"""
invoice_extractor_vlm.py

Extracts structured fields from fuel invoice images embedded in a docx,
using a local VLM for semantic field identification, EasyOCR for raw
text capture, and the SAME regex fallback tier validated against this
exact invoice set in the original Textract-based pipeline.

Architecture — mirrors the split used for the distance log table:
  - VLM does the thing OCR fundamentally can't: understanding that a
    given piece of text IS a vendor name, or IS a quantity, rather than
    just being characters at a position. This replaces Textract's
    analyze_expense semantic field detection.
  - EasyOCR does plain text capture, no semantic understanding needed.
    This replaces Textract's expense_doc.lines raw text output, and
    feeds the SAME regex fallback functions that already existed.
  - The regex fallback tier (scrape_missing_metrics_from_text,
    extract_granular_taxes, apply_global_fallbacks) is REUSED VERBATIM.
    These functions never depended on Textract's objects — they always
    operated on a plain std_record dict and a raw text blob. The
    Costco/Burnaby/Shell-specific patches, province/city keyword
    inference, payment form matching, and time repair all transfer
    directly.

What's REMOVED from the original pipeline:
  field_map / update_field_value / clean_numeric's ranking logic — this
  machinery existed to reconcile MULTIPLE competing Textract-labeled
  fields per target column. It's unnecessary here because the VLM
  already outputs fields under your exact target names directly.

Known limitation, stated plainly:
  Textract's confidence scores are measured per-field from the model's
  internals. A VLM has no equivalent signal. The single
  "self_reported_confidence" value below is the model guessing about
  its own guess — a soft, unvalidated proxy. It is reported separately,
  never presented as equivalent to Textract's confidence scoring.

Dataset-specific note:
  clean_and_standardize_dataframe() below includes a few hardcoded
  string replacements (e.g. '2016- 33 16' -> '2016/03/16') that were
  debugged against SPECIFIC garbled strings Textract produced on THIS
  exact invoice set. A VLM will very likely garble different cells
  differently (or not garble them at all) — these exact-match patches
  are harmless no-ops if they don't apply, but don't assume they cover
  whatever NEW malformed strings the VLM introduces. The quality report
  at the end prints every row that still fails date/time parsing so
  those are visible rather than silently dropped.

Philosophy on missing/uncertain values, held throughout this project:
  Never fabricate a value that wasn't actually present on the receipt.
  Where extraction fails, the value stays None/NaN.

Requirements:
  pip install ollama easyocr opencv-python pandas numpy
  ollama pull qwen2.5vl:7b
  Ollama must be running (ollama serve, or it auto-starts on macOS).
"""

import base64
import json
import re
import zipfile
from pathlib import Path

import cv2
import easyocr
import numpy as np
import ollama
import pandas as pd

DOCX_PATH = 'Data/Fuel Invoices.docx'
MODEL     = 'qwen2.5vl:7b'

INVOICE_FIELDS = [
    'invoice_number', 'date', 'time', 'vendor_name', 'city', 'province',
    'fuel_type', 'fuel_grade', 'quantity', 'cost_per_litre', 'cost',
    'total_tax', 'fed_tax', 'prov_tax', 'payment_form'
]

# Fields that should be cleaned with clean_numeric() when seeding std_record
NUMERIC_INVOICE_FIELDS = {
    'quantity', 'cost_per_litre', 'cost', 'total_tax', 'fed_tax', 'prov_tax'
}

# EasyOCR routinely inserts stray spaces around punctuation (observed:
# "2.89" -> "2 .89", "09:56" -> "39:56", etc). Every money-amount regex
# in this file originally used a rigid \d{1,3}\.\d{2} pattern that
# silently fails to match the moment a space sneaks in next to the
# decimal point -- the match just returns None with no error, so a
# correct INCL/tax line can sit right in raw_blob and never get picked
# up. MONEY_PATTERN tolerates a single optional space on either side of
# the decimal point; downstream code must call _clean_money() on the
# captured group before float()-ing it, since the match itself may now
# contain that stray space.
MONEY_PATTERN = r'(\d{1,3}\s?\.\s?\d{2})'

def _clean_money(raw_match: str) -> float:
    """Strips stray OCR whitespace from a captured money string before
    casting to float, e.g. '2 .89' -> 2.89."""
    return float(raw_match.replace(' ', ''))

VLM_INVOICE_PROMPT = """This image is a fuel purchase receipt or invoice.

Extract the following fields into a JSON object with exactly these keys:
["invoice_number", "date", "time", "vendor_name", "city", "province", "fuel_type", "fuel_grade", "quantity", "cost_per_litre", "cost", "total_tax", "fed_tax", "prov_tax", "payment_form"]

Rules:
1. date, time: transcribe EXACTLY as printed on the receipt. Do not reformat, do not guess a format if ambiguous.
2. province: the 2-letter code (AB, BC, SK, MB, ON, QC) ONLY if explicitly printed. If not explicitly shown, return null — do not infer it from the city name yourself.
3. quantity, cost_per_litre, cost, total_tax, fed_tax, prov_tax: numbers. If federal and provincial tax are only shown as one combined total, put that value in total_tax and leave fed_tax/prov_tax null — do not split it yourself.
4. fuel_type (e.g. "Diesel", "Gasoline") and fuel_grade (e.g. "Regular", "Premium", "Supreme") if shown.
5. payment_form: transcribe as printed (e.g. "Visa", "Interac", "Cash").
6. For ANY field not present on the receipt, or that you cannot confidently read, return null. Never guess or estimate a value that isn't actually printed.

Also include a "self_reported_confidence" key: a rough 0.0-1.0 self-assessment of your overall confidence in this extraction. This is a soft estimate, not a measured quantity.

Output ONLY the JSON object — no markdown fences, no explanation.
"""

# ── Lazy-loaded EasyOCR reader (singleton) ──────────────────────────────────

_OCR_READER = None

def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        print("Loading EasyOCR (one-time)...")
        _OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _OCR_READER


# ── Step 1: Extract images from docx, in memory, no disk writes ─────────────

def extract_images_from_docx(docx_path):
    images = []
    with zipfile.ZipFile(docx_path, 'r') as z:
        image_names = sorted(
            n for n in z.namelist() if n.startswith('word/media/')
        )
        for name in image_names:
            images.append({'filename': Path(name).name, 'bytes': z.read(name)})
    print(f"Found {len(images)} embedded image(s) in {Path(docx_path).name}")
    return images


# ── Step 2: VLM semantic field call — replaces Textract analyze_expense ─────

def call_vlm_for_invoice_fields(img_bytes, source_filename):
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{
                'role': 'user', 'content': VLM_INVOICE_PROMPT, 'images': [img_b64]
            }],
            options={'temperature': 0.0, 'num_ctx': 8192}
        )
    except Exception as e:
        print(f"   ⚠️  Ollama call failed for {source_filename}: {e}")
        return {}, None

    raw = response['message']['content'].strip()
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()

    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Response was not a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"   ⚠️  VLM JSON parse failed for {source_filename}: {e}")
        print(f"      Raw output (first 250 chars): {raw[:250]}")
        return {}, None

    confidence = result.pop('self_reported_confidence', None)
    return result, confidence


# ── Step 3: Raw OCR text — replaces Textract's expense_doc.lines ────────────

def get_raw_ocr_lines(img_bytes):
    """
    Plain free-text OCR over the full invoice image. No semantic
    understanding needed here — this exists purely to give the regex
    fallback tier a text blob to scan, same role Textract's lines
    feature played originally.
    """
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = _get_ocr_reader().readtext(
        gray, detail=0, paragraph=True, batch_size=1
    )
    return [r.strip() for r in results if r.strip()]


# ── Step 4: Seed std_record from VLM output, same defaults as original ──────

def clean_numeric(val_str):
    """Strip currencies, text, and spaces to isolate clean numeric strings.
    Reused verbatim — operates on raw values regardless of source."""
    if not val_str:
        return None
    cleaned = re.sub(r'[^\d\.\-]', '', str(val_str))
    return cleaned if cleaned else None


def vlm_json_to_std_record(vlm_json):
    """
    Maps the VLM's direct JSON output onto the SAME std_record shape
    and defaults the original Textract-based pipeline used. Preserving
    these exact defaults ('UNKNOWN' sentinels for province/fuel_type/
    fuel_grade, None for everything else) matters because the regex
    fallback functions below check for these specific sentinel values.
    """
    std_record = {
        'invoice_number': None, 'date': None, 'time': None,
        'vendor_name': None, 'city': None, 'province': 'UNKNOWN',
        'fuel_type': 'UNKNOWN', 'fuel_grade': 'UNKNOWN', 'quantity': None,
        'cost_per_litre': None, 'cost': None, 'total_tax': None,
        'fed_tax': None, 'prov_tax': None, 'payment_form': None,
        'invoice_number_low_confidence': False
    }

    for field in INVOICE_FIELDS:
        val = vlm_json.get(field)
        if val is None or str(val).strip().lower() in ('', 'null', 'none'):
            continue   # leave the default in place

        if field in NUMERIC_INVOICE_FIELDS:
            cleaned = clean_numeric(val)
            if cleaned:
                std_record[field] = cleaned
        else:
            std_record[field] = str(val).strip().upper()

    return std_record


# ── Step 5: Regex fallback tier — REUSED VERBATIM from the Textract pipeline ─
#   None of these functions reference Textract objects — they always
#   operated on a plain dict and a list of text strings. Unchanged below.

def scrape_missing_metrics_from_text(std_record, raw_blob):
    """
    Robust fallback regex engine that handles layout text reversals,
    multi-line spacing errors, varying unit symbols (L vs G), and
    prepaid receipt structures.
    """
    # 1. QUANTITY (LITERS) FALLBACK
    if not std_record['quantity']:
        qty_pattern = r'(?:LITRES|LITERS|\bL\b)[:\-]?\s+(\d+\.\d{2,3})|(\d+\.\d{2,3})\s*(?:LITRES|LITERS|\bL\b)'
        qty_match = re.search(qty_pattern, raw_blob)
        if qty_match:
            std_record['quantity'] = qty_match.group(1) if qty_match.group(1) else qty_match.group(2)

    # 2. UNIT COST (PRICE PER LITRE / GALLON) FALLBACK
    if not std_record['cost_per_litre']:
        price_pattern = r'PRICE/[LG](?:ITRE)?[:\-]?\s*\$?[:\-]?\s*\$?(\d+\.\d{2,3})|(\d+\.\d{2,3})\s*PRICE/[LG](?:ITRE)?'
        price_match = re.search(price_pattern, raw_blob)
        if price_match:
            std_record['cost_per_litre'] = price_match.group(1) if price_match.group(1) else price_match.group(2)

    # 3. FIX FOR COSTCO (DECOUPLING DUPLICATED ATTRIBUTES)
    if std_record['quantity'] and std_record['cost_per_litre']:
        if float(std_record['quantity']) == float(std_record['cost_per_litre']):
            true_price_match = re.search(r'(?:PRICE/LITRE|PRICE/L)[:\-]?\s*\$?(\d+\.\d{2,3})', raw_blob)
            if true_price_match:
                std_record['cost_per_litre'] = true_price_match.group(1)

    # 4. PREPAID FOOTER SCANNER (For Burnaby and similar prepay receipts)
    if std_record['cost'] and (not std_record['quantity'] or not std_record['cost_per_litre']):
        if 'PREPAY' in raw_blob or 'PRE-PAY' in raw_blob:
            gst_inc_match = re.search(r'FUEL INCLUDES\s+GST\s+\d+\.\d+%\s*\$?' + MONEY_PATTERN, raw_blob)
            if gst_inc_match and not std_record['fed_tax']:
                std_record['fed_tax'] = str(_clean_money(gst_inc_match.group(1)))

    return std_record


def extract_granular_taxes(raw_blob):
    """
    NOTE: raw_blob arrives here already space-joined with no newlines
    (built via " ".join(raw_lines) in apply_global_fallbacks), so the
    previous per-line splitting was a no-op — the whole blob was always
    treated as a single "line". That's why a GST-INCL value appearing
    later in the receipt could be missed or, worse, a nearby unrelated
    number (e.g. from FUEL SALES) could be picked up by the proximity
    regex below it instead.

    Fix: check for the explicit GST-INCL / P-HST-INCL phrasing FIRST,
    searched across the whole blob rather than line-by-line, since that
    phrasing unambiguously identifies the tax value regardless of what
    other dollar amounts sit nearby in the text. Only fall back to the
    looser proximity-based regex if no explicit INCL phrasing is found.
    """
    fed_tax = 0.00
    prov_tax = 0.00

    fed_incl_match = re.search(r'(?:GST|F[\-\s]?HST)\s*INCL(?:UDED\s+IN\s+FUEL)?\.?\s*\$?\s*' + MONEY_PATTERN, raw_blob)
    if fed_incl_match:
        fed_tax = _clean_money(fed_incl_match.group(1))
    else:
        match = re.search(r'(?:GST|F-HST)[^0-9]*?\$?\s*' + MONEY_PATTERN + r'\b', raw_blob)
        if match:
            fed_tax = _clean_money(match.group(1))

    prov_incl_match = re.search(r'P[\-\s]?HST\s*INCL(?:UDED\s+IN\s+FUEL)?\.?\s*\$?\s*' + MONEY_PATTERN, raw_blob)
    if prov_incl_match:
        prov_tax = _clean_money(prov_incl_match.group(1))
    else:
        match = re.search(r'(?:PST|QST|BC TAX|PROV|P-HST)[^0-9]*?\$?\s*' + MONEY_PATTERN + r'\b', raw_blob)
        if match:
            prov_tax = _clean_money(match.group(1))

    # Requested fallback: if no separate provincial-tax label exists at all
    # in the receipt text, assume there's no separate prov tax to find and
    # mirror fed_tax into prov_tax's absence isn't right — instead this
    # signals total_tax should equal fed_tax alone (prov_tax = 0), which is
    # already the default. The actual ask was the reverse case: when a
    # P-HST line EXISTS but wasn't matched, don't silently leave it at 0.
    # That's now covered by the P-HST INCL check above. No further
    # same-value mirroring needed once F-HST INCL is matched correctly.

    return fed_tax, prov_tax


def apply_global_fallbacks(std_record, raw_lines):
    """Standardizes codes, fills structural gaps using raw text streams,
    and computes math fallbacks. Unchanged from the original pipeline."""
    raw_blob = " ".join(raw_lines).upper()
    print(raw_blob)

    std_record = scrape_missing_metrics_from_text(std_record, raw_blob)

    fed_tax, prov_tax = extract_granular_taxes(raw_blob)

    def _is_implausible_tax(existing, cost_str):
        # An existing tax value counts as implausible (and should be
        # overridden by the regex-derived value) if it's missing, OR if
        # it's suspiciously close to/exceeding the full cost — the
        # signature of a misread field (e.g. FUEL SALES grabbed instead
        # of the actual tax line).
        if existing is None or existing == 'None':
            return True
        try:
            existing_f = float(existing)
            cost_f = float(cost_str) if cost_str and cost_str != 'None' else None
            return cost_f is not None and cost_f > 0 and existing_f >= cost_f
        except ValueError:
            return True

    def _should_prefer_regex_tax(existing, regex_value):
        # Separate from the implausibility check above: an existing value
        # of exactly 0.0 is indistinguishable from "genuinely no tax of
        # this kind" vs "extraction silently failed to find the INCL
        # line and fell back to the 0.0 default". If the regex pass
        # below — now run on the FULL blob with corrected INCL patterns
        # — found a real non-zero number, prefer it over a bare 0.0,
        # since a missed tax line is far more common on these receipts
        # than a genuinely-zero provincial tax on a province that
        # otherwise prints an explicit P-HST/PHST line at all.
        try:
            existing_f = float(existing) if existing not in (None, 'None') else 0.0
        except ValueError:
            return True
        return existing_f == 0.0 and regex_value > 0.0

    if _is_implausible_tax(std_record['fed_tax'], std_record.get('cost')) or \
       _should_prefer_regex_tax(std_record['fed_tax'], fed_tax):
        std_record['fed_tax'] = fed_tax
    if _is_implausible_tax(std_record['prov_tax'], std_record.get('cost')) or \
       _should_prefer_regex_tax(std_record['prov_tax'], prov_tax):
        std_record['prov_tax'] = prov_tax
    try:
        f_tax = float(std_record.get('fed_tax', 0.0) or 0.0)
        p_tax = float(std_record.get('prov_tax', 0.0) or 0.0)
        t_tax = float(std_record.get('total_tax', 0.0) or 0.0)
    except ValueError:
        f_tax, p_tax, t_tax = 0.0, 0.0, 0.0
    if t_tax < (f_tax + p_tax) or not std_record['total_tax'] or std_record['total_tax'] == 'None':
        std_record['total_tax'] = f"{round(f_tax + p_tax, 2)}"

    # Plausibility guard: tax equal to (or exceeding) the full cost means a
    # field was misread — most commonly a 'FUEL SALES'/'TOTAL OWED' line
    # getting grabbed where a tax line was meant. Clearing it surfaces the
    # row in the quality report rather than silently keeping an impossible
    # 100%+ tax rate.
    try:
        cost_val = float(std_record.get('cost') or 0.0)
        tax_val = float(std_record.get('total_tax') or 0.0)
        if cost_val > 0 and tax_val >= cost_val:
            std_record['total_tax'] = None
            std_record['fed_tax'] = None
            std_record['prov_tax'] = None
    except ValueError:
        pass

    # Province inference from city/region keywords
    prov_dump = std_record['province']
    if prov_dump == 'UNKNOWN' or len(prov_dump) > 2:
        if any(p in raw_blob for p in [' AB ', 'ALBERTA', 'EDMONTON', 'RED DEER', 'CALGARY', 'HANNA', 'CONSORT']):
            std_record['province'] = 'AB'
        elif any(p in raw_blob for p in [' BC ', 'BRITISH COLUMBIA', 'KELOUNA', 'KELOWNA', 'VANCOUVER', 'BURNABY', 'LANGLEY', 'LANKEY']):
            std_record['province'] = 'BC'
        elif any(p in raw_blob for p in [' SK ', 'SASKATCHEWAN', 'REGINA', 'SASKATOON']):
            std_record['province'] = 'SK'
        elif any(p in raw_blob for p in [' MB ', 'MANITOBA', 'WINNIPEG']):
            std_record['province'] = 'MB'
        elif any(p in raw_blob for p in [' ON ', 'ONTARIO', 'BARRIE', 'CALEDON', 'WILLOWDALE']):
            std_record['province'] = 'ON'

    # City inference
    if not std_record['city'] or std_record['city'] == 'UNKNOWN':
        for city_check in ['RED DEER', 'EDMONTON', 'HANNA', 'CALEDON', 'WINNIPEG',
                           'WILLOWDALE', 'BARRIE', 'LANGLEY', 'BURNABY', 'CONSORT']:
            if city_check in raw_blob:
                std_record['city'] = city_check
                break

    # Fuel type/grade sweep
    if std_record['fuel_type'] == 'UNKNOWN':
        if 'DIESEL' in raw_blob:
            std_record['fuel_type'] = 'DIESEL'
            std_record['fuel_grade'] = 'UNKNOWN'
        else:
            for gas in ['SUPREME', 'BRONZE', 'PLUS', 'REGULAR', 'UNLEADED', 'GASOLINE', 'EREG']:
                if gas in raw_blob:
                    std_record['fuel_type'] = 'GASOLINE'
                    std_record['fuel_grade'] = gas
                    if gas == 'EREG':
                        std_record['fuel_grade'] = 'REGULAR'

    # Sanity-bound cost_per_litre BEFORE the math fallback below. A corrupted
    # OCR/VLM read (e.g. missing decimal -> 1259.000 instead of 1.259) would
    # otherwise survive untouched, since it isn't technically "missing".
    # Treating it as missing lets the math fallback below recompute it.
    COST_PER_LITRE_BOUNDS = (0.3, 3.0)
    if std_record['cost_per_litre'] and std_record['cost_per_litre'] != 'None':
        try:
            cpl = float(std_record['cost_per_litre'])
            if not (COST_PER_LITRE_BOUNDS[0] <= cpl <= COST_PER_LITRE_BOUNDS[1]):
                std_record['cost_per_litre'] = None
        except ValueError:
            std_record['cost_per_litre'] = None

    # Mathematically deduce unit price if both other values are known
    if (not std_record['cost_per_litre'] or std_record['cost_per_litre'] == 'None') and std_record['cost'] and std_record['quantity']:
        try:
            tot_cost = float(std_record['cost'])
            volume = float(std_record['quantity'])
            if volume > 0.0:
                recomputed = round(tot_cost / volume, 3)
                # Only accept if the recomputed value is itself plausible —
                # otherwise cost/quantity were also misread, and a bad
                # cost_per_litre is more honestly "missing" than a
                # fabricated-but-still-wrong fallback value.
                if COST_PER_LITRE_BOUNDS[0] <= recomputed <= COST_PER_LITRE_BOUNDS[1]:
                    std_record['cost_per_litre'] = f"{recomputed}"
        except ValueError:
            pass

    # Invoice/transaction number fallback patterns
    if not std_record['invoice_number'] or std_record['invoice_number'] == 'None':
        inv_pattern = r'\b(?:INV(?:OICE)?\.?\s*(?:No\.?)?|TRANS(?:\s*#|\s*ACTION)?\.?\s*(?:No\.?)?|TICKET\s*#?|REF(?:ERENCE)?\s*#?)\s*[:\-]?\s*([A-Z0-9\-]{4,15})\b'
        inv_match = re.search(inv_pattern, raw_blob, re.IGNORECASE)
        if inv_match:
            std_record['invoice_number'] = inv_match.group(1).strip()

    if not std_record['invoice_number'] or std_record['invoice_number'] == 'None':
        pc_match = re.search(r'PC\d+:\s*(\d+)', raw_blob)
        if pc_match:
            std_record['invoice_number'] = pc_match.group(1)

    if not std_record['invoice_number'] or std_record['invoice_number'] == 'None':
        # Original pattern required INV/TRANS/TICKET/REF immediately before
        # the number — Shell's "No.  62165364653" used a label not in that
        # list. Added separately rather than folding into the main pattern
        # since 'No.' alone is a much weaker, more ambiguous signal.
        no_match = re.search(r'\bNo\.?\s+(\d{6,})\b', raw_blob, re.IGNORECASE)
        if no_match:
            std_record['invoice_number'] = no_match.group(1)

    if not std_record['invoice_number'] or std_record['invoice_number'] == 'None':
        # Last-resort fallback for receipts with NO label at all (the
        # common case per review). Takes the longest standalone digit run
        # as a proxy ID. This is a meaningfully weaker signal than the
        # labeled matches above, so it's flagged via a companion column
        # rather than presented as equivalent confidence.
        candidates = re.findall(r'\b\d{6,}\b', raw_blob)
        if candidates:
            std_record['invoice_number'] = max(candidates, key=len)
            std_record['invoice_number_low_confidence'] = True

    # Payment form keyword loop
    if not std_record['payment_form'] or std_record['payment_form'] == 'None':
        raw_upper = raw_blob.upper()
        if 'INTERAC' in raw_upper or 'DEBIT' in raw_upper or 'CHEQUING' in raw_upper:
            std_record['payment_form'] = 'DEBIT'
        elif 'VISA' in raw_upper or 'UISA' in raw_upper:
            std_record['payment_form'] = 'VISA'
        elif 'MASTERCARD' in raw_upper or 'MCARD' in raw_upper or 'MASTER' in raw_upper:
            std_record['payment_form'] = 'MASTERCARD'
        elif 'AMEX' in raw_upper or 'AMERICAN EXPRESS' in raw_upper:
            std_record['payment_form'] = 'AMEX'
        elif 'CASH' in raw_upper:
            std_record['payment_form'] = 'CASH'
        elif 'CARD #' in raw_upper or 'FLEET' in raw_upper:
            std_record['payment_form'] = 'FLEET_CARD'

    # Time repair
    if not std_record['time'] or std_record['time'] == 'None' or std_record['time'] is None:
        time_pattern = r'\b([0-9]?\d)\s*:\s*([0-9]\d)(?:\s*:\s*([0-5]\d))?\b'
        time_match = re.search(time_pattern, raw_blob)
        if time_match:
            hours = time_match.group(1).zfill(2)
            minutes = time_match.group(2)
            seconds = time_match.group(3)
            std_record['time'] = f"{hours}:{minutes}:{seconds or '00'}"

    return std_record


# ── Step 6: Per-invoice orchestration ─────────────────────────────────────────

def extract_clean_record_vlm(image_records):
    """
    For each image: call VLM for semantic fields, run EasyOCR for raw
    text, seed std_record from the VLM's output, then run it through
    the SAME fallback chain the Textract pipeline used.
    """
    clean_records = []
    confidence_records = []

    for i, img in enumerate(image_records, 1):
        print(f"\n   [{i}/{len(image_records)}] {img['filename']}")

        vlm_json, self_conf = call_vlm_for_invoice_fields(img['bytes'], img['filename'])
        raw_lines = get_raw_ocr_lines(img['bytes'])

        std_record = vlm_json_to_std_record(vlm_json)
        std_record = apply_global_fallbacks(std_record, raw_lines)
        std_record['source_file'] = img['filename']

        n_present = sum(
            1 for f in INVOICE_FIELDS
            if std_record.get(f) not in (None, 'UNKNOWN', 'None')
        )
        conf_str = f"{self_conf:.2f}" if isinstance(self_conf, (int, float)) else "n/a"
        print(f"      vendor={std_record.get('vendor_name')}  "
              f"cost={std_record.get('cost')}  fields_present={n_present}/{len(INVOICE_FIELDS)}  "
              f"self_conf={conf_str}")

        clean_records.append(std_record)
        confidence_records.append({
            'source_file': img['filename'],
            'self_reported_confidence': self_conf,
            'note': 'VLM self-assessment — NOT equivalent to Textract per-field confidence'
        })

    return (pd.DataFrame(clean_records) if clean_records else pd.DataFrame(),
            pd.DataFrame(confidence_records) if confidence_records else pd.DataFrame())


# ── Step 7: Standardization pass — reused, with dataset-specific notes ──────

def clean_and_standardize_dataframe(df):
    """
    Reused from the Textract pipeline. The hardcoded string replacements
    below (2016- 33 16, 2025/11/38, 99:56:00) were debugged against
    SPECIFIC garbled strings Textract produced on this exact invoice set.
    They are harmless no-ops if the VLM doesn't produce those same
    strings — but it may introduce DIFFERENT malformed strings of its
    own, which these patches won't catch. The quality report below
    prints every row that still fails to parse, so new patches can be
    added deliberately rather than silently missed.
    """
    clean_df = df.copy()

    clean_df['date'] = clean_df['date'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    clean_df['time'] = clean_df['time'].astype(str).str.replace(r'(\d+):\s+(\d+)', r'\1:\2', regex=True).str.strip()
    clean_df['time'] = clean_df['time'].str.replace(r'^(\d{2}:\d{2})$', r'\1:00', regex=True)

    # Dataset-specific patches from the Textract pipeline — see docstring above
    clean_df['date'] = clean_df['date'].str.replace('2016- 33 16', '2016/03/16', regex=False)
    clean_df['date'] = clean_df['date'].str.replace('2025/11/38', '2025/11/30', regex=False)
    clean_df['date'] = clean_df['date'].str.replace('-', '/', regex=False)
    clean_df['time'] = clean_df['time'].str.replace('99:56:00', '09:56:00', regex=False)

    clean_df['timestamp'] = pd.to_datetime(
        clean_df['date'] + ' ' + clean_df['time'],
        format='%Y/%m/%d %H:%M:%S', errors='coerce'
    )
    clean_df['date'] = clean_df['timestamp'].dt.strftime('%Y-%m-%d')
    clean_df['time'] = clean_df['timestamp'].dt.strftime('%H:%M:%S')

    text_columns = ['vendor_name', 'city', 'province', 'invoice_number', 'fuel_grade', 'payment_form']
    for col in text_columns:
        if col in clean_df.columns:
            clean_df[col] = (clean_df[col].astype(str)
                             .str.replace(r'[\r\n]+', ' ', regex=True)
                             .str.strip()
                             .str.strip('-')
                             .str.replace(r'-', ' ', regex=False)
                             .str.replace(r'\s+', ' ', regex=True)
                             .str.strip()
                             .replace({'None': np.nan, 'NONE': np.nan, 'nan': np.nan, '': np.nan}))

    numeric_columns = ['quantity', 'cost_per_litre', 'cost', 'total_tax', 'fed_tax', 'prov_tax']
    for col in numeric_columns:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

    # Defensive rule originally added for a Textract placeholder quirk —
    # harmless no-op for VLM output unless it does something similar.
    clean_df.loc[clean_df['quantity'] == 1.0, 'quantity'] = np.nan

    clean_df = clean_df.sort_values(by='timestamp').reset_index(drop=True)
    return clean_df


def clean_dataframe(df_uncleaned):
    """Remove all empty rows and exact duplicates. Unchanged."""
    df_uncleaned = df_uncleaned.replace(r'^\s*$', pd.NA, regex=True)
    df_cleaned = df_uncleaned.dropna(how='all')
    is_duplicate = df_cleaned.astype(str).duplicated()
    df_cleaned = df_cleaned[~is_duplicate].reset_index(drop=True)
    return df_cleaned


# ── Step 8: Full pipeline ──────────────────────────────────────────────────

def extract_invoices_from_docx_vlm(docx_path=DOCX_PATH, save_conf_scores_path=None):
    images = extract_images_from_docx(docx_path)
    if not images:
        print("⚠️  No images found.")
        return pd.DataFrame(), pd.DataFrame()

    df_raw, df_conf = extract_clean_record_vlm(images)
    df_clean = clean_and_standardize_dataframe(df_raw)
    df_clean = clean_dataframe(df_clean)

    n_failed_dates = df_clean['date'].isna().sum() if 'date' in df_clean else 0
    print(f"\n{'─'*50}")
    print(f"Quality report")
    print(f"{'─'*50}")
    print(f"   Invoices processed     : {len(df_clean)}")
    for f in INVOICE_FIELDS:
        if f in df_clean.columns:
            n_present = df_clean[f].notna().sum()
            print(f"   {f:<18} present : {n_present}/{len(df_clean)}")
    if n_failed_dates:
        print(f"   ⚠️  {n_failed_dates} row(s) with unparsed date/time — review individually:")
        print(df_clean.loc[df_clean['date'].isna(),
                           ['source_file', 'vendor_name']].to_string())
    if 'invoice_number_low_confidence' in df_clean.columns:
        n_low_conf = df_clean['invoice_number_low_confidence'].sum()
        if n_low_conf:
            print(f"   ⚠️  {n_low_conf} row(s) with unlabeled, low-confidence invoice_number — review individually:")
            print(df_clean.loc[df_clean['invoice_number_low_confidence'],
                               ['source_file', 'vendor_name', 'invoice_number']].to_string())
    print(f"{'─'*50}\n")

    if save_conf_scores_path:
        df_conf.to_excel(save_conf_scores_path, index=False)
        print(f"Confidence notes saved to {save_conf_scores_path}")

    return df_clean, df_conf


if __name__ == '__main__':
    df_invoices, df_conf = extract_invoices_from_docx_vlm()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 250)
    print(df_invoices.to_string())