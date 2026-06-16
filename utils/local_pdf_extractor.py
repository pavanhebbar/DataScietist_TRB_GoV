"""
local_pdf_extractor.py

Hybrid table extraction for scanned IFTA distance logs.

Pipeline per row:
  1. Morphological grid-line detection — finds actual pixel positions of
     printed table lines on this scan (measured, not predicted).
  2. DATE: ONE VLM call, isolated single-cell crop. DATE has a fixed
     grammar (DD/M/YY) that benefits from a model reasoning about
     plausible digit/separator patterns — the one field where VLM
     genuinely outperforms raw OCR here.
  3. START_KM, END_KM, TOTAL_KM: strict cell-by-cell EasyOCR with digit
     allowlist. Unconstrained digit strings — no internal grammar for a
     VLM to exploit, so OCR is equally capable and far faster.
  4. STARTING_POINT, DESTINATION: isolated EasyOCR free-text read.
  5. The 13 provincial KMS/Fuel columns: strict cell-by-cell EasyOCR with
     digit allowlist.
  6. Odometer continuity cross-check: START_KM of row N should equal
     END_KM of row N-1. Used ONLY to recover a null primary value, never
     to overwrite a value that was actually read. Disagreements between
     two present values are flagged, not resolved.

Philosophy on missing/uncertain values, held throughout:
  Never fabricate or interpolate a value that wasn't actually read. This
  matters because the next stage of this project is anomaly detection —
  silently filling a gap with a plausible guess could mask the very
  irregularity the model exists to catch. Where extraction fails, the
  value stays null/NaN and is reported as such.

Requirements:
  pip install ollama opencv-python easyocr pdf2image pandas numpy
  ollama pull qwen2.5vl:7b
  Ollama must be running (ollama serve, or it auto-starts on macOS).
"""

import base64
import json
import re

import cv2
import easyocr
import numpy as np
import ollama
import pandas as pd
from pdf2image import convert_from_path

# ── Configuration ────────────────────────────────────────────────────────────

PDF_PATH = 'Data/Distance log 2.pdf'
DPI      = 300
MODEL    = 'qwen2.5vl:7b'

COLUMN_NAMES = [
    'DATE', 'START_KM', 'STARTING_POINT', 'DESTINATION', 'END_KM', 'TOTAL_KM',
    'AB_KMS', 'BC_KMS', 'SK_KMS', 'MB_KMS', 'ON_KMS', 'QC_KMS', 'YT_KMS',
    'AB_FUEL', 'BC_FUEL', 'SK_FUEL', 'MB_FUEL', 'ON_FUEL', 'QC_FUEL'
]
RIGHT_BLOCK_COLS = COLUMN_NAMES[6:]
NUMERIC_COLS     = set(COLUMN_NAMES) - {'DATE', 'STARTING_POINT', 'DESTINATION'}

# Column index boundaries (col_pos has len(COLUMN_NAMES)+1 entries):
#   DATE: col_pos[0]->[1]  START_KM: [1]->[2]  STARTING_POINT: [2]->[3]
#   DESTINATION: [3]->[4]  END_KM: [4]->[5]  TOTAL_KM: [5]->[6]
#   provincial cols: [6]->[19]

MAX_PLAUSIBLE_PROVINCIAL_KM   = 3000
MAX_PLAUSIBLE_PROVINCIAL_FUEL = 1500
NORMALIZE_DATE_CHARACTERS     = True

DATE_PROMPT = """This image is a single cell containing a date, written DD/M/YY (e.g. "02/5/22").

Transcribe exactly what is written. If the cell is blank or you cannot confidently read it with confidence > 90\%, return the single word: null

Output ONLY the date string (or null) — no quotes, no explanation, no markdown.
"""

DATE_DEBUG_PROMPT = """This image is a single cell containing a date, written DD/M/YY (e.g. "02/5/22").

Transcribe exactly what is written.

Output ONLY the date string and the confidence score — no quotes,
no explanation, no markdown.
"""

DATE_KM_PROMPT = """This image is a crop from the SAME row of a table.
It contains exactly two cells, left to right: ["DATE", "START_KM"]

DATE is written DD/M/YY, e.g. "02/5/22". Transcribe exactly as written,
do not reformat. If the date cell is blank or you cannot confidently read
the date with confidence > 90\%, return the single word: null

Output ONLY the date string (or null) — no quotes, no explanation, no markdown.
"""

DATE_KM_DEBUG_PROMPT = """This image is a crop from the SAME row of a table.
It contains exactly two cells, left to right: ["DATE", "START_KM"]

DATE is written DD/M/YY, e.g. "02/5/22". Transcribe exactly as written,
do not reformat. If the date cell is blank or you cannot confidently read
the date with confidence > 90\%, return the single word: null

Output ONLY the date string and the confidence score — no quotes,
no explanation, no markdown.
"""

# ── Lazy-loaded EasyOCR reader (singleton) ──────────────────────────────────

_OCR_READER = None

def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        print("Loading EasyOCR (one-time)...")
        _OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _OCR_READER


# ── Step 1: Preprocessing + grid-line detection ───────────────────────────────

def preprocess(pil_image):
    img  = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    return gray, binary


def find_grid_lines(binary, h, w):
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 7, 2))
    h_mask   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, h // 12))
    v_mask   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    return h_mask, v_mask


def cluster_positions(mask, axis, min_gap=12):
    projection = mask.sum(axis=axis).astype(float)
    projection = projection / (projection.max() + 1e-9)
    lit = np.where(projection > 0.08)[0]
    if len(lit) == 0:
        return []
    clusters, group = [], [lit[0]]
    for px in lit[1:]:
        if px - group[-1] > min_gap:
            clusters.append(int(np.mean(group)))
            group = []
        group.append(px)
    clusters.append(int(np.mean(group)))
    return clusters


# ── Step 2: DATE — the only VLM call per row ─────────────────────────────────

def crop_to_base64_png(gray_img, y1, y2, x1, x2, pad=6):
    crop = gray_img[max(0, y1 - pad): y2 + pad, max(0, x1 - pad): x2 + pad]
    success, buf = cv2.imencode('.png', crop)
    return base64.b64encode(buf).decode('utf-8')


def ocr_date_with_vlm(gray_img, y1, y2, x1, x2, max_retries=0,
                      xback2=None):
    """
    The only VLM call per row. DATE has a fixed DD/M/YY grammar, which is
    where a model's contextual reasoning genuinely helps over raw OCR —
    every other field in this table is an unconstrained digit string or
    free text with no equivalent grammar to exploit.

    One retry on null, since prior testing showed nulls can occur on
    cells that are visually unambiguous (a model inference quirk, not an
    image-quality wall) — a second identical call sometimes resolves it.
    """
    img_b64 = crop_to_base64_png(gray_img, y1, y2, x1, x2)
    img_b64_back = None
    if xback2 is not None:
        img_b64_back = crop_to_base64_png(
            gray_img, y1, y2, x1, xback2)

    for attempt in range(max_retries + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{'role': 'user', 'content': DATE_PROMPT,
                           'images': [img_b64]}],
                options={'temperature': 0.0, 'num_ctx': 4096}
            )
        except Exception as e:
            print(f"      ⚠️  Ollama call failed (attempt {attempt+1}): {e}")
            continue

        text = response['message']['content'].strip()
        if text.lower() not in ('null', 'none', ''):
            return text

        if img_b64_back is not None:
            print (f"Attempting to read date from Date + start_km cell crop")
            response_date2 = ollama.chat(
                model=MODEL,
                messages=[{'role': 'user', 'content': DATE_KM_PROMPT,
                            'images': [img_b64_back]}],
                options={'temperature': 0.0, 'num_ctx': 4096}
            )
            text_date2 = response_date2['message']['content'].strip()
            if text_date2.lower() not in ('null', 'none', ''):
                return text_date2

        if attempt < max_retries:
            print(f"      ↻ DATE null on attempt {attempt+1}, retrying")

    return None


# ── Step 3: Numeric KM fields — strict EasyOCR, no VLM ───────────────────────

def ocr_numeric_cell(gray_img, y1, y2, x1, x2, pad=4):
    """
    Strict digit-only EasyOCR for START_KM/END_KM/TOTAL_KM and the 13
    provincial columns. No internal grammar for a VLM to exploit here —
    these are plain digit strings, so OCR is equally capable and far
    faster than a model call.
    """
    crop = gray_img[max(0, y1 + pad): y2 - pad, max(0, x1 + pad): x2 - pad]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    if crop.shape[1] < 80:
        scale = max(2, 80 // crop.shape[1])
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    results = _get_ocr_reader().readtext(
        crop, allowlist='0123456789', detail=0, paragraph=False, batch_size=1
    )
    text = re.sub(r'[^0-9]', '', ' '.join(results))
    return text if text else None


# ── Step 4: STARTING_POINT/DESTINATION — isolated free-text EasyOCR ──────────

def ocr_text_cell(gray_img, y1, y2, x1, x2, pad=4):
    crop = gray_img[max(0, y1 + pad): y2 - pad, max(0, x1 + pad): x2 - pad]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    if crop.shape[1] < 150:
        scale = max(2, 150 // crop.shape[1])
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    results = _get_ocr_reader().readtext(
        crop, detail=0, paragraph=True, batch_size=1
    )
    text = re.sub(r'\s+', ' ', ' '.join(results).strip())
    return text if text else None


# ── Step 5: Page-level orchestration ──────────────────────────────────────────

def extract_page(gray, binary, page_num, verbose=True):
    h, w = binary.shape
    h_mask, v_mask = find_grid_lines(binary, h, w)
    row_pos = cluster_positions(h_mask, axis=1, min_gap=12)
    col_pos = cluster_positions(v_mask, axis=0, min_gap=18)

    if verbose:
        print(f"      Grid: {len(row_pos)} row lines × {len(col_pos)} col lines")

    if len(row_pos) < 3 or len(col_pos) < len(COLUMN_NAMES) + 1:
        print(f"      ⚠️  Grid detection insufficient — skipping page")
        return []

    accepted, skipped = [], 0

    for r in range(1, len(row_pos) - 1):    # row 0 = header, skip it
        y1, y2 = row_pos[r], row_pos[r + 1]
        if (y2 - y1) < 10:
            continue

        row_data = {}

        # DATE — the only VLM call this row
        row_data['DATE'] = ocr_date_with_vlm(
            gray, y1, y2, col_pos[0], col_pos[1], xback2=col_pos[2])

        # START_KM, END_KM, TOTAL_KM — strict OCR
        row_data['START_KM'] = ocr_numeric_cell(gray, y1, y2, col_pos[1], col_pos[2])
        row_data['END_KM']   = ocr_numeric_cell(gray, y1, y2, col_pos[4], col_pos[5])
        row_data['TOTAL_KM'] = ocr_numeric_cell(gray, y1, y2, col_pos[5], col_pos[6])

        # STARTING_POINT, DESTINATION — isolated free-text OCR
        row_data['STARTING_POINT'] = ocr_text_cell(gray, y1, y2, col_pos[2], col_pos[3])
        row_data['DESTINATION']    = ocr_text_cell(gray, y1, y2, col_pos[3], col_pos[4])

        # Right block — strict per-cell OCR
        for c in range(6, len(COLUMN_NAMES)):
            col_name = COLUMN_NAMES[c]
            x1, x2   = col_pos[c], col_pos[c + 1]
            row_data[col_name] = ocr_numeric_cell(gray, y1, y2, x1, x2)

        key_vals = [str(row_data.get(k) or '') for k in
                   ('START_KM', 'END_KM', 'TOTAL_KM')]
        if any(re.fullmatch(r'\d{2,}', v) for v in key_vals):
            accepted.append(row_data)
        else:
            skipped += 1

    if verbose and skipped:
        print(f"      (dropped {skipped} empty/unreadable rows)")

    return accepted


# ── Step 6: Date character normalization (NOT interpolation) ─────────────────

def normalize_date_chars(date_str):
    if not NORMALIZE_DATE_CHARACTERS or not isinstance(date_str, str):
        return date_str
    s = date_str.strip()
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('l', '1').replace('I', '1')
    s = s.replace('S', '5').replace('s', '5')
    s = s.replace('B', '8')
    return s


def parse_dates(df):
    df['DATE_RAW'] = df['DATE']
    cleaned = df['DATE'].apply(normalize_date_chars)
    parsed = pd.to_datetime(cleaned, format='%d/%m/%y', errors='coerce')

    bad = parsed.isna()
    for fmt in ['%d/%m/%Y', '%d-%m-%y']:
        if not bad.any():
            break
        fix = pd.to_datetime(cleaned[bad], format=fmt, errors='coerce')
        parsed[bad & fix.notna()] = fix[fix.notna()]
        bad = parsed.isna()

    n_failed = parsed.isna().sum()
    print(f"\n   Date parsing: {len(df) - n_failed}/{len(df)} parsed successfully")
    if n_failed > 0:
        print(f"   {n_failed} row(s) left as NaT:")
        print(df.loc[parsed.isna(), ['DATE_RAW', 'STARTING_POINT', 'TOTAL_KM']].to_string())

    return parsed


# ── Step 7: Implausible value flagging (NOT correction) ──────────────────────

def flag_implausible_provincial_values(df):
    n_flagged = 0
    for col in RIGHT_BLOCK_COLS:
        bound = (MAX_PLAUSIBLE_PROVINCIAL_KM if 'KMS' in col
                else MAX_PLAUSIBLE_PROVINCIAL_FUEL)
        too_high = df[col] > bound
        n_flagged += too_high.sum()
        df.loc[too_high, col] = np.nan
    if n_flagged:
        print(f"   ⚠️  {n_flagged} provincial value(s) exceeded plausibility bounds — marked unreadable")
    return df


# ── Step 8: Odometer continuity cross-check ───────────────────────────────────

def cross_validate_odometer_continuity(df):
    """
    START_KM of row N should equal END_KM of row N-1 — both record the
    same physical odometer reading from adjacent log entries. Recovers a
    null START_KM from a second, independent cell on the page; never
    overwrites a value that was actually read. Disagreements between two
    present values are flagged, not resolved — could be a real signal.
    """
    df = df.reset_index(drop=True).copy()
    df['START_KM_RECOVERED'] = False
    recovered = 0
    flagged_mismatches = []

    for i in range(1, len(df)):
        prev_end   = df.loc[i - 1, 'END_KM']
        this_start = df.loc[i, 'START_KM']

        if pd.isna(this_start) and pd.notna(prev_end):
            df.loc[i, 'START_KM'] = prev_end
            df.loc[i, 'START_KM_RECOVERED'] = True
            recovered += 1
        elif pd.notna(this_start) and pd.notna(prev_end):
            if abs(this_start - prev_end) > 5:
                flagged_mismatches.append((i, prev_end, this_start))

    print(f"\n   Odometer continuity check:")
    print(f"      Recovered {recovered} missing START_KM value(s)")
    if flagged_mismatches:
        print(f"      ⚠️  {len(flagged_mismatches)} mismatch(es) — left as-is, may be a real signal:")
        for i, pe, ts in flagged_mismatches:
            print(f"         Row {i}: prior END_KM={pe}, this START_KM={ts}")

    return df


# ── Step 9: Full multi-page pipeline ──────────────────────────────────────────

def extract_distance_log(pdf_path=PDF_PATH, dpi=DPI):
    print(f"Converting PDF at {dpi} DPI...")
    pages = convert_from_path(pdf_path, dpi=dpi)
    print(f"   {len(pages)} pages found\n")

    all_rows, page_counts = [], []
    for i, page in enumerate(pages):
        print(f"   Page {i + 1}...")
        gray, binary = preprocess(page)
        rows = extract_page(gray, binary, i)
        print(f"      ✅ {len(rows)} rows")
        page_counts.append(len(rows))
        all_rows.extend(rows)

    print(f"\n{'─' * 50}\nPer-page : {page_counts}\nTotal    : {sum(page_counts)}")

    if not all_rows:
        print("⚠️  Nothing extracted.")
        return pd.DataFrame(columns=COLUMN_NAMES)

    df = pd.DataFrame(all_rows)
    for col in COLUMN_NAMES:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMN_NAMES]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = flag_implausible_provincial_values(df)
    df['DATE'] = parse_dates(df)

    for col in ('STARTING_POINT', 'DESTINATION'):
        df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

    df = cross_validate_odometer_continuity(df)

    print(f"── Quality report ──────────────────────────────────")
    print(f"   Rows extracted       : {len(df)}")
    print(f"   Valid DATE           : {df['DATE'].notna().sum()}/{len(df)}")
    print(f"   Valid TOTAL_KM       : {df['TOTAL_KM'].notna().sum()}/{len(df)}")
    print(f"   STARTING_POINT null  : {df['STARTING_POINT'].isna().sum()}")
    print(f"   DESTINATION null     : {df['DESTINATION'].isna().sum()}")
    print(f"   START_KM recovered   : {df['START_KM_RECOVERED'].sum()}")
    print(f"   Shape                : {df.shape}")
    print(f"────────────────────────────────────────────────────\n")

    return df


if __name__ == '__main__':
    df_raw = extract_distance_log()
    print(df_raw[['DATE','START_KM','STARTING_POINT','DESTINATION',
                  'END_KM','TOTAL_KM']].to_string())
    print("\n── NaN counts ──────────────────────────────────────")
    print(df_raw.isna().sum().to_string())