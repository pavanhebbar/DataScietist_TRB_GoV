# ─────────────────────────────────────────────────────────────────────────────
# local_pdf_extractor_vlm.py
#
# Uses a local vision-language model (via Ollama) to read each full page
# of the scanned distance log directly, instead of cropping individual cells.
#
# Why this is different from grid + cell OCR:
#   - The model reads the WHOLE table at once and can use row/column context
#     to disambiguate handwriting — e.g. it can recognise that AB KMs is
#     never realistically >1500 for one trip, so it won't merge "1050"
#     with an adjacent "7" into "10507" the way isolated cell OCR did.
#   - No grid line detection, no per-cell cropping, no character whitelists.
#   - Critically: the prompt explicitly instructs null over guessing.
#     Missing or illegible values stay missing — never fabricated. This
#     matters because synthesizing a plausible date or distance would
#     corrupt the exact signal an anomaly detector relies on.
#
# Requirements:
#   pip install ollama pdf2image pandas
#   ollama pull qwen2.5vl:7b
# ─────────────────────────────────────────────────────────────────────────────

import base64
import io
import json
import re

import ollama
import pandas as pd
from pdf2image import convert_from_path

PDF_PATH = 'Data/Distancelog1page_test.pdf'
MODEL    = 'qwen2.5vl:7b'   # swap for minicpm-v or llava:13b if needed
DPI      = 220              # lower than 300 — keeps the image VLM-friendly

COLUMN_NAMES = [
    'DATE', 'START_KM', 'STARTING_POINT', 'DESTINATION', 'END_KM', 'TOTAL_KM',
    'AB_KMS', 'BC_KMS', 'SK_KMS', 'MB_KMS', 'ON_KMS', 'QC_KMS', 'YT_KMS',
    'AB_FUEL', 'BC_FUEL', 'SK_FUEL', 'MB_FUEL', 'ON_FUEL', 'QC_FUEL'
]

EXTRACTION_PROMPT = f"""This image is one page of a scanned fuel/distance log used for trucking IFTA compliance.

Extract the table into a JSON array. Each table row becomes one JSON object with exactly these keys, in this order:
{json.dumps(COLUMN_NAMES)}

Rules — follow exactly:
1. DATE is written DD/M/YY (e.g. "02/5/22"). Transcribe exactly as written.
2. All _KM and _FUEL fields are numbers. If a cell is empty, smudged, or illegible, use null. Never guess a value you cannot actually read.
3. STARTING_POINT and DESTINATION are place names like "Nisku, AB".
4. Ignore the header row, page numbers, and handwritten margin notes ("to F1" etc.) that aren't part of the grid.
5. Skip entirely blank rows.
6. Output ONLY the JSON array — no explanation, no markdown fences, no commentary.
"""


def page_to_base64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def extract_page_with_vlm(pil_image, page_num):
    img_b64 = page_to_base64(pil_image)

    response = ollama.chat(
        model=MODEL,
        messages=[{
            'role': 'user',
            'content': EXTRACTION_PROMPT,
            'images': [img_b64]
        }],
        options={'temperature': 0.0, 'num_ctx': 8192}   # deterministic
    )

    raw = response['message']['content'].strip()
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()

    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"      ⚠️  Page {page_num}: JSON parse failed ({e})")
        print(f"      Raw output (first 400 chars):\n{raw[:400]}")
        return []

    if not isinstance(rows, list):
        print(f"      ⚠️  Page {page_num}: expected a list, got {type(rows)}")
        return []

    print(f"      ✅ Page {page_num}: {len(rows)} rows")
    return rows


def extract_distance_log_vlm(pdf_path=PDF_PATH, dpi=DPI):
    print(f"Converting PDF at {dpi} DPI...")
    pages = convert_from_path(pdf_path, dpi=dpi)
    print(f"   {len(pages)} pages\n")

    all_rows = []
    for i, page in enumerate(pages):
        print(f"   Page {i + 1}...")
        rows = extract_page_with_vlm(page, i + 1)
        all_rows.extend(rows)

    if not all_rows:
        print("⚠️  Nothing extracted.")
        return pd.DataFrame(columns=COLUMN_NAMES)

    df = pd.DataFrame(all_rows)

    for col in COLUMN_NAMES:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMN_NAMES]

    # Numeric coercion — unreadable cells stay NaN, never silently zero
    numeric_cols = [c for c in COLUMN_NAMES
                    if c not in ('DATE', 'STARTING_POINT', 'DESTINATION')]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"\n── Quality report ─────────────────────────────")
    print(f"   Rows extracted     : {len(df)}")
    print(f"   DATE present       : {df['DATE'].notna().sum()}/{len(df)}")
    print(f"   TOTAL_KM present   : {df['TOTAL_KM'].notna().sum()}/{len(df)}")
    print(f"────────────────────────────────────────────────\n")

    return df


df_raw = extract_distance_log_vlm()
print(df_raw.head(20).to_string())
print("\n── NaN counts ──────────────────────────────")
print(df_raw.isna().sum().to_string())