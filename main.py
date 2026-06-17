"""
IFTA Audit ML Pipeline — Main Orchestrator
Run from project root: python main.py
    python main.py --skip-extraction
    python main.py --skip-extraction --skip-feature-engineering
    python main.py --skip-visualization

Project structure:
    main.py
    utils/
        extract_utils.py
        merge_utils.py
        feature_engineering.py
        analysis.py
        visualization.py
    Data/
        Fuel Invoices.docx
        Distance log 1.xlsx
        Distance log 2.pdf
        outputs/          ← generated automatically
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
UTILS_DIR  = ROOT_DIR  / 'utils'
RAW_DIR    = ROOT_DIR  / 'Data'
OUTPUT_DIR = ROOT_DIR  / 'Data' / 'outputs'
sys.path.insert(0, str(UTILS_DIR))

# ── Imports ───────────────────────────────────────────────────────────────────
from extract_utils import readfile_to_df
from merge_utils import build_unified_ml_dataset
from feature_engineering import engineer_ifta_features
from analysis import execute_full_audit_pipeline, \
                     contamination_sensitivity_check
from visualization import run_full_eda_pipeline

# ── File paths (update filenames to match yours) ──────────────────────────────
INVOICE_FILE   = RAW_DIR / 'Fuel Invoices.docx'
DISTANCE_LOG1  = RAW_DIR / 'Distance log 1.xlsx'
DISTANCE_LOG2  = RAW_DIR / 'Distance log 2.pdf'

CONTAMINATION_RATE = 0.03

# Date columns that need re-parsing when a cached CSV is reloaded
# rather than freshly computed in this run.
DATE_COLS = ['trip_date', 'invoice_date']


# ── Stage functions ───────────────────────────────────────────────────────────

def stage1_extract():
    """Extract raw data from all three source files."""
    print("\n" + "=" * 50)
    print("STAGE 1 — DATA EXTRACTION")
    print("=" * 50)

    # Invoices: images inside .docx → Textract analyze_expense
    print("\n📄 Extracting invoices from .docx (Textract analyze_expense)...")
    df_invoices, _ = readfile_to_df(str(INVOICE_FILE), filequant='invoice')
    print(f"   {len(df_invoices)} invoice records")

    # Distance Log 1: Excel spreadsheet (2016–2021) — truck 1
    print("\n📊 Loading Distance Log 1 (Excel)...")
    df_log1, _ = readfile_to_df(str(DISTANCE_LOG1), filequant='distance',
                                truck_number=1)
    print(f"   {len(df_log1)} rows")

    # Distance Log 2: Multi-page PDF → Textract async via S3 — truck 2
    print("\n📋 Extracting Distance Log 2 (PDF via Textract async)...")
    df_log2, _ = readfile_to_df(str(DISTANCE_LOG2), filequant='distance',
                                truck_number=2)
    print(f"   {len(df_log2)} rows")

    return df_invoices, df_log1, df_log2


def stage2_merge(df_invoices, df_log1, df_log2):
    """Merge all sources into one ML-ready dataset."""
    print("\n" + "=" * 60)
    print("STAGE 2 — MERGING & PREPARATION")
    print("=" * 60)

    df_ml = build_unified_ml_dataset(df_log1, df_log2, df_invoices)
    print(f"\n   ✅ Unified dataset: {df_ml.shape[0]} rows × {df_ml.shape[1]} cols")

    vouched = (df_ml['matching_invoice_number'] != 'NO MATCHING INVOICE').sum()
    print(f"    Vouched trips (invoice matched): {vouched} / {len(df_ml)}")
    print(f"    Unvouched trips:  {len(df_ml) - vouched} / {len(df_ml)}")

    _save(df_ml, 'ml_ready_dataset.csv')
    return df_ml


def stage3_features(df_ml):
    """Engineer IFTA-relevant features."""
    print("\n" + "=" * 60)
    print("STAGE 3 — FEATURE ENGINEERING")
    print("=" * 60)

    df_feat = engineer_ifta_features(df_ml)

    print(f"\n   📊 Fuel source breakdown:")
    print(df_feat['fuel_source'].value_counts().to_string())

    missing_fuel = df_feat['fuel_source'].eq('missing').sum()
    missing_dist = df_feat['total_km'].eq(0).sum()
    print(f"\n    Records with missing fuel data: {missing_fuel}")
    print(f"     Records with zero distance:    {missing_dist}")

    print(f"\n   Feature matrix: {df_feat.shape[0]} rows × {df_feat.shape[1]} cols")
    _save(df_feat, 'featured_dataset.csv')
    return df_feat


def stage4_analysis(df_feat):
    """Run sensitivity check then full anomaly detection pipeline."""
    print("\n" + "=" * 60)
    print("STAGE 4 — ANOMALY DETECTION")
    print("=" * 60)

    print("\n Contamination rate sensitivity check...")
    sensitivity_df = contamination_sensitivity_check(df_feat)
    _save(sensitivity_df, 'contamination_sensitivity.csv')

    df_clean, df_audit, df_importance = execute_full_audit_pipeline(
        df_feat, contamination=CONTAMINATION_RATE
    )

    _save(df_audit,      'audit_ledger.csv')
    _save(df_clean,      'clean_trips.csv')
    _save(df_importance, 'feature_importance.csv')
    return df_clean, df_audit, df_importance


def print_summary(df_audit, df_importance):
    """Final summary printed to console."""
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — AUDIT SUMMARY")
    print("=" * 60)
    print(f"\n🎯 Total flagged records : {len(df_audit)}")
    print(f"\n📊 By detection stage:")
    print(df_audit['anomaly_source'].value_counts().to_string())
    print(f"\n🔑 Top anomaly drivers:")
    print(df_importance.head(5).to_string(index=False))
    print(f"\n💾 All outputs saved to: {OUTPUT_DIR}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(df, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"   💾 Saved → {path.name}")


def _load(filename, date_cols=None):
    """Loads a previously-saved stage output, for use with --skip-* flags."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected cached file not found: {path}\n"
            f"Run the pipeline at least once without this skip flag first."
        )
    parse_dates = [c for c in (date_cols or []) ]
    df = pd.read_csv(path, parse_dates=parse_dates)
    print(f"   📂 Loaded ← {path.name} ({len(df)} rows)")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='IFTA Audit ML Pipeline')
    parser.add_argument(
        '--skip-extraction', action='store_true',
        help='Skip Stage 1 (extraction) AND Stage 2 (merge) — there is no '
             'standalone cache point between them — and load '
             'Data/outputs/ml_ready_dataset.csv from a previous run instead.'
    )
    parser.add_argument(
        '--skip-feature-engineering', action='store_true',
        help='Skip Stage 3 and load Data/outputs/featured_dataset.csv '
             'from a previous run instead.'
    )
    parser.add_argument(
        '--skip-visualization', action='store_true',
        help='Skip the EDA visualization pipeline.'
    )
    args = parser.parse_args()

    if args.skip_extraction:
        print("\n⏭  Skipping extraction + merge (no cache point exists "
              "between them) — loading ml_ready_dataset.csv")
        df_ml = _load('ml_ready_dataset.csv', date_cols=DATE_COLS)
    else:
        df_invoices, df_log1, df_log2 = stage1_extract()
        df_ml = stage2_merge(df_invoices, df_log1, df_log2)

    if args.skip_feature_engineering:
        print("\n⏭  Skipping feature engineering — loading featured_dataset.csv")
        df_feat = _load('featured_dataset.csv', date_cols=DATE_COLS)
    else:
        df_feat = stage3_features(df_ml)

    if args.skip_visualization:
        print("\n⏭  Skipping visualization")
    else:
        print("\n" + "=" * 60)
        print("VISUALIZATION — EDA PIPELINE")
        print("=" * 60)
        run_full_eda_pipeline(df_feat)

    df_clean, df_audit, df_importance = stage4_analysis(df_feat)
    print_summary(df_audit, df_importance)


if __name__ == '__main__':
    main()