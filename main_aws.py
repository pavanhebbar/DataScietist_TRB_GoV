"""
IFTA AWS Pipeline — Master Orchestrator
Runs the full end-to-end pipeline in order:
  Step 1: Extract raw files → S3 (Textract + boto3)
  Step 2: Deploy Athena schema and merged view
  Step 3: Build engineered feature table (Athena CTAS → Parquet)
  Step 4: Run audit pipeline (Athena Stage 1 + SageMaker Stage 2 + S3 merge)

Run from project root:
    python main_aws.py
    python main_aws.py --skip-extraction   # if data already in S3
    python main_aws.py --skip-schema       # if tables already deployed
    python main_aws.py --contamination 0.05
"""

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR     = Path(__file__).parent
UTILS_AWS    = ROOT_DIR / 'utils_aws'
sys.path.insert(0, str(UTILS_AWS))

# ── Pipeline module imports ───────────────────────────────────────────────────
from extract_utils_aws      import run_cloud_extraction_pipeline
from merge_utils_athena     import deploy_athena_schema
from feature_engineering_aws import build_analytical_features_table
from analysis_aws           import run_aws_audit_pipeline

# ── Configuration ─────────────────────────────────────────────────────────────
PRIMARY_BUCKET   = 'prh-ifta-audit-lake'
DATA_LAKE_REGION = 'ca-west-1'

# Local raw files — update paths to match your directory structure
TARGET_FILES = [
    './Data/Distance log 1.xlsx',
    './Data/Distance log 2.pdf',
    './Data/Fuel Invoices.docx',
]
FILE_TYPES = ['distance', 'distance', 'invoice']


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(step_num, title):
    print(f"\n{'=' * 60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'=' * 60}")


def _elapsed(start):
    secs = time.time() - start
    mins, secs = divmod(int(secs), 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_extract(bucket_name, skip=False):
    """Extract raw files via Textract and upload CSVs to S3."""
    _banner(1, 'DATA EXTRACTION → S3')

    if skip:
        print('   ⏭  Skipped — using existing S3 data.')
        return

    missing = [f for f in TARGET_FILES if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Raw files not found locally:\n  " + "\n  ".join(missing) +
            "\n  Update TARGET_FILES in main_aws.py to match your paths."
        )

    t = time.time()
    run_cloud_extraction_pipeline(
        file_list=TARGET_FILES,
        file_quant_list=FILE_TYPES,
        bucket_name=bucket_name
    )
    print(f"\n   ✅ Extraction complete ({_elapsed(t)})")
    print(f"   → s3://{bucket_name}/ocr-extracted-csv/")


def step2_schema(bucket_name, region, skip=False):
    """Deploy Athena external tables and merged view."""
    _banner(2, 'ATHENA SCHEMA + MERGED VIEW DEPLOYMENT')

    if skip:
        print('   ⏭  Skipped — using existing Athena tables.')
        return

    t = time.time()
    deploy_athena_schema(bucket_name=bucket_name, region=region)
    print(f"\n   ✅ Schema deployed ({_elapsed(t)})")
    print(f"   → Tables: ifta_extracted_invoices, ifta_extracted_distance_logs")
    print(f"   → View:   ifta_audit_merged_view")


def step3_features(bucket_name, region):
    """Build engineered feature table via Athena CTAS → Parquet."""
    _banner(3, 'FEATURE ENGINEERING → PARQUET (Athena CTAS)')

    t = time.time()
    build_analytical_features_table(bucket_name=bucket_name, region=region)
    print(f"\n   ✅ Feature table built ({_elapsed(t)})")
    print(f"   → s3://{bucket_name}/engineered-features/")
    print(f"   → Table: ifta_audit_features_table")


def step4_audit(bucket_name, region, contamination):
    """
    Run the two-stage audit pipeline:
      Stage 1 — Deterministic 3σ rules in Athena
      Stage 2 — Isolation Forest in SageMaker Processing Job
      Stage 3 — Merge and write master audit ledger to S3
    """
    _banner(4, 'AUDIT PIPELINE (Athena + SageMaker + S3)')

    t = time.time()
    df_master = run_aws_audit_pipeline(
        contamination=contamination,
        bucket_name=bucket_name,
        region=region
    )
    print(f"\n   ✅ Audit pipeline complete ({_elapsed(t)})")
    print(f"   → s3://{bucket_name}/audit-results/master_audit_ledger.csv")
    return df_master


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df_master, total_start):
    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Total runtime : {_elapsed(total_start)}")
    print(f"  Total flagged : {len(df_master)} records")

    if 'anomaly_source' in df_master.columns:
        print(f"\n  By detection stage:")
        for source, count in df_master['anomaly_source'].value_counts().items():
            print(f"    {source:<35} {count:>4} records")

    if 'deterministic_rule' in df_master.columns:
        det = df_master[df_master['anomaly_source'] == 'Deterministic Z-Score Rule']
        if not det.empty:
            print(f"\n  Top deterministic rules fired:")
            for rule, count in det['deterministic_rule'].value_counts().head(5).items():
                print(f"    {rule:<35} {count:>4} records")

    print(f"\n  S3 outputs:")
    print(f"    s3://{PRIMARY_BUCKET}/audit-stage1/flagged/")
    print(f"    s3://{PRIMARY_BUCKET}/audit-stage2/")
    print(f"    s3://{PRIMARY_BUCKET}/audit-results/master_audit_ledger.csv")
    print(f"\n  Query results in Athena:")
    print(f"    SELECT * FROM default.ifta_audit_features_table LIMIT 10;")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IFTA AWS Audit Pipeline'
    )
    parser.add_argument(
        '--skip-extraction', action='store_true',
        help='Skip Step 1 — use CSV files already in S3'
    )
    parser.add_argument(
        '--skip-schema', action='store_true',
        help='Skip Step 2 — use Athena tables already deployed'
    )
    parser.add_argument(
        '--contamination', type=float, default=0.03,
        help='Isolation Forest contamination rate (default: 0.03)'
    )
    parser.add_argument(
        '--bucket', type=str, default=PRIMARY_BUCKET,
        help=f'S3 bucket name (default: {PRIMARY_BUCKET})'
    )
    parser.add_argument(
        '--region', type=str, default=DATA_LAKE_REGION,
        help=f'AWS region (default: {DATA_LAKE_REGION})'
    )
    args = parser.parse_args()

    print(f"\n  Bucket     : s3://{args.bucket}")
    print(f"  Region     : {args.region}")
    print(f"  Contamination rate: {args.contamination:.0%}")

    total_start = time.time()

    try:
        step1_extract(args.bucket, skip=args.skip_extraction)
        step2_schema(args.bucket, args.region, skip=args.skip_schema)
        step3_features(args.bucket, args.region)
        df_master = step4_audit(args.bucket, args.region, args.contamination)
        print_summary(df_master, total_start)

    except FileNotFoundError as e:
        print(f"\n❌ File error: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Pipeline failed at runtime: {e}")
        raise


if __name__ == '__main__':
    main()