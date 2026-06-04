"""
AWS Audit Pipeline Orchestrator
Stage 1: Deterministic rules via Athena SQL
Stage 2: Isolation Forest via SageMaker Processing Job
Stage 3: Merge and write final audit ledger to S3
"""

import io
import time
import boto3
import pandas as pd
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from pathlib import Path
UTILS_AWS_DIR = Path(__file__).parent

DATA_LAKE_REGION = 'ca-west-1'
PRIMARY_BUCKET   = 'prh-ifta-audit-lake'


# ── Stage 1: Athena Deterministic Rules ──────────────────────────────────────

STAGE1_FLAGGED_PREFIX   = 'audit-stage1/flagged/'
STAGE1_CLEAN_PREFIX     = 'audit-stage1/clean-pool/'

def run_athena_stage1(bucket_name=PRIMARY_BUCKET, region=DATA_LAKE_REGION):
    """
    Runs the deterministic 3σ and IFTA compliance rules entirely in Athena SQL.
    Writes two CTAS outputs to S3:
      - audit-stage1/flagged/    → deterministic violations
      - audit-stage1/clean-pool/ → records passed to Isolation Forest
    
    Offloads Stage 1 compute to Athena serverless — no local pandas needed.
    """
    athena = boto3.client('athena', region_name=region)
    s3     = boto3.resource('s3', region_name=region)

    # Clear old CTAS outputs to prevent collision errors
    for prefix in [STAGE1_FLAGGED_PREFIX, STAGE1_CLEAN_PREFIX]:
        s3.Bucket(bucket_name).objects.filter(Prefix=prefix).delete()

    # Build the deterministic rules entirely in SQL
    # Uses approx_percentile for median (robust central tendency for rolling avgs)
    stage1_sql = f"""
    WITH population_stats AS (
        SELECT
            avg(total_km)                                    AS mean_km,
            stddev_samp(total_km)                            AS std_km,
            avg(total_fuel_litres)                           AS mean_fuel,
            stddev_samp(total_fuel_litres)                   AS std_fuel,
            avg(fuel_litres_per_km)                          AS mean_intensity,
            stddev_samp(fuel_litres_per_km)                  AS std_intensity,
            avg(cycle_fuel_per_km)                           AS mean_cycle,
            stddev_samp(cycle_fuel_per_km)                   AS std_cycle,
            avg(km_per_tank)                                 AS mean_tank_dist,
            stddev_samp(km_per_tank)                         AS std_tank_dist,
            avg(avg_fuel_per_tank)                           AS mean_tank_fuel,
            stddev_samp(avg_fuel_per_tank)                   AS std_tank_fuel,
            avg(dist_to_weekly_ratio)                        AS mean_rw_dist,
            stddev_samp(dist_to_weekly_ratio)                AS std_rw_dist,
            avg(fuel_to_weekly_ratio)                        AS mean_rw_fuel,
            stddev_samp(fuel_to_weekly_ratio)                AS std_rw_fuel,
            avg(dist_to_monthly_ratio)                       AS mean_rm_dist,
            stddev_samp(dist_to_monthly_ratio)               AS std_rm_dist,
            avg(fuel_to_monthly_ratio)                       AS mean_rm_fuel,
            stddev_samp(fuel_to_monthly_ratio)               AS std_rm_fuel,
            approx_percentile(weekly_avg_distance_km,  0.5)  AS median_w_dist,
            stddev_samp(weekly_avg_distance_km)              AS std_w_dist,
            approx_percentile(weekly_avg_fuel_litres,  0.5)  AS median_w_fuel,
            stddev_samp(weekly_avg_fuel_litres)              AS std_w_fuel,
            approx_percentile(monthly_avg_distance_km, 0.5)  AS median_m_dist,
            stddev_samp(monthly_avg_distance_km)             AS std_m_dist,
            approx_percentile(monthly_avg_fuel_litres, 0.5)  AS median_m_fuel,
            stddev_samp(monthly_avg_fuel_litres)             AS std_m_fuel
        FROM default.ifta_audit_features_table
        WHERE total_km IS NOT NULL
    ),
    evaluated AS (
        SELECT f.*,
            CASE
                WHEN f.total_km > s.mean_km + 3 * s.std_km
                    THEN 'dist_z_score'
                WHEN f.total_fuel_litres > s.mean_fuel + 3 * s.std_fuel
                    THEN 'fuel_z_score'
                WHEN f.fuel_litres_per_km > s.mean_intensity + 3 * s.std_intensity
                    THEN 'trip_intensity_z'
                WHEN f.cycle_fuel_per_km > s.mean_cycle + 3 * s.std_cycle
                  OR f.cycle_fuel_per_km < 0.05
                    THEN 'cycle_intensity_violation'
                WHEN f.km_per_tank > s.mean_tank_dist + 3 * s.std_tank_dist
                    THEN 'tank_distance_z'
                WHEN f.avg_fuel_per_tank > s.mean_tank_fuel + 3 * s.std_tank_fuel
                  OR f.avg_fuel_per_tank < 100
                    THEN 'tank_fuel_z'
                WHEN f.km_reconciliation_gap >= 10
                    THEN 'km_reconciliation_gap'
                WHEN f.fuel_source_reliability = 0
                    THEN 'missing_fuel'
                WHEN f.odometer_gap >= 50
                    THEN 'odometer_gap'
                WHEN f.ab_fuel_prop = 1.0 AND f.ab_dist_prop = 0.0
                    THEN 'ifta_boundary_violation'
                WHEN f.dist_to_weekly_ratio  > s.mean_rw_dist + 3 * s.std_rw_dist
                  OR f.fuel_to_weekly_ratio  > s.mean_rw_fuel + 3 * s.std_rw_fuel
                    THEN 'weekly_ratio_outlier'
                WHEN f.dist_to_monthly_ratio > s.mean_rm_dist + 3 * s.std_rm_dist
                  OR f.fuel_to_monthly_ratio > s.mean_rm_fuel + 3 * s.std_rm_fuel
                    THEN 'monthly_ratio_outlier'
                WHEN f.weekly_avg_distance_km > s.median_w_dist + 3 * s.std_w_dist
                  OR f.weekly_avg_fuel_litres > s.median_w_fuel + 3 * s.std_w_fuel
                  OR (f.weekly_avg_distance_km > 0 AND (
                        f.weekly_avg_distance_km < s.median_w_dist - 3 * s.std_w_dist
                     OR f.weekly_avg_fuel_litres < s.median_w_fuel - 3 * s.std_w_fuel))
                    THEN 'weekly_macro_drift'
                WHEN f.monthly_avg_distance_km > s.median_m_dist + 3 * s.std_m_dist
                  OR f.monthly_avg_fuel_litres > s.median_m_fuel + 3 * s.std_m_fuel
                  OR (f.monthly_avg_distance_km > 0 AND (
                        f.monthly_avg_distance_km < s.median_m_dist - 3 * s.std_m_dist
                     OR f.monthly_avg_fuel_litres < s.median_m_fuel - 3 * s.std_m_fuel))
                    THEN 'monthly_macro_drift'
                ELSE NULL
            END AS deterministic_rule
        FROM default.ifta_audit_features_table f
        CROSS JOIN population_stats s
    )
    """

    # CTAS 1: Write flagged records to S3
    flagged_sql = f"""
    CREATE TABLE default.ifta_stage1_flagged
    WITH (format='PARQUET',
          external_location='s3://{bucket_name}/{STAGE1_FLAGGED_PREFIX}')
    AS
    {stage1_sql}
    SELECT *, 'Deterministic Z-Score Rule' AS anomaly_source, -1.0 AS anomaly_score
    FROM evaluated
    WHERE deterministic_rule IS NOT NULL;
    """

    # CTAS 2: Write clean pool to S3 for Stage 2
    clean_sql = f"""
    CREATE TABLE default.ifta_stage1_clean_pool
    WITH (format='PARQUET',
          external_location='s3://{bucket_name}/{STAGE1_CLEAN_PREFIX}')
    AS
    {stage1_sql}
    SELECT *
    FROM evaluated
    WHERE deterministic_rule IS NULL;
    """

    print("\n🔍 Stage 1: Running deterministic rules in Athena...")

    for label, sql in [('DROP old tables', None),
                       ('flagged records', flagged_sql),
                       ('clean pool',      clean_sql)]:
        if sql is None:
            _athena_query("DROP TABLE IF EXISTS default.ifta_stage1_flagged;",
                          athena, bucket_name)
            _athena_query("DROP TABLE IF EXISTS default.ifta_stage1_clean_pool;",
                          athena, bucket_name)
        else:
            print(f"   Writing {label}...")
            _athena_query(sql, athena, bucket_name)

    print("   ✅ Stage 1 complete.")


# ── Stage 2: SageMaker Processing Job ────────────────────────────────────────

def run_sagemaker_stage2(contamination=0.03,
                          bucket_name=PRIMARY_BUCKET,
                          region=DATA_LAKE_REGION):

    try:
        role = sagemaker.get_execution_role()
    except Exception:
        role = 'arn:aws:iam::082787299163:role/SageMakerExecutionRole'

    # Region is passed through a boto3 session → sagemaker session
    # NOT directly to SKLearnProcessor
    boto_session      = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role,
        instance_type='ml.t3.medium',
        instance_count=1,
        base_job_name='ifta-audit-isolation-forest',
        sagemaker_session=sagemaker_session  # ← region comes through here
    )

    print("\n🌲 Stage 2: Submitting Isolation Forest to SageMaker Processing...")
    print(UTILS_AWS_DIR / 'analysis_bysagemaker.py')
    print((UTILS_AWS_DIR / 'analysis_bysagemaker.py').exists())
    processor.run(
        code=str(UTILS_AWS_DIR / 'analysis_bysagemaker.py'),
        inputs=[
            ProcessingInput(
                source=f's3://{bucket_name}/{STAGE1_CLEAN_PREFIX}',
                destination='/opt/ml/processing/input/features'
            )
        ],
        outputs=[
            ProcessingOutput(
                source='/opt/ml/processing/output',
                destination=f's3://{bucket_name}/audit-stage2/'
            )
        ],
        arguments=['--contamination', str(contamination)],
        wait=True,
        logs=True
    )


# ── Stage 3: Merge and write final audit ledger ───────────────────────────────

def merge_and_save_final_ledger(bucket_name=PRIMARY_BUCKET,
                                 region=DATA_LAKE_REGION):
    """
    Reads Stage 1 (Athena CTAS) and Stage 2 (SageMaker) outputs from S3,
    merges into a single audit ledger, and writes the final result.
    """
    s3 = boto3.client('s3', region_name=region)

    def read_parquet_prefix(prefix):
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        keys = [o['Key'] for o in response.get('Contents', [])
                if o['Key'].endswith('.parquet') or o['Key'].endswith('.csv')]
        frames = []
        for key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            body = obj['Body'].read()
            if key.endswith('.parquet'):
                frames.append(pd.read_parquet(io.BytesIO(body)))
            else:
                frames.append(pd.read_csv(io.BytesIO(body)))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print("\n📋 Stage 3: Merging audit ledger...")
    df_stage1 = read_parquet_prefix(STAGE1_FLAGGED_PREFIX)
    df_stage2 = read_parquet_prefix('audit-stage2/stage2_flagged')

    df_master = pd.concat([df_stage1, df_stage2], ignore_index=True, sort=False)

    buf = io.StringIO()
    df_master.to_csv(buf, index=False, lineterminator='\n')
    s3.put_object(
        Bucket=bucket_name,
        Key='audit-results/master_audit_ledger.csv',
        Body=buf.getvalue()
    )

    print(f"   ✅ Master ledger: {len(df_master)} records")
    print(f"      Stage 1 (Athena):    {len(df_stage1)}")
    print(f"      Stage 2 (SageMaker): {len(df_stage2)}")
    print(f"      → s3://{bucket_name}/audit-results/master_audit_ledger.csv")
    return df_master


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_aws_audit_pipeline(contamination=0.03,
                            bucket_name=PRIMARY_BUCKET,
                            region=DATA_LAKE_REGION):
    print("\n" + "=" * 60)
    print("IFTA AWS AUDIT PIPELINE")
    print("=" * 60)

    run_athena_stage1(bucket_name, region)
    run_sagemaker_stage2(contamination, bucket_name, region)
    df_master = merge_and_save_final_ledger(bucket_name, region)

    print("\n🎯 Pipeline complete.")
    return df_master


# ── Helpers ───────────────────────────────────────────────────────────────────

def _athena_query(sql, client, bucket_name):
    response = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={'Database': 'default'},
        ResultConfiguration={
            'OutputLocation': f's3://{bucket_name}/athena-query-results/'
        }
    )
    qid = response['QueryExecutionId']
    while True:
        state = client.get_query_execution(
            QueryExecutionId=qid
        )['QueryExecution']['Status']['State']
        if state == 'SUCCEEDED':
            return qid
        if state in ['FAILED', 'CANCELLED']:
            raise Exception(f"Athena query {qid} failed")
        time.sleep(2)


if __name__ == '__main__':
    run_aws_audit_pipeline()