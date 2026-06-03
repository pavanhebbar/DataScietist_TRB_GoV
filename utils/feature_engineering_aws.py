"""Python program to engineer features"""

import boto3
import time


def build_analytical_features_table(bucket_name='prh-ifta-audit-lake',
                                    region='ca-west-1'):
    """
    Compiles the advanced temporal window features into a physical,
    pre-computed Parquet table in Athena/S3, utilizing TRY_CAST for data safety.
    """
    athena_client = boto3.client('athena', region_name=region)
    s3_resource = boto3.resource('s3')
    
    table_name = "default.ifta_audit_features_table"
    target_s3_location = f"s3://{bucket_name}/engineered-features/"
    
    # 1. Drop old table structure
    print(f"Dropping old table structure for {table_name}...")
    response = athena_client.start_query_execution(
        QueryString=f"DROP TABLE IF EXISTS {table_name};",
        QueryExecutionContext={'Database': 'default'},
        ResultConfiguration={'OutputLocation': f's3://{bucket_name}/athena-query-results/'}
    )
    wait_for_athena_query(athena_client, response['QueryExecutionId'])

    # 2. Clear out physical S3 backend objects to prevent CTAS collision errors
    print("Clearing old data files from S3 directory...")
    bucket = s3_resource.Bucket(bucket_name)
    bucket.objects.filter(Prefix="engineered-features/").delete()

    # 3. Comprehensive Feature Compilation SQL (CTAS)
    ctas_sql = f"""
    CREATE TABLE {table_name}
    WITH (
        format = 'PARQUET',
        external_location = '{target_s3_location}'
    ) AS
    WITH base_metrics AS (
        SELECT
            trip_date,
            TRY_CAST(TRIM(trip_date) AS DATE) AS t_date,
            trip_origin, trip_destination, trip_start_time, trip_end_time,
            start_odometer, end_odometer, distance_km, vin_or_truck_number,
            ab_kms, bc_kms, sk_kms, mb_kms, on_kms,
            ab_fuel, bc_fuel, sk_fuel, mb_fuel, on_fuel, log_source_file,
            invoice_number, invoice_date, invoice_time, vendor_name,
            purchase_city, purchase_province, fuel_type, fuel_grade,
            fuel_litres_purchased, cost_per_litre, fuel_cost,
            total_tax, fed_tax, prov_tax, payment_form, invoice_source_file,
            days_difference,
            
            (COALESCE(ab_kms, 0) + COALESCE(bc_kms, 0) + COALESCE(sk_kms, 0) + COALESCE(mb_kms, 0) + COALESCE(on_kms, 0)) AS total_km,
            
            COALESCE(
                fuel_litres_purchased, 
                (COALESCE(ab_fuel, 0) + COALESCE(bc_fuel, 0) + COALESCE(sk_fuel, 0) + COALESCE(mb_fuel, 0) + COALESCE(on_fuel, 0))
            ) AS total_fuel_litres
            
        FROM default.ifta_audit_merged_view
    ),
    filtered_metrics AS (
        -- New Step: Explicitly filter out any rows that failed the date conversion
        -- This keeps your downstream temporal window calculations clean and error-free
        SELECT * FROM base_metrics
        WHERE t_date IS NOT NULL
    ),
    temporal_fuel_windows AS (
        SELECT 
            *,
            -- Lookback/Lookahead 3 days to find ANY fuel logged/purchased for this specific truck
            SUM(total_fuel_litres) OVER(
                PARTITION BY vin_or_truck_number 
                ORDER BY t_date 
                RANGE BETWEEN INTERVAL '3' DAY PRECEDING AND INTERVAL '3' DAY FOLLOWING
            ) AS rolling_3d_fuel_window,
            
            -- Locate the exact date of the next fuel purchase event for this truck
            MIN(CASE WHEN total_fuel_litres > 0 THEN t_date END) OVER(
                PARTITION BY vin_or_truck_number 
                ORDER BY t_date 
                ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
            ) AS next_fuel_event_date,
            
            -- Dynamic grouping to bound segments leading up to a fuel purchase
            COUNT(CASE WHEN total_fuel_litres > 0 THEN 1 END) OVER(
                PARTITION BY vin_or_truck_number 
                ORDER BY t_date 
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            ) AS fuel_segment_block
        FROM filtered_metrics
    ),
    calculated_efficiency AS (
        SELECT 
            *,
            -- Compute distance accumulated within the current fueling segment block
            SUM(total_km) OVER(
                PARTITION BY vin_or_truck_number, fuel_segment_block
            ) AS total_km_accumulated_before_next_fuel,
            
            -- Days until the next fuel receipt appears
            DATE_DIFF('day', t_date, next_fuel_event_date) AS days_until_next_fuel,
            
            -- Downstream ratios
            total_fuel_litres / NULLIF(total_km, 0) AS fuel_litres_per_km,
            fuel_cost / NULLIF(total_km, 0) AS fuel_cost_per_km,
            
            -- Unvouched trip if truck moved but 0 fuel exists across the entire 6-day window
            CASE 
                WHEN total_km > 0 AND (rolling_3d_fuel_window IS NULL OR rolling_3d_fuel_window = 0) THEN 1 
                ELSE 0 
            END AS is_unvouched_trip,
            
            -- High-probability missing invoice indicator (>7 days between fueling logs)
            CASE 
                WHEN DATE_DIFF('day', t_date, next_fuel_event_date) > 7 THEN 1 
                ELSE 0 
            END AS missing_invoice_gap_flag
            
        FROM temporal_fuel_windows
    ),
    jurisdictional_proportions AS (
        SELECT 
            *,
            (COALESCE(ab_kms, 0) / NULLIF(total_km, 0)) AS ab_dist_prop,
            (COALESCE(ab_fuel, 0) / NULLIF(total_fuel_litres, 0)) AS ab_fuel_prop,
            ((COALESCE(ab_kms, 0) / NULLIF(total_km, 0)) - (COALESCE(ab_fuel, 0) / NULLIF(total_fuel_litres, 0))) AS ab_dist_fuel_variance,
            
            (COALESCE(bc_kms, 0) / NULLIF(total_km, 0)) AS bc_dist_prop,
            (COALESCE(bc_fuel, 0) / NULLIF(total_fuel_litres, 0)) AS bc_fuel_prop,
            ((COALESCE(bc_kms, 0) / NULLIF(total_km, 0)) - (COALESCE(bc_fuel, 0) / NULLIF(total_fuel_litres, 0))) AS bc_dist_fuel_variance,
            
            (COALESCE(sk_kms, 0) / NULLIF(total_km, 0)) AS sk_dist_prop,
            (COALESCE(sk_fuel, 0) / NULLIF(total_fuel_litres, 0)) AS sk_fuel_prop,
            ((COALESCE(sk_kms, 0) / NULLIF(total_km, 0)) - (COALESCE(sk_fuel, 0) / NULLIF(total_fuel_litres, 0))) AS sk_dist_fuel_variance,
            
            (COALESCE(mb_kms, 0) / NULLIF(total_km, 0)) AS mb_dist_prop,
            (COALESCE(mb_fuel, 0) / NULLIF(total_fuel_litres, 0)) AS mb_fuel_prop,
            ((COALESCE(mb_kms, 0) / NULLIF(total_km, 0)) - (COALESCE(mb_fuel, 0) / NULLIF(total_fuel_litres, 0))) AS mb_dist_fuel_variance,
            
            (COALESCE(on_kms, 0) / NULLIF(total_km, 0)) AS on_dist_prop,
            (COALESCE(on_fuel, 0) / NULLIF(total_fuel_litres, 0)) AS on_fuel_prop,
            ((COALESCE(on_kms, 0) / NULLIF(total_km, 0)) - (COALESCE(on_fuel, 0) / NULLIF(total_fuel_litres, 0))) AS on_dist_fuel_variance
        FROM calculated_efficiency
    ),
    daily_timeline_aggregates AS (
        SELECT 
            t_date,
            SUM(total_km) AS daily_sum_km,
            SUM(total_fuel_litres) AS daily_sum_fuel
        FROM jurisdictional_proportions
        GROUP BY t_date
    ),
    temporal_moving_windows AS (
        SELECT 
            t_date,
            AVG(daily_sum_km) OVER(ORDER BY t_date RANGE BETWEEN INTERVAL '1' DAY PRECEDING AND INTERVAL '1' DAY PRECEDING) AS daily_avg_distance_km,
            AVG(daily_sum_km) OVER(ORDER BY t_date RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND INTERVAL '1' DAY PRECEDING) AS weekly_avg_distance_km,
            AVG(daily_sum_km) OVER(ORDER BY t_date RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND INTERVAL '1' DAY PRECEDING) AS monthly_avg_distance_km,
            
            AVG(daily_sum_fuel) OVER(ORDER BY t_date RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND INTERVAL '1' DAY PRECEDING) AS weekly_avg_fuel_litres,
            AVG(daily_sum_fuel) OVER(ORDER BY t_date RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND INTERVAL '1' DAY PRECEDING) AS monthly_avg_fuel_litres
        FROM daily_timeline_aggregates
    )
    SELECT 
        p.*,
        t.daily_avg_distance_km,
        t.weekly_avg_distance_km,
        t.monthly_avg_distance_km,
        t.weekly_avg_fuel_litres,
        t.monthly_avg_fuel_litres,
        
        p.distance_km / NULLIF(t.daily_avg_distance_km, 0) AS ratio_trip_to_daily_km,
        p.distance_km / NULLIF(t.weekly_avg_distance_km, 0) AS ratio_trip_to_weekly_km,
        p.distance_km / NULLIF(t.monthly_avg_distance_km, 0) AS ratio_trip_to_monthly_km
    FROM jurisdictional_proportions p
    LEFT JOIN temporal_moving_windows t ON p.t_date = t.t_date;
    """

    print("Compiling Feature Engineering Table via CTAS...")
    response = athena_client.start_query_execution(
        QueryString=ctas_sql,
        QueryExecutionContext={'Database': 'default'},
        ResultConfiguration={'OutputLocation': f's3://{bucket_name}/athena-query-results/'}
    )
    
    wait_for_athena_query(athena_client, response['QueryExecutionId'])
    print(f"Physical feature matrix built successfully at: {target_s3_location}")

def wait_for_athena_query(client, query_id):
    """Helper engine polling routine."""
    while True:
        status = client.get_query_execution(QueryExecutionId=query_id)
        state = status['QueryExecution']['Status']['State']
        if state == 'SUCCEEDED':
            break
        elif state in ['FAILED', 'CANCELLED']:
            raise Exception(f"Athena transaction {query_id} aborted: {status['QueryExecution']['Status'].get('StateChangeReason')}")
        time.sleep(2)

if __name__ == "__main__":
    build_analytical_features_table()