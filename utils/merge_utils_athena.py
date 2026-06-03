"""Python program to merge pandas dataframes of different quantities."""

import boto3
import time


def execute_athena_query(query_string, bucket_name, region='ca-west-1',
                         database='default'):
    """Helper function to execute Athena SQL queries and poll for completion."""
    athena_client = boto3.client('athena', region_name=region)
    
    response = athena_client.start_query_execution(
        QueryString=query_string,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={
            'OutputLocation': f's3://{bucket_name}/athena-query-results/'
        }
    )
    
    query_id = response['QueryExecutionId']
    print(f"Executing Athena Query ID: {query_id}")
    
    while True:
        status = athena_client.get_query_execution(QueryExecutionId=query_id)
        state = status['QueryExecution']['Status']['State']
        
        if state == 'SUCCEEDED':
            print("Athena query execution completed successfully.")
            return query_id
        elif state in ['FAILED', 'CANCELLED']:
            reason = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
            raise Exception(f"Athena query failed: {reason}")
            
        time.sleep(2)


def deploy_athena_schema(bucket_name='prh-ifta-audit-lake', region='ca-west-1'):
    """Deploys the external table schemas and merged audit feature views over your S3 CSV directories."""
    print("\n-- Deploying Athena Relational Database Schema & Views --")
    
    # 1. Invoices Table Schema
    invoices_sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS default.ifta_extracted_invoices (
        invoice_number STRING,
        date STRING,
        time STRING,
        vendor_name STRING,
        city STRING,
        province STRING,
        fuel_type STRING,
        fuel_grade STRING,
        quantity DOUBLE,
        cost_per_litre DOUBLE,
        cost DOUBLE,
        total_tax DOUBLE,
        fed_tax DOUBLE,
        prov_tax DOUBLE,
        payment_form STRING,
        source_file STRING
    )
    ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
    LOCATION 's3://{bucket_name}/ocr-extracted-csv/invoices/'
    TBLPROPERTIES ('skip.header.line.count'='1');
    """
    
    # 2. Distance Logs Table Schema
    distance_logs_sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS default.ifta_extracted_distance_logs (
        trip_date STRING,
        trip_origin STRING,
        trip_destination STRING,
        trip_start_time STRING,
        trip_end_time STRING,
        start_odometer DOUBLE,
        end_odometer DOUBLE,
        distance_km DOUBLE,
        vin_or_truck_number STRING,
        ab_kms DOUBLE,
        bc_kms DOUBLE,
        sk_kms DOUBLE,
        mb_kms DOUBLE,
        on_kms DOUBLE,
        ab_fuel DOUBLE,
        bc_fuel DOUBLE,
        sk_fuel DOUBLE,
        mb_fuel DOUBLE,
        on_fuel DOUBLE,
        source_file STRING
    )
    ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
    LOCATION 's3://{bucket_name}/ocr-extracted-csv/distance_logs/'
    TBLPROPERTIES ('skip.header.line.count'='1');
    """

    # 3. Merged Audit Feature View (+/- 2 Day Fuzzy Match + Proximity Ranking)
    merged_view_sql = """
    CREATE OR REPLACE VIEW default.ifta_audit_merged_view AS
    WITH parsed_invoices AS (
        SELECT *,
            CAST(date AS DATE) AS inv_date
        FROM default.ifta_extracted_invoices
        WHERE date IS NOT NULL AND date != 'nan'
    ),
    parsed_logs AS (
        SELECT 
            trip_date, trip_origin, trip_destination, trip_start_time,
            trip_end_time, start_odometer, end_odometer, vin_or_truck_number,
            ab_kms, bc_kms, sk_kms, mb_kms, on_kms,
            ab_fuel, bc_fuel, sk_fuel, mb_fuel, on_fuel, source_file,
            CAST(trip_date AS DATE) AS log_date,
            COALESCE(distance_km, (end_odometer - start_odometer)) AS distance_km
        FROM default.ifta_extracted_distance_logs
        WHERE trip_date IS NOT NULL AND trip_date != 'nan'
    ),
    fuzzy_matched_data AS (
        SELECT 
            -- Distance Log Columns (The Primary Stream)
            d.trip_date,
            d.trip_origin,
            d.trip_destination,
            d.trip_start_time,
            d.trip_end_time,
            d.start_odometer,
            d.end_odometer,
            d.distance_km,
            d.vin_or_truck_number,
            d.ab_kms, d.bc_kms, d.sk_kms, d.mb_kms, d.on_kms,
            d.ab_fuel, d.bc_fuel, d.sk_fuel, d.mb_fuel, d.on_fuel,
            d.source_file AS log_source_file,

            -- Appended Invoice Columns (Trailing Block)
            i.invoice_number,
            i.date AS invoice_date,
            i.time AS invoice_time,
            i.vendor_name,
            i.city AS purchase_city,
            i.province AS purchase_province,
            i.fuel_type,
            i.fuel_grade,
            i.quantity AS fuel_litres_purchased,
            i.cost_per_litre,
            i.cost AS fuel_cost,
            i.total_tax,
            i.fed_tax,
            i.prov_tax,
            i.payment_form,
            i.source_file AS invoice_source_file,

            -- Matching Metadata
            abs(date_diff('day', i.inv_date, d.log_date)) AS days_difference,

            -- Rank matches per distance log entry
            ROW_NUMBER() OVER(
                PARTITION BY d.vin_or_truck_number, d.trip_date, d.trip_start_time 
                ORDER BY abs(date_diff('day', i.inv_date, d.log_date)) ASC
            ) AS proximity_rank

        FROM parsed_logs d
        LEFT JOIN parsed_invoices i
            ON abs(date_diff('day', i.inv_date, d.log_date)) <= 2
    )
    SELECT * FROM fuzzy_matched_data 
    WHERE proximity_rank = 1;
    """
    # Drop old tables
    print("Dropping old table definitions if they exist...")
    execute_athena_query(
        "DROP TABLE IF EXISTS default.ifta_extracted_invoices;", bucket_name,
        region)
    execute_athena_query(
        "DROP TABLE IF EXISTS default.ifta_extracted_distance_logs;",
        bucket_name, region)
    execute_athena_query(
        "DROP VIEW IF EXISTS default.ifta_audit_merged_view;",
        bucket_name, region)

    # Execute the table builds first
    print("Building raw tables...")
    execute_athena_query(invoices_sql, bucket_name, region)
    execute_athena_query(distance_logs_sql, bucket_name, region)

    # Execute the view build dynamically on top of the tables
    print("Compiling fuzzy-match feature view...")
    execute_athena_query(merged_view_sql, bucket_name, region)
    
    print("All schemas and views successfully deployed.")


# --- Execution Entry Point ---
if __name__ == "__main__":
    PRIMARY_BUCKET = "prh-ifta-audit-lake"
    deploy_athena_schema(bucket_name=PRIMARY_BUCKET)

    print("\n Data is structured and ready for analytical views.")