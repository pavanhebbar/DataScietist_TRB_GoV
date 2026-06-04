"""Python program to extract IFTA relevant data.

Things to complete by tonight:
1. Check for rows where quantities are missing
2. Resolve into federal and proincial taxes

"""

import io
import boto3
from botocore.exceptions import ClientError
import docx2txt
import numpy as np
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
import re
import shutil
from textractor import Textractor
from textractor.data.constants import TextractFeatures


# ── Column alias maps ─────────────────────────────────────────────────────────
# Maps every known variation → standard column name
# Matching is done on lowercased, stripped, whitespace-normalized column names

DISTANCE_COLUMN_ALIASES = {
    # trip_date
    'date': 'trip_date', 'trip date': 'trip_date',
    'trip_date': 'trip_date',

    # trip_origin
    'origin': 'trip_origin', 'trip origin': 'trip_origin',
    'trip_origin': 'trip_origin', 'starting point': 'trip_origin',
    'start location': 'trip_origin', 'from': 'trip_origin',

    # trip_destination
    'destination': 'trip_destination', 'trip destination': 'trip_destination',
    'trip_destination': 'trip_destination', 'to': 'trip_destination',
    'end location': 'trip_destination',

    # trip_start_time
    'start time': 'trip_start_time', 'trip_start_time': 'trip_start_time',
    'trip start time': 'trip_start_time', 'departure time': 'trip_start_time',

    # trip_end_time
    'end time': 'trip_end_time', 'trip_end_time': 'trip_end_time',
    'trip end time': 'trip_end_time', 'arrival time': 'trip_end_time',

    # start_odometer
    'start_odometer': 'start_odometer', 'start odometer': 'start_odometer',
    'start km': 'start_odometer', 'start kms': 'start_odometer',
    'odometer start': 'start_odometer', 'begin odometer': 'start_odometer',

    # end_odometer
    'end_odometer': 'end_odometer', 'end odometer': 'end_odometer',
    'end km': 'end_odometer', 'end kms': 'end_odometer',
    'odometer end': 'end_odometer', 'finish odometer': 'end_odometer',

    # distance_km
    'distance_km': 'distance_km', 'distance km': 'distance_km',
    'distance': 'distance_km', 'total km': 'distance_km',
    'total kms': 'distance_km', 'total_km': 'distance_km',
    'km': 'distance_km', 'kms': 'distance_km',

    # vin_or_truck_number
    'vin_or_truck_number': 'vin_or_truck_number', 'vin': 'vin_or_truck_number',
    'truck number': 'vin_or_truck_number', 'truck no': 'vin_or_truck_number',
    'vehicle id': 'vin_or_truck_number', 'unit': 'vin_or_truck_number',
    'unit number': 'vin_or_truck_number',

    # provincial km columns
    'ab kms': 'ab_kms', 'ab km': 'ab_kms', 'ab_kms': 'ab_kms',
    'alberta km': 'ab_kms', 'alberta kms': 'ab_kms',
    'bc kms': 'bc_kms', 'bc km': 'bc_kms', 'bc_kms': 'bc_kms',
    'british columbia km': 'bc_kms',
    'sk kms': 'sk_kms', 'sk km': 'sk_kms', 'sk_kms': 'sk_kms',
    'saskatchewan km': 'sk_kms',
    'mb kms': 'mb_kms', 'mb km': 'mb_kms', 'mb_kms': 'mb_kms',
    'manitoba km': 'mb_kms',
    'on kms': 'on_kms', 'on km': 'on_kms', 'on_kms': 'on_kms',
    'ontario km': 'on_kms',

    # provincial fuel columns
    'ab fuel': 'ab_fuel', 'ab_fuel': 'ab_fuel', 'alberta fuel': 'ab_fuel',
    'bc fuel': 'bc_fuel', 'bc_fuel': 'bc_fuel', 'british columbia fuel': 'bc_fuel',
    'sk fuel': 'sk_fuel', 'sk_fuel': 'sk_fuel', 'saskatchewan fuel': 'sk_fuel',
    'mb fuel': 'mb_fuel', 'mb_fuel': 'mb_fuel', 'manitoba fuel': 'mb_fuel',
    'on fuel': 'on_fuel', 'on_fuel': 'on_fuel', 'ontario fuel': 'on_fuel',
}

INVOICE_COLUMN_ALIASES = {
    # invoice_number
    'invoice_number': 'invoice_number', 'invoice number': 'invoice_number',
    'invoice no': 'invoice_number', 'invoice #': 'invoice_number',
    'receipt number': 'invoice_number', 'receipt no': 'invoice_number',
    'receipt #': 'invoice_number', 'receipt_number': 'invoice_number',
    'transaction number': 'invoice_number', 'ref number': 'invoice_number',

    # date
    'date': 'date', 'transaction date': 'date', 'invoice date': 'date',
    'purchase date': 'date',

    # time
    'time': 'time', 'transaction time': 'time', 'purchase time': 'time',

    # vendor_name
    'vendor_name': 'vendor_name', 'vendor name': 'vendor_name',
    'vendor': 'vendor_name', 'store': 'vendor_name',
    'store name': 'vendor_name', 'merchant': 'vendor_name',
    'retailer': 'vendor_name', 'company': 'vendor_name',

    # city
    'city': 'city', 'location': 'city', 'town': 'city',
    'purchase location': 'city', 'purchase city': 'city',

    # province
    'province': 'province', 'prov': 'province', 'state': 'province',
    'purchase province': 'province', 'jurisdiction': 'province',

    # fuel_type
    'fuel_type': 'fuel_type', 'fuel type': 'fuel_type',
    'type': 'fuel_type', 'product': 'fuel_type', 'fuel': 'fuel_type',

    # fuel_grade
    'fuel_grade': 'fuel_grade', 'fuel grade': 'fuel_grade',
    'grade': 'fuel_grade', 'product grade': 'fuel_grade',

    # quantity
    'quantity': 'quantity', 'litres': 'quantity', 'liters': 'quantity',
    'volume': 'quantity', 'qty': 'quantity', 'amount (l)': 'quantity',
    'fuel quantity': 'quantity', 'fuel litres': 'quantity',
    'litres purchased': 'quantity',

    # cost_per_litre
    'cost_per_litre': 'cost_per_litre', 'cost per litre': 'cost_per_litre',
    'price per litre': 'cost_per_litre', 'price/l': 'cost_per_litre',
    '$/l': 'cost_per_litre', 'unit price': 'cost_per_litre',
    'price per liter': 'cost_per_litre', 'rate': 'cost_per_litre',

    # cost
    'cost': 'cost', 'total cost': 'cost', 'total': 'cost',
    'amount': 'cost', 'fuel cost': 'cost', 'subtotal': 'cost',
    'net amount': 'cost',

    # total_tax
    'total_tax': 'total_tax', 'total tax': 'total_tax',
    'tax': 'total_tax', 'taxes': 'total_tax', 'tax total': 'total_tax',

    # fed_tax
    'fed_tax': 'fed_tax', 'federal tax': 'fed_tax', 'gst': 'fed_tax',
    'hst': 'fed_tax', 'federal': 'fed_tax',

    # prov_tax
    'prov_tax': 'prov_tax', 'provincial tax': 'prov_tax', 'pst': 'prov_tax',
    'provincial': 'prov_tax',

    # payment_form
    'payment_form': 'payment_form', 'payment form': 'payment_form',
    'form of payment': 'payment_form', 'payment': 'payment_form',
    'payment method': 'payment_form', 'payment type': 'payment_form',
    'tender': 'payment_form',
}


def standardize_df_columns(data_df, file_cols, filequant='distance'):
    """
    Maps raw extracted column names to standard schema column names
    using a fuzzy alias lookup before falling back to NaN for
    genuinely missing columns.

    Matching is case-insensitive and whitespace-normalized so that
    'AB KMs', 'ab kms', 'AB Kms' all map correctly to 'ab_kms'.

    Prints a diagnostic report showing what was matched, renamed,
    and filled with NaN — useful for debugging new source files.
    """
    alias_map = (DISTANCE_COLUMN_ALIASES if filequant == 'distance'
                 else INVOICE_COLUMN_ALIASES)

    # Normalize incoming column names for matching
    # Keep original names for the actual rename operation
    normalized_to_original = {
        col.lower().strip().replace('_', ' '): col
        for col in data_df.columns
    }

    rename_map   = {}   # original_name → standard_name
    matched      = []
    renamed      = []
    filled_nan   = []

    for std_col in file_cols:
        if std_col == 'source_file':
            continue  # always added separately in readfile_to_df

        # Check if standard name already exists exactly
        if std_col in data_df.columns:
            matched.append(std_col)
            continue

        # Normalize and look up in alias map
        found = False
        for norm_col, orig_col in normalized_to_original.items():
            mapped = alias_map.get(norm_col)
            if mapped == std_col and orig_col not in rename_map:
                rename_map[orig_col] = std_col
                renamed.append(f"'{orig_col}' → '{std_col}'")
                found = True
                break

        if not found:
            filled_nan.append(std_col)

    # Apply renames
    data_df = data_df.rename(columns=rename_map)

    # Fill genuinely missing columns with NaN
    for col in filled_nan:
        data_df[col] = np.nan

    # Reindex to standard column order, dropping any extra columns
    data_df = data_df.reindex(columns=file_cols)

    # Diagnostic report
    print(f"\n   📋 Column standardization report ({filequant}):")
    if matched:
        print(f"      ✅ Exact match  ({len(matched)}): {matched}")
    if renamed:
        print(f"      🔄 Renamed      ({len(renamed)}): {renamed}")
    if filled_nan:
        print(f"      ⚠️  Filled NaN   ({len(filled_nan)}): {filled_nan}")

    return data_df


def check_file(filename):
    """Check if file exists and return file type."""
    filename = Path(filename)
    if filename.is_file():
        return filename.suffix
    return False


def check_upload_s3(localfile, bucket_name=None, s3_prefix='raw/'):
    """Check if file is in S3 bucket, else upload"""
    if bucket_name is None:
        bucket_name='ifta-storage-prh'

    file_name = Path(localfile).name
    s3_key = f"{s3_prefix}{file_name}"

    # Create bucket if it does not exist
    s3_client = boto3.client('s3', region_name='ca-west-1')
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as err:
        error_code = err.response['Error']['Code']
        if error_code in ['404', 'NoSuchBucket']:
            print(f"Bucket '{bucket_name}' not found. Creating it now...")
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': 'ca-west-1'}
            )
        else:
            raise err

    # Check if file exists in bucket
    try:
        # Check metadata without downloading the full file
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"[S3 Cache Hit] '{file_name}' already exists in" +
              f"s3://{bucket_name}/{s3_key}. Skipping upload.")
    except ClientError as e:
        # If a 404 error is returned, the file does not exist yet
        if e.response['Error']['Code'] == '404':
            print(f"[S3 Cache Miss] '{file_name}' not found in cloud storage.")
            print("Uploading now...")
            try:
                s3_client.upload_file(localfile, bucket_name, s3_key)
                print(f"Successfully uploaded '{file_name}' to s3://{bucket_name}/{s3_key}")
            except Exception as upload_error:
                print(f"Error uploading file to S3: {upload_error}")
                raise upload_error
        else:
            # Re-raise any unexpected AWS errors (e.g., credential or permission issues)
            raise e

    return s3_key


def run_textract_smallpdf(pdffile):
    """Run textract for single page pdfs."""
    # Check if file exists:
    if check_file(pdffile) != '.pdf':
        raise ValueError("File is not found or is not a pdf file")

    print("Using the synchronous local stream to extract data.")
    extractor = Textractor(region_name='ca-central-1')
    document = extractor.analyze_document(
        file_source=pdffile,
        features=[TextractFeatures.TABLES]
    )
    if not document.tables:
        print("No structural tables detected on this page.")
        return pd.DataFrame()

    print(f"Extracted {len(document.tables)} table(s) from 1 page pdf.")
    return document.tables


def run_textract_largepdf(pdffile, bucket_name='textract-storage-prh'):
    """Run textract for multipage pdfs and combine the tables.

    Makes use of the S3 client services and NextToken pagination.
    """
    # Check if file exists
    if check_file(pdffile) != '.pdf':
        raise ValueError("File is not found or is not a pdf file")

    # Create bucket if it does not exist
    s3_client = boto3.client('s3', region_name='ca-central-1')
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as err:
        error_code = err.response['Error']['Code']
        if error_code in ['404', 'NoSuchBucket']:
            print(f"Bucket '{bucket_name}' not found. Creating it now...")
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': 'ca-central-1'}
            )
        else:
            raise err

    # Inititalize and run Textractor
    print("Using asynchronous pipelines for multi-page pdfs.")
    extractor = Textractor(region_name='ca-central-1')
    document = extractor.start_document_analysis(
        file_source=pdffile, s3_upload_path=f"s3://{bucket_name}/",
        features=[TextractFeatures.TABLES], save_image=False
    )
    if not document.tables:
        print("No structural tables detected on this page.")
        return pd.DataFrame()
    print("Extracted all tables from multiple pages")
    return document.tables


def get_df_texttables_special(textract_tables):
    """Special case when only the first page has the column names."""
    all_table_dfs = []
    all_conf_dfs = []
    column_headers = None

    for i, table in enumerate(textract_tables):
        if i == 0:
            df_temp = table.to_pandas(use_columns=True)
            column_headers = df_temp.columns.tolist()
            conf_df = get_conf_df_textracttable(
                table, column_headers, skipheader=True)
        else:
            df_temp = table.to_pandas(use_columns=False)
            # Check for same number of columns
            if df_temp.shape[1] == len(column_headers):
                df_temp.columns = column_headers
            else:
                raise ValueError(f"Size mismatch at table {i}")
            conf_df = get_conf_df_textracttable(
                table, column_headers, skipheader=False)

        all_table_dfs.append(df_temp)
        all_conf_dfs.append(conf_df)

    df_combined = pd.concat(all_table_dfs, ignore_index=True)
    conf_df_comb = pd.concat(all_conf_dfs, ignore_index=True)
    return df_combined, conf_df_comb


def get_df_from_textract_tables(textract_tables, column_names=None,
                                column_names_everytable=True):
    """Get pandas datafram from tables extracted by Textractor.
    
    Uses rigid column names. If there is a column name mismatch, then
    a warning is given but it will combine the columns as long as the length
    is the same. If lengths are not the same, then error is thrown
    """
    if not textract_tables:
        return pd.DataFrame(), pd.DataFrame()

    # If there is only one table. Just return the dataframe of the table
    if len(textract_tables) == 1:
        table_df = textract_tables[0].to_pandas(use_columns=True)
        column_names = table_df.columns.tolist()
        conf_df = get_conf_df_textracttable(textract_tables[0], column_names)
        return table_df, conf_df

    # If column names are not listed in every table, use the special case
    if not column_names_everytable:
        return get_df_texttables_special(textract_tables)
    if column_names is None:
        # Use column names from the first table
        column_names = [
            cell.text.strip() for cell in textract_tables[0].table_cells
            if cell.row_index == 1]

    all_table_dfs = []
    all_confs_dfs = []
    for i, table in enumerate(textract_tables):
        df_temp = table.to_pandas(use_columns=True)
        columns_temp = df_temp.columns.tolist()
        # Check size of tables
        if len(columns_temp) != len(column_names):
            raise ValueError(f"Size mismatch at table {i}")
        for (col1, col2) in zip(column_names, columns_temp):
            if col1 != col2.strip():
                print(f"Warning: Column name {col2} does not match with {col1}")
                print(f"Column name will be replaced with {col1}")

        df_temp.columns = column_names
        conf_df_temp = get_conf_df_textracttable(table, column_names)
        all_table_dfs.append(df_temp)
        all_confs_dfs.append(conf_df_temp)

    df_combined = pd.concat(all_table_dfs, ignore_index=True)
    conf_df_combined = pd.concat(all_confs_dfs, ignore_index=True)
    return df_combined, conf_df_combined


def get_conf_df_textracttable(textract_table, column_names, skipheader=True):
    "Get confidence scores of an extracted table from textract."
    sorted_cells = sorted(textract_table.table_cells,
                          key=lambda c: (c.row_index, c.col_index))
    cell_confs = [cell.confidence for cell in sorted_cells]
    table_conf = np.array(cell_confs).reshape(
        textract_table.row_count, textract_table.column_count)
    if skipheader:
        conf_df = pd.DataFrame(table_conf[1:, :], columns=column_names)
    else:
        conf_df = pd.DataFrame(table_conf, columns=column_names)
    return conf_df


def extract_images_docx(docxfile, target_dir):
    """Extract images from the docx files."""
    print(f"Extracting images to temporary folder {target_dir}")
    docx2txt.process(docxfile, str(target_dir))
    return sorted(Path(target_dir).glob("*"))


def process_images_textractor(image_paths):
    """Extract data from images embedded in docx file"""
    extractor = Textractor(region_name='ca-central-1')
    raw_exp_data = []

    for i, path in enumerate(image_paths, 1):
        print(f"...Processing {i}th image: name:{path.name}...")
        expense_doc = extractor.analyze_expense(file_source=str(path))

        if expense_doc.expense_documents:
            exp_data_full = expense_doc.expense_documents[0]
            summ_dict = []  # Safe initialization for both blocks!

            if exp_data_full.summary_fields_list:
                for field in exp_data_full.summary_fields_list:
                    f_type = field.type.text if field.type else ""
                    f_conf = field.type.confidence if field.type else 0.0
                    f_label = field.key.text if field.key else ""
                    f_value = field.value.text if field.value else ""
                    summ_dict.append({
                        'type' : str(f_type).strip().upper(),
                        'conf': float(f_conf),
                        'label': str(f_label).strip().lower(), 
                        'value': str(f_value).strip().upper(),
                    })

            # Add fields from line item groups
            if exp_data_full.line_items_groups:
                for group in exp_data_full.line_items_groups:
                    for row in group.rows:
                        for cell in row.expenses:
                            f_type = (cell.type.text
                                      if cell.type else "LINE_ITEM")
                            f_conf = cell.type.confidence if cell.type else 0.0
                            f_label = cell.type.text if cell.type else ""
                            f_value = cell.value.text if cell.value else ""

                            if f_value:
                                summ_dict.append({
                                    'type': str(f_type).strip().upper(),
                                    'conf': float(f_conf),
                                    'label': str(f_label).strip().lower(),
                                    'value': str(f_value).strip().upper(),
                                })
        # Get raw text lines in case they were missed in summary
        if expense_doc.lines:
            raw_text_lines = [line.text.strip()
                              for line in expense_doc.lines if line.text]
        else:
            raw_text_lines = []

        # Append summ_dict and raw text lines
        raw_exp_data.append({'fields': summ_dict, 'raw_lines': raw_text_lines})

    return raw_exp_data

def clean_numeric(val_str):
    """Strip currencies, text, and spaces to isolate clean numeric strings."""
    if not val_str:
        return None
    cleaned = re.sub(r'[^\d\.\-]', '', val_str)
    return cleaned if cleaned else None


def update_field_value(f_type, f_label, fval, fconf, match_fields,
                       currstd_fval, curr_stdfconf, curr_ftype_rank=100,
                       clean=False):
    """Check if the input field should update std_field based on field rank."""
    std_fval = currstd_fval
    std_fconf = curr_stdfconf

    # Check rank
    ftype_pos = match_fields.index(f_type) if f_type in match_fields else 100
    flabel_pos = match_fields.index(f_label) if f_label in match_fields else 100
    best_pos = min(ftype_pos, flabel_pos)

    # Update value if better rank
    if best_pos < curr_ftype_rank and fval:
        clean_fval = clean_numeric(fval) if clean else str(fval).upper().strip()
        if clean_fval:
            std_fval = clean_fval
            std_fconf = fconf
            curr_ftype_rank = best_pos

    return std_fval, std_fconf, curr_ftype_rank


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
    # If Textract misaligns bounding boxes and assigns the quantity to the unit price slot
    if std_record['quantity'] and std_record['cost_per_litre']:
        if float(std_record['quantity']) == float(std_record['cost_per_litre']):
            # Scan the raw text stream for the true standalone pricing pattern (e.g., "$1.179")
            # looking for a number following a standard pattern that isn't the volume number
            true_price_match = re.search(
                r'(?:PRICE/LITRE|PRICE/L)[:\-]?\s*\$?(\d+\.\d{2,3})', raw_blob)
            if true_price_match:
                std_record['cost_per_litre'] = true_price_match.group(1)

    # 4. PREPAID FOOTER SCANNER (For Burnaby and similar prepay receipts)
    # If we have a cost (like $30.00) but are missing either volume metrics
    if std_record['cost'] and (not std_record['quantity'] or not std_record['cost_per_litre']):
        # If the receipt contains a prepaid indicator line
        if 'PREPAY' in raw_blob or 'PRE-PAY' in raw_blob:
            # Look for tax inclusions or text fragments that pinpoint real calculated metrics
            # Shell receipts hide the GST tax calculation total down at the bottom
            gst_inc_match = re.search(
                r'FUEL INCLUDES\s+GST\s+\d+\.\d+%\s*\$?(\d+\.\d{2})', raw_blob)
            if gst_inc_match and not std_record['fed_tax']:
                std_record['fed_tax'] = gst_inc_match.group(1)

    return std_record


def extract_granular_taxes(raw_blob):
    """Extract provincial and federal taxes"""
    fed_tax = 0.00
    prov_tax = 0.00

    # Split text into uppercase string components
    lines = [line.upper() for line in raw_blob.split('\n')]

    for line in lines:
        # Step A: Process Federal (GST/HST)
        if 'GST' in line or 'F-HST' in line:
            # 1. Look for currency format directly associated with the keyword token
            # Avoids corporate business numbers completely by requiring exactly two decimal places
            match = re.search(r'(?:GST|F-HST)[^0-9]*?\$?\s*(\d{1,3}\.\d{2})\b', line)
            if match:
                fed_tax = float(match.group(1))
            # 2. Fallback: Catch trailing inclusive amounts only if it matches standard decimal format
            elif 'INCL' in line:
                match_incl = re.search(r'\$\s*(\d{1,3}\.\d{2})', line)
                if match_incl:
                    fed_tax = float(match_incl.group(1))
                    print(fed_tax)

        # Step B: Process Provincial (PST/QST/BC TAX)
        if any(token in line for token in ['PST', 'QST', 'BC TAX', 'PROV',
                                           'P-HST']):
            match = re.search(
                r'(?:PST|QST|BC TAX|PROV|P-HST)[^0-9]*?\$?\s*(\d{1,3}\.\d{2})\b', line)
            if match:
                prov_tax = float(match.group(1))
            elif 'INCL' in line:
                # Scans specific trailing provincial configurations
                match_incl = re.search(
                    r'P\-HST\s*INCL\s*\$\s*(\d{1,3}\.\d{2})', line)
                if match_incl:
                    prov_tax = float(match_incl.group(1))

    return fed_tax, prov_tax


def apply_global_fallbacks(std_record, raw_lines):
    """Standardize codes, fill structural gaps, and compute math fallbacks."""
    raw_blob = " ".join(raw_lines).upper()

    # Functionality A: Run regex layout text scrape if fields came up empty
    std_record = scrape_missing_metrics_from_text(std_record, raw_blob)

    # Get taxes
    fed_tax, prov_tax = extract_granular_taxes(raw_blob)
    if std_record['fed_tax'] is None:
        std_record['fed_tax'] = fed_tax
    if std_record['prov_tax'] is None:
        std_record['prov_tax'] = prov_tax
    try:
        f_tax = float(std_record.get('fed_tax', 0.0) or 0.0)
        p_tax = float(std_record.get('prov_tax', 0.0) or 0.0)
        t_tax = float(std_record.get('total_tax', 0.0) or 0.0)
    except ValueError:
        f_tax, p_tax, t_tax = 0.0, 0.0, 0.0
    if t_tax < (f_tax + p_tax) or not std_record['total_tax'] or std_record['total_tax'] == 'None':
        std_record['total_tax'] = f"{round(f_tax + p_tax, 2)}"

    # Functionality B: Map variable regional province text blocks to 2-letter IFTA codes
    prov_dump = std_record['province']
    if prov_dump == 'UNKNOWN' or len(prov_dump) > 2:
        if any(p in raw_blob for p in [' AB ', 'ALBERTA', 'EDMONTON',
                                       'RED DEER', 'CALGARY']):
            std_record['province'] = 'AB'
        elif any(p in raw_blob for p in [' BC ', 'BRITISH COLUMBIA', 'KELOUNA',
                                         'VANCOUVER', 'BURNABY']):
            std_record['province'] = 'BC'
        elif any(p in raw_blob for p in [' SK ', 'SASKATCHEWAN', 'REGINA',
                                         'SASKATOON']):
            std_record['province'] = 'SK'
        elif any(p in raw_blob for p in [' MB ', 'MANITOBA', 'WINNIPEG']):
            std_record['province'] = 'MB'
        elif any(p in raw_blob for p in [' ON ', 'ONTARIO', 'BARRIE',
                                         'CALEDON', 'WILLOWDALE']):
            std_record['province'] = 'ON'

    # Fill city
    if not std_record['city'] or std_record['city'] == 'UNKNOWN':
        for city_check in ['RED DEER', 'EDMONTON', 'HANNA', 'CALEDON',
                           'WINNIPEG', 'WILLOWDALE', 'BARRIE', 'LANGLEY',
                           'BURNABY', 'CONSORT']:
            if city_check in raw_blob:
                std_record['city'] = city_check
                break

    # Functionality C: Standardize missing Fuel Type strings by sweeping raw layout lines
    if std_record['fuel_type'] == 'UNKNOWN':
        if 'DIESEL' in raw_blob:
            std_record['fuel_type'] = 'DIESEL'
            std_record['fuel_grade'] = 'UNKNOWN'
        else:
            for gas in ['SUPREME', 'BRONZE', 'PLUS', 'REGULAR', 'UNLEADED',
                        'GASOLINE', 'EREG']:
                if gas in raw_blob:
                    std_record['fuel_type'] = 'GASOLINE'
                    std_record['fuel_grade'] = gas
                    if gas == 'EREG':
                        std_record['fuel_grade'] = 'REGULAR'

    # Mathematically deduce volume if structural fields and regex both missed it
    if (not std_record['cost_per_litre'] or std_record['cost_per_litre'] ==
        'None') and std_record['cost'] and std_record['quantity']:
        try:
            tot_cost = float(std_record['cost'])
            volume = float(std_record['quantity'])

            # Make sure we don't divide by zero or process anomalous quantities like 1
            if volume > 0.0:  
                calculated_unit_price = round(tot_cost / volume, 3)
                std_record['cost_per_litre'] = f"{calculated_unit_price}"
        except ValueError:
            pass

    # Robust Invoice/Transaction Number Fallback
    if (not std_record['invoice_number'] or std_record['invoice_number'] ==
            'None'):
        # Pattern covers: "INV No. 123", "TRANS #: 123", "TICKET # 123", or "REF #: 123"
        inv_pattern = r'\b(?:INV(?:OICE)?\.?\s*(?:No\.?)?|TRANS(?:\s*#|\s*ACTION)?\.?\s*(?:No\.?)?|TICKET\s*#?|REF(?:ERENCE)?\s*#?)\s*[:\-]?\s*([A-Z0-9\-]{4,15})\b'
        inv_match = re.search(inv_pattern, raw_blob, re.IGNORECASE)
        if inv_match:
            std_record['invoice_number'] = inv_match.group(1).strip()

    if not std_record['invoice_number'] or std_record['invoice_number'] == 'None':
        # Find the PC sequence (e.g., PC0970310:3537601)
        pc_match = re.search(r'PC\d+:\s*(\d+)', raw_blob)
        if pc_match:
            std_record['invoice_number'] = pc_match.group(1)

    # Payment Form Fallback Loop
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

    # Robust Time Fallback Loop
    if not std_record['time'] or std_record['time'] == 'None' or std_record['time'] is None:
        # Matches: "13:13:37", "19:05", or "16: 27" (handles optional spaces and optional seconds)
        time_pattern = r'\b([0-9]?\d)\s*:\s*([0-9]\d)(?:\s*:\s*([0-5]\d))?\b'
        time_match = re.search(time_pattern, raw_blob)

        if time_match:
            hours = time_match.group(1).zfill(2)
            minutes = time_match.group(2)
            seconds = time_match.group(3)

            if seconds:
                std_record['time'] = f"{hours}:{minutes}:{seconds}"
            else:
                std_record['time'] = f"{hours}:{minutes}:00"

    return std_record


def extract_clean_record(raw_exp_data):
    """Selects and standardizes core variables using a configuration matrix and multi-layered fallbacks.    """
    clean_records = []
    clean_confscores = []

    # Centralized configuration mapping matrix
    field_map = {
        'invoice_number': ['INVOICE_RECEIPT_ID', 'receipt no', 'invoice no',
                           'inv no.'],
        'date': ['INVOICE_RECEIPT_DATE', 'date:'],
        'time': ['TRANSACTION_TIME', 'time:'],
        'vendor_name': ['VENDOR_NAME'],
        'city': ['CITY'],
        'province': ['STATE'],
        'fuel_type': ['FUEL'],
        'fuel_grade': ['product', 'grade', 'fuel'],
        'quantity': ['QUANTITY', 'litres', 'l', 'litres :', 'quantity',
                     'volume'],
        'cost_per_litre': ['UNIT_PRICE', 'price/litre', 'price/l', 'unit_price',
                           'price/g'],
        'cost': ['TOTAL', 'AMOUNT_PAID', 'PRICE', 'fuel sales', 'AMOUNT_DUE',
                 'total fuel'],
        'total_tax': ['TAX'],
        # 'fed_tax':['gst included:', 'fhst included in fuel', 'gst included',
        #           '* f-hst incl$', 'gst', 'f-hst'],
        # 'prov_tax':['* p-hst incl$', 'pst', 'p-hst'],
        'fed_tax': ['fhst included in fuel', 'fuel includes gst 5.0%'],
        'prov_tax':['phst included in fuel', 'fuel includes pst 7.0%'],
        'payment_form': ['PAYMENT_TYPE'],
    }

    clean_numeric_bool = [
        False, False, False, False, False, False, 
        False, False, True, True, True, True, True, True, False]

    for record in raw_exp_data:
        # Uniform target layout template matching the required elements exactly
        std_record = {
            'invoice_number': None, 'date': None, 'time': None,
            'vendor_name': None, 'city': None, 'province': 'UNKNOWN',
            'fuel_type': 'UNKNOWN', 'fuel_grade': 'UNKNOWN', 'quantity': None,
            'cost_per_litre': None, 'cost': None, 'total_tax': None,
            'fed_tax': None, 'prov_tax': None,
            'payment_form': None
        }

        confidence_scores = {
            'invoice_number': 0.0, 'date': 0.0, 'time': 0.0,
            'vendor_name': 0.0, 'city': 0.0, 'province': 0.0,
            'fuel_type': 0.0, 'fuel_grade': 0.0, 'quantity': 0.0,
            'cost_per_litre': 0.0, 'cost': 0.0, 'total_tax': 0.0,
            'fed_tax': 0.0, 'prov_tax': 0.0, 'payment_form': 0.0
        }

        curr_std_ranks = [100] * len(field_map)

        summ_list = record.get('fields', [])
        raw_lines = record.get('raw_lines', [])

        # Run through structural summary layout fields
        for field in summ_list:
            f_label = field['label'].strip().lower() if field['label'] else ""
            f_type = field['type']
            f_value = field['value']
            f_conf = field['conf']

            for i, std_key in enumerate(field_map.keys()):
                # Update value if needed
                updated_val, updated_conf, updated_rank = update_field_value(
                    f_type=f_type, 
                    f_label=f_label, 
                    fval=f_value, 
                    fconf=f_conf,
                    match_fields=field_map[std_key],
                    currstd_fval=std_record[std_key],
                    curr_stdfconf= confidence_scores[std_key],
                    curr_ftype_rank=curr_std_ranks[i],
                    clean=clean_numeric_bool[i]
                )

                std_record[std_key] = updated_val
                confidence_scores[std_key] = updated_conf
                curr_std_ranks[i] = updated_rank

        # Check for missing values in the raw lines
        std_record = apply_global_fallbacks(std_record, raw_lines)
        clean_records.append(std_record)
        clean_confscores.append(confidence_scores)

    return (pd.DataFrame(clean_records) if clean_records else pd.DataFrame(),
            (pd.DataFrame(clean_confscores)
            if clean_confscores else pd.DataFrame()))


def clean_and_standardize_dataframe(df):
    """Standardize dataframe values."""
    clean_df = df.copy()
    
    # 1. FORCE UNIFORM STRING CLEANING FOR DATE AND TIME
    clean_df['date'] = clean_df['date'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    clean_df['time'] = clean_df['time'].astype(str).str.replace(r'(\d+):\s+(\d+)', r'\1:\2', regex=True).str.strip()
    clean_df['time'] = clean_df['time'].str.replace(r'^(\d{2}:\d{2})$', r'\1:00', regex=True)
    
    # Fix specific typos
    clean_df['date'] = clean_df['date'].str.replace('2016- 33 16', '2016/03/16', regex=False)
    clean_df['date'] = clean_df['date'].str.replace('2025/11/38', '2025/11/30', regex=False)
    clean_df['date'] = clean_df['date'].str.replace('-', '/', regex=False)
    clean_df['time'] = clean_df['time'].str.replace('99:56:00', '09:56:00', regex=False)

    # 2. PARSE UNIFIED DATETIME TIMESTAMP
    clean_df['timestamp'] = pd.to_datetime(
        clean_df['date'] + ' ' + clean_df['time'], 
        format='%Y/%m/%d %H:%M:%S',
        errors='coerce'
    )
    clean_df['date'] = clean_df['timestamp'].dt.strftime('%Y-%m-%d')
    clean_df['time'] = clean_df['timestamp'].dt.strftime('%H:%M:%S')

    # 3. CLEAN TEXT FIELDS WITH CUSTOM HYPHEN/SPACE RULES
    text_columns = ['vendor_name', 'city', 'province', 'invoice_number', 'fuel_grade', 'payment_form']
    for col in text_columns:
        if col in clean_df.columns:
            clean_df[col] = (clean_df[col].astype(str)
                             .str.replace(r'[\r\n]+', ' ', regex=True) # Wipes out raw newlines
                             .str.strip()                             # Drops leading/trailing spaces
                             .str.strip('-')                          # Drops leading/trailing hyphens
                             .str.replace(r'-', ' ', regex=False)     # Converts middle hyphens to spaces
                             .str.replace(r'\s+', ' ', regex=True)     # Shrinks any resulting double spaces
                             .str.strip()                             # Final safety strip
                             .replace({'None': np.nan, 'NONE': np.nan, 'nan': np.nan, '': np.nan}))
            
    # 4. FORCE NUMERIC FIELDS TO FLOATS
    numeric_columns = ['quantity', 'cost_per_litre', 'cost', 'total_tax', 'fed_tax', 'prov_tax']
    for col in numeric_columns:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            
    # 5. ENFORCE SYSTEM CONSISTENCY RULES
    clean_df.loc[clean_df['quantity'] == 1.0, 'quantity'] = np.nan
    
    # Sort chronologically by your new timestamp
    clean_df = clean_df.sort_values(by='timestamp').reset_index(drop=True)
    
    return clean_df


def run_textractor_docx_invoices(docxfile, temp_imgdir='./temp', debug=False):
    """"""
    temp_imgdir = Path(temp_imgdir)
    temp_imgdir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract images into a temporary directory
        image_paths = extract_images_docx(docxfile, temp_imgdir)
        if not image_paths:
            return pd.DataFrame()

        #Run Textractor
        raw_data = process_images_textractor(image_paths)
        invoice_dataframe, inv_conf_scores = extract_clean_record(raw_data)
        return (clean_and_standardize_dataframe(invoice_dataframe),
                inv_conf_scores)

    finally:
        # So that the temp file is deleted even when there is err
        if not debug:
            if temp_imgdir.exists():
                print(f"Deleting temporary directory: {temp_imgdir}")
                shutil.rmtree(temp_imgdir)
            else:
                print('Temp. folder not deleted given debug status')


def clean_dataframe(df_uncleaned, conf_df_uncleaned=None):
    """Remove all empty  and duplicate rows"""
    if df_uncleaned.empty:
        return df_uncleaned, conf_df_uncleaned
    # Remove empty rows
    df_uncleaned = df_uncleaned.replace(r'^\s*$', pd.NA, regex=True)
    valid_row_mask = df_uncleaned.notna().any(axis=1)
    df_cleaned = df_uncleaned[valid_row_mask]

    # Remove duplicates
    is_duplicate = df_cleaned.astype(str).duplicated()
    df_cleaned = df_cleaned[~is_duplicate].reset_index(drop=True)
    if conf_df_uncleaned is not None:
        conf_df_cleaned = conf_df_uncleaned[valid_row_mask]
        conf_df_cleaned = conf_df_cleaned[~is_duplicate].reset_index(drop=True)
    return df_cleaned, conf_df_cleaned


def export_dataframe_to_s3_csv(df, target_prefix, base_filename, bucket_name):
    """
    Exports using pipe delimiter to avoid Athena SerDe issues with
    quoted strings containing commas (e.g. 'Nisku, AB').
    Also explicitly drops pandas index to prevent column count mismatch.
    """
    s3_client = boto3.client('s3')
    csv_buffer = io.StringIO()

    df.to_csv(
        csv_buffer,
        index=False,          # never export pandas index
        header=True,
        sep='|',        # pipe delimiter — no conflict with city/province commas
        lineterminator='\n'
    )

    target_key = (
        f"{target_prefix}{Path(base_filename).stem.replace(' ', '_')}_extracted.csv"
    )

    s3_client.put_object(
        Bucket=bucket_name,
        Key=target_key,
        Body=csv_buffer.getvalue()
    )
    print(f"Staged extraction table at: s3://{bucket_name}/{target_key}")


def readfile_to_df(localfile, filequant='distance', bucket_name=None):
    """Read one file and return pandas dataframe."""
    # Initialize targeted empty layouts matching your required structural elements perfectly
    invoice_columns = [
        'invoice_number', 'date', 'time', 'vendor_name', 'city', 'province',
        'fuel_type', 'fuel_grade', 'quantity', 'cost_per_litre', 'cost', 
        'total_tax', 'fed_tax', 'prov_tax', 'payment_form', 'source_file'
    ]
    
    distance_columns = [
        'trip_date', 'trip_origin', 'trip_destination', 'trip_start_time', 'trip_end_time',
        'start_odometer', 'end_odometer', 'distance_km', 'vin_or_truck_number',
        'ab_kms', 'bc_kms', 'sk_kms', 'mb_kms', 'on_kms',
        'ab_fuel', 'bc_fuel', 'sk_fuel', 'mb_fuel', 'on_fuel', 'source_file'
    ]
    filetype = check_file(localfile)
    if not filetype:
        raise ValueError("File extension not found.")

    if filequant == 'distance':
        std_cols = distance_columns
    elif filequant == 'invoice':
        std_cols = invoice_columns
    else:
        raise ValueError("Filequant can only be 'distance' of 'invoice'")

    if filetype in ['.docx', '.doc']:
        datatable_df, conf_scores_df = run_textractor_docx_invoices(localfile)
    elif filetype == '.pdf':
        pdf_pagecount = len(PdfReader(localfile).pages)
        if pdf_pagecount == 1:
            pdf_tables = run_textract_smallpdf(localfile)
        else:
            if bucket_name is None:
                bucket_name = 'pavan-ifta-audit-lake'
            pdf_tables = run_textract_largepdf(
                localfile, bucket_name=f"{bucket_name}-textract-temp")
        datatable_df, conf_scores_df = get_df_from_textract_tables(
            pdf_tables, column_names_everytable=True)
    elif filetype in ['.xlsx', '.xls']:
        datatable_df = pd.read_excel(localfile)
        conf_scores_df = pd.DataFrame(
            np.ones_like(datatable_df.values, dtype=float)*100.,
            columns=datatable_df.columns)
    elif filetype == '.csv':
        datatable_df = pd.read_csv(localfile)
        conf_scores_df = pd.DataFrame(
            np.ones_like(datatable_df.values, dtype=float)*100.,
            columns=datatable_df.columns)
    else:
        print("Files can only be PDF, EXCEL, CSV or DOC/DOCX")
        raise ValueError(f"No support for file type: {filetype}")

    datatable_df['source_file'] = Path(localfile).name
    conf_scores_df['source_file'] = 100.0

    # Cleaning
    df_cleaned, conf_cleaned = clean_dataframe(datatable_df, conf_scores_df)
    df_standardized = standardize_df_columns(df_cleaned, std_cols, filequant)
    conf_standardized = standardize_df_columns(conf_cleaned, std_cols,
                                               filequant)

    return (df_standardized, conf_standardized)
    


def run_cloud_extraction_pipeline(
        file_list, file_quant_list=None, bucket_name='prh-ifta-audit-lake',
        bucket_csvfolder="ocr-extracted-csv",
        bucket_conffolder="ocr-extraction-confs"):
    """
    Automated main execution pipeline block. Dispatches files to your 
    existing parsing engine and saves standardized csv rows directly to S3.
    """
    if file_quant_list is None:
        file_quant_list = ['distance', 'distance', 'invoice']

    dftables_list = []
    conf_scores_list = []
    count = 0
    for local_file, doctype in zip(file_list, file_quant_list):
        print(f"\n-- Initiating cloud extraction sequence for: {local_file} --")
        # Upload raw files
        check_upload_s3(local_file, bucket_name=bucket_name, s3_prefix='raw/')

        # Extract dataframe and confidence scores from local files
        df_table, conf_scores = readfile_to_df(
            local_file, doctype, bucket_name=bucket_name)
        df_table.to_csv(f"Temp_{doctype}_{count}.csv")
        if doctype == 'invoice':
            dynamic_csv_folder = f"{bucket_csvfolder}/invoices/"
            dynamic_conf_folder = f"{bucket_conffolder}/invoices/"
        else:
            dynamic_csv_folder = f"{bucket_csvfolder}/distance_logs/"
            dynamic_conf_folder = f"{bucket_conffolder}/distance_logs/"
        export_dataframe_to_s3_csv(
            df_table, dynamic_csv_folder, local_file, bucket_name)
        export_dataframe_to_s3_csv(
            conf_scores, dynamic_conf_folder, local_file, bucket_name)
        dftables_list.append(df_table)
        conf_scores_list.append(conf_scores)
        count += 1

    return dftables_list, conf_scores_list


# --- Execution Entry Point ---
if __name__ == "__main__":
    # Define your specific operational files
    target_files = [
        "../Data/Distance log 1.xlsx",
        "../Data/Distance log 2.pdf",
        "../Data/Fuel Invoices.docx"
    ]

    file_types = ['distance', 'distance', 'invoice']
    PRIMARY_BUCKET = "prh-ifta-audit-lake"

    run_cloud_extraction_pipeline(
        file_list=target_files, 
        file_quant_list=file_types, 
        bucket_name=PRIMARY_BUCKET
    )
