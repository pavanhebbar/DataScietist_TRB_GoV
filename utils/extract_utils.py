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


def check_file(filename):
    """Check if file exists and return file type."""
    filename = Path(filename)
    if filename.is_file():
        return filename.suffix
    return False

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
    column_headers = None

    for i, table in enumerate(textract_tables):
        if i == 0:
            df_temp = table.to_pandas(use_columns=True)
            column_headers = df_temp.columns.tolist()
        else:
            df_temp = table.to_pandas(use_columns=False)
            # Check for same number of columns
            if df_temp.shape[1] == len(column_headers):
                df_temp.columns = column_headers
            else:
                raise ValueError(f"Size mismatch at table {i}")

        all_table_dfs.append(df_temp)

    df_combined = pd.concat(all_table_dfs, ignore_index=True)
    return df_combined


def get_df_from_textract_tables(textract_tables, column_names=None,
                                column_names_everytable=True):
    """Get pandas datafram from tables extracted by Textractor.
    
    Uses rigid column names. If there is a column name mismatch, then
    a warning is given but it will combine the columns as long as the length
    is the same. If lengths are not the same, then error is thrown
    """
    if not textract_tables:
        return pd.DataFrame()
    
    # If there is only one table. Just return the dataframe of the table
    if len(textract_tables) == 1:
        return textract_tables[0].to_pandas(use_columns=True)

    # If column names are not listed in every table, use the special case
    if not column_names_everytable:
        return get_df_texttables_special(textract_tables)
    
    if column_names is None:
        # Use column names from the first table
        column_names = [
            cell.text.strip() for cell in textract_tables[0].table_cells
            if cell.row_index == 1]

    all_table_dfs = []
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
        all_table_dfs.append(df_temp)

    df_combined = pd.concat(all_table_dfs, ignore_index=True)
    return df_combined


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
            true_price_match = re.search(r'(?:PRICE/LITRE|PRICE/L)[:\-]?\s*\$?(\d+\.\d{2,3})', raw_blob)
            if true_price_match:
                std_record['cost_per_litre'] = true_price_match.group(1)

    # 4. PREPAID FOOTER SCANNER (For Burnaby and similar prepay receipts)
    # If we have a cost (like $30.00) but are missing either volume metrics
    if std_record['cost'] and (not std_record['quantity'] or not std_record['cost_per_litre']):
        # If the receipt contains a prepaid indicator line
        if 'PREPAY' in raw_blob or 'PRE-PAY' in raw_blob:
            # Look for tax inclusions or text fragments that pinpoint real calculated metrics
            # Shell receipts hide the GST tax calculation total down at the bottom
            gst_inc_match = re.search(r'FUEL INCLUDES\s+GST\s+\d+\.\d+%\s*\$?(\d+\.\d{2})', raw_blob)
            if gst_inc_match and not std_record['fed_tax']:
                std_record['fed_tax'] = gst_inc_match.group(1)

    return std_record


def extract_granular_taxes(raw_blob):
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
            # Protect against picking up a percentage number like 7.0% by looking for standard dollar values
            match = re.search(r'(?:PST|QST|BC TAX|PROV|P-HST)[^0-9]*?\$?\s*(\d{1,3}\.\d{2})\b', line)
            if match:
                prov_tax = float(match.group(1))
            elif 'INCL' in line:
                # Scans specific trailing provincial configurations
                match_incl = re.search(r'P\-HST\s*INCL\s*\$\s*(\d{1,3}\.\d{2})', line)
                if match_incl:
                    prov_tax = float(match_incl.group(1))

    return fed_tax, prov_tax


def apply_global_fallbacks(std_record, raw_lines):
    """Standardizes codes, fills structural gaps using raw text streams, and computes math fallbacks."""
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

    # Functionality D: Mathematically deduce volume if structural fields and regex both missed it
    if (not std_record['cost_per_litre'] or std_record['cost_per_litre'] == 'None') and std_record['cost'] and std_record['quantity']:
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
    if not std_record['invoice_number'] or std_record['invoice_number'] == 'None':
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


def clean_dataframe(df_uncleaned):
    """Remove all empty rows"""
    df_uncleaned = df_uncleaned.replace(r'^\s*$', pd.NA, regex=True)
    df_cleaned = df_uncleaned.dropna(how='all')
    return df_cleaned.reset_index(drop=True)


def get_dataframe_onefile(datafile):
    """Extract tables from Excel file."""
    filetype = check_file(datafile)
    if not filetype:
        raise ValueError("File not found.")
    # Check if file can be read by pandas read_table
    if filetype == '.xlsx' or filetype == '.xls':
        data_table = pd.read_excel(datafile)
    elif filetype == '.csv':
        data_table = pd.read_csv(datafile)
    elif filetype == '.pdf':
        pdf_pagecount = len(PdfReader(datafile).pages)
        if pdf_pagecount == 1:
            tables = run_textract_smallpdf(datafile)
        else:
            tables = run_textract_largepdf(datafile)
        data_table = get_df_from_textract_tables(
            tables, column_names_everytable=True)
    elif filetype == '.docx' or filetype == '.doc':
        pass
        # code to load docx table as dataframe
    return clean_dataframe(data_table)


def clean_distance_log(log_path):
    # Load the uploaded distance log file
    df_log = pd.read_csv(log_path)
    
    # Standardize empty strings or spaces in names
    df_log['Origin'] = df_log['Origin'].str.strip()
    df_log['Destination'] = df_log['Destination'].str.strip()
    
    # Fix the mixed/irregular date formatting strings dynamically
    # This natively reads configurations like 'Jan 7 2023', '2016-03-16', and '03-06-2016'
    df_log['trip_date'] = pd.to_datetime(df_log['Date'].str.strip(), errors='coerce')
    
    # Drop rows that don't have valid dates since we can't align them chronologically
    df_log = df_log.dropna(subset=['trip_date'])
    
    # Set up a timestamp anchor for the merge (defaulting to the start of the log day)
    df_log['timestamp'] = df_log['trip_date']
    
    return df_log.sort_values('timestamp')

def merge_invoices_with_distance_logs(df_clean_invoices, clean_log_path):
    # 1. Clean and fetch the distance logging profiles
    df_log = clean_distance_log(clean_log_path)
    
    # 2. Ensure your invoice dataframe is explicitly sorted by its unified timestamp
    df_inv = df_clean_invoices.sort_values('timestamp')
    
    # 3. Execute an asof merge mapping by date proximity matching backwards
    # This links the transaction to the closest recorded trip on or right before that date
    merged_df = pd.merge_asof(
        df_inv,
        df_log,
        on='timestamp',
        by='province',  # CRITICAL: Ensures the invoice province matches the log jurisdiction destination profile
        direction='nearest'
    )
    
    # 4. Generate and Map the Explicitly Requested Evaluation Columns
    merged_df['trip_origin'] = merged_df['Origin']
    merged_df['trip_destination'] = merged_df['Destination']
    merged_df['start_odometer'] = merged_df['Start_Odometer']
    merged_df['end_odometer'] = merged_df['End_Odometer']
    merged_df['distance_km'] = merged_df['Distance_km']
    
    # Set default tracking flags for fields missing from the physical log copies
    merged_df['trip_start_time'] = 'UNKNOWN'
    merged_df['trip_end_time'] = 'UNKNOWN'
    merged_df['VIN_or_truck_number'] = 'FLEET_UNIT_01'
    
    # Determine distance traveled by jurisdiction dynamically based on log maps
    # If the trip ended in the matching invoice province, assign the full distance to it
    merged_df['distance_traveled_by_jurisdiction'] = merged_df.apply(
        lambda r: f"{r['province']}: {r['distance_km']} km" if pd.notna(r['distance_km']) else "UNKNOWN", 
        axis=1
    )
    
    # 5. Drop intermediate duplicate columns to keep the output pristine
    columns_to_drop = ['Origin', 'Destination', 'Start_Odometer', 'End_Odometer', 'Distance_km', 'Date']
    merged_df = merged_df.drop(columns=[c for c in columns_to_drop if c in merged_df.columns])
    
    return merged_df


def compile_master_distance_log(df_log1_raw, df_log2_raw):
    # --- PROCESS DISTANCE LOG 1 (Spreadsheet) ---
    log1 = df_log1_raw.copy()
    log1_clean = pd.DataFrame()
    
    # Safely handle potential variations in column naming
    log1_clean['date'] = pd.to_datetime(log1['Date'])
    log1_clean['trip_origin'] = log1['Origin'].astype(str).str.strip()
    log1_clean['trip_destination'] = log1['Destination'].astype(str).str.strip()
    log1_clean['start_odometer'] = pd.to_numeric(log1['Start_Odometer'], errors='coerce')
    log1_clean['end_odometer'] = pd.to_numeric(log1['End_Odometer'], errors='coerce')
    log1_clean['distance_km'] = pd.to_numeric(log1['Distance_km'], errors='coerce')
    
    # Initialize empty jurisdiction columns for Log 1
    juris_cols = ['AB KMs', 'BC KMs', 'SK KMs', 'MB KMs', 'ON KMs', 'QC KMs', 'YT KMs',
                  'AB Fuel', 'BC Fuel', 'SK Fuel', 'MB Fuel', 'ON Fuel', 'QC Fuel']
    for col in juris_cols:
        log1_clean[col] = np.nan

    # --- PROCESS DISTANCE LOG 2 (Textractor DataFrame) ---
    log2 = df_log2_raw.copy()
    log2_clean = pd.DataFrame()
    
    # Fix the YY date formats (e.g., '02/5/22' -> 2022-05-02)
    log2_clean['date'] = pd.to_datetime(log2['DATE'].astype(str).str.strip(), format='%d/%m/%y', errors='coerce')
    log2_clean['trip_origin'] = log2['STARTING POINT'].astype(str).str.strip()
    log2_clean['trip_destination'] = log2['DESTINATION'].astype(str).str.strip()
    log2_clean['start_odometer'] = pd.to_numeric(log2['START KM'], errors='coerce')
    log2_clean['end_odometer'] = pd.to_numeric(log2['END KM'], errors='coerce')
    log2_clean['distance_km'] = pd.to_numeric(log2['TOTAL KM'], errors='coerce')
    
    # Direct pass-through for the detailed IFTA metrics
    for col in juris_cols:
        if col in log2.columns:
            log2_clean[col] = pd.to_numeric(log2[col], errors='coerce')
            
    # --- COMBINE AND CLEAN ---
    # Merge both tables vertically
    df_master_log = pd.concat([log1_clean, log2_clean], ignore_index=True)
    
    # Drop rows that don't have valid dates or odometer entries
    df_master_log = df_master_log.dropna(subset=['date', 'start_odometer'])
    
    # Apply your custom string rules: drop hyphens at edges, switch interior hyphens to spaces
    for col in ['trip_origin', 'trip_destination']:
        df_master_log[col] = (df_master_log[col]
                              .str.strip('- ')
                              .str.replace('-', ' ', regex=False)
                              .str.replace(r'\s+', ' ', regex=True))
        
    # Sort chronologically so pd.merge_asof can look back through time correctly
    df_master_log = df_master_log.sort_values(by='date').reset_index(drop=True)
    
    return df_master_log