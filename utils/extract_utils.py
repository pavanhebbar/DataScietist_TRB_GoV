"""Python program to extract IFTA relevant data."""

import io
import boto3
from botocore.exceptions import ClientError
import docx2txt
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

            if exp_data_full.summary_fields_list:
                # Get summary fields
                summ_dict = []
                for field in exp_data_full.summary_fields_list:
                    f_type = field.type.text if field.type else ""
                    f_label = field.key.text if field.key else ""
                    f_value = field.value.text if field.value else ""
                    summ_dict.append({
                        'type' : str(f_type).strip().upper(),
                        'label': str(f_label).strip(),
                        'value': str(f_value).strip().lower(),
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
    """Helper to strip currencies, spaces, etc. from numeric data."""
    if not val_str:
        return None
    # Strip symbols and chars
    cleaned = re.sub(r'[^\d\.\-]', '', val_str)
    return cleaned if cleaned else None


def update_field_value(f_type, f_label, f_val, match_fields,
                       currstd_fval, curr_ftype_rank=50, clean=False):
    """Check whether the input field should be used to update std_field."""
    std_fval = currstd_fval
    if f_type in match_fields:
        ftype_pos = match_fields.index(f_type)
    else:
        ftype_pos = 100
    if f_label in match_fields:
        flabel_pos = match_fields.index(f_label)
    else:
        flabel_pos = 100
    if flabel_pos < ftype_pos:
        ftype_pos = flabel_pos

    if ftype_pos < curr_ftype_rank and std_fval:
            clean_fval = clean_numeric(f_val) if clean else f_val.upper()
            if clean_fval:
                std_fval = clean_fval
                curr_ftype_rank = ftype_pos

    return std_fval, curr_ftype_rank


def extract_clean_record(raw_exp_data):
    """Select only relevant fields."""
    clean_records = []
    # Mapping record keys with expected field types.
    field_map = {
        'province': ['STATE'], 'date': ['INVOICE_RECEIPT_DATE'],
        'quantity': ['litres', 'l'],
        'cost':['TOTAL', 'AMOUNT_PAID', 'fuel sales', 'AMOUNT_DUE'],
        'fuel_type': ['product', 'grade', 'fuel'],
        'cost_per_litre': ['UNIT_PRICE']}
    for record in raw_exp_data:
        # Skeletal clean record
        std_record = {
            'province': 'UNKNOWN', 'date': None, 'qunatity': None,
            'cost': None, 'fuel_type': 'UNKNOWN',
            'cost_per_litre': None
        }
        clean_numeric_bool = [False, False, True, True, False, True]
        curr_std_ranks = [50, 50, 50, 50, 50, 50]
        summ_list = record['fields']
        for field in summ_list:
            for i, std_key in enumerate(std_record):
                std_fval, curr_std_ranks = update_field_value(
                    field['type'], field['label'], field['value'],
                    field_map[std_key], std_record[std_key], curr_std_ranks[i],
                    clean_numeric_bool[i]
                )
        




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
        return process_images_textractor(image_paths)

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
