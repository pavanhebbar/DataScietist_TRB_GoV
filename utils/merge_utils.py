"""Python program to merge pandas dataframes of different quantities."""

import numpy as np
import pandas as pd

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


def build_unified_ml_dataset(df_log1, df_log2, df_invoices):
    l1 = df_log1.copy()
    l2 = df_log2.copy()
    inv = df_invoices.copy()
    
    # Clean up whitespace and lowercase column names
    l1.columns = l1.columns.astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    l2.columns = l2.columns.astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    inv.columns = inv.columns.astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Convert dates to standard Datetime objects safely
    l1['date'] = pd.to_datetime(l1['date'], errors='coerce')
    l2['date'] = pd.to_datetime(l2['date'], errors='coerce')
    inv['date'] = pd.to_datetime(inv['date'], errors='coerce')
    
    # ----------------------------------------------------
    # PHASE A: Process Distance Log 1 (2016-2021 Excel) using Invoices
    # ----------------------------------------------------
    # Initialize explicit arrays to guarantee correct row-by-row mapping
    n_rows = len(l1)
    matching_invoice_number = ['NO MATCHING INVOICE'] * n_rows
    invoice_date = [pd.NaT] * n_rows
    invoice_time = [pd.NaT] * n_rows
    invoice_location = ['NO MATCHING INVOICE'] * n_rows
    total_cost = [np.nan] * n_rows
    
    ab_km = np.zeros(n_rows)
    bc_km = np.zeros(n_rows)
    mb_km = np.zeros(n_rows)
    on_km = np.zeros(n_rows)
    
    ab_fuel = np.full(n_rows, np.nan)
    bc_fuel = np.full(n_rows, np.nan)
    mb_fuel = np.full(n_rows, np.nan)
    on_fuel = np.full(n_rows, np.nan)
    
    l1_dist_col = 'distance_km' if 'distance_km' in l1.columns else ('distance km' if 'distance km' in l1.columns else l1.columns[0])
    l1_total_km = pd.to_numeric(l1[l1_dist_col], errors='coerce').fillna(0.0).values

    for idx, row in l1.reset_index(drop=True).iterrows():
        if pd.isna(row['date']):
            continue
        matching_inv = inv[
            (inv['date'] >= row['date']) & 
            (inv['date'] <= row['date'] + pd.Timedelta(days=7))
        ]
        if not matching_inv.empty:
            target_inv = matching_inv.sort_values('date').iloc[0]
            prov = str(target_inv.get('province', 'UNKNOWN')).upper().strip()
            city = str(target_inv.get('city', 'UNKNOWN')).upper().strip()
            qty = pd.to_numeric(target_inv.get('quantity', 0.0))
            cost_val = pd.to_numeric(target_inv.get('cost', 0.0))
            
            matching_invoice_number[idx] = str(target_inv.get('invoice_number', target_inv.get('receipt_number', 'UNKNOWN')))
            invoice_date[idx] = target_inv['date']
            
            # Formats time cleanly as a string 'HH:MM:SS' instead of tuple components
            t_val = target_inv.get('time')
            if pd.notna(t_val):
                invoice_time[idx] = str(t_val).strip()
                
            invoice_location[idx] = f"{city}, {prov}" if prov != 'UNKNOWN' else 'NO MATCHING INVOICE'
            total_cost[idx] = cost_val
            
            # Map Jurisdictional Distance and Fuel using the found invoice province
            dist = l1_total_km[idx]
            if prov == 'AB':
                ab_km[idx] = dist; ab_fuel[idx] = qty
            elif prov == 'BC':
                bc_km[idx] = dist; bc_fuel[idx] = qty
            elif prov == 'MB':
                mb_km[idx] = dist; mb_fuel[idx] = qty
            elif prov == 'ON':
                on_km[idx] = dist; on_fuel[idx] = qty

    def find_column_value(df, variations_list):
        for variant in variations_list:
            if variant in df.columns:
                return df[variant]
        return 'UNKNOWN'

    log1_mapped = pd.DataFrame({
        'trip_date': l1['date'],
        'trip_origin': find_column_value(l1, ['origin', 'trip origin', 'trip_origin', 'starting point']),
        'trip_destination': find_column_value(l1, ['trip destination', 'trip_destination', 'destination']),
        'start_odometer': pd.to_numeric(find_column_value(l1, ['start odometer', 'start_odometer', 'start km']), errors='coerce').fillna(0.0),
        'end_odometer': pd.to_numeric(find_column_value(l1, ['end odometer', 'end_odometer', 'end km']), errors='coerce').fillna(0.0),
        'total_km': l1_total_km,
        
        'ab_km': ab_km,
        'bc_km': bc_km,
        'mb_km': mb_km,
        'on_km': on_km,
        
        'ab_fuel': ab_fuel,
        'bc_fuel': bc_fuel,
        'mb_fuel': mb_fuel,
        'on_fuel': on_fuel,
        
        'matching_invoice_number': matching_invoice_number,
        'invoice_date': invoice_date,
        'invoice_time': invoice_time,
        'invoice_location': invoice_location,
        'total_cost': total_cost,
        'source_log': 'Log_1_Excel'
    })

    # ----------------------------------------------------
    # PHASE B: Process Distance Log 2 (2022 PDF)
    # ----------------------------------------------------
    raw_ab_fuel = pd.to_numeric(find_column_value(l2, ['ab fuel', 'ab_fuel']), errors='coerce').fillna(0.0)
    raw_bc_fuel = pd.to_numeric(find_column_value(l2, ['bc fuel', 'bc_fuel']), errors='coerce').fillna(0.0)
    
    log2_mapped = pd.DataFrame({
        'trip_date': l2['date'],
        'trip_origin': find_column_value(l2, ['starting point', 'starting_point', 'origin', 'trip origin']),
        'trip_destination': find_column_value(l2, ['destination', 'trip destination']),
        'start_odometer': pd.to_numeric(find_column_value(l2, ['start km', 'start_km', 'start odometer']), errors='coerce').fillna(0.0),
        'end_odometer': pd.to_numeric(find_column_value(l2, ['end km', 'end_km', 'end odometer']), errors='coerce').fillna(0.0),
        'total_km': pd.to_numeric(find_column_value(l2, ['total km', 'total_km']), errors='coerce').fillna(0.0),
        
        'ab_km': pd.to_numeric(find_column_value(l2, ['ab kms', 'ab_kms', 'ab']), errors='coerce').fillna(0.0),
        'bc_km': pd.to_numeric(find_column_value(l2, ['bc kms', 'bc_kms', 'bc']), errors='coerce').fillna(0.0),
        'mb_km': pd.to_numeric(find_column_value(l2, ['mb kms', 'mb_kms', 'mb']), errors='coerce').fillna(0.0),
        'on_km': 0.0,
        
        'ab_fuel': np.where(raw_ab_fuel > 0, raw_ab_fuel, np.nan),
        'bc_fuel': np.where(raw_bc_fuel > 0, raw_bc_fuel, np.nan),
        'mb_fuel': np.nan,
        'on_fuel': np.nan,
        
        'matching_invoice_number': 'NO MATCHING INVOICE',
        'invoice_date': pd.NaT,
        'invoice_time': pd.NaT,
        'invoice_location': 'NO MATCHING INVOICE',
        'total_cost': np.nan,
        'source_log': 'Log_2_PDF'
    })

    # ----------------------------------------------------
    # PHASE C: Combine, Datatype Cast, and Sort
    # ----------------------------------------------------
    df_ml_ready = pd.concat([log1_mapped, log2_mapped], ignore_index=True)
    
    df_ml_ready['trip_date'] = pd.to_datetime(df_ml_ready['trip_date'], errors='coerce')
    df_ml_ready['invoice_date'] = pd.to_datetime(df_ml_ready['invoice_date'], errors='coerce')
    
    df_ml_ready = df_ml_ready.dropna(subset=['trip_date'])
    df_ml_ready = df_ml_ready.sort_values('trip_date').reset_index(drop=True)
    
    return df_ml_ready