"""Python program to merge pandas dataframes of different quantities."""

import numpy as np
import pandas as pd


def build_unified_ml_dataset(df_log1, df_log2, df_invoices, days_window=3):
    """"Build final dataset me merging distance logs and invoices."""
    l1 = df_log1.copy()
    l2 = df_log2.copy()
    inv = df_invoices.copy()

    # Clean up whitespace and lowercase column names, just for safety
    l1.columns = l1.columns.astype(str).str.lower().str.replace(
        r'\s+', ' ', regex=True).str.strip()
    l2.columns = l2.columns.astype(str).str.lower().str.replace(
        r'\s+', ' ', regex=True).str.strip()
    inv.columns = inv.columns.astype(str).str.lower().str.replace(
        r'\s+', ' ', regex=True).str.strip()

    # Distance logs arrive from readfile_to_df already standardized to
    # extract_utils.py's distance_columns schema — 'trip_date' (not
    # 'date'), 'trip_origin', 'trip_destination', 'start_odometer',
    # 'end_odometer', 'distance_km', 'ab_kms'/'bc_kms'/etc, 'ab_fuel'/etc.
    # Invoices keep their own schema, where 'date' IS the correct name.
    l1['trip_date'] = pd.to_datetime(l1['trip_date'], errors='coerce')
    l2['trip_date'] = pd.to_datetime(l2['trip_date'], errors='coerce')
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
    sk_km = np.zeros(n_rows)

    ab_fuel = np.full(n_rows, np.nan)
    bc_fuel = np.full(n_rows, np.nan)
    mb_fuel = np.full(n_rows, np.nan)
    on_fuel = np.full(n_rows, np.nan)
    sk_fuel = np.full(n_rows, np.nan)

    # 'distance_km' is guaranteed present by the standardized schema —
    # no fallback chain needed.
    l1_total_km = pd.to_numeric(l1['distance_km'],
                                errors='coerce').fillna(0.0).values

    for idx, row in l1.reset_index(drop=True).iterrows():
        if pd.isna(row['trip_date']):
            continue
        matching_inv = inv[
            (inv['date'] >= row['trip_date'] - pd.Timedelta(days=days_window)) &
            (inv['date'] <= row['trip_date'] + pd.Timedelta(days=days_window))
        ]
        if not matching_inv.empty:
            matching_inv = matching_inv.copy()
            matching_inv['_date_diff'] = (
                 matching_inv['date'] - row['trip_date']).abs()
            target_inv = matching_inv.sort_values('_date_diff').iloc[0]
            prov = str(target_inv.get('province', 'UNKNOWN')).upper().strip()
            city = str(target_inv.get('city', 'UNKNOWN')).upper().strip()
            qty = pd.to_numeric(target_inv.get('quantity', 0.0))
            cost_val = pd.to_numeric(target_inv.get('cost', 0.0))

            matching_invoice_number[idx] = str(
                target_inv.get('invoice_number',
                               target_inv.get('receipt_number', 'UNKNOWN')))
            invoice_date[idx] = target_inv['date']

            t_val = target_inv.get('time')
            if pd.notna(t_val):
                invoice_time[idx] = str(t_val).strip()

            invoice_location[idx] = (f"{city}, {prov}"
                                     if prov != 'UNKNOWN'
                                     else 'NO MATCHING INVOICE')
            total_cost[idx] = cost_val

            # Map Jurisdictional Distance and Fuel using the found invoice province
            dist = l1_total_km[idx]
            if prov == 'AB':
                ab_km[idx] = dist
                ab_fuel[idx] = qty
            elif prov == 'BC':
                bc_km[idx] = dist
                bc_fuel[idx] = qty
            elif prov == 'MB':
                mb_km[idx] = dist
                mb_fuel[idx] = qty
            elif prov == 'ON':
                on_km[idx] = dist
                on_fuel[idx] = qty
            elif prov == 'SK':
                sk_km[idx] = dist
                sk_fuel[idx] = qty

    log1_mapped = pd.DataFrame({
        'trip_date': l1['trip_date'],
        'trip_origin': l1['trip_origin'],
        'trip_destination': l1['trip_destination'],
        'start_odometer': pd.to_numeric(l1['start_odometer'],
                                        errors='coerce').fillna(0.0),
        'end_odometer': pd.to_numeric(l1['end_odometer'],
                                      errors='coerce').fillna(0.0),
        'total_km': l1_total_km,

        'ab_km': ab_km,
        'bc_km': bc_km,
        'mb_km': mb_km,
        'on_km': on_km,
        'sk_km': sk_km,

        'ab_fuel': ab_fuel,
        'bc_fuel': bc_fuel,
        'mb_fuel': mb_fuel,
        'on_fuel': on_fuel,
        'sk_fuel': sk_fuel,

        'vin_or_truck_number': l1['vin_or_truck_number'],
        'matching_invoice_number': matching_invoice_number,
        'invoice_date': invoice_date,
        'invoice_time': invoice_time,
        'invoice_location': invoice_location,
        'total_cost': total_cost,
        'source_log': l1['source_file']
    })

    # ----------------------------------------------------
    # PHASE B: Process Distance Log 2 (2022 PDF)
    # ----------------------------------------------------
    raw_ab_fuel = pd.to_numeric(l2['ab_fuel'], errors='coerce').fillna(0.0)
    raw_bc_fuel = pd.to_numeric(l2['bc_fuel'], errors='coerce').fillna(0.0)
    l2_start_odo = pd.to_numeric(l2['start_odometer'],
                                 errors='coerce').fillna(0.0)
    l2_end_odo = pd.to_numeric(l2['end_odometer'],
                               errors='coerce').fillna(0.0)
    l2_total_km = (l2_end_odo - l2_start_odo).clip(lower=0)

    log2_mapped = pd.DataFrame({
        'trip_date': l2['trip_date'],
        'trip_origin': l2['trip_origin'],
        'trip_destination': l2['trip_destination'],
        'start_odometer':   l2_start_odo,
        'end_odometer':     l2_end_odo,
        'total_km':         l2_total_km,

        'ab_km': pd.to_numeric(l2['ab_kms'], errors='coerce').fillna(0.0),
        'bc_km': pd.to_numeric(l2['bc_kms'], errors='coerce').fillna(0.0),
        'mb_km': pd.to_numeric(l2['mb_kms'], errors='coerce').fillna(0.0),
        'on_km': pd.to_numeric(l2['on_kms'], errors='coerce').fillna(0.0),
        'sk_km': pd.to_numeric(l2['sk_kms'], errors='coerce').fillna(0.0),

        'ab_fuel': np.where(raw_ab_fuel > 0, raw_ab_fuel, np.nan),
        'bc_fuel': np.where(raw_bc_fuel > 0, raw_bc_fuel, np.nan),
        'mb_fuel': np.nan,
        'on_fuel': np.nan,
        'sk_fuel': np.nan,

        'vin_or_truck_number': l2['vin_or_truck_number'],
        'matching_invoice_number': 'NO MATCHING INVOICE',
        'invoice_date': pd.NaT,
        'invoice_time': pd.NaT,
        'invoice_location': 'NO MATCHING INVOICE',
        'total_cost': np.nan,
        'source_log': l2['source_file']
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