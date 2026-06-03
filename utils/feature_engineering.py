import pandas as pd
import numpy as np

# =====================================================================
# 1. INDEPENDENT MODULES (Helper Functions)
# =====================================================================

def build_nearby_fuel_lookup(df, window_days=3):
    """
    For each trip, scans all rows within ±window_days for self-reported
    provincial fuel entries. Returns:
      - nearby_prov_fuel: nearest fuel quantity found in the window
      - nearby_window_km: sum of total_km across all trips in that window
        (used to compute window_efficiency = nearby_prov_fuel / nearby_window_km)

    IFTA context: drivers often fuel once per multi-day cycle. The window
    captures the fuelling cycle rather than the single trip.
    Window set to ±3 days based on operational fuelling behaviour assumption.
    """
    fuel_cols = ['ab_fuel', 'bc_fuel', 'mb_fuel', 'on_fuel', 'sk_fuel']
    prov_fuel_sum = df[fuel_cols].sum(axis=1, min_count=1)
    prov_fuel_sum = prov_fuel_sum.where(prov_fuel_sum > 0)

    # Build date-indexed series for fuel and km lookups
    fuel_by_date = pd.Series(
        prov_fuel_sum.values,
        index=df['trip_date']
    ).sort_index()

    km_by_date = pd.Series(
        df['total_km'].values,
        index=df['trip_date']
    ).sort_index()

    nearby_prov_fuel  = pd.Series(np.nan, index=df.index)
    nearby_window_km  = pd.Series(np.nan, index=df.index)

    for idx, trip_date in zip(df.index, df['trip_date']):
        if pd.isna(trip_date):
            continue

        lo = trip_date - pd.Timedelta(days=window_days)
        hi = trip_date + pd.Timedelta(days=window_days)

        window_fuel = fuel_by_date[
            (fuel_by_date.index >= lo) & (fuel_by_date.index <= hi)
        ].dropna()

        window_km = km_by_date[
            (km_by_date.index >= lo) & (km_by_date.index <= hi)
        ]

        if not window_fuel.empty:
            # Nearest entry by date
            seconds = (window_fuel.index - trip_date).total_seconds().values
            nearest_idx = np.argmin(np.abs(seconds))
            nearby_prov_fuel[idx] = window_fuel.iloc[nearest_idx]

        if not window_km.empty:
            nearby_window_km[idx] = window_km.sum()

    return nearby_prov_fuel, nearby_window_km


def reconcile_fuel_sources(df, window_days=3):
    """
    Builds total_fuel_litres and fuel_source using three-tier priority:
    1. invoice     — verified Textract extract, matched within ±3 days
    2. provincial_log — self-reported, matched within ±3 day window
    3. missing     — no fuel record of any kind within ±3 days

    Also returns nearby_window_km for window-level efficiency calculation.
    """
    fuel_cols = ['ab_fuel', 'bc_fuel', 'mb_fuel', 'on_fuel']
    provincial_sum = df[fuel_cols].sum(axis=1, min_count=1)
    provincial_sum = provincial_sum.where(provincial_sum > 0)

    nearby_prov_fuel, nearby_window_km = build_nearby_fuel_lookup(
        df, window_days=window_days
    )

    has_invoice      = df['matching_invoice_number'] != 'NO MATCHING INVOICE'
    has_row_fuel     = provincial_sum.notna()
    has_nearby_fuel  = nearby_prov_fuel.notna()

    # Priority: invoice → same-row provincial → nearby provincial → missing
    conditions = [
        has_invoice,
        has_row_fuel & ~has_invoice,
        has_nearby_fuel & ~has_row_fuel & ~has_invoice,
    ]
    choices = ['invoice', 'provincial_log', 'provincial_log_nearby']

    fuel_source = pd.Series(
        np.select(conditions, choices, default='missing'),
        index=df.index
    )

    # Total fuel: invoice/row fuel takes priority, nearby fills gaps
    total_fuel = provincial_sum.combine_first(nearby_prov_fuel)

    return total_fuel, fuel_source, nearby_prov_fuel, nearby_window_km


def calculate_efficiency_and_costs(df, window_days=3):
    """
    Computes two complementary efficiency metrics:

    trip_fuel_per_km:   provincial_sum  ÷ total_km
                        Single-trip efficiency — catches per-trip anomalies

    window_fuel_per_km: nearby_prov_fuel ÷ nearby_window_km
                        Fuelling-cycle efficiency — catches patterns that
                        look fine per trip but anomalous over a cycle

    Both are retained as features. In an IFTA audit, auditors examine both
    the individual receipt and the overall fuelling pattern.
    """
    total_fuel, fuel_source, nearby_prov_fuel, nearby_window_km = \
        reconcile_fuel_sources(df, window_days=window_days)

    # Trip-level efficiency
    trip_fuel_per_km = np.where(
        total_fuel.isna() | (df['total_km'] == 0),
        np.nan,
        total_fuel / df['total_km']
    )

    # Window-level efficiency
    window_fuel_per_km = np.where(
        nearby_prov_fuel.isna() | nearby_window_km.isna() | (nearby_window_km == 0),
        np.nan,
        nearby_prov_fuel / nearby_window_km
    )

    # Cost per km (uses verified total_cost only)
    fuel_cost_per_km = np.where(
        df['total_cost'].isna() | (df['total_km'] == 0),
        np.nan,
        df['total_cost'] / df['total_km']
    )

    return (total_fuel, fuel_source, nearby_prov_fuel, nearby_window_km,
            trip_fuel_per_km, window_fuel_per_km, fuel_cost_per_km)


def calculate_provincial_sum_km(df):
    """
    Sums all provincial km columns into one total.
    
    IFTA context: this should equal total_km for compliant records.
    A mismatch between provincial_sum_km and total_km means the
    jurisdictional breakdown doesn't account for all distance driven —
    a direct compliance gap.
    """
    km_cols = ['ab_km', 'bc_km', 'mb_km', 'on_km', 'sk_km']
    provincial_sum_km = df[km_cols].sum(axis=1, min_count=1)
    return provincial_sum_km


def flag_compliance_gaps(df):
    """Flags binary indicators targeting unvouched trips for audit analysis."""
    return np.where(df['matching_invoice_number'] == 'NO MATCHING INVOICE', 1,
                    0)


def calculate_jurisdictional_proportions(df, total_fuel_series):
    """Computes variances between where miles were driven vs where fuel was bought."""
    proportions_dict = {}
    
    for prov in ['ab', 'bc', 'mb', 'on', 'sk']:
        km_col = f'{prov}_km'
        fuel_col = f'{prov}_fuel'
        
        # 1. Distance Proportion
        proportions_dict[f'{prov}_dist_prop'] = np.where(
            df['total_km'] == 0,
            0.0,
            df[km_col] / df['total_km']
        )
        
        # 2. Fuel Proportion
        proportions_dict[f'{prov}_fuel_prop'] = np.where(
            total_fuel_series.isna() | (total_fuel_series == 0),
            0.0,
            df[fuel_col].fillna(0.0) / total_fuel_series
        )
        
        # 3. Spatial Variance (Distance % minus Fuel %)
        proportions_dict[f'{prov}_dist_fuel_variance'] = (
            proportions_dict[f'{prov}_dist_prop'] - proportions_dict[f'{prov}_fuel_prop']
        )
        
    return pd.DataFrame(proportions_dict, index=df.index)


def extract_temporal_trends(df):
    """
    Computes rolling averages using index-based mapping to bypass merge issues.
    """
    # 1. Aggregate to daily
    daily_stats = df.groupby('trip_date').agg({
        'total_km': 'sum',
        'total_fuel_litres': 'sum'
    })
    
    # 2. Calculate rolling (Ensure index is sorted for rolling to work!)
    daily_stats = daily_stats.sort_index()
    rolling_7d = daily_stats.rolling(window='7D', closed='left')
    rolling_30d = daily_stats.rolling(window='30D', closed='left')
    
    daily_stats['w_dist'] = rolling_7d['total_km'].mean()
    daily_stats['w_fuel'] = rolling_7d['total_fuel_litres'].mean()
    daily_stats['m_dist'] = rolling_30d['total_km'].mean()
    daily_stats['m_fuel'] = rolling_30d['total_fuel_litres'].mean()
    
    # 3. MAPPING instead of MERGING
    # This takes each 'trip_date' in your df and looks it up in the daily_stats index
    df['weekly_avg_distance_km'] = df['trip_date'].map(daily_stats['w_dist'])
    df['weekly_avg_fuel_litres'] = df['trip_date'].map(daily_stats['w_fuel'])
    df['monthly_avg_distance_km'] = df['trip_date'].map(daily_stats['m_dist'])
    df['monthly_avg_fuel_litres'] = df['trip_date'].map(daily_stats['m_fuel'])
    
    # 4. Fill gaps only after mapping
    return df.fillna(0.0)


def calculate_temporal_ratios(df):
    """
    Computes how much each trip deviates from its rolling baseline.
    Must be called AFTER extract_temporal_trends so the avg columns exist.
    
    IFTA context: a single trip that is 3x the weekly average is suspicious
    even if its absolute fuel/distance values look plausible in isolation.
    """
    df['dist_to_weekly_ratio'] = df.apply(
        lambda r: r['total_km'] / r['weekly_avg_distance_km']
        if r['weekly_avg_distance_km'] > 0 else 1.0, axis=1
    )
    df['fuel_to_weekly_ratio'] = df.apply(
        lambda r: r['total_fuel_litres'] / r['weekly_avg_fuel_litres']
        if r['weekly_avg_fuel_litres'] > 0 else 1.0, axis=1
    )
    df['dist_to_monthly_ratio'] = df.apply(
        lambda r: r['total_km'] / r['monthly_avg_distance_km']
        if r['monthly_avg_distance_km'] > 0 else 1.0, axis=1
    )
    df['fuel_to_monthly_ratio'] = df.apply(
        lambda r: r['total_fuel_litres'] / r['monthly_avg_fuel_litres']
        if r['monthly_avg_fuel_litres'] > 0 else 1.0, axis=1
    )
    return df


def calculate_odometer_gap(df):
    """
    Difference between a trip's start_odometer and the previous trip's 
    end_odometer. A large gap means undocumented distance — the single 
    biggest red flag in IFTA audits.
    
    IFTA context: if end_odometer is 50,000 and next trip starts at 51,200,
    there are 1,200 undocumented km that need to be explained.
    """
    df_sorted = df.sort_values('trip_date').copy()
    df_sorted['prev_end_odometer'] = df_sorted['end_odometer'].shift(1)
    df_sorted['odometer_gap'] = (
        df_sorted['start_odometer'] - df_sorted['prev_end_odometer']
    )
    # Negative gaps = odometer reset or data error, treat as missing
    df_sorted['odometer_gap'] = df_sorted['odometer_gap'].clip(lower=0)
    return df_sorted['odometer_gap']


def encode_fuel_source_reliability(fuel_source_series):
    """
    Ordinal encoding of fuel source by confidence level.
    The model can use this as a continuous risk signal.
    
    invoice=3 (verified), provincial_log=2 (self-reported same row),
    provincial_log_nearby=1 (self-reported nearby), missing=0 (red flag)
    """
    reliability_map = {
        'invoice': 3,
        'provincial_log': 2,
        'provincial_log_nearby': 1,
        'missing': 0
    }
    return fuel_source_series.map(reliability_map).fillna(0).astype(int)


def flag_cross_border_trips(df):
    """
    Binary flag: 1 if origin and destination are in different provinces.
    Cross-border trips are the core IFTA compliance event — anomalies
    on these trips are higher audit priority than intra-provincial ones.
    """
    province_keywords = ['AB', 'BC', 'MB', 'ON', 'SK', 'QC', 'YT']

    def extract_province(location_str):
        if pd.isna(location_str):
            return None
        loc = str(location_str).upper()
        for prov in province_keywords:
            if prov in loc:
                return prov
        return None

    origin_prov = df['trip_origin'].apply(extract_province)
    dest_prov   = df['trip_destination'].apply(extract_province)

    return np.where(
        origin_prov.isna() | dest_prov.isna(),
        np.nan,
        (origin_prov != dest_prov).astype(int)
    )

# =====================================================================
# 2. MASTER ORCHESTRATOR PIPELINE
# =====================================================================

def engineer_ifta_features(df_ml_ready, window_days=3):
    """Add engineered features."""
    df_feat = df_ml_ready.copy()

    # Module 1: Reconciled fuel + dual efficiency metrics
    # Unpack all 7 return values — nearby_prov_fuel and nearby_window_km
    # are now explicit columns, not hidden intermediates
    (total_fuel, fuel_source,
     nearby_prov_fuel, nearby_window_km,
     trip_fuel_per_km, window_fuel_per_km,
     cost_per_km) = calculate_efficiency_and_costs(df_feat, window_days=window_days)

    df_feat['total_fuel_litres']       = total_fuel
    df_feat['fuel_source']             = fuel_source
    df_feat['fuel_source_reliability'] = encode_fuel_source_reliability(fuel_source)
    df_feat['nearby_prov_fuel']        = nearby_prov_fuel    # ← was missing
    df_feat['nearby_window_km']        = nearby_window_km    # ← was missing
    df_feat['fuel_litres_per_km']      = trip_fuel_per_km
    df_feat['window_fuel_per_km']      = window_fuel_per_km
    df_feat['fuel_cost_per_km']        = cost_per_km

    # Module 2: Provincial km sum + mismatch vs total_km
    df_feat['provincial_sum_km']       = calculate_provincial_sum_km(df_feat)  # ← new
    df_feat['km_reconciliation_gap']   = (                                      # ← new
        df_feat['total_km'] - df_feat['provincial_sum_km']
    ).clip(lower=0)
    # A non-zero gap means total distance driven exceeds what's jurisdictionally
    # accounted for — a direct IFTA reporting gap

    # Module 3: Odometer continuity
    df_feat['odometer_gap']            = calculate_odometer_gap(df_feat)

    # Module 4: Cross-border flag (post-hoc filter, not model feature)
    df_feat['cross_border_trip']       = flag_cross_border_trips(df_feat)

    # Module 5: Jurisdictional proportions
    jurisdiction_features = calculate_jurisdictional_proportions(df_feat, total_fuel)
    df_feat = pd.concat([df_feat, jurisdiction_features], axis=1)

    # Module 6: Temporal rolling averages
    df_feat = extract_temporal_trends(df_feat)

    # Module 7: Temporal deviation ratios — MUST come after Module 6
    df_feat = calculate_temporal_ratios(df_feat)    # ← was missing

    return df_feat