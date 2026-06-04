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
    Three-tier fuel source priority:
    1. invoice              — Textract-verified, matched within ±window_days
    2. provincial_log       — self-reported, present on this exact row
    3. provincial_log_nearby — self-reported, nearest cycle fuel within
                               window_days of this trip
    4. missing              — no fuel record within window_days in either
                               direction; trip is unverifiable

    window_days is the maximum acceptable gap to the nearest fuel event.
    Beyond this, the trip has no verifiable fuel anchor for IFTA purposes.
    """
    fuel_cols = ['ab_fuel', 'bc_fuel', 'mb_fuel', 'on_fuel', 'sk_fuel']
    provincial_sum = df[fuel_cols].sum(axis=1, min_count=1)
    provincial_sum = provincial_sum.where(provincial_sum > 0)

    km_per_tank, avg_fuel_per_tank, min_gap_days = \
        calculate_fuelling_cycle_features(df, max_gap_days=window_days)

    has_invoice    = df['matching_invoice_number'] != 'NO MATCHING INVOICE'
    has_row_fuel   = provincial_sum.notna()
    has_cycle_fuel = min_gap_days <= window_days  # at least one fill-up nearby

    conditions = [
        has_invoice,
        has_row_fuel   & ~has_invoice,
        has_cycle_fuel & ~has_row_fuel & ~has_invoice,
    ]
    choices = ['invoice', 'provincial_log', 'provincial_log_nearby']

    fuel_source = pd.Series(
        np.select(conditions, choices, default='missing'),
        index=df.index
    )

    # Row-level fuel takes priority; cycle average fills gaps
    total_fuel = provincial_sum.combine_first(avg_fuel_per_tank)

    return total_fuel, fuel_source, km_per_tank, avg_fuel_per_tank


def calculate_efficiency_and_costs(df, window_days=3):
    """
    trip_fuel_per_km:  total_fuel  ÷ total_km   — single trip signal
    cycle_fuel_per_km: avg_fuel_per_tank ÷ km_per_tank — tank range signal
    """
    total_fuel, fuel_source, km_per_tank, avg_fuel_per_tank = \
        reconcile_fuel_sources(df, window_days=window_days)

    trip_fuel_per_km = np.where(
        total_fuel.isna() | (df['total_km'] == 0),
        np.nan,
        total_fuel / df['total_km']
    )

    cycle_fuel_per_km = np.where(
        avg_fuel_per_tank.isna() | km_per_tank.isna() | (km_per_tank == 0),
        np.nan,
        avg_fuel_per_tank / km_per_tank
    )

    fuel_cost_per_km = np.where(
        df['total_cost'].isna() | (df['total_km'] == 0),
        np.nan,
        df['total_cost'] / df['total_km']
    )

    return (total_fuel, fuel_source,
            km_per_tank, avg_fuel_per_tank,
            trip_fuel_per_km, cycle_fuel_per_km,
            fuel_cost_per_km)


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
    Flags trips where the start_odometer of a trip doesn't follow on from
    the end_odometer of the previous trip for the SAME truck.

    Computed per source_log (truck) to avoid false gaps at the boundary
    between Log 1 (2016-2021) and Log 2 (2022), which are two different
    vehicles with independent odometer sequences.

    Returns a binary flag: 1 if undocumented gap > 50 km, 0 otherwise.
    NaN where odometer data is missing for either the current or prior trip.

    IFTA context: a 50 km gap means ~50 km of unlogged distance that cannot
    be allocated to any jurisdiction — a direct compliance failure.
    """
    df_sorted = df.sort_values('trip_date').copy()

    # Shift within each truck's log — never carry the last row of Log 1
    # into the first row of Log 2
    df_sorted['prev_end_odometer'] = (
        df_sorted.groupby('source_log')['end_odometer'].shift(1)
    )

    raw_gap = (
        df_sorted['start_odometer'] - df_sorted['prev_end_odometer']
    ).clip(lower=0)

    # Binary flag — NaN when either odometer reading is missing
    has_both = df_sorted['start_odometer'].notna() & df_sorted['prev_end_odometer'].notna()
    odometer_gap_flag = pd.Series(np.nan, index=df_sorted.index)
    odometer_gap_flag[has_both] = (raw_gap[has_both] > 50).astype(int)

    return odometer_gap_flag


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


def calculate_fuelling_cycle_features(df, max_gap_days=3):
    """
    For each trip, finds the fuelling cycle it belongs to by locating the 
    nearest preceding and following fuel purchase events.

    Returns:
      km_per_tank:       Total km driven across ALL trips between the two 
                         surrounding fuel events. Shared by all trips in the 
                         same cycle. Physically validates tank range.
      avg_fuel_per_tank: Average of the two surrounding fill-up quantities.
                         Used as the denominator-neutral fuel estimate for 
                         cycle efficiency.
      min_gap_days:      Days to the nearest fuel event in either direction.
                         Controls fuel_source = 'missing' when > max_gap_days.

    IFTA context: auditors think in fuelling cycles, not fixed date windows.
    A truck with a 400L tank cannot travel 3000 km between fill-ups.
    The ±max_gap_days constraint flags trips with no fuel record nearby,
    which in an IFTA audit means the distance is unverifiable.
    """
    fuel_cols = ['ab_fuel', 'bc_fuel', 'mb_fuel', 'on_fuel', 'sk_fuel']
    prov_fuel_sum = df[fuel_cols].sum(axis=1, min_count=1)
    prov_fuel_sum = prov_fuel_sum.where(prov_fuel_sum > 0)

    # Aggregate by date — multiple fill-ups on same day are summed
    fuel_events = (
        pd.Series(prov_fuel_sum.values, index=df['trip_date'])
        .dropna()
        .groupby(level=0)
        .sum()
        .sort_index()
    )

    km_by_date = (
        pd.Series(df['total_km'].values, index=df['trip_date'])
        .groupby(level=0)
        .sum()
        .sort_index()
    )

    fuel_dates        = fuel_events.index
    km_per_tank       = pd.Series(np.nan, index=df.index)
    avg_fuel_per_tank = pd.Series(np.nan, index=df.index)
    min_gap_days      = pd.Series(np.inf, index=df.index)

    for idx, trip_date in zip(df.index, df['trip_date']):
        if pd.isna(trip_date):
            continue

        # searchsorted: pos = insertion point after all equal values
        pos = fuel_dates.searchsorted(trip_date, side='right')

        prev_fuel_date = fuel_dates[pos - 1] if pos > 0 else None
        next_fuel_date = fuel_dates[pos]     if pos < len(fuel_dates) else None

        prev_gap = (trip_date - prev_fuel_date).days \
                   if prev_fuel_date is not None else np.inf
        next_gap = (next_fuel_date - trip_date).days \
                   if next_fuel_date is not None else np.inf

        min_gap_days[idx] = min(prev_gap, next_gap)

        # --- km_per_tank: all km strictly between the two fuel events ---
        if prev_fuel_date is not None and next_fuel_date is not None:
            # Trips after prev fill-up up to and including next fill-up date
            cycle_km = km_by_date[
                (km_by_date.index > prev_fuel_date) &
                (km_by_date.index <= next_fuel_date)
            ].sum()
            km_per_tank[idx]       = cycle_km if cycle_km > 0 else np.nan
            avg_fuel_per_tank[idx] = (
                fuel_events[prev_fuel_date] + fuel_events[next_fuel_date]
            ) / 2

        # Edge case: trip is before any fill-up
        elif next_fuel_date is not None:
            cycle_km = km_by_date[
                km_by_date.index <= next_fuel_date
            ].sum()
            km_per_tank[idx]       = cycle_km if cycle_km > 0 else np.nan
            avg_fuel_per_tank[idx] = fuel_events[next_fuel_date]

        # Edge case: trip is after the last fill-up
        elif prev_fuel_date is not None:
            cycle_km = km_by_date[
                km_by_date.index > prev_fuel_date
            ].sum()
            km_per_tank[idx]       = cycle_km if cycle_km > 0 else np.nan
            avg_fuel_per_tank[idx] = fuel_events[prev_fuel_date]

    return km_per_tank, avg_fuel_per_tank, min_gap_days

# =====================================================================
# 2. MASTER ORCHESTRATOR PIPELINE
# =====================================================================

def engineer_ifta_features(df_ml_ready, window_days=3):
    df_feat = df_ml_ready.copy()

    # Module 1: Fuel reconciliation + efficiency
    (total_fuel, fuel_source,
     km_per_tank, avg_fuel_per_tank,
     trip_fuel_per_km, cycle_fuel_per_km,
     cost_per_km) = calculate_efficiency_and_costs(df_feat, window_days=window_days)

    df_feat['total_fuel_litres']       = total_fuel
    df_feat['fuel_source']             = fuel_source
    df_feat['fuel_source_reliability'] = encode_fuel_source_reliability(fuel_source)
    df_feat['km_per_tank']             = km_per_tank       # cycle distance
    df_feat['avg_fuel_per_tank']       = avg_fuel_per_tank # cycle fuel anchor
    df_feat['fuel_litres_per_km']      = trip_fuel_per_km  # per-trip signal
    df_feat['cycle_fuel_per_km']       = cycle_fuel_per_km # tank-range signal
    df_feat['fuel_cost_per_km']        = cost_per_km

    # Module 2: Provincial km sum + reconciliation gap
    df_feat['provincial_sum_km']     = calculate_provincial_sum_km(df_feat)
    df_feat['km_reconciliation_gap'] = (
        df_feat['total_km'] - df_feat['provincial_sum_km']
    ).clip(lower=0)

    # Module 3: Odometer continuity
    df_feat['odometer_gap'] = calculate_odometer_gap(df_feat)

    # Module 4: Cross-border flag (post-hoc filter only, not in model)
    df_feat['cross_border_trip'] = flag_cross_border_trips(df_feat)

    # Module 5: Jurisdictional proportions
    jurisdiction_features = calculate_jurisdictional_proportions(df_feat, total_fuel)
    df_feat = pd.concat([df_feat, jurisdiction_features], axis=1)

    # Module 6: Temporal rolling averages
    df_feat = extract_temporal_trends(df_feat)

    # Module 7: Deviation ratios — must follow Module 6
    df_feat = calculate_temporal_ratios(df_feat)

    return df_feat