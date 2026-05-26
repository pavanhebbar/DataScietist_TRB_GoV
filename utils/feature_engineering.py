import pandas as pd
import numpy as np

# =====================================================================
# 1. INDEPENDENT MODULES (Helper Functions)
# =====================================================================

def calculate_efficiency_and_costs(df):
    """Computes overall consumption rates and financial metrics per kilometer."""
    fuel_cols = ['ab_fuel', 'bc_fuel', 'mb_fuel', 'on_fuel']
    
    # Sum across provincial fuel columns per row, keeping completely NaN rows as NaN
    total_fuel = df[fuel_cols].sum(axis=1, min_count=1)
    
    # Calculate Litres per KM with 0.0 safety guardrails for missing data or 0 km trips
    fuel_litres_per_km = np.where(
        total_fuel.isna() | (df['total_km'] == 0),
        0.0,
        total_fuel / df['total_km']
    )
    
    # Calculate Cost per KM with safety guardrails
    fuel_cost_per_km = np.where(
        df['total_cost'].isna() | (df['total_km'] == 0),
        0.0,
        df['total_cost'] / df['total_km']
    )
    
    return total_fuel, fuel_litres_per_km, fuel_cost_per_km


def flag_compliance_gaps(df):
    """Flags binary indicators targeting unvouched trips for audit analysis."""
    return np.where(df['matching_invoice_number'] == 'NO MATCHING INVOICE', 1, 0)


def calculate_jurisdictional_proportions(df, total_fuel_series):
    """Computes variances between where miles were driven vs where fuel was bought."""
    proportions_dict = {}
    
    for prov in ['ab', 'bc', 'mb', 'on']:
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


# =====================================================================
# 2. MASTER ORCHESTRATOR PIPELINE
# =====================================================================

def engineer_ifta_features(df_ml_ready):
    """
    Orchestrator function that maps clean source data to individual 
    feature engineering modules and outputs a unified dataset for ML.
    """
    # Create working copy
    df_feat = df_ml_ready.copy()
    
    # Module 1: Core Consumption Metrics
    total_fuel, litres_per_km, cost_per_km = calculate_efficiency_and_costs(df_feat)
    df_feat['total_fuel_litres'] = total_fuel
    df_feat['fuel_litres_per_km'] = litres_per_km
    df_feat['fuel_cost_per_km'] = cost_per_km
    
    # Module 2: Audit Compliance Flagging
    df_feat['is_unvouched_trip'] = flag_compliance_gaps(df_feat)
    
    # Module 3: Cross-Border Variance Metrics
    jurisdiction_features = calculate_jurisdictional_proportions(df_feat, total_fuel)
    df_feat = pd.concat([df_feat, jurisdiction_features], axis=1)
    
    # Module 4: Temporal Behavior Components
    df_feat = extract_temporal_trends(df_feat)
    
    return df_feat