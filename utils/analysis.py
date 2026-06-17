"""Python program from analysis and anomaly detection."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

def extract_deterministic_outliers(df):
    """
    STAGE 1: Mathematically Rigorous Rules-Based Filtering
    Identifies and removes explicit > 3 sigma breaches and compliance violations 
    calculated dynamically from the population statistics.
    """
    print("🧹 Stage 1: Dynamically calculating population Z-scores for deterministic filtering...")

    # 1. Total Distance and Total Fuel,  absolute bounds (> 3 Sigma)
    mean_km, std_km = df['total_km'].mean(), df['total_km'].std()
    mean_fuel, std_fuel = (
        df['total_fuel_litres'].mean(), df['total_fuel_litres'].std())

    rule_dist_z = df['total_km'] > (mean_km + 3 * std_km)
    rule_fuel_z = df['total_fuel_litres'] > (mean_fuel + 3 * std_fuel)

    # Per tank distance and fuel absolute bounds ( > 3 sigma)
    mean_avg_fuel, std_avg_fuel = (
        df['avg_fuel_per_tank'].mean(), df['avg_fuel_per_tank'].std())
    mean_dist_tank, std_dist_tank = (
        df['km_per_tank'].mean(), df['km_per_tank'].std())

    rule_avgfuel_z = (
        (df['avg_fuel_per_tank'] > (mean_avg_fuel + 3*std_avg_fuel)) |
        (df['avg_fuel_per_tank'] < 100))
    rule_avgdist_z = df['km_per_tank'] > (mean_dist_tank + 3*std_dist_tank)


    # 2. Fuel Intensity bounds (> 3 Sigma) and < 0.05 L/km
    rule_intensity_z1 = (
        df['fuel_litres_per_km'] > (df['fuel_litres_per_km'].mean() +
                                    3 * df['fuel_litres_per_km'].std()))
    rule_intensity_z2 = (
        df['cycle_fuel_per_km'] > (df['cycle_fuel_per_km'].mean() +
                                   3 * df['cycle_fuel_per_km'].std()))
    rule_intensity_z3  = (df['cycle_fuel_per_km'] < 0.05)
    rule_intensity_z = (rule_intensity_z1 | rule_intensity_z2 |
                        rule_intensity_z3)

    # Fuel cost
    rule_cost_high = (df['fuel_cost_per_km'] > 1)

    # Remove clear flags:
    rule_dist_mismatch = (df['km_reconciliation_gap'] >= 10)
    rule_missing_fuel = (df['fuel_source_reliability'] == 0)
    rule_odometer_gap = (df['odometer_gap'] == 1)

    # 3. IFTA Boundary Compliance Violation (Fuel in Alberta, but 0 Distance)
    rule_ifta_breach = (df['ab_fuel_prop'] == 1.0) & (df['ab_dist_prop'] == 0.0)

    # 4. Weekly and Monthly Ratio Deviations (> 3 Sigma)
    mean_rw_dist, std_rw_dist = df['dist_to_weekly_ratio'].mean(), df['dist_to_weekly_ratio'].std()
    mean_rw_fuel, std_rw_fuel = df['fuel_to_weekly_ratio'].mean(), df['fuel_to_weekly_ratio'].std()
    mean_rm_dist, std_rm_dist = df['dist_to_monthly_ratio'].mean(), df['dist_to_monthly_ratio'].std()
    mean_rm_fuel, std_rm_fuel = df['fuel_to_monthly_ratio'].mean(), df['fuel_to_monthly_ratio'].std()

    rule_weekly_ratio_outlier = ((
        df['dist_to_weekly_ratio'] > (mean_rw_dist + 3 * std_rw_dist)) |
        (df['fuel_to_weekly_ratio'] > (mean_rw_fuel + 3 * std_rw_fuel)))

    rule_monthly_ratio_outlier = (
        (df['dist_to_monthly_ratio'] > (mean_rm_dist + 3 * std_rm_dist)) |
        (df['fuel_to_monthly_ratio'] > (mean_rm_fuel + 3 * std_rm_fuel)))

    # 5. Moving Average Macro-Drift (> 3 Sigma)
    mean_w_avg_km, std_w_avg_km = (
        df['weekly_avg_distance_km'].median(),
        df['weekly_avg_distance_km'].std())
    mean_w_avg_L, std_w_avg_L = (
        df['weekly_avg_fuel_litres'].median(),
        df['weekly_avg_fuel_litres'].std())
    mean_m_avg_km, std_m_avg_km = (
        df['monthly_avg_distance_km'].median(),
        df['monthly_avg_distance_km'].std())
    mean_m_avg_L, std_m_avg_L = (
        df['monthly_avg_fuel_litres'].median(),
        df['monthly_avg_fuel_litres'].std())
    
    rule_macro_drift_week = (
        (df['weekly_avg_distance_km'] > (mean_w_avg_km + 3 * std_w_avg_km)) |
        (df['weekly_avg_fuel_litres'] > (mean_w_avg_L + 3 * std_w_avg_L)) |
        (
            (df['weekly_avg_distance_km'] > 0) &
            (
                (df['weekly_avg_distance_km'] < (mean_w_avg_km -
                                                 3 * std_w_avg_km)) |
                (df['weekly_avg_fuel_litres'] < (mean_w_avg_L -
                                                 3 * std_w_avg_L))
            )
            )
    )

    rule_macro_drift_month = (
        (df['monthly_avg_distance_km'] > (mean_m_avg_km + 3 * std_m_avg_km)) |
        (df['monthly_avg_fuel_litres'] > (mean_m_avg_L + 3 * std_m_avg_L)) |
        (
            (df['monthly_avg_distance_km'] > 0) &
            (
                (df['monthly_avg_distance_km'] < (mean_m_avg_km -
                                                  3 * std_m_avg_km)) |
                (df['monthly_avg_fuel_litres'] < (mean_m_avg_L -
                                                  3 * std_m_avg_L))
            )
        )
    )

    # Master logical OR to catch all deterministic violations
    deterministic_mask = (
        rule_dist_z | rule_fuel_z | rule_intensity_z | rule_cost_high |
        rule_dist_mismatch | rule_missing_fuel | rule_odometer_gap |
        rule_avgdist_z | rule_avgfuel_z | rule_ifta_breach |
        rule_weekly_ratio_outlier | rule_monthly_ratio_outlier |
        rule_macro_drift_week | rule_macro_drift_month
    )
    
    # Split the dataset
    df_breaches = df[deterministic_mask].copy()
    df_clean_pool = df[~deterministic_mask].copy()
    
    df_breaches['anomaly_source'] = 'Deterministic'
    df_breaches['anomaly_score'] = -1.0
    # ── Primary Driver: first matching rule wins (priority order) ────────
    # Reindex each rule mask to df_breaches so np.select works correctly
    _b = df_breaches.index
    primary_conditions = [
        rule_missing_fuel.reindex(_b, fill_value=False),
        rule_odometer_gap.reindex(_b, fill_value=False),
        rule_dist_mismatch.reindex(_b, fill_value=False),
        (rule_dist_z | rule_fuel_z |
         rule_avgfuel_z | rule_avgdist_z).reindex(_b, fill_value=False),
        (rule_intensity_z | rule_ifta_breach).reindex(_b, fill_value=False),
        rule_cost_high.reindex(_b, fill_value=False),
        (rule_weekly_ratio_outlier | rule_monthly_ratio_outlier |
         rule_macro_drift_week | rule_macro_drift_month).reindex(
             _b, fill_value=False),
    ]
    primary_labels = [
        'Missing fuel',
        'Odometer gap',
        'Distance mismatch',
        'Fuel quantity or km',
        'Fuel efficiency mismatch',
        'Fuel cost eff mismatch',
        'Temporal trend mismatch',
    ]
    df_breaches['Primary Driver'] = np.select(
        primary_conditions, primary_labels, default='Rule-Based Violation'
    )

    # ── Secondary Driver: next co-occurring rule, excluding the primary ──
    secondary_conditions = [
        (df_breaches['Primary Driver'] != 'Distance mismatch') &
        rule_dist_mismatch.reindex(_b, fill_value=False),

        (df_breaches['Primary Driver'] != 'Temporal trend mismatch') &
        (rule_weekly_ratio_outlier | rule_monthly_ratio_outlier |
         rule_macro_drift_week | rule_macro_drift_month).reindex(
             _b, fill_value=False),

        (df_breaches['Primary Driver'] != 'Fuel efficiency mismatch') &
        (rule_intensity_z | rule_ifta_breach).reindex(_b, fill_value=False),

        (df_breaches['Primary Driver'] != 'Fuel quantity or km') &
        (rule_dist_z | rule_fuel_z |
         rule_avgfuel_z | rule_avgdist_z).reindex(_b, fill_value=False),

        (df_breaches['Primary Driver'] != 'Fuel cost eff mismatch') &
        rule_cost_high.reindex(_b, fill_value=False),
    ]
    secondary_labels = [
        'Distance mismatch',
        'Temporal trend mismatch',
        'Fuel efficiency mismatch',
        'Fuel quantity or km',
        'Fuel cost eff mismatch'
    ]
    df_breaches['Secondary Driver'] = np.select(
        secondary_conditions, secondary_labels, default='None'
    )

    df_breaches['Multi-Feature Interaction Spike'] = 'N/A'
    df_breaches['Route Context'] = df_breaches.apply(lambda r: f"{r.get('trip_origin', 'UNK')} ➔ {r.get('trip_destination', 'UNK')}", axis=1)

    print(f"   👉 Isolated {len(df_breaches)} records violating the > 3σ or IFTA constraints.")
    print(f"   👉 Kept {len(df_clean_pool)} records for unsupervised Isolation Forest modeling.")

    _print_driver_summary(df_breaches)
    return df_clean_pool, df_breaches


def _print_driver_summary(df_breaches):
    """
    Prints a formatted terminal table showing how many flagged records
    each rule category contributes to as Primary or Secondary Driver.
    Called automatically at the end of extract_deterministic_outliers.
    """
    all_drivers = [
        'Missing fuel',
        'Odometer gap',
        'Distance mismatch',
        'Fuel quantity or km',
        'Fuel efficiency mismatch',
        'Temporal trend mismatch',
        'Fuel cost eff mismatch',
        'Rule-Based Violation',   # fallback — should ideally be 0
    ]

    primary_counts = df_breaches['Primary Driver'].value_counts()

    # Exclude 'None' from secondary — it means no co-occurring rule fired
    secondary_counts = df_breaches[
        df_breaches['Secondary Driver'] != 'None'
    ]['Secondary Driver'].value_counts()

    # Build rows — only include drivers that appear at least once
    rows = []
    for driver in all_drivers:
        p = primary_counts.get(driver, 0)
        s = secondary_counts.get(driver, 0)
        if p > 0 or s > 0:
            rows.append((driver, p, s, p + s))

    # Sort by total contribution descending
    rows.sort(key=lambda x: x[3], reverse=True)

    w = 30   # driver name column width
    print(f"\n   {'Deterministic Audit Ledger — Driver Contribution Summary'}")
    print(f"   {'═' * 62}")
    print(f"{'Driver Category':<{w}}  {'Primary':>8}  {'Secondary':>10}  {'Total':>7}")
    print(f"   {'─' * 62}")
    for driver, p, s, total in rows:
        print(f"   {driver:<{w}}  {p:>8}  {s:>10}  {total:>7}")
    print(f"   {'─' * 62}")
    n_with_secondary = (df_breaches['Secondary Driver'] != 'None').sum()
    print(f"   {'Total flagged records':<{w}}  "
          f"{len(df_breaches):>8}  {n_with_secondary:>10}  {'—':>7}")
    print(f"   {'═' * 62}\n")


def extract_robust_anomaly_reasons(iso_forest, df_statistical, X_scaled, features):
    """
    Explains local anomalies by measuring how much neutralizing a specific 
    feature to its mean baseline improves the model's anomaly path-length score.
    """
    base_scores = iso_forest.score_samples(X_scaled)
    statistical_indices = df_statistical['pool_matrix_idx'].values
    
    anomaly_diagnostics = []
    
    for i, matrix_idx in enumerate(statistical_indices):
        row_data = X_scaled[matrix_idx].copy()
        feature_contributions = {}
        
        # Systematically ablate each feature to zero out its anomaly footprint
        for feat_idx, feat_name in enumerate(features):
            modified_row = row_data.copy()
            modified_row[feat_idx] = 0.0  # Population mean under StandardScaler
            
            new_score = iso_forest.score_samples([modified_row])[0]
            
            # Impact is the scale of the drop into the negative zone caused by this feature
            score_delta = new_score - base_scores[matrix_idx]
            feature_contributions[feat_name] = score_delta
            
        # Sort by impact
        sorted_drivers = sorted(feature_contributions.items(), key=lambda item: item[1], reverse=True)
        
        top_1_feat, top_1_impact = sorted_drivers[0]
        top_2_feat, top_2_impact = sorted_drivers[1] if len(sorted_drivers) > 1 else ("None", 0.0)
        
        anomaly_diagnostics.append({
            'Primary Driver': top_1_feat,
            'Secondary Driver': top_2_feat,
            'Multi-Feature Interaction Spike': 'High' if abs(top_1_impact - top_2_impact) < 0.15 else 'Single-Feature Dominated'
        })
        
    return pd.DataFrame(anomaly_diagnostics, index=df_statistical.index)

def calculate_global_importance(df_statistical):
    """
    Computes global feature importance natively by taking the mean absolute 
    ablation impact across all flagged statistical anomalies.
    """
    print("\n📊 Aggregating Global Feature Importances from anomaly drivers...")
    
    # If no statistical anomalies were found, we can't aggregate drivers
    if df_statistical.empty:
        print("   ⚠️ No statistical anomalies to evaluate for global importance.")
        return pd.DataFrame()

    driver_counts = pd.concat([df_statistical['Primary Driver'],
                               df_statistical['Secondary Driver']]).value_counts()
    driver_df = pd.DataFrame({'Feature': driver_counts.index,
                              'Selection Frequency': driver_counts.values})
    driver_df = driver_df[driver_df['Feature'] != 'None'].sort_values(by='Selection Frequency', ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=driver_df.head(10), x='Selection Frequency', y='Feature', palette='viridis')
    plt.title('Global Feature Importance: Anomaly Isolation Impact Frequency', fontweight='bold', fontsize=12)
    plt.xlabel('Number of Trips Driven by Feature')
    plt.ylabel('Feature Space')
    plt.tight_layout()
    plt.show()
    
    return driver_df


def run_unsupervised_anomaly_detection(df_pool, contamination_rate=0.03):
    """
    STAGE 2: Isolation Forest Anomaly Detection with integrated Local Feature Ablation.
    """
    print("\n🌲 Stage 2: Fitting Isolation Forest on sanitized operational data...")
    
    base_features = [
        'total_km',
        'total_fuel_litres',
        'fuel_litres_per_km',        # per-trip efficiency
        'cycle_fuel_per_km',         # fuelling-cycle efficiency (tank range)
        'fuel_cost_per_km',
        'km_per_tank',               # total cycle distance
        'avg_fuel_per_tank',         # anchor fuel quantity for the cycle
        'provincial_sum_km',         # jurisdictional km sum
        'km_reconciliation_gap',     # unaccounted distance
        'fuel_source_reliability',   # 0=missing → 3=invoice
        'odometer_gap',              # undocumented km between trips
        'weekly_avg_distance_km',    'weekly_avg_fuel_litres',
        'dist_to_weekly_ratio',      'fuel_to_weekly_ratio',
        'dist_to_monthly_ratio',     'fuel_to_monthly_ratio',
    ]

    provinces = ['ab', 'bc', 'mb', 'on', 'sk']
    jurisdiction_features = []
    for prov in provinces:
        jurisdiction_features.extend([f'{prov}_dist_prop', f'{prov}_fuel_prop', f'{prov}_dist_fuel_variance'])
        
    features = base_features + jurisdiction_features
    
    # Clean and fill missing regional properties before scaling
    df_modeling = df_pool.copy()
    df_modeling[jurisdiction_features] = df_modeling[jurisdiction_features].fillna(0.0)
    df_modeling = df_modeling.dropna(subset=base_features).copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_modeling[features])
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=200, contamination=contamination_rate, random_state=42, n_jobs=-1
    )
    
    df_modeling['forest_prediction'] = iso_forest.fit_predict(X_scaled)
    df_modeling['anomaly_score'] = iso_forest.score_samples(X_scaled)
    df_modeling['anomaly_source'] = 'Isolation Forest'
    df_modeling['Route Context'] = df_modeling.apply(lambda r: f"{r.get('trip_origin', 'UNK')} ➔ {r.get('trip_destination', 'UNK')}", axis=1)
    
    # Add matrix array positions so the ablation loops look at the correct row records
    df_modeling['pool_matrix_idx'] = np.arange(len(df_modeling))
    
    # Split outcomes
    df_statistical = df_modeling[df_modeling['forest_prediction'] == -1].copy()
    df_validated_clean = df_modeling[df_modeling['forest_prediction'] == 1].copy()
    
    # Local Feature Explanations applied straight to the statistical outliers
    if not df_statistical.empty:
        print("🔎 Quantifying local feature ablation contributions for outliers...")
        df_explanations = extract_robust_anomaly_reasons(iso_forest, df_statistical, X_scaled, features)
        df_statistical = df_statistical.join(df_explanations)
    else:
        df_statistical['Primary Driver'] = pd.Series(dtype='str')
        df_statistical['Secondary Driver'] = pd.Series(dtype='str')
        df_statistical['Multi-Feature Interaction Spike'] = pd.Series(dtype='str')

    # Drop processing helper column
    df_statistical = df_statistical.drop(columns=['pool_matrix_idx'])
    df_validated_clean = df_validated_clean.drop(columns=['pool_matrix_idx'])
    
    print(f"   👉 Flagged {len(df_statistical)} latent structural anomalies.")
    print(f"   👉 Verified {len(df_validated_clean)} clean operational runs.")
    
    # Call corrected global plot using the calculated drivers ledger directly
    df_importance = calculate_global_importance(df_statistical)
    
    return df_validated_clean, df_statistical, df_importance


def contamination_sensitivity_check(df_pool, rates=None):
    """
    Runs Isolation Forest at multiple contamination rates to justify
    the chosen threshold. Prints flagged record counts and plots the curve.
    
    IFTA context: helps determine what % of trips warrant audit review
    without overwhelming investigators with false positives.
    """
    if rates is None:
        rates = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

    base_features = [
        'total_km',
        'total_fuel_litres',
        'fuel_litres_per_km',        # per-trip efficiency
        'fuel_cost_per_km',
        'cycle_fuel_per_km',         # fuelling-cycle efficiency (tank range)
        'km_per_tank',               # total cycle distance
        'avg_fuel_per_tank',         # anchor fuel quantity for the cycle
        'provincial_sum_km',         # jurisdictional km sum
        'km_reconciliation_gap',     # unaccounted distance
        'fuel_source_reliability',   # 0=missing → 3=invoice
        'odometer_gap',              # undocumented km between trips
        'weekly_avg_distance_km',    'weekly_avg_fuel_litres',
        'dist_to_weekly_ratio',      'fuel_to_weekly_ratio',
        'dist_to_monthly_ratio',     'fuel_to_monthly_ratio',
    ]

    df_modeling = df_pool.dropna(subset=base_features).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_modeling[base_features])

    results = []
    print("\n📊 Contamination Rate Sensitivity Check:")
    print(f"   {'Rate':>8} | {'Flagged':>8} | {'% of Pool':>10}")
    print("   " + "-" * 32)

    for rate in rates:
        iso = IsolationForest(
            n_estimators=200, contamination=rate,
            random_state=42, n_jobs=-1
        )
        preds = iso.fit_predict(X_scaled)
        n_flagged = (preds == -1).sum()
        pct = n_flagged / len(df_modeling) * 100
        results.append({'rate': rate, 'flagged': n_flagged, 'pct': pct})
        print(f"   {rate:>8.0%} | {n_flagged:>8} | {pct:>9.1f}%")

    # Plot the curve
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 4))
    plt.plot(results_df['rate'] * 100, results_df['flagged'],
             marker='o', color='steelblue', linewidth=2)
    plt.axvline(x=3, color='red', linestyle='--',
                label='Selected threshold (3%)')
    plt.title('Isolation Forest: Flagged Records vs Contamination Rate',
              fontweight='bold')
    plt.xlabel('Contamination Rate (%)')
    plt.ylabel('Flagged Records')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results_df


def execute_full_audit_pipeline(df_raw, contamination=0.03):
    """Complete Pipeline Orchestration Block."""
    df_features = df_raw.copy()

    df_pool, df_deterministic = extract_deterministic_outliers(df_features)
    (df_clean, df_statistical,
     df_importance) = run_unsupervised_anomaly_detection(
         df_pool, contamination_rate=contamination)
    
    df_master_audit = pd.concat([df_deterministic, df_statistical], axis=0, sort=False)
    
    print(f"\n🎯 Pipeline Complete. Master Audit Ledger rows generated: {len(df_master_audit)}")
    return df_clean, df_master_audit, df_importance