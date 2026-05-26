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
    
    # 1. Total Distance and Total Fuel absolute bounds (> 3 Sigma)
    mean_km, std_km = df['total_km'].mean(), df['total_km'].std()
    mean_fuel, std_fuel = df['total_fuel_litres'].mean(), df['total_fuel_litres'].std()
    
    rule_dist_z = df['total_km'] > (mean_km + 3 * std_km)
    rule_fuel_z = df['total_fuel_litres'] > (mean_fuel + 3 * std_fuel)
    
    # 2. Fuel Intensity bounds (> 3 Sigma)
    rule_intensity_z = df['fuel_litres_per_km'] > (df['fuel_litres_per_km'].mean() + 3 * df['fuel_litres_per_km'].std())
    
    # 3. IFTA Boundary Compliance Violation (Fuel in Alberta, but 0 Distance)
    rule_ifta_breach = (df['ab_fuel_prop'] == 1.0) & (df['ab_dist_prop'] == 0.0)
    
    # 4. Weekly and Monthly Ratio Deviations (> 3 Sigma)
    mean_rw_dist, std_rw_dist = df['dist_to_weekly_ratio'].mean(), df['dist_to_weekly_ratio'].std()
    mean_rw_fuel, std_rw_fuel = df['fuel_to_weekly_ratio'].mean(), df['fuel_to_weekly_ratio'].std()
    mean_rm_dist, std_rm_dist = df['dist_to_monthly_ratio'].mean(), df['dist_to_monthly_ratio'].std()
    mean_rm_fuel, std_rm_fuel = df['fuel_to_monthly_ratio'].mean(), df['fuel_to_monthly_ratio'].std()
    
    rule_weekly_ratio_outlier = (df['dist_to_weekly_ratio'] > (mean_rw_dist + 3 * std_rw_dist)) | \
                                (df['fuel_to_weekly_ratio'] > (mean_rw_fuel + 3 * std_rw_fuel))
                                
    rule_monthly_ratio_outlier = (df['dist_to_monthly_ratio'] > (mean_rm_dist + 3 * std_rm_dist)) | \
                                 (df['fuel_to_monthly_ratio'] > (mean_rm_fuel + 3 * std_rm_fuel))
                                 
    # 5. Moving Average Macro-Drift (> 3 Sigma)
    mean_w_avg_km, std_w_avg_km = df['weekly_avg_distance_km'].mean(), df['weekly_avg_distance_km'].std()
    mean_w_avg_L, std_w_avg_L = df['weekly_avg_fuel_litres'].mean(), df['weekly_avg_fuel_litres'].std()
    mean_m_avg_km, std_m_avg_km = df['monthly_avg_distance_km'].mean(), df['monthly_avg_distance_km'].std()
    mean_m_avg_L, std_m_avg_L = df['monthly_avg_fuel_litres'].mean(), df['monthly_avg_fuel_litres'].std()
    
    rule_macro_drift = (df['weekly_avg_distance_km'] > (mean_w_avg_km + 3 * std_w_avg_km)) | \
                        (df['weekly_avg_fuel_litres'] > (mean_w_avg_L + 3 * std_w_avg_L)) | \
                        (df['monthly_avg_distance_km'] > (mean_m_avg_km + 3 * std_m_avg_km)) | \
                        (df['monthly_avg_fuel_litres'] > (mean_m_avg_L + 3 * std_m_avg_L))

    # Master logical OR to catch all deterministic violations
    deterministic_mask = (
        rule_dist_z | rule_fuel_z | rule_intensity_z | 
        rule_ifta_breach | rule_weekly_ratio_outlier | 
        rule_monthly_ratio_outlier | rule_macro_drift
    )
    
    # Split the dataset
    df_breaches = df[deterministic_mask].copy()
    df_clean_pool = df[~deterministic_mask].copy()
    
    df_breaches['anomaly_source'] = 'Deterministic Z-Score Rule'
    df_breaches['anomaly_score'] = -1.0
    df_breaches['Primary Driver'] = 'Rule-Based Violation'
    df_breaches['Secondary Driver'] = 'None'
    df_breaches['Multi-Feature Interaction Spike'] = 'N/A'
    df_breaches['Route Context'] = df_breaches.apply(lambda r: f"{r.get('trip_origin', 'UNK')} ➔ {r.get('trip_destination', 'UNK')}", axis=1)
    
    print(f"   👉 Isolated {len(df_breaches)} records violating the > 3σ or IFTA constraints.")
    print(f"   👉 Kept {len(df_clean_pool)} records for unsupervised Isolation Forest modeling.")
    
    return df_clean_pool, df_breaches

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

def calculate_global_importance(df_statistical, features):
    """
    Computes global feature importance natively by taking the mean absolute 
    ablation impact across all flagged statistical anomalies.
    """
    print("\n📊 Aggregating Global Feature Importances from anomaly drivers...")
    
    # If no statistical anomalies were found, we can't aggregate drivers
    if df_statistical.empty:
        print("   ⚠️ No statistical anomalies to evaluate for global importance.")
        return pd.DataFrame()

    # We can extract this directly if you run a quick loop or track the raw deltas,
    # but for a fast, clean presentation slide, we can look at the distribution 
    # of the Primary and Secondary drivers flagged in the ledger:
    driver_counts = pd.concat([df_statistical['Primary Driver'], df_statistical['Secondary Driver']]).value_counts()
    driver_df = pd.DataFrame({'Feature': driver_counts.index, 'Selection Frequency': driver_counts.values})
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
        'total_km', 'total_fuel_litres', 'fuel_litres_per_km',
        'weekly_avg_distance_km', 'weekly_avg_fuel_litres',
        'dist_to_weekly_ratio', 'fuel_to_weekly_ratio',
        'dist_to_monthly_ratio', 'fuel_to_monthly_ratio'
    ]
    
    provinces = ['ab', 'bc', 'mb', 'on']
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
    df_importance = calculate_global_importance(df_statistical, features)
    
    return df_validated_clean, df_statistical, df_importance

def execute_full_audit_pipeline(df_raw, contamination=0.03):
    """Complete Pipeline Orchestration Block."""
    df_features = df_raw.copy()
    
    df_features['dist_to_weekly_ratio'] = df_features.apply(lambda r: r['total_km'] / r['weekly_avg_distance_km'] if r['weekly_avg_distance_km'] > 0 else 1.0, axis=1)
    df_features['fuel_to_weekly_ratio'] = df_features.apply(lambda r: r['total_fuel_litres'] / r['weekly_avg_fuel_litres'] if r['weekly_avg_fuel_litres'] > 0 else 1.0, axis=1)
    df_features['dist_to_monthly_ratio'] = df_features.apply(lambda r: r['total_km'] / r['monthly_avg_distance_km'] if r['monthly_avg_distance_km'] > 0 else 1.0, axis=1)
    df_features['fuel_to_monthly_ratio'] = df_features.apply(lambda r: r['total_fuel_litres'] / r['monthly_avg_fuel_litres'] if r['monthly_avg_fuel_litres'] > 0 else 1.0, axis=1)

    df_pool, df_deterministic = extract_deterministic_outliers(df_features)
    (df_clean, df_statistical,
     df_importance) = run_unsupervised_anomaly_detection(
         df_pool, contamination_rate=contamination)
    
    df_master_audit = pd.concat([df_deterministic, df_statistical], axis=0, sort=False)
    
    print(f"\n🎯 Pipeline Complete. Master Audit Ledger rows generated: {len(df_master_audit)}")
    return df_clean, df_master_audit, df_importance