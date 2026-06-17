"""
Script that runs inside a SageMaker Processing Job container.
Reads clean pool parquet from /opt/ml/processing/input,
runs Isolation Forest, writes results to /opt/ml/processing/output.

SageMaker automatically syncs these paths to/from S3.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
# from analysis import run_unsupervised_anomaly_detection, \
#                     contamination_sensitivity_check, \
#                     calculate_global_importance

INPUT_DIR  = Path('/opt/ml/processing/input/features')
OUTPUT_DIR = Path('/opt/ml/processing/output')


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
    df_importance = calculate_global_importance(df_statistical, features)
    
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

    results_df = pd.DataFrame(results)
    return results_df


def main(contamination):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Try .parquet extension first
    parquet_files = list(INPUT_DIR.rglob('*.parquet'))

    # Athena CTAS sometimes writes parquet files without extension
    if not parquet_files:
        all_files = [f for f in INPUT_DIR.rglob('*') if f.is_file()]
        print(f"No .parquet files found. Trying all files: {all_files}")
        parquet_files = all_files

    if not parquet_files:
        raise FileNotFoundError(f"No files at all found under {INPUT_DIR}")

    print(f"Loading {len(parquet_files)} file(s)...")
    frames = []
    for f in parquet_files:
        try:
            frames.append(pd.read_parquet(f))
            print(f"   ✅ Read: {f.name}")
        except Exception as e:
            print(f"   ⚠️ Could not read {f.name}: {e}")

    if not frames:
        raise ValueError("All files failed to load as parquet")

    df_pool = pd.concat(frames, ignore_index=True)
    print(f"Clean pool: {df_pool.shape[0]} rows × {df_pool.shape[1]} cols")

    # Sensitivity check
    sensitivity_df = contamination_sensitivity_check(df_pool)
    sensitivity_df.to_csv(
        OUTPUT_DIR / 'contamination_sensitivity.csv', index=False)

    # Stage 2: Isolation Forest
    df_clean, df_statistical, df_importance = run_unsupervised_anomaly_detection(
        df_pool, contamination_rate=contamination
    )

    # Save outputs — SageMaker uploads these to S3 automatically
    df_statistical.to_csv(OUTPUT_DIR / 'stage2_flagged.csv',      index=False)
    df_clean.to_csv(      OUTPUT_DIR / 'stage2_clean.csv',        index=False)
    df_importance.to_csv( OUTPUT_DIR / 'feature_importance.csv',  index=False)

    print(f"Stage 2 complete. Flagged: {len(df_statistical)} | Clean: {len(df_clean)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contamination', type=float, default=0.03)
    args = parser.parse_args()
    main(args.contamination)