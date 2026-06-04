"""Python program to visualize IFTA logs."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def set_plot_style():
    """Configures high-quality professional visual standards for slides."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    sns.set_palette("muted")


def plot_total_km_vs_fuel(df, save_path='eda_1_total_km_vs_fuel.png'):
    """
    1. Scatter Plot: Total distance driven vs Total fuel consumed.
    Labels the extreme physical outlier (> 3000 km) using routing context.
    """
    plt.figure(figsize=(11, 7))

    palette = {
        'invoice': '#27ae60',
        'provincial_log': '#2980b9',
        'provincial_log_nearby': '#f39c12',
        'missing': '#e74c3c'
    }

    ax = sns.scatterplot(
        data=df, x='total_km', y='total_fuel_litres',
        hue='fuel_source', palette=palette,
        s=150, edgecolor='black', alpha=0.85
    )
    
    mean_x, std_x = df['total_km'].mean(), df['total_km'].std()
    mean_y, std_y = df['total_fuel_litres'].mean(), df['total_fuel_litres'].std()
    
    for i in [1, 2, 3]:
        color = '#e67e22' if i == 1 else ('#d35400' if i == 2 else '#c0392b')
        ls = '--' if i == 1 else (':' if i == 2 else '-.')
        plt.axvline(mean_x + i * std_x, color=color, linestyle=ls, alpha=0.3)
        if mean_x - i * std_x >= 0: plt.axvline(mean_x - i * std_x, color=color, linestyle=ls, alpha=0.3)
        plt.axhline(mean_y + i * std_y, color=color, linestyle=ls, alpha=0.3)
        if mean_y - i * std_y >= 0: plt.axhline(mean_y - i * std_y, color=color, linestyle=ls, alpha=0.3)
            
    outlier = df[df['total_km'] > 3000]
    if not outlier.empty:
        for idx, row in outlier.iterrows():
            plt.annotate(
                f"High Risk Outlier\nRoute: {row['trip_origin']} ➔ {row['trip_destination']}\nDistance: {row['total_km']:.0f} km",
                xy=(row['total_km'], row['total_fuel_litres']),
                xytext=(row['total_km'] - 850, row['total_fuel_litres'] + 150),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fce4d6", ec="#c0392b", lw=1.5)
            )
    missing_fuel = df[df['fuel_source'] == 'missing']
    if not missing_fuel.empty:
        for idx, row in missing_fuel.iterrows():
            plt.annotate(
                f"Missing fuel\nRoute: {row['trip_origin']} ➔ {row['trip_destination']}\nDistance: {row['total_km']:.0f} km",
                xy=(row['total_km'], row['total_fuel_litres']),
                xytext=(row['total_km'] - 850, row['total_fuel_litres'] + 150),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fce4d6", ec="#c0392b", lw=1.5))
            
    plt.title('Fleet Physical Baseline: Total Fuel vs. Distance Logged', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Total Distance (km)')
    plt.ylabel('Total Fuel (L)')
    plt.legend(title='Is Unvouched Trip?', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_efficiency_distributions_hybrid(
        df, save_path='eda_2_efficiency_distributions.png'):
    """
    2. Hybrid Distribution Check:
    - Fuel Intensity (L/km): Includes ALL records, plus highlights and labels the severe > 2.5 L/km outlier.
    - Cost Efficiency ($/km): Includes VOUCHED records ONLY to prevent artificial $0 skew.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    datasets = {
        'fuel_litres_per_km': df.copy(),
        # Replace is_unvouched_trip==0 with invoice-only
        'fuel_cost_per_km': df[df['fuel_source'] == 'invoice']
    }
    metrics = ['fuel_litres_per_km', 'fuel_cost_per_km']
    colors = ['#27ae60', '#2980b9']
    titles = ['Fleet Distribution: Fuel Intensity (All Trips)', 'Vouched Distribution: Cost Efficiency (Invoiced Only)']
    x_labels = ['Fuel Intensity (L/km)', 'Cost Efficiency ($/km)']
    
    for idx, col in enumerate(metrics):
        target_df = datasets[col]
        data_series = target_df[col].dropna()
        
        # Plot histogram and extract bin/count data to place annotation comfortably
        n, bins, patches = axes[idx].hist(data_series, bins=12, color=colors[idx], edgecolor='black', alpha=0.7)
        sns.kdeplot(data_series, ax=axes[idx], color=colors[idx], linewidth=2, linestyle='-')
        
        mean_val = data_series.mean()
        std_val = data_series.std()
        
        axes[idx].axvline(mean_val, color='black', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
        for i in [1, 2, 3]:
            color = '#e67e22' if i == 1 else ('#d35400' if i == 2 else '#c0392b')
            ls = '--' if i == 1 else (':' if i == 2 else '-.')
            if (mean_val - i * std_val) >= 0:
                axes[idx].axvline(
                    mean_val - i * std_val, color=color, linestyle=ls,
                    alpha=0.6)
            axes[idx].axvline(mean_val + i * std_val, color=color, linestyle=ls, alpha=0.6)
            
        # Target Annotation for the massive fuel intensity spike (>2.5 L/km) on Subplot 0
        if col == 'fuel_litres_per_km':
            intensity_outliers = target_df[
                target_df['fuel_litres_per_km'] > 2.5]
            if not intensity_outliers.empty:
                for enum_idx, (o_idx, o_row) in enumerate(
                            intensity_outliers.iterrows()):
                    y_text_position = max(n) * (0.55 - (enum_idx * 0.18))
                    axes[idx].annotate(
                        f"Critical Intensity Spike\nRoute: {o_row['trip_origin']} ➔ {o_row['trip_destination']}\nValue: {o_row['fuel_litres_per_km']:.2f} L/km",
                        xy=(o_row['fuel_litres_per_km'], 1),  # Pointing near the bottom of the outlier bin
                        xytext=(o_row['fuel_litres_per_km'] - 1,
                                y_text_position),  # Shifted left
                        arrowprops=dict(facecolor='black', shrink=0.1, width=1, headwidth=6),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fce4d6", ec="#c0392b", lw=1.5)
                    )
            
        axes[idx].set_title(titles[idx], fontweight='bold', fontsize=12)
        axes[idx].set_xlabel(x_labels[idx])
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend(frameon=True, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_jurisdiction_proportions(
        df, save_path='eda_3_jurisdiction_proportions.png'):
    """
    3. Scatter Plot: Compare jurisdiction-level distance vs fuel proportions.
    Directly targets and labels (Distance=0.0, Fuel=1.0) for Alberta.
    """
    provinces = ['ab', 'bc', 'mb', 'on']
    long_data = []
    
    for prov in provinces:
        for idx, row in df.iterrows():
            if pd.notna(row[f'{prov}_dist_prop']) or pd.notna(row[f'{prov}_fuel_prop']):
                long_data.append({
                    'Province': prov.upper(),
                    'Distance Proportion': row[f'{prov}_dist_prop'],
                    'Fuel Proportion': row[f'{prov}_fuel_prop'],
                    'Variance': row[f'{prov}_dist_fuel_variance'],
                    'trip_origin': row.get('trip_origin', 'Unknown'),
                    'trip_destination': row.get('trip_destination', 'Unknown')
                })
                
    df_long = pd.DataFrame(long_data)
    
    if not df_long.empty:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df_long, x='Distance Proportion', y='Fuel Proportion', hue='Variance',
            palette='coolwarm', style='Province', s=200, edgecolor='black', alpha=0.9
        )
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7, label='Perfect Parity Line')
        
        ab_anomalies = df_long[(df_long['Province'] == 'AB') & (df_long['Distance Proportion'] == 0.0) & (df_long['Fuel Proportion'] == 1.0)]
        if not ab_anomalies.empty:
            for idx, row in ab_anomalies.iterrows():
                plt.annotate(
                    f"IFTA Breach: Fuel entry with Zero Distance\nRoute: {row['trip_origin']} ➔ {row['trip_destination']}",
                    xy=(0.0, 1.0), xytext=(0.15, 0.92),
                    arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fce4d6", ec="#c0392b", lw=1.5)
                )

        mean_x, std_x = df_long['Distance Proportion'].mean(), df_long['Distance Proportion'].std()
        mean_y, std_y = df_long['Fuel Proportion'].mean(), df_long['Fuel Proportion'].std()
        for i in [1, 2, 3]:
            color = '#e67e22' if i == 1 else '#c0392b'
            plt.axvline(mean_x + i * std_x, color=color, linestyle=':', alpha=0.3)
            plt.axhline(mean_y + i * std_y, color=color, linestyle=':', alpha=0.3)
        
        plt.title('Cross-Jurisdictional Alignment: Distance vs. Fuel Share', fontsize=14, pad=15, fontweight='bold')
        plt.xlabel('Proportion of Total Distance per Trip')
        plt.ylabel('Proportion of Total Fuel per Trip')
        plt.legend(title='Jurisdiction Audit Metrics', frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()


def plot_baselines_deviation(
        df, save_path_weekly='eda_4a_weekly_deviation.png',
        save_path_monthly='eda_4b_monthly_deviation.png'):
    """
    4. Deviation Analysis: Ratio of current trip metrics compared to rolling baselines.
    Annotates trips extending beyond +3 sigma volatility thresholds on either axis.
    """
    df_copy = df.copy()
    df_copy['dist_to_weekly_ratio'] = df_copy.apply(lambda r: r['total_km'] / r['weekly_avg_distance_km'] if r['weekly_avg_distance_km'] > 0 else 1, axis=1)
    df_copy['fuel_to_weekly_ratio'] = df_copy.apply(lambda r: r['total_fuel_litres'] / r['weekly_avg_fuel_litres'] if r['weekly_avg_fuel_litres'] > 0 else 1, axis=1)
    df_copy['dist_to_monthly_ratio'] = df_copy.apply(lambda r: r['total_km'] / r['monthly_avg_distance_km'] if r['monthly_avg_distance_km'] > 0 else 1, axis=1)
    df_copy['fuel_to_monthly_ratio'] = df_copy.apply(lambda r: r['total_fuel_litres'] / r['monthly_avg_fuel_litres'] if r['monthly_avg_fuel_litres'] > 0 else 1, axis=1)

    # --- Plot 4A: Weekly Base Volatility ---
    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=df_copy, x='dist_to_weekly_ratio', y='fuel_to_weekly_ratio', s=150, color='#9b59b6', edgecolor='black', alpha=0.8)
    plt.axvline(x=1, color='black', linestyle='-', alpha=0.4)
    plt.axhline(y=1, color='black', linestyle='-', alpha=0.4)
    
    mean_wx, std_wx = df_copy['dist_to_weekly_ratio'].mean(), df_copy['dist_to_weekly_ratio'].std()
    mean_wy, std_wy = df_copy['fuel_to_weekly_ratio'].mean(), df_copy['fuel_to_weekly_ratio'].std()
    thresh_wx, thresh_wy = mean_wx + 3 * std_wx, mean_wy + 3 * std_wy
    
    for i in [1, 2, 3]:
        color = '#e67e22' if i == 1 else '#c0392b'
        ls = '--' if i == 1 else ':'
        plt.axvline(mean_wx + i * std_wx, color=color, linestyle=ls, alpha=0.3)
        plt.axhline(mean_wy + i * std_wy, color=color, linestyle=ls, alpha=0.3)
        if mean_wx - i * std_wx >= 0: plt.axvline(mean_wx - i * std_wx, color=color, linestyle=ls, alpha=0.3)
        if mean_wy - i * std_wy >= 0: plt.axhline(mean_wy - i * std_wy, color=color, linestyle=ls, alpha=0.3)
        
    weekly_outliers = df_copy[(df_copy['dist_to_weekly_ratio'] > thresh_wx) | (df_copy['fuel_to_weekly_ratio'] > thresh_wy)]
    for idx, row in weekly_outliers.head(2).iterrows():
        plt.annotate(
            f"Volatility Outlier\n{row['trip_origin']} ➔ {row['trip_destination']}",
            xy=(row['dist_to_weekly_ratio'], row['fuel_to_weekly_ratio']),
            xytext=(row['dist_to_weekly_ratio'] - 0.5, row['fuel_to_weekly_ratio'] + 0.5),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.5, headwidth=5),
            fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#f2e6ff", ec="#8e44ad", lw=1)
        )

    plt.title('Operational Volatility: Deviations from Weekly Baselines', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Distance Ratio (Actual / Weekly Moving Average)')
    plt.ylabel('Fuel Ratio (Actual / Weekly Moving Average)')
    plt.tight_layout()
    plt.savefig(save_path_weekly, dpi=300)
    plt.show()

    # --- Plot 4B: Monthly Base Macro-Trends ---
    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=df_copy, x='dist_to_monthly_ratio', y='fuel_to_monthly_ratio', s=150, color='#e67e22', edgecolor='black', alpha=0.8)
    plt.axvline(x=1, color='black', linestyle='-', alpha=0.4)
    plt.axhline(y=1, color='black', linestyle='-', alpha=0.4)
    
    mean_mx, std_mx = df_copy['dist_to_monthly_ratio'].mean(), df_copy['dist_to_monthly_ratio'].std()
    mean_my, std_my = df_copy['fuel_to_monthly_ratio'].mean(), df_copy['fuel_to_monthly_ratio'].std()
    thresh_mx, thresh_my = mean_mx + 3 * std_mx, mean_my + 3 * std_my
    
    for i in [1, 2, 3]:
        color = '#e67e22' if i == 1 else '#c0392b'
        ls = '--' if i == 1 else ':'
        plt.axvline(mean_mx + i * std_mx, color=color, linestyle=ls, alpha=0.3)
        plt.axhline(mean_my + i * std_my, color=color, linestyle=ls, alpha=0.3)
        if mean_mx - i * std_mx >= 0:
            plt.axvline(mean_mx - i * std_mx, color=color, linestyle=ls,
                        alpha=0.3)
        if mean_my - i * std_my >= 0:
            plt.axhline(mean_my - i * std_my, color=color, linestyle=ls,
                        alpha=0.3)
        
    monthly_outliers = df_copy[(df_copy['dist_to_monthly_ratio'] > thresh_mx) | (df_copy['fuel_to_monthly_ratio'] > thresh_my)]
    for idx, row in monthly_outliers.head(2).iterrows():
        plt.annotate(
            f"Macro Outlier\n{row['trip_origin']} ➔ {row['trip_destination']}",
            xy=(row['dist_to_monthly_ratio'], row['fuel_to_monthly_ratio']),
            xytext=(row['dist_to_monthly_ratio'] - 0.5, row['fuel_to_monthly_ratio'] + 0.5),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.5, headwidth=5),
            fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#fff2e6", ec="#d35400", lw=1)
        )

    plt.title('Macro-Operational Volatility: Deviations from Monthly Baselines', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Distance Ratio (Actual / Monthly Moving Average)')
    plt.ylabel('Fuel Ratio (Actual / Monthly Moving Average)')
    plt.tight_layout()
    plt.savefig(save_path_monthly, dpi=300)
    plt.show()


def plot_baseline_historical_correlations(
        df, save_path='eda_5_baseline_correlations.png'):
    """
    5. Baseline Consistency Check:
    Plots moving averages against each other and labels structural tracking drift (>3 sigma).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Subplot A: Weekly Baselines ---
    sns.scatterplot(data=df, x='weekly_avg_distance_km', y='weekly_avg_fuel_litres', s=150, color='#8e44ad', edgecolor='black', alpha=0.75, ax=axes[0])
    mean_wx, std_wx = df['weekly_avg_distance_km'].median(), df['weekly_avg_distance_km'].std()
    mean_wy, std_wy = df['weekly_avg_fuel_litres'].median(), df['weekly_avg_fuel_litres'].std()

    axes[0].axvline(mean_wx, color='black', linestyle='-', alpha=0.5, label=fr'$\tilde\mu$_dist: {mean_wx:.0f}')
    axes[0].axhline(mean_wy, color='black', linestyle='-', alpha=0.5, label=fr'$\tilde\mu$_fuel: {mean_wy:.0f}')
    
    for i in [1, 2, 3]:
        color = '#e67e22' if i == 1 else '#c0392b'
        ls = '--' if i == 1 else ':'
        axes[0].axvline(mean_wx + i * std_wx, color=color, linestyle=ls, alpha=0.3)
        axes[0].axhline(mean_wy + i * std_wy, color=color, linestyle=ls, alpha=0.3)
        if mean_wx - i * std_wx >= 0:
            axes[0].axvline(mean_wx - i * std_wx, color=color, linestyle=ls,
                            alpha=0.3)
        if mean_wy - i * std_wy >= 0:
            axes[0].axhline(mean_wy - i * std_wy, color=color, linestyle=ls,
                            alpha=0.3)
        
    w_drift = df[(df['weekly_avg_distance_km'] > (mean_wx + 2*std_wx)) |
                 (df['weekly_avg_fuel_litres'] < (mean_wy - 2*std_wy)) &
                 (df['weekly_avg_distance_km'] > 0)]
    for enum_idx, (idx, row) in enumerate(w_drift.iterrows()):
        axes[0].annotate(
            f"Structural Drift\nRoute: {row['trip_origin']} ➔ {row['trip_destination']}",
            xy=(row['weekly_avg_distance_km'], row['weekly_avg_fuel_litres']),
            xytext=(row['weekly_avg_distance_km'] - 600 + enum_idx*300,
                    row['weekly_avg_fuel_litres'] + 100 + enum_idx*100),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.5, headwidth=5),
            fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#f2e6ff", ec="#8e44ad", lw=1)
        )
    axes[0].set_title('Macro Profile: Weekly Rolling Baselines', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Moving Average Distance (km)')
    axes[0].set_ylabel('Moving Average Fuel (L)')
    axes[0].legend(frameon=True, fontsize=10)
    
    # --- Subplot B: Monthly Baselines ---
    sns.scatterplot(data=df, x='monthly_avg_distance_km', y='monthly_avg_fuel_litres', s=150, color='#d35400', edgecolor='black', alpha=0.75, ax=axes[1])
    mean_mx, std_mx = df['monthly_avg_distance_km'].median(), df['monthly_avg_distance_km'].std()
    mean_my, std_my = df['monthly_avg_fuel_litres'].median(), df['monthly_avg_fuel_litres'].std()
    
    axes[1].axvline(mean_mx, color='black', linestyle='-', alpha=0.5, label=fr'$\tilde\mu$_dist: {mean_mx:.0f}')
    axes[1].axhline(mean_my, color='black', linestyle='-', alpha=0.5, label=fr'$\tilde\mu$_fuel: {mean_my:.0f}')
    
    for i in [1, 2, 3]:
        color = '#e67e22' if i == 1 else '#c0392b'
        ls = '--' if i == 1 else ':'
        axes[1].axvline(mean_mx + i * std_mx, color=color, linestyle=ls, alpha=0.3)
        axes[1].axhline(mean_my + i * std_my, color=color, linestyle=ls, alpha=0.3)
        if mean_mx - i * std_mx >= 0:
            axes[1].axvline(mean_mx - i * std_mx, color=color, linestyle=ls,
                            alpha=0.3)
        if mean_my - i * std_my >= 0:
            axes[1].axhline(mean_my - i * std_my, color=color, linestyle=ls,
                            alpha=0.3)
        
    m_drift = df[(df['monthly_avg_distance_km'] > (mean_mx + 2*std_mx)) |
                 (df['monthly_avg_fuel_litres'] < (mean_my - 2*std_my)) &
                 (df['monthly_avg_distance_km']) > 0]
    for enum_idx, (idx, row) in enumerate(m_drift.iterrows()):
        axes[1].annotate(
            f"Macro Drift\nRoute: {row['trip_origin']} ➔ {row['trip_destination']}",
            xy=(row['monthly_avg_distance_km'], row['monthly_avg_fuel_litres']),
            xytext=(row['monthly_avg_distance_km'] - 600 + enum_idx*400,
                    row['monthly_avg_fuel_litres'] + 100 - enum_idx*200),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.5, headwidth=5),
            fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#fff2e6", ec="#d35400", lw=1)
        )
    axes[1].set_title('Macro Profile: Monthly Rolling Baselines', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Moving Average Distance (km)')
    axes[1].set_ylabel('Moving Average Fuel (L)')
    axes[1].legend(frameon=True, fontsize=10)
    
    plt.suptitle('Fleet Baseline Consistency: Distance vs. Fuel Macro-Correlations', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_window_efficiency_features(df, 
                                    save_path='eda_6_window_efficiency.png'):
    """
    6. Three-panel histogram for the new ±3 day window features.
    Coloured by fuel_source to show how data quality affects each distribution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    features = ['cycle_fuel_per_km', 'km_per_tank', 'avg_fuel_per_tank']
    titles   = [
        'Window Efficiency (L/km) — Fuelling Cycle View',
        'Window Distance (km) — ±3 Day Coverage',
        'Nearby Fuel (L) — Closest Provincial Record'
    ]
    xlabels = ['L/km (±3 day window)', 'Total km (±3 day window)', 'Litres (±3 day window)']
    colors   = ['#1abc9c', '#3498db', '#9b59b6']

    for i, (feat, title, xlabel, color) in enumerate(
        zip(features, titles, xlabels, colors)
    ):
        data = df[feat].dropna()
        if data.empty:
            axes[i].set_title(f'{title}\n(No data)')
            continue

        mean_val = data.median()
        std_val  = data.std()

        axes[i].hist(data, bins=15, color=color,
                     edgecolor='black', alpha=0.7)
        sns.kdeplot(data, ax=axes[i], color=color,
                    linewidth=2, linestyle='-')

        axes[i].axvline(mean_val, color='black', linewidth=2,
                        label=f'Mean: {mean_val:.2f}')
        for sigma, ls in zip([1, 2, 3, -1, -2, -3], ['--', ':', '-.']*2):
            c = '#e67e22' if sigma == 1 else '#c0392b'
            axes[i].axvline(mean_val + sigma * std_val,
                            color=c, linestyle=ls, alpha=0.6,
                            label=f'+{sigma}σ: {mean_val + sigma*std_val:.2f}')

        axes[i].set_title(title, fontweight='bold', fontsize=11)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel('Frequency')
        axes[i].legend(fontsize=9, frameon=True)

    plt.suptitle('±3-Day Fuelling Window Feature Distributions',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_trip_vs_window_efficiency(df,
                                   save_path='eda_7_trip_vs_window_efficiency.png'):
    """
    7. Scatter: per-trip efficiency vs window efficiency, coloured by fuel_source.
    The interesting records are where the two metrics DISAGREE — a normal
    trip efficiency but anomalous window efficiency suggests the fuel record
    belongs to a different trip cycle entirely.
    """
    df_plot = df[['km_per_tank', 'avg_fuel_per_tank', 'cycle_fuel_per_km',
                  'fuel_source', 'trip_origin', 'trip_destination']].dropna()

    if df_plot.empty:
        print("⚠️ No data for trip vs window efficiency plot.")
        return

    palette = {
        'invoice': '#27ae60',
        'provincial_log': '#2980b9',
        'provincial_log_nearby': '#f39c12',
        'missing': '#e74c3c'
    }

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot, x='km_per_tank', y='avg_fuel_per_tank',
        hue='fuel_source', palette=palette,
        s=150, edgecolor='black', alpha=0.85
    )

    # Parity line — points on this line have consistent trip vs window efficiency
    max_val = max(df_plot['km_per_tank'].max(),
                  df_plot['avg_fuel_per_tank'].max())
    plt.plot([0, max_val], [0, 0.55*max_val], color='gray',
             linestyle='--', alpha=0.6, label='0.55 L/km')
    plt.plot([0, max_val], [0, 1.5*max_val], color='gray',
             linestyle='-.', alpha=0.6, label='1.5 L/km')
    plt.plot([0, max_val], [0, 0.06*max_val], color='gray',
             linestyle=':', alpha=0.6, label='0.06 L/km')

    # Annotate records with high disagreement from mean cycle_fuel_per_km
    mean_wfpm = df_plot['cycle_fuel_per_km'].median()
    std_wfpm = df_plot['cycle_fuel_per_km'].std()
    df_plot['z_score_wfpm'] = ((df_plot['cycle_fuel_per_km'] - mean_wfpm) /
                               std_wfpm)
    wfpm_drift = df[(df_plot['z_score_wfpm'] > 3) |
                    (df_plot['cycle_fuel_per_km'] < 0.05)]
    for enum_idx, (_, row) in enumerate(wfpm_drift.iterrows()):
        plt.annotate(
            f"Cycle Mismatch\n{row['trip_origin']} ➔ {row['trip_destination']}",
            xy=(row['km_per_tank'], row['avg_fuel_per_tank']),
            xytext=(row['km_per_tank'] + 500*enum_idx,
                    row['avg_fuel_per_tank'] + 500 - 75*enum_idx),
            arrowprops=dict(facecolor='black', shrink=0.01,
                            width=1, headwidth=6),
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="#fce4d6", ec="#c0392b", lw=1.5)
        )

    plt.title('Fuelling cycle efficiency y source',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('±3 Day window distance before fuel(km)')
    plt.ylabel('±3 Day window nearest Fuel quantity (L)')
    plt.legend(title='Nearby fuel vs distance', frameon=True)
    plt.ylim(-50, df_plot['avg_fuel_per_tank'].max()*1.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_fuel_source_breakdown(df,
                               save_path='eda_8_fuel_source_breakdown.png'):
    """
    8. Fuel source quality breakdown — a data quality slide.
    Shows what proportion of trips have verified vs self-reported vs missing fuel.
    Critical for framing the model's reliability to auditors.
    """
    counts = df['fuel_source'].value_counts()

    # Ordered by reliability for clear communication
    order   = ['invoice', 'provincial_log',
                'provincial_log_nearby', 'missing']
    palette = ['#27ae60', '#2980b9', '#f39c12', '#e74c3c']
    labels  = {
        'invoice': 'Invoice Verified',
        'provincial_log': 'Provincial Log (same row)',
        'provincial_log_nearby': 'Provincial Log (±3 day)',
        'missing': 'No Fuel Record'
    }

    ordered_counts = pd.Series({
        k: counts.get(k, 0) for k in order
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    bars = axes[0].bar(
        [labels[k] for k in order],
        ordered_counts.values,
        color=palette, edgecolor='black', alpha=0.85
    )
    axes[0].bar_label(bars, fmt='%d', padding=4, fontsize=11)
    axes[0].set_title('Trip Count by Fuel Source',
                      fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Number of Trips')
    axes[0].set_xticklabels(
        [labels[k] for k in order], rotation=15, ha='right'
    )

    # Pie chart — proportion view
    axes[1].pie(
        ordered_counts.values,
        labels=[labels[k] for k in order],
        colors=palette, autopct='%1.1f%%',
        startangle=140, pctdistance=0.75,
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
    )
    axes[1].set_title('Fuel Source Reliability Breakdown (%)',
                      fontweight='bold', fontsize=12)

    plt.suptitle('Data Quality Assessment: Fuel Record Coverage',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def run_full_eda_pipeline(df_input):
    print("🚀 Running clean, publication-grade modular EDA validations...")
    set_plot_style()

    plot_total_km_vs_fuel(df_input)            # updated for fuel_source
    plot_efficiency_distributions_hybrid(df_input)  # updated for fuel_source
    plot_jurisdiction_proportions(df_input)
    plot_baselines_deviation(df_input)
    plot_baseline_historical_correlations(df_input)
    plot_window_efficiency_features(df_input)  # new
    plot_trip_vs_window_efficiency(df_input)   # new
    plot_fuel_source_breakdown(df_input)       # new

    print("📊 All EDA visual assets generated and exported successfully!")
