#!/usr/bin/env python3
"""
Multi-Window Normalization Comparison for Regression Models

Tests 13 regression models across 4 different rolling window sizes:
- 72 hours (3 days)
- 168 hours (7 days)
- 336 hours (14 days)
- 720 hours (30 days)

This answers: Does normalization window size affect model performance?
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import json
from datetime import datetime

# Configuration
DATA_PATH = '/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/processed/bitcoin_lstm_features_v1.6_final.csv'
OUTPUT_DIR = Path('results/analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Window sizes to test (hours)
WINDOW_SIZES = [72, 168, 336, 720]  # 3, 7, 14, 30 days

# Train/val/test split ratios
TRAIN_SPLIT = 0.60
VAL_SPLIT = 0.20
TEST_SPLIT = 0.20
RANDOM_STATE = 42

print("="*80)
print("MULTI-WINDOW NORMALIZATION EXPERIMENTS - REGRESSION MODELS")
print("="*80)
print(f"Window sizes to test: {WINDOW_SIZES}")
print(f"Models per window: 13 (5 linear + 8 tree-based)")
print(f"Total experiments: {len(WINDOW_SIZES) * 13}")
print("="*80)

# =============================================================================
# DATA LOADING
# =============================================================================

df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"\nData: {df.shape[0]:,} samples ({df['timestamp'].min()} to {df['timestamp'].max()})")

# =============================================================================
# NORMALIZATION FUNCTION
# =============================================================================

def apply_rolling_normalization(df, feature_cols, window=720):
    """Apply rolling window z-score normalization."""
    df_norm = df.copy()
    scaling_params = {}

    for col in feature_cols:
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero

        df_norm[f'{col}_norm'] = (df[col] - rolling_mean) / rolling_std

        # Store parameters for denormalization
        scaling_params[col] = {
            'mean': rolling_mean,
            'std': rolling_std
        }

    return df_norm, scaling_params

def denormalize_predictions(y_pred_norm, scaling_params, target_col='dvol'):
    """Convert normalized predictions back to raw scale."""
    mean = scaling_params[target_col]['mean'].iloc[-1]  # Use last value
    std = scaling_params[target_col]['std'].iloc[-1]
    return y_pred_norm * std + mean

# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

# Linear model specifications (5 models)
linear_specs = [
    ('OLS_NoLags', ['dvol', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']),
    ('OLS_NoLags_Jumps', ['dvol', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                        'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d']),
    ('HAR_RV', ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d']),
    ('OLS_WithLags', ['dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                      'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']),
    ('OLS_WithLags_Jumps', ['dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                            'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                            'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d']),
]

# Tree model specifications (8 models)
tree_specs = [
    ('RF_NoLag', 4, ['dvol', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'], False),
    ('RF_Lags', 7, ['dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                   'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'], False),
    ('RF_NoLag_Jumps', 8, ['dvol', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                          'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'], False),
    ('RF_Lags_Jumps', 11, ['dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                         'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                         'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'], False),
    ('XGB_NoLag', 4, ['dvol', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'], True),
    ('XGB_NoLag_Jumps', 8, ['dvol', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                          'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'], True),
    ('XGB_Lags', 7, ['dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                   'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'], True),
    ('XGB_Lags_Jumps', 11, ['dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                         'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                         'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'], True),
]

# =============================================================================
# EXPERIMENT FUNCTION FOR ONE WINDOW SIZE
# =============================================================================

def run_experiments_for_window(window_size):
    """Run all 13 models for a specific window size."""

    print(f"\n{'='*60}")
    print(f"WINDOW: {window_size} hours ({window_size//24} days)")
    print(f"{'='*60}")

    # Apply normalization with specified window
    all_features = []
    for spec in linear_specs:
        all_features.extend([f for f in spec[1] if f not in all_features])
    for spec in tree_specs:
        all_features.extend([f for f in spec[2] if f not in all_features])

    df_norm, scaling_params = apply_rolling_normalization(df, all_features, window=window_size)

    # Train/val/test split
    n = len(df_norm)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    train_df = df_norm.iloc[:n_train].copy()
    val_df = df_norm.iloc[n_train:n_train + n_val].copy()
    test_df = df_norm.iloc[n_train + n_val:].copy()

    # Create targets (normalized next-period DVOL)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df['target'] = train_df['dvol_norm'].shift(-1)
    val_df['target'] = val_df['dvol_norm'].shift(-1)
    test_df['target'] = test_df['dvol_norm'].shift(-1)

    # Drop NaN from targets (this also removes corresponding rows from features)
    train_df = train_df.dropna(subset=['target'])
    val_df = val_df.dropna(subset=['target'])
    test_df = test_df.dropna(subset=['target'])

    y_train = train_df['target'].values
    y_val = val_df['target'].values
    y_test = test_df['target'].values

    print(f"Samples: {len(y_train)} train | {len(y_val)} val | {len(y_test)} test")

    results = {}

    # -------------------------------------------------------------------------
    # LINEAR MODELS (5)
    # -------------------------------------------------------------------------

    for name, features in linear_specs:
        # Get normalized feature names
        feature_cols = [f'{f}_norm' for f in features]

        # Prepare data
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        X_test = test_df[feature_cols].values

        # Train model
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)

        # Predict
        y_pred_test = model.predict(X_test)

        # Denormalize predictions for evaluation
        y_test_denorm = denormalize_predictions(y_test, scaling_params, 'dvol')
        y_pred_denorm = denormalize_predictions(y_pred_test, scaling_params, 'dvol')

        # Metrics
        r2_norm = r2_score(y_test, y_pred_test)
        r2 = r2_score(y_test_denorm, y_pred_denorm)
        rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
        mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
        directional = ((y_test_denorm > 0) == (y_pred_denorm > 0)).mean()

        results[name] = {
            'r2_norm': float(r2_norm),
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'directional': float(directional),
            'features': len(features)
        }

    # -------------------------------------------------------------------------
    # TREE MODELS (8)
    # -------------------------------------------------------------------------

    for name, n_features, features, is_xgb in tree_specs:
        feature_cols = [f'{f}_norm' for f in features]

        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        X_test = test_df[feature_cols].values

        if is_xgb:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        # Denormalize
        y_test_denorm = denormalize_predictions(y_test, scaling_params, 'dvol')
        y_pred_denorm = denormalize_predictions(y_pred_test, scaling_params, 'dvol')

        results[name] = {
            'r2_norm': float(r2_score(y_test, y_pred_test)),
            'r2': float(r2_score(y_test_denorm, y_pred_denorm)),
            'rmse': float(np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))),
            'mae': float(mean_absolute_error(y_test_denorm, y_pred_denorm)),
            'directional': float(((y_test_denorm > 0) == (y_pred_denorm > 0)).mean()),
            'features': n_features
        }

    return results

# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================

all_window_results = {}

for window in WINDOW_SIZES:
    print(f"\nRunning experiments for window: {window}h ({window//24}d)...")
    all_window_results[window] = run_experiments_for_window(window)

# =============================================================================
# COMPILE COMPARISON RESULTS
# =============================================================================

print("\n" + "="*80)
print("MULTI-WINDOW COMPARISON RESULTS")
print("="*80)

window_comparison = []

for window in WINDOW_SIZES:
    results = all_window_results[window]
    for name, metrics in results.items():
        window_comparison.append({
            'window_hours': window,
            'window_days': window // 24,
            'model': name,
            'features': metrics['features'],
            'r2_norm': metrics['r2_norm'],
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'directional': metrics['directional']
        })

comparison_df = pd.DataFrame(window_comparison)

# Full comparison table
print("\n" + "-"*80)
print("ALL MODELS - ALL WINDOWS (Ranking by R²)")
print("-"*80)
print(comparison_df.sort_values('r2', ascending=False).to_string(index=False))

# Best model by window
print("\n" + "="*80)
print("BEST MODEL BY WINDOW SIZE")
print("="*80)

for window in WINDOW_SIZES:
    window_data = comparison_df[comparison_df['window_hours'] == window]
    if not window_data.empty:
        best = window_data.loc[window_data['r2'].idxmax()]
        print(f"\n{window}h ({window//24}d): {best['model']}")
        print(f"  R²_norm: {best['r2_norm']:.4f} | R²: {best['r2']:.4f} | RMSE: {best['rmse']:.2f}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

avg_r2_by_window = comparison_df.groupby('window_hours')['r2'].mean().to_dict()
best_r2_by_window = comparison_df.groupby('window_hours')['r2'].max().to_dict()

print(f"\nAverage R² by window:")
for w in WINDOW_SIZES:
    print(f"  {w}h ({w//24}d): {avg_r2_by_window[w]:.4f}")

print(f"\nBest R² by window:")
for w in WINDOW_SIZES:
    print(f"  {w}h ({w//24}d): {best_r2_by_window[w]:.4f}")

overall_best_window = max(avg_r2_by_window, key=avg_r2_by_window.get)
print(f"\nBest average window: {overall_best_window}h ({avg_r2_by_window[overall_best_window]:.4f})")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output_file = OUTPUT_DIR / f'multi_window_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

summary = {
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'window_sizes_tested': WINDOW_SIZES,
    'total_experiments': len(WINDOW_SIZES) * 13,
    'results_by_window': {w: all_window_results[w] for w in WINDOW_SIZES},
    'summary_statistics': {
        'average_r2_by_window': avg_r2_by_window,
        'best_r2_by_window': best_r2_by_window,
        'overall_best_window': int(overall_best_window)
    }
}

with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to: {output_file}")

# Also save CSV
csv_file = OUTPUT_DIR / f'multi_window_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
comparison_df.to_csv(csv_file, index=False)
print(f"CSV saved to: {csv_file}")

print("\n" + "="*80)
print("EXPERIMENTS COMPLETE")
print("="*80)
