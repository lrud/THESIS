#!/usr/bin/env python3
"""Multi-window comparison with industry standard directional accuracy."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
DATA_PATH = 'data/processed/bitcoin_lstm_features_v1.6_final.csv'
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f'Data loaded: {df.shape[0]:,} samples')

# Apply rolling normalization function
def apply_rolling_normalization(df, feature_cols, window=720):
    df_norm = df.copy()
    scaling_params = {}

    for col in feature_cols:
        if col in ['lee_mykland_jump', 'jump_indicator']:
            df_norm[col] = df[col]
            continue
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, min_periods=1).std().replace(0, 1)
        df_norm[f'{col}_norm'] = (df[col] - rolling_mean) / rolling_std
        scaling_params[col] = {'mean': rolling_mean.iloc[-1], 'std': rolling_std.iloc[-1]}

    df_norm['dvol_rolling_mean'] = df['dvol'].rolling(window=window, min_periods=1).mean()
    df_norm['dvol_rolling_std'] = df['dvol'].rolling(window=window, min_periods=1).std().replace(0, 1)
    df_norm['timestamp'] = df['timestamp']
    return df_norm, scaling_params

def denormalize_predictions(y_pred_norm, df_norm, indices):
    mean = df_norm.loc[indices, 'dvol_rolling_mean'].values
    std = df_norm.loc[indices, 'dvol_rolling_std'].values
    return y_pred_norm * std + mean

# Industry standard directional accuracy (Pesaran-Timmermann, 1992)
def directional_accuracy(y_true, y_pred):
    """
    Mean Directional Accuracy (MDA) per Pesaran & Timmermann (1992).

    Compares: sgn(A_t - A_{t-1}) vs sgn(F_t - A_{t-1})
    """
    actual_direction = np.sign(y_true[1:] - y_true[:-1])
    predicted_direction = np.sign(y_pred[:-1] - y_true[:-1])
    valid = (actual_direction != 0)
    correct = (actual_direction[valid] == predicted_direction[valid])
    return (correct.sum() / valid.sum() * 100) if valid.sum() > 0 else 0.0

# Model specifications
linear_specs = [
    ('OLS_NoLags', ['transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']),
    ('OLS_NoLags_Jumps', ['transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                        'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d']),
    ('HAR_RV', ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d']),
    ('OLS_WithLags', ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                      'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']),
    ('OLS_WithLags_Jumps', ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                            'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                            'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d']),
]

tree_specs = [
    ('RF_NoLag', 4, ['transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'], False),
    ('RF_Lags', 7, ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                   'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'], False),
    ('RF_NoLag_Jumps', 8, ['transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                          'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'], False),
    ('RF_Lags_Jumps', 11, ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                         'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                         'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'], False),
    ('XGB_NoLag', 4, ['transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'], True),
    ('XGB_NoLag_Jumps', 8, ['transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                          'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'], True),
    ('XGB_Lags', 7, ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                   'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'], True),
    ('XGB_Lags_Jumps', 11, ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                         'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
                         'lee_mykland_jump', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'], True),
]

WINDOW_SIZES = [72, 168, 336, 720]

print('Starting multi-window experiments with industry standard directional accuracy...')
print('='*100)

all_window_results = {}

for window in WINDOW_SIZES:
    print(f'\nWINDOW: {window}h ({window//24}d)')
    print('-'*80)

    # Build features list
    all_features = ['dvol']
    for spec in linear_specs:
        all_features.extend([f for f in spec[1] if f not in all_features])
    for spec in tree_specs:
        all_features.extend([f for f in spec[2] if f not in all_features])

    df_norm, _ = apply_rolling_normalization(df, all_features, window=window)

    # Train/val/test split
    n = len(df_norm)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)

    train_df = df_norm.iloc[:n_train].copy()
    val_df = df_norm.iloc[n_train:n_train + n_val].copy()
    test_df = df_norm.iloc[n_train + n_val:].copy()

    train_df['target'] = train_df['dvol_norm'].shift(-1)
    val_df['target'] = val_df['dvol_norm'].shift(-1)
    test_df['target'] = test_df['dvol_norm'].shift(-1)

    train_df['actual_dvol'] = train_df['dvol'].shift(-1)
    val_df['actual_dvol'] = val_df['dvol'].shift(-1)
    test_df['actual_dvol'] = test_df['dvol'].shift(-1)

    feature_cols_to_check = [f'{f}_norm' for f in all_features if f not in ['lee_mykland_jump', 'jump_indicator']]
    feature_cols_to_check.append('lee_mykland_jump')

    train_df = train_df.dropna(subset=['target'] + feature_cols_to_check)
    val_df = val_df.dropna(subset=['target'] + feature_cols_to_check)
    test_df = test_df.dropna(subset=['target'] + feature_cols_to_check)

    y_train = train_df['target'].values
    y_val = val_df['target'].values
    y_test = test_df['target'].values

    print(f'Samples: {len(y_train)} train | {len(y_val)} val | {len(y_test)} test')

    results = {}

    # Linear models
    for name, features in linear_specs:
        final_features = []
        for f in features:
            if f in ['lee_mykland_jump', 'jump_indicator']:
                final_features.append(f)
            else:
                final_features.append(f'{f}_norm')

        X_train = train_df[final_features].values
        X_val = val_df[final_features].values
        X_test = test_df[final_features].values

        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        y_pred_denorm = denormalize_predictions(y_pred_test, test_df, test_df.index)
        y_test_denorm = denormalize_predictions(y_test, test_df, test_df.index)

        dir_acc = directional_accuracy(y_test_denorm, y_pred_denorm)

        results[name] = {
            'r2': float(r2_score(y_test_denorm, y_pred_denorm)),
            'rmse': float(np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))),
            'mae': float(mean_absolute_error(y_test_denorm, y_pred_denorm)),
            'dir_acc': float(dir_acc),
            'features': len(features)
        }

    # Tree models
    for name, n_features, features, is_xgb in tree_specs:
        final_features = []
        for f in features:
            if f in ['lee_mykland_jump', 'jump_indicator']:
                final_features.append(f)
            else:
                final_features.append(f'{f}_norm')

        X_train = train_df[final_features].values
        X_val = val_df[final_features].values
        X_test = test_df[final_features].values

        if is_xgb:
            model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                        min_samples_split=10, min_samples_leaf=4,
                                        random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        y_pred_denorm = denormalize_predictions(y_pred_test, test_df, test_df.index)
        y_test_denorm = denormalize_predictions(y_test, test_df, test_df.index)

        dir_acc = directional_accuracy(y_test_denorm, y_pred_denorm)

        results[name] = {
            'r2': float(r2_score(y_test_denorm, y_pred_denorm)),
            'rmse': float(np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))),
            'mae': float(mean_absolute_error(y_test_denorm, y_pred_denorm)),
            'dir_acc': float(dir_acc),
            'features': n_features
        }

    all_window_results[window] = results

print('\n' + '='*100)
print('RESULTS SUMMARY - MULTI-WINDOW COMPARISON WITH INDUSTRY STANDARD DIRECTIONAL ACCURACY')
print('Formula: Pesaran-Timmermann (1992) - sgn(A_t - A_{t-1}) vs sgn(F_t - A_{t-1})')
print('='*100)

# Print results by window
for window in WINDOW_SIZES:
    print(f'\n{window}h ({window//24}d) Window:')
    print('-'*100)
    print(f"{'Model':<22} {'Feats':>5} {'R²':>9} {'RMSE':>8} {'MAE':>8} {'Dir%':>7}")
    print('-'*100)

    results = all_window_results[window]
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['dir_acc'], reverse=True):
        print(f'{name:<22} {metrics["features"]:>5} {metrics["r2"]:>9.4f} {metrics["rmse"]:>8.2f} {metrics["mae"]:>8.2f} {metrics["dir_acc"]:>6.1f}%')

print('\n' + '='*100)
print('BEST DIRECTIONAL ACCURACY BY WINDOW')
print('='*100)

for window in WINDOW_SIZES:
    results = all_window_results[window]
    best = max(results.items(), key=lambda x: x[1]['dir_acc'])
    print(f'{window}h ({window//24}d): {best[0]:<22} Dir% = {best[1]["dir_acc"]:.1f}% | R² = {best[1]["r2"]:.4f}')

print('\n' + '='*100)
print('HAR_RV PERFORMANCE ACROSS ALL WINDOWS')
print('='*100)

for window in WINDOW_SIZES:
    har_rv = all_window_results[window]['HAR_RV']
    print(f'{window}h ({window//24}d): Dir% = {har_rv["dir_acc"]:.1f}%, R² = {har_rv["r2"]:.4f}, RMSE = {har_rv["rmse"]:.2f}')

print('\n' + '='*100)
print('AVERAGE DIRECTIONAL ACCURACY BY WINDOW')
print('='*100)

for window in WINDOW_SIZES:
    results = all_window_results[window]
    avg_dir = np.mean([m['dir_acc'] for m in results.values()])
    print(f'{window}h ({window//24}d): Avg Dir% = {avg_dir:.1f}%')

# Save results
import json
results_to_save = {}
for window, models in all_window_results.items():
    results_to_save[str(window)] = models

with open('results/analysis/multi_window_dir_acc_results.json', 'w') as f:
    json.dump(results_to_save, f, indent=2)

print('\n' + '='*100)
print('Results saved to: results/analysis/multi_window_dir_acc_results.json')
print('='*100)
