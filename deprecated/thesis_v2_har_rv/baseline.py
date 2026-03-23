"""
Baseline Model Runners for HAR-RV Analysis.

This module contains functions for running various baseline models (OLS, HAR-RV variants,
Random Forest, XGBoost) with jump-focused evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from typing import Dict


def _load_and_prepare_data(data_path: str, output_dir: str, data_version: str):
    """Helper function to load and prepare data for all baseline models."""
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    jump_masks_path = f'{output_dir}/jump_period_masks_{data_version}.csv'
    if not Path(jump_masks_path).exists():
        raise FileNotFoundError(f"Jump masks not found at {jump_masks_path}")

    df_masks = pd.read_csv(jump_masks_path)

    min_len = min(len(df), len(df_masks))
    df = df.iloc[:min_len].copy()
    df_masks = df_masks.iloc[:min_len].copy()

    df['dvol_change'] = df['dvol'].shift(-1) - df['dvol']
    df = df.dropna(subset=['dvol_change'])
    df_masks = df_masks.iloc[:len(df)].copy()

    return df, df_masks


def _create_splits(df_clean, df_masks_clean):
    """Helper function to create train/test splits."""
    n = len(df_clean)
    train_size = int(n * 0.7)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_size] = True
    test_mask = ~train_mask
    jump_mask = df_masks_clean['jump_indicator'].values == 1
    return n, train_size, train_mask, test_mask, jump_mask


def _calculate_metrics(actual, pred, mask):
    """Helper function to calculate performance metrics."""
    if mask.sum() == 0:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'samples': 0}
    masked_actual = actual[mask]
    masked_pred = pred[mask]
    return {
        'r2': float(r2_score(masked_actual, masked_pred)),
        'rmse': float(np.sqrt(mean_squared_error(masked_actual, masked_pred))),
        'mae': float(mean_absolute_error(masked_actual, masked_pred)),
        'samples': int(mask.sum())
    }


def run_phase1_baseline_analysis(data_path: str, data_version: str = 'v1.1',
                                  output_dir: str = 'results/thesis_v2'):
    """Run Phase 1 OLS Baseline Evaluation."""
    print("=" * 80)
    print(f"PHASE 1C: OLS BASELINE EVALUATION ({data_version})")
    print("=" * 80)

    df, df_masks = _load_and_prepare_data(data_path, output_dir, data_version)

    feature_cols = [
        'dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'
    ]

    valid_mask = df[feature_cols + ['dvol_change']].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    df_masks_clean = df_masks[valid_mask].copy()

    n, train_size, train_mask, test_mask, jump_mask = _create_splits(df_clean, df_masks_clean)

    X = df_clean[feature_cols].values
    y = df_clean['dvol_change'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ols_model = LinearRegression(fit_intercept=True)
    ols_model.fit(X_scaled[train_mask], y[train_mask])
    y_pred = ols_model.predict(X_scaled)

    results = {
        'model_specification': {
            'model_type': 'vanilla_ols',
            'target': 'next_dvol_change',
            'features': feature_cols,
            'feature_count': len(feature_cols)
        },
        'data_info': {
            'total_observations': int(n),
            'training_samples': int(train_mask.sum()),
            'testing_samples': int(test_mask.sum()),
            'jump_periods_test': int((test_mask & jump_mask).sum()),
            'split_date': df_clean['timestamp'].iloc[train_size].isoformat()
        },
        'performance_metrics': {
            'training': {
                'jump_periods': _calculate_metrics(y, y_pred, train_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, train_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, train_mask)
            },
            'testing': {
                'jump_periods': _calculate_metrics(y, y_pred, test_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, test_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, test_mask)
            }
        }
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'ols_baseline_{data_version}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    return results


def run_phase1_baseline_with_diagnostics(data_path: str, data_version: str = 'v1.1',
                                         output_dir: str = 'results/thesis_v2'):
    """Run Phase 1 OLS Baseline with comprehensive statistical diagnostics."""
    from .diagnostics import calculate_statistical_diagnostics
    from .visualization import create_statistical_diagnostics_summary

    print("=" * 80)
    print(f"PHASE 1C+: OLS BASELINE WITH STATISTICAL DIAGNOSTICS ({data_version})")
    print("=" * 80)

    df, df_masks = _load_and_prepare_data(data_path, output_dir, data_version)

    feature_cols = [
        'dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'
    ]

    valid_mask = df[feature_cols + ['dvol_change']].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    df_masks_clean = df_masks[valid_mask].copy()

    n, train_size, train_mask, test_mask, jump_mask = _create_splits(df_clean, df_masks_clean)

    X = df_clean[feature_cols].values
    y = df_clean['dvol_change'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ols_model = LinearRegression(fit_intercept=True)
    ols_model.fit(X_scaled[train_mask], y[train_mask])
    y_pred = ols_model.predict(X_scaled)

    def get_diagnostics(actual, pred, mask):
        if mask.sum() == 0:
            return None
        return calculate_statistical_diagnostics(
            actual[mask], pred[mask], feature_cols,
            ols_model.coef_, X_scaled[train_mask], y[train_mask], train_mask.sum()
        )

    diagnostics = {
        'jump_periods': get_diagnostics(y, y_pred, test_mask & jump_mask),
        'normal_periods': get_diagnostics(y, y_pred, test_mask & ~jump_mask),
        'overall': get_diagnostics(y, y_pred, test_mask)
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'ols_baseline_diagnostics_{data_version}.json'
    with open(output_file, 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)

    create_statistical_diagnostics_summary(diagnostics, data_version, output_dir)
    return diagnostics


def run_random_forest_baseline(data_path: str, data_version: str = 'v1.1',
                                output_dir: str = 'results/thesis_v2',
                                n_estimators: int = 100, max_depth: int = 10):
    """Run Random Forest baseline with jump-focused evaluation."""
    print("=" * 80)
    print(f"RANDOM FOREST BASELINE ({data_version})")
    print("=" * 80)

    df, df_masks = _load_and_prepare_data(data_path, output_dir, data_version)

    feature_cols = [
        'dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'
    ]

    valid_mask = df[feature_cols + ['dvol_change']].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    df_masks_clean = df_masks[valid_mask].copy()

    n, train_size, train_mask, test_mask, jump_mask = _create_splits(df_clean, df_masks_clean)

    X = df_clean[feature_cols].values
    y = df_clean['dvol_change'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_scaled[train_mask], y[train_mask])
    y_pred = rf_model.predict(X_scaled)

    results = {
        'model_specification': {
            'model_type': 'random_forest',
            'target': 'next_dvol_change',
            'features': feature_cols,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'random_state': 42
            },
            'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_.tolist()))
        },
        'data_info': {
            'total_observations': int(n),
            'training_samples': int(train_mask.sum()),
            'testing_samples': int(test_mask.sum()),
            'jump_periods_test': int((test_mask & jump_mask).sum())
        },
        'performance_metrics': {
            'training': {
                'jump_periods': _calculate_metrics(y, y_pred, train_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, train_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, train_mask)
            },
            'testing': {
                'jump_periods': _calculate_metrics(y, y_pred, test_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, test_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, test_mask)
            }
        }
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'random_forest_baseline_{data_version}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_xgboost_baseline(data_path: str, data_version: str = 'v1.1',
                         output_dir: str = 'results/thesis_v2',
                         n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
    """Run XGBoost baseline with jump-focused evaluation."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("Error: xgboost not installed. Run: pip install xgboost")
        return None

    print("=" * 80)
    print(f"XGBOOST BASELINE ({data_version})")
    print("=" * 80)

    df, df_masks = _load_and_prepare_data(data_path, output_dir, data_version)

    feature_cols = [
        'dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'
    ]

    valid_mask = df[feature_cols + ['dvol_change']].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    df_masks_clean = df_masks[valid_mask].copy()

    n, train_size, train_mask, test_mask, jump_mask = _create_splits(df_clean, df_masks_clean)

    X = df_clean[feature_cols].values
    y = df_clean['dvol_change'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb_model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_scaled[train_mask], y[train_mask])
    y_pred = xgb_model.predict(X_scaled)

    results = {
        'model_specification': {
            'model_type': 'xgboost',
            'target': 'next_dvol_change',
            'features': feature_cols,
            'hyperparameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'feature_importance': dict(zip(feature_cols, xgb_model.feature_importances_.tolist()))
        },
        'data_info': {
            'total_observations': int(n),
            'training_samples': int(train_mask.sum()),
            'testing_samples': int(test_mask.sum()),
            'jump_periods_test': int((test_mask & jump_mask).sum())
        },
        'performance_metrics': {
            'training': {
                'jump_periods': _calculate_metrics(y, y_pred, train_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, train_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, train_mask)
            },
            'testing': {
                'jump_periods': _calculate_metrics(y, y_pred, test_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, test_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, test_mask)
            }
        }
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'xgboost_baseline_{data_version}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_har_rv_volatility_focused(data_path: str, data_version: str = 'v1.1',
                                   output_dir: str = 'results/thesis_v2'):
    """Run HAR-RV with only volatility lag features."""
    df, df_masks = _load_and_prepare_data(data_path, output_dir, data_version)

    feature_cols = ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d']

    valid_mask = df[feature_cols + ['dvol_change']].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    df_masks_clean = df_masks[valid_mask].copy()

    n, train_size, train_mask, test_mask, jump_mask = _create_splits(df_clean, df_masks_clean)

    X = df_clean[feature_cols].values
    y = df_clean['dvol_change'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_scaled[train_mask], y[train_mask])
    y_pred = model.predict(X_scaled)

    results = {
        'model_specification': {
            'model_type': 'har_rv_volatility_focused',
            'target': 'next_dvol_change',
            'features': feature_cols,
            'coefficients': dict(zip(feature_cols, model.coef_.tolist())),
            'intercept': float(model.intercept_)
        },
        'data_info': {
            'total_observations': int(n),
            'training_samples': int(train_mask.sum()),
            'testing_samples': int(test_mask.sum()),
            'jump_periods_test': int((test_mask & jump_mask).sum())
        },
        'performance_metrics': {
            'training': {
                'jump_periods': _calculate_metrics(y, y_pred, train_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, train_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, train_mask)
            },
            'testing': {
                'jump_periods': _calculate_metrics(y, y_pred, test_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, test_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, test_mask)
            }
        }
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'har_rv_volatility_focused_{data_version}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_har_rv_comprehensive(data_path: str, data_version: str = 'v1.1',
                             output_dir: str = 'results/thesis_v2'):
    """Run HAR-RV with all ML features."""
    df, df_masks = _load_and_prepare_data(data_path, output_dir, data_version)

    feature_cols = [
        'dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'
    ]

    valid_mask = df[feature_cols + ['dvol_change']].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    df_masks_clean = df_masks[valid_mask].copy()

    n, train_size, train_mask, test_mask, jump_mask = _create_splits(df_clean, df_masks_clean)

    X = df_clean[feature_cols].values
    y = df_clean['dvol_change'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_scaled[train_mask], y[train_mask])
    y_pred = model.predict(X_scaled)

    results = {
        'model_specification': {
            'model_type': 'har_rv_comprehensive',
            'target': 'next_dvol_change',
            'features': feature_cols,
            'coefficients': dict(zip(feature_cols, model.coef_.tolist())),
            'intercept': float(model.intercept_)
        },
        'data_info': {
            'total_observations': int(n),
            'training_samples': int(train_mask.sum()),
            'testing_samples': int(test_mask.sum()),
            'jump_periods_test': int((test_mask & jump_mask).sum())
        },
        'performance_metrics': {
            'training': {
                'jump_periods': _calculate_metrics(y, y_pred, train_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, train_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, train_mask)
            },
            'testing': {
                'jump_periods': _calculate_metrics(y, y_pred, test_mask & jump_mask),
                'normal_periods': _calculate_metrics(y, y_pred, test_mask & ~jump_mask),
                'overall': _calculate_metrics(y, y_pred, test_mask)
            }
        }
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'har_rv_comprehensive_{data_version}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_comprehensive_baseline_comparison(data_path: str, data_version: str = 'v1.1',
                                          output_dir: str = 'results/thesis_v2'):
    """Run all baseline models and create comparison table visualization."""
    from .visualization import create_baseline_comparison_table

    print("=" * 80)
    print(f"COMPREHENSIVE BASELINE COMPARISON ({data_version})")
    print("=" * 80)

    results = {}

    ols_results = run_phase1_baseline_analysis(data_path, data_version, output_dir)
    if ols_results:
        results['OLS (All Features)'] = ols_results

    har_rv_vol_results = run_har_rv_volatility_focused(data_path, data_version, output_dir)
    if har_rv_vol_results:
        results['HAR-RV (Volatility Lags)'] = har_rv_vol_results

    har_rv_comp_results = run_har_rv_comprehensive(data_path, data_version, output_dir)
    if har_rv_comp_results:
        results['HAR-RV (All Features)'] = har_rv_comp_results

    rf_results = run_random_forest_baseline(data_path, data_version, output_dir)
    if rf_results:
        results['Random Forest'] = rf_results

    xgb_results = run_xgboost_baseline(data_path, data_version, output_dir)
    if xgb_results:
        results['XGBoost'] = xgb_results

    create_baseline_comparison_table(results, data_version, output_dir)
    return results
