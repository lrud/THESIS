#!/usr/bin/env python3
"""
Tail Risk Analysis and Naive Benchmark Comparison for LSTM DVOL Forecasting

This script performs two complementary analyses:

1. VALUE-AT-RISK (VaR) BACKTESTING
   Tests if the model underestimates tail risk (dangerous for trading decisions).
   Computes VaR at multiple confidence levels and validates coverage.

2. NAIVE BENCHMARK COMPARISON
   Compares LSTM performance against simple baseline models:
   - Persistence: tomorrow = today
   - Mean: tomorrow = historical mean
   - Random Walk: random perturbation around today

All values are computed dynamically from data - NO HARDCODED VALUES.

Author: Thesis V2 Research
Date: 2026-01-02
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.modeling.model import LSTM_DVOL
from scripts.modeling.data_loader_jump_aware import JumpAwareLSTMDataset


def load_trained_model(model_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Dict]:
    """
    Load trained LSTM model and its configuration.

    Args:
        model_path: Path to .pth checkpoint file
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded model in eval mode
        config: Model configuration from results JSON
    """
    print("=" * 80)
    print("LOADING TRAINED MODEL")
    print("=" * 80)

    # Load state dict from checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Find corresponding results JSON file for config
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results" / "cli_training"

    # Search all date subfolders for matching result
    config = None
    for date_folder in sorted(results_dir.iterdir(), reverse=True):
        if date_folder.is_dir():
            for json_file in date_folder.glob("*.json"):
                try:
                    with open(json_file) as f:
                        result_data = json.load(f)
                    # Check if this result matches our model
                    if result_data.get('model_path', '') == model_path or \
                       model_path.name in result_data.get('model_path', ''):
                        config = result_data.get('config', {})
                        print(f"\nFound config in: {json_file}")
                        break
                except:
                    pass
            if config:
                break

    # If no config found in results, use defaults based on model name
    if config is None:
        print(f"\nWarning: No config found in results, using defaults from model name")
        config = {
            'hidden_size': 512,
            'num_layers': 7,
            'dropout': 0.5,
            'input_size': 11  # Jump-aware features
        }

    print(f"\nModel configuration:")
    print(f"  Hidden size: {config.get('hidden_size', 'N/A')}")
    print(f"  Num layers: {config.get('num_layers', 'N/A')}")
    print(f"  Dropout: {config.get('dropout', 'N/A')}")
    print(f"  Input size: {config.get('input_size', 'N/A')}")

    # Reconstruct model
    input_size = config.get('input_size', 11)
    hidden_size = config.get('hidden_size', 512)
    num_layers = config.get('num_layers', 7)
    dropout = config.get('dropout', 0.5)

    model = LSTM_DVOL(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        l2_reg=0
    ).to(device)

    # Handle DataParallel wrapper
    state_dict = checkpoint
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    # Calculate parameters
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nModel loaded successfully from: {model_path}")
    print(f"Device: {device}")
    print(f"Total parameters: {n_params:,}")

    return model, config


def load_test_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load test set data and actual DVOL values.

    Returns:
        y_true: Actual DVOL values (original scale)
        timestamps: Test set timestamps
        jump_indicators: Jump indicators for test set
    """
    print("\n" + "=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)

    # Load full dataset
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Apply same split as training (60/20/20)
    n = len(df)
    train_size = int(n * 0.6)
    val_size = int(n * 0.2)
    test_start_idx = train_size + val_size

    # Extract test set
    test_df = df.iloc[test_start_idx:].copy()

    # Drop NaN rows (same as training)
    feature_cols = [
        'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv',
        'dvol_rv_spread', 'jump_indicator', 'jump_magnitude',
        'days_since_jump', 'jump_cluster_7d'
    ]

    test_df = test_df.dropna(subset=feature_cols + ['dvol']).reset_index(drop=True)

    # Account for sequence_length and rolling_window
    sequence_length = 24
    window_size = 720
    valid_start_idx = window_size + sequence_length

    if valid_start_idx >= len(test_df):
        raise ValueError(f"Test set too small: {len(test_df)} < {valid_start_idx}")

    y_true = test_df['dvol'].iloc[valid_start_idx:].values
    timestamps = test_df['timestamp'].iloc[valid_start_idx:].values
    jump_indicators = test_df['jump_indicator'].iloc[valid_start_idx:].values

    print(f"\nTest set:")
    print(f"  Period: {timestamps[0]} to {timestamps[-1]}")
    print(f"  Samples: {len(y_true):,}")
    print(f"  DVOL range: {y_true.min():.2f} - {y_true.max():.2f}")
    print(f"  DVOL mean: {y_true.mean():.2f} +/- {y_true.std():.2f}")
    print(f"  Jump samples: {jump_indicators.sum():,} ({jump_indicators.mean()*100:.1f}%)")

    return y_true, timestamps, jump_indicators


def get_model_predictions(model: torch.nn.Module, data_path: str,
                         device: str = 'cuda') -> np.ndarray:
    """
    Generate model predictions on test set.

    Args:
        model: Trained LSTM model
        data_path: Path to data file
        device: 'cuda' or 'cpu'

    Returns:
        y_pred: Model predictions (original scale)
    """
    print("\n" + "=" * 80)
    print("GENERATING MODEL PREDICTIONS")
    print("=" * 80)

    # Load full dataset
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Apply same split
    n = len(df)
    train_size = int(n * 0.6)
    val_size = int(n * 0.2)

    train_data = df.iloc[:train_size].copy()
    val_data = df.iloc[train_size:train_size + val_size].copy()
    test_data = df.iloc[train_size + val_size:].copy()

    # Clean data
    feature_cols = [
        'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv',
        'dvol_rv_spread', 'jump_indicator', 'jump_magnitude',
        'days_since_jump', 'jump_cluster_7d'
    ]

    test_data = test_data.dropna(subset=feature_cols + ['dvol']).reset_index(drop=True)

    # Create test dataset
    test_dataset = JumpAwareLSTMDataset(
        test_data,
        sequence_length=24,
        window_size=720,
        mode='test'
    )

    # Generate predictions
    predictions = []
    rolling_stats_list = []

    with torch.no_grad():
        for X_batch, y_batch, w_batch, stats_batch in test_dataset:
            X_tensor = torch.FloatTensor(X_batch).unsqueeze(0).to(device)

            pred_normalized = model(X_tensor).cpu().numpy()[0, 0]
            stats = np.array(stats_batch)

            # Inverse transform
            mean = stats[0]
            std = stats[1]
            pred_original = pred_normalized * std + mean

            predictions.append(pred_original)
            rolling_stats_list.append(stats)

    y_pred = np.array(predictions)

    print(f"\nPredictions generated: {len(y_pred):,}")
    print(f"  Pred range: {y_pred.min():.2f} - {y_pred.max():.2f}")
    print(f"  Pred mean: {y_pred.mean():.2f} +/- {y_pred.std():.2f}")

    return y_pred


# ============================================================================
# VALUE-AT-RISK (VaR) ANALYSIS
# ============================================================================

def calculate_var(errors: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Value-at-Risk at confidence level (1-alpha).

    VaR is the (1-alpha)-quantile of losses.
    For volatility forecasting, we use absolute errors.

    Args:
        errors: Absolute errors |y_true - y_pred|
        alpha: Significance level (0.05 = 95% VaR, 0.01 = 99% VaR)

    Returns:
        VaR value at (1-alpha) confidence level
    """
    return np.percentile(errors, (1 - alpha) * 100)


def calculate_var_backtest(y_true: np.ndarray, y_pred: np.ndarray,
                          alpha_levels: list = [0.05, 0.01]) -> Dict:
    """
    Perform VaR backtesting at multiple confidence levels.

    Tests whether the model's prediction errors exceed expected VaR thresholds
    more often than theoretically permissible.

    Args:
        y_true: Actual DVOL values
        y_pred: Predicted DVOL values
        alpha_levels: List of significance levels to test

    Returns:
        Dictionary with VaR metrics for each alpha level
    """
    print("\n" + "=" * 80)
    print("VALUE-AT-RISK (VaR) BACKTESTING")
    print("=" * 80)

    errors = np.abs(y_true - y_pred)
    error_std = errors.std()
    error_mean = errors.mean()

    print(f"\nError statistics:")
    print(f"  Mean absolute error: {error_mean:.4f}")
    print(f"  Std of errors: {error_std:.4f}")
    print(f"  Max error: {errors.max():.4f}")
    print(f"  Min error: {errors.min():.4f}")

    results = {}

    for alpha in alpha_levels:
        var_value = calculate_var(errors, alpha)

        # Count exceedances (errors > VaR)
        exceedances = errors > var_value
        exceedance_count = exceedances.sum()
        coverage = exceedance_count / len(errors)

        # Expected exceedances for valid VaR
        expected_exceedances = alpha * len(errors)

        # Kupiec test approximation (unconditional coverage)
        # Tests if observed exceedances differ significantly from expected
        lr_uc = (
            2 * (exceedance_count * np.log(coverage) if coverage > 0 else 0)
            - 2 * (exceedance_count * np.log(alpha) if alpha > 0 else 0)
            + 2 * ((len(errors) - exceedance_count) * np.log(1 - coverage) if coverage < 1 else 0)
            - 2 * ((len(errors) - exceedance_count) * np.log(1 - alpha) if alpha < 1 else 0)
        )

        # Critical value at 95% confidence (chi-square with 1 dof)
        critical_value = 3.841
        reject_null = lr_uc > critical_value

        results[f'alpha_{alpha}'] = {
            'alpha': alpha,
            'confidence_level': f"{(1-alpha)*100:.0f}%",
            'var_value': float(var_value),
            'var_pct_of_mean': float(var_value / y_true.mean() * 100),
            'exceedance_count': int(exceedance_count),
            'exceedance_rate': float(coverage),
            'expected_exceedance_rate': alpha,
            'expected_count': int(expected_exceedances),
            'lr_uc_statistic': float(lr_uc),
            'reject_null': bool(reject_null),
            'underestimates_risk': bool(coverage > alpha)
        }

        print(f"\n{(1-alpha)*100:.0f}% VaR:")
        print(f"  VaR value: {var_value:.4f} ({var_value/y_true.mean()*100:.1f}% of mean DVOL)")
        print(f"  Exceedances: {exceedance_count}/{len(errors)} ({coverage*100:.2f}%)")
        print(f"  Expected: {expected_exceedances:.0f} ({alpha*100:.1f}%)")
        print(f"  Underestimates risk: {coverage > alpha}")
        if reject_null:
            print(f"  WARNING: Kupiec test REJECTS valid VaR model (LR={lr_uc:.2f} > {critical_value})")

    return results


def calculate_tail_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          jump_indicators: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate tail risk metrics for extreme errors.

    Focuses on worst-case forecasting performance.

    Args:
        y_true: Actual DVOL values
        y_pred: Predicted DVOL values
        jump_indicators: Optional binary array indicating jump periods

    Returns:
        Dictionary with tail metrics
    """
    print("\n" + "=" * 80)
    print("TAIL RISK METRICS")
    print("=" * 80)

    errors = np.abs(y_true - y_pred)
    pct_errors = np.abs((y_true - y_pred) / y_true) * 100

    results = {
        'p95_error': float(np.percentile(errors, 95)),
        'p99_error': float(np.percentile(errors, 99)),
        'p95_pct_error': float(np.percentile(pct_errors, 95)),
        'p99_pct_error': float(np.percentile(pct_errors, 99)),
        'max_error': float(errors.max()),
        'max_pct_error': float(pct_errors.max()),
        'worst_10_avg_error': float(np.mean(np.sort(errors)[-10:])),
        'tail_skewness': float(errors.mean() / np.median(errors)),
    }

    print(f"\nAbsolute errors:")
    print(f"  95th percentile: {results['p95_error']:.4f}")
    print(f"  99th percentile: {results['p99_error']:.4f}")
    print(f"  Maximum: {results['max_error']:.4f}")
    print(f"  Worst 10 average: {results['worst_10_avg_error']:.4f}")

    print(f"\nPercentage errors:")
    print(f"  95th percentile: {results['p95_pct_error']:.2f}%")
    print(f"  99th percentile: {results['p99_pct_error']:.2f}%")
    print(f"  Maximum: {results['max_pct_error']:.2f}%")

    print(f"\nTail characteristics:")
    print(f"  Tail skewness: {results['tail_skewness']:.2f} (>1 = heavy tail)")

    # Jump-specific tail metrics
    if jump_indicators is not None and jump_indicators.sum() > 0:
        jump_errors = errors[jump_indicators > 0]
        normal_errors = errors[jump_indicators == 0]

        results['jump_avg_error'] = float(jump_errors.mean())
        results['normal_avg_error'] = float(normal_errors.mean())
        results['jump_error_ratio'] = float(jump_errors.mean() / normal_errors.mean())

        print(f"\nJump vs Normal periods:")
        print(f"  Jump avg error: {results['jump_avg_error']:.4f}")
        print(f"  Normal avg error: {results['normal_avg_error']:.4f}")
        print(f"  Ratio (jump/normal): {results['jump_error_ratio']:.2f}x")

    return results


# ============================================================================
# NAIVE BENCHMARK COMPARISON
# ============================================================================

def naive_penchmark(y_train: np.ndarray, n_test: int,
                   method: str = 'persistence') -> np.ndarray:
    """
    Generate naive baseline forecasts.

    Args:
        y_train: Training set DVOL values
        n_test: Number of test samples to forecast
        method: 'persistence', 'mean', or 'random_walk'

    Returns:
        forecasts: Array of naive forecasts
    """
    if method == 'persistence':
        # Tomorrow equals today
        last_value = y_train[-1]
        return np.full(n_test, last_value)

    elif method == 'mean':
        # Tomorrow equals historical mean
        historical_mean = y_train.mean()
        return np.full(n_test, historical_mean)

    elif method == 'random_walk':
        # Random perturbation around last value
        last_value = y_train[-1]
        std = y_train.std()
        return last_value + np.random.randn(n_test) * std * 0.1

    else:
        raise ValueError(f"Unknown naive method: {method}")


def compare_to_benchmarks(y_true: np.ndarray, y_pred_lstm: np.ndarray,
                         y_train: np.ndarray, seed: int = 42) -> Dict:
    """
    Compare LSTM performance against naive benchmarks.

    Args:
        y_true: Actual DVOL values (test set)
        y_pred_lstm: LSTM predictions
        y_train: Training set DVOL values (for naive forecasts)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with benchmark comparison results
    """
    print("\n" + "=" * 80)
    print("NAIVE BENCHMARK COMPARISON")
    print("=" * 80)

    np.random.seed(seed)
    n_test = len(y_true)

    # Calculate LSTM metrics
    lstm_mae = np.mean(np.abs(y_true - y_pred_lstm))
    lstm_rmse = np.sqrt(np.mean((y_true - y_pred_lstm)**2))
    lstm_r2 = 1 - np.sum((y_true - y_pred_lstm)**2) / np.sum((y_true - y_true.mean())**2)

    print(f"\nLSTM Performance:")
    print(f"  MAE: {lstm_mae:.4f}")
    print(f"  RMSE: {lstm_rmse:.4f}")
    print(f"  R²: {lstm_r2:.4f}")

    results = {
        'lstm': {
            'mae': float(lstm_mae),
            'rmse': float(lstm_rmse),
            'r2': float(lstm_r2)
        }
    }

    # Test each naive benchmark
    for method in ['persistence', 'mean', 'random_walk']:
        y_pred_naive = naive_penchmark(y_train, n_test, method)

        mae = np.mean(np.abs(y_true - y_pred_naive))
        rmse = np.sqrt(np.mean((y_true - y_pred_naive)**2))
        r2 = 1 - np.sum((y_true - y_pred_naive)**2) / np.sum((y_true - y_true.mean())**2)

        improvement_mae = (mae - lstm_mae) / mae * 100
        improvement_rmse = (rmse - lstm_rmse) / rmse * 100

        results[method] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mae_improvement_vs_lstm': float(improvement_mae),
            'rmse_improvement_vs_lstm': float(improvement_rmse)
        }

        print(f"\n{method.upper()}:")
        print(f"  MAE: {mae:.4f} (LSTM improvement: {improvement_mae:+.1f}%)")
        print(f"  RMSE: {rmse:.4f} (LSTM improvement: {improvement_rmse:+.1f}%)")
        print(f"  R²: {r2:.4f}")

    # LSTM vs Naive summary
    lstm_beats_persistence = lstm_mae < results['persistence']['mae']
    lstm_beats_mean = lstm_mae < results['mean']['mae']

    print(f"\nSummary:")
    print(f"  LSTM beats persistence: {lstm_beats_persistence}")
    print(f"  LSTM beats mean: {lstm_beats_mean}")

    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Execute complete tail risk and benchmark analysis."""

    # Paths - computed dynamically, no hardcoded values
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / "deep_512x7_jump_aware_best.pth"
    data_path = project_root / "data" / "processed" / "bitcoin_lstm_features_v1.1_complete_with_jumps.csv"
    output_path = project_root / "results" / "analysis" / "tail_risk_and_benchmarks.json"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Device detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "=" * 80)
    print("TAIL RISK AND BENCHMARK ANALYSIS")
    print(f"Model: {model_path.name}")
    print(f"Data: v1.1 complete")
    print("=" * 80)

    # Load model
    model, config = load_trained_model(str(model_path), device)

    # Load test data
    y_true, timestamps, jump_indicators = load_test_data(str(data_path))

    # Get predictions
    y_pred = get_model_predictions(model, str(data_path), device)

    # Get training data for naive benchmarks
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    train_size = int(n * 0.6)
    train_df = df.iloc[:train_size].dropna(subset=['dvol'])
    y_train = train_df['dvol'].values

    # Run analyses
    var_results = calculate_var_backtest(y_true, y_pred)
    tail_results = calculate_tail_metrics(y_true, y_pred, jump_indicators)
    benchmark_results = compare_to_benchmarks(y_true, y_pred, y_train)

    # Compile results
    output_results = {
        'model_path': str(model_path),
        'data_path': str(data_path),
        'test_period': {
            'start': str(timestamps[0]),
            'end': str(timestamps[-1]),
            'n_samples': len(y_true)
        },
        'var_backtesting': var_results,
        'tail_metrics': tail_results,
        'benchmark_comparison': benchmark_results,
        'summary': {
            'lstm_r2': float(benchmark_results['lstm']['r2']),
            'lstm_mae': float(benchmark_results['lstm']['mae']),
            'beats_persistence': bool(benchmark_results['lstm']['mae'] < benchmark_results['persistence']['mae']),
            'beats_mean': bool(benchmark_results['lstm']['mae'] < benchmark_results['mean']['mae']),
            'underestimates_tail_risk_5pct': bool(var_results['alpha_0.05']['underestimates_risk']),
            'underestimates_tail_risk_1pct': bool(var_results['alpha_0.01']['underestimates_risk']),
        }
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(output_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    # Print summary
    print("\nKEY FINDINGS:")
    print(f"  1. LSTM R²: {output_results['summary']['lstm_r2']:.4f}")
    print(f"  2. Beats persistence: {output_results['summary']['beats_persistence']}")
    print(f"  3. Beats mean: {output_results['summary']['beats_mean']}")
    print(f"  4. Underestimates 5% VaR: {output_results['summary']['underestimates_tail_risk_5pct']}")
    print(f"  5. Underestimates 1% VaR: {output_results['summary']['underestimates_tail_risk_1pct']}")

    return output_results


if __name__ == '__main__':
    main()
