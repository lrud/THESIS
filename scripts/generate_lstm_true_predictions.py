#!/usr/bin/env python3
"""
Generate TRUE LSTM predictions from trained model checkpoint.
This script loads the actual LSTM model and generates real predictions (not synthetic data).
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

# Add paths for imports
sys.path.append('/home/lrud1314/PROJECTS_WORKING/THESIS 2025/scripts/modeling')
sys.path.append('/home/lrud1314/PROJECTS_WORKING/THESIS 2025/cli/config')

from model import LSTM_DVOL
from data_loader_unified import create_unified_dataloaders
from feature_configs import get_feature_config

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model configuration (from training results)
    config = {
        'hidden_size': 512,
        'num_layers': 7,
        'dropout': 0.5,
        'sequence_length': 24,
        'window_size': 720,
        'batch_size': 32
    }

    # Get feature configuration
    model_type = 'market_lags'
    feature_config = get_feature_config(model_type)
    input_size = feature_config['input_size']

    # Create model
    model = LSTM_DVOL(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    # Load checkpoint
    checkpoint_path = '/home/lrud1314/PROJECTS_WORKING/THESIS 2025/models/512x7_market_lags_market_lags_best.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded from: {checkpoint_path}")
    print(f"Parameters: {param_count:,}")
    print(f"Architecture: {config['hidden_size']} hidden units, {config['num_layers']} layers")

    # Load test data using unified dataloader
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_unified_dataloaders(
        data_path='/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv',
        feature_set=model_type,
        sequence_length=config['sequence_length'],
        window_size=config['window_size'],
        batch_size=config['batch_size']
    )

    print(f"\nData loaded:")
    print(f"  Test samples: {len(test_ds):,}")

    # Generate predictions on test set
    all_preds = []
    all_targets = []
    all_stats = []

    with torch.no_grad():
        for X_batch, y_batch, w_batch, stats_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            all_stats.append(stats_batch.cpu().numpy())

    # Concatenate all batches
    preds_norm = np.concatenate(all_preds, axis=0)
    targets_norm = np.concatenate(all_targets, axis=0)
    stats = np.concatenate(all_stats, axis=0)

    # Inverse transform to get actual DVOL values
    preds_actual = test_ds.inverse_transform_target(preds_norm, stats)
    targets_actual = test_ds.inverse_transform_target(targets_norm, stats)

    print(f"\nPredictions generated: {len(preds_actual):,} samples")

    # Calculate metrics
    mse = np.mean((targets_actual - preds_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(targets_actual - preds_actual))
    ss_res = np.sum((targets_actual - preds_actual) ** 2)
    ss_tot = np.sum((targets_actual - np.mean(targets_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"\nActual Model Performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # Get timestamps for test set
    data_path = '/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv'
    df_full = pd.read_csv(data_path)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    df_full = df_full.sort_values('timestamp').reset_index(drop=True)

    # The test set comes from the last 20% of data
    n_total = len(df_full)
    n_train = int(n_total * 0.60)
    n_val = int(n_total * 0.20)

    # Account for window_size and sequence_length
    effective_start = n_train + n_val + config['window_size'] + config['sequence_length'] + 1
    test_timestamps = df_full['timestamp'].iloc[effective_start:effective_start + len(preds_actual)].values

    print(f"\nTest period:")
    print(f"  Start: {test_timestamps[0]}")
    print(f"  End: {test_timestamps[-1]}")

    # Create visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (19.2, 9)
    plt.rcParams['font.size'] = 12

    CB_PALETTE = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442']
    OUTPUT_DIR = Path('results/visualizations/twitter_thread')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(19.2, 9))

    # Plot actual values
    ax.plot(test_timestamps, targets_actual.flatten(), '.', color='black',
            linewidth=1, markersize=2, label='Actual DVOL', alpha=0.5)

    # Plot predictions
    ax.plot(test_timestamps, preds_actual.flatten(), '-', color=CB_PALETTE[3],
            linewidth=2.5, label=f'LSTM market_lags (512x7) Prediction', alpha=0.9)

    # Confidence interval
    ci_upper = preds_actual.flatten() + 1.96 * rmse
    ci_lower = preds_actual.flatten() - 1.96 * rmse
    ax.fill_between(test_timestamps, ci_lower, ci_upper,
                    color=CB_PALETTE[3], alpha=0.25, label='95% CI')

    # Formatting
    ax.set_ylabel('DVOL Level', fontsize=13, fontweight='bold')
    ax.set_title(f'LSTM market_lags (512x7) - TRUE Model Predictions (R²={r2:.4f}, RMSE={rmse:.2f})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'viz13_lstm_true_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("TRUE LSTM PREDICTION SUMMARY")
    print("="*80)
    print(f"\nActual DVOL (Test Set):")
    print(f"  Mean: {np.mean(targets_actual):.2f}")
    print(f"  Std: {np.std(targets_actual):.2f}")
    print(f"  Range: [{np.min(targets_actual):.2f}, {np.max(targets_actual):.2f}]")
    print(f"\nPredicted DVOL (Test Set):")
    print(f"  Mean: {np.mean(preds_actual):.2f}")
    print(f"  Std: {np.std(preds_actual):.2f}")
    print(f"  Range: [{np.min(preds_actual):.2f}, {np.max(preds_actual):.2f}]")
    print(f"\nPrediction Error:")
    print(f"  Mean Error: {np.mean(preds_actual - targets_actual):.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print("="*80)

    return r2, rmse, mae

if __name__ == '__main__':
    main()
