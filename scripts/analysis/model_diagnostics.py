"""
Diagnostic script to identify issues with LSTM model predictions.
Investigates why predictions appear as a straight line.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modeling'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler

from data_loader import load_features_data, create_sequences, split_temporal_data, normalize_features
from model import LSTM_DVOL

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def check_data_loading():
    """Check raw data loading."""
    print("="*80)
    print("1. CHECKING RAW DATA LOADING")
    print("="*80)
    
    df = load_features_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    print(f"\nBasic statistics:")
    print(df.describe())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df


def check_sequence_creation(df):
    """Check sequence creation logic."""
    print("\n" + "="*80)
    print("2. CHECKING SEQUENCE CREATION")
    print("="*80)
    
    feature_cols = [
        'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'
    ]
    target_col = 'dvol'
    
    X, y, timestamps = create_sequences(df, feature_cols, target_col, 
                                        sequence_length=24, forecast_horizon=24)
    
    print(f"\nSequence shapes:")
    print(f"  X: {X.shape} (samples, timesteps, features)")
    print(f"  y: {y.shape} (samples, 1)")
    print(f"  timestamps: {len(timestamps)}")
    
    print(f"\nFirst sequence:")
    print(f"  X[0] shape: {X[0].shape}")
    print(f"  y[0]: {y[0]}")
    print(f"  Timestamp: {timestamps[0]}")
    
    print(f"\nX[0] statistics (first sequence):")
    print(f"  Mean per feature: {X[0].mean(axis=0)}")
    print(f"  Std per feature: {X[0].std(axis=0)}")
    
    print(f"\nTarget (y) statistics BEFORE normalization:")
    print(f"  Mean: {y.mean():.4f}")
    print(f"  Std: {y.std():.4f}")
    print(f"  Min: {y.min():.4f}")
    print(f"  Max: {y.max():.4f}")
    print(f"  First 10 values: {y[:10].flatten()}")
    
    return X, y, timestamps


def check_normalization(X, y):
    """Check normalization behavior."""
    print("\n" + "="*80)
    print("3. CHECKING NORMALIZATION")
    print("="*80)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal_data(
        X, y, train_ratio=0.6, val_ratio=0.2
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val: {X_val.shape[0]}")
    print(f"  Test: {X_test.shape[0]}")
    
    print(f"\nBEFORE normalization:")
    print(f"  y_train - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    print(f"  y_val - Mean: {y_val.mean():.4f}, Std: {y_val.std():.4f}")
    print(f"  y_test - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
    
    # Normalize
    X_train_norm, X_val_norm, X_test_norm, scaler_X = normalize_features(
        X_train, X_val, X_test
    )
    
    # Normalize target
    scaler_y = StandardScaler()
    y_train_norm = scaler_y.fit_transform(y_train)
    y_val_norm = scaler_y.transform(y_val)
    y_test_norm = scaler_y.transform(y_test)
    
    print(f"\nAFTER normalization:")
    print(f"  y_train - Mean: {y_train_norm.mean():.4f}, Std: {y_train_norm.std():.4f}")
    print(f"  y_val - Mean: {y_val_norm.mean():.4f}, Std: {y_val_norm.std():.4f}")
    print(f"  y_test - Mean: {y_test_norm.mean():.4f}, Std: {y_test_norm.std():.4f}")
    
    print(f"\nScaler parameters:")
    print(f"  scaler_y.mean_: {scaler_y.mean_}")
    print(f"  scaler_y.scale_: {scaler_y.scale_}")
    
    print(f"\nFirst 20 normalized y_train values:")
    print(y_train_norm[:20].flatten())
    
    return X_train_norm, y_train_norm, X_test_norm, y_test_norm, scaler_y


def check_model_predictions(X_train, y_train, X_test, y_test, scaler_y):
    """Check what the model is actually predicting."""
    print("\n" + "="*80)
    print("4. CHECKING MODEL PREDICTIONS")
    print("="*80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM_DVOL(input_size=7, hidden_size=128, num_layers=2, dropout=0.3, l2_reg=1e-4)
    model.load_state_dict(torch.load('models/lstm_baseline_best.pth'))
    model = model.to(device)
    model.eval()
    
    # Get predictions on a small subset
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test[:100]).to(device)
        y_pred_norm = model(X_test_tensor).cpu().numpy()
    
    print(f"\nNormalized predictions (first 20):")
    print(y_pred_norm[:20].flatten())
    
    print(f"\nNormalized prediction statistics:")
    print(f"  Mean: {y_pred_norm.mean():.4f}")
    print(f"  Std: {y_pred_norm.std():.4f}")
    print(f"  Min: {y_pred_norm.min():.4f}")
    print(f"  Max: {y_pred_norm.max():.4f}")
    print(f"  Unique values: {len(np.unique(y_pred_norm))}")
    
    # Inverse transform
    y_pred_orig = scaler_y.inverse_transform(y_pred_norm)
    y_test_orig = scaler_y.inverse_transform(y_test[:100])
    
    print(f"\nOriginal scale predictions (first 20):")
    print(y_pred_orig[:20].flatten())
    
    print(f"\nOriginal scale prediction statistics:")
    print(f"  Mean: {y_pred_orig.mean():.4f}")
    print(f"  Std: {y_pred_orig.std():.4f}")
    print(f"  Min: {y_pred_orig.min():.4f}")
    print(f"  Max: {y_pred_orig.max():.4f}")
    print(f"  Unique values: {len(np.unique(y_pred_orig))}")
    
    print(f"\nActual test values (first 20):")
    print(y_test_orig[:20].flatten())
    
    print(f"\nActual test statistics:")
    print(f"  Mean: {y_test_orig.mean():.4f}")
    print(f"  Std: {y_test_orig.std():.4f}")
    print(f"  Min: {y_test_orig.min():.4f}")
    print(f"  Max: {y_test_orig.max():.4f}")
    
    # Check if predictions are nearly constant
    if y_pred_orig.std() < 1.0:
        print(f"\n⚠️  WARNING: Predictions have very low variance (std={y_pred_orig.std():.4f})")
        print(f"⚠️  Model is essentially predicting a constant value!")
    
    return y_pred_orig, y_test_orig


def visualize_diagnostics(y_pred, y_test):
    """Create diagnostic visualizations."""
    print("\n" + "="*80)
    print("5. CREATING DIAGNOSTIC VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Predictions vs Actuals (time series)
    axes[0, 0].plot(y_test, label='Actual', alpha=0.7, linewidth=2)
    axes[0, 0].plot(y_pred, label='Predicted', alpha=0.7, linewidth=2)
    axes[0, 0].set_title('Predictions vs Actuals (Time Series)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('DVOL')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Scatter plot
    axes[0, 1].scatter(y_test, y_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0, 1].set_title('Actual vs Predicted (Scatter)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Actual DVOL')
    axes[0, 1].set_ylabel('Predicted DVOL')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Residuals
    residuals = y_test - y_pred
    axes[0, 2].plot(residuals, alpha=0.7)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 2].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('Residual (Actual - Predicted)')
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Residual distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Prediction distribution
    axes[1, 1].hist(y_test, bins=50, alpha=0.7, label='Actual', edgecolor='black')
    axes[1, 1].hist(y_pred, bins=50, alpha=0.7, label='Predicted', edgecolor='black')
    axes[1, 1].set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('DVOL')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/model_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/visualizations/model_diagnostics.png")
    
    # Additional plot: Zoomed in predictions
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.plot(y_test[:50], label='Actual', alpha=0.7, linewidth=3, marker='o', markersize=5)
    ax.plot(y_pred[:50], label='Predicted', alpha=0.7, linewidth=3, marker='s', markersize=5)
    ax.set_title('Predictions vs Actuals (First 50 Samples - ZOOMED)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('DVOL', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/visualizations/predictions_zoomed.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/visualizations/predictions_zoomed.png")


def main():
    """Run all diagnostics."""
    print("\n" + "="*80)
    print("LSTM MODEL DIAGNOSTIC ANALYSIS")
    print("="*80 + "\n")
    
    # Step 1: Check data loading
    df = check_data_loading()
    
    # Step 2: Check sequence creation
    X, y, timestamps = check_sequence_creation(df)
    
    # Step 3: Check normalization
    X_train, y_train, X_test, y_test, scaler_y = check_normalization(X, y)
    
    # Step 4: Check model predictions
    y_pred, y_test_subset = check_model_predictions(X_train, y_train, X_test, y_test, scaler_y)
    
    # Step 5: Visualize
    visualize_diagnostics(y_pred, y_test_subset)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("="*80)
    print("\nCheck the following files:")
    print("  - results/visualizations/model_diagnostics.png")
    print("  - results/visualizations/predictions_zoomed.png")
    print("\nLook for:")
    print("  1. Are predictions nearly constant?")
    print("  2. Are residuals systematic (not random)?")
    print("  3. Is the prediction distribution much narrower than actual?")
    print("  4. Are there issues with data scaling?")


if __name__ == '__main__':
    main()
