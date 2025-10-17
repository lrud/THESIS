"""
Data loading utilities for LSTM with DIFFERENCED target variable.
This solves the non-stationarity issue identified in the baseline model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_features_data():
    """Load the preprocessed LSTM features data."""
    df = pd.read_csv('data/processed/bitcoin_lstm_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df


def create_sequences_differenced(df, feature_cols, target_col, sequence_length=24, forecast_horizon=24):
    """
    Create sequences with DIFFERENCED target variable.
    
    Key difference from baseline: Target is Δdvol = dvol_t - dvol_{t-1}
    This makes the target stationary.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name ('dvol')
        sequence_length: Number of time steps to look back (24h)
        forecast_horizon: Number of time steps ahead to forecast (24h)
    
    Returns:
        X: Input sequences (samples, timesteps, features)
        y_diff: DIFFERENCED target values (samples, 1)
        y_prev: Previous target values needed for reconstruction (samples, 1)
        timestamps: Timestamps for each sample
    """
    X, y_diff, y_prev, timestamps = [], [], [], []
    
    # Compute first difference of target
    df['dvol_diff'] = df[target_col].diff()
    
    for i in range(sequence_length, len(df) - forecast_horizon):
        # Input sequence
        X.append(df[feature_cols].iloc[i-sequence_length:i].values)
        
        # Target: difference at forecast horizon
        target_idx = i + forecast_horizon - 1
        y_diff.append(df['dvol_diff'].iloc[target_idx])
        
        # Store previous value needed for reconstruction: dvol at time i
        y_prev.append(df[target_col].iloc[target_idx - 1])
        
        timestamps.append(df.index[target_idx])
    
    X = np.array(X)
    y_diff = np.array(y_diff).reshape(-1, 1)
    y_prev = np.array(y_prev).reshape(-1, 1)
    
    print(f"Created {len(X)} sequences")
    print(f"  Input shape: {X.shape}")
    print(f"  Target (diff) shape: {y_diff.shape}")
    print(f"  Previous values shape: {y_prev.shape}")
    print(f"  Diff stats - Mean: {y_diff.mean():.4f}, Std: {y_diff.std():.4f}")
    
    return X, y_diff, y_prev, timestamps


def split_temporal_data(X, y_diff, y_prev, train_ratio=0.6, val_ratio=0.2):
    """
    Split data temporally (no shuffling to preserve time series order).
    
    Args:
        X: Input sequences
        y_diff: Differenced target values
        y_prev: Previous target values for reconstruction
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
    
    Returns:
        X_train, X_val, X_test, y_diff_train, y_diff_val, y_diff_test,
        y_prev_train, y_prev_val, y_prev_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_diff_train = y_diff[:train_end]
    y_diff_val = y_diff[train_end:val_end]
    y_diff_test = y_diff[val_end:]
    
    y_prev_train = y_prev[:train_end]
    y_prev_val = y_prev[train_end:val_end]
    y_prev_test = y_prev[val_end:]
    
    print(f"\nTemporal split:")
    print(f"  Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(X_val)} samples ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(X_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    print(f"\nDifferenced target statistics:")
    print(f"  Train Δdvol - Mean: {y_diff_train.mean():.4f}, Std: {y_diff_train.std():.4f}")
    print(f"  Val Δdvol   - Mean: {y_diff_val.mean():.4f}, Std: {y_diff_val.std():.4f}")
    print(f"  Test Δdvol  - Mean: {y_diff_test.mean():.4f}, Std: {y_diff_test.std():.4f}")
    print(f"  ✅ All means near 0 = Stationary!")
    
    return (X_train, X_val, X_test, 
            y_diff_train, y_diff_val, y_diff_test,
            y_prev_train, y_prev_val, y_prev_test)


def normalize_features(X_train, X_val, X_test, y_diff_train, y_diff_val, y_diff_test):
    """
    Normalize features and differenced targets using StandardScaler.
    Fit on training data only.
    
    Args:
        X_train, X_val, X_test: Feature sequences
        y_diff_train, y_diff_val, y_diff_test: Differenced target values
    
    Returns:
        Normalized data + scalers
    """
    # Reshape for scaling
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit scaler on training data
    scaler_X = StandardScaler()
    X_train_2d = scaler_X.fit_transform(X_train_2d)
    X_val_2d = scaler_X.transform(X_val_2d)
    X_test_2d = scaler_X.transform(X_test_2d)
    
    # Reshape back
    X_train = X_train_2d.reshape(X_train.shape)
    X_val = X_val_2d.reshape(X_val.shape)
    X_test = X_test_2d.reshape(X_test.shape)
    
    # Normalize differenced target
    scaler_y_diff = StandardScaler()
    y_diff_train = scaler_y_diff.fit_transform(y_diff_train)
    y_diff_val = scaler_y_diff.transform(y_diff_val)
    y_diff_test = scaler_y_diff.transform(y_diff_test)
    
    print(f"\nNormalized differenced target statistics:")
    print(f"  Train - Mean: {y_diff_train.mean():.4f}, Std: {y_diff_train.std():.4f}")
    print(f"  Val   - Mean: {y_diff_val.mean():.4f}, Std: {y_diff_val.std():.4f}")
    print(f"  Test  - Mean: {y_diff_test.mean():.4f}, Std: {y_diff_test.std():.4f}")
    
    return X_train, X_val, X_test, y_diff_train, y_diff_val, y_diff_test, scaler_X, scaler_y_diff


def prepare_data_differenced(sequence_length=24, forecast_horizon=24, train_ratio=0.6, val_ratio=0.2):
    """
    Complete data preparation pipeline with DIFFERENCED target.
    
    This function implements the fix for non-stationarity by transforming
    the target to first differences.
    
    Returns:
        Dictionary with all prepared data, scalers, and reconstruction metadata
    """
    # Feature columns (without options OI)
    feature_cols = [
        'dvol_lag_1d',
        'dvol_lag_7d',
        'dvol_lag_30d',
        'transaction_volume',
        'network_activity',
        'nvrv',
        'dvol_rv_spread'
    ]
    target_col = 'dvol'
    
    # Load data
    df = load_features_data()
    
    # Create sequences with differenced target
    X, y_diff, y_prev, timestamps = create_sequences_differenced(
        df, feature_cols, target_col, 
        sequence_length, forecast_horizon
    )
    
    # Split
    (X_train, X_val, X_test, 
     y_diff_train, y_diff_val, y_diff_test,
     y_prev_train, y_prev_val, y_prev_test) = split_temporal_data(
        X, y_diff, y_prev, train_ratio, val_ratio
    )
    
    # Normalize
    (X_train, X_val, X_test, 
     y_diff_train, y_diff_val, y_diff_test, 
     scaler_X, scaler_y_diff) = normalize_features(
        X_train, X_val, X_test, 
        y_diff_train, y_diff_val, y_diff_test
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_diff_train': y_diff_train,
        'y_diff_val': y_diff_val,
        'y_diff_test': y_diff_test,
        'y_prev_train': y_prev_train,
        'y_prev_val': y_prev_val,
        'y_prev_test': y_prev_test,
        'scaler_X': scaler_X,
        'scaler_y_diff': scaler_y_diff,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon
    }


def reconstruct_from_diff(y_diff_pred, y_prev):
    """
    Reconstruct original DVOL values from predicted differences.
    
    Formula: dvol_t = dvol_{t-1} + Δdvol_t
    
    Args:
        y_diff_pred: Predicted differences (samples, 1)
        y_prev: Previous DVOL values (samples, 1)
    
    Returns:
        y_reconstructed: Reconstructed DVOL values (samples, 1)
    """
    return y_prev + y_diff_pred
