"""
Data loading and preprocessing utilities for LSTM model.

Handles:
- Loading Bitcoin LSTM features dataset
- Creating time series sequences
- Train/val/test splitting
- Feature normalization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_features_data(filepath='data/processed/bitcoin_lstm_features.csv'):
    """Load the Bitcoin LSTM features dataset."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.set_index('timestamp')
    return df.dropna()


def create_sequences(df, feature_cols, target_col, sequence_length=24, forecast_horizon=24):
    """
    Create sequences for LSTM training.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        sequence_length: Number of past timesteps (hours)
        forecast_horizon: Hours ahead to forecast
    
    Returns:
        X (sequences), y (targets), timestamps
    """
    X_sequences = []
    y_targets = []
    timestamps = []
    
    for i in range(len(df) - sequence_length - forecast_horizon):
        # Extract sequence
        seq = df[feature_cols].iloc[i:i+sequence_length].values
        
        # Extract target (DVOL at t+forecast_horizon)
        target = df[target_col].iloc[i+sequence_length+forecast_horizon-1]
        
        X_sequences.append(seq)
        y_targets.append(target)
        timestamps.append(df.index[i+sequence_length+forecast_horizon-1])
    
    return np.array(X_sequences), np.array(y_targets).reshape(-1, 1), timestamps


def split_temporal_data(X, y, train_ratio=0.6, val_ratio=0.2):
    """
    Split data temporally (no shuffling for time series).
    
    Args:
        X: Feature sequences
        y: Targets
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Normalize features and targets using StandardScaler.
    Fit on training data only.
    
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
    
    # Normalize target
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y


def prepare_data(sequence_length=24, forecast_horizon=24, train_ratio=0.6, val_ratio=0.2):
    """
    Complete data preparation pipeline.
    
    Returns:
        Dictionary with all prepared data and scalers
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
    
    # Create sequences
    X, y, timestamps = create_sequences(
        df, feature_cols, target_col, 
        sequence_length, forecast_horizon
    )
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal_data(
        X, y, train_ratio, val_ratio
    )
    
    # Normalize
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = normalize_features(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon
    }
