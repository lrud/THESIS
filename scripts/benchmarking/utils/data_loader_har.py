"""
Data loading utilities for HAR-RV model.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def load_dvol_data() -> pd.DataFrame:
    df = pd.read_csv('../../data/processed/bitcoin_lstm_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    print(f"Loaded {len(df)} samples from {df.index.min()} to {df.index.max()}")
    return df


def calculate_realized_volatility(df: pd.DataFrame, 
                                  price_col: str = None,
                                  return_col: str = None,
                                  window: int = 24) -> pd.Series:
    if return_col is not None and return_col in df.columns:
        returns = df[return_col]
    elif price_col is not None and price_col in df.columns:
        returns = df[price_col].pct_change()
    else:
        raise ValueError("Must provide either price_col or return_col")
    
    rv = returns.rolling(window=window).apply(
        lambda x: np.sqrt(np.sum(x**2)) * 100,
        raw=True
    )
    return rv


def prepare_har_rv_data(forecast_horizon: int = 24,
                        train_ratio: float = 0.6,
                        val_ratio: float = 0.2) -> Dict[str, np.ndarray]:
    df = load_dvol_data()
    rv_series = df['dvol'].values
    timestamps = df.index.values
    
    valid_idx = ~np.isnan(rv_series)
    rv_series = rv_series[valid_idx]
    timestamps = timestamps[valid_idx]
    
    print(f"\nDVOL: Mean={np.mean(rv_series):.2f}, Std={np.std(rv_series):.2f}, "
          f"Min={np.min(rv_series):.2f}, Max={np.max(rv_series):.2f}")
    
    n = len(rv_series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    rv_train = rv_series[:train_end]
    rv_val = rv_series[:val_end]
    rv_test = rv_series
    
    timestamps_train = timestamps[:train_end]
    timestamps_val = timestamps[:val_end]
    timestamps_test = timestamps
    
    print(f"\nTemporal split:")
    print(f"  Train: {len(rv_train):,} ({train_ratio*100:.0f}%) [{timestamps_train[0]} to {timestamps_train[-1]}]")
    print(f"  Val:   {val_end - train_end:,} ({val_ratio*100:.0f}%) [{timestamps[train_end]} to {timestamps[val_end-1]}]")
    print(f"  Test:  {n - val_end:,} ({(1-train_ratio-val_ratio)*100:.0f}%) [{timestamps[val_end]} to {timestamps[-1]}]")
    
    print(f"\nSplit statistics:")
    print(f"  Train - Mean: {np.mean(rv_train):.2f}, Std: {np.std(rv_train):.2f}")
    print(f"  Val   - Mean: {np.mean(rv_series[train_end:val_end]):.2f}, Std: {np.std(rv_series[train_end:val_end]):.2f}")
    print(f"  Test  - Mean: {np.mean(rv_series[val_end:]):.2f}, Std: {np.std(rv_series[val_end:]):.2f}")
    
    return {
        'rv_train': rv_train,
        'rv_val': rv_val,
        'rv_test': rv_test,
        'timestamps_train': timestamps_train,
        'timestamps_val': timestamps_val,
        'timestamps_test': timestamps_test,
        'forecast_horizon': forecast_horizon,
        'split_indices': {
            'train_end': train_end,
            'val_end': val_end,
            'total': n
        }
    }


def extract_predictions_for_split(predictions: np.ndarray,
                                  split_indices: Dict[str, int],
                                  split_name: str) -> np.ndarray:
    if split_name == 'train':
        return predictions[:split_indices['train_end'] - 22 - 24 + 1]
    elif split_name == 'val':
        start = split_indices['train_end'] - 22 - 24 + 1
        end = split_indices['val_end'] - 22 - 24 + 1
        return predictions[start:end]
    elif split_name == 'test':
        start = split_indices['val_end'] - 22 - 24 + 1
        return predictions[start:]
    else:
        raise ValueError(f"Unknown split: {split_name}")


def get_aligned_actuals(rv_series: np.ndarray,
                       forecast_horizon: int = 24,
                       monthly_lag: int = 22) -> np.ndarray:
    start_idx = monthly_lag + forecast_horizon - 1
    return rv_series[start_idx:]
