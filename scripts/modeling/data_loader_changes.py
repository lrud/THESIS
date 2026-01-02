#!/usr/bin/env python3
"""
Data Loader for LSTM with Differenced Target (Δdvol)
====================================================

Loads data and creates sequences for LSTM models predicting next-period DVOL changes.
This enables direct comparison to XGBoost Spec D benchmarks.

Target: Δdvol = dvol_{t+1} - dvol_t (next-period change)
Features: 7 core features (consistent with rolling/jump-aware loaders)

Difference from rolling/jump-aware:
- Target is differenced (stationary by construction)
- No rolling window normalization needed
- Direct comparison to XGChange benchmarks (R²=0.50 baseline)

Author: Claude Code Assistant
Date: December 30, 2025
Purpose: Option C - Train both LSTM variants (levels + changes)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class DifferencedLSTMDataset(Dataset):
    """
    Dataset for LSTM with differenced target (Δdvol).

    Predicts next-period DVOL change: dvol_{t+1} - dvol_t

    This is the standard econometric approach for non-stationary series.
    Enables direct comparison to XGBoost Spec D benchmarks.
    """

    def __init__(self, data, sequence_length=24, mode='train', jump_aware=False):
        """
        Args:
            data: DataFrame with DVOL, features, and optional jump indicators
            sequence_length: Hours of history to use (24h = 1 day)
            mode: 'train', 'val', or 'test'
            jump_aware: If True, include jump_indicator in data for separate evaluation
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.mode = mode
        self.jump_aware = jump_aware

        # Core features (same as rolling/jump-aware loaders for consistency)
        self.feature_cols = [
            'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
            'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'
        ]

        self.target_col = 'dvol'
        self.change_col = 'dvol_change'

        # Create target if not exists
        if self.change_col not in self.data.columns:
            self.data[self.change_col] = self.data[self.target_col].shift(-1) - self.data[self.target_col]

        # Drop rows with NaN in features or target
        self.data = self.data.dropna(subset=self.feature_cols + [self.change_col])

        # If jump_aware, keep jump_indicator for evaluation
        if self.jump_aware and 'jump_indicator' in self.data.columns:
            self.data = self.data.dropna(subset=['jump_indicator'])
        else:
            self.jump_aware = False

        self._prepare_data()

        print(f"{mode.upper()} set: {len(self.X):,} samples")
        print(f"  Features: {len(self.feature_cols)} ({', '.join(self.feature_cols)})")
        if self.jump_aware:
            jump_pct = (self.jump_indicator == 1).sum() / len(self.jump_indicator) * 100
            print(f"  Jump samples: {(self.jump_indicator == 1).sum():,} ({jump_pct:.1f}%)")
        print(f"  Target stats: mean={self.y.mean():.4f}, std={self.y.std():.4f}")

    def _prepare_data(self):
        """Normalize features and create sequences."""
        # Normalize features using StandardScaler
        scaler = StandardScaler()
        self.data[self.feature_cols] = scaler.fit_transform(self.data[self.feature_cols])

        # Create sequences
        X_list = []
        y_list = []
        jump_list = []

        for i in range(self.sequence_length, len(self.data)):
            # Input sequence
            X_seq = self.data[self.feature_cols].iloc[i-self.sequence_length:i].values
            X_list.append(X_seq)

            # Target: next-period change
            y_val = self.data[self.change_col].iloc[i]
            y_list.append(y_val)

            # Jump indicator (if available)
            if self.jump_aware:
                jump_list.append(self.data['jump_indicator'].iloc[i])

        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32).reshape(-1, 1)

        if self.jump_aware:
            self.jump_indicator = np.array(jump_list, dtype=np.int32)
        else:
            self.jump_indicator = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.jump_aware:
            return (
                torch.FloatTensor(self.X[idx]),
                torch.FloatTensor(self.y[idx]),
                torch.LongTensor([self.jump_indicator[idx]])
            )
        else:
            return (
                torch.FloatTensor(self.X[idx]),
                torch.FloatTensor(self.y[idx])
            )


class SpecDLSTMDataset(Dataset):
    """
    Dataset for LSTM with Spec D features (XGBoost benchmark match).

    Uses the same feature set as XGBoost Spec D:
    - nvrv_diff instead of nvrv
    - dvol_change_lag_1, dvol_change_lag_24 (momentum features)

    This enables the fairest possible comparison to XGBoost Spec D (R²=0.50).
    """

    def __init__(self, data, sequence_length=24, mode='train', jump_aware=False):
        """
        Args:
            data: DataFrame with DVOL and features
            sequence_length: Hours of history to use
            mode: 'train', 'val', or 'test'
            jump_aware: If True, include jump_indicator for separate evaluation
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.mode = mode
        self.jump_aware = jump_aware

        # Spec D features (matching XGBoost benchmark)
        self.feature_cols = [
            'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
            'transaction_volume', 'network_activity', 'nvrv_diff', 'dvol_rv_spread',
            'dvol_change_lag_1', 'dvol_change_lag_24'
        ]

        self.target_col = 'dvol'
        self.change_col = 'dvol_change'

        # Create Spec D features if not exists
        self._create_spec_d_features()

        # Drop rows with NaN
        self.data = self.data.dropna(subset=self.feature_cols + [self.change_col])

        if self.jump_aware and 'jump_indicator' in self.data.columns:
            self.data = self.data.dropna(subset=['jump_indicator'])
        else:
            self.jump_aware = False

        self._prepare_data()

        print(f"{mode.upper()} set (Spec D): {len(self.X):,} samples")
        print(f"  Features: {len(self.feature_cols)} (Spec D: nvrv_diff, dvol_change_lag)")
        if self.jump_aware:
            jump_pct = (self.jump_indicator == 1).sum() / len(self.jump_indicator) * 100
            print(f"  Jump samples: {(self.jump_indicator == 1).sum():,} ({jump_pct:.1f}%)")
        print(f"  Target stats: mean={self.y.mean():.4f}, std={self.y.std():.4f}")

    def _create_spec_d_features(self):
        """Create Spec D features to match XGBoost benchmark."""
        # dvol_change (next-period change)
        if self.change_col not in self.data.columns:
            self.data[self.change_col] = self.data[self.target_col].shift(-1) - self.data[self.target_col]

        # nvrv_diff: first difference of NVRV
        if 'nvrv_diff' not in self.data.columns:
            self.data['nvrv_diff'] = self.data['nvrv'].diff()

        # dvol_change_lag_1: 1-hour lagged change
        if 'dvol_change_lag_1' not in self.data.columns:
            self.data['dvol_change_lag_1'] = self.data[self.change_col].shift(1)

        # dvol_change_lag_24: 24-hour lagged change
        if 'dvol_change_lag_24' not in self.data.columns:
            self.data['dvol_change_lag_24'] = self.data[self.change_col].shift(24)

    def _prepare_data(self):
        """Normalize features and create sequences."""
        scaler = StandardScaler()
        self.data[self.feature_cols] = scaler.fit_transform(self.data[self.feature_cols])

        X_list = []
        y_list = []
        jump_list = []

        for i in range(self.sequence_length, len(self.data)):
            X_seq = self.data[self.feature_cols].iloc[i-self.sequence_length:i].values
            X_list.append(X_seq)

            y_val = self.data[self.change_col].iloc[i]
            y_list.append(y_val)

            if self.jump_aware:
                jump_list.append(self.data['jump_indicator'].iloc[i])

        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32).reshape(-1, 1)

        if self.jump_aware:
            self.jump_indicator = np.array(jump_list, dtype=np.int32)
        else:
            self.jump_indicator = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.jump_aware:
            return (
                torch.FloatTensor(self.X[idx]),
                torch.FloatTensor(self.y[idx]),
                torch.LongTensor([self.jump_indicator[idx]])
            )
        else:
            return (
                torch.FloatTensor(self.X[idx]),
                torch.FloatTensor(self.y[idx])
            )


def create_changes_dataloaders(data_path, sequence_length=24, batch_size=32,
                               train_ratio=0.7, val_ratio=0.15, jump_aware=False,
                               use_spec_d=False):
    """
    Create train/val/test dataloaders for differenced target LSTM.

    Args:
        data_path: Path to v1.1 data file
        sequence_length: LSTM input sequence length
        batch_size: Batch size for training
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        jump_aware: If True, include jump_indicator for separate evaluation
        use_spec_d: If True, use Spec D features (XGBoost match)

    Returns:
        train_loader, val_loader, test_loader, test_dataset
    """
    # Load data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded data from: {data_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Split temporally (no shuffling for time series)
    n_samples = len(df) - sequence_length
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Create datasets
    if use_spec_d:
        DatasetClass = SpecDLSTMDataset
    else:
        DatasetClass = DifferencedLSTMDataset

    train_df = df.iloc[:train_end + sequence_length].copy()
    val_df = df.iloc[train_end:val_end + sequence_length].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nSplit ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%}")

    train_dataset = DatasetClass(train_df, sequence_length, mode='train', jump_aware=jump_aware)
    val_dataset = DatasetClass(val_df, sequence_length, mode='val', jump_aware=jump_aware)
    test_dataset = DatasetClass(test_df, sequence_length, mode='test', jump_aware=jump_aware)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_dataset


if __name__ == "__main__":
    # Test the data loader
    import sys
    sys.path.append('scripts/modeling')

    DATA_PATH = 'data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv'

    print("="*70)
    print("TESTING: Differenced LSTM Data Loader")
    print("="*70)

    # Test 1: Core features (7 features)
    print("\n" + "-"*70)
    print("TEST 1: Core Features (7 features, no jump awareness)")
    print("-"*70)

    train_loader, val_loader, test_loader, test_dataset = create_changes_dataloaders(
        DATA_PATH,
        sequence_length=24,
        batch_size=32,
        jump_aware=False,
        use_spec_d=False
    )

    # Test 2: Spec D features (9 features)
    print("\n" + "-"*70)
    print("TEST 2: Spec D Features (9 features, matches XGBoost)")
    print("-"*70)

    train_loader, val_loader, test_loader, test_dataset = create_changes_dataloaders(
        DATA_PATH,
        sequence_length=24,
        batch_size=32,
        jump_aware=False,
        use_spec_d=True
    )

    # Test 3: Jump-aware evaluation
    print("\n" + "-"*70)
    print("TEST 3: Jump-Aware Evaluation (core features + jump indicator)")
    print("-"*70)

    train_loader, val_loader, test_loader, test_dataset = create_changes_dataloaders(
        DATA_PATH,
        sequence_length=24,
        batch_size=32,
        jump_aware=True,
        use_spec_d=False
    )

    print("\n" + "="*70)
    print("DATA LOADER TEST COMPLETE")
    print("="*70)
