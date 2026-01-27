"""Unified data loader for all LSTM model variants."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


FEATURE_SETS = {
    'market': ['transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'],
    'market_jumps': [
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
        'jump_indicator', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'
    ],
    'market_lags': [
        'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread'
    ],
    'jump_aware': [
        'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
        'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread',
        'jump_indicator', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'
    ]
}


class UnifiedLSTMDataset(Dataset):
    """Unified dataset for all LSTM model variants with configurable feature sets."""

    def __init__(self, data, feature_set, sequence_length=24, window_size=720, mode='train'):
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.mode = mode
        self.feature_cols = FEATURE_SETS[feature_set]
        self.use_weighting = 'jump' in feature_set
        self.target_col = 'dvol'

        self.data = self.data.dropna(subset=self.feature_cols + [self.target_col])
        self._apply_rolling_normalization()
        self.X, self.y, self.weights, self.rolling_stats = self._prepare_sequences()

    def _apply_rolling_normalization(self):
        """Apply 720h rolling window normalization to all features and target."""
        for col in self.feature_cols:
            if col in ['jump_indicator', 'jump_cluster_7d']:
                self.data[f'{col}_normalized'] = self.data[col]
                continue

            rolling_mean = self.data[col].rolling(self.window_size, min_periods=1).mean()
            rolling_std = self.data[col].rolling(self.window_size, min_periods=1).std()
            rolling_std = rolling_std.replace(0, 1e-8)
            self.data[f'{col}_normalized'] = (self.data[col] - rolling_mean) / rolling_std

        target_rolling_mean = self.data[self.target_col].rolling(self.window_size, min_periods=1).mean()
        target_rolling_std = self.data[self.target_col].rolling(self.window_size, min_periods=1).std()
        target_rolling_std = target_rolling_std.replace(0, 1e-8)
        self.data['target_normalized'] = (self.data[self.target_col] - target_rolling_mean) / target_rolling_std
        self.data['target_rolling_mean'] = target_rolling_mean
        self.data['target_rolling_std'] = target_rolling_std

        self.data = self.data[self.window_size:]

    def _prepare_sequences(self):
        """Create LSTM sequences with 1-hour ahead forecast."""
        X_list, y_list, weight_list, stats_list = [], [], [], []

        normalized_features = [
            f'{col}_normalized' if col not in ['jump_indicator', 'jump_cluster_7d'] else col
            for col in self.feature_cols
        ]

        for i in range(self.sequence_length, len(self.data) - 1):
            X_seq = self.data[normalized_features].iloc[i - self.sequence_length:i].values
            X_list.append(X_seq)

            y_val = self.data['target_normalized'].iloc[i + 1]
            y_list.append(y_val)

            if self.use_weighting:
                is_jump = self.data['jump_indicator'].iloc[i + 1]
                weight_list.append(2.0 if is_jump else 1.0)
            else:
                weight_list.append(1.0)

            rolling_mean = self.data['target_rolling_mean'].iloc[i + 1]
            rolling_std = self.data['target_rolling_std'].iloc[i + 1]
            stats_list.append([rolling_mean, rolling_std])

        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(weight_list, dtype=np.float32),
            np.array(stats_list, dtype=np.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.y[idx]]),
            torch.FloatTensor([self.weights[idx]]),
            torch.FloatTensor(self.rolling_stats[idx])
        )

    def inverse_transform_target(self, normalized_target, rolling_stats):
        """Convert normalized predictions back to original DVOL scale."""
        if isinstance(normalized_target, torch.Tensor):
            normalized_target = normalized_target.cpu().numpy()
        if isinstance(rolling_stats, torch.Tensor):
            rolling_stats = rolling_stats.cpu().numpy()

        mean = rolling_stats[:, 0:1]
        std = rolling_stats[:, 1:2]
        return normalized_target * std + mean


def create_unified_dataloaders(
    data_path='data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv',
    feature_set='market',
    sequence_length=24,
    window_size=720,
    batch_size=32,
    val_ratio=0.2,
    test_ratio=0.2
):
    """Create train/val/test dataloaders for unified LSTM training."""
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    n = len(df)
    train_size = int(n * (1 - val_ratio - test_ratio))
    val_size = int(n * val_ratio)

    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    train_dataset = UnifiedLSTMDataset(train_data, feature_set, sequence_length, window_size, 'train')
    val_dataset = UnifiedLSTMDataset(val_data, feature_set, sequence_length, window_size, 'val')
    test_dataset = UnifiedLSTMDataset(test_data, feature_set, sequence_length, window_size, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
