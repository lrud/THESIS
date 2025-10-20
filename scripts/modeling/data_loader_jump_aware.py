import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class JumpAwareLSTMDataset(Dataset):
    """
    Dataset for LSTM with jump-aware features and loss weighting.
    
    Combines:
    1. Rolling window normalization (preserves feature relationships)
    2. Jump detection features (indicator, magnitude, timing)
    3. Sample weighting (emphasize jump periods)
    
    Key innovation: Differentiates normal vs. crisis periods for forecasting.
    """
    
    def __init__(self, data, sequence_length=24, window_size=720, mode='train'):
        """
        Args:
            data: DataFrame with DVOL, features, and jump indicators
            sequence_length: Hours of history to use (24h = 1 day)
            window_size: Rolling window for normalization (720h = 30 days)
            mode: 'train', 'val', or 'test' (affects normalization)
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.mode = mode
        
        self.feature_cols = [
            'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
            'transaction_volume', 'network_activity', 'nvrv', 
            'dvol_rv_spread',
            'jump_indicator', 'jump_magnitude', 'days_since_jump', 'jump_cluster_7d'
        ]
        
        self.target_col = 'dvol'
        
        self.data = self.data.dropna(subset=self.feature_cols + [self.target_col])
        
        self._apply_rolling_normalization()
        
        self.X, self.y, self.weights, self.rolling_stats = self._prepare_sequences()
        
        print(f"{mode.upper()} set: {len(self.X):,} samples")
        print(f"  Features: {len(self.feature_cols)} ({', '.join(self.feature_cols)})")
        print(f"  Jump samples: {self.weights[self.weights > 1.0].shape[0]:,} ({self.weights[self.weights > 1.0].shape[0]/len(self.weights)*100:.1f}%)")
        print(f"  Mean sample weight: {self.weights.mean():.3f}")
        print(f"  Target stats: mean={self.y.mean():.2f}, std={self.y.std():.2f}")
        
    def _apply_rolling_normalization(self):
        """
        Apply rolling window normalization to preserve feature relationships.
        Critical: Prevents global normalization bias when regime shifts occur.
        """
        for col in self.feature_cols:
            if col in ['jump_indicator']:
                continue
            
            rolling_mean = self.data[col].rolling(
                window=self.window_size, 
                min_periods=1
            ).mean()
            rolling_std = self.data[col].rolling(
                window=self.window_size, 
                min_periods=1
            ).std()
            
            rolling_std = rolling_std.replace(0, 1e-8)
            
            self.data[f'{col}_normalized'] = (
                (self.data[col] - rolling_mean) / rolling_std
            )
        
        target_rolling_mean = self.data[self.target_col].rolling(
            window=self.window_size,
            min_periods=1
        ).mean()
        target_rolling_std = self.data[self.target_col].rolling(
            window=self.window_size,
            min_periods=1
        ).std()
        
        target_rolling_std = target_rolling_std.replace(0, 1e-8)
        
        self.data['target_normalized'] = (
            (self.data[self.target_col] - target_rolling_mean) / target_rolling_std
        )
        
        self.data['target_rolling_mean'] = target_rolling_mean
        self.data['target_rolling_std'] = target_rolling_std
        
        self.data = self.data[self.window_size:]
        
    def _prepare_sequences(self):
        """
        Create LSTM input sequences with jump-aware weights.
        
        Returns:
            X: (n_samples, seq_len, n_features) input sequences
            y: (n_samples,) targets
            weights: (n_samples,) sample weights (higher for jump periods)
            rolling_stats: (n_samples, 2) for inverse transform
        """
        X_list = []
        y_list = []
        weight_list = []
        stats_list = []
        
        normalized_features = [
            f'{col}_normalized' if col not in ['jump_indicator'] else col
            for col in self.feature_cols
        ]
        
        for i in range(self.sequence_length, len(self.data)):
            X_seq = self.data[normalized_features].iloc[i-self.sequence_length:i].values
            X_list.append(X_seq)
            
            y_val = self.data['target_normalized'].iloc[i]
            y_list.append(y_val)
            
            is_jump = self.data['jump_indicator'].iloc[i]
            weight = 2.0 if is_jump else 1.0
            weight_list.append(weight)
            
            rolling_mean = self.data['target_rolling_mean'].iloc[i]
            rolling_std = self.data['target_rolling_std'].iloc[i]
            stats_list.append([rolling_mean, rolling_std])
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        weights = np.array(weight_list, dtype=np.float32)
        rolling_stats = np.array(stats_list, dtype=np.float32)
        
        return X, y, weights, rolling_stats
    
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
        """
        Convert normalized predictions back to original DVOL scale.
        
        Args:
            normalized_target: Normalized DVOL values
            rolling_stats: (mean, std) from dataset
        
        Returns:
            Original scale DVOL values
        """
        if isinstance(normalized_target, torch.Tensor):
            normalized_target = normalized_target.cpu().numpy()
        if isinstance(rolling_stats, torch.Tensor):
            rolling_stats = rolling_stats.cpu().numpy()
        
        mean = rolling_stats[:, 0:1]
        std = rolling_stats[:, 1:2]
        
        original = normalized_target * std + mean
        
        return original


def create_jump_aware_dataloaders(
    data_path='data/processed/bitcoin_lstm_features_with_jumps.csv',
    sequence_length=24,
    window_size=720,
    batch_size=32,
    val_ratio=0.2,
    test_ratio=0.2,
    random_state=42
):
    """
    Create train/val/test dataloaders with jump-aware features.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("=" * 80)
    print("LOADING JUMP-AWARE LSTM DATA")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nDataset: {len(df):,} observations")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Jump samples: {df['jump_indicator'].sum():,} ({df['jump_indicator'].mean()*100:.1f}%)")
    
    n = len(df)
    train_size = int(n * (1 - val_ratio - test_ratio))
    val_size = int(n * val_ratio)
    
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_data):,} ({len(train_data)/n*100:.1f}%)")
    print(f"  Val:   {len(val_data):,} ({len(val_data)/n*100:.1f}%)")
    print(f"  Test:  {len(test_data):,} ({len(test_data)/n*100:.1f}%)")
    print()
    
    train_dataset = JumpAwareLSTMDataset(
        train_data, sequence_length, window_size, mode='train'
    )
    val_dataset = JumpAwareLSTMDataset(
        val_data, sequence_length, window_size, mode='val'
    )
    test_dataset = JumpAwareLSTMDataset(
        test_data, sequence_length, window_size, mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print("=" * 80)
    print()
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_jump_aware_dataloaders()
    
    print("Batch example:")
    for X_batch, y_batch, w_batch, stats_batch in train_loader:
        print(f"  X shape: {X_batch.shape}")
        print(f"  y shape: {y_batch.shape}")
        print(f"  weights shape: {w_batch.shape}")
        print(f"  stats shape: {stats_batch.shape}")
        print(f"  Sample weight range: {w_batch.min():.2f} - {w_batch.max():.2f}")
        break
