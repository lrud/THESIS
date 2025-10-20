import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

class RollingWindowDataLoader:
    """
    Data loader with ROLLING WINDOW NORMALIZATION for target variable.
    
    This solves non-stationarity WITHOUT differencing:
    - Adapts to regime changes (mean shift from 69→47)
    - Preserves predictable relationships (features → DVOL level)
    - Avoids trivial solution (predict no change)
    
    Key difference from differenced approach:
    - Differenced: Predicts Δdvol ≈ 0 (naive persistence)
    - Rolling: Predicts deviation from local mean (genuine forecasting)
    """
    
    def __init__(self, data_path, sequence_length=24, forecast_horizon=24, 
                 rolling_window=720, batch_size=32, train_ratio=0.6, val_ratio=0.2):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.rolling_window = rolling_window  # 720h = 30 days
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        self.feature_scaler = StandardScaler()
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.train_rolling_stats = None
        self.val_rolling_stats = None
        self.test_rolling_stats = None
    
    def prepare_data(self):
        """Load, normalize, and create sequences."""
        df = self._load_data()
        
        df = self._apply_rolling_normalization(df)
        
        self._create_sequences(df)
        
        self._normalize_features()
        
        self._create_dataloaders()
        
        print("Data preparation complete (Rolling Window Normalization)")
        print(f"  Rolling window: {self.rolling_window}h ({self.rolling_window/24:.1f} days)")
        print(f"  Train: {len(self.X_train):,} samples")
        print(f"  Val:   {len(self.X_val):,} samples")
        print(f"  Test:  {len(self.X_test):,} samples")
        print(f"  Features: {self.X_train.shape[2]}")
    
    def _load_data(self):
        """Load preprocessed features."""
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df
    
    def _apply_rolling_normalization(self, df):
        """
        Apply rolling window normalization to target variable.
        
        For each timestamp t:
        dvol_normalized_t = (dvol_t - mean(dvol_{t-W:t})) / std(dvol_{t-W:t})
        
        Where W = rolling_window size (e.g., 720h = 30 days)
        """
        rolling_mean = df['dvol'].rolling(
            window=self.rolling_window,
            min_periods=self.rolling_window
        ).mean()
        
        rolling_std = df['dvol'].rolling(
            window=self.rolling_window,
            min_periods=self.rolling_window
        ).std()
        
        df['dvol_normalized'] = (df['dvol'] - rolling_mean) / (rolling_std + 1e-8)
        
        df['rolling_mean'] = rolling_mean
        df['rolling_std'] = rolling_std
        
        df = df.dropna(subset=['dvol_normalized', 'rolling_mean', 'rolling_std'])
        
        print(f"\nRolling normalization statistics:")
        print(f"  Original DVOL: mean={df['dvol'].mean():.2f}, std={df['dvol'].std():.2f}")
        print(f"  Normalized DVOL: mean={df['dvol_normalized'].mean():.2f}, std={df['dvol_normalized'].std():.2f}")
        print(f"  Samples after dropna: {len(df):,}")
        
        return df
    
    def _create_sequences(self, df):
        """Create input-output sequences."""
        feature_cols = [
            'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
            'transaction_volume', 'network_activity',
            'nvrv', 'dvol_rv_spread'
        ]
        
        X_all, y_all, rolling_mean_all, rolling_std_all = [], [], [], []
        
        for i in range(self.sequence_length, len(df) - self.forecast_horizon):
            X_all.append(df[feature_cols].iloc[i-self.sequence_length:i].values)
            
            target_idx = i + self.forecast_horizon - 1
            y_all.append(df['dvol_normalized'].iloc[target_idx])
            rolling_mean_all.append(df['rolling_mean'].iloc[target_idx])
            rolling_std_all.append(df['rolling_std'].iloc[target_idx])
        
        X_all = np.array(X_all)
        y_all = np.array(y_all).reshape(-1, 1)
        rolling_mean_all = np.array(rolling_mean_all).reshape(-1, 1)
        rolling_std_all = np.array(rolling_std_all).reshape(-1, 1)
        
        n_samples = len(X_all)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        self.X_train = X_all[:train_end]
        self.y_train = y_all[:train_end]
        self.train_rolling_stats = {
            'mean': rolling_mean_all[:train_end],
            'std': rolling_std_all[:train_end]
        }
        
        self.X_val = X_all[train_end:val_end]
        self.y_val = y_all[train_end:val_end]
        self.val_rolling_stats = {
            'mean': rolling_mean_all[train_end:val_end],
            'std': rolling_std_all[train_end:val_end]
        }
        
        self.X_test = X_all[val_end:]
        self.y_test = y_all[val_end:]
        self.test_rolling_stats = {
            'mean': rolling_mean_all[val_end:],
            'std': rolling_std_all[val_end:]
        }
    
    def _normalize_features(self):
        """Normalize features using StandardScaler (fitted on train set only)."""
        n_samples_train, n_timesteps, n_features = self.X_train.shape
        
        X_train_reshaped = self.X_train.reshape(-1, n_features)
        self.feature_scaler.fit(X_train_reshaped)
        
        self.X_train = self.feature_scaler.transform(X_train_reshaped).reshape(
            n_samples_train, n_timesteps, n_features
        )
        
        n_samples_val = self.X_val.shape[0]
        self.X_val = self.feature_scaler.transform(
            self.X_val.reshape(-1, n_features)
        ).reshape(n_samples_val, n_timesteps, n_features)
        
        n_samples_test = self.X_test.shape[0]
        self.X_test = self.feature_scaler.transform(
            self.X_test.reshape(-1, n_features)
        ).reshape(n_samples_test, n_timesteps, n_features)
    
    def _create_dataloaders(self):
        """Create PyTorch DataLoaders."""
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train)
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.FloatTensor(self.y_val)
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(self.X_test),
            torch.FloatTensor(self.y_test)
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
    
    def inverse_transform_target(self, y_normalized, split='test'):
        """
        Convert normalized predictions back to original DVOL scale.
        
        dvol_t = dvol_normalized_t * rolling_std_t + rolling_mean_t
        """
        if split == 'train':
            rolling_stats = self.train_rolling_stats
        elif split == 'val':
            rolling_stats = self.val_rolling_stats
        else:
            rolling_stats = self.test_rolling_stats
        
        y_normalized = y_normalized.reshape(-1, 1)
        
        dvol_original = y_normalized * rolling_stats['std'] + rolling_stats['mean']
        
        return dvol_original.flatten()
    
    def get_split_stats(self):
        """Get statistics for each split (for analysis)."""
        stats = {
            'train': {
                'normalized_mean': self.y_train.mean(),
                'normalized_std': self.y_train.std(),
                'rolling_mean_avg': self.train_rolling_stats['mean'].mean(),
                'rolling_std_avg': self.train_rolling_stats['std'].mean()
            },
            'val': {
                'normalized_mean': self.y_val.mean(),
                'normalized_std': self.y_val.std(),
                'rolling_mean_avg': self.val_rolling_stats['mean'].mean(),
                'rolling_std_avg': self.val_rolling_stats['std'].mean()
            },
            'test': {
                'normalized_mean': self.y_test.mean(),
                'normalized_std': self.y_test.std(),
                'rolling_mean_avg': self.test_rolling_stats['mean'].mean(),
                'rolling_std_avg': self.test_rolling_stats['std'].mean()
            }
        }
        return stats
