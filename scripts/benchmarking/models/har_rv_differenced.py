"""
HAR-RV Model with DIFFERENCED target variable.
Addresses the same non-stationarity issue as the LSTM model.
Reference: Corsi (2009) with first-difference transformation
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class HARRVDiffConfig:
    daily_lag: int = 1
    weekly_lag: int = 5
    monthly_lag: int = 22
    forecast_horizon: int = 1
    include_intercept: bool = True
    
    def __post_init__(self):
        assert self.daily_lag > 0
        assert self.weekly_lag >= self.daily_lag
        assert self.monthly_lag >= self.weekly_lag
        assert self.forecast_horizon > 0


class HARRVDifferenced:
    
    def __init__(self, config: HARRVDiffConfig = None):
        self.config = config or HARRVDiffConfig()
        self.model = LinearRegression(fit_intercept=self.config.include_intercept)
        self.is_fitted = False
        self.feature_names = None
        self.coef_dict = None
        
    def _create_har_features_diff(self, rv_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create HAR features with DIFFERENCED target.
        Target is Δdvol_t = dvol_t - dvol_{t-1}
        """
        n = len(rv_series)
        max_lag = self.config.monthly_lag
        horizon = self.config.forecast_horizon
        n_samples = n - max_lag - horizon
        
        if n_samples <= 0:
            raise ValueError(f"Series too short. Need {max_lag + horizon}, got {n}")
        
        X = np.zeros((n_samples, 3))
        y_diff = np.zeros(n_samples)
        y_prev = np.zeros(n_samples)
        
        for i in range(n_samples):
            idx = i + max_lag - 1
            X[i, 0] = rv_series[idx - self.config.daily_lag + 1]
            week_start = idx - self.config.weekly_lag + 1
            X[i, 1] = np.mean(rv_series[week_start:idx + 1])
            month_start = idx - self.config.monthly_lag + 1
            X[i, 2] = np.mean(rv_series[month_start:idx + 1])
            
            target_idx = idx + horizon
            y_diff[i] = rv_series[target_idx] - rv_series[target_idx - 1]
            y_prev[i] = rv_series[target_idx - 1]
        
        return X, y_diff, y_prev
    
    def fit(self, rv_series: np.ndarray) -> 'HARRVDifferenced':
        X, y_diff, _ = self._create_har_features_diff(rv_series)
        self.model.fit(X, y_diff)
        self.is_fitted = True
        
        self.feature_names = ['RV_daily', 'RV_weekly', 'RV_monthly']
        self.coef_dict = {
            'intercept': self.model.intercept_ if self.config.include_intercept else 0.0,
            'beta_daily': self.model.coef_[0],
            'beta_weekly': self.model.coef_[1],
            'beta_monthly': self.model.coef_[2]
        }
        
        print(f"HAR-RV (Differenced) fitted - Intercept: {self.coef_dict['intercept']:.6f}, "
              f"β_d: {self.coef_dict['beta_daily']:.6f}, "
              f"β_w: {self.coef_dict['beta_weekly']:.6f}, "
              f"β_m: {self.coef_dict['beta_monthly']:.6f}")
        return self
    
    def predict(self, rv_series: np.ndarray, return_reconstruction: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict differenced values and optionally reconstruct to absolute scale.
        
        Returns:
            y_diff_pred: Predicted differences
            y_reconstructed: Reconstructed absolute values (if return_reconstruction=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, _, y_prev = self._create_har_features_diff(rv_series)
        y_diff_pred = self.model.predict(X)
        
        if return_reconstruction:
            y_reconstructed = y_prev + y_diff_pred
            return y_diff_pred, y_reconstructed
        else:
            return y_diff_pred, None
    
    def get_coefficients(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.coef_dict.copy()
    
    def get_r_squared(self, rv_series: np.ndarray, on_diff: bool = True) -> float:
        """
        Calculate R² on differenced or reconstructed scale.
        
        Args:
            rv_series: RV series
            on_diff: If True, R² on differences; if False, on reconstructed values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X, y_diff_true, y_prev = self._create_har_features_diff(rv_series)
        y_diff_pred = self.model.predict(X)
        
        if on_diff:
            ss_res = np.sum((y_diff_true - y_diff_pred) ** 2)
            ss_tot = np.sum((y_diff_true - np.mean(y_diff_true)) ** 2)
        else:
            y_true = y_prev + y_diff_true
            y_pred = y_prev + y_diff_pred
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')
    
    def __repr__(self) -> str:
        status = "Fitted" if self.is_fitted else "Not fitted"
        return (f"HAR-RV-Diff ({status}) - Lags: d={self.config.daily_lag}, "
                f"w={self.config.weekly_lag}, m={self.config.monthly_lag}, "
                f"horizon={self.config.forecast_horizon}")


def create_har_rv_differenced(daily_lag: int = 1,
                               weekly_lag: int = 5,
                               monthly_lag: int = 22,
                               forecast_horizon: int = 1) -> HARRVDifferenced:
    config = HARRVDiffConfig(
        daily_lag=daily_lag,
        weekly_lag=weekly_lag,
        monthly_lag=monthly_lag,
        forecast_horizon=forecast_horizon
    )
    return HARRVDifferenced(config)
