"""
HAR-RV Model for volatility forecasting.
Reference: Corsi (2009)
Model: RV_t+h = β₀ + β_d·RV_t + β_w·RV_t^(week) + β_m·RV_t^(month) + ε_t
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class HARRVConfig:
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


class HARRV:
    
    def __init__(self, config: HARRVConfig = None):
        self.config = config or HARRVConfig()
        self.model = LinearRegression(fit_intercept=self.config.include_intercept)
        self.is_fitted = False
        self.feature_names = None
        self.coef_dict = None
        
    def _create_har_features(self, rv_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(rv_series)
        max_lag = self.config.monthly_lag
        horizon = self.config.forecast_horizon
        n_samples = n - max_lag - horizon + 1
        
        if n_samples <= 0:
            raise ValueError(f"Series too short. Need {max_lag + horizon}, got {n}")
        
        X = np.zeros((n_samples, 3))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            idx = i + max_lag - 1
            X[i, 0] = rv_series[idx - self.config.daily_lag + 1]
            week_start = idx - self.config.weekly_lag + 1
            X[i, 1] = np.mean(rv_series[week_start:idx + 1])
            month_start = idx - self.config.monthly_lag + 1
            X[i, 2] = np.mean(rv_series[month_start:idx + 1])
            y[i] = rv_series[idx + horizon]
        
        return X, y
    
    def fit(self, rv_series: np.ndarray) -> 'HARRV':
        X, y = self._create_har_features(rv_series)
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.feature_names = ['RV_daily', 'RV_weekly', 'RV_monthly']
        self.coef_dict = {
            'intercept': self.model.intercept_ if self.config.include_intercept else 0.0,
            'beta_daily': self.model.coef_[0],
            'beta_weekly': self.model.coef_[1],
            'beta_monthly': self.model.coef_[2]
        }
        
        print(f"HAR-RV fitted - Intercept: {self.coef_dict['intercept']:.6f}, "
              f"β_d: {self.coef_dict['beta_daily']:.6f}, "
              f"β_w: {self.coef_dict['beta_weekly']:.6f}, "
              f"β_m: {self.coef_dict['beta_monthly']:.6f}")
        return self
    
    def predict(self, rv_series: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X, _ = self._create_har_features(rv_series)
        return self.model.predict(X)
    
    def get_coefficients(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.coef_dict.copy()
    
    def get_r_squared(self, rv_series: np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        X, y = self._create_har_features(rv_series)
        return self.model.score(X, y)
    
    def __repr__(self) -> str:
        status = "Fitted" if self.is_fitted else "Not fitted"
        return (f"HAR-RV ({status}) - Lags: d={self.config.daily_lag}, "
                f"w={self.config.weekly_lag}, m={self.config.monthly_lag}, "
                f"horizon={self.config.forecast_horizon}")


def create_har_rv_model(daily_lag: int = 1,
                        weekly_lag: int = 5,
                        monthly_lag: int = 22,
                        forecast_horizon: int = 1) -> HARRV:
    config = HARRVConfig(
        daily_lag=daily_lag,
        weekly_lag=weekly_lag,
        monthly_lag=monthly_lag,
        forecast_horizon=forecast_horizon
    )
    return HARRV(config)
