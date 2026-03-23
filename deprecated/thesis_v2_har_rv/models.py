"""
HAR-RV Volatility Forecasting Models.

This module provides the core HAR-RV (Heterogeneous Autoregressive Realized Volatility)
model implementation supporting both standard and differenced targets.

Reference: Corsi (2009) - "A simple approximate long-memory model of realized volatility"
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass
from typing import Dict, Tuple, Union, Optional
from pathlib import Path


@dataclass
class HARRVConfig:
    """Configuration for HAR-RV model."""
    daily_lag: int = 1
    weekly_lag: int = 5
    monthly_lag: int = 22
    forecast_horizon: int = 1
    include_intercept: bool = True
    difference_target: bool = False

    def __post_init__(self):
        assert self.daily_lag > 0
        assert self.weekly_lag >= self.daily_lag
        assert self.monthly_lag >= self.weekly_lag
        assert self.forecast_horizon > 0


class HARRV:
    """Unified HAR-RV model supporting both standard and differenced targets."""

    def __init__(self, config: HARRVConfig = None):
        self.config = config or HARRVConfig()
        self.model = LinearRegression(fit_intercept=self.config.include_intercept)
        self.is_fitted = False
        self.feature_names = None
        self.coef_dict = None

    def _create_har_features(self, rv_series: np.ndarray) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Create HAR features."""
        n = len(rv_series)
        max_lag = self.config.monthly_lag
        horizon = self.config.forecast_horizon

        if self.config.difference_target:
            n_samples = n - max_lag - horizon
        else:
            n_samples = n - max_lag - horizon + 1

        if n_samples <= 0:
            raise ValueError(f"Series too short. Need {max_lag + horizon}, got {n}")

        X = np.zeros((n_samples, 3))

        for i in range(n_samples):
            idx = i + max_lag - 1
            X[i, 0] = rv_series[idx - self.config.daily_lag + 1]
            week_start = idx - self.config.weekly_lag + 1
            X[i, 1] = np.mean(rv_series[week_start:idx + 1])
            month_start = idx - self.config.monthly_lag + 1
            X[i, 2] = np.mean(rv_series[month_start:idx + 1])

        if self.config.difference_target:
            y_diff = np.zeros(n_samples)
            y_prev = np.zeros(n_samples)

            for i in range(n_samples):
                idx = i + max_lag - 1
                target_idx = idx + horizon
                y_diff[i] = rv_series[target_idx] - rv_series[target_idx - 1]
                y_prev[i] = rv_series[target_idx - 1]

            return X, y_diff, y_prev
        else:
            y = np.zeros(n_samples)
            for i in range(n_samples):
                idx = i + max_lag - 1
                y[i] = rv_series[idx + horizon]

            return X, y

    def fit(self, rv_series: np.ndarray) -> 'HARRV':
        """Fit the HAR-RV model."""
        if self.config.difference_target:
            X, y_diff, _ = self._create_har_features(rv_series)
            self.model.fit(X, y_diff)
            model_type = "HAR-RV (Differenced)"
        else:
            X, y = self._create_har_features(rv_series)
            self.model.fit(X, y)
            model_type = "HAR-RV"

        self.is_fitted = True
        self.feature_names = ['RV_daily', 'RV_weekly', 'RV_monthly']
        self.coef_dict = {
            'intercept': self.model.intercept_ if self.config.include_intercept else 0.0,
            'beta_daily': self.model.coef_[0],
            'beta_weekly': self.model.coef_[1],
            'beta_monthly': self.model.coef_[2]
        }

        print(f"{model_type} fitted - Intercept: {self.coef_dict['intercept']:.6f}, "
              f"β_d: {self.coef_dict['beta_daily']:.6f}, "
              f"β_w: {self.coef_dict['beta_weekly']:.6f}, "
              f"β_m: {self.coef_dict['beta_monthly']:.6f}")
        return self

    def predict(self, rv_series: np.ndarray,
                return_reconstruction: bool = True) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray]
    ]:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.config.difference_target:
            X, _, y_prev = self._create_har_features(rv_series)
            y_diff_pred = self.model.predict(X)

            if return_reconstruction:
                y_reconstructed = y_prev + y_diff_pred
                return y_diff_pred, y_reconstructed
            else:
                return y_diff_pred, None
        else:
            X, _ = self._create_har_features(rv_series)
            return self.model.predict(X)

    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.coef_dict.copy()

    def get_r_squared(self, rv_series: np.ndarray,
                     on_diff: Optional[bool] = None) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.config.difference_target:
            X, y_diff_true, y_prev = self._create_har_features(rv_series)
            y_diff_pred = self.model.predict(X)

            if on_diff is None:
                on_diff = True
            elif not self.config.difference_target:
                on_diff = False

            if on_diff:
                ss_res = np.sum((y_diff_true - y_diff_pred) ** 2)
                ss_tot = np.sum((y_diff_true - np.mean(y_diff_true)) ** 2)
            else:
                y_true = y_prev + y_diff_true
                y_pred = y_prev + y_diff_pred
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        else:
            X, y_true = self._create_har_features(rv_series)
            y_pred = self.model.predict(X)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')

    def __repr__(self) -> str:
        status = "Fitted" if self.is_fitted else "Not fitted"
        diff_str = "-Diff" if self.config.difference_target else ""
        return (f"HAR-RV{diff_str} ({status}) - Lags: d={self.config.daily_lag}, "
                f"w={self.config.weekly_lag}, m={self.config.monthly_lag}, "
                f"horizon={self.config.forecast_horizon}")


def create_har_rv_model(daily_lag: int = 1,
                        weekly_lag: int = 5,
                        monthly_lag: int = 22,
                        forecast_horizon: int = 1,
                        difference_target: bool = False) -> HARRV:
    """Factory function to create HAR-RV model."""
    config = HARRVConfig(
        daily_lag=daily_lag,
        weekly_lag=weekly_lag,
        monthly_lag=monthly_lag,
        forecast_horizon=forecast_horizon,
        difference_target=difference_target
    )
    return HARRV(config)


def create_har_rv_differenced(daily_lag: int = 1,
                              weekly_lag: int = 5,
                              monthly_lag: int = 22,
                              forecast_horizon: int = 1) -> HARRV:
    """Backward compatibility function for differenced HAR-RV."""
    return create_har_rv_model(
        daily_lag=daily_lag,
        weekly_lag=weekly_lag,
        monthly_lag=monthly_lag,
        forecast_horizon=forecast_horizon,
        difference_target=True
    )


def create_comprehensive_har_rv_model(features: list = None,
                                     forecast_horizon: int = 1,
                                     difference_target: bool = False) -> LinearRegression:
    """Create comprehensive HAR-RV model using actual dataset features."""
    if features is None:
        features = ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                   'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']

    model = LinearRegression(fit_intercept=True)
    return model


def evaluate_comprehensive_har_rv(df: pd.DataFrame,
                                dvol_col: str = 'dvol',
                                features: list = None,
                                train_split: float = 0.7,
                                forecast_horizon: int = 1) -> Dict:
    """Evaluate comprehensive HAR-RV model using actual dataset features."""
    if features is None:
        features = ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                   'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']

    df_clean = df[[dvol_col] + features].dropna()
    n = len(df_clean)
    train_size = int(n * train_split)

    train_data = df_clean.iloc[:train_size]
    test_data = df_clean.iloc[train_size:]

    X_train = train_data[features]
    y_train = train_data[dvol_col].shift(-forecast_horizon).dropna()

    X_test = test_data[features]
    y_test = test_data[dvol_col].shift(-forecast_horizon).dropna()

    X_train = X_train.iloc[:len(y_train)]
    X_test = X_test.iloc[:len(y_test)]

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    return {
        'model': model,
        'features': features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'coefficients': dict(zip(features, model.coef_)),
        'intercept': model.intercept
    }
