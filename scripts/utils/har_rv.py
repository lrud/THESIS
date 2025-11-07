"""
Consolidated HAR-RV Model for volatility forecasting.

This module provides a unified HAR-RV implementation that can handle both
standard and differenced target variables.

Reference: Corsi (2009)
Model: RV_t+h = β₀ + β_d·RV_t + β_w·RV_t^(week) + β_m·RV_t^(month) + ε_t

Author: Claude Code Assistant
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import Dict, Tuple, Union, Optional


@dataclass
class HARRVConfig:
    """Configuration for HAR-RV model."""
    daily_lag: int = 1
    weekly_lag: int = 5
    monthly_lag: int = 22
    forecast_horizon: int = 1
    include_intercept: bool = True
    difference_target: bool = False  # NEW: Unified parameter

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
        Tuple[np.ndarray, np.ndarray],  # Standard case
        Tuple[np.ndarray, np.ndarray, np.ndarray]  # Differenced case
    ]:
        """
        Create HAR features.

        Returns:
            For standard: X, y
            For differenced: X, y_diff, y_prev
        """
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
        np.ndarray,  # Standard case
        Tuple[np.ndarray, np.ndarray]  # Differenced case
    ]:
        """
        Make predictions.

        Args:
            rv_series: Input RV series
            return_reconstruction: For differenced models, whether to return reconstructed values

        Returns:
            Standard: y_pred
            Differenced: (y_diff_pred, y_reconstructed) if return_reconstruction=True
                      (y_diff_pred, None) if return_reconstruction=False
        """
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
        """
        Calculate R² score.

        Args:
            rv_series: RV series
            on_diff: For differenced models, whether to calculate R² on differences or reconstructed values
                    If None, uses differenced scale for differenced models
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.config.difference_target:
            X, y_diff_true, y_prev = self._create_har_features(rv_series)
            y_diff_pred = self.model.predict(X)

            # Determine whether to calculate on differenced or reconstructed scale
            if on_diff is None:
                on_diff = True  # Default for differenced models
            elif not self.config.difference_target:
                on_diff = False  # Always false for standard models

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
    """
    Factory function to create HAR-RV model.

    Args:
        daily_lag: Daily lag (default: 1)
        weekly_lag: Weekly lag (default: 5)
        monthly_lag: Monthly lag (default: 22)
        forecast_horizon: Forecast horizon (default: 1)
        difference_target: Whether to use differenced target (default: False)

    Returns:
        Configured HAR-RV model
    """
    config = HARRVConfig(
        daily_lag=daily_lag,
        weekly_lag=weekly_lag,
        monthly_lag=monthly_lag,
        forecast_horizon=forecast_horizon,
        difference_target=difference_target
    )
    return HARRV(config)


# Backward compatibility aliases
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