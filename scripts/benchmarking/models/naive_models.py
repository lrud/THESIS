"""
Naive baseline models for volatility forecasting.
"""

import numpy as np
from typing import Dict


class NaivePersistence:
    
    def __init__(self, forecast_horizon: int = 1):
        self.forecast_horizon = forecast_horizon
        self.is_fitted = True
        
    def fit(self, y_series: np.ndarray) -> 'NaivePersistence':
        return self
    
    def predict(self, y_series: np.ndarray) -> np.ndarray:
        n = len(y_series)
        n_predictions = n - self.forecast_horizon
        predictions = np.zeros(n_predictions)
        
        for i in range(n_predictions):
            predictions[i] = y_series[i + self.forecast_horizon - 1]
        
        return predictions
    
    def __repr__(self) -> str:
        return f"NaivePersistence(h={self.forecast_horizon})"


class NaiveDrift:
    
    def __init__(self, forecast_horizon: int = 1):
        self.forecast_horizon = forecast_horizon
        self.drift = None
        self.is_fitted = False
        
    def fit(self, y_series: np.ndarray) -> 'NaiveDrift':
        differences = np.diff(y_series)
        self.drift = np.mean(differences)
        self.is_fitted = True
        return self
    
    def predict(self, y_series: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        n = len(y_series)
        n_predictions = n - self.forecast_horizon
        predictions = np.zeros(n_predictions)
        
        for i in range(n_predictions):
            predictions[i] = y_series[i + self.forecast_horizon - 1] + self.forecast_horizon * self.drift
        
        return predictions
    
    def get_drift(self) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.drift
    
    def __repr__(self) -> str:
        if self.is_fitted:
            return f"NaiveDrift(h={self.forecast_horizon}, drift={self.drift:.6f})"
        return f"NaiveDrift(h={self.forecast_horizon}, not fitted)"


class NaiveMean:
    
    def __init__(self, forecast_horizon: int = 1):
        self.forecast_horizon = forecast_horizon
        self.mean = None
        self.is_fitted = False
        
    def fit(self, y_series: np.ndarray) -> 'NaiveMean':
        self.mean = np.mean(y_series)
        self.is_fitted = True
        return self
    
    def predict(self, y_series: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        n = len(y_series)
        n_predictions = n - self.forecast_horizon
        predictions = np.full(n_predictions, self.mean)
        
        return predictions
    
    def get_mean(self) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.mean
    
    def __repr__(self) -> str:
        if self.is_fitted:
            return f"NaiveMean(h={self.forecast_horizon}, mean={self.mean:.2f})"
        return f"NaiveMean(h={self.forecast_horizon}, not fitted)"


class NaiveMovingAverage:
    
    def __init__(self, window: int = 5, forecast_horizon: int = 1):
        self.window = window
        self.forecast_horizon = forecast_horizon
        self.is_fitted = True
        
    def fit(self, y_series: np.ndarray) -> 'NaiveMovingAverage':
        return self
    
    def predict(self, y_series: np.ndarray) -> np.ndarray:
        n = len(y_series)
        n_predictions = n - self.forecast_horizon
        predictions = np.zeros(n_predictions)
        
        for i in range(n_predictions):
            end_idx = i + self.forecast_horizon - 1
            start_idx = max(0, end_idx - self.window + 1)
            predictions[i] = np.mean(y_series[start_idx:end_idx + 1])
        
        return predictions
    
    def __repr__(self) -> str:
        return f"NaiveMovingAverage(window={self.window}, h={self.forecast_horizon})"


def create_naive_models(forecast_horizon: int = 24) -> Dict[str, object]:
    return {
        'Persistence': NaivePersistence(forecast_horizon),
        'Drift': NaiveDrift(forecast_horizon),
        'Mean': NaiveMean(forecast_horizon),
        'MA5': NaiveMovingAverage(window=5, forecast_horizon=forecast_horizon),
        'MA22': NaiveMovingAverage(window=22, forecast_horizon=forecast_horizon)
    }


def get_aligned_actuals_naive(y_series: np.ndarray, forecast_horizon: int = 24) -> np.ndarray:
    n = len(y_series)
    n_predictions = n - forecast_horizon
    actuals = y_series[forecast_horizon:]
    return actuals[:n_predictions]
