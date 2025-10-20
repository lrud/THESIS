"""
Benchmarking module for comparing LSTM with baseline models.

This module provides implementations of standard volatility forecasting
models for benchmarking against the LSTM model.

Available models:
- HAR-RV: Heterogeneous Autoregressive Realized Volatility

Usage:
    from benchmarking import create_har_rv_model, prepare_har_rv_data
    
    model = create_har_rv_model(forecast_horizon=24)
    data = prepare_har_rv_data()
    model.fit(data['rv_train'])
"""

from .models.har_rv_model import HARRV, HARRVConfig, create_har_rv_model
from .utils.data_loader_har import prepare_har_rv_data, load_dvol_data
from .utils.evaluator_har import calculate_metrics, plot_predictions_comparison

__all__ = [
    'HARRV',
    'HARRVConfig',
    'create_har_rv_model',
    'prepare_har_rv_data',
    'load_dvol_data',
    'calculate_metrics',
    'plot_predictions_comparison',
]

__version__ = '1.0.0'
