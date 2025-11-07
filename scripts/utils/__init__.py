"""
Shared utilities for Bitcoin DVOL forecasting project.

This module contains consolidated utilities used across different model types
and training approaches to eliminate code duplication.

Modules:
- metrics: Unified evaluation metrics calculation
- har_rv: Consolidated HAR-RV model implementation
"""

from .metrics import calculate_metrics, print_metrics_comparison, calculate_model_performance_summary
from .har_rv import HARRV, create_har_rv_model, create_har_rv_differenced

__all__ = [
    'calculate_metrics',
    'print_metrics_comparison',
    'calculate_model_performance_summary',
    'HARRV',
    'create_har_rv_model',
    'create_har_rv_differenced'
]