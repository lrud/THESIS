"""
Shared utilities for Bitcoin DVOL forecasting project.

This module contains consolidated utilities used across different model types
and training approaches to eliminate code duplication.

Modules:
- metrics: Unified evaluation metrics calculation

Note: HAR-RV functionality has been deprecated. For HAR-RV analysis, use
  notebooks/benchmarking.ipynb which provides comprehensive baseline analysis.
"""

from .metrics import calculate_metrics, print_metrics_comparison, calculate_model_performance_summary

__all__ = [
    'calculate_metrics',
    'print_metrics_comparison',
    'calculate_model_performance_summary',
]
