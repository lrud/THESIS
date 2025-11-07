"""
Consolidated evaluation metrics for all model types.

This module provides unified metrics calculation functions that can be used
across LSTM models, HAR-RV models, and other forecasting approaches.

Author: Claude Code Assistant
"""

import numpy as np
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard evaluation metrics for forecasting models.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary containing RMSE, MAE, MAPE, R², and Directional Accuracy
    """
    # Ensure arrays are flat
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Basic metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # R² calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')

    # Directional accuracy
    if len(y_true) > 1:
        true_direction = np.sign(y_true[1:] - y_true[:-1])
        pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
        direction_correct = np.sum(true_direction == pred_direction)
        directional_accuracy = (direction_correct / (len(y_true) - 1)) * 100
    else:
        directional_accuracy = 0.0

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'Directional_Accuracy_%': directional_accuracy
    }


def print_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                            title: str = "Model Performance Comparison"):
    """
    Print formatted comparison of metrics across multiple models.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        title: Title for the comparison table
    """
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")

    metric_names = list(next(iter(metrics_dict.values())).keys())
    model_names = list(metrics_dict.keys())

    header = f"{'Metric':<25}"
    for model_name in model_names:
        header += f"{model_name:>20}"
    print(header)
    print("-" * len(header))

    for metric in metric_names:
        row = f"{metric:<25}"
        for model_name in model_names:
            value = metrics_dict[model_name][metric]
            row += f"{value:>20.4f}"
        print(row)

    print(f"{'='*80}\n")


def calculate_model_performance_summary(y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      model_name: str = "Model") -> Dict[str, float]:
    """
    Calculate and format comprehensive performance summary for a single model.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        model_name: Name of the model for reporting

    Returns:
        Dictionary of performance metrics
    """
    metrics = calculate_metrics(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name} Performance Summary")
    print(f"{'='*60}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:25s}: {value:10.4f}")
    print(f"{'='*60}\n")

    return metrics