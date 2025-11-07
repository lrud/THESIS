"""
Evaluation utilities for HAR-RV model.

Note: This module uses consolidated metrics from scripts.utils.metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import sys
import os

# Add parent directory to path for utils import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.metrics import calculate_metrics, print_metrics_comparison


def plot_predictions_comparison(y_true: np.ndarray,
                               predictions_dict: Dict[str, np.ndarray],
                               dataset_name: str = "Test Set",
                               save_path: Optional[str] = None,
                               max_samples: int = 500):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    n_samples = min(len(y_true), max_samples)
    indices = range(n_samples)
    y_true_plot = y_true[:n_samples]
    colors = plt.cm.Set2(range(len(predictions_dict)))
    
    ax1 = axes[0]
    ax1.plot(indices, y_true_plot, label='Actual', 
             color='black', linewidth=2, alpha=0.8, linestyle='-')
    
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        y_pred_plot = y_pred[:n_samples]
        ax1.plot(indices, y_pred_plot, label=model_name,
                color=colors[i], linewidth=1.5, alpha=0.7, linestyle='--')
    
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('DVOL', fontsize=12)
    ax1.set_title(f'{dataset_name} - Predictions vs Actuals', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax2 = axes[1]
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        residuals = (y_true - y_pred)[:n_samples]
        ax2.scatter(indices, residuals, label=f'{model_name} Residuals',
                   color=colors[i], alpha=0.5, s=10)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax2.set_title('Prediction Residuals', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_scatter_comparison(y_true: np.ndarray,
                           predictions_dict: Dict[str, np.ndarray],
                           dataset_name: str = "Test Set",
                           save_path: Optional[str] = None):
    n_models = len(predictions_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > n_cols else axes
    
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[i] if n_models > 1 else axes[0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        metrics = calculate_metrics(y_true, y_pred)
        
        ax.set_xlabel('Actual DVOL', fontsize=11)
        ax.set_ylabel('Predicted DVOL', fontsize=11)
        ax.set_title(f'{model_name}\nR² = {metrics["R²"]:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{dataset_name} - Actual vs Predicted', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_metrics_to_csv(metrics_dict: Dict[str, Dict[str, float]],
                        save_path: str):
    import pandas as pd
    df = pd.DataFrame(metrics_dict).T
    df.index.name = 'Model'
    df.to_csv(save_path)
    print(f"Metrics saved to: {save_path}")
    return df
