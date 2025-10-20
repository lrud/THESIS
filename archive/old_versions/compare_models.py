"""
Comparison between LSTM and HAR-RV models.
Usage: python scripts/benchmarking/compare_models.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from evaluator_har import print_metrics_comparison, plot_predictions_comparison


def load_lstm_results():
    try:
        lstm_metrics = pd.read_csv('results/csv/lstm_differenced_metrics.csv', index_col=0)
        print("Loaded LSTM metrics")
        return lstm_metrics
    except FileNotFoundError:
        print("LSTM metrics not found. Run LSTM model first.")
        return None


def load_har_rv_results():
    try:
        har_metrics = pd.read_csv('results/csv/har_rv_metrics.csv', index_col=0)
        print("Loaded HAR-RV metrics")
        return har_metrics
    except FileNotFoundError:
        print("HAR-RV metrics not found. Run HAR-RV model first.")
        return None


def create_comparison_table(lstm_metrics, har_metrics):
    lstm_test = lstm_metrics.loc['test'] if 'test' in lstm_metrics.index else lstm_metrics.iloc[-1]
    har_test = har_metrics.loc['HAR-RV_Test'] if 'HAR-RV_Test' in har_metrics.index else har_metrics.iloc[-1]
    
    comparison = {
        'LSTM (Differenced)': lstm_test.to_dict(),
        'HAR-RV': har_test.to_dict()
    }
    
    print_metrics_comparison(comparison, title="LSTM vs HAR-RV: Test Set Performance")
    
    print(f"\n{'='*80}")
    print(f"RELATIVE IMPROVEMENTS (LSTM vs HAR-RV)")
    print(f"{'='*80}\n")
    
    for metric in lstm_test.index:
        lstm_val = lstm_test[metric]
        har_val = har_test[metric]
        
        if metric in ['R²', 'Directional_Accuracy_%']:
            improvement = ((lstm_val - har_val) / abs(har_val)) * 100 if har_val != 0 else 0
            better_model = 'LSTM' if lstm_val > har_val else 'HAR-RV'
        else:
            improvement = ((har_val - lstm_val) / abs(har_val)) * 100 if har_val != 0 else 0
            better_model = 'LSTM' if lstm_val < har_val else 'HAR-RV'
        
        print(f"{metric:<25}: {improvement:>8.2f}% ({better_model} better)")
    
    print(f"{'='*80}\n")
    
    return comparison


def plot_comparative_bar_chart(comparison, save_path='results/visualizations/model_comparison_bars.png'):
    """
    Create bar chart comparing model performances.
    
    Args:
        comparison: Dictionary with model metrics
        save_path: Path to save the plot
    """
    # Prepare data
    models = list(comparison.keys())
    metrics = ['R²', 'RMSE', 'MAE', 'MAPE', 'Directional_Accuracy_%']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c']  # Blue for LSTM, Red for HAR-RV
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [comparison[model][metric] for model in models]
        
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight better model
        if metric in ['R²', 'Directional_Accuracy_%']:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
    
    # Hide the 6th subplot
    axes[5].set_visible(False)
    
    plt.suptitle('Model Performance Comparison (Test Set)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_executive_summary(comparison, save_path='results/HAR_RV_BENCHMARK_SUMMARY.md'):
    """
    Create a markdown summary document.
    
    Args:
        comparison: Dictionary with model metrics
        save_path: Path to save the summary
    """
    lstm_metrics = comparison['LSTM (Differenced)']
    har_metrics = comparison['HAR-RV']
    
    summary = f"""# HAR-RV Benchmark Results

## Executive Summary

This document summarizes the performance of the HAR-RV benchmark model compared to the LSTM model for forecasting Bitcoin DVOL (implied volatility).

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}

---

## Model Specifications

### LSTM (Differenced)
- **Architecture:** 2-layer LSTM with 128 hidden units
- **Input:** 24-hour sequences of 7 features
- **Target:** First differences of DVOL (Δdvol)
- **Regularization:** 0.3 dropout, 1e-4 L2 penalty
- **Training:** Early stopping with patience=15

### HAR-RV
- **Architecture:** Linear regression with 3 components
- **Components:**
  - Daily: DVOL_t (lag-1)
  - Weekly: 5-day average
  - Monthly: 22-day average
- **Target:** DVOL (absolute values, not differenced)
- **Parameters:** 4 (intercept + 3 coefficients)

---

## Test Set Performance

| Metric | LSTM (Differenced) | HAR-RV | Winner |
|--------|-------------------|---------|--------|
| **R²** | {lstm_metrics['R²']:.4f} | {har_metrics['R²']:.4f} | {'LSTM' if lstm_metrics['R²'] > har_metrics['R²'] else 'HAR-RV'} |
| **RMSE** | {lstm_metrics['RMSE']:.4f} | {har_metrics['RMSE']:.4f} | {'LSTM' if lstm_metrics['RMSE'] < har_metrics['RMSE'] else 'HAR-RV'} |
| **MAE** | {lstm_metrics['MAE']:.4f} | {har_metrics['MAE']:.4f} | {'LSTM' if lstm_metrics['MAE'] < har_metrics['MAE'] else 'HAR-RV'} |
| **MAPE** | {lstm_metrics['MAPE']:.2f}% | {har_metrics['MAPE']:.2f}% | {'LSTM' if lstm_metrics['MAPE'] < har_metrics['MAPE'] else 'HAR-RV'} |
| **Directional Accuracy** | {lstm_metrics['Directional_Accuracy_%']:.2f}% | {har_metrics['Directional_Accuracy_%']:.2f}% | {'LSTM' if lstm_metrics['Directional_Accuracy_%'] > har_metrics['Directional_Accuracy_%'] else 'HAR-RV'} |

---

## Key Findings

### 1. Model Complexity vs Performance

"""
    
    # Add interpretation
    if lstm_metrics['R²'] > har_metrics['R²'] + 0.1:
        summary += """- **LSTM shows substantial improvement** over HAR-RV, suggesting that the additional complexity and feature engineering provide meaningful predictive power.
- The non-linear patterns captured by LSTM are valuable for DVOL forecasting.
"""
    elif abs(lstm_metrics['R²'] - har_metrics['R²']) < 0.1:
        summary += """- **Performance is comparable** between LSTM and HAR-RV, suggesting that much of DVOL's predictability comes from linear autocorrelation patterns.
- This raises questions about whether LSTM's complexity is justified.
- Consider using HAR-RV for production due to its simplicity and interpretability.
"""
    else:
        summary += """- **HAR-RV outperforms LSTM**, which is concerning and suggests:
  - Potential overfitting in the LSTM model
  - The differencing transformation may have removed valuable information
  - Simple linear patterns dominate DVOL dynamics
"""
    
    summary += f"""
### 2. Directional Accuracy

- LSTM: {lstm_metrics['Directional_Accuracy_%']:.2f}%
- HAR-RV: {har_metrics['Directional_Accuracy_%']:.2f}%

Both models show directional accuracy near 50%, suggesting that while they capture magnitude well, predicting the direction of volatility changes remains challenging.

### 3. Error Magnitudes

- LSTM MAPE: {lstm_metrics['MAPE']:.2f}%
- HAR-RV MAPE: {har_metrics['MAPE']:.2f}%

"""
    
    if lstm_metrics['MAPE'] < 5:
        summary += "LSTM shows excellent accuracy with very low percentage errors.\n"
    
    summary += f"""
---

## Recommendations

### Next Steps

1. **Additional Benchmarks:**
   - Implement GARCH model for volatility clustering
   - Test naive persistence (today's value = tomorrow's forecast)
   - Try ARIMA/ARFIMA for autoregressive baselines

2. **Model Refinement:**
"""
    
    if lstm_metrics['R²'] < har_metrics['R²']:
        summary += """   - Investigate LSTM overfitting concerns
   - Consider simpler LSTM architecture
   - Try alternative target transformations (percentage changes, log returns)
   - Implement rolling-window normalization
"""
    else:
        summary += """   - Explore attention mechanisms for LSTM
   - Test ensemble methods (LSTM + HAR-RV)
   - Investigate feature importance in LSTM
"""
    
    summary += """
3. **Production Considerations:**
   - HAR-RV is computationally efficient and interpretable
   - LSTM requires GPU for real-time inference
   - Consider hybrid approach: HAR-RV for baseline, LSTM for refinement

---

## Visualizations

See `results/visualizations/` for:
- `model_comparison_bars.png` - Side-by-side metric comparison
- `har_rv_predictions_test.png` - HAR-RV test set predictions
- `har_rv_scatter_test.png` - Actual vs predicted scatter plot

---

## Conclusion

"""
    
    if lstm_metrics['R²'] > 0.5 and har_metrics['R²'] > 0.5:
        summary += "Both models demonstrate strong predictive power for DVOL forecasting. "
    
    if abs(lstm_metrics['R²'] - har_metrics['R²']) < 0.05:
        summary += """The similar performance between LSTM and HAR-RV suggests that DVOL dynamics are largely driven by linear autocorrelation patterns. The added complexity of LSTM may not be justified unless substantial improvements can be achieved through further tuning or architecture changes.
"""
    elif lstm_metrics['R²'] > har_metrics['R²']:
        summary += """LSTM's superior performance validates the use of deep learning for this task, though the relatively modest improvement over HAR-RV should be considered in production deployment decisions.
"""
    else:
        summary += """HAR-RV's superior performance is a red flag for the LSTM model and warrants investigation into potential overfitting or methodological issues.
"""
    
    with open(save_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {save_path}")


def main():
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON: LSTM vs HAR-RV")
    print(f"{'='*80}\n")
    
    print("Loading model results...")
    lstm_metrics = load_lstm_results()
    har_metrics = load_har_rv_results()
    
    if lstm_metrics is None or har_metrics is None:
        print("\nCannot proceed without both model results.")
        print("Please run both models first:")
        print("  1. python scripts/modeling/main_differenced.py")
        print("  2. python scripts/benchmarking/main_har_rv.py")
        return
    
    print("\nCreating comparison...")
    comparison = create_comparison_table(lstm_metrics, har_metrics)
    
    print("\nGenerating visualizations...")
    plot_comparative_bar_chart(comparison)
    
    print(f"\n{'='*80}")
    print(f"Comparison complete")
    print(f"{'='*80}\n")
    print(f"Output files:")
    print(f"  - results/visualizations/model_comparison_bars.png")
    print(f"\n")


if __name__ == '__main__':
    main()
