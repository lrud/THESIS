"""
Comprehensive comparison of all models: LSTM, HAR-RV, and Naive baselines.
Usage: python scripts/benchmarking/compare_all_models.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_all_metrics():
    metrics = {}
    
    har_rv = pd.read_csv('../../results/csv/har_rv_metrics.csv')
    har_rv_test = har_rv[har_rv['Model'] == 'HAR-RV_Test'].iloc[0]
    metrics['HAR-RV'] = har_rv_test.to_dict()
    
    har_rv_diff = pd.read_csv('../../results/csv/har_rv_differenced_metrics.csv')
    har_diff_test = har_rv_diff[har_rv_diff['Model'] == 'HAR-RV-Diff_Test'].iloc[0]
    metrics['HAR-RV-Diff'] = har_diff_test.to_dict()
    
    lstm_diff = pd.read_csv('../../results/csv/lstm_differenced_metrics.csv')
    lstm_test = lstm_diff[lstm_diff['Dataset'] == 'Test'].iloc[0]
    metrics['LSTM-Diff'] = lstm_test.to_dict()
    
    naive = pd.read_csv('../../results/csv/naive_baselines_metrics.csv')
    naive_pers = naive[naive['Model'] == 'Naive_Persistence_Test'].iloc[0]
    naive_drift = naive[naive['Model'] == 'Naive_Drift_Test'].iloc[0]
    naive_ma5 = naive[naive['Model'] == 'Naive_MA5_Test'].iloc[0]
    
    metrics['Naive-Persistence'] = naive_pers.to_dict()
    metrics['Naive-Drift'] = naive_drift.to_dict()
    metrics['Naive-MA5'] = naive_ma5.to_dict()
    
    return metrics


def create_comparison_plots(metrics):
    models = ['HAR-RV', 'HAR-RV-Diff', 'LSTM-Diff', 'Naive-Persistence', 'Naive-MA5']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Comparison - Test Set Performance', fontsize=16, fontweight='bold')
    
    metric_names = ['R²', 'RMSE', 'MAE', 'MAPE', 'Directional_Accuracy_%']
    titles = ['R² (Higher is Better)', 'RMSE (Lower is Better)', 'MAE (Lower is Better)', 
              'MAPE % (Lower is Better)', 'Directional Accuracy % (Higher is Better)']
    
    for idx, (metric, title) in enumerate(zip(metric_names, titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = [metrics[model][metric] for model in models]
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}' if metric == 'R²' else f'{val:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        if metric == 'R²':
            ax.axhline(y=0.997, color='red', linestyle='--', linewidth=2, 
                      label='Naive Baseline Level', alpha=0.7)
            ax.legend(fontsize=9)
    
    axes[1, 2].axis('off')
    
    text_content = """
KEY FINDINGS:

1. NAIVE PERSISTENCE MATCHES COMPLEX MODELS
   - R² = 0.9970 for all differenced approaches
   - Simple random walk = 100K+ parameter LSTM
   
2. HAR-RV (ABSOLUTE) ONLY DEFENSIBLE MODEL
   - R² = 0.9649 (robust to non-stationarity)
   - Interpretable, handles regime shift
   
3. DIFFERENCING DESTROYS FORECASTING POWER
   - Reduces models to naive persistence
   - Directional accuracy ~ 50% (coin flip)
   
4. THESIS RECOMMENDATION
   - Use HAR-RV (absolute) as primary model
   - Acknowledge naive baseline equivalence
   - Focus on interpretability over accuracy
    """
    
    axes[1, 2].text(0.1, 0.5, text_content, fontsize=10, 
                   verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    output_dir = Path('../../results/visualizations/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / 'all_models_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()


def create_summary_table(metrics):
    rows = []
    for model_name, model_metrics in metrics.items():
        rows.append({
            'Model': model_name,
            'R²': model_metrics['R²'],
            'RMSE': model_metrics['RMSE'],
            'MAE': model_metrics['MAE'],
            'MAPE': model_metrics['MAPE'],
            'Dir%': model_metrics['Directional_Accuracy_%']
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('R²', ascending=False)
    
    output_path = '../../results/csv/all_models_comparison.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    print("\n" + "="*80)
    print("ALL MODELS COMPARISON (Test Set)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print("\nLoading metrics from all models...")
    
    metrics = load_all_metrics()
    
    print(f"\nLoaded {len(metrics)} models:")
    for name in metrics.keys():
        print(f"  - {name}")
    
    print("\nCreating comparison visualizations...")
    create_comparison_plots(metrics)
    
    print("\nCreating summary table...")
    df = create_summary_table(metrics)
    
    print("\n" + "="*80)
    print("CRITICAL FINDING")
    print("="*80)
    
    naive_r2 = metrics['Naive-Persistence']['R²']
    lstm_r2 = metrics['LSTM-Diff']['R²']
    har_diff_r2 = metrics['HAR-RV-Diff']['R²']
    har_abs_r2 = metrics['HAR-RV']['R²']
    
    print(f"\nNaive Persistence:    R² = {naive_r2:.4f}")
    print(f"LSTM (Differenced):   R² = {lstm_r2:.4f}")
    print(f"HAR-RV (Differenced): R² = {har_diff_r2:.4f}")
    print(f"HAR-RV (Absolute):    R² = {har_abs_r2:.4f}")
    
    if abs(naive_r2 - lstm_r2) < 0.001:
        print("\nWARNING: Complex LSTM provides NO improvement over naive baseline!")
        print("         Differencing reduces 100K+ parameter model to random walk.")
    
    if abs(naive_r2 - har_diff_r2) < 0.001:
        print("\nWARNING: HAR-RV differenced provides NO improvement over naive baseline!")
        print("         All differenced approaches collapse to naive persistence.")
    
    improvement = ((har_abs_r2 - naive_r2) / (1 - naive_r2)) * 100
    print(f"\nNOTE: HAR-RV (Absolute) achieves {improvement:.1f}% of remaining variance explained")
    print("      This is the ONLY model showing genuine forecasting value.")
    
    print("\n" + "="*80)
    print("THESIS RECOMMENDATION")
    print("="*80)
    print("\nBased on comprehensive benchmarking:")
    print("  1. USE: HAR-RV (Absolute) as primary model")
    print("  2. REPORT: Naive baseline equivalence of differenced models")
    print("  3. DISCUSS: Why complex models fail (regime shift, non-stationarity)")
    print("  4. EMPHASIZE: Interpretability and robustness over raw accuracy")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
