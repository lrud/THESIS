"""
Create effective visualizations for model comparison that avoid R² skew and highlight directional accuracy.
Focus on telling the story: trivial solutions vs genuine forecasting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns


def load_all_metrics():
    """Load all model metrics."""
    metrics = {}

    try:
        # Naive models
        naive = pd.read_csv('../../results/csv/metrics/naive_baselines_metrics.csv')
        naive_pers = naive[naive['Model'] == 'Naive_Persistence_Test'].iloc[0]
        naive_drift = naive[naive['Model'] == 'Naive_Drift_Test'].iloc[0]
        naive_ma5 = naive[naive['Model'] == 'Naive_MA5_Test'].iloc[0]

        metrics['Naive-Persistence'] = naive_pers.to_dict()
        metrics['Naive-Drift'] = naive_drift.to_dict()
        metrics['Naive-MA5'] = naive_ma5.to_dict()

        # HAR-RV models
        har_rv = pd.read_csv('../../results/csv/metrics/har_rv_metrics.csv')
        har_rv_test = har_rv[har_rv['Model'] == 'HAR-RV_Test'].iloc[0]
        metrics['HAR-RV'] = har_rv_test.to_dict()

        har_rv_diff = pd.read_csv('../../results/csv/metrics/har_rv_differenced_metrics.csv')
        har_diff_test = har_rv_diff[har_rv_diff['Model'] == 'HAR-RV-Diff_Test'].iloc[0]
        metrics['HAR-RV-Diff'] = har_diff_test.to_dict()

        # LSTM models
        lstm_diff = pd.read_csv('../../results/csv/metrics/lstm_differenced_metrics.csv')
        lstm_test = lstm_diff[lstm_diff['Dataset'] == 'Test'].iloc[0]
        metrics['LSTM-Diff'] = lstm_test.to_dict()

        lstm_rolling = pd.read_csv('../../results/csv/metrics/lstm_rolling_metrics.csv')
        rolling_test = lstm_rolling[lstm_rolling['Dataset'] == 'Test'].iloc[0]
        metrics['LSTM-Rolling'] = rolling_test.to_dict()

        lstm_jump = pd.read_csv('../../results/csv/metrics/lstm_jump_aware_metrics.csv')
        jump_test = lstm_jump[lstm_jump['split'] == 'test'].iloc[0]
        metrics['LSTM-Jump-Aware'] = jump_test.to_dict()

    except Exception as e:
        print(f"Warning: Could not load some metrics: {e}")

    return metrics


def get_metric_value(model_dict, metric_name):
    """Helper function to get metric value with different naming conventions."""
    if metric_name == 'R²':
        if 'R²' in model_dict:
            return float(model_dict['R²'])
        elif 'overall_r2' in model_dict:
            return float(model_dict['overall_r2'])
    elif metric_name == 'Directional_Accuracy_%':
        if 'Directional_Accuracy_%' in model_dict:
            return float(model_dict['Directional_Accuracy_%'])
        elif 'overall_directional_accuracy' in model_dict:
            return float(model_dict['overall_directional_accuracy'])
    elif metric_name == 'MAPE':
        if 'MAPE' in model_dict:
            return float(model_dict['MAPE'])
        elif 'overall_mape' in model_dict:
            return float(model_dict['overall_mape'])
    elif metric_name == 'RMSE':
        if 'RMSE' in model_dict:
            return float(model_dict['RMSE'])
        elif 'overall_rmse' in model_dict:
            return float(model_dict['overall_rmse'])
    return None


def create_trivial_vs_genuine_plot(metrics):
    """Plot 1: Trivial vs Genuine Solutions - R² focus."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Get models and classify
    models = list(metrics.keys())
    r2_values = [get_metric_value(m, 'R²') for m in metrics.values()]

    naive_r2 = get_metric_value(metrics['Naive-Persistence'], 'R²')

    # Classify models
    trivial_models = []
    genuine_models = []

    for model_name, model_metrics in metrics.items():
        r2 = get_metric_value(model_metrics, 'R²')
        if abs(r2 - naive_r2) < 0.01:  # Within 1% of naive
            trivial_models.append((model_name, r2))
        elif r2 < 0.95:  # Significantly different from naive
            genuine_models.append((model_name, r2))

    # Left plot: All models with focus on R² range
    all_names = [m[0] for m in trivial_models + genuine_models]
    all_r2 = [m[1] for m in trivial_models + genuine_models]

    colors = ['#E74C3C'] * len(trivial_models) + ['#27AE60'] * len(genuine_models)

    bars1 = ax1.bar(range(len(all_names)), all_r2, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add reference line for naive baseline
    ax1.axhline(y=naive_r2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Naive Baseline')

    # Formatting
    ax1.set_title('All Models: R² Performance\n(Red = Trivial, Green = Genuine)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R²', fontsize=12)
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_xticks(range(len(all_names)))
    ax1.set_xticklabels(all_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0.85, 1.0)

    # Add value labels on bars
    for bar, val in zip(bars1, all_r2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Right plot: Focus on genuine models with better scale
    if genuine_models:
        genuine_names = [m[0] for m in genuine_models]
        genuine_r2 = [m[1] for m in genuine_models]

        bars2 = ax2.bar(range(len(genuine_names)), genuine_r2, color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax2.set_title('Genuine Forecasting Models Only\n(R² Range Expanded)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('R²', fontsize=12)
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_xticks(range(len(genuine_names)))
        ax2.set_xticklabels(genuine_names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0.8, 0.9)

        # Add value labels
        for bar, val in zip(bars2, genuine_r2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Genuine Models Found', ha='center', va='center', fontsize=14)
        ax2.set_title('No Genuine Forecasting Models', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_directional_accuracy_plot(metrics):
    """Plot 2: Directional Accuracy - The Real Story."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    models = list(metrics.keys())
    dir_acc = [get_metric_value(m, 'Directional_Accuracy_%') for m in metrics.values()]

    # Color based on whether model is trivial or genuine
    naive_r2 = get_metric_value(metrics['Naive-Persistence'], 'R²')
    colors = []
    for model_name, model_metrics in metrics.items():
        r2 = get_metric_value(model_metrics, 'R²')
        if abs(r2 - naive_r2) < 0.01:  # Trivial
            colors.append('#E74C3C')
        else:  # Genuine
            colors.append('#27AE60')

    # Create horizontal bar chart
    bars = ax.barh(range(len(models)), dir_acc, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add 50% reference line (random guessing)
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Guessing (50%)')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, dir_acc)):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')

    # Add model labels
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # Formatting
    ax.set_title('Directional Accuracy: The Real Performance Indicator\n(Red = Trivial, Green = Genuine)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Directional Accuracy (%)', fontsize=14)
    ax.set_ylabel('Models', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(45, max(dir_acc) + 5)

    # Add annotations for key insights
    ax.annotate('Genuine forecasting\n(>52% direction)',
                xy=(52.8, 6), xytext=(55, 6),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, fontweight='bold')

    if 'LSTM-Jump-Aware' in models:
        jump_idx = models.index('LSTM-Jump-Aware')
        ax.annotate('Crisis robust\n(54.1% crisis accuracy)',
                    xy=(48.8, jump_idx), xytext=(60, jump_idx),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


def create_effectiveness_scatter(metrics):
    """Plot 3: R² vs Directional Accuracy Scatter."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    models = list(metrics.keys())
    r2_values = [get_metric_value(m, 'R²') for m in metrics.values()]
    dir_acc = [get_metric_value(m, 'Directional_Accuracy_%') for m in metrics.values()]

    # Classify and color
    naive_r2 = get_metric_value(metrics['Naive-Persistence'], 'R²')
    colors = []
    sizes = []

    for model_name, model_metrics in metrics.items():
        r2 = get_metric_value(model_metrics, 'R²')
        if abs(r2 - naive_r2) < 0.01:  # Trivial
            colors.append('#E74C3C')
            sizes.append(100)
        else:  # Genuine
            colors.append('#27AE60')
            sizes.append(150)

    # Create scatter plot
    scatter = ax.scatter(r2_values, dir_acc, c=colors, s=sizes, alpha=0.8, edgecolor='black', linewidth=2)

    # Add reference lines
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Guessing')
    ax.axvline(x=naive_r2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Naive Baseline R²')

    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (r2_values[i], dir_acc[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    # Create quadrants
    ax.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=52, color='gray', linestyle=':', alpha=0.5)

    # Quadrant labels
    ax.text(0.987, 56, 'Trivial\nSolutions', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='#E74C3C', alpha=0.3))
    ax.text(0.88, 56, 'Genuine\nForecasting', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='#27AE60', alpha=0.3))

    # Formatting
    ax.set_title('R² vs Directional Accuracy: Model Effectiveness\n(Green = Good, Red = Illusory)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('R²', fontsize=14)
    ax.set_ylabel('Directional Accuracy (%)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.85, 1.0)
    ax.set_ylim(45, 60)

    plt.tight_layout()
    return fig


def create_maPE_comparison(metrics):
    """Plot 4: MAPE - Practical Error Metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    models = list(metrics.keys())
    mape_values = [get_metric_value(m, 'MAPE') for m in metrics.values()]

    # Color based on classification
    naive_r2 = get_metric_value(metrics['Naive-Persistence'], 'R²')
    colors = []
    for model_name, model_metrics in metrics.items():
        r2 = get_metric_value(model_metrics, 'R²')
        if abs(r2 - naive_r2) < 0.01:  # Trivial
            colors.append('#E74C3C')
        else:  # Genuine
            colors.append('#27AE60')

    # Create bar chart
    bars = ax.bar(models, mape_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, mape_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(mape_values)*0.02,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_title('MAPE: Practical Forecasting Error\n(Lower = Better, but must be > trivial)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_xlabel('Models', fontsize=12)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("\n" + "="*80)
    print("CREATING EFFECTIVE MODEL COMPARISON VISUALIZATIONS")
    print("="*80)

    metrics = load_all_metrics()
    print(f"Loaded {len(metrics)} models")

    # Create output directory
    output_dir = Path('../../results/visualizations/effective_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    print("\n1. Creating Trivial vs Genuine R² comparison...")
    fig1 = create_trivial_vs_genuine_plot(metrics)
    fig1.savefig(output_dir / 'trivial_vs_genuine_r2.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'trivial_vs_genuine_r2.png'}")

    print("\n2. Creating Directional Accuracy focus...")
    fig2 = create_directional_accuracy_plot(metrics)
    fig2.savefig(output_dir / 'directional_accuracy_focus.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'directional_accuracy_focus.png'}")

    print("\n3. Creating Effectiveness Scatter...")
    fig3 = create_effectiveness_scatter(metrics)
    fig3.savefig(output_dir / 'effectiveness_scatter.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'effectiveness_scatter.png'}")

    print("\n4. Creating MAPE comparison...")
    fig4 = create_maPE_comparison(metrics)
    fig4.savefig(output_dir / 'mape_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'mape_comparison.png'}")

    plt.close('all')

    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    print("\nFour effective visualizations created:")
    print("1. Trivial vs Genuine R² - Shows the statistical illusion")
    print("2. Directional Accuracy Focus - Real forecasting performance")
    print("3. Effectiveness Scatter - R² vs Directional trade-offs")
    print("4. MAPE Comparison - Practical error metrics")
    print("\nThese tell the story: High R² ≠ Good Model")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()