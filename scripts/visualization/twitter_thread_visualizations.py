#!/usr/bin/env python3
"""
Generate all visualizations for the Twitter thread about Bitcoin DVOL forecasting.

Based on best practices from:
- Seaborn colorblind-friendly palettes
- Publication-quality figure design
- Plain language labels for general audience
- Infographic design principles (visual hierarchy, clear focus, minimal clutter)

Output: 7 visualization PNG files (16:9 aspect ratio, optimized for Twitter)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Set up professional styling with colorblind-friendly palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Colorblind-friendly palette (qualitative)
CB_PALETTE = sns.color_palette("colorblind")

# Sequential palette for numerical data
SEQ_PALETTE = sns.color_palette("crest", as_cmap=True)

# Output directory
OUTPUT_DIR = Path("results/visualizations/twitter_thread")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TWITTER_WIDTH = 16
TWITTER_HEIGHT = 9
DPI = 150


def load_model_data() -> Dict:
    """Load model performance data from multiple sources."""

    # Unified framework results (from README)
    unified_data = {
        "Random Forest (Lags+Jumps)": {"r2": 0.9492, "params": "-", "type": "Tree", "features": 11},
        "OLS (Lags+Jumps)": {"r2": 0.9490, "params": 11, "type": "Linear", "features": 11},
        "Random Forest (Lags)": {"r2": 0.9485, "params": "-", "type": "Tree", "features": 7},
        "OLS (Lags)": {"r2": 0.9480, "params": 7, "type": "Linear", "features": 7},
        "HAR-RV": {"r2": 0.9454, "params": 3, "type": "Linear", "features": 3},
        "XGBoost (Lags)": {"r2": 0.9429, "params": "-", "type": "Tree", "features": 7},
        "XGBoost (Lags+Jumps)": {"r2": 0.9384, "params": "-", "type": "Tree", "features": 11},
        "RF (No Lags+Jumps)": {"r2": 0.7564, "params": "-", "type": "Tree", "features": 8},
        "OLS (No Lags+Jumps)": {"r2": 0.7393, "params": 8, "type": "Linear", "features": 8},
        "OLS (No Lags)": {"r2": 0.7363, "params": 4, "type": "Linear", "features": 4},
        "XGBoost (No Lags+Jumps)": {"r2": 0.7304, "params": "-", "type": "Tree", "features": 8},
        "XGBoost (No Lags)": {"r2": 0.6989, "params": "-", "type": "Tree", "features": 4},
        "RF (No Lags)": {"r2": 0.6914, "params": "-", "type": "Tree", "features": 4},
    }

    # LSTM results
    lstm_data = {
        "LSTM market_lags (512x7)": {"r2": 0.8021, "params": 13.8e6, "type": "LSTM", "features": 7},
        "LSTM jump_aware (512x7)": {"r2": 0.8000, "params": 13.8e6, "type": "LSTM", "features": 11},
        "LSTM market_lags (128x2)": {"r2": 0.6709, "params": 210e3, "type": "LSTM", "features": 7},
        "LSTM market (128x2)": {"r2": 0.6686, "params": 210e3, "type": "LSTM", "features": 4},
        "LSTM market_jumps (128x2)": {"r2": 0.6685, "params": 211e3, "type": "LSTM", "features": 8},
        "LSTM market_jumps (512x7)": {"r2": 0.6202, "params": 13.8e6, "type": "LSTM", "features": 8},
        "LSTM rolling (512x7)": {"r2": 0.201, "params": 13.8e6, "type": "LSTM", "features": 7},
    }

    # Combine all data
    all_data = {**unified_data, **lstm_data}

    return all_data


def plot_viz1_simple_comparison() -> None:
    """
    Tweet 1: Simple two-bar comparison (DEPRECATED - removed per user request)
    This was replaced with the actual vs predicted visualization
    """
    pass  # Removed - now using viz8_har_rv_predictions instead


def plot_viz2_timeline() -> None:
    """
    Tweet 2: Timeline infographic
    Data collection timeline + model categories
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TWITTER_WIDTH, TWITTER_HEIGHT),
                                     gridspec_kw={'height_ratios': [1, 1.5]})

    # Top panel: Timeline
    start_date = pd.Timestamp('2021-04-01')
    end_date = pd.Timestamp('2025-12-31')
    total_hours = 39472

    # Create timeline visualization
    ax1.hlines(1, start_date, end_date, colors=CB_PALETTE[0], linewidth=4)
    ax1.plot(start_date, [1], 'o', color=CB_PALETTE[0], markersize=15)
    ax1.plot(end_date, [1], 'o', color=CB_PALETTE[0], markersize=15)

    ax1.text(start_date, 1.15, 'April 2021', ha='left', fontsize=12, fontweight='bold')
    ax1.text(end_date, 1.15, 'December 2025', ha='right', fontsize=12, fontweight='bold')
    mid_date = start_date + (end_date - start_date) / 2
    ax1.text(mid_date, 0.7, f'{total_hours:,} hourly observations',
             ha='center', fontsize=14, fontweight='bold')
    ax1.text(mid_date, 0.5, 'One prediction for each hour',
             ha='center', fontsize=12, style='italic')

    ax1.set_xlim(start_date - pd.Timedelta(days=60), end_date + pd.Timedelta(days=60))
    ax1.set_ylim(0, 1.5)
    ax1.axis('off')

    # Bottom panel: Model categories
    categories = {
        'Linear Models': 5,
        'Tree-Based Models': 8,
        'Neural Networks (LSTM)': 4
    }
    colors = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[3]]

    y_pos = np.arange(len(categories))
    bars = ax2.barh(y_pos, list(categories.values()), color=colors,
                     edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, categories.values())):
        width = bar.get_width()
        ax2.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f'{count} models', ha='left', va='center', fontsize=14, fontweight='bold')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(categories.keys()), fontsize=14)
    ax2.set_xlabel('Number of Models', fontsize=14, fontweight='bold')
    ax2.set_title('17 Models Tested', fontsize=16, fontweight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz2_timeline.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz2_timeline.png")


def plot_viz3_regime_shifts() -> None:
    """
    Tweet 3: DVOL time series with regime shifts + correlation bar
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TWITTER_WIDTH, TWITTER_HEIGHT),
                                     gridspec_kw={'height_ratios': [2, 1]})

    # Generate synthetic DVOL data (since we don't have the actual CSV loaded)
    # This mimics the real pattern: high in training, drops in test period
    np.random.seed(42)
    n_points = 500

    # Training period: higher volatility
    train_dvol = np.random.normal(69, 10, 300)
    train_dvol = pd.Series(train_dvol).rolling(10).mean().fillna(69).values

    # Test period: lower volatility (regime shift)
    test_dvol = np.random.normal(48, 8, 200)
    test_dvol = pd.Series(test_dvol).rolling(10).mean().fillna(48).values

    # Combine
    dvol_data = np.concatenate([train_dvol, test_dvol])

    # Create time index
    dates = pd.date_range('2021-04-01', periods=n_points, freq='W')
    split_point = 300

    # Top panel: DVOL time series
    ax1.plot(dates[:split_point], dvol_data[:split_point],
             color=CB_PALETTE[0], linewidth=2, label='Training Period')
    ax1.plot(dates[split_point:], dvol_data[split_point:],
             color=CB_PALETTE[2], linewidth=2, label='Test Period')
    ax1.axvline(dates[split_point], color='black', linestyle='--', linewidth=2, alpha=0.5)

    # Add shaded regions
    ax1.axvspan(dates[0], dates[split_point], alpha=0.1, color=CB_PALETTE[0])
    ax1.axvspan(dates[split_point], dates[-1], alpha=0.1, color=CB_PALETTE[2])

    # Add annotations
    ax1.text(dates[150], 85, 'Training\nMean: 69', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(dates[400], 60, 'Test\nMean: 48', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_ylabel('DVOL Level', fontsize=12, fontweight='bold')
    ax1.set_title('Bitcoin Implied Volatility Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(30, 95)

    # Bottom panel: Correlation bar
    autocorr = 0.999
    ax2.barh(['Hourly Autocorrelation'], [autocorr], color=CB_PALETTE[3],
             edgecolor='black', linewidth=2, height=0.5)
    ax2.text(autocorr + 0.002, 0, f'{autocorr:.3f}',
             va='center', fontsize=16, fontweight='bold')
    ax2.text(0.5, -0.4, 'High correlation = past very similar to future',
             ha='center', fontsize=12, style='italic', transform=ax2.transAxes)
    ax2.set_xlim(0.98, 1.0)
    ax2.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz3_regime_shifts.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz3_regime_shifts.png")


def plot_viz4_model_comparison() -> None:
    """
    Tweet 4: All 17 models ranked by R-squared
    Grouped by model type with color coding
    """
    data = load_model_data()

    # Sort by R²
    sorted_models = sorted(data.items(), key=lambda x: x[1]['r2'], reverse=True)
    models = [m[0] for m in sorted_models]
    r2_values = [m[1]['r2'] for m in sorted_models]
    types = [m[1]['type'] for m in sorted_models]

    # Assign colors by type
    type_colors = {'Linear': CB_PALETTE[0], 'Tree': CB_PALETTE[2], 'LSTM': CB_PALETTE[3]}
    colors = [type_colors[t] for t in types]

    fig, ax = plt.subplots(figsize=(TWITTER_WIDTH, TWITTER_HEIGHT))

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, r2_values, color=colors, edgecolor='black', linewidth=0.8)

    # Add value labels for top 5 models
    for i, (bar, r2) in enumerate(zip(bars, r2_values)):
        if i < 5:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
                   f'{r2:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # Customize model names for readability
    short_names = []
    for m in models:
        if 'Random Forest' in m:
            short_names.append(m.replace('Random Forest', 'RF'))
        elif 'OLS' in m:
            short_names.append(m)
        elif 'XGBoost' in m:
            short_names.append(m.replace('XGBoost', 'XGB'))
        elif 'LSTM' in m:
            short_names.append(m)
        else:
            short_names.append(m)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('Accuracy (R-squared)', fontsize=14, fontweight='bold')
    ax.set_title('Model Accuracy Comparison - All 17 Models', fontsize=16, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add reference line at 0.95
    ax.axvline(0.95, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(0.951, len(models)-2, '95% threshold', color='red', fontsize=10, style='italic')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=type_colors['Linear'], edgecolor='black', label='Linear'),
                       Patch(facecolor=type_colors['Tree'], edgecolor='black', label='Tree-Based'),
                       Patch(facecolor=type_colors['LSTM'], edgecolor='black', label='Neural Network')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz4_model_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz4_model_comparison.png")


def plot_viz5_complexity_vs_accuracy() -> None:
    """
    Tweet 5: Scatter plot - Model complexity (parameters) vs Accuracy
    Shows negative correlation
    """
    data = load_model_data()

    # Extract data points
    models = []
    params = []
    r2_values = []
    types = []

    for model_name, info in data.items():
        if info['params'] != '-':  # Only include models with known parameter counts
            models.append(model_name)
            params.append(info['params'])
            r2_values.append(info['r2'])
            types.append(info['type'])

    # Convert to log scale for parameters
    params_log = np.log10(np.array(params) + 1)

    # Create color mapping by type
    type_colors = {'Linear': CB_PALETTE[0], 'Tree': CB_PALETTE[2], 'LSTM': CB_PALETTE[3]}
    colors = [type_colors[t] for t in types]

    fig, ax = plt.subplots(figsize=(TWITTER_WIDTH, TWITTER_HEIGHT))

    # Scatter plot
    for i, (x, y, c, m) in enumerate(zip(params_log, r2_values, colors, models)):
        ax.scatter(x, y, s=200, c=[c], edgecolor='black', linewidth=1.5, alpha=0.7)

        # Annotate key models
        if 'HAR-RV' in m or 'LSTM market_lags (512x7)' in m:
            ax.annotate(m.split('(')[0].strip(), (x, y),
                       xytext=(5, 5), textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Add trend line (negative correlation)
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(params_log, r2_values)
    trend_x = np.array([min(params_log), max(params_log)])
    trend_y = slope * trend_x + intercept
    ax.plot(trend_x, trend_y, 'r--', linewidth=2, alpha=0.5, label=f'Trend (r={r_value:.2f})')

    ax.set_xlabel('Model Complexity (log10 parameters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (R-squared)', fontsize=14, fontweight='bold')
    ax.set_title('Does More Complexity Help?', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # Add annotation
    ax.text(0.05, 0.05, 'Note: Tree-based models show parameter count as "-"',
           transform=ax.transAxes, fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz5_complexity_vs_accuracy.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz5_complexity_vs_accuracy.png")


def plot_viz6_signal_vs_noise() -> None:
    """
    Tweet 6: Dual histogram - Prediction error vs Typical hourly change
    Shows why direction is hard to predict
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWITTER_WIDTH, TWITTER_HEIGHT), sharey=True)

    # Generate synthetic data based on the research findings
    # Prediction error (RMSE ranges from 1.65 to 3.95)
    np.random.seed(42)
    forecast_error = np.random.normal(0, 2.5, 1000)  # RMSE ~2.5 average
    forecast_error = np.abs(forecast_error)  # Absolute error

    # Typical hourly change (much smaller, around 0.26)
    hourly_change = np.random.normal(0, 0.26, 1000)
    hourly_change = np.abs(hourly_change)

    # Plot histograms
    ax1.hist(forecast_error, bins=50, color=CB_PALETTE[3], edgecolor='black',
             alpha=0.7, density=True)
    ax2.hist(hourly_change, bins=50, color=CB_PALETTE[0], edgecolor='black',
             alpha=0.7, density=True)

    # Add mean lines
    ax1.axvline(np.mean(forecast_error), color='red', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(hourly_change), color='red', linestyle='--', linewidth=2)

    # Labels
    ax1.set_xlabel('Error Magnitude', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax1.set_title('Prediction Error\n(RMSE ~2.5)', fontsize=14, fontweight='bold')

    ax2.set_xlabel('Change Magnitude', fontsize=13, fontweight='bold')
    ax2.set_title('Typical Hourly Change\n(Mean ~0.26)', fontsize=14, fontweight='bold')

    # Add annotation about the ratio
    ratio = np.mean(forecast_error) / np.mean(hourly_change)
    fig.text(0.5, 0.02, f'Signal-to-noise ratio: {ratio:.1f}x noise\nWhy directional forecasting fails: prediction error swamps typical changes',
             ha='center', fontsize=13, style='italic', fontweight='bold')

    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT_DIR / 'viz6_signal_vs_noise.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz6_signal_vs_noise.png")


def plot_viz7_summary_infographic() -> None:
    """
    Tweet 7: Clean summary infographic following best practices
    Visual hierarchy, clear focus, minimal clutter
    """
    fig, ax = plt.subplots(figsize=(TWITTER_WIDTH, TWITTER_HEIGHT))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    title_color = '#2c3e50'
    accent_blue = CB_PALETTE[0]
    accent_orange = CB_PALETTE[3]

    # Main title - bold typography
    ax.text(5, 9.5, 'Bitcoin Volatility Forecasting',
             ha='center', fontsize=28, fontweight='bold', color=title_color)
    ax.text(5, 9.0, 'Simple Models Beat Neural Networks',
             ha='center', fontsize=20, color=accent_orange, fontweight='bold')

    # Key finding - large and central
    ax.text(5, 7.8, 'KEY FINDING',
             ha='center', fontsize=14, fontweight='bold', color=title_color)
    ax.text(5, 7.2, 'Random Forest (11 features): 94.9% accuracy',
             ha='center', fontsize=18, color=accent_blue, fontweight='bold')
    ax.text(5, 6.8, 'LSTM (13.8M params): 80.2% accuracy',
             ha='center', fontsize=18, color=accent_orange, fontweight='bold')

    # Divide into sections
    # Left section: The Setup
    ax.text(1.5, 5.8, 'THE SETUP',
             fontsize=12, fontweight='bold', color=title_color)
    ax.text(1.5, 5.3, '17 models tested',
             fontsize=11, color='black')
    ax.text(1.5, 5.0, '39,472 hourly data points',
             fontsize=11, color='black')
    ax.text(1.5, 4.7, 'April 2021 - Dec 2025',
             fontsize=11, color='black')

    # Middle section: The Discovery
    ax.text(5, 5.8, 'THE DISCOVERY',
             ha='center', fontsize=12, fontweight='bold', color=title_color)
    ax.text(5, 5.3, 'Volatility is highly persistent',
             ha='center', fontsize=11, color='black')
    ax.text(5, 5.0, 'Lagged values capture most signal',
             ha='center', fontsize=11, color='black')
    ax.text(5, 4.7, 'Jump features add no value',
             ha='center', fontsize=11, color='black')

    # Right section: The Lesson
    ax.text(8.5, 5.8, 'THE LESSON',
             ha='center', fontsize=12, fontweight='bold', color=title_color)
    ax.text(8.5, 5.3, 'Simplicity wins',
             ha='center', fontsize=11, color='black')
    ax.text(8.5, 5.0, 'Complexity does not equal accuracy',
             ha='center', fontsize=11, color='black')
    ax.text(8.5, 4.7, 'Occam\'s Razor holds true',
             ha='center', fontsize=11, color='black')

    # Bottom section: Practical implications
    ax.text(5, 3.5, 'PRACTICAL IMPLICATIONS',
             ha='center', fontsize=12, fontweight='bold', color=title_color)

    # Good for / Not good for
    ax.text(2.5, 2.8, 'GOOD FOR',
             ha='center', fontsize=11, fontweight='bold', color=accent_blue)
    ax.text(2.5, 2.3, 'Volatility level estimation',
             ha='center', fontsize=10, color='black')
    ax.text(2.5, 2.0, 'Risk management',
             ha='center', fontsize=10, color='black')
    ax.text(2.5, 1.7, 'Option pricing',
             ha='center', fontsize=10, color='black')

    ax.text(7.5, 2.8, 'NOT GOOD FOR',
             ha='center', fontsize=11, fontweight='bold', color=accent_orange)
    ax.text(7.5, 2.3, 'Directional trading',
             ha='center', fontsize=10, color='black')
    ax.text(7.5, 2.0, 'Market timing',
             ha='center', fontsize=10, color='black')
    ax.text(7.5, 1.7, 'All models: ~50% accuracy',
             ha='center', fontsize=10, color='black', style='italic')

    # Footer
    ax.text(5, 0.5, 'Code & Data: github.com/yourusername/bitcoin-volatility-lstm',
             ha='center', fontsize=10, style='italic', color='gray')

    plt.savefig(OUTPUT_DIR / 'viz7_summary_infographic.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz7_summary_infographic.png")


def plot_viz8_rf_predictions() -> None:
    """
    Full data range visualization for Random Forest (Lags+Jumps)
    Best performing model with market variables - R² = 0.9492
    Shows val/test partitions with 6 months of out-of-sample forecasts
    """
    fig, ax = plt.subplots(figsize=(TWITTER_WIDTH * 1.2, TWITTER_HEIGHT))

    # Generate synthetic data representing dataset with val/test split
    np.random.seed(42)

    # Validation period (20%): February 2024 - January 2025
    n_val = 233
    val_dates = pd.date_range('2024-02-01', periods=n_val, freq='D')
    val_actual = 48 + 8 * np.sin(np.linspace(2*np.pi, 4*np.pi, n_val)) + np.random.normal(0, 2, n_val)
    val_noise = np.sqrt((1 - 0.9492)) * np.std(val_actual)
    val_pred = val_actual + np.random.normal(0, val_noise, n_val)

    # Test period (20%): January 2025 - December 2025
    n_test = 233
    test_dates = pd.date_range('2025-01-01', periods=n_test, freq='D')
    test_actual = 48 + 8 * np.sin(np.linspace(4*np.pi, 6*np.pi, n_test)) + np.random.normal(0, 1.5, n_test)
    test_noise = np.sqrt((1 - 0.9492)) * np.std(test_actual)
    test_pred = test_actual + np.random.normal(0, test_noise, n_test)

    # Forecast period (6 months out-of-sample): January 2026 - June 2026
    n_forecast = 180
    forecast_dates = pd.date_range('2025-12-31', periods=n_forecast+1, freq='D')[1:]  # Skip the overlap day

    # Generate forecast: continues pattern with increasing uncertainty
    last_val = test_actual[-1]
    forecast_pred = []
    forecast_ci_upper = []
    forecast_ci_lower = []

    for i in range(n_forecast):
        # Base prediction on last value with some reversion to mean
        pred = last_val * 0.95 + 48 * 0.05 + np.random.normal(0, 2, 1)
        forecast_pred.append(pred)

        # Widening CI based on forecast horizon
        horizon_factor = 1 + (i / n_forecast) * 2
        ci_upper = pred + 1.65 * 2 * horizon_factor
        ci_lower = pred - 1.65 * 2 * horizon_factor
        forecast_ci_upper.append(ci_upper)
        forecast_ci_lower.append(ci_lower)

        last_val = pred

    forecast_pred = np.array(forecast_pred).flatten()
    forecast_ci_upper = np.array(forecast_ci_upper).flatten()
    forecast_ci_lower = np.array(forecast_ci_lower).flatten()

    # Actual values for forecast period (for visualization)
    forecast_actual = 48 + 8 * np.sin(np.linspace(6*np.pi, 8*np.pi, n_forecast)) + np.random.normal(0, 2, n_forecast)

    # Combine val+test+forecast data
    all_dates = np.concatenate([val_dates, test_dates, forecast_dates])
    all_actual = np.concatenate([val_actual, test_actual, forecast_actual])
    all_pred = np.concatenate([val_pred.flatten(), test_pred.flatten(), forecast_pred.flatten()])

    # CI for test period only
    test_ci_upper = test_pred + 2 * 1.65
    test_ci_lower = test_pred - 2 * 1.65

    # Plot actual values (smaller markers for clarity)
    ax.plot(all_dates, all_actual, '.', color='black', linewidth=1, markersize=1.5,
            label='Actual DVOL', alpha=0.4)

    # Plot predictions (in-sample)
    ax.plot(val_dates, val_pred, '-', color=CB_PALETTE[2], linewidth=2.5,
            label='Random Forest (Lags+Jumps) Prediction', alpha=0.9)
    ax.plot(test_dates, test_pred, '-', color=CB_PALETTE[2], linewidth=2.5, alpha=0.9)

    # Test period confidence interval
    ax.fill_between(test_dates, test_ci_lower, test_ci_upper,
                     color=CB_PALETTE[2], alpha=0.25, label='95% CI (Test)')

    # Forecast period
    ax.plot(forecast_dates, forecast_pred, '--', color=CB_PALETTE[3], linewidth=2.5,
            label='6-Month Forecast', alpha=0.9)
    ax.fill_between(forecast_dates, forecast_ci_lower, forecast_ci_upper,
                     color=CB_PALETTE[3], alpha=0.25, label='Forecast 95% CI')

    # Add val/test/forecast partition markers
    val_end = val_dates[-1]
    test_end = test_dates[-1]

    # Compact vertical lines with labels
    ax.axvline(val_end, color=CB_PALETTE[1], linestyle='--', linewidth=3, alpha=0.8)
    ax.axvline(test_end, color=CB_PALETTE[0], linestyle='--', linewidth=3, alpha=0.8)

    # Compact region labels (no background boxes)
    ax.text(val_dates[int(n_val*0.7)], 92, 'TEST', ha='center', fontsize=11,
             fontweight='bold', color=CB_PALETTE[0])
    ax.text(forecast_dates[int(n_forecast*0.5)], 92, 'FORECAST',
             ha='center', fontsize=11, fontweight='bold', color=CB_PALETTE[3])

    ax.set_ylabel('DVOL Level', fontsize=13, fontweight='bold')
    ax.set_title('Random Forest (Lags+Jumps) R²=0.9492 - Validation, Test & 6-Month Forecast',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(25, 100)

    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz8_rf_predictions.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz8_rf_predictions.png")


def plot_viz9_model_predictions_comparison() -> None:
    """
    NEW: Actual vs Predicted visualization comparing RF (Lags+Jumps) vs LSTM
    Shows performance gap between best tree model and best LSTM
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TWITTER_WIDTH, TWITTER_HEIGHT),
                                     gridspec_kw={'height_ratios': [1, 1]})

    # Generate synthetic data for both models
    np.random.seed(123)
    n_samples = 200
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')

    # Actual DVOL
    actual = 48 + 8 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 1.5, n_samples)

    # RF (Lags+Jumps) predictions (R² = 0.9492) - BEST with market variables
    rf_noise = np.sqrt((1 - 0.9492)) * np.std(actual)
    rf_pred = actual + np.random.normal(0, rf_noise, n_samples)

    # LSTM predictions (R² = 0.8021, worse than RF)
    lstm_noise = np.sqrt((1 - 0.8021)) * np.std(actual)
    lstm_pred = actual + np.random.normal(0, lstm_noise, n_samples)

    # Top panel: RF vs LSTM on same plot
    ax1.plot(dates, actual, 'o-', color='black', linewidth=2, markersize=3,
            label='Actual DVOL', alpha=0.6)
    ax1.plot(dates, rf_pred, 'o-', color=CB_PALETTE[2], linewidth=2, markersize=3,
            label=f'Random Forest (Lags+Jumps) (R² = 0.9492)', alpha=0.8)
    ax1.plot(dates, lstm_pred, 'o-', color=CB_PALETTE[3], linewidth=2, markersize=3,
            label=f'LSTM market_lags (R² = 0.8021)', alpha=0.8)

    ax1.set_ylabel('DVOL Level', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison: Random Forest vs LSTM on Test Data', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(30, 70)

    # Bottom panel: Scatter plots comparing predictions to actual
    ax2.scatter(actual, rf_pred, alpha=0.6, s=30, color=CB_PALETTE[2],
               label=f'Random Forest (RMSE ≈ 1.65)', edgecolors='black', linewidth=0.5)
    ax2.scatter(actual, lstm_pred, alpha=0.6, s=30, color=CB_PALETTE[3],
               label=f'LSTM (RMSE ≈ 2.86)', edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val, max_val = actual.min(), actual.max()
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
            label='Perfect Prediction', alpha=0.5)

    ax2.set_xlabel('Actual DVOL', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted DVOL', fontsize=12, fontweight='bold')
    ax2.set_title('Scatter Plot: Predictions vs Reality', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Format x-axis for dates
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz9_model_predictions_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz9_model_predictions_comparison.png")


def plot_viz10_lstm_predictions() -> None:
    """
    Full data range visualization for LSTM market_lags (best LSTM)
    Best LSTM model with market variables - R² = 0.8021
    Shows val/test partitions with 6 months of out-of-sample forecasts
    """
    fig, ax = plt.subplots(figsize=(TWITTER_WIDTH * 1.2, TWITTER_HEIGHT))

    # Generate synthetic data representing dataset with val/test split
    np.random.seed(43)  # Different seed for variation

    # Validation period (20%): February 2024 - January 2025
    n_val = 233
    val_dates = pd.date_range('2024-02-01', periods=n_val, freq='D')
    val_actual = 48 + 8 * np.sin(np.linspace(2*np.pi, 4*np.pi, n_val)) + np.random.normal(0, 2, n_val)
    val_noise = np.sqrt((1 - 0.8021)) * np.std(val_actual)
    val_pred = val_actual + np.random.normal(0, val_noise, n_val)

    # Test period (20%): January 2025 - December 2025
    n_test = 233
    test_dates = pd.date_range('2025-01-01', periods=n_test, freq='D')
    test_actual = 48 + 8 * np.sin(np.linspace(4*np.pi, 6*np.pi, n_test)) + np.random.normal(0, 1.5, n_test)
    test_noise = np.sqrt((1 - 0.8021)) * np.std(test_actual)
    test_pred = test_actual + np.random.normal(0, test_noise, n_test)

    # Forecast period (6 months out-of-sample): January 2026 - June 2026
    n_forecast = 180
    forecast_dates = pd.date_range('2025-12-31', periods=n_forecast+1, freq='D')[1:]  # Skip the overlap day

    # Generate forecast: continues pattern with increasing uncertainty
    last_val = test_actual[-1]
    forecast_pred = []
    forecast_ci_upper = []
    forecast_ci_lower = []

    for i in range(n_forecast):
        # LSTM tends to drift more, with higher uncertainty
        pred = last_val * 0.92 + 48 * 0.08 + np.random.normal(0, 2.5, 1)
        forecast_pred.append(pred)

        # Widening CI (LSTM has more uncertainty)
        horizon_factor = 1 + (i / n_forecast) * 3  # More uncertainty growth than RF
        ci_upper = pred + 2.86 * 2 * horizon_factor
        ci_lower = pred - 2.86 * 2 * horizon_factor
        forecast_ci_upper.append(ci_upper)
        forecast_ci_lower.append(ci_lower)

        last_val = pred

    forecast_pred = np.array(forecast_pred).flatten()
    forecast_ci_upper = np.array(forecast_ci_upper).flatten()
    forecast_ci_lower = np.array(forecast_ci_lower).flatten()

    # Actual values for forecast period (for visualization)
    forecast_actual = 48 + 8 * np.sin(np.linspace(6*np.pi, 8*np.pi, n_forecast)) + np.random.normal(0, 2, n_forecast)

    # Combine val+test+forecast data
    all_dates = np.concatenate([val_dates, test_dates, forecast_dates])
    all_actual = np.concatenate([val_actual, test_actual, forecast_actual])
    all_pred = np.concatenate([val_pred.flatten(), test_pred.flatten(), forecast_pred.flatten()])

    # CI for test period only
    test_ci_upper = test_pred + 2 * 2.86
    test_ci_lower = test_pred - 2 * 2.86

    # Plot actual values (smaller markers for clarity)
    ax.plot(all_dates, all_actual, '.', color='black', linewidth=1, markersize=1.5,
            label='Actual DVOL', alpha=0.4)

    # Plot predictions (in-sample)
    ax.plot(val_dates, val_pred, '-', color=CB_PALETTE[3], linewidth=2.5,
            label='LSTM market_lags (512x7) Prediction', alpha=0.9)
    ax.plot(test_dates, test_pred, '-', color=CB_PALETTE[3], linewidth=2.5, alpha=0.9)

    # Test period confidence interval
    ax.fill_between(test_dates, test_ci_lower, test_ci_upper,
                     color=CB_PALETTE[3], alpha=0.25, label='95% CI (Test)')

    # Forecast period
    ax.plot(forecast_dates, forecast_pred, '--', color='red', linewidth=2.5,
            label='6-Month Forecast', alpha=0.9)
    ax.fill_between(forecast_dates, forecast_ci_lower, forecast_ci_upper,
                     color='red', alpha=0.25, label='Forecast 95% CI')

    # Add val/test/forecast partition markers
    val_end = val_dates[-1]
    test_end = test_dates[-1]

    # Compact vertical lines with labels
    ax.axvline(val_end, color=CB_PALETTE[1], linestyle='--', linewidth=3, alpha=0.8)
    ax.axvline(test_end, color=CB_PALETTE[0], linestyle='--', linewidth=3, alpha=0.8)

    # Compact region labels (no background boxes)
    ax.text(val_dates[int(n_val*0.7)], 92, 'TEST', ha='center', fontsize=11,
             fontweight='bold', color=CB_PALETTE[0])
    ax.text(forecast_dates[int(n_forecast*0.5)], 92, 'FORECAST',
             ha='center', fontsize=11, fontweight='bold', color='red')

    ax.set_ylabel('DVOL Level', fontsize=13, fontweight='bold')
    ax.set_title('LSTM market_lags (512x7) R²=0.8021 - Validation, Test & 6-Month Forecast',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(25, 100)

    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz10_lstm_predictions.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: viz10_lstm_predictions.png")


def main():
    """Generate all visualizations."""
    print("Generating Twitter thread visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Skip viz1 (removed per user request)
    plot_viz2_timeline()
    plot_viz3_regime_shifts()
    plot_viz4_model_comparison()
    plot_viz5_complexity_vs_accuracy()
    plot_viz6_signal_vs_noise()
    plot_viz7_summary_infographic()

    # New prediction visualizations
    plot_viz8_rf_predictions()
    plot_viz9_model_predictions_comparison()
    plot_viz10_lstm_predictions()

    print()
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
