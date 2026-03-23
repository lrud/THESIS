#!/usr/bin/env python
"""
Comprehensive Statistical Investigation of Bitcoin DVOL Data

Analyzes the statistical properties of DVOL data to inform modeling decisions:
- Normality and heavy-tail tests
- Stationarity tests
- Heteroskedasticity and volatility clustering
- Asymmetry/leverage effects
- Regime change detection
- Distribution fitting
- Long memory analysis

Based on research findings for heavy-tailed volatility forecasting.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("BITCOIN DVOL STATISTICAL INVESTIGATION")
print("=" * 90)
print()

# Load data
print("Loading data...")
data_path = Path('data/processed/bitcoin_lstm_features_v1.1_complete_with_jumps.csv')
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Total observations: {len(df)}")
print()

# Extract DVOL series
dvol = df['dvol'].values
dvol_diff = np.diff(dvol)  # First differences
dvol_log_diff = np.diff(np.log(dvol))  # Log differences

# ============================================================================
# SECTION 1: NORMALITY & HEAVY TAIL ANALYSIS
# ============================================================================
print("=" * 90)
print("SECTION 1: NORMALITY & HEAVY TAIL ANALYSIS")
print("=" * 90)
print()

# Basic statistics
mean_dvol = np.mean(dvol)
std_dvol = np.std(dvol)
skew_dvol = stats.skew(dvol)
kurtosis_dvol = stats.kurtosis(dvol)  # Excess kurtosis (Gaussian = 0)
jarque_bera, jb_pvalue = stats.jarque_bera(dvol)

print("Basic Statistics:")
print(f"  Mean: {mean_dvol:.4f}")
print(f"  Std Dev: {std_dvol:.4f}")
print(f"  Min: {np.min(dvol):.4f}")
print(f"  Max: {np.max(dvol):.4f}")
print(f"  Range: {np.max(dvol) - np.min(dvol):.4f}")
print()

print("Shape Statistics:")
print(f"  Skewness: {skew_dvol:.4f}")
print(f"  Excess Kurtosis: {kurtosis_dvol:.4f} (Gaussian = 0)")
print(f"  Interpretation: {'Heavy-tailed' if kurtosis_dvol > 0 else 'Light-tailed'}")
print()

# Jarque-Bera test
print("Jarque-Bera Test (Normality):")
print(f"  Statistic: {jarque_bera:.4f}")
print(f"  P-value: {jb_pvalue:.2e}")
print(f"  Result: {'Reject Normality' if jb_pvalue < 0.05 else 'Cannot Reject Normality'}")
print()

# Shapiro-Wilk test (on subset due to sample size)
sample_size = min(5000, len(dvol))
shapiro_stat, shapiro_p = stats.shapiro(np.random.choice(dvol, sample_size, replace=False))
print(f"Shapiro-Wilk Test (n={sample_size}):")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.2e}")
print(f"  Result: {'Reject Normality' if shapiro_p < 0.05 else 'Cannot Reject Normality'}")
print()

# Anderson-Darling test
ad_result = stats.anderson(dvol, dist='norm')
print("Anderson-Darling Test (vs Normal):")
print(f"  Statistic: {ad_result.statistic:.4f}")
print(f"  Critical Values (5%): {ad_result.critical_values[2]:.4f}")
print(f"  Result: {'Reject Normality at 5%' if ad_result.statistic > ad_result.critical_values[2] else 'Cannot Reject Normality at 5%'}")
print()

# Hill Estimator for Tail Index
def hill_estimator(data, tail_fraction=0.1):
    """Estimate tail index using Hill estimator."""
    sorted_data = np.sort(data)[::-1]
    k = int(tail_fraction * len(data))
    tail_data = sorted_data[:k] - sorted_data[k]
    tail_data = tail_data[tail_data > 0]
    if len(tail_data) == 0:
        return np.nan
    hill_index = 1 / np.mean(np.log(tail_data / tail_data[-1]))
    return hill_index

# Use positive deviations from mean for tail estimation
positive_tail = dvol[dvol > mean_dvol] - mean_dvol
hill_idx = hill_estimator(positive_tail, tail_fraction=0.05)

print("Tail Analysis (Hill Estimator):")
print(f"  Tail Index: {hill_idx:.4f}")
print(f"  Interpretation:")
print(f"    - α < 2: Infinite variance (very heavy tails)")
print(f"    - 2 < α < 4: Finite variance, infinite kurtosis")
print(f"    - α > 4: Finite kurtosis (approaching Gaussian)")
print()

# ============================================================================
# SECTION 2: STATIONARITY TESTS
# ============================================================================
print("=" * 90)
print("SECTION 2: STATIONARITY TESTS")
print("=" * 90)
print()

# ADF Test
adf_stat, adf_pvalue, adf_usedlag, adf_nobs, adf_crit, adf_icbest = adfuller(dvol, maxlag=30)
print("Augmented Dickey-Fuller Test:")
print(f"  Statistic: {adf_stat:.4f}")
print(f"  P-value: {adf_pvalue:.2e}")
print(f"  Critical Values (5%): {adf_crit['5%']:.4f}")
print(f"  Result: {'Reject Unit Root (Stationary)' if adf_pvalue < 0.05 else 'Cannot Reject Unit Root (Non-Stationary)'}")
print()

# KPSS Test
kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss(dvol, regression='c', nlags='auto')
print("KPSS Test:")
print(f"  Statistic: {kpss_stat:.4f}")
print(f"  P-value: {kpss_pvalue:.2e}")
print(f"  Critical Values (5%): {kpss_crit['5%']:.4f}")
print(f"  Result: {'Reject Stationarity' if kpss_pvalue < 0.05 else 'Cannot Reject Stationarity'}")
print()

# ============================================================================
# SECTION 2.5: STATIONARITY TESTS ACROSS ROLLING NORMALIZATION WINDOWS
# ============================================================================

window_sizes = [72, 168, 336, 720]
features_to_test = [c for c in ['dvol', 'dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d',
                                'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']
                    if c in df.columns]

normalization_stationarity_results = {}
stationarity_counts = {w: 0 for w in window_sizes}

print(f"{'Feature':<25} {'Window':>10} {'ADF p':>10} {'KPSS p':>10} {'Stationary':>12}")
print("-" * 75)

for feature in features_to_test:
    feature_results = {}
    for window in window_sizes:
        rolling_mean = df[feature].rolling(window=window, min_periods=1).mean()
        rolling_std = df[feature].rolling(window=window, min_periods=1).std().replace(0, 1)
        normalized = ((df[feature] - rolling_mean) / rolling_std).dropna()

        adf_stat, adf_p, _, _, _, _ = adfuller(normalized, maxlag=30)
        kpss_stat, kpss_p, _, _ = kpss(normalized, regression='c', nlags='auto')

        stationary = adf_p < 0.05 and kpss_p >= 0.05
        if stationary:
            stationarity_counts[window] += 1

        feature_results[window] = {
            'adf_stat': float(adf_stat),
            'adf_pvalue': float(adf_p),
            'kpss_stat': float(kpss_stat),
            'kpss_pvalue': float(kpss_p),
            'stationary': bool(stationary)
        }
        print(f"{feature:<25} {window:>4}h        {adf_p:>10.2e} {kpss_p:>10.2e} {stationary!s:>12}")

    normalization_stationarity_results[feature] = feature_results

print()
print(f"{'Window':>10} {'Stationary Features':>20} {'Percentage':>12}")
print("-" * 45)
for w in window_sizes:
    pct = (stationarity_counts[w] / len(features_to_test)) * 100
    print(f"{w:>4}h        {stationarity_counts[w]:>10}/{len(features_to_test)}      {pct:>10.1f}%")

best_window = max(stationarity_counts, key=stationarity_counts.get)
print(f"\nBest window: {best_window}h ({best_window/24:.0f}d) - {stationarity_counts[best_window]}/{len(features_to_test)} features")
print()

# ============================================================================
# SECTION 3: HETEROSKEDASTICITY & VOLATILITY CLUSTERING
# ============================================================================
print("=" * 90)
print("SECTION 3: HETEROSKEDASTICITY & VOLATILITY CLUSTERING")
print("=" * 90)
print()

# ARCH-LM Test
arch_stat, arch_pvalue, lm_stat, lm_pvalue = het_arch(dvol_diff, maxlag=30)
print("ARCH-LM Test (for ARCH effects):")
print(f"  Statistic: {arch_stat:.4f}")
print(f"  P-value: {arch_pvalue:.2e}")
print(f"  Result: {'Reject No ARCH (Volatility Clustering Present)' if arch_pvalue < 0.05 else 'Cannot Reject No ARCH'}")
print()

# Ljung-Box test on squared residuals
squared_diff = dvol_diff ** 2
lb_stat_10 = acorr_ljungbox(squared_diff, lags=[10], return_df=True)
lb_stat_30 = acorr_ljungbox(squared_diff, lags=[30], return_df=True)

print("Ljung-Box Test on Squared Returns (Volatility Clustering):")
print(f"  Lag 10: Statistic={lb_stat_10['lb_stat'].values[0]:.2f}, p-value={lb_stat_10['lb_pvalue'].values[0]:.2e}")
print(f"  Lag 30: Statistic={lb_stat_30['lb_stat'].values[0]:.2f}, p-value={lb_stat_30['lb_pvalue'].values[0]:.2e}")
print()

# ============================================================================
# SECTION 4: ASYMMETRY & LEVERAGE EFFECT
# ============================================================================
print("=" * 90)
print("SECTION 4: ASYMMETRY & LEVERAGE EFFECT")
print("=" * 90)
print()

# Separate positive and negative shocks
positive_shocks = dvol_diff[dvol_diff > 0]
negative_shocks = dvol_diff[dvol_diff < 0]
abs_negative = np.abs(negative_shocks)

print("Shock Analysis:")
print(f"  Number of positive shocks: {len(positive_shocks)}")
print(f"  Number of negative shocks: {len(negative_shocks)}")
print(f"  Mean positive shock: {np.mean(positive_shocks):.4f}")
print(f"  Mean |negative| shock: {np.mean(abs_negative):.4f}")
print()

# Leverage effect test (correlation between current return and future volatility)
def leverage_effect_test(returns, lags=[1, 5, 10, 24]):
    """Test if negative returns predict higher future volatility."""
    results = {}
    squared_returns = returns ** 2
    for lag in lags:
        if lag >= len(returns):
            continue
        corr = np.corrcoef(returns[:-lag], squared_returns[lag:])[0, 1]
        results[f'lag_{lag}'] = corr
    return results

leverage_results = leverage_effect_test(dvol_diff, lags=[1, 5, 10, 24, 168])

print("Leverage Effect (Correlation: Return_t vs Volatility_{t+lag}):")
for lag, corr in leverage_results.items():
    significance = "Significant" if abs(corr) > 0.05 else "Not Significant"
    direction = "Negative (Leverage)" if corr < 0 else "Positive"
    print(f"  {lag}: r={corr:.4f} ({direction}) - {significance}")
print()

# Sign bias test
def sign_bias_test(returns):
    """Test if negative shocks have different impact than positive."""
    pos_idx = returns > 0
    neg_idx = returns < 0

    # Squared returns after positive vs negative shocks
    squared = returns ** 2
    volatility_after_pos = []
    volatility_after_neg = []

    for i in range(1, len(returns)):
        if pos_idx[i-1]:
            volatility_after_pos.append(squared[i])
        elif neg_idx[i-1]:
            volatility_after_neg.append(squared[i])

    t_stat, p_value = stats.ttest_ind(volatility_after_neg, volatility_after_pos, equal_var=False)
    return {
        'mean_after_pos': np.mean(volatility_after_pos),
        'mean_after_neg': np.mean(volatility_after_neg),
        'ratio': np.mean(volatility_after_neg) / np.mean(volatility_after_pos),
        't_stat': t_stat,
        'p_value': p_value
    }

sign_bias = sign_bias_test(dvol_diff)
print("Sign Bias Test (Volatility After Negative vs Positive Shocks):")
print(f"  Mean volatility after positive shock: {sign_bias['mean_after_pos']:.4f}")
print(f"  Mean volatility after negative shock: {sign_bias['mean_after_neg']:.4f}")
print(f"  Ratio (neg/pos): {sign_bias['ratio']:.4f}")
print(f"  T-statistic: {sign_bias['t_stat']:.4f}")
print(f"  P-value: {sign_bias['p_value']:.2e}")
print(f"  Result: {'Significant Asymmetry' if sign_bias['p_value'] < 0.05 else 'No Significant Asymmetry'}")
print()

# ============================================================================
# SECTION 5: REGIME CHANGE DETECTION
# ============================================================================
print("=" * 90)
print("SECTION 5: REGIME CHANGE DETECTION")
print("=" * 90)
print()

# ICSS algorithm for variance change points
def icss_algorithm(returns, significance=0.05):
    """Iterative Cumulative Sum of Squares algorithm for change points."""
    n = len(returns)
    critical_value = 1.358  # For 5% significance

    # Center the series
    centered = returns - np.mean(returns)

    # Cumulative sum of squares
    css = np.cumsum(centered ** 2)
    css_normalized = css / css[-1]

    # Test statistic
    D = np.max(np.abs(css_normalized - np.arange(1, n+1) / n))
    critical = critical_value / np.sqrt(n)

    if D > critical:
        # Find change point
        change_point = np.argmax(np.abs(css_normalized - np.arange(1, n+1) / n))
        return change_point, D, critical
    return None, D, critical

# Apply ICSS recursively
def find_change_points(returns, min_segment=100):
    """Recursively find change points."""
    change_points = []

    def search(segment, offset=0):
        if len(segment) < min_segment:
            return
        cp, D, critical = icss_algorithm(segment)
        if cp is not None and cp > min_segment and cp < len(segment) - min_segment:
            change_points.append(offset + cp)
            search(segment[:cp], offset)
            search(segment[cp:], offset + cp)

    search(returns)
    return sorted(change_points)

change_points = find_change_points(dvol_diff[:10000], min_segment=500)  # Use subset for speed

print("ICSS Algorithm (Variance Change Points):")
if change_points:
    print(f"  Found {len(change_points)} change points")
    print(f"  Change point indices: {change_points[:10]}..." if len(change_points) > 10 else f"  Change point indices: {change_points}")
else:
    print("  No significant change points detected")
print()

# Rolling variance analysis to visualize regimes
window = 720  # 30 days
rolling_var = pd.Series(dvol_diff).rolling(window=window).var()
var_stats = {
    'min': rolling_var.min(),
    'max': rolling_var.max(),
    'mean': rolling_var.mean(),
    'std': rolling_var.std(),
    'cv': rolling_var.std() / rolling_var.mean()  # Coefficient of variation
}

print("Rolling Variance Statistics (720-hour window):")
print(f"  Min: {var_stats['min']:.4f}")
print(f"  Max: {var_stats['max']:.4f}")
print(f"  Mean: {var_stats['mean']:.4f}")
print(f"  Std: {var_stats['std']:.4f}")
print(f"  CV (std/mean): {var_stats['cv']:.4f}")
print(f"  Range ratio: {var_stats['max'] / var_stats['min']:.2f}x")
print()

# ============================================================================
# SECTION 6: DISTRIBUTION FITTING
# ============================================================================
print("=" * 90)
print("SECTION 6: DISTRIBUTION FITTING")
print("=" * 90)
print()

# Fit Student's t-distribution
def fit_students_t(data):
    """Fit Student's t distribution using MLE."""
    def neg_log_likelihood(params):
        df, loc, scale = params
        return -np.sum(stats.t.logpdf(data, df, loc=loc, scale=scale))

    # Initial guess
    df_guess = max(3, stats.skew(data)**2 + 3)
    initial = [df_guess, np.mean(data), np.std(data)]

    # Bounds
    bounds = [(2.1, 100), (None, None), (0.001, None)]

    result = minimize(neg_log_likelihood, initial, bounds=bounds, method='L-BFGS-B')
    df, loc, scale = result.x

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.t.cdf(x, df, loc=loc, scale=scale))

    return {'df': df, 'loc': loc, 'scale': scale, 'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue}

t_fit = fit_students_t(dvol_diff)

print("Student's t-Distribution Fit:")
print(f"  Degrees of Freedom: {t_fit['df']:.2f}")
print(f"  Location: {t_fit['loc']:.4f}")
print(f"  Scale: {t_fit['scale']:.4f}")
print(f"  KS Statistic: {t_fit['ks_stat']:.4f}")
print(f"  KS P-value: {t_fit['ks_pvalue']:.2e}")
print(f"  Interpretation:")
print(f"    - df < 5: Very heavy tails")
print(f"    - 5 < df < 10: Moderately heavy tails")
print(f"    - df > 10: Approaching Gaussian")
print()

# Compare to Gaussian fit
gaussian_params = stats.norm.fit(dvol_diff)
ks_gaussian, ks_gaussian_p = stats.kstest(dvol_diff, lambda x: stats.norm.cdf(x, *gaussian_params))

print("Gaussian Distribution Fit (for comparison):")
print(f"  KS Statistic: {ks_gaussian:.4f}")
print(f"  KS P-value: {ks_gaussian_p:.2e}")
print(f"  Better fit: {'Student t' if t_fit['ks_stat'] < ks_gaussian else 'Gaussian'}")
print()

# ============================================================================
# SECTION 7: LONG MEMORY ANALYSIS
# ============================================================================
print("=" * 90)
print("SECTION 7: LONG MEMORY ANALYSIS")
print("=" * 90)
print()

# Hurst Exponent (R/S analysis)
def hurst_rs(series, max_lag=500):
    """Calculate Hurst exponent using R/S analysis."""
    lags = range(10, min(max_lag, len(series)//2))
    tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]

    # Log-log regression
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] / 2
    return hurst

hurst_exp = hurst_rs(dvol, max_lag=500)

print("Hurst Exponent (R/S Analysis):")
print(f"  H = {hurst_exp:.4f}")
print(f"  Interpretation:")
print(f"    - H < 0.5: Mean-reverting (anti-persistent)")
print(f"    - H ≈ 0.5: Random walk / Brownian motion")
print(f"    - H > 0.5: Trending / long memory")
print()

# Autocorrelation analysis
def acf_analysis(series, max_lag=168):
    """Analyze autocorrelation decay."""
    acf_values = [1.0]  # Lag 0
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        acf_values.append(corr)
    return acf_values

acf_vals = acf_analysis(dvol, max_lag=168)

print("Autocorrelation at Key Lags:")
key_lags = [1, 6, 24, 48, 168]
for lag in key_lags:
    if lag < len(acf_vals):
        print(f"  Lag {lag}h: {acf_vals[lag]:.4f}")
print()

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("=" * 90)
print("SUMMARY & MODELING RECOMMENDATIONS")
print("=" * 90)
print()

summary = {
    'date': '2026-02-17',
    'n_observations': int(len(df)),
    'date_range': {
        'start': str(df['timestamp'].min().date()),
        'end': str(df['timestamp'].max().date())
    },
    'dvol_statistics': {
        'mean': float(mean_dvol),
        'std': float(std_dvol),
        'skewness': float(skew_dvol),
        'kurtosis': float(kurtosis_dvol)
    },
    'normality_tests': {
        'jarque_bera_pvalue': float(jb_pvalue),
        'shapiro_pvalue': float(shapiro_p),
        'reject_normality': bool(jb_pvalue < 0.05)
    },
    'tail_analysis': {
        'hill_estimator': float(hill_idx),
        'students_t_df': float(t_fit['df'])
    },
    'stationarity': {
        'raw_dvol': {
            'adf_pvalue': float(adf_pvalue),
            'kpss_pvalue': float(kpss_pvalue),
            'is_stationary': bool(adf_pvalue < 0.05),
            'conflict': bool(adf_pvalue < 0.05 and kpss_pvalue < 0.05)
        },
        'rolling_normalized': {
            f'{w}h': {
                'features_tested': len(features_to_test),
                'features_stationary': stationarity_counts[w],
                'percentage_stationary': float((stationarity_counts[w] / len(features_to_test)) * 100),
                'results_by_feature': {
                    feat: normalization_stationarity_results[feat][w]
                    for feat in normalization_stationarity_results
                }
            }
            for w in window_sizes
        },
        'best_window_hours': int(best_window),
        'best_window_days': int(best_window / 24),
        'best_window_stationary_pct': float((stationarity_counts[best_window] / len(features_to_test)) * 100)
    },
    'volatility_clustering': {
        'arch_lm_pvalue': float(arch_pvalue),
        'has_arch_effects': bool(arch_pvalue < 0.05)
    },
    'leverage_effect': {
        'lag_1_corr': float(leverage_results.get('lag_1', 0)),
        'sign_bias_pvalue': float(sign_bias['p_value']),
        'has_asymmetry': bool(sign_bias['p_value'] < 0.05)
    },
    'regime_changes': {
        'n_change_points': len(change_points),
        'rolling_variance_cv': float(var_stats['cv'])
    },
    'long_memory': {
        'hurst_exponent': float(hurst_exp)
    }
}

# Generate recommendations
recommendations = []

# Normality / Heavy tails
if kurtosis_dvol > 3:
    recommendations.append({
        'category': 'Distribution',
        'finding': f"Heavy-tailed distribution (kurtosis = {kurtosis_dvol:.2f})",
        'recommendation': 'Use Student t-distribution or quantile loss functions',
        'priority': 'HIGH'
    })

if t_fit['df'] < 10:
    recommendations.append({
        'category': 'Distribution',
        'finding': f"Student t fit: df = {t_fit['df']:.2f} (very heavy tails)",
        'recommendation': 'Quantile regression or robust loss functions recommended',
        'priority': 'HIGH'
    })

# Stationarity
raw_conflict = adf_pvalue < 0.05 and kpss_pvalue < 0.05
if raw_conflict:
    recommendations.append({
        'category': 'Stationarity',
        'finding': f'ADF/KPSS conflict (ADF p={adf_pvalue:.2e}, KPSS p={kpss_pvalue:.2e})',
        'recommendation': 'Rolling normalization required',
        'priority': 'HIGH'
    })

best_window_pct = (stationarity_counts[best_window] / len(features_to_test)) * 100
if best_window_pct < 80:
    recommendations.append({
        'category': 'Normalization',
        'finding': f'{best_window_pct:.0f}% features stationary at best window ({best_window}h)',
        'recommendation': 'Consider first-differencing for non-stationary features',
        'priority': 'MEDIUM'
    })

# Volatility clustering
if arch_pvalue < 0.05:
    recommendations.append({
        'category': 'Volatility',
        'finding': 'Significant volatility clustering (ARCH effects)',
        'recommendation': 'GARCH-style modeling or regime-switching approaches',
        'priority': 'HIGH'
    })

# Leverage effect
if sign_bias['p_value'] < 0.05:
    recommendations.append({
        'category': 'Asymmetry',
        'finding': f"Significant asymmetry: neg/pos ratio = {sign_bias['ratio']:.2f}",
        'recommendation': 'Implement asymmetric response (GJR-GARCH style)',
        'priority': 'MEDIUM'
    })

# Regime changes
if var_stats['cv'] > 0.5:
    recommendations.append({
        'category': 'Regimes',
        'finding': f"High variance variability (CV = {var_stats['cv']:.2f})",
        'recommendation': 'Regime-switching models or adaptive window sizing',
        'priority': 'MEDIUM'
    })

# Long memory
if hurst_exp > 0.6:
    recommendations.append({
        'category': 'Memory',
        'finding': f"Long memory detected (H = {hurst_exp:.4f})",
        'recommendation': 'Include long-lag features (HAR-RV style)',
        'priority': 'MEDIUM'
    })
elif hurst_exp < 0.4:
    recommendations.append({
        'category': 'Memory',
        'finding': f"Mean-reversion detected (H = {hurst_exp:.4f})",
        'recommendation': 'Ornstein-Uhlenbeck or mean-reverting models',
        'priority': 'LOW'
    })

# Print recommendations
print("Priority Recommendations:")
print("-" * 90)

priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
recommendations.sort(key=lambda x: priority_order[x['priority']])

for i, rec in enumerate(recommendations, 1):
    print(f"{i}. [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
    print(f"   Finding: {rec['finding']}")
    print()

summary['recommendations'] = recommendations

# Save results
output_dir = Path('results/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'statistical_investigation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("=" * 90)
print(f"Results saved to: {output_dir / 'statistical_investigation_summary.json'}")
print("=" * 90)
