#!/usr/bin/env python3
"""
Standard Lee-Mykland (2008) Jump Detection Implementation

This script implements the standard Lee-Mykland jump detection test
as specified in the academic literature:
Lee, S. and Mykland, P.A. (2008): "Jumps in financial markets."

Reference implementation based on:
- Original Lee & Mykland (2008) paper
- linuskohl GitHub implementation
- highfrequency R package documentation

Author: Claude AI
Date: 2026-02-25
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys

def lee_mykland_test(price_series, significance_level=0.0001, k=None):
    """
    Standard Lee-Mykland jump detection test.

    Parameters
    ----------
    price_series : pd.Series
        Price or value series (e.g., DVOL values)
    significance_level : float
        Significance level for jump detection (default: 0.0001 = 0.01%)
        This corresponds to detecting jumps in the top 0.01% of movements
    k : int, optional
        Window size for bipower variation estimation.
        If None, uses k = ceil(sqrt(n)) where n is sample size.

    Returns
    -------
    jumps : pd.Series
        Binary series indicating jumps (1) and non-jumps (0)
    test_stats : pd.Series
        T statistics (normalized test statistics)
    L_stats : pd.Series
        Raw L statistics (returns / estimated volatility)
    threshold : float
        The critical value (beta_star) for jump detection

    References
    ----------
    Lee, S. and Mykland, P.A. (2008): "Jumps in financial markets."
    Journal of Econometrics.

    Notes
    -----
    The test statistic is:
        L_t = r_t / σ_t
    where r_t are log returns and σ_t is estimated using bipower variation.

    The normalized test statistic is:
        T_t = (|L_t| - C_n) × S_n
    where S_n and C_n are normalizing constants based on sample size.

    Jumps are detected when T_t > β*, where β* is from Gumbel distribution.
    """
    # Convert to numpy array and handle missing values
    prices = price_series.dropna().values
    n = len(prices)

    if n < 10:
        raise ValueError("Insufficient data for Lee-Mykland test (need at least 10 observations)")

    # 1. Calculate log returns: r_t = log(S_t) - log(S_{t-1})
    log_returns = np.diff(np.log(prices))

    # Pad with NaN at start to align with original index
    log_returns = np.concatenate([[np.nan], log_returns])

    # 2. Calculate Realized Bipower Variation (BPV)
    # BPV_t = |r_{t-1}| × |r_t|
    # This is a robust estimator of integrated variance
    abs_returns = np.abs(log_returns)

    # BPV uses adjacent absolute returns (excluding the initial NaN)
    # We need n-1 BPV values from n returns
    valid_returns = abs_returns[1:]  # Skip initial NaN
    bpv = valid_returns[:-1] * valid_returns[1:]

    # Pad to match original length with NaN at start
    bpv = np.concatenate([[np.nan, np.nan], bpv])

    # 3. Estimate instantaneous volatility using rolling window
    # Window size k: typically k = ceil(sqrt(n)) or a fixed parameter
    if k is None:
        k = int(np.ceil(np.sqrt(n)))
    else:
        k = max(k, 3)  # Need at least 3 observations

    # Rolling mean of BPV to estimate variance
    bpv_rolling = pd.Series(bpv).rolling(
        window=k,
        min_periods=1
    ).mean().values

    # Instantaneous volatility: σ_t = sqrt(BPV_t)
    sigma_t = np.sqrt(bpv_rolling)

    # 4. Calculate L-statistic: L_t = r_t / σ_t
    L_t = log_returns / sigma_t
    L_t[0] = np.nan  # First observation has no return

    # 5. Calculate normalizing constants (Extreme Value Theory)
    c = np.sqrt(2 / np.pi)
    S_n = c * np.sqrt(2 * np.log(n))
    C_n = (S_n / c) - (np.log(np.pi * np.log(n))) / (2 * c * S_n)

    # 6. Calculate normalized test statistic
    # T_t = (|L_t| - C_n) × S_n
    T_t = (np.abs(L_t) - C_n) * S_n

    # 7. Calculate critical value (threshold) from Gumbel distribution
    # β* = -log(-log(1 - α))
    # For α = 0.0001, β* ≈ 9.21
    beta_star = -np.log(-np.log(1 - significance_level))

    # 8. Detect jumps
    jumps = (T_t > beta_star).astype(int)
    jumps[np.isnan(T_t)] = 0

    # Create pandas Series with original index
    result_jumps = pd.Series(jumps, index=price_series.index)
    result_T = pd.Series(T_t, index=price_series.index)
    result_L = pd.Series(L_t, index=price_series.index)

    return result_jumps, result_T, result_L, beta_star


def lee_mykland_test_hv(returns, significance_level=0.0001, k=None):
    """
    Lee-Mykland test for pre-calculated returns (e.g., DVOL percentage changes).

    This variant is useful when working with returns rather than prices.
    For DVOL analysis, percentage changes are equivalent to log returns for small changes.

    Parameters
    ----------
    returns : pd.Series
        Return series (already calculated)
    significance_level : float
        Significance level for jump detection
    k : int, optional
        Window size for bipower variation estimation

    Returns
    -------
    jumps, test_stats, L_stats, threshold : tuple
    """
    # Remove NaN values for calculation
    returns_clean = returns.dropna()
    n = len(returns_clean)

    if n < 10:
        raise ValueError("Insufficient data for Lee-Mykland test")

    # For returns, we use them directly
    r = returns_clean.values

    # Calculate Realized Bipower Variation
    # BPV_t = |r_{t-1}| × |r_t|
    abs_r = np.abs(r)
    bpv = abs_r[:-1] * abs_r[1:]

    # Rolling window for volatility estimation
    if k is None:
        k = int(np.ceil(np.sqrt(n)))
    k = max(k, 3)

    bpv_rolling = pd.Series(bpv).rolling(
        window=k,
        min_periods=1
    ).mean().values

    sigma_t = np.sqrt(bpv_rolling)

    # L-statistic: L_t = r_t / σ_t
    # Alignment: BPV_t uses r_{t-1} and r_t, so sigma_t corresponds to r_t position
    # r[0] has no BPV before it → NaN
    # r[1] corresponds to BPV_0
    L_t = np.full(n, np.nan)
    L_t[1:] = r[1:] / sigma_t  # sigma_t has n-1 values, same as bpv

    # Normalizing constants
    c = np.sqrt(2 / np.pi)
    S_n = c * np.sqrt(2 * np.log(n))
    C_n = (S_n / c) - (np.log(np.pi * np.log(n))) / (2 * c * S_n)

    # Normalized test statistic
    T_t = (np.abs(L_t) - C_n) * S_n

    # Critical value
    beta_star = -np.log(-np.log(1 - significance_level))

    # Detect jumps
    jumps = (T_t > beta_star).astype(int)
    jumps[np.isnan(T_t)] = 0

    # Reconstruct Series with original index
    result_jumps = pd.Series(0, index=returns.index)
    result_jumps.loc[returns_clean.index] = jumps

    result_T = pd.Series(np.nan, index=returns.index)
    result_T.loc[returns_clean.index] = T_t

    result_L = pd.Series(np.nan, index=returns.index)
    result_L.loc[returns_clean.index] = L_t

    return result_jumps, result_T, result_L, beta_star


def compare_jump_methods(df, price_col='dvol'):
    """
    Compare the standard Lee-Mykland implementation with the current v1.3 method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    price_col : str
        Column name for price/volatility data

    Returns
    -------
    comparison : dict
        Dictionary with comparison statistics
    """
    print("=" * 80)
    print("LEE-MYKLAND JUMP DETECTION COMPARISON")
    print("=" * 80)
    print()

    # Standard Lee-Mykland (from prices)
    print("Running standard Lee-Mykland test (from prices)...")
    jumps_lm_std, T_std, L_std, beta_std = lee_mykland_test(
        df[price_col],
        significance_level=0.0001
    )

    print(f"  Critical value (β*): {beta_std:.4f}")
    print(f"  Jumps detected: {jumps_lm_std.sum()} ({jumps_lm_std.sum()/len(jumps_lm_std)*100:.2f}%)")
    print(f"  Max T-statistic: {T_std.max():.4f}")
    print(f"  Mean |L|: {np.abs(L_std).mean():.4f}")
    print()

    # Alternative: from percentage changes
    print("Running standard Lee-Mykland test (from returns)...")
    returns = df[price_col].pct_change()
    jumps_lm_ret, T_ret, L_ret, beta_ret = lee_mykland_test_hv(
        returns,
        significance_level=0.0001
    )

    print(f"  Jumps detected: {jumps_lm_ret.sum()} ({jumps_lm_ret.sum()/len(jumps_lm_ret)*100:.2f}%)")
    print()

    # Current v1.3 method
    print("Current v1.3 method (data-driven)...")
    v13_jumps = df['lee_mykland_jump'].sum()
    print(f"  Jumps detected: {v13_jumps} ({v13_jumps/len(df)*100:.2f}%)")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    comparison = {
        'standard_lm_price': {
            'jumps': int(jumps_lm_std.sum()),
            'pct': jumps_lm_std.sum() / len(jumps_lm_std) * 100,
            'threshold': float(beta_std),
            'max_T': float(T_std.max())
        },
        'standard_lm_returns': {
            'jumps': int(jumps_lm_ret.sum()),
            'pct': jumps_lm_ret.sum() / len(jumps_lm_ret) * 100,
            'threshold': float(beta_ret),
            'max_T': float(T_ret.max())
        },
        'v13_current': {
            'jumps': int(v13_jumps),
            'pct': v13_jumps / len(df) * 100,
            'threshold': 'data-driven (~93.52)'
        }
    }

    # Print comparison table
    print(f"{'Method':<30} {'Jumps':<10} {'%':<10} {'Threshold':<15}")
    print("-" * 70)
    print(f"{'Standard LM (from prices)':<30} {comparison['standard_lm_price']['jumps']:<10} "
          f"{comparison['standard_lm_price']['pct']:<10.2f} {comparison['standard_lm_price']['threshold']:<15.4f}")
    print(f"{'Standard LM (from returns)':<30} {comparison['standard_lm_returns']['jumps']:<10} "
          f"{comparison['standard_lm_returns']['pct']:<10.2f} {comparison['standard_lm_returns']['threshold']:<15.4f}")
    print(f"{'v1.3 Current (data-driven)':<30} {comparison['v13_current']['jumps']:<10} "
          f"{comparison['v13_current']['pct']:<10.2f} {comparison['v13_current']['threshold']:<15}")
    print()

    # Overlap analysis
    overlap_std_v13 = (jumps_lm_std == 1) & (df['lee_mykland_jump'] == 1)
    print(f"Overlap (Standard LM ∩ v1.3): {overlap_std_v13.sum()} jumps")

    only_std = (jumps_lm_std == 1) & (df['lee_mykland_jump'] == 0)
    only_v13 = (jumps_lm_std == 0) & (df['lee_mykland_jump'] == 1)
    print(f"Only Standard LM: {only_std.sum()} jumps")
    print(f"Only v1.3: {only_v13.sum()} jumps")
    print()

    return comparison, jumps_lm_std, T_std, L_std


def main():
    """Main function to run the comparison on v1.3 dataset."""
    import sys
    import os
    sys.path.insert(0, '/home/lrud1314/PROJECTS_WORKING/THESIS 2025')

    INPUT_FILE = "data/processed/bitcoin_lstm_features_v1.3_complete_fixed.csv"

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Loaded: {len(df):,} rows")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    # Run comparison
    comparison, jumps_std, T_stats, L_stats = compare_jump_methods(df, 'dvol')

    # Save results for possible dataset update
    print("=" * 80)
    print("Saving results...")
    print()

    output_dir = "results/analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Save the standard Lee-Mykland jump series
    results_df = df[['timestamp', 'dvol', 'lee_mykland_jump']].copy()
    results_df['lee_mykland_jump_standard'] = jumps_std.values
    results_df['T_statistic'] = T_stats.values
    results_df['L_statistic'] = L_stats.values

    output_file = f"{output_dir}/lee_mykland_standard_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"  Saved comparison to: {output_file}")
    print()

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the comparison results")
    print("2. If using standard Lee-Mykland, update v1.3 dataset with new jumps")
    print("3. Update progress.md with findings")
    print()


if __name__ == "__main__":
    main()
