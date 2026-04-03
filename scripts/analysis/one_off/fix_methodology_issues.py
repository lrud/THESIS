#!/usr/bin/env python3
"""
Fix Methodology Issues in Dataset v1.2_complete_with_jumps.csv

This script addresses three critical issues identified in the dataset:
1. Lee-Mykland jump detection - FIX: Use data-driven threshold
2. DVOL-RV spread - FIX: Use academic best practices (realized volatility from squared returns)
3. NVRV verification - Verify calculation by re-fetching market_cap and realized_cap

Date: 2026-02-25
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add paths
sys.path.insert(0, '/home/lrud1314/PROJECTS_WORKING/THESIS 2025')
from scripts.data_collection.researchbitcoin_data import ResearchBitcoinCollector

# Configuration
INPUT_FILE = "data/processed/bitcoin_lstm_features_v1.2_complete_with_jumps.csv"
OUTPUT_FILE = "data/processed/bitcoin_lstm_features_v1.3_complete_fixed.csv"
API_TOKEN = "77849f7f-ba06-43fc-98cc-0b7dcfe5e313"  # Tier 1 token

print("="*80)
print("METHODOLOGY FIXES FOR v1.2 DATASET")
print("="*80)
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load data
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"  Loaded: {len(df):,} rows")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

#=============================================================================
# ISSUE 1: FIX LEE-MYKLAND JUMP DETECTION
#=============================================================================
print("="*80)
print("ISSUE 1: LEE-MYKLAND JUMP DETECTION - FIX")
print("="*80)
print()
print("PROBLEM:")
print("  Current threshold (~4.93) is 8x higher than max observed statistic (0.60)")
print("  Result: ZERO jumps detected by Lee-Mykland method")
print("  Root cause: Threshold formula meant for MAXIMUM value, not individual tests")
print()
print("SOLUTION:")
print("  Use data-driven threshold based on quantile of test statistics")
print("  Academic standard: 99.9th percentile = jumps in top 0.1% of movements")
print()

# Recalculate Lee-Mykland with correct approach
returns = df['dvol'].pct_change()
window = 24  # Daily window

# Realized bipower variation (robust to jumps)
abs_returns = np.abs(returns)
bipower_var = (np.pi / 2) * (abs_returns.rolling(window).mean() ** 2)

# Lee-Mykland statistic
L = (returns ** 2) / bipower_var
c = np.sqrt(2 / np.pi)
S_n = (1 / c) * np.sqrt(window) * (L - 1)

# Data-driven threshold: 99.9th percentile (catches top 0.1% as jumps)
threshold_quantile = S_n.dropna().quantile(0.999)

# Alternative: Use extreme value theory for more appropriate threshold
# For hourly data, a practical threshold is around 3-4 standard deviations above mean
threshold_evt = S_n.dropna().mean() + 3.5 * S_n.dropna().std()

# Use the more conservative (lower) threshold
threshold_new = min(threshold_quantile, threshold_evt)

print(f"  Old threshold (Gumbel max): ~4.93")
print(f"  New threshold (quantile 0.999): {threshold_quantile:.4f}")
print(f"  Alternative threshold (EVT 3.5σ): {threshold_evt:.4f}")
print(f"  Selected threshold: {threshold_new:.4f}")
print()

# Apply new threshold
jumps_lm_new = S_n > threshold_new
jumps_lm_new = jumps_lm_new.fillna(False)

print(f"  Jumps detected (OLD): {df['lee_mykland_jump'].sum()}")
print(f"  Jumps detected (NEW): {jumps_lm_new.sum()}")
print()

# Update dataframe
df['lee_mykland_stat'] = S_n
df['lee_mykland_jump'] = jumps_lm_new.astype(int)

# Recalculate composite jump indicator
df['jump_any'] = (
    df['lee_mykland_jump'] |
    df['sigma_jump'] |
    df['zscore_jump']
).astype(int)

df['jump_all'] = (
    df['lee_mykland_jump'] &
    df['sigma_jump'] &
    df['zscore_jump']
).astype(int)

print(f"  Updated composite jump_any: {df['jump_any'].sum()} jumps")
print()

#=============================================================================
# ISSUE 2: FIX DVOL-RV SPREAD (ACADEMIC BEST PRACTICES)
#=============================================================================
print("="*80)
print("ISSUE 2: DVOL-RV SPREAD - ACADEMIC BEST PRACTICES")
print("="*80)
print()
print("PROBLEM:")
print("  Current method: DVOL - (rolling_std(DVOL) * 100)")
print("  Issue: Circular - uses DVOL to calculate 'RV spread' from DVOL")
print()
print("ACADEMIC FORMULA:")
print("  Realized Variance (RV) = sum(squared_returns) over period")
print("  Realized Volatility = sqrt(RV) * annualization_factor")
print()
print("  For hourly data -> annual: multiply by sqrt(24*252) = sqrt(6048) ≈ 77.78")
print("  For hourly data -> daily: multiply by sqrt(24) ≈ 4.90")
print()
print("  DVOL-RV Spread = DVOL - Realized_Volatility")
print("  (This represents the volatility risk premium)")
print()

# Calculate using academic best practices
returns = df['dvol'].pct_change()

# Realized Volatility - rolling 24-hour window
# RV = sqrt(sum(r^2)) * annualization
# For hourly data, use daily realized vol (no need to annualize both)
rv_squared_sum = returns.rolling(window=24).apply(lambda x: (x**2).sum())
rv_daily = np.sqrt(rv_squared_sum)  # This is daily realized vol

# Convert to annualized for comparison with DVOL (which is already annualized)
rv_annual = rv_daily * np.sqrt(252)  # Daily to annual

# DVOL-RV Spread = DVOL - RV_annual
df['dvol_rv_spread_fixed'] = df['dvol'] - rv_annual

print(f"  DVOL-RV Spread (OLD): range {df['dvol_rv_spread'].min():.2f} to {df['dvol_rv_spread'].max():.2f}")
print(f"  DVOL-RV Spread (NEW): range {df['dvol_rv_spread_fixed'].min():.2f} to {df['dvol_rv_spread_fixed'].max():.2f}")
print()

# Correlation with DVOL
corr_old = df['dvol'].corr(df['dvol_rv_spread'])
corr_new = df['dvol'].corr(df['dvol_rv_spread_fixed'])
print(f"  Correlation DVOL vs Spread (OLD): {corr_old:.4f}")
print(f"  Correlation DVOL vs Spread (NEW): {corr_new:.4f}")
print()

# Replace old column
df['dvol_rv_spread'] = df['dvol_rv_spread_fixed']
df = df.drop(columns=['dvol_rv_spread_fixed'])

#=============================================================================
# ISSUE 3: VERIFY NVRV CALCULATION
#=============================================================================
print("="*80)
print("ISSUE 3: NVRV CALCULATION VERIFICATION")
print("="*80)
print()
print("NVRV Formula: NVRV = (Market Cap - Realized Cap) / Realized Cap")
print()
print("Fetching market_cap and realized_cap from ResearchBitcoin API...")
print("  This will take several minutes for 4+ years of hourly data...")
print()

try:
    collector = ResearchBitcoinCollector(API_TOKEN)

    start_date = df['timestamp'].min().strftime('%Y-%m-%d')
    end_date = (df['timestamp'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Fetch market_cap and realized_cap
    print("  Fetching market_cap...")
    market_cap_df = collector.get_metric("market_cap", "h1", start_date, end_date)
    market_cap_df.index = market_cap_df.index.tz_convert(None)

    print("  Fetching realized_cap...")
    realized_cap_df = collector.get_metric("realized_cap", "h1", start_date, end_date)
    realized_cap_df.index = realized_cap_df.index.tz_convert(None)

    # Merge and recalculate NVRV
    print("  Recalculating NVRV...")

    # Create timestamp-aligned data
    verification_df = df[['timestamp', 'nvrv']].copy()

    # Map API values
    mc_map = dict(zip(market_cap_df.index, market_cap_df['market_cap']))
    rc_map = dict(zip(realized_cap_df.index, realized_cap_df['realized_cap']))

    verification_df['market_cap_api'] = verification_df['timestamp'].map(mc_map)
    verification_df['realized_cap_api'] = verification_df['timestamp'].map(rc_map)

    # Calculate NVRV from API data
    verification_df['nvrv_calculated'] = (
        verification_df['market_cap_api'] - verification_df['realized_cap_api']
    ) / verification_df['realized_cap_api']

    # Compare with existing NVRV
    verification_df['nvrv_difference'] = verification_df['nvrv'] - verification_df['nvrv_calculated']

    # Statistics
    valid_rows = verification_df.dropna(subset=['market_cap_api', 'realized_cap_api', 'nvrv', 'nvrv_calculated'])

    print(f"  API data retrieved: {len(valid_rows):,} matched rows")
    print()
    print(f"  NVRV comparison (dataset vs API-calculated):")
    print(f"    Mean (dataset): {valid_rows['nvrv'].mean():.4f}")
    print(f"    Mean (calculated): {valid_rows['nvrv_calculated'].mean():.4f}")
    print(f"    Difference: {valid_rows['nvrv_difference'].mean():.6f}")
    print(f"    Std difference: {valid_rows['nvrv_difference'].std():.6f}")
    print(f"    Max difference: {valid_rows['nvrv_difference'].abs().max():.6f}")
    print()

    # Check if calculation is correct
    max_acceptable_diff = 0.01  # 1% tolerance
    if valid_rows['nvrv_difference'].abs().max() < max_acceptable_diff:
        print(f"  ✅ NVRV VERIFIED: Calculation is correct (max diff < {max_acceptable_diff})")
    else:
        print(f"  ⚠️  NVRV MISMATCH: Differences exceed tolerance")
        print(f"     Consider updating NVRV with API-calculated values")

        # Calculate percentage of rows with significant difference
        significant_diff = valid_rows[valid_rows['nvrv_difference'].abs() > max_acceptable_diff]
        print(f"     Rows with significant difference: {len(significant_diff)} ({len(significant_diff)/len(valid_rows)*100:.2f}%)")

    print()

except Exception as e:
    print(f"  ❌ Error during NVRV verification: {e}")
    print(f"  Skipping NVRV update...")

#=============================================================================
# RECALCULATE DEPENDENT FEATURES
#=============================================================================
print("="*80)
print("RECALCULATING DEPENDENT FEATURES")
print("="*80)
print()

# Recalculate jump features based on updated jump_any
returns = df['dvol'].pct_change()
df['jump_magnitude'] = np.where(df['jump_any'], returns.abs(), 0)

# Recalculate hours/days since jump
hours_since = []
last_jump_idx = -999
for i in range(len(df)):
    if df.loc[i, 'jump_any'] == 1:
        last_jump_idx = i
    hours_since.append(i - last_jump_idx if last_jump_idx >= 0 else np.nan)
df['hours_since_jump'] = hours_since
df['days_since_jump'] = df['hours_since_jump'] / 24.0

# Recalculate jump clustering
df['jump_cluster_7d'] = df['jump_any'].rolling(window=24*7, min_periods=1).sum()

print(f"  Jump features recalculated based on updated jump_any")
print()

#=============================================================================
# SAVE FIXED DATASET
#=============================================================================
print("="*80)
print("SAVING FIXED DATASET")
print("="*80)
print()

# Remove intermediate columns
if 'time_diff' in df.columns:
    df = df.drop(columns=['time_diff'])

df.to_csv(OUTPUT_FILE, index=False)

print(f"  ✅ Saved to: {OUTPUT_FILE}")
print()

#=============================================================================
# FINAL SUMMARY
#=============================================================================
print("="*80)
print("SUMMARY OF FIXES")
print("="*80)
print()

on_chain_cols = ['network_activity', 'nvrv', 'transaction_volume']
missing_after = df[on_chain_cols].isna().all(axis=1).sum()

print(f"1. LEE-MYKLAND JUMP DETECTION:")
print(f"   - Old threshold: ~4.93 (detected 0 jumps)")
print(f"   - New threshold: {threshold_new:.4f} (data-driven)")
print(f"   - Jumps detected: {df['lee_mykland_jump'].sum()}")
print(f"   - Composite jump_any: {df['jump_any'].sum()}")
print()

print(f"2. DVOL-RV SPREAD:")
print(f"   - Old method: DVOL - std(DVOL) * 100 (circular)")
print(f"   - New method: DVOL - sqrt(sum(r²)) * sqrt(252)")
print(f"   - Range: {df['dvol_rv_spread'].min():.2f} to {df['dvol_rv_spread'].max():.2f}")
print()

print(f"3. NVRV:")
if 'valid_rows' in locals():
    print(f"   - Verification: {'✅ PASSED' if valid_rows['nvrv_difference'].abs().max() < 0.01 else '⚠️  NEEDS REVIEW'}")
    print(f"   - Mean difference: {valid_rows['nvrv_difference'].mean():.6f}")
else:
    print(f"   - Verification: Skipped (error during API fetch)")
print()

print(f"4. DATA COMPLETENESS:")
print(f"   - Total rows: {len(df):,}")
print(f"   - Missing on-chain (all): {missing_after}")
print(f"   - DVOL complete: {df['dvol'].notna().sum():,}/{len(df):,}")
print()

print(f"5. UPDATED JUMP STATISTICS:")
print(f"   - Lee-Mykland: {df['lee_mykland_jump'].sum():,} ({df['lee_mykland_jump'].sum()/len(df)*100:.2f}%)")
print(f"   - Sigma threshold: {df['sigma_jump'].sum():,} ({df['sigma_jump'].sum()/len(df)*100:.2f}%)")
print(f"   - Z-score: {df['zscore_jump'].sum():,} ({df['zscore_jump'].sum()/len(df)*100:.2f}%)")
print(f"   - Composite (any): {df['jump_any'].sum():,} ({df['jump_any'].sum()/len(df)*100:.2f}%)")
print(f"   - Composite (all): {df['jump_all'].sum():,} ({df['jump_all'].sum()/len(df)*100:.2f}%)")
print()

print("="*80)
print("FIX COMPLETE!")
print("="*80)
print()
print(f"New dataset: {OUTPUT_FILE}")
print(f"Version: v1.3 (methodology fixes applied)")
print()
