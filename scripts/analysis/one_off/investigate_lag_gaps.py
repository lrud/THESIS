#!/usr/bin/env python3
"""
Investigate Lag Feature Missing Values in v1.4 Dataset

This script investigates the unexpected missing values in dvol_lag_30d and dvol_lag_7d
to determine the root cause.

Date: 2026-02-25
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("LAG FEATURE MISSING VALUES INVESTIGATION")
print("=" * 80)
print()

# Load both datasets
print("Loading datasets...")
v13 = pd.read_csv("data/processed/bitcoin_lstm_features_v1.3_complete_fixed.csv")
v14 = pd.read_csv("data/processed/bitcoin_lstm_features_v1.4_standard_lm.csv")

v13['timestamp'] = pd.to_datetime(v13['timestamp'])
v14['timestamp'] = pd.to_datetime(v14['timestamp'])

print(f"v1.3: {len(v13):,} rows")
print(f"v1.4: {len(v14):,} rows")
print()

#=============================================================================
# 1. Compare missing value patterns
#=============================================================================
print("=" * 80)
print("1. MISSING VALUE PATTERNS COMPARISON")
print("=" * 80)
print()

lag_cols = ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d']

for col in lag_cols:
    v13_missing = v13[col].isna().sum()
    v14_missing = v14[col].isna().sum()
    match = "✅ MATCH" if v13_missing == v14_missing else "❌ DIFFER"
    print(f"{col}:")
    print(f"  v1.3: {v13_missing} missing")
    print(f"  v1.4: {v14_missing} missing")
    print(f"  {match}")
    print()

#=============================================================================
# 2. Check if missing patterns are identical
#=============================================================================
print("=" * 80)
print("2. MISSING LOCATION COMPARISON")
print("=" * 80)
print()

for col in lag_cols:
    v13_na_idx = set(v13[v13[col].isna()].index)
    v14_na_idx = set(v14[v14[col].isna()].index)

    if v13_na_idx == v14_na_idx:
        print(f"{col}: ✅ IDENTICAL missing locations")
    else:
        only_v13 = len(v13_na_idx - v14_na_idx)
        only_v14 = len(v14_na_idx - v13_na_idx)
        overlap = len(v13_na_idx & v14_na_idx)
        print(f"{col}: ❌ DIFFERENT missing locations")
        print(f"  Only in v1.3: {only_v13}")
        print(f"  Only in v1.4: {only_v14}")
        print(f"  Overlap: {overlap}")
    print()

#=============================================================================
# 3. Check if missing values correlate with DVOL missing
#=============================================================================
print("=" * 80)
print("3. DVOL COMPLETENESS CHECK")
print("=" * 80)
print()

v13_dvol_missing = v13['dvol'].isna().sum()
v14_dvol_missing = v14['dvol'].isna().sum()

print(f"v1.3 DVOL missing: {v13_dvol_missing}")
print(f"v1.4 DVOL missing: {v14_dvol_missing}")
print()

# Check if lag feature NaNs occur at same locations as DVOL NaNs
print("Checking if lag NaNs correlate with DVOL NaNs...")
for col in lag_cols:
    lag_na = v14[col].isna()
    dvol_na = v14['dvol'].isna()

    # Count where both are NaN
    both_na = (lag_na & dvol_na).sum()
    lag_only_na = (lag_na & ~dvol_na).sum()

    print(f"{col}:")
    print(f"  Both lag & DVOL NaN: {both_na}")
    print(f"  Only lag NaN: {lag_only_na}")
    print()

#=============================================================================
# 4. Examine specific missing periods
#=============================================================================
print("=" * 80)
print("4. MISSING VALUE TIMELINE ANALYSIS")
print("=" * 80)
print()

# Find rows where dvol_lag_30d is missing (excluding first 720 startup rows)
lag_30d_na = v14[v14['dvol_lag_30d'].isna()].copy()

# Separate startup vs unexpected
startup_rows = 720
startup_na = lag_30d_na[lag_30d_na.index < startup_rows]
unexpected_na = lag_30d_na[lag_30d_na.index >= startup_rows]

print(f"dvol_lag_30d missing:")
print(f"  First 720 rows (expected): {len(startup_na)}")
print(f"  After row 720 (unexpected): {len(unexpected_na)}")
print()

if len(unexpected_na) > 0:
    print("Unexpected missing periods (sample):")
    sample = unexpected_na[['timestamp', 'dvol', 'dvol_lag_30d', 'dvol_lag_7d']].head(20)
    for idx, row in sample.iterrows():
        print(f"  Row {idx}: {row['timestamp']} - DVOL={row['dvol']}, lag_30d={row['dvol_lag_30d']}, lag_7d={row['dvol_lag_7d']}")
    print()

    # Check for continuous blocks
    print("Analyzing missing blocks...")
    unexpected_indices = unexpected_na.index.tolist()
    blocks = []
    if unexpected_indices:
        current_block = [unexpected_indices[0]]
        for i in range(1, len(unexpected_indices)):
            if unexpected_indices[i] == unexpected_indices[i-1] + 1:
                current_block.append(unexpected_indices[i])
            else:
                blocks.append(current_block)
                current_block = [unexpected_indices[i]]
        blocks.append(current_block)

    print(f"  Found {len(blocks)} separate missing blocks:")
    for i, block in enumerate(blocks[:10]):  # Show first 10
        start_ts = v14.loc[block[0], 'timestamp']
        end_ts = v14.loc[block[-1], 'timestamp']
        print(f"    Block {i+1}: rows {block[0]}-{block[-1]} ({len(block)} rows)")
        print(f"             {start_ts} to {end_ts}")
    if len(blocks) > 10:
        print(f"    ... and {len(blocks) - 10} more blocks")
    print()

#=============================================================================
# 5. Verify lag calculation correctness
#=============================================================================
print("=" * 80)
print("5. LAG CALCULATION VERIFICATION")
print("=" * 80)
print()

# Manually calculate lag_30d for a sample and compare
print("Verifying lag calculation on sample of non-missing values...")
non_na_lag30 = v14[v14['dvol_lag_30d'].notna()].index

if len(non_na_lag30) > 0:
    # Check first 100 non-missing rows
    sample_indices = non_na_lag30[:100]
    mismatches = 0

    for idx in sample_indices:
        if idx >= 720:  # Only check after startup period
            expected_lag = v14.loc[idx - 720, 'dvol']
            actual_lag = v14.loc[idx, 'dvol_lag_30d']

            if not np.isnan(expected_lag) and not np.isnan(actual_lag):
                if abs(expected_lag - actual_lag) > 0.0001:
                    mismatches += 1

    print(f"  Checked {len(sample_indices)} rows")
    print(f"  Mismatches: {mismatches}")
    print(f"  ✅ Lag calculation appears correct" if mismatches == 0 else "❌ Lag calculation has issues")
    print()

#=============================================================================
# 6. Summary and root cause analysis
#=============================================================================
print("=" * 80)
print("6. ROOT CAUSE ANALYSIS")
print("=" * 80)
print()

print("Key Findings:")
print()

# Check if v1.3 has same issue
v13_lag_30d_na_after_startup = v13[v14['dvol_lag_30d'].isna() & (v14.index >= 720)]
print(f"1. Inheritance from v1.3:")
print(f"   - Missing pattern in v1.4 matches v1.3: {'YES' if v13['dvol_lag_30d'].isna().sum() == v14['dvol_lag_30d'].isna().sum() else 'NO'}")
print()

print(f"2. DVOL source data:")
print(f"   - DVOL has NO missing values: {v14['dvol'].notna().all()}")
print(f"   - Lag NaNs occur where DVOL is present: {lag_only_na > 0}")
print()

print(f"3. Lag calculation:")
print(f"   - Lag shift operation appears correct: {mismatches == 0}")
print()

print("CONCLUSION:")
print()
if v13['dvol_lag_30d'].isna().sum() == v14['dvol_lag_30d'].isna().sum():
    print("✅ The lag feature missing pattern was INHERITED from v1.3")
    print("   This is NOT a new issue introduced in v1.4")
    print()
    print("ROOT CAUSE: The lag features were calculated incorrectly in the original")
    print("   dataset creation (likely in an earlier version before v1.3), and this")
    print("   pattern has been preserved through subsequent versions.")
    print()
    print("RECOMMENDATION: Recalculate lag features from DVOL using shift(24), shift(168),")
    print("   shift(720) to ensure all lag values are properly computed.")
else:
    print("❌ Unexpected difference between v1.3 and v1.4")
    print("   Further investigation needed.")

print()
print("=" * 80)
