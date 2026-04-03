#!/usr/bin/env python3
"""
Update v1.3 Dataset with Standard Lee-Mykland Jump Detection

This script creates v1.4 of the dataset, replacing the data-driven jump
detection with the academically correct Lee-Mykland (2008) implementation.

Date: 2026-02-25
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add project path
sys.path.insert(0, '/home/lrud1314/PROJECTS_WORKING/THESIS 2025')

# Import standard Lee-Mykland implementation
from scripts.analysis.standard_lee_mykland import lee_mykland_test

# Configuration
INPUT_FILE = "data/processed/bitcoin_lstm_features_v1.3_complete_fixed.csv"
OUTPUT_FILE = "data/processed/bitcoin_lstm_features_v1.4_standard_lm.csv"

print("=" * 80)
print("CREATING v1.4 DATASET WITH STANDARD LEE-MYKLAND JUMPS")
print("=" * 80)
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load data
print("Loading v1.3 dataset...")
df = pd.read_csv(INPUT_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"  Loaded: {len(df):,} rows")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

# Store old jump statistics for comparison
old_lm_jumps = df['lee_mykland_jump'].sum()
old_jump_any = df['jump_any'].sum()

# Run standard Lee-Mykland test
print("=" * 80)
print("APPLYING STANDARD LEE-MYKLAND (2008) JUMP DETECTION")
print("=" * 80)
print()

print("Running standard Lee-Mykland test on DVOL...")
jumps_standard, T_stats, L_stats, beta_star = lee_mykland_test(
    df['dvol'],
    significance_level=0.0001
)

print(f"  Critical value (β*): {beta_star:.4f}")
print(f"  Jumps detected: {jumps_standard.sum()} ({jumps_standard.sum()/len(jumps_standard)*100:.2f}%)")
print(f"  Max T-statistic: {T_stats.max():.4f}")
print(f"  Mean |L|: {np.abs(L_stats).mean():.4f}")
print()

# Comparison with v1.3
print("=" * 80)
print("COMPARISON: v1.3 vs Standard Lee-Mykland")
print("=" * 80)
print()

overlap = (jumps_standard == 1) & (df['lee_mykland_jump'] == 1)
only_standard = (jumps_standard == 1) & (df['lee_mykland_jump'] == 0)
only_v13 = (jumps_standard == 0) & (df['lee_mykland_jump'] == 1)

print(f"v1.3 Lee-Mykland jumps: {old_lm_jumps} ({old_lm_jumps/len(df)*100:.2f}%)")
print(f"Standard Lee-Mykland jumps: {jumps_standard.sum()} ({jumps_standard.sum()/len(df)*100:.2f}%)")
print(f"  Overlap: {overlap.sum()} jumps")
print(f"  Only Standard: {only_standard.sum()} jumps")
print(f"  Only v1.3: {only_v13.sum()} jumps")
print()

# Update dataframe with standard Lee-Mykland results
print("=" * 80)
print("UPDATING DATASET")
print("=" * 80)
print()

# Save old values for reference
df['lee_mykland_jump_v13'] = df['lee_mykland_jump']
df['lee_mykland_stat_v13'] = df['lee_mykland_stat']

# Update with standard Lee-Mykland values
df['lee_mykland_stat'] = L_stats.values
df['lee_mykland_jump'] = jumps_standard.values
df['lee_mykland_T_statistic'] = T_stats.values
df['lee_mykland_threshold'] = beta_star

# Recalculate composite jump indicators
# The standard Lee-Mykland is now the authoritative jump detection method
# We keep sigma_jump and zscore_jump as complementary indicators
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

print(f"  Updated lee_mykland_stat: L-statistic values")
print(f"  Updated lee_mykland_jump: {df['lee_mykland_jump'].sum()} jumps")
print(f"  Added lee_mykland_T_statistic: Normalized test statistic")
print(f"  Added lee_mykland_threshold: {beta_star:.4f}")
print()

# Recalculate derived jump features
print("Recalculating derived jump features...")

returns = df['dvol'].pct_change()
df['jump_magnitude'] = np.where(df['jump_any'], returns.abs(), 0)

# Recalculate hours/days since jump (using updated jump_any)
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

print(f"  Updated jump_magnitude based on jump_any")
print(f"  Updated hours_since_jump and days_since_jump")
print(f"  Updated jump_cluster_7d")
print()

# Print updated jump statistics
print("=" * 80)
print("UPDATED JUMP STATISTICS (v1.4)")
print("=" * 80)
print()

print(f"Lee-Mykland: {df['lee_mykland_jump'].sum():,} ({df['lee_mykland_jump'].sum()/len(df)*100:.2f}%)")
print(f"Sigma threshold: {df['sigma_jump'].sum():,} ({df['sigma_jump'].sum()/len(df)*100:.2f}%)")
print(f"Z-score: {df['zscore_jump'].sum():,} ({df['zscore_jump'].sum()/len(df)*100:.2f}%)")
print(f"Composite (any): {df['jump_any'].sum():,} ({df['jump_any'].sum()/len(df)*100:.2f}%)")
print(f"Composite (all): {df['jump_all'].sum():,} ({df['jump_all'].sum()/len(df)*100:.2f}%)")
print()

# Save v1.4 dataset
print("=" * 80)
print("SAVING v1.4 DATASET")
print("=" * 80)
print()

# Drop temporary columns if any
if 'time_diff' in df.columns:
    df = df.drop(columns=['time_diff'])

df.to_csv(OUTPUT_FILE, index=False)

print(f"  ✅ Saved to: {OUTPUT_FILE}")
print()

# Final summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"Input file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print()

print("Changes from v1.3 to v1.4:")
print()
print("  1. Lee-Mykland jump detection:")
print(f"     - Old (v1.3): {old_lm_jumps} jumps (data-driven threshold)")
print(f"     - New (v1.4): {df['lee_mykland_jump'].sum()} jumps (standard Lee-Mykland 2008)")
print(f"     - Change: {df['lee_mykland_jump'].sum() - old_lm_jumps:+d} jumps")
print()
print("  2. Methodology:")
print("     - v1.3: Data-driven threshold (99.9th percentile)")
print("     - v1.4: Standard Gumbel EV threshold (β* = 9.21)")
print()
print("  3. Academic validity:")
print("     - v1.4 now follows Lee & Mykland (2008) specification")
print("     - Jumps detected align with academic standards")
print()

# Data quality check
print("Data quality check:")
print()
on_chain_cols = ['network_activity', 'nvrv', 'transaction_volume']
missing_after = df[on_chain_cols].isna().all(axis=1).sum()
print(f"  Total rows: {len(df):,}")
print(f"  Missing on-chain (all): {missing_after}")
print(f"  DVOL complete: {df['dvol'].notna().sum():,}/{len(df):,}")
print()

print("=" * 80)
print("v1.4 DATASET CREATION COMPLETE!")
print("=" * 80)
print()
print("The v1.4 dataset uses the academically correct Lee-Mykland (2008)")
print("jump detection method and is ready for model training.")
print()
