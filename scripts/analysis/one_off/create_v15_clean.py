#!/usr/bin/env python3
"""
Create v1.5 Dataset - Remove Composite Jump Indicators

This script creates v1.5 of the dataset by removing the composite jump indicators
(jump_any, jump_all) that lack academic justification, and recalculating dependent
features using only the academically sound Lee-Mykland jump detection.

Date: 2026-02-25
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
INPUT_FILE = "data/processed/bitcoin_lstm_features_v1.4_standard_lm.csv"
OUTPUT_FILE = "data/processed/bitcoin_lstm_features_v1.5_clean.csv"

print("=" * 80)
print("CREATING v1.5 DATASET - REMOVE COMPOSITE JUMP INDICATORS")
print("=" * 80)
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load v1.4 data
print("Loading v1.4 dataset...")
df = pd.read_csv(INPUT_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"  Loaded: {len(df):,} rows")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

# Store old values for comparison
old_jump_any = df['jump_any'].sum()
old_jump_all = df['jump_all'].sum()
old_lm_jumps = df['lee_mykland_jump'].sum()

print("=" * 80)
print("REMOVING COMPOSITE JUMP INDICATORS")
print("=" * 80)
print()

print("Rationale: jump_any and jump_all combine different jump detection methods")
print("(Lee-Mykland, sigma threshold, z-score) without academic justification.")
print()

print("Columns to remove:")
print("  - jump_any (OR of three methods)")
print("  - jump_all (AND of three methods)")
print("  - jump_indicator (redundant)")
print()

# Check what jump_indicator contains
if 'jump_indicator' in df.columns:
    ji_matches_any = (df['jump_indicator'] == df['jump_any']).all()
    print(f"  jump_indicator matches jump_any: {ji_matches_any}")
    print(f"  → jump_indicator is redundant and will be removed")
    print()

# Remove composite columns
cols_to_drop = ['jump_any', 'jump_all', 'jump_indicator']
cols_existing = [c for c in cols_to_drop if c in df.columns]

print(f"Removing: {cols_existing}")
df = df.drop(columns=cols_existing)
print()

print("=" * 80)
print("RECALCULATING JUMP-DEPENDENT FEATURES")
print("=" * 80)
print()

print("Using only lee_mykland_jump (academically justified Lee-Mykland 2008)")
print()

# Recalculate hours_since_jump based on lee_mykland_jump only
print("Calculating hours_since_jump and days_since_jump...")
hours_since = []
last_jump_idx = -999
for i in range(len(df)):
    if df.loc[i, 'lee_mykland_jump'] == 1:
        last_jump_idx = i
    hours_since.append(i - last_jump_idx if last_jump_idx >= 0 else np.nan)

df['hours_since_jump'] = hours_since
df['days_since_jump'] = df['hours_since_jump'] / 24.0
print(f"  First jump at row: {hours_since.index(0) if 0 in hours_since else 'N/A'}")
print()

# Recalculate jump_cluster_7d based on lee_mykland_jump only
print("Calculating jump_cluster_7d (Lee-Mykland jumps in past 7 days)...")
df['jump_cluster_7d'] = df['lee_mykland_jump'].rolling(window=24*7, min_periods=1).sum()
print(f"  Max cluster size: {df['jump_cluster_7d'].max()}")
print(f"  Mean cluster size: {df['jump_cluster_7d'].mean():.2f}")
print()

# Recalculate jump_magnitude based on lee_mykland_jump only
print("Calculating jump_magnitude...")
returns = df['dvol'].pct_change()
df['jump_magnitude'] = np.where(df['lee_mykland_jump'], returns.abs(), 0)
print(f"  Non-zero magnitudes: {(df['jump_magnitude'] > 0).sum()}")
print(f"  Mean magnitude (when > 0): {df[df['jump_magnitude'] > 0]['jump_magnitude'].mean():.4f}")
print()

print("=" * 80)
print("v1.5 DATASET SUMMARY")
print("=" * 80)
print()

print("Changes from v1.4:")
print()
print(f"1. Removed composite jump indicators:")
print(f"   - jump_any: {old_jump_any} jumps (removed)")
print(f"   - jump_all: {old_jump_all} jumps (removed)")
print()
print(f"2. Jump-dependent features recalculated using lee_mykland_jump only:")
print(f"   - hours_since_jump")
print(f"   - days_since_jump")
print(f"   - jump_cluster_7d")
print(f"   - jump_magnitude")
print()
print(f"3. Preserved individual jump detection methods:")
print(f"   - lee_mykland_jump: {old_lm_jumps} (Standard Lee-Mykland 2008)")
print(f"   - sigma_jump: {df['sigma_jump'].sum()} (complementary)")
print(f"   - zscore_jump: {df['zscore_jump'].sum()} (complementary)")
print()

# Data quality check
print("=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)
print()

on_chain_cols = ['network_activity', 'nvrv', 'transaction_volume']
missing_after = df[on_chain_cols].isna().all(axis=1).sum()

print(f"  Total rows: {len(df):,}")
print(f"  Missing on-chain (all): {missing_after}")
print(f"  DVOL complete: {df['dvol'].notna().sum():,}/{len(df):,}")
print()

# Lag feature verification (confirming they're correct)
print(f"  Lag features (expected startup missing only):")
lag_info = [
    ('dvol_lag_1d', 24),
    ('dvol_lag_7d', 168),
    ('dvol_lag_30d', 720)
]
for col, expected_startup in lag_info:
    actual_missing = df[col].isna().sum()
    status = "✅" if actual_missing == expected_startup else "❌"
    print(f"    {status} {col}: {actual_missing} missing (expected: {expected_startup})")
print()

print(f"  Columns: {len(df.columns)}")
print()

# Save v1.5 dataset
print("=" * 80)
print("SAVING v1.5 DATASET")
print("=" * 80)
print()

df.to_csv(OUTPUT_FILE, index=False)

print(f"  ✅ Saved to: {OUTPUT_FILE}")
print()

print("=" * 80)
print("v1.5 CREATION COMPLETE!")
print("=" * 80)
print()

print("Summary:")
print()
print("The v1.5 dataset removes composite jump indicators (jump_any, jump_all)")
print("that lacked academic justification. All jump-dependent features are now")
print("calculated using only the standard Lee-Mykland (2008) jump detection method.")
print()
print("Use v1.5 for model training - it is academically rigorous and clean.")
print()

# Print column list for reference
print("Columns in v1.5:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
print()
