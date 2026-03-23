#!/usr/bin/env python3
"""
Create v1.6 Dataset - Final Clean Version with Lee-Mykland Only

This script creates v1.6 by removing alternative jump detection methods
(sigma_jump, zscore_jump) and keeping only the academically rigorous
Lee-Mykland (2008) jump detection.

Author: Claude AI
Date: 2026-02-25
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
INPUT_FILE = "data/processed/bitcoin_lstm_features_v1.5_clean.csv"
OUTPUT_FILE = "data/processed/bitcoin_lstm_features_v1.6_final.csv"

print("=" * 80)
print("CREATING v1.6 DATASET - LEE-MYKLAND ONLY")
print("=" * 80)
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load v1.5 data
print("Loading v1.5 dataset...")
df = pd.read_csv(INPUT_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"  Loaded: {len(df):,} rows")
print(f"  Columns: {len(df.columns)}")
print()

print("=" * 80)
print("REMOVING ALTERNATIVE JUMP DETECTION METHODS")
print("=" * 80)
print()

print("Rationale: Lee-Mykland (2008) is the academically rigorous gold standard.")
print("Sigma threshold and z-score are separate methods not needed for Lee-Mykland.")
print()

# Store stats before removal
old_sigma_jumps = df['sigma_jump'].sum()
old_zscore_jumps = df['zscore_jump'].sum()
lm_jumps = df['lee_mykland_jump'].sum()

print(f"Jump statistics before removal:")
print(f"  Lee-Mykland: {lm_jumps} ({lm_jumps/len(df)*100:.2f}%)")
print(f"  Sigma threshold: {old_sigma_jumps} ({old_sigma_jumps/len(df)*100:.2f}%)")
print(f"  Z-score: {old_zscore_jumps} ({old_zscore_jumps/len(df)*100:.2f}%)")
print()

# Columns to remove
cols_to_remove = [
    'sigma_jump',      # Alternative method
    'sigma_threshold',  # Only used for sigma_jump
    'zscore_jump',      # Alternative method
    'return_zscore'     # Only used for zscore_jump
]

# Verify columns exist
cols_existing = [c for c in cols_to_remove if c in df.columns]

print(f"Removing {len(cols_existing)} columns:")
for col in cols_existing:
    print(f"  - {col}")

# Remove columns
df = df.drop(columns=cols_existing)
print()

print("=" * 80)
print("v1.6 DATASET SUMMARY")
print("=" * 80)
print()

print(f"Final column count: {len(df.columns)} (was 23)")
print()

print("Columns in v1.6:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
print()

# Data quality check
print("=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)
print()

on_chain_cols = ['network_activity', 'nvrv', 'transaction_volume']
missing_after = df[on_chain_cols].isna().all(axis=1).sum()

print(f"  Total rows: {len(df):,}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Missing on-chain (all): {missing_after}")
print(f"  DVOL complete: {df['dvol'].notna().sum():,}/{len(df):,}")
print()

print(f"Jump detection (Lee-Mykland only):")
print(f"  Jumps detected: {lm_jumps} ({lm_jumps/len(df)*100:.2f}%)")
print(f"  Threshold: {df['lee_mykland_threshold'].iloc[0]:.4f}")
print(f"  Max T-statistic: {df['lee_mykland_T_statistic'].max():.2f}")
print()

# Save v1.6 dataset
print("=" * 80)
print("SAVING v1.6 DATASET")
print("=" * 80)
print()

df.to_csv(OUTPUT_FILE, index=False)

print(f"  ✅ Saved to: {OUTPUT_FILE}")
print()

print("=" * 80)
print("v1.6 CREATION COMPLETE!")
print("=" * 80)
print()

print("Summary:")
print()
print("The v1.6 dataset is the final, clean version with:")
print("  ✅ Only Lee-Mykland (2008) jump detection")
print("  ✅ Academically rigorous and publication-ready")
print("  ✅ 19 columns (down from 23)")
print("  ✅ All alternative methods removed")
print()
print("USE v1.6 FOR ALL MODEL TRAINING.")
print()
