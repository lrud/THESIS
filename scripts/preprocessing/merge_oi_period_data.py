#!/usr/bin/env python3
"""
Merge DVOL, features, and OI data for the OI availability period.

Purpose:
- Combine DVOL features, network activity, and options OI
- Filter to OI availability period (May 22, 2025 - Oct 16, 2025)
- Prepare for OI-augmented LSTM model training

Output:
- data/processed/oi_period_merged_data.csv
  Contains: timestamp, dvol, DVOL lags, network metrics, OI data
  Period: May 22 - Oct 16, 2025 (3,535 hours with OI)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

print("=" * 80)
print("MERGING DATA FOR OI AVAILABILITY PERIOD")
print("=" * 80)

# Load all datasets
print("\n1. Loading datasets...")
print("-" * 80)

# DVOL and features
print("   Loading DVOL features (bitcoin_lstm_features.csv)...")
features = pd.read_csv('data/processed/bitcoin_lstm_features.csv')
features['timestamp'] = pd.to_datetime(features['timestamp'])

# Options OI
print("   Loading OI bootstrap (options_oi_hourly.csv)...")
oi = pd.read_csv('data/processed/options_oi_hourly.csv')
oi['timestamp'] = pd.to_datetime(oi['timestamp'])

# Raw DVOL data (for any additional processing if needed)
print("   Loading raw DVOL (bitcoin_dvol_hourly_complete.csv)...")
raw_dvol = pd.read_csv('data/raw/bitcoin_dvol_hourly_complete.csv')
raw_dvol['timestamp_unix'] = pd.to_numeric(raw_dvol['timestamp'])
raw_dvol['timestamp_dt'] = pd.to_datetime(raw_dvol['timestamp_unix'] / 1000, unit='s')

print(f"\n   Features shape: {features.shape}")
print(f"   OI shape: {oi.shape}")
print(f"   Raw DVOL shape: {raw_dvol.shape}")

# Define OI period
oi_start = pd.Timestamp('2025-05-22')
oi_end = pd.Timestamp('2025-10-16 23:00:00')

print(f"\n2. OI Availability Period")
print("-" * 80)
print(f"   Start: {oi_start.date()}")
print(f"   End: {oi_end.date()}")

# Filter features to OI period
print(f"\n3. Filtering data to OI period...")
print("-" * 80)

features_oi_period = features[
    (features['timestamp'] >= oi_start) & 
    (features['timestamp'] <= oi_end)
].copy()

print(f"   Features in OI period: {len(features_oi_period)} records")
print(f"   Features date range: {features_oi_period['timestamp'].min()} to {features_oi_period['timestamp'].max()}")

oi_data = oi[
    (oi['timestamp'] >= oi_start) & 
    (oi['timestamp'] <= oi_end)
].copy()

print(f"   OI in OI period: {len(oi_data)} records")
print(f"   OI date range: {oi_data['timestamp'].min()} to {oi_data['timestamp'].max()}")

# Merge on timestamp
print(f"\n4. Merging datasets...")
print("-" * 80)

# Sort both by timestamp for proper merging
features_oi_period = features_oi_period.sort_values('timestamp').reset_index(drop=True)
oi_data = oi_data.sort_values('timestamp').reset_index(drop=True)

# Merge on exact timestamp match
merged = pd.merge(
    features_oi_period,
    oi_data,
    on='timestamp',
    how='inner'  # Keep only records where both have data
)

print(f"   Merged records: {len(merged)}")
print(f"   Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")

# Check for gaps
print(f"\n5. Data continuity check...")
print("-" * 80)

# Convert to datetime if not already
merged['timestamp'] = pd.to_datetime(merged['timestamp'])
merged_sorted = merged.sort_values('timestamp')

# Calculate gaps between records
time_diffs = merged_sorted['timestamp'].diff()
max_gap = time_diffs.max()
gaps_over_1h = (time_diffs > pd.Timedelta(hours=1.5)).sum()

print(f"   Total records: {len(merged_sorted)}")
print(f"   Max gap between records: {max_gap}")
print(f"   Number of gaps > 1.5 hours: {gaps_over_1h}")
print(f"   Continuity: {(1 - gaps_over_1h/len(merged_sorted))*100:.2f}%")

# Display column summary
print(f"\n6. Final merged dataset structure...")
print("-" * 80)
print(f"   Columns ({len(merged.columns)}):")
for i, col in enumerate(merged.columns, 1):
    print(f"      {i:2d}. {col}")

print(f"\n   Data types:")
for col in merged.columns:
    print(f"      {col:30s} {str(merged[col].dtype):15s}")

# Basic statistics
print(f"\n7. Data statistics...")
print("-" * 80)
numeric_cols = merged.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    print(f"\n   {col}:")
    print(f"      Mean: {merged[col].mean():.2f}")
    print(f"      Std:  {merged[col].std():.2f}")
    print(f"      Min:  {merged[col].min():.2f}")
    print(f"      Max:  {merged[col].max():.2f}")
    print(f"      NaN:  {merged[col].isna().sum()}")

# Check for missing values
print(f"\n8. Missing values check...")
print("-" * 80)
missing = merged.isna().sum()
if missing.sum() > 0:
    print(f"   Columns with missing values:")
    for col in missing[missing > 0].index:
        print(f"      {col}: {missing[col]} values ({missing[col]/len(merged)*100:.1f}%)")
else:
    print(f"   ✓ No missing values")

# Save merged dataset
output_file = 'data/processed/oi_period_merged_data.csv'
print(f"\n9. Saving merged dataset...")
print("-" * 80)
merged.to_csv(output_file, index=False)
print(f"   ✓ Saved to: {output_file}")
print(f"   Size: {os.path.getsize(output_file) / 1024:.1f} KB")
print(f"   Records: {len(merged)}")

# Display sample of merged data
print(f"\n10. Sample of merged data (first 5 records)...")
print("-" * 80)
print(merged.head().to_string())

print(f"\n11. Sample of merged data (last 5 records)...")
print("-" * 80)
print(merged.tail().to_string())

print(f"\n" + "=" * 80)
print("✓ MERGE COMPLETE")
print("=" * 80)
print(f"\nSummary:")
print(f"  • Period: May 22 - Oct 16, 2025")
print(f"  • Records: {len(merged)}")
print(f"  • Features: DVOL (with lags), network activity, NVRV, OI aggregates")
print(f"  • Ready for: OI-augmented LSTM training")
print(f"  • File: {output_file}")
