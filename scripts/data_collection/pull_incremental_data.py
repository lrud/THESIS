#!/usr/bin/env python3
"""
Incremental Data Pull Script
Uses existing classes to pull missing data from Oct 15, 2025 to present.
"""

import sys
import os
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.append("/home/lrud1314/PROJECTS_WORKING/THESIS 2025/scripts/data_collection")
sys.path.append("/home/lrud1314/PROJECTS_WORKING/THESIS 2025/deribit_data_collector")

from researchbitcoin_data import ResearchBitcoinCollector
from btc_volatility_collector import BitcoinVolatilityCollector

BASE_DIR = Path("/home/lrud1314/PROJECTS_WORKING/THESIS 2025")
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = DATA_DIR / "archive"

# Version info
V1_0_DATE = "2025-10-15"
V1_1_DATE = datetime.now().strftime("%Y-%m-%d")


def archive_datasets():
    """Archive current v1.0 datasets."""
    print("\n" + "="*60)
    print("STEP 1: ARCHIVING V1.0 DATASETS")
    print("="*60)

    v1_0_dir = ARCHIVE_DIR / f"v1.0_{V1_0_DATE}"
    v1_0_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("data/raw/bitcoin_dvol_hourly_complete.csv", "bitcoin_dvol_hourly_complete.csv"),
        ("data/raw/bitcoin_nvrv_hourly_20251015.csv", "bitcoin_nvrv_hourly_20251015.csv"),
        ("data/processed/bitcoin_lstm_features.csv", "bitcoin_lstm_features.csv"),
    ]

    for src_path, dest_name in files:
        src = BASE_DIR / src_path
        dest = v1_0_dir / dest_name
        if src.exists() and not dest.exists():
            shutil.copy2(src, dest)
            print(f"  ✅ Archived: {dest_name}")
        elif dest.exists():
            print(f"  ⊙ Already archived: {dest_name}")

    return v1_0_dir


def pull_dvol_data():
    """Pull incremental DVOL data."""
    print("\n" + "="*60)
    print("STEP 2: PULLING INCREMENTAL DVOL DATA")
    print("="*60)

    collector = BitcoinVolatilityCollector("BTC")
    df = collector.collect_incremental_volatility(save=True)
    return df


def pull_on_chain_data():
    """Pull incremental on-chain data."""
    print("\n" + "="*60)
    print("STEP 3: PULLING INCREMENTAL ON-CHAIN DATA")
    print("="*60)

    api_token = os.getenv("RESEARCH_BITCOIN_API_TOKEN")
    if not api_token:
        print("⚠️  RESEARCH_BITCOIN_API_TOKEN not set - skipping on-chain data")
        return None

    collector = ResearchBitcoinCollector(api_token)
    df = collector.collect_incremental()
    return df


def merge_dvol_data(df_new):
    """Merge new DVOL data with existing."""
    print("\n" + "="*60)
    print("STEP 4: MERGING DVOL DATA")
    print("="*60)

    if df_new is None:
        print("⚠️  No new DVOL data to merge")
        return None

    # Load existing
    df_old = pd.read_csv(BASE_DIR / "data/raw/bitcoin_dvol_hourly_complete.csv")
    first_col = df_old.columns[0]
    df_old[first_col] = pd.to_datetime(df_old[first_col])
    df_old = df_old.rename(columns={first_col: 'timestamp'})

    # Check continuity
    gap = (df_new['timestamp'].min() - df_old['timestamp'].max()).total_seconds() / 3600
    print(f"  Gap: {gap:.1f} hours")

    # Merge
    df_merged = pd.concat([df_old, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last')
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

    # Save
    v1_1_dir = ARCHIVE_DIR / f"v1.1_{V1_1_DATE}"
    v1_1_dir.mkdir(exist_ok=True)
    output_file = v1_1_dir / "bitcoin_dvol_hourly_complete.csv"
    df_merged.to_csv(output_file, index=False)

    print(f"  ✅ Merged: {len(df_old)} + {len(df_new)} = {len(df_merged)} records")
    print(f"  Range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")

    return df_merged


def merge_on_chain_data(df_new):
    """Merge new on-chain data with existing."""
    print("\n" + "="*60)
    print("STEP 5: MERGING ON-CHAIN DATA")
    print("="*60)

    if df_new is None:
        print("⚠️  No new on-chain data to merge")
        return None

    # Load existing
    df_old = pd.read_csv(BASE_DIR / "data/raw/bitcoin_nvrv_hourly_20251015.csv")
    first_col = df_old.columns[0]
    df_old[first_col] = pd.to_datetime(df_old[first_col])
    df_old = df_old.rename(columns={first_col: 'timestamp'})

    # Check continuity
    gap = (df_new.index.min() - df_old['timestamp'].max()).total_seconds() / 3600
    print(f"  Gap: {gap:.1f} hours")

    # Merge
    df_new_reset = df_new.reset_index()
    df_merged = pd.concat([df_old, df_new_reset], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last')
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

    # Save
    v1_1_dir = ARCHIVE_DIR / f"v1.1_{V1_1_DATE}"
    v1_1_dir.mkdir(exist_ok=True)
    output_file = v1_1_dir / f"bitcoin_nvrv_hourly_{datetime.now().strftime('%Y%m%d')}.csv"
    df_merged.to_csv(output_file, index=False)

    print(f"  ✅ Merged: {len(df_old)} + {len(df_new)} = {len(df_merged)} records")
    print(f"  Range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")

    return df_merged


def main():
    print("\n" + "="*60)
    print("INCREMENTAL DATA COLLECTION")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Archive v1.0
    v1_0_dir = archive_datasets()

    # Pull DVOL
    df_dvol = pull_dvol_data()

    # Pull on-chain
    df_onchain = pull_on_chain_data()

    # Merge DVOL
    df_merged_dvol = merge_dvol_data(df_dvol)

    # Merge on-chain
    df_merged_onchain = merge_on_chain_data(df_onchain)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"v1.0 archived: {v1_0_dir}")
    if df_merged_dvol is not None:
        print(f"DVOL merged: {len(df_merged_dvol):,} records")
    if df_merged_onchain is not None:
        print(f"On-chain merged: {len(df_merged_onchain):,} records")

    v1_1_dir = ARCHIVE_DIR / f"v1.1_{V1_1_DATE}"
    print(f"\nv1.1 datasets: {v1_1_dir}")


if __name__ == "__main__":
    main()
