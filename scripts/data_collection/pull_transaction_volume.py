#!/usr/bin/env python3
"""
Pull Missing Transaction Volume Data
Pulls transaction_volume (spent_volume_usd) from Bitcoin Researcher's Lab API
for the missing period: Oct 15, 2025 to Dec 28, 2025
"""

import sys
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.append("/home/lrud1314/PROJECTS_WORKING/THESIS 2025/scripts/data_collection")
from researchbitcoin_data import ResearchBitcoinCollector

BASE_DIR = Path("/home/lrud1314/PROJECTS_WORKING/THESIS 2025")
DATA_DIR = BASE_DIR / "data"

def pull_transaction_volume(start_date: str = "2025-10-15", end_date: str = "2025-12-28"):
    """Pull transaction volume data for specified date range."""
    api_token = os.getenv("RESEARCH_BITCOIN_API_TOKEN")
    if not api_token:
        print("ERROR: RESEARCH_BITCOIN_API_TOKEN not set")
        return None

    print("="*60)
    print("PULLING TRANSACTION VOLUME DATA")
    print("="*60)
    print(f"Date range: {start_date} to {end_date}")

    collector = ResearchBitcoinCollector(api_token)

    # Pull spent_tx_volume_usd metric (transaction_volume)
    print("\nCollecting spent_tx_volume_usd (transaction_volume)...")
    df = collector.get_metric("spent_tx_volume_usd", resolution="h1", from_time=start_date)

    # Filter to end date (handle timezone-aware comparison)
    df = df[df.index <= pd.to_datetime(end_date).tz_localize('UTC')]

    print(f"Collected: {len(df)} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df

def main():
    df_vol = pull_transaction_volume()

    if df_vol is not None:
        # Save to archive
        output_dir = BASE_DIR / "data/archive/v1.1_2025-12-29"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "transaction_volume_hourly.csv"
        df_vol.to_csv(output_file)
        print(f"\nSaved to: {output_file}")

        # Display summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Records: {len(df_vol)}")
        print(f"Date range: {df_vol.index.min()} to {df_vol.index.max()}")
        print(f"Mean volume: ${df_vol['spent_tx_volume_usd'].mean():,.0f}")
        print(f"Median volume: ${df_vol['spent_tx_volume_usd'].median():,.0f}")

if __name__ == "__main__":
    main()
