#!/usr/bin/env python3
"""
Fill Missing Data Gaps Script

Retrieves missing DVOL and on-chain data for all identified gaps in the dataset.
Gaps are identified from the current dataset and filled using existing API collectors.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from researchbitcoin_data import ResearchBitcoinCollector
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../deribit_data_collector"))
from btc_volatility_collector import BitcoinVolatilityCollector

# Configuration
BASE_DIR = Path("/home/lrud1314/PROJECTS_WORKING/THESIS 2025")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
DERIBIT_DIR = DATA_DIR / "deribit"

# Current dataset
CURRENT_DATASET = PROCESSED_DIR / "bitcoin_lstm_features_v1.1_complete_with_jumps.csv"
OUTPUT_VERSION = "v1.2"


def identify_gaps(df: pd.DataFrame) -> List[Dict]:
    """
    Identify all time gaps in the dataset.

    Returns list of gap dictionaries with start, end, and duration info.
    """
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    df_sorted['time_diff'] = df_sorted['timestamp'].diff()
    df_sorted['gap_hours'] = df_sorted['time_diff'].dt.total_seconds() / 3600

    gaps = df_sorted[df_sorted['gap_hours'] > 1].copy()

    gap_list = []
    for idx in gaps.index:
        gap_start = df_sorted.iloc[idx - 1]['timestamp']
        gap_end = df_sorted.iloc[idx]['timestamp']
        gap_hours = gaps.loc[idx, 'gap_hours']

        gap_list.append({
            'index': idx,
            'gap_start': gap_start,
            'gap_end': gap_end,
            'gap_hours': gap_hours,
            'missing_start': gap_start + timedelta(hours=1),
            'missing_end': gap_end
        })

    # Sort by gap size (descending) to prioritize large gaps
    gap_list.sort(key=lambda x: x['gap_hours'], reverse=True)

    return gap_list


def retrieve_dvol_for_gap(gap: Dict) -> pd.DataFrame:
    """
    Retrieve DVOL data for a specific gap period using Deribit API.
    """
    print(f"\n{'='*60}")
    print(f"Retrieving DVOL for gap: {gap['gap_start']} → {gap['gap_end']}")
    print(f"Duration: {gap['gap_hours']:.0f} hours")
    print(f"{'='*60}")

    collector = BitcoinVolatilityCollector("BTC")

    # Convert to millisecond timestamps for API
    start_ts = int((gap['missing_start']).timestamp() * 1000)
    end_ts = int((gap['missing_end']).timestamp() * 1000)

    try:
        df = collector.collect_incremental_volatility(
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            save=False  # Don't save individually, will merge
        )

        if df is not None and len(df) > 0:
            print(f"  ✅ Retrieved {len(df)} DVOL records")
            return df
        else:
            print(f"  ⚠️  No DVOL data returned for this gap")
            return None

    except Exception as e:
        print(f"  ❌ Error retrieving DVOL: {e}")
        return None


def retrieve_onchain_for_gap(gap: Dict, api_token: str) -> pd.DataFrame:
    """
    Retrieve on-chain metrics for a specific gap period.
    """
    print(f"\n{'='*60}")
    print(f"Retrieving On-Chain Metrics for gap")
    print(f"{'='*60}")

    if not api_token:
        print("  ⚠️  No API token - skipping on-chain data")
        return None

    collector = ResearchBitcoinCollector(api_token)

    # Format dates for API
    from_date = gap['missing_start'].strftime('%Y-%m-%d')
    to_date = gap['missing_end'].strftime('%Y-%m-%d')

    metrics = ["price", "market_cap", "realized_cap", "transaction_volume", "txs_n"]
    dfs = []

    for metric in metrics:
        print(f"  Pulling {metric}...", end=" ")
        try:
            df = collector.get_metric(
                metric=metric,
                resolution="h1",
                from_time=from_date,
                to_time=to_date
            )
            if df is not None and len(df) > 0:
                df.columns = [metric]
                dfs.append(df)
                print(f"✅ {len(df)} records")
            else:
                print(f"⚠️  No data")
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"❌ Error: {e}")

    if not dfs:
        return None

    # Merge all metrics
    result = pd.concat(dfs, axis=1)
    print(f"  ✅ Combined {len(result)} records with {len(result.columns)} metrics")

    return result


def merge_gap_data(
    df_existing: pd.DataFrame,
    dvol_data: pd.DataFrame,
    onchain_data: pd.DataFrame,
    gap: Dict
) -> pd.DataFrame:
    """
    Merge retrieved gap data with existing dataset.
    """
    if dvol_data is None and onchain_data is None:
        print("  ⚠️  No data to merge for this gap")
        return df_existing

    # Prepare DVOL data
    if dvol_data is not None:
        dvol_df = dvol_data[['timestamp', 'dvol_close']].copy()
        dvol_df = dvol_df.rename(columns={'dvol_close': 'dvol'})
    else:
        dvol_df = None

    # Prepare on-chain data
    if onchain_data is not None:
        onchain_df = onchain_data.reset_index()
        onchain_df['timestamp'] = pd.to_datetime(onchain_df['timestamp'])
    else:
        onchain_df = None

    # Merge gap data
    if dvol_df is not None:
        gap_data = dvol_df
        if onchain_df is not None:
            gap_data = pd.merge(gap_data, onchain_df, on='timestamp', how='outer')
    elif onchain_df is not None:
        gap_data = onchain_df
    else:
        return df_existing

    # Calculate NVRV if we have the components
    if 'market_cap' in gap_data.columns and 'realized_cap' in gap_data.columns:
        gap_data['nvrv'] = (gap_data['market_cap'] - gap_data['realized_cap']) / gap_data['realized_cap']

    # Add placeholder columns for features that need recalculation
    for col in ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d', 'dvol_rv_spread',
                'lee_mykland_stat', 'lee_mykland_jump', 'sigma_threshold', 'sigma_jump',
                'return_zscore', 'zscore_jump', 'jump_any', 'jump_all', 'jump_indicator',
                'jump_magnitude', 'hours_since_jump', 'days_since_jump', 'jump_cluster_7d']:
        if col not in gap_data.columns:
            gap_data[col] = np.nan

    # Ensure correct column order
    expected_cols = df_existing.columns.tolist()
    gap_data = gap_data.reindex(columns=expected_cols)

    # Merge with existing data
    df_merged = pd.concat([df_existing, gap_data], ignore_index=True)
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

    # Remove any exact duplicates
    df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last')

    print(f"  ✅ Merged: {len(df_existing)} → {len(df_merged)} records")

    return df_merged


def recalculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate lag features and other derived features.
    """
    print(f"\n{'='*60}")
    print(f"RECALCULATING DERIVED FEATURES")
    print(f"{'='*60}")

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate lag features (time-aware would be better, but using row shift for consistency)
    print("  Calculating lag features...")
    df['dvol_lag_1d'] = df['dvol'].shift(24)
    df['dvol_lag_7d'] = df['dvol'].shift(24 * 7)
    df['dvol_lag_30d'] = df['dvol'].shift(24 * 30)

    # Calculate DVOL-RV spread (placeholder - requires realized vol calculation)
    if 'dvol_rv_spread' not in df.columns or df['dvol_rv_spread'].isna().all():
        df['dvol_rv_spread'] = np.nan  # Will need proper RV calculation

    print("  ✅ Lag features recalculated")

    return df


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("DATA GAP FILLING SCRIPT")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {CURRENT_DATASET}")
    print(f"Output Version: {OUTPUT_VERSION}")

    # Load existing data
    print(f"\nLoading existing dataset...")
    df = pd.read_csv(CURRENT_DATASET)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Loaded: {len(df):,} records")
    print(f"  Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Identify gaps
    print(f"\nIdentifying gaps...")
    gaps = identify_gaps(df)
    print(f"  Found {len(gaps)} gaps")

    if len(gaps) == 0:
        print("  ✅ No gaps found - dataset is complete!")
        return

    # Show gap summary
    print(f"\nGap Summary:")
    print(f"  Critical (>100h): {sum(1 for g in gaps if g['gap_hours'] > 100)}")
    print(f"  Large (24-100h): {sum(1 for g in gaps if 24 <= g['gap_hours'] <= 100)}")
    print(f"  Medium (2-24h): {sum(1 for g in gaps if 2 < g['gap_hours'] < 24)}")
    print(f"  Small (2h): {sum(1 for g in gaps if g['gap_hours'] == 2)}")

    # Get API token
    api_token = os.getenv("RESEARCH_BITCOIN_API_TOKEN")
    if not api_token:
        print("\n⚠️  RESEARCH_BITCOIN_API_TOKEN not set")
        print("    On-chain metrics will not be retrieved")

    # Ask which gaps to fill
    print(f"\n{'='*60}")
    print("Top 10 Largest Gaps:")
    print(f"{'='*60}")
    for i, gap in enumerate(gaps[:10], 1):
        print(f"  {i}. {gap['gap_start']} → {gap['gap_end']} ({gap['gap_hours']:.0f}h)")

    # Fill gaps
    print(f"\n{'='*60}")
    print("FILLING GAPS")
    print(f"{'='*60}")

    filled_count = 0
    skipped_count = 0

    for i, gap in enumerate(gaps, 1):
        print(f"\n[{i}/{len(gaps)}] Processing gap...")

        # Retrieve DVOL data
        dvol_data = retrieve_dvol_for_gap(gap)

        # Retrieve on-chain data
        onchain_data = None
        if api_token and gap['gap_hours'] >= 2:  # Only pull on-chain for larger gaps
            onchain_data = retrieve_onchain_for_gap(gap, api_token)

        # Merge data
        if dvol_data is not None or onchain_data is not None:
            df = merge_gap_data(df, dvol_data, onchain_data, gap)
            filled_count += 1
        else:
            skipped_count += 1

    # Recalculate features
    df = recalculate_features(df)

    # Save output
    output_file = PROCESSED_DIR / f"bitcoin_lstm_features_{OUTPUT_VERSION}_complete.csv"
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Input records: {len(pd.read_csv(CURRENT_DATASET)):,}")
    print(f"  Output records: {len(df):,}")
    print(f"  Records added: {len(df) - len(pd.read_csv(CURRENT_DATASET)):,}")
    print(f"  Gaps filled: {filled_count}")
    print(f"  Gaps skipped: {skipped_count}")
    print(f"\n  Output: {output_file}")
    print(f"\n⚠️  Next: Re-run jump detection analysis")
    print(f"     python scripts/analysis/jump_detection_analysis.py \\")
    print(f"       --data-path {output_file} \\")
    print(f"       --data-version {OUTPUT_VERSION}")


if __name__ == "__main__":
    main()
