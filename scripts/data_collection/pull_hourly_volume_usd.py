#!/usr/bin/env python3
"""
Pull Hourly Transaction Volume in USD
Combines /v1/stats/tx/volume (Satoshis) with /v1/markets/price/usd/ to calculate hourly USD volume.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://api.researchbitcoin.net"

def get_hourly_transaction_volume_usd(start_ts, end_ts, api_token=None):
    """
    Fetches hourly transaction volume and converts to USD.

    Args:
        start_ts: Start Unix timestamp
        end_ts: End Unix timestamp
        api_token: Optional API token

    Returns:
        List of dicts with timestamp, volume_btc, price_usd, volume_usd
    """
    print(f"Fetching hourly transaction volume from {datetime.fromtimestamp(start_ts, tz=timezone.utc)} to {datetime.fromtimestamp(end_ts, tz=timezone.utc)}...")

    # 1. Get hourly volume in Satoshis
    volume_url = f"{BASE_URL}/v1/stats/tx/volume"
    volume_params = {
        'from': start_ts,
        'to': end_ts,
        'window': 3600  # 1-hour window
    }

    headers = {}
    if api_token:
        headers['Authorization'] = f'Bearer {api_token}'

    try:
        volume_response = requests.get(volume_url, params=volume_params, headers=headers, timeout=30)
        volume_response.raise_for_status()
        volume_data = volume_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching volume data: {e}")
        return []

    if not volume_data:
        print("No volume data returned")
        return []

    print(f"Retrieved {len(volume_data)} hourly data points")

    hourly_results = []

    # 2. For each hour, get the price and calculate USD value
    for i, (hour_timestamp, satoshi_volume) in enumerate(sorted(volume_data.items())):
        if i % 100 == 0:
            print(f"  Processing hour {i+1}/{len(volume_data)}...")

        # Get the price at the start of the hour
        price_url = f"{BASE_URL}/v1/markets/price/usd/"
        price_params = {'timestamp': hour_timestamp}

        try:
            price_response = requests.get(price_url, params=price_params, headers=headers, timeout=10)
            price_response.raise_for_status()
            price_data = price_response.json()
            price_usd = price_data.get('price')

            if price_usd is not None:
                # 3. Perform the calculation
                btc_volume = int(satoshi_volume) / 100_000_000
                usd_volume = btc_volume * price_usd

                result = {
                    'timestamp': datetime.fromtimestamp(int(hour_timestamp), tz=timezone.utc),
                    'volume_btc': btc_volume,
                    'price_usd': price_usd,
                    'volume_usd': usd_volume
                }
                hourly_results.append(result)

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching price for timestamp {hour_timestamp}: {e}")
            continue

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    return hourly_results


def main():
    # Date range: Oct 15, 2025 to Dec 28, 2025
    start_date = datetime(2025, 10, 15, tzinfo=timezone.utc)
    end_date = datetime(2025, 12, 29, tzinfo=timezone.utc)  # End of Dec 28

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    # Optional: Use API token if available
    api_token = None  # Add token if required

    print("="*60)
    print("PULLING HOURLY TRANSACTION VOLUME (USD)")
    print("="*60)
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Hours: {(end_ts - start_ts) // 3600}")
    print()

    results = get_hourly_transaction_volume_usd(start_ts, end_ts, api_token)

    if results:
        # Convert to DataFrame
        df = pd.DataFrame(results)
        df.set_index('timestamp', inplace=True)

        # Save to archive
        output_dir = Path("/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/archive/v1.1_2025-12-29")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "transaction_volume_hourly_usd.csv"

        # Save with volume_usd column renamed to transaction_volume for consistency
        df_save = df[['volume_usd']].copy()
        df_save.columns = ['transaction_volume']
        df_save.to_csv(output_file)

        print()
        print("="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Records pulled: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Mean hourly volume: ${df['volume_usd'].mean():,.0f}")
        print(f"Median hourly volume: ${df['volume_usd'].median():,.0f}")
        print(f"\nSaved to: {output_file}")

        # Display sample
        print(f"\nSample data (first 5 hours):")
        print(df[['volume_usd']].head())
    else:
        print("No data retrieved")


if __name__ == "__main__":
    main()
