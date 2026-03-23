#!/usr/bin/env python3

import requests
import pandas as pd
import time
from datetime import datetime, timezone
import os

class ResearchBitcoinCollector:
    def __init__(self, api_token: str):
        # v1 API for volume, v2 API for other metrics
        self.base_url_v1 = "https://api.researchbitcoin.net/v1"
        self.base_url_v2 = "https://api.researchbitcoin.net/v2"
        self.api_token = api_token
        self.headers = {"X-API-Token": api_token}

    def get_metric(self, metric: str, resolution: str = "h1", from_time: str = None, to_time: str = None) -> pd.DataFrame:
        # Map metrics to API endpoints
        # All use v2 API with from_time/to_time parameters
        endpoint_map = {
            "transaction_volume": ("v2", "volume", "spent_tx_outputs_count"),
            "spent_volume_usd": ("v2", "volume", "spent_tx_volume_usd"),
            "price": ("v2", "price", "price"),
            "mvrv": ("v2", "market_value_to_realized_value", "mvrv"),
            "market_cap": ("v2", "marketcap", "market_cap"),
            "realized_cap": ("v2", "realizedcap", "realized_cap"),
            "net_unrealized_profit_loss": ("v2", "net_unrealized_profit_loss", "net_unrealized_profit_loss"),
            "txs_n": ("v2", "network_statistics", "txs_n"),
            "spent_tx_volume_usd": ("v2", "volume", "spent_tx_volume_usd")
        }

        if metric not in endpoint_map:
            raise ValueError(f"Unsupported metric: {metric}")

        api_version, endpoint_group, data_field = endpoint_map[metric]

        # All endpoints now use v2 API with from_time/to_time parameters
        url = f"{self.base_url_v2}/{endpoint_group}/{data_field}"
        params = {"resolution": resolution, "output_format": "csv"}
        if from_time:
            params["from_time"] = from_time
        if to_time:
            params["to_time"] = to_time
        response = requests.get(url, headers=self.headers, params=params)

        response.raise_for_status()

        # API returns CSV format
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

        # v2 format: time column
        df["timestamp"] = pd.to_datetime(df["time"])
        # Get the metric column (second column after 'time')
        metric_col = df.columns[1]
        df = df.rename(columns={metric_col: metric})

        df.set_index("timestamp", inplace=True)
        return df[[metric]]
    
    def collect_dataset(self, start_date: str = "2021-06-01", end_date: str = "2025-10-15", resolution: str = "h1") -> pd.DataFrame:
        metrics = {
            "price": "price",
            "market_cap": "market_cap",
            "realized_cap": "realized_cap",
            "net_unrealized_profit_loss": "net_unrealized_profit_loss",
            "volume_usd": "transaction_volume",  # Updated metric name
            "network_activity": "txs_n"
        }
        
        dfs = []
        for name, metric in metrics.items():
            print(f"Collecting {name} from {start_date} to {end_date}")
            df = self.get_metric(metric, resolution, start_date, end_date)
            df.columns = [name]
            dfs.append(df)
            time.sleep(1)
        
        result = pd.concat(dfs, axis=1)
        result = result.dropna()
        result = result.sort_index()
        
        # Calculate NVRV (Net Value to Realized Value)
        # NVRV = (Market Cap - Realized Cap) / Realized Cap
        if 'market_cap' in result.columns and 'realized_cap' in result.columns:
            result['nvrv'] = (result['market_cap'] - result['realized_cap']) / result['realized_cap']
            print("NVRV calculated from (Market Cap - Realized Cap) / Realized Cap")
            
            # Also calculate NUPL ratio for reference (Net Unrealized P&L / Market Cap)
            if 'net_unrealized_profit_loss' in result.columns:
                # The API returns NUPL as already normalized, so this is the ratio
                result['nupl_ratio'] = result['net_unrealized_profit_loss']
                print("NUPL ratio preserved from API (already normalized)")
        else:
            print("Warning: Could not calculate NVRV - missing required metrics")
        
        return result
    
    def collect_incremental(self, start_date: str = None, end_date: str = None):
        """Collect incremental data from existing dataset end date to specified end date."""
        # Find existing data end date
        nvrv_path = "/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/raw/bitcoin_nvrv_hourly_20251015.csv"
        try:
            df_existing = pd.read_csv(nvrv_path)
            first_col = df_existing.columns[0]
            df_existing[first_col] = pd.to_datetime(df_existing[first_col])
            last_date = df_existing[first_col].max()
            if start_date is None:
                start_date = (last_date + pd.Timedelta(hours=1)).strftime('%Y-%m-%d')
        except:
            start_date = "2025-10-15"

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Collecting incremental data from {start_date} to {end_date}")

        return self.collect_dataset(start_date, end_date, "h1")

    def save_data(self, df: pd.DataFrame, filename: str):
        output_path = f"/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/raw/{filename}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        print(f"Data saved to {output_path}")

def main():
    api_token = os.getenv("RESEARCH_BITCOIN_API_TOKEN")
    if not api_token:
        print("ERROR: API token required. Set RESEARCH_BITCOIN_API_TOKEN environment variable.")
        return
        
    collector = ResearchBitcoinCollector(api_token)
    
    try:
        print("Testing API connection...")
        # Test with a small request first
        test_df = collector.get_metric("price", "d1", "2025-10-10", "2025-10-15")
        print(f"API connection successful. Sample data shape: {test_df.shape}")
        
        print("Collecting historical Bitcoin data with NVRV - HOURLY (2021-2025)...")
        # Full historical range to match DVOL dataset exactly
        start_str = "2021-03-24"  # Match DVOL start date
        end_str = "2025-10-15"    # Match DVOL end date
        
        print(f"Date range: {start_str} to {end_str}")
        df = collector.collect_dataset(start_str, end_str, "h1")
        
        filename = f"bitcoin_nvrv_hourly_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join("data", "raw", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
    except requests.exceptions.HTTPError as e:
        if "expired" in str(e).lower():
            print("ERROR: API token has expired (90-day lifetime)")
            print("To renew:")
            print("1. Visit https://beta.thebitcoinresearcher.net/v2/token")
            print("2. Log in to your account")
            print("3. Generate a new API token")
            print("4. Update RESEARCH_BITCOIN_API_TOKEN environment variable")
        else:
            print(f"API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()