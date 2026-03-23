#!/usr/bin/env python3
"""
Bitcoin Options Data Collector for Volatility Forecasting Research
Extends existing dataset from 2021-04-23 to present
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import deribit_data as dm

class BitcoinVolatilityCollector:
    def __init__(self, currency="BTC"):
        """Initialize the data collector"""
        self.currency = currency
        self.data_collector = dm.Options(currency)
        self.base_path = "/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/deribit"
        self.existing_data_path = "/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/processed/bitcoin_lstm_features.csv"
        os.makedirs(self.base_path, exist_ok=True)

        # Load existing data to understand our dataset
        self.existing_data = self.load_existing_data()

    def load_existing_data(self):
        """Load existing Bitcoin LSTM data to understand date range"""
        try:
            df = pd.read_csv(self.existing_data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"📊 Loaded existing data: {len(df)} records")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
        except Exception as e:
            print(f"⚠️  Could not load existing data: {e}")
            return None

    def get_data_gap_info(self):
        """Calculate the gap between existing data and current date"""
        if self.existing_data is None:
            return None

        latest_date = self.existing_data['timestamp'].max()
        current_date = datetime.now()
        gap_days = (current_date - latest_date).days

        print(f"📈 Latest data: {latest_date}")
        print(f"📅 Current date: {current_date}")
        print(f"⏰ Data gap: {gap_days} days")

        return {
            'latest_date': latest_date,
            'current_date': current_date,
            'gap_days': gap_days
        }

    def collect_incremental_volatility(self, start_timestamp=None, end_timestamp=None, save=True):
        """Collect incremental volatility data using get_volatility_index_data API with pagination."""
        print("📊 Collecting incremental Deribit volatility data (full historical access)...")

        if start_timestamp is None:
            # Load existing data to find end date
            dvol_path = "/home/lrud1314/PROJECTS_WORKING/THESIS 2025/data/raw/bitcoin_dvol_hourly_complete.csv"
            try:
                df_existing = pd.read_csv(dvol_path)
                # First column is timestamp in milliseconds (e.g., 1616544000000)
                first_col = df_existing.columns[0]
                # Convert millisecond timestamps to datetime
                last_timestamp_ms = df_existing[first_col].iloc[-1]
                last_date = pd.to_datetime(last_timestamp_ms, unit='ms')
                start_timestamp = int((last_date + pd.Timedelta(hours=1)).timestamp() * 1000)
                print(f"  Existing data ends: {last_date}")
            except:
                start_timestamp = int((datetime.now() - pd.Timedelta(days=90)).timestamp() * 1000)

        if end_timestamp is None:
            end_timestamp = int(datetime.now().timestamp() * 1000)

        print(f"  Pulling from {datetime.fromtimestamp(start_timestamp/1000).strftime('%Y-%m-%d %H:%M')}")
        print(f"            to {datetime.fromtimestamp(end_timestamp/1000).strftime('%Y-%m-%d %H:%M')}")

        import requests
        url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
        all_data = []
        current_start = start_timestamp
        continuation = None
        page = 0

        while current_start < end_timestamp:
            page += 1
            print(f"  Page {page}...", end=" ")

            params = {
                "currency": "BTC",
                "start_timestamp": current_start,
                "end_timestamp": end_timestamp,
                "resolution": "3600"
            }
            if continuation:
                params["continuation"] = continuation

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    print(f"Error: {data['error']}")
                    break

                if "result" in data and "data" in data["result"]:
                    page_data = data["result"]["data"]
                    all_data.extend(page_data)
                    print(f"{len(page_data)} records")

                    continuation = data["result"].get("continuation")
                    if continuation and len(page_data) == 1000:
                        current_start = continuation
                    else:
                        break
                else:
                    print("No data")
                    break

            except Exception as e:
                print(f"Error: {e}")
                break

        if not all_data:
            print("❌ No data retrieved")
            return None

        # Convert to DataFrame and deduplicate
        df = pd.DataFrame(all_data, columns=['timestamp', 'dvol_open', 'dvol_high', 'dvol_low', 'dvol_close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"  ✅ Retrieved {len(df)} records")
        print(f"  Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.base_path}/deribit_dvol_incremental_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 Saved: {filename}")

        return df

    def collect_historical_volatility(self, save=True):
        """Collect historical volatility data from Deribit (covers last 15 days)"""
        print("📊 Collecting Deribit historical volatility data (last 15 days)...")
        try:
            df = self.data_collector.get_hist_vol(save_csv=False)

            # Convert to timezone-aware if needed
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            # Convert to naive datetime for consistency with existing data
            df.index = df.index.tz_localize(None)

            # Rename column to match existing data format
            df.rename(columns={f'{self.currency.lower()}_hist_vol': 'dvol_deribit'}, inplace=True)

            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.base_path}/deribit_hist_vol_{timestamp}.csv"
                df.to_csv(filename)
                print(f"💾 Saved: {filename}")

                # Also save to processed location for easy access
                processed_file = f"{self.base_path}/../processed/deribit_hist_vol_latest.csv"
                df.to_csv(processed_file)
                print(f"💾 Latest data saved to: {processed_file}")

            return df
        except Exception as e:
            print(f"❌ Error collecting historical volatility: {e}")
            return None

    def collect_options_snapshot(self, max_workers=10, save=True):
        """Collect current options chain snapshot"""
        print(f"📊 Collecting options chain data ({max_workers} workers)...")
        try:
            # Get options list first
            options_list = self.data_collector.get_options_list()
            print(f"📋 Found {len(options_list)} options")

            # Filter for reasonable expiry periods (within 365 days)
            current_timestamp = datetime.now().timestamp() * 1000
            max_expiry_timestamp = current_timestamp + (365 * 24 * 60 * 60 * 1000)  # 1 year ahead

            options_list = options_list[options_list['expiration_timestamp'] <= max_expiry_timestamp]
            print(f"📋 Filtered to {len(options_list)} options within 1 year")

            # Modify URLs based on filtered options
            urls = []
            for instrument_name in options_list['instrument_name']:
                url = self.data_collector.url + 'get_order_book?instrument_name=' + instrument_name
                urls.append(url)

            print(f"🔄 Processing {len(urls)} options...")

            # Use ThreadPoolExecutor with our specified max_workers
            from concurrent.futures import ThreadPoolExecutor
            raw_data = []

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for i, asset in enumerate(pool.map(self.data_collector.request_get, urls)):
                    raw_data.append(asset)
                    if (i + 1) % 50 == 0:
                        print(f"  Progress: {i + 1}/{len(urls)} options processed")

            # Process the collected data
            df = pd.DataFrame(raw_data)

            # Add option_type (call/put) and extract strike/expiry
            df['option_type'] = [df.instrument_name[i][-1] for i in range(len(df))]
            df['strike'] = df['instrument_name'].str.extract(r'-(\d+)[CP]$').astype(float)

            # Extract expiry date from instrument name
            df['expiry_date'] = df['instrument_name'].str.extract(r'^[A-Z]+-(\d+[A-Z]+\d+)-')[0]

            # Convert expiry timestamp
            df['expiry_timestamp'] = pd.to_datetime(df['expiration_timestamp'], unit='ms')
            df['days_to_expiry'] = (df['expiry_timestamp'] - datetime.now()).dt.days

            # Calculate moneyness
            if 'underlying_price' in df.columns:
                df['moneyness'] = df['strike'] / df['underlying_price']

            # Add collection timestamp
            df['collection_timestamp'] = datetime.now()

            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.base_path}/options_chain_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"💾 Saved: {filename}")

                # Also save a latest version
                latest_file = f"{self.base_path}/../processed/options_chain_latest.csv"
                df.to_csv(latest_file, index=False)
                print(f"💾 Latest options saved to: {latest_file}")

            return df
        except Exception as e:
            print(f"❌ Error collecting options data: {e}")
            return None

    def extract_volatility_surface(self, options_df):
        """Extract volatility surface from options data"""
        print("📈 Extracting volatility surface...")

        if options_df is None or options_df.empty:
            print("❌ No options data provided")
            return None

        try:
            # Filter for options with mark_iv and valid data
            df_with_iv = options_df[
                (options_df['mark_iv'].notna()) &
                (options_df['mark_iv'] > 0) &
                (options_df['strike'] > 0) &
                (options_df['days_to_expiry'] > 0)
            ].copy()

            if df_with_iv.empty:
                print("❌ No valid options with implied volatility found")
                return None

            # Create volatility surface data
            vol_surface = df_with_iv[[
                'strike', 'days_to_expiry', 'mark_iv', 'option_type',
                'underlying_price', 'last_price', 'open_interest', 'moneyness'
            ]].copy()

            # Add term structure data
            vol_surface_sorted = vol_surface.sort_values(['days_to_expiry', 'strike'])

            print(f"✅ Volatility surface extracted: {len(vol_surface)} data points")
            print(f"📊 Calls: {len(vol_surface[vol_surface['option_type'] == 'C'])}")
            print(f"📊 Puts: {len(vol_surface[vol_surface['option_type'] == 'P'])}")

            return vol_surface_sorted

        except Exception as e:
            print(f"❌ Error extracting volatility surface: {e}")
            return None

    def create_data_summary(self, hist_vol_df, options_df, vol_surface_df):
        """Create a comprehensive summary of the collected data"""
        print("📋 Creating data summary...")

        # Get gap info
        gap_info = self.get_data_gap_info()

        summary = {
            'collection_time': datetime.now().isoformat(),
            'currency': self.currency,
            'existing_dataset': {
                'records': len(self.existing_data) if self.existing_data is not None else 0,
                'date_range': {
                    'start': self.existing_data['timestamp'].min().isoformat() if self.existing_data is not None else None,
                    'end': self.existing_data['timestamp'].max().isoformat() if self.existing_data is not None else None
                }
            },
            'data_gap': {
                'latest_date': gap_info['latest_date'].isoformat() if gap_info else None,
                'current_date': gap_info['current_date'].isoformat() if gap_info else None,
                'gap_days': gap_info['gap_days'] if gap_info else None
            },
            'deribit_historical_volatility': {
                'records': len(hist_vol_df) if hist_vol_df is not None else 0,
                'date_range': {
                    'start': hist_vol_df.index.min().isoformat() if hist_vol_df is not None else None,
                    'end': hist_vol_df.index.max().isoformat() if hist_vol_df is not None else None
                },
                'latest_volatility': float(hist_vol_df.iloc[-1, 0]) if hist_vol_df is not None else None
            },
            'options_chain': {
                'total_options': len(options_df) if options_df is not None else 0,
                'calls': len(options_df[options_df['option_type'] == 'C']) if options_df is not None else 0,
                'puts': len(options_df[options_df['option_type'] == 'P']) if options_df is not None else 0,
                'options_with_iv': len(options_df[options_df['mark_iv'].notna()]) if options_df is not None else 0,
                'expiry_range': {
                    'min_dte': int(options_df['days_to_expiry'].min()) if options_df is not None else None,
                    'max_dte': int(options_df['days_to_expiry'].max()) if options_df is not None else None
                }
            },
            'volatility_surface': {
                'data_points': len(vol_surface_df) if vol_surface_df is not None else 0,
                'min_strike': float(vol_surface_df['strike'].min()) if vol_surface_df is not None else None,
                'max_strike': float(vol_surface_df['strike'].max()) if vol_surface_df is not None else None,
                'min_dte': int(vol_surface_df['days_to_expiry'].min()) if vol_surface_df is not None else None,
                'max_dte': int(vol_surface_df['days_to_expiry'].max()) if vol_surface_df is not None else None,
                'min_iv': float(vol_surface_df['mark_iv'].min()) if vol_surface_df is not None else None,
                'max_iv': float(vol_surface_df['mark_iv'].max()) if vol_surface_df is not None else None
            }
        }

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"{self.base_path}/data_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"💾 Summary saved: {summary_file}")
        return summary

    def main_collection_routine(self):
        """Execute the complete data collection routine"""
        print("🚀 Bitcoin Options Data Collection for Volatility Research")
        print("=" * 60)

        # Show existing data info
        self.get_data_gap_info()

        # Collect Deribit historical volatility
        hist_vol_df = self.collect_historical_volatility()

        # Collect current options chain
        options_df = self.collect_options_snapshot(max_workers=15)

        # Extract volatility surface
        vol_surface_df = self.extract_volatility_surface(options_df)

        # Create comprehensive summary
        summary = self.create_data_summary(hist_vol_df, options_df, vol_surface_df)

        # Print final summary
        print("\n📊 COLLECTION SUMMARY:")
        print(f"Existing Dataset: {summary['existing_dataset']['records']} records")
        print(f"Existing Range: {summary['existing_dataset']['date_range']['start']} to {summary['existing_dataset']['date_range']['end']}")
        print(f"Data Gap: {summary['data_gap']['gap_days']} days")
        print(f"Deribit Historical Volatility: {summary['deribit_historical_volatility']['records']} records")
        print(f"Latest Deribit Volatility: {summary['deribit_historical_volatility']['latest_volatility']:.2f}%")
        print(f"Total Options: {summary['options_chain']['total_options']}")
        print(f"Options with IV: {summary['options_chain']['options_with_iv']}")
        print(f"Volatility Surface Points: {summary['volatility_surface']['data_points']}")

        if vol_surface_df is not None:
            print(f"Strike Range: {summary['volatility_surface']['min_strike']:,.0f} - {summary['volatility_surface']['max_strike']:,.0f}")
            print(f"DTE Range: {summary['volatility_surface']['min_dte']} - {summary['volatility_surface']['max_dte']} days")
            print(f"IV Range: {summary['volatility_surface']['min_iv']:.1f}% - {summary['volatility_surface']['max_iv']:.1f}%")

        print("\n✅ Data collection complete!")
        print(f"📁 Data saved to: {self.base_path}")
        print(f"📁 Processed data: {self.base_path}/../processed/")

        return hist_vol_df, options_df, vol_surface_df, summary

def main():
    """Main execution function"""
    collector = BitcoinVolatilityCollector("BTC")
    return collector.main_collection_routine()

if __name__ == "__main__":
    main()