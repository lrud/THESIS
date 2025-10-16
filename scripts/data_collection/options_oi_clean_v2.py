"""
Deribit Bitcoin Options Open Interest Data Collection

Aggregates open interest across all BTC option contracts from Deribit API.
OI reported in BTC notional value (1 BTC per contract).

Output: Hourly time series with total/call/put OI and put-call ratios.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fetch_btc_price(deribit):
    """Fetch current BTC/USD price from Deribit."""
    try:
        ticker = deribit.fetch_ticker('BTC/USD')
        price = ticker.get('last')
        if price:
            return float(price)
    except:
        pass
    
    try:
        ticker = deribit.fetch_ticker('BTC-PERPETUAL')
        price = ticker.get('last')
        if price:
            return float(price)
    except:
        pass
    
    return None


def parse_option_symbol(symbol):
    """
    Parse Deribit option symbol.
    Format: BTC/USD:BTC-YYMMDD-STRIKE-TYPE
    Example: BTC/USD:BTC-251226-100000-C
    """
    try:
        # Extract BTC-YYMMDD-STRIKE-TYPE part
        match = re.search(r'BTC-(\d{6})-(\d+)-([CP])', symbol)
        if not match:
            return None
        
        expiry = match.group(1)
        strike = float(match.group(2))
        option_type = 'call' if match.group(3) == 'C' else 'put'
        
        return {
            'expiry': expiry,
            'strike_price': strike,
            'type': option_type,
            'symbol': symbol
        }
    except:
        return None


def fetch_all_options_oi_snapshot(deribit, btc_price):
    """
    Aggregate open interest across all BTC option contracts.
    OI values are summed in BTC notional (1 BTC per contract).
    """
    print(f"[{datetime.now()}] Fetching BTC options OI snapshot...")
    print(f"  Current BTC price: ${btc_price:,.2f}")
    
    try:
        markets = deribit.load_markets()
        btc_options = [
            s for s, m in markets.items() 
            if m['type'] == 'option' and 'BTC-' in s and '/USD:' in s
        ]
        
        print(f"  Found {len(btc_options)} BTC/USD options contracts")
        
        oi_records = []
        call_oi_btc = 0.0
        put_oi_btc = 0.0
        strikes_with_oi = set()
        fetch_errors = 0
        zero_oi_count = 0
        
        for i, symbol in enumerate(btc_options):
            try:
                ticker = deribit.fetch_ticker(symbol)
                raw_oi = ticker.get('info', {}).get('open_interest')
                
                if raw_oi is None:
                    zero_oi_count += 1
                    continue
                
                try:
                    oi_btc = float(raw_oi)
                except (ValueError, TypeError):
                    zero_oi_count += 1
                    continue
                
                if oi_btc <= 0:
                    zero_oi_count += 1
                    continue
                
                parsed = parse_option_symbol(symbol)
                if not parsed:
                    continue
                
                oi_records.append({
                    'symbol': symbol,
                    'expiry': parsed['expiry'],
                    'strike': parsed['strike_price'],
                    'type': parsed['type'],
                    'oi_btc': oi_btc,
                })
                
                if parsed['type'] == 'call':
                    call_oi_btc += oi_btc
                else:
                    put_oi_btc += oi_btc
                
                strikes_with_oi.add(parsed['strike_price'])
                
                if (i + 1) % 200 == 0:
                    print(f"  Processed {i + 1}/{len(btc_options)} options...")
                    
            except Exception as e:
                fetch_errors += 1
                pass
        
        total_oi_btc = call_oi_btc + put_oi_btc
        put_call_ratio = put_oi_btc / max(call_oi_btc, 0.01) if call_oi_btc > 0 else 0
        timestamp = datetime.utcnow()
        
        result = {
            'timestamp': timestamp,
            'timestamp_unix': int(timestamp.timestamp() * 1000),
            'total_oi_btc': total_oi_btc,
            'call_oi_btc': call_oi_btc,
            'put_oi_btc': put_oi_btc,
            'num_strikes': len(strikes_with_oi),
            'put_call_ratio': put_call_ratio,
        }
        
        print(f"\n  Snapshot complete")
        print(f"    Total OI:         {total_oi_btc:>12,.2f} BTC")
        print(f"    Call OI:          {call_oi_btc:>12,.2f} BTC")
        print(f"    Put OI:           {put_oi_btc:>12,.2f} BTC")
        print(f"    Put/Call Ratio:   {put_call_ratio:>12.4f}")
        print(f"    Distinct Strikes: {len(strikes_with_oi):>12,}")
        
        return result
        
    except Exception as e:
        print(f"  Error fetching OI: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None


def load_or_create_oi_file():
    """Load existing OI time series or create new DataFrame."""
    oi_file = 'data/processed/options_oi_clean.csv'
    
    if os.path.exists(oi_file):
        df = pd.read_csv(oi_file, parse_dates=['timestamp'])
        print(f"Loaded existing OI series: {len(df)} records")
        return df
    else:
        print("Creating new OI time series")
        return pd.DataFrame()


def append_snapshot(df, snapshot):
    """Append new OI snapshot to time series, avoiding duplicates."""
    if snapshot is None:
        return df
    
    ts = snapshot['timestamp'].replace(minute=0, second=0, microsecond=0)
    snapshot_copy = snapshot.copy()
    snapshot_copy['timestamp'] = ts
    
    if not df.empty:
        latest_ts = df['timestamp'].max()
        if pd.notna(latest_ts):
            latest_ts = latest_ts.replace(minute=0, second=0, microsecond=0)
            if ts <= latest_ts:
                print(f"  Skipping duplicate timestamp: {ts}")
                return df
    
    new_row = pd.DataFrame([snapshot_copy])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
    
    return df


def save_oi_file(df):
    """Save OI time series to CSV."""
    os.makedirs('data/processed', exist_ok=True)
    oi_file = 'data/processed/options_oi_clean.csv'
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_csv(oi_file, index=False)
    print(f"Saved OI time series: {len(df)} records to {oi_file}")
    
    return df



def backfill_historical_snapshots(deribit, hours_to_collect=24*30):
    """
    Collect current snapshot for time series building.
    Deribit API does not provide historical OI via public endpoints.
    Run hourly via cron for continuous data collection.
    """
    print("\n" + "=" * 80)
    print("HISTORICAL BACKFILL MODE")
    print("=" * 80)
    print(f"Target: {hours_to_collect} hours of data ({hours_to_collect//24} days)")
    print("NOTE: Deribit public API provides current snapshots only.")
    print("=" * 80)
    
    df = load_or_create_oi_file()
    
    if not df.empty:
        hours_collected = len(df)
        latest_ts = df['timestamp'].max()
        print(f"\nExisting data: {hours_collected} snapshots")
        print(f"Latest: {latest_ts}")
        
        hours_needed = hours_to_collect - hours_collected
        if hours_needed <= 0:
            print(f"Already have {hours_collected} hours of data")
            return df
        
        print(f"Need {hours_needed} more hours of data")
    else:
        hours_needed = hours_to_collect
        print(f"\nNo existing data. Collecting current snapshot.")
    
    print("\nCollecting current snapshot...")
    btc_price = fetch_btc_price(deribit)
    if not btc_price:
        print("  Failed to fetch BTC price")
        return df
    
    snapshot = fetch_all_options_oi_snapshot(deribit, btc_price)
    if snapshot:
        df = append_snapshot(df, snapshot)
        df = save_oi_file(df)
    
    print(f"\nCollection complete: {len(df)} total snapshots")
    print("\nRecommendation: Run hourly via cron")
    print("  crontab -e")
    print("  0 * * * * cd /path/to/project && python3 scripts/data_collection/options_oi_clean_v2.py")
    
    return df


def main():
    """Main execution function."""
    print("=" * 80)
    print("DERIBIT BITCOIN OPTIONS OPEN INTEREST COLLECTION")
    print(f"Started: {datetime.now()}")
    print("=" * 80)
    
    import argparse
    parser = argparse.ArgumentParser(description='Collect Bitcoin options OI data from Deribit')
    parser.add_argument('--backfill', action='store_true', 
                        help='Backfill mode: collect current snapshot with instructions')
    parser.add_argument('--hours', type=int, default=24*30,
                        help='Target hours of data (default: 720)')
    args = parser.parse_args()
    
    try:
        print("\nConnecting to Deribit...")
        deribit = ccxt.deribit()
        
        if args.backfill:
            print("\nBACKFILL MODE")
            df = backfill_historical_snapshots(deribit, hours_to_collect=args.hours)
            return
        
        print("\nFetching BTC/USD price...")
        btc_price = fetch_btc_price(deribit)
        if not btc_price:
            print("  Failed to fetch BTC price")
            return
        print(f"  BTC Price: ${btc_price:,.2f}")
        
        print("\nFetching OI snapshot...")
        snapshot = fetch_all_options_oi_snapshot(deribit, btc_price)
        
        if not snapshot:
            print("Failed to fetch OI snapshot")
            return
        
        print("\nLoading historical OI series...")
        df = load_or_create_oi_file()
        
        if not df.empty:
            print(f"\n  Existing data statistics:")
            print(f"    Records:     {len(df):>12,}")
            print(f"    Date range:  {df['timestamp'].min()} to {df['timestamp'].max()}")
            hours_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            print(f"    Time span:   {hours_span:>12,.1f} hours ({hours_span/24:.1f} days)")
            print(f"    Mean OI:     {df['total_oi_btc'].mean():>12,.2f} BTC")
            print(f"    Max OI:      {df['total_oi_btc'].max():>12,.2f} BTC")
            print(f"    Min OI:      {df['total_oi_btc'].min():>12,.2f} BTC")
        
        print("\nAppending new snapshot...")
        df = append_snapshot(df, snapshot)
        
        print("\nSaving OI time series...")
        df = save_oi_file(df)
        
        print("\n" + "=" * 80)
        print("COLLECTION COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
