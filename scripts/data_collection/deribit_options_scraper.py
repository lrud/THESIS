#!/usr/bin/env python3
"""
Deribit BTC Options Daily Scraper

Collects daily snapshots of all BTC options including:
- Mark price, bid/ask, mid-price
- Open interest
- Implied volatility (mark IV)
- Underlying BTC price
- Greeks (if available)

Runs daily and appends to CSV file for continuous data collection.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path
import logging
import sys

# Configuration
BASE_URL = "https://www.deribit.com/api/v2/public"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "btc_options_daily_snapshots.csv"
LOG_FILE = OUTPUT_DIR / "scraper.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_btc_options_snapshot():
    """
    Get current snapshot of all BTC options from Deribit.
    Returns list of dicts with option data.
    """
    try:
        params = {
            "currency": "BTC",
            "kind": "option"
        }
        
        response = requests.get(
            f"{BASE_URL}/get_book_summary_by_currency",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        instruments = data.get("result", [])
        
        logger.info(f"Retrieved {len(instruments)} BTC option instruments")
        return instruments
        
    except Exception as e:
        logger.error(f"Error fetching options data: {e}")
        return []


def process_snapshot(instruments, snapshot_time):
    """
    Process raw instrument data into structured format.
    """
    processed_data = []
    
    for inst in instruments:
        try:
            # Parse instrument name to get strike, expiration, option type
            name = inst.get("instrument_name", "")
            parts = name.split("-")
            
            if len(parts) >= 4:
                expiration_str = parts[1]
                strike = parts[2]
                option_type = parts[3]
            else:
                expiration_str = None
                strike = None
                option_type = None
            
            record = {
                # Timestamp
                "timestamp_utc": snapshot_time,
                "timestamp_ms": int(snapshot_time.timestamp() * 1000),
                
                # Instrument info
                "instrument_name": name,
                "expiration_str": expiration_str,
                "strike": strike,
                "option_type": option_type,
                "creation_timestamp": inst.get("creation_timestamp"),
                
                # Pricing data
                "mark_price": inst.get("mark_price"),
                "mid_price": inst.get("mid_price"),
                "bid_price": inst.get("bid_price"),
                "ask_price": inst.get("ask_price"),
                "last_price": inst.get("last"),
                
                # Market data
                "open_interest": inst.get("open_interest"),
                "volume_24h": inst.get("volume"),
                "volume_usd": inst.get("volume_usd"),
                
                # Volatility
                "mark_iv": inst.get("mark_iv"),
                
                # Underlying
                "underlying_price": inst.get("underlying_price"),
                "underlying_index": inst.get("underlying_index"),
                
                # Statistics
                "high_24h": inst.get("high"),
                "low_24h": inst.get("low"),
                "price_change_24h": inst.get("price_change"),
                
                # Other
                "interest_rate": inst.get("interest_rate"),
                "base_currency": inst.get("base_currency"),
                "quote_currency": inst.get("quote_currency"),
            }
            
            processed_data.append(record)
            
        except Exception as e:
            logger.warning(f"Error processing instrument {inst.get('instrument_name')}: {e}")
            continue
    
    return processed_data


def save_snapshot(data, output_file):
    """
    Save snapshot data to CSV, appending to existing file if present.
    """
    df = pd.DataFrame(data)
    
    # Check if file exists
    if output_file.exists():
        # Append to existing file
        df.to_csv(output_file, mode='a', header=False, index=False)
        logger.info(f"Appended {len(df)} records to {output_file}")
    else:
        # Create new file with header
        df.to_csv(output_file, index=False)
        logger.info(f"Created new file with {len(df)} records at {output_file}")


def main():
    """
    Main scraper execution.
    """
    logger.info("=" * 70)
    logger.info("Starting Deribit BTC Options Scraper")
    logger.info("=" * 70)
    
    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get current time
    snapshot_time = datetime.now(timezone.utc)
    logger.info(f"Snapshot time: {snapshot_time}")
    
    # Fetch data
    logger.info("Fetching options data from Deribit...")
    instruments = get_btc_options_snapshot()
    
    if not instruments:
        logger.error("No data retrieved. Exiting.")
        return 1
    
    # Process data
    logger.info("Processing snapshot data...")
    processed = process_snapshot(instruments, snapshot_time)
    
    if not processed:
        logger.error("No data processed. Exiting.")
        return 1
    
    logger.info(f"Processed {len(processed)} instruments")
    
    # Save data
    logger.info("Saving data...")
    save_snapshot(processed, OUTPUT_FILE)
    
    # Summary statistics
    df = pd.DataFrame(processed)
    logger.info("\n" + "=" * 70)
    logger.info("Snapshot Summary:")
    logger.info(f"  Total instruments: {len(df)}")
    logger.info(f"  With open interest > 0: {(df['open_interest'] > 0).sum()}")
    logger.info(f"  With mark price: {df['mark_price'].notna().sum()}")
    logger.info(f"  With mark IV: {df['mark_iv'].notna().sum()}")
    logger.info(f"  Avg underlying price: ${df['underlying_price'].iloc[0]:,.2f}" if len(df) > 0 else "  N/A")
    logger.info("=" * 70)
    
    logger.info("Scraper completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
