#!/usr/bin/env python3

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

class DVOLCollector:
    def __init__(self):
        self.base_url = "https://www.deribit.com/api/v2"
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.request_delay = 0.2
    
    def collect_dvol_data(self, start_date="2021-03-24", end_date="2025-10-15"):
        print(f"Collecting DVOL data: {start_date} to {end_date}")
        
        all_data = []
        current_start = start_date
        
        while current_start <= end_date:
            chunk_end = self._add_months(current_start, 1)
            if chunk_end > end_date:
                chunk_end = end_date
            
            chunk_data = self._fetch_dvol_chunk(current_start, chunk_end)
            
            if chunk_data is not None and not chunk_data.empty:
                all_data.append(chunk_data)
                print(f"Retrieved {len(chunk_data)} records for {current_start}")
            
            time.sleep(self.request_delay)
            current_start = self._add_days(chunk_end, 1)
        
        if not all_data:
            return None
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    def _fetch_dvol_chunk(self, start_date, end_date):
        start_timestamp = self._date_to_timestamp(start_date)
        end_timestamp = self._date_to_timestamp(end_date)
        
        endpoint = f"{self.base_url}/public/get_volatility_index_data"
        params = {
            'currency': 'BTC',
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'resolution': '3600'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'error' in data:
                print(f"API Error: {data['error']}")
                return None
            
            candles = data.get('result', {}).get('data', [])
            if not candles:
                return None
            
            return self._process_dvol_data(candles)
            
        except Exception as e:
            print(f"Error fetching {start_date}-{end_date}: {e}")
            return None
    
    def _process_dvol_data(self, candles):
        processed_data = []
        
        for candle in candles:
            if len(candle) >= 5:
                timestamp_ms, open_val, high_val, low_val, close_val = candle[:5]
                dt = pd.to_datetime(timestamp_ms, unit='ms', utc=True)
                
                processed_data.append({
                    'timestamp': timestamp_ms,
                    'datetime': dt,
                    'date': dt.strftime('%Y-%m-%d'),
                    'hour': dt.hour,
                    'dvol_open': float(open_val),
                    'dvol_high': float(high_val),
                    'dvol_low': float(low_val), 
                    'dvol_close': float(close_val)
                })
        
        return pd.DataFrame(processed_data).sort_values('timestamp').reset_index(drop=True)
    
    def save_data(self, df, filename="bitcoin_dvol_hourly_complete.csv"):
        output_path = f"data/raw/{filename}"
        
        df_save = df.copy()
        df_save['datetime_utc'] = df_save['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        cols = ['timestamp', 'datetime_utc', 'date', 'hour', 
                'dvol_open', 'dvol_high', 'dvol_low', 'dvol_close']
        df_save = df_save[cols]
        
        df_save.to_csv(output_path, index=False, float_format='%.6f')
        print(f"Data saved: {output_path}")
        return output_path
    
    def _date_to_timestamp(self, date_str):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _add_months(self, date_str, months):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        new_month = dt.month + months
        new_year = dt.year + (new_month - 1) // 12
        new_month = (new_month - 1) % 12 + 1
        
        try:
            new_dt = dt.replace(year=new_year, month=new_month)
        except ValueError:
            import calendar
            last_day = calendar.monthrange(new_year, new_month)[1]
            new_dt = dt.replace(year=new_year, month=new_month, day=min(dt.day, last_day))
        
        return new_dt.strftime("%Y-%m-%d")

    def _add_days(self, date_str, days):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        new_dt = dt + timedelta(days=days)
        return new_dt.strftime("%Y-%m-%d")


def main():
    print("Deribit DVOL Data Collection")
    
    collector = DVOLCollector()
    df = collector.collect_dvol_data()
    
    if df is not None:
        collector.save_data(df)
        print("DVOL data collection complete")
    else:
        print("Failed to collect DVOL data")


if __name__ == "__main__":
    main()