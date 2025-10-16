#!/usr/bin/env python3
"""
Regression Analysis: Additional Covariates vs DVOL
Tests statistical significance of new covariates for DVOL forecasting
"""

import pandas as pd
import numpy as np
from scipy import stats
import requests
from datetime import datetime, timedelta

def collect_order_flow():
    """Collect recent Deribit order flow data"""
    print("Collecting Deribit order flow data...")
    try:
        base_url = "https://www.deribit.com/api/v2"
        
        # Get recent trades
        url = f"{base_url}/public/get_last_trades_by_instrument"
        params = {
            'instrument_name': 'BTC-PERPETUAL',
            'count': 5000,
            'sorting': 'new_first'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data['result']['trades']:
            trades_df = pd.DataFrame(data['result']['trades'])
            
            # Parse timestamp
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='ms')
            trades_df['hour'] = trades_df['timestamp'].dt.floor('h')
            
            # Separate buy and sell volume
            trades_df['buy_volume'] = trades_df.apply(
                lambda x: x['amount'] if x['direction'] == 'buy' else 0, axis=1
            )
            trades_df['sell_volume'] = trades_df.apply(
                lambda x: x['amount'] if x['direction'] == 'sell' else 0, axis=1
            )
            
            # Aggregate hourly
            hourly_flow = trades_df.groupby('hour').agg({
                'buy_volume': 'sum',
                'sell_volume': 'sum'
            }).reset_index()
            
            hourly_flow['order_flow_imbalance'] = (
                (hourly_flow['buy_volume'] - hourly_flow['sell_volume']) / 
                (hourly_flow['buy_volume'] + hourly_flow['sell_volume'] + 1e-8)
            ) * 100
            
            print(f"  ✓ Collected {len(hourly_flow)} hourly order flow records")
            return hourly_flow
        else:
            print("  ✗ No trades returned")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return pd.DataFrame()

def collect_realized_moments():
    """Collect realized skewness and kurtosis"""
    print("Collecting realized higher moments...")
    try:
        base_url = "https://min-api.cryptocompare.com"
        
        url = f"{base_url}/data/v2/histohour"
        params = {
            'fsym': 'BTC',
            'tsym': 'USD',
            'limit': 2000
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data['Response'] == 'Success':
            df = pd.DataFrame(data['Data']['Data'])
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df['close'] = df['close'].astype(float)
            df['returns'] = df['close'].pct_change()
            
            # Calculate realized moments
            df['realized_skew'] = df['returns'].rolling(24).skew()
            df['realized_kurt'] = df['returns'].rolling(24).kurt()
            
            result = df[['timestamp', 'realized_skew', 'realized_kurt']].copy()
            print(f"  ✓ Calculated {len(result)} hourly moment records")
            return result
        else:
            print(f"  ✗ Error: {data}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return pd.DataFrame()

def collect_active_addresses():
    """Collect daily active addresses and interpolate to hourly"""
    print("Collecting active addresses...")
    try:
        base_url = "https://community-api.coinmetrics.io/v4"
        
        url = f"{base_url}/timeseries/asset-metrics"
        params = {
            'assets': 'btc',
            'metrics': 'AdrActCnt',
            'frequency': '1d',
            'start_time': '2021-04-23',
            'end_time': '2025-10-14'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['time'])
            df['active_addresses_daily'] = df['AdrActCnt'].astype(float)
            
            result = df[['timestamp', 'active_addresses_daily']].copy()
            print(f"  ✓ Collected {len(result)} daily active address records")
            return result
        else:
            print("  ✗ No data returned")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return pd.DataFrame()

def analyze_correlations(dvol_df, covariate_df, covariate_name):
    """Analyze correlation between DVOL and a covariate"""
    
    # Merge on timestamp
    if 'hour' in covariate_df.columns:
        merged = pd.merge(dvol_df, covariate_df, left_on='timestamp', right_on='hour', how='inner')
    elif 'timestamp' in covariate_df.columns:
        merged = pd.merge(dvol_df, covariate_df, on='timestamp', how='inner')
    else:
        return None
    
    if len(merged) < 3:
        return None
    
    # Get feature column
    feature_col = [c for c in covariate_df.columns if c not in ['timestamp', 'hour']][0]
    
    # Calculate correlation
    valid_idx = merged[feature_col].notna()
    if valid_idx.sum() < 3:
        return None
    
    corr, p_val = stats.pearsonr(merged.loc[valid_idx, feature_col], 
                                 merged.loc[valid_idx, 'dvol'])
    
    r_squared = corr ** 2
    
    return {
        'covariate': covariate_name,
        'correlation': corr,
        'p_value': p_val,
        'r_squared': r_squared,
        'n_obs': valid_idx.sum(),
        'significant': 'Yes' if p_val < 0.05 else 'No'
    }

def main():
    print("\n" + "="*90)
    print("COVARIATE SIGNIFICANCE ANALYSIS FOR DVOL FORECASTING")
    print("="*90 + "\n")
    
    # Load current DVOL data
    dvol_df = pd.read_csv('data/processed/bitcoin_lstm_features.csv')
    dvol_df['timestamp'] = pd.to_datetime(dvol_df['timestamp'])
    
    print(f"Current DVOL dataset: {len(dvol_df)} observations\n")
    
    # Collect covariates
    covariates = {}
    
    order_flow_df = collect_order_flow()
    if not order_flow_df.empty:
        covariates['Order Flow Imbalance'] = order_flow_df
    
    moments_df = collect_realized_moments()
    if not moments_df.empty:
        covariates['Realized Skewness'] = moments_df[['timestamp', 'realized_skew']].copy()
        covariates['Realized Kurtosis'] = moments_df[['timestamp', 'realized_kurt']].copy()
    
    addresses_df = collect_active_addresses()
    if not addresses_df.empty:
        # Ensure timezone-naive for merge
        addresses_df['timestamp'] = pd.to_datetime(addresses_df['timestamp']).dt.tz_localize(None)
        
        # Interpolate daily to hourly
        full_range = pd.date_range(dvol_df['timestamp'].min(), dvol_df['timestamp'].max(), freq='h')
        addresses_hourly = pd.DataFrame({'timestamp': full_range})
        addresses_hourly = pd.merge_asof(addresses_hourly, addresses_df.sort_values('timestamp'),
                                        on='timestamp', direction='forward')
        addresses_hourly['active_addresses_hourly'] = addresses_hourly['active_addresses_daily'].fillna(method='ffill')
        covariates['Active Addresses'] = addresses_hourly[['timestamp', 'active_addresses_hourly']].copy()
    
    # Run correlations
    print("\n" + "="*90)
    print("CORRELATION ANALYSIS RESULTS")
    print("="*90 + "\n")
    
    results = []
    for cov_name, cov_df in covariates.items():
        print(f"\nAnalyzing {cov_name}...")
        result = analyze_correlations(dvol_df, cov_df, cov_name)
        if result:
            results.append(result)
            print(f"  Correlation: {result['correlation']:.6f}")
            print(f"  P-Value:     {result['p_value']:.4e}")
            print(f"  R²:          {result['r_squared']:.6f}")
            print(f"  N Obs:       {result['n_obs']}")
            print(f"  Significant: {result['significant']}")
        else:
            print(f"  ✗ Insufficient data for analysis")
    
    # Summary table
    if results:
        results_df = pd.DataFrame(results).sort_values('correlation', key=abs, ascending=False)
        
        print("\n" + "="*90)
        print("SUMMARY TABLE (Sorted by |Correlation|)")
        print("="*90)
        print(results_df.to_string(index=False))
        
        print("\n" + "="*90)
        print("INTERPRETATION")
        print("="*90)
        
        sig_results = results_df[results_df['significant'] == 'Yes']
        if len(sig_results) > 0:
            print(f"\n✅ Found {len(sig_results)} statistically significant covariate(s):")
            for _, row in sig_results.iterrows():
                print(f"   • {row['covariate']:30s}: r = {row['correlation']:7.4f} (p < 0.05)")
        else:
            print("\n⚠️  No statistically significant covariates found at p < 0.05")
        
        print("\n🔍 Interpretation Guidance:")
        print("   • |r| > 0.3: Meaningful economic relationship")
        print("   • |r| > 0.5: Strong predictive relationship")
        print("   • p < 0.05:  Statistically significant at 95% confidence")
        print("   • R²:        % of DVOL variance explained by covariate")
    else:
        print("\n✗ No covariates collected successfully")
    
    print("\n" + "="*90 + "\n")

if __name__ == "__main__":
    main()
