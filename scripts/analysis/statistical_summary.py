#!/usr/bin/env python3
"""
Statistical Analysis of Bitcoin LSTM Features Dataset
Lightweight summary statistics and correlation analysis with DVOL as dependent variable
"""

import pandas as pd
import numpy as np
from scipy import stats

def load_data():
    """Load processed feature dataset"""
    df = pd.read_csv('data/processed/bitcoin_lstm_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def compute_summary_stats(df):
    """Compute descriptive statistics for all features"""
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    print(df.describe().to_string())
    print()

def compute_correlations(df):
    """Compute correlations with DVOL as dependent variable"""
    print("=" * 80)
    print("CORRELATIONS WITH DVOL (Dependent Variable)")
    print("=" * 80)
    
    features = ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d', 
                'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']
    
    correlations = []
    for feature in features:
        corr, p_value = stats.pearsonr(df[feature].dropna(), df['dvol'].loc[df[feature].notna()])
        correlations.append({
            'Feature': feature,
            'Correlation': corr,
            'P-Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
    print(corr_df.to_string(index=False))
    print()

def compute_autocorrelation(df):
    """Compute autocorrelation of DVOL at key lags"""
    print("=" * 80)
    print("DVOL AUTOCORRELATION")
    print("=" * 80)
    
    lags = [1, 24, 168, 720]  # 1 hour, 1 day, 1 week, 1 month
    lag_names = ['1 hour', '1 day (24h)', '1 week (168h)', '1 month (720h)']
    
    for lag, name in zip(lags, lag_names):
        if lag < len(df):
            acf = df['dvol'].autocorr(lag=lag)
            print(f"Lag {name:15s}: {acf:7.4f}")
    print()

def compute_predictive_power(df):
    """Compute R² for lagged DVOL features"""
    print("=" * 80)
    print("DVOL LAG PREDICTIVE POWER (R² vs current DVOL)")
    print("=" * 80)
    
    lag_features = ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d']
    
    for feature in lag_features:
        valid_idx = df[feature].notna()
        r_squared = (stats.pearsonr(df.loc[valid_idx, feature], 
                                    df.loc[valid_idx, 'dvol'])[0])**2
        print(f"{feature:18s}: R² = {r_squared:.4f} ({r_squared*100:.2f}%)")
    print()

def compute_feature_ranges(df):
    """Compute time-series ranges and volatility"""
    print("=" * 80)
    print("FEATURE RANGES & VOLATILITY")
    print("=" * 80)
    
    features = ['dvol', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']
    
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        std_val = df[feature].std()
        cv = (std_val / df[feature].mean()) * 100  # Coefficient of variation
        print(f"{feature:20s}: Min={min_val:12.2f} | Max={max_val:12.2f} | Std={std_val:12.2f} | CV={cv:6.2f}%")
    print()

def main():
    df = load_data()
    
    print(f"\nDataset: {len(df):,} observations | Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    compute_summary_stats(df)
    compute_correlations(df)
    compute_autocorrelation(df)
    compute_predictive_power(df)
    compute_feature_ranges(df)
    
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
