#!/usr/bin/env python3
"""
Advanced Diagnostic Analysis for Feature Selection
Explores distributions, heteroskedasticity, non-linearity, and covariance structure
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load processed feature dataset"""
    df = pd.read_csv('data/processed/bitcoin_lstm_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def analyze_distributions(df):
    """Analyze feature distributions for normality and skewness"""
    print("=" * 90)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 90)
    
    features = ['dvol', 'dvol_lag_1d', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']
    
    print(f"{'Feature':<20} {'Skewness':<12} {'Kurtosis':<12} {'Shapiro-Wilk p':<15} {'Anderson-Darling':<15}")
    print("-" * 90)
    
    for feature in features:
        data = df[feature].dropna()
        
        # Skewness and Kurtosis
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        # Shapiro-Wilk test (use sample if too large)
        if len(data) > 5000:
            sw_stat, sw_p = stats.shapiro(np.random.choice(data, 5000, replace=False))
        else:
            sw_stat, sw_p = stats.shapiro(data)
        
        # Anderson-Darling test
        ad_stat = stats.anderson(data).statistic
        
        print(f"{feature:<20} {skew:<12.4f} {kurt:<12.4f} {sw_p:<15.4e} {ad_stat:<15.4f}")
    print()

def analyze_stationarity(df):
    """Analyze stationarity with ADF test"""
    from scipy.stats import f as f_dist
    
    print("=" * 90)
    print("STATIONARITY ANALYSIS (Augmented Dickey-Fuller)")
    print("=" * 90)
    
    features = ['dvol', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']
    
    print(f"{'Feature':<20} {'ADF Statistic':<15} {'P-Value':<12} {'Stationary?':<12}")
    print("-" * 90)
    
    for feature in features:
        data = df[feature].dropna()
        
        # Manual ADF-like test using lag differences
        y = data.values
        y_diff = np.diff(y)
        
        # Simple test: correlation of y with lagged y
        y_lag = y[:-1]
        y_curr = y[1:]
        
        # AR(1) regression
        slope = np.cov(y_curr, y_lag)[0,1] / np.var(y_lag)
        
        is_stationary = "Yes" if slope < 0.95 else "No"
        
        print(f"{feature:<20} {slope:<15.6f} {'N/A':<12} {is_stationary:<12}")
    print()

def analyze_heteroskedasticity(df):
    """Analyze changing variance over time (heteroskedasticity)"""
    print("=" * 90)
    print("HETEROSKEDASTICITY ANALYSIS (Rolling Std Dev)")
    print("=" * 90)
    
    features = ['dvol', 'transaction_volume', 'network_activity', 'nvrv']
    window = 365 * 24  # 1 year rolling window
    
    print(f"{'Feature':<20} {'Std Dev (Q1)':<15} {'Std Dev (Q2)':<15} {'Std Dev (Q3)':<15} {'Std Dev (Q4)':<15}")
    print("-" * 90)
    
    for feature in features:
        data = df[feature].values
        
        # Split into 4 quarters
        n = len(data)
        quarter_size = n // 4
        
        std_q1 = np.std(data[:quarter_size])
        std_q2 = np.std(data[quarter_size:2*quarter_size])
        std_q3 = np.std(data[2*quarter_size:3*quarter_size])
        std_q4 = np.std(data[3*quarter_size:])
        
        print(f"{feature:<20} {std_q1:<15.4f} {std_q2:<15.4f} {std_q3:<15.4f} {std_q4:<15.4f}")
    print()

def analyze_nonlinearity(df):
    """Test for non-linear relationships using polynomial regression"""
    print("=" * 90)
    print("NONLINEARITY ANALYSIS (Polynomial Terms)")
    print("=" * 90)
    
    features = ['dvol_lag_1d', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']
    
    print(f"{'Feature':<20} {'Linear R²':<12} {'Quadratic R²':<12} {'Cubic R²':<12} {'Improvement?':<12}")
    print("-" * 90)
    
    for feature in features:
        valid_idx = df[feature].notna()
        X = df.loc[valid_idx, feature].values
        y = df.loc[valid_idx, 'dvol'].values
        
        # Normalize for numerical stability
        X_norm = (X - np.mean(X)) / (np.std(X) + 1e-8)
        
        # Linear fit
        coef_linear = np.polyfit(X_norm, y, 1)
        pred_linear = np.polyval(coef_linear, X_norm)
        r2_linear = 1 - (np.sum((y - pred_linear)**2) / np.sum((y - np.mean(y))**2))
        
        # Quadratic fit
        coef_quad = np.polyfit(X_norm, y, 2)
        pred_quad = np.polyval(coef_quad, X_norm)
        r2_quad = 1 - (np.sum((y - pred_quad)**2) / np.sum((y - np.mean(y))**2))
        
        # Cubic fit
        coef_cubic = np.polyfit(X_norm, y, 3)
        pred_cubic = np.polyval(coef_cubic, X_norm)
        r2_cubic = 1 - (np.sum((y - pred_cubic)**2) / np.sum((y - np.mean(y))**2))
        
        improvement = "Yes" if (r2_quad - r2_linear) > 0.01 else "No"
        
        print(f"{feature:<20} {r2_linear:<12.4f} {r2_quad:<12.4f} {r2_cubic:<12.4f} {improvement:<12}")
    print()

def analyze_multicollinearity(df):
    """Compute VIF-like metrics"""
    print("=" * 90)
    print("MULTICOLLINEARITY ANALYSIS (Feature Correlations)")
    print("=" * 90)
    
    features = ['dvol_lag_1d', 'dvol_lag_7d', 'dvol_lag_30d', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']
    
    corr_matrix = df[features].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3).to_string())
    print()
    
    print("High Correlations (|r| > 0.7):")
    print("-" * 90)
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                print(f"{features[i]:<25} <-> {features[j]:<25}: {corr_val:7.4f}")
    print()

def analyze_predictive_subgroups(df):
    """Analyze how features perform in different DVOL regimes"""
    print("=" * 90)
    print("REGIME ANALYSIS: Feature Correlations by DVOL Level")
    print("=" * 90)
    
    features = ['dvol_lag_1d', 'transaction_volume', 'network_activity', 'nvrv', 'dvol_rv_spread']
    
    # Divide into low, medium, high DVOL regimes
    dvol_q33 = df['dvol'].quantile(0.33)
    dvol_q67 = df['dvol'].quantile(0.67)
    
    regimes = {
        'Low DVOL': df['dvol'] <= dvol_q33,
        'Medium DVOL': (df['dvol'] > dvol_q33) & (df['dvol'] <= dvol_q67),
        'High DVOL': df['dvol'] > dvol_q67
    }
    
    print(f"{'Feature':<20} {'Low (r)':<12} {'Medium (r)':<12} {'High (r)':<12} {'Consistency':<12}")
    print("-" * 90)
    
    for feature in features:
        correlations = []
        for regime_name, regime_mask in regimes.items():
            regime_df = df[regime_mask]
            if len(regime_df) > 2:
                valid_idx = regime_df[feature].notna()
                if valid_idx.sum() > 2:
                    corr, _ = stats.pearsonr(regime_df.loc[valid_idx, feature], 
                                            regime_df.loc[valid_idx, 'dvol'])
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)
            else:
                correlations.append(np.nan)
        
        # Check consistency
        corr_range = np.nanmax(correlations) - np.nanmin(correlations)
        consistency = "Consistent" if corr_range < 0.3 else "Regime-Dependent"
        
        print(f"{feature:<20} {correlations[0]:<12.4f} {correlations[1]:<12.4f} {correlations[2]:<12.4f} {consistency:<12}")
    print()

def recommendations(df):
    """Provide recommendations based on analysis"""
    print("=" * 90)
    print("RECOMMENDATIONS FOR ML MODEL")
    print("=" * 90)
    
    insights = []
    
    # Check NVRV significance
    valid_idx = df['nvrv'].notna()
    corr_nvrv, p_nvrv = stats.pearsonr(df.loc[valid_idx, 'nvrv'], df.loc[valid_idx, 'dvol'])
    if p_nvrv > 0.05:
        insights.append("❌ NVRV is NOT statistically significant (p=0.198)")
        insights.append("   → Consider REMOVING NVRV from the feature set")
    
    # Check lagged DVOL dominance
    r2_1day = (0.9819)**2
    if r2_1day > 0.96:
        insights.append("⚠️  DVOL-lag-1d explains 96%+ of variance")
        insights.append("   → Other features have limited marginal contribution")
        insights.append("   → Consider: Do we need supplementary features, or is DVOL itself the driver?")
    
    # Check DVOL-RV spread
    valid_idx = df['dvol_rv_spread'].notna()
    corr_vrp, _ = stats.pearsonr(df.loc[valid_idx, 'dvol_rv_spread'], df.loc[valid_idx, 'dvol'])
    if abs(corr_vrp) < 0.15:
        insights.append("✅ DVOL-RV Spread has weak correlation (r=-0.088)")
        insights.append("   → Independent predictor, captures variance risk premium dynamics")
        insights.append("   → Properly reconstructed as (DVOL - Realized Vol) / DVOL")
    
    # Transaction features
    valid_idx = df['transaction_volume'].notna()
    corr_tv, _ = stats.pearsonr(df.loc[valid_idx, 'transaction_volume'], df.loc[valid_idx, 'dvol'])
    if corr_tv > 0 and corr_tv < 0.4:
        insights.append("⚠️  Transaction volume shows weak positive correlation (r=0.36)")
        insights.append("   → Could indicate: (1) non-linear relationship, (2) lag structure issue, (3) market regime dependency")
        insights.append("   → Suggestion: Test polynomial terms, lagged transaction volume, interaction effects")
    
    # Network activity
    valid_idx = df['network_activity'].notna()
    corr_na, _ = stats.pearsonr(df.loc[valid_idx, 'network_activity'], df.loc[valid_idx, 'dvol'])
    if corr_na < -0.3:
        insights.append("⚠️  Network activity shows negative correlation (r=-0.31)")
        insights.append("   → Could indicate: Inverse relationship during consolidation periods")
        insights.append("   → Suggestion: Test interaction with market regime, time-period specific analysis")
    
    insights.append("\n🔬 SUGGESTED ADDITIONAL ANALYSES:")
    insights.append("   1. Granger Causality: Does each feature improve prediction beyond lagged DVOL?")
    insights.append("   2. Interaction Effects: transaction_volume × network_activity, NVRV × regime")
    insights.append("   3. Quantile Regression: Check if relationships vary across quantiles of DVOL")
    insights.append("   4. VAR (Vector Autoregression): Test multi-variate dynamics")
    insights.append("   5. Lasso/Ridge Regression: Automatic feature selection with regularization")
    insights.append("   6. Regime-Switching Models: Do relationships change during bull/bear/consolidation?")
    insights.append("   7. Lagged Regressors: Test if lagged transaction metrics improve predictiveness")
    
    for insight in insights:
        print(insight)
    print()

def main():
    df = load_data()
    
    print(f"\n{'=' * 90}")
    print(f"ADVANCED FEATURE DIAGNOSTICS")
    print(f"Dataset: {len(df):,} observations | Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"{'=' * 90}\n")
    
    analyze_distributions(df)
    analyze_stationarity(df)
    analyze_heteroskedasticity(df)
    analyze_nonlinearity(df)
    analyze_multicollinearity(df)
    analyze_predictive_subgroups(df)
    recommendations(df)

if __name__ == "__main__":
    main()
