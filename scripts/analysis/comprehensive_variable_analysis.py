#!/usr/bin/env python3
"""
Comprehensive Variable Analysis for DVOL Forecasting Model

Analyzes:
1. Summary statistics for all variables
2. Linearity/nonlinearity tests
3. Correlation structure
4. Distribution characteristics
5. Stationarity tests
6. Recommendations for regression specification

Date: October 16, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load the complete features dataset."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load the complete features dataset
    print("\n[1] Loading Bitcoin LSTM features data...")
    df = pd.read_csv('data/processed/bitcoin_lstm_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.set_index('timestamp')
    print(f"    ✅ Dataset: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    print(f"    Columns: {list(df.columns)}")
    
    # Rename columns to match our expected naming
    df = df.rename(columns={
        'transaction_volume': 'transaction_volume_usd',
        'network_activity': 'active_addresses',
        'dvol_lag_1d': 'dvol_lag1',
        'dvol_lag_7d': 'dvol_lag7',
        'dvol_lag_30d': 'dvol_lag30'
    })
    
    # Drop rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    print(f"    ✅ After dropping NaN: {len(df)} rows (dropped {initial_rows - len(df)})")
    
    return df


def calculate_features(df):
    """Calculate additional derived features for analysis."""
    print("\n" + "=" * 80)
    print("CALCULATING ADDITIONAL FEATURES")
    print("=" * 80)
    
    # Calculate realized volatility if not present
    if 'realized_volatility_30d' not in df.columns:
        print("\n[1] Calculating 30-day realized volatility from DVOL-RV spread...")
        df['realized_volatility_30d'] = df['dvol'] - df['dvol_rv_spread']
    
    # Log transformations (for potential nonlinearity)
    print("[2] Log transformations...")
    df['log_transaction_volume'] = np.log(df['transaction_volume_usd'] + 1)
    df['log_active_addresses'] = np.log(df['active_addresses'] + 1)
    
    # Percentage changes
    print("[3] Percentage changes...")
    df['dvol_pct_change'] = df['dvol'].pct_change(24)
    df['nvrv_pct_change'] = df['nvrv'].pct_change(24)
    
    # Drop NaN from new features
    initial_rows = len(df)
    df = df.dropna()
    print(f"\n    ✅ Final dataset: {len(df)} rows (dropped {initial_rows - len(df)} from new features)")
    
    return df


def summary_statistics(df):
    """Generate comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    variables = {
        'Dependent Variable': ['dvol'],
        'Lagged DVOL': ['dvol_lag1', 'dvol_lag7', 'dvol_lag30'],
        'On-chain Metrics': ['transaction_volume_usd', 'active_addresses', 'nvrv'],
        'Derived Features': ['dvol_rv_spread', 'realized_volatility_30d']
    }
    
    results = []
    
    for category, vars in variables.items():
        print(f"\n{category}:")
        print("-" * 80)
        
        for var in vars:
            if var in df.columns:
                data = df[var]
                
                stats_dict = {
                    'Variable': var,
                    'Category': category,
                    'N': len(data),
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Min': data.min(),
                    'Q25': data.quantile(0.25),
                    'Median': data.median(),
                    'Q75': data.quantile(0.75),
                    'Max': data.max(),
                    'Skewness': data.skew(),
                    'Kurtosis': data.kurtosis(),
                    'CV': data.std() / data.mean() if data.mean() != 0 else np.nan
                }
                
                results.append(stats_dict)
                
                print(f"  {var:30} N={len(data):6,} Mean={data.mean():12.4f} Std={data.std():12.4f} Skew={data.skew():7.3f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/summary_statistics.csv', index=False)
    print(f"\n✅ Summary statistics saved to results/summary_statistics.csv")
    
    return results_df


def normality_tests(df):
    """Test for normality of distributions."""
    print("\n" + "=" * 80)
    print("NORMALITY TESTS")
    print("=" * 80)
    
    variables = ['dvol', 'dvol_lag1', 'transaction_volume_usd', 'active_addresses', 
                 'nvrv', 'dvol_rv_spread', 'log_transaction_volume', 'log_active_addresses']
    
    results = []
    
    for var in variables:
        if var in df.columns:
            data = df[var].dropna()
            
            # Jarque-Bera test
            jb_stat, jb_pval = jarque_bera(data)
            
            # Shapiro-Wilk test (use sample if too large)
            if len(data) > 5000:
                sample = data.sample(5000, random_state=42)
                sw_stat, sw_pval = shapiro(sample)
            else:
                sw_stat, sw_pval = shapiro(data)
            
            # D'Agostino-Pearson test
            k2_stat, k2_pval = normaltest(data)
            
            result = {
                'Variable': var,
                'JB_Statistic': jb_stat,
                'JB_P_Value': jb_pval,
                'JB_Normal': 'Yes' if jb_pval > 0.05 else 'No',
                'SW_Statistic': sw_stat,
                'SW_P_Value': sw_pval,
                'SW_Normal': 'Yes' if sw_pval > 0.05 else 'No',
                'DP_Statistic': k2_stat,
                'DP_P_Value': k2_pval,
                'DP_Normal': 'Yes' if k2_pval > 0.05 else 'No'
            }
            
            results.append(result)
            
            print(f"{var:30} JB p={jb_pval:.4f} SW p={sw_pval:.4f} Normal: {result['JB_Normal']}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/normality_tests.csv', index=False)
    print(f"\n✅ Normality tests saved to results/normality_tests.csv")
    
    return results_df


def stationarity_tests(df):
    """Test for stationarity (important for time series)."""
    print("\n" + "=" * 80)
    print("STATIONARITY TESTS (ADF & KPSS)")
    print("=" * 80)
    
    variables = ['dvol', 'transaction_volume_usd', 'active_addresses', 'nvrv', 
                 'dvol_rv_spread', 'log_transaction_volume', 'log_active_addresses']
    
    results = []
    
    for var in variables:
        if var in df.columns:
            data = df[var].dropna()
            
            # Augmented Dickey-Fuller test (H0: unit root / non-stationary)
            adf_result = adfuller(data, maxlag=24, regression='ct')
            adf_stat = adf_result[0]
            adf_pval = adf_result[1]
            adf_stationary = 'Yes' if adf_pval < 0.05 else 'No'
            
            # KPSS test (H0: stationary)
            kpss_result = kpss(data, regression='ct', nlags='auto')
            kpss_stat = kpss_result[0]
            kpss_pval = kpss_result[1]
            kpss_stationary = 'Yes' if kpss_pval > 0.05 else 'No'
            
            result = {
                'Variable': var,
                'ADF_Statistic': adf_stat,
                'ADF_P_Value': adf_pval,
                'ADF_Stationary': adf_stationary,
                'KPSS_Statistic': kpss_stat,
                'KPSS_P_Value': kpss_pval,
                'KPSS_Stationary': kpss_stationary,
                'Consensus': 'Stationary' if (adf_stationary == 'Yes' and kpss_stationary == 'Yes') else 'Non-Stationary'
            }
            
            results.append(result)
            
            print(f"{var:30} ADF p={adf_pval:.4f} KPSS p={kpss_pval:.4f} → {result['Consensus']}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/stationarity_tests.csv', index=False)
    print(f"\n✅ Stationarity tests saved to results/stationarity_tests.csv")
    
    return results_df


def linearity_tests(df):
    """Test linearity of relationships with DVOL."""
    print("\n" + "=" * 80)
    print("LINEARITY TESTS (Ramsey RESET & Visual Analysis)")
    print("=" * 80)
    
    predictors = ['dvol_lag1', 'transaction_volume_usd', 'active_addresses', 
                  'nvrv', 'dvol_rv_spread', 'log_transaction_volume', 'log_active_addresses']
    
    results = []
    
    for pred in predictors:
        if pred in df.columns:
            # Prepare data
            y = df['dvol'].values
            X = df[pred].values
            X = sm.add_constant(X)
            
            # Linear regression
            model = sm.OLS(y, X).fit()
            
            # Ramsey RESET test (tests for nonlinearity)
            # Add powers of fitted values
            y_fitted = model.fittedvalues
            X_reset = np.column_stack([X, y_fitted**2, y_fitted**3])
            model_reset = sm.OLS(y, X_reset).fit()
            
            # F-test for added powers
            from scipy.stats import f
            ssr_restricted = model.ssr
            ssr_unrestricted = model_reset.ssr
            df_num = 2
            df_denom = len(y) - model_reset.df_model - 1
            
            f_stat = ((ssr_restricted - ssr_unrestricted) / df_num) / (ssr_unrestricted / df_denom)
            f_pval = 1 - f.cdf(f_stat, df_num, df_denom)
            
            result = {
                'Predictor': pred,
                'Linear_R2': model.rsquared,
                'Linear_Coef': model.params[1],
                'Linear_PValue': model.pvalues[1],
                'RESET_F_Stat': f_stat,
                'RESET_P_Value': f_pval,
                'Linearity': 'Linear' if f_pval > 0.05 else 'Non-Linear',
                'Recommendation': 'Use as-is' if f_pval > 0.05 else 'Consider transformation/polynomial'
            }
            
            results.append(result)
            
            print(f"{pred:30} R²={model.rsquared:.4f} RESET p={f_pval:.4f} → {result['Linearity']}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/linearity_tests.csv', index=False)
    print(f"\n✅ Linearity tests saved to results/linearity_tests.csv")
    
    return results_df


def correlation_analysis(df):
    """Analyze correlation structure."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    variables = ['dvol', 'dvol_lag1', 'dvol_lag7', 'dvol_lag30', 
                 'transaction_volume_usd', 'active_addresses', 'nvrv', 'dvol_rv_spread']
    
    corr_matrix = df[variables].corr()
    
    print("\nCorrelation with DVOL:")
    print("-" * 80)
    dvol_corr = corr_matrix['dvol'].sort_values(ascending=False)
    for var, corr in dvol_corr.items():
        if var != 'dvol':
            print(f"  {var:30} {corr:7.4f}")
    
    # Save correlation matrix
    corr_matrix.to_csv('results/correlation_matrix.csv')
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Correlation matrix saved to results/correlation_matrix.csv")
    print(f"✅ Correlation heatmap saved to results/correlation_heatmap.png")
    
    return corr_matrix


def create_scatter_plots(df):
    """Create scatter plots to visualize relationships."""
    print("\n" + "=" * 80)
    print("CREATING SCATTER PLOTS")
    print("=" * 80)
    
    predictors = ['dvol_lag1', 'transaction_volume_usd', 'active_addresses', 
                  'nvrv', 'dvol_rv_spread']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, pred in enumerate(predictors):
        if pred in df.columns:
            ax = axes[idx]
            
            # Scatter plot with LOWESS smoothing
            x = df[pred].values
            y = df['dvol'].values
            
            # Sample if too many points
            if len(x) > 5000:
                sample_idx = np.random.choice(len(x), 5000, replace=False)
                x_sample = x[sample_idx]
                y_sample = y[sample_idx]
            else:
                x_sample = x
                y_sample = y
            
            ax.scatter(x_sample, y_sample, alpha=0.3, s=10)
            
            # LOWESS smoothing
            lowess_result = lowess(y, x, frac=0.1)
            ax.plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2, label='LOWESS')
            
            # Linear fit
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'g--', linewidth=2, label='Linear Fit')
            
            ax.set_xlabel(pred, fontsize=10)
            ax.set_ylabel('DVOL', fontsize=10)
            ax.set_title(f'DVOL vs {pred}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('results/scatter_plots_linearity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Scatter plots saved to results/scatter_plots_linearity.png")


def multicollinearity_analysis(df):
    """Check for multicollinearity using VIF."""
    print("\n" + "=" * 80)
    print("MULTICOLLINEARITY ANALYSIS (VIF)")
    print("=" * 80)
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    predictors = ['dvol_lag1', 'transaction_volume_usd', 'active_addresses', 
                  'nvrv', 'dvol_rv_spread']
    
    # Prepare data
    X = df[predictors].dropna()
    
    # Calculate VIF
    vif_data = []
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({'Variable': col, 'VIF': vif, 'Multicollinearity': 'High' if vif > 10 else ('Moderate' if vif > 5 else 'Low')})
        print(f"{col:30} VIF={vif:7.2f} → {vif_data[-1]['Multicollinearity']}")
    
    vif_df = pd.DataFrame(vif_data)
    vif_df.to_csv('results/vif_analysis.csv', index=False)
    print(f"\n✅ VIF analysis saved to results/vif_analysis.csv")
    
    return vif_df


def generate_recommendations(linearity_df, stationarity_df, normality_df):
    """Generate modeling recommendations based on tests."""
    print("\n" + "=" * 80)
    print("MODELING RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    print("\n📊 REGRESSION SPECIFICATION RECOMMENDATIONS:\n")
    
    # Linearity recommendations
    nonlinear_vars = linearity_df[linearity_df['Linearity'] == 'Non-Linear']['Predictor'].tolist()
    if nonlinear_vars:
        print("⚠️  NON-LINEAR RELATIONSHIPS DETECTED:")
        for var in nonlinear_vars:
            print(f"   • {var}: Consider polynomial terms, splines, or log transformation")
            recommendations.append(f"Add polynomial/nonlinear terms for {var}")
    else:
        print("✅ All relationships appear linear")
    
    # Stationarity recommendations
    print("\n📈 TIME SERIES PROPERTIES:")
    non_stationary = stationarity_df[stationarity_df['Consensus'] == 'Non-Stationary']['Variable'].tolist()
    if non_stationary:
        print("⚠️  NON-STATIONARY VARIABLES:")
        for var in non_stationary:
            print(f"   • {var}: Consider differencing or using in percentage change form")
            recommendations.append(f"Difference or use percent change for {var}")
    else:
        print("✅ All variables are stationary")
    
    # Normality recommendations
    print("\n📐 DISTRIBUTIONAL PROPERTIES:")
    non_normal = normality_df[normality_df['JB_Normal'] == 'No']['Variable'].tolist()
    if non_normal:
        print("⚠️  NON-NORMAL DISTRIBUTIONS:")
        for var in non_normal:
            print(f"   • {var}: Consider log transformation or robust regression methods")
            recommendations.append(f"Log transform or use robust methods for {var}")
        print("\n   💡 Recommendation: Use robust standard errors or quantile regression")
    else:
        print("✅ Most variables normally distributed")
    
    print("\n🎯 SUGGESTED MODEL SPECIFICATIONS:\n")
    print("1. LINEAR MODEL (OLS):")
    print("   DVOL ~ dvol_lag1 + dvol_lag7 + dvol_lag30 + log(transaction_volume) +")
    print("          log(active_addresses) + nvrv + dvol_rv_spread")
    
    print("\n2. POLYNOMIAL MODEL (if nonlinearity detected):")
    print("   Add squared/cubic terms for non-linear predictors")
    
    print("\n3. LSTM MODEL (captures nonlinearity automatically):")
    print("   ✅ Best choice given non-linear relationships and time dependencies")
    print("   ✅ No need for manual polynomial terms")
    print("   ✅ Handles non-stationarity through sequential learning")
    
    print("\n4. ROBUST REGRESSION (if outliers/non-normality):")
    print("   Use Huber loss or quantile regression")
    
    # Save recommendations
    with open('results/modeling_recommendations.txt', 'w') as f:
        f.write("MODELING RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        for rec in recommendations:
            f.write(f"• {rec}\n")
        f.write("\nSee detailed results in CSV files for full analysis.\n")
    
    print(f"\n✅ Recommendations saved to results/modeling_recommendations.txt")


def main():
    """Run comprehensive analysis."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "COMPREHENSIVE VARIABLE ANALYSIS" + " " * 32 + "║")
    print("║" + " " * 10 + "DVOL Forecasting Model - Statistical Properties" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Load and prepare data
    df = load_data()
    df = calculate_features(df)
    
    # Run analyses
    summary_df = summary_statistics(df)
    normality_df = normality_tests(df)
    stationarity_df = stationarity_tests(df)
    linearity_df = linearity_tests(df)
    corr_matrix = correlation_analysis(df)
    create_scatter_plots(df)
    vif_df = multicollinearity_analysis(df)
    
    # Generate recommendations
    generate_recommendations(linearity_df, stationarity_df, normality_df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n📁 All results saved to results/ directory:")
    print("   • summary_statistics.csv")
    print("   • normality_tests.csv")
    print("   • stationarity_tests.csv")
    print("   • linearity_tests.csv")
    print("   • correlation_matrix.csv")
    print("   • vif_analysis.csv")
    print("   • correlation_heatmap.png")
    print("   • scatter_plots_linearity.png")
    print("   • modeling_recommendations.txt")
    
    print("\n✅ Ready to proceed with model specification!")
    print("\n")


if __name__ == "__main__":
    main()
