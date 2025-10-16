# LSTM Forecasting of Bitcoin Implied Volatility (DVOL)

## Objective

Develop and evaluate a parsimonious LSTM model to forecast next-day Bitcoin implied volatility (DVOL) using academically justified on-chain and derivatives-market features, validated by statistical accuracy and economic significance.

## Current Status

**Phase:** Data collection and feature engineering

**Completed:**
- DVOL hourly data collection (38,789 observations, 2021-2025)
- On-chain metrics collection (39,864 observations, 2021-2025)
- Realized volatility calculations
- Feature validation and exploratory analysis

**In Progress:**
- Options open interest data sourcing (OptionsDX evaluation)
- DVOL-RV spread calculation
- Feature preprocessing pipeline

**Next:**
- LSTM architecture implementation
- Model training and hyperparameter optimization
- Benchmark comparison (HAR-RV, GARCH, naive models)
- Economic validation via delta-neutral trading strategies


## Model Specification

### Dependent Variable
- **DVOL**: Deribit 30-day implied volatility index (daily)

### Core Predictors

**1. Lagged DVOL** (1-day, 7-day, 30-day)
- Lagged implied volatility explains 25% of future variance (Fleming et al. 2001)
- Daily autocorrelation ρ ≈ 0.80 (Christensen & Prabhala 1998)
- Boosts HAR-RV R² by 10-15% (Bollerslev et al. 2009)

**2. Transaction Volume (USD)**
- Volume→volatility Granger causality: 89.02% rejection of no-causality null (Yamak et al. 2019)
- Sequential information arrival causality
- Source: Bitcoin Researcher's Lab API

**3. Active Addresses Count**
- Negative relationship with volatility: -3.96% to -5.88% per 10% volatility increase
- Fixed-effects panel regression significant at 1% (Fiaschetti et al. 2024)
- Source: Bitcoin Researcher's Lab API

**4. Network Value to Realized Value (NVRV)**
- Strongest correlation with BTC price among on-chain metrics (Iraizoz Sánchez 2023)
- Profitable short-term trading: Sharpe 0.41
- Superior measure of holder P&L vs. MVRV (Yang & Fantazzini 2022)
- Formula: (Market Cap - Realized Cap) / Realized Cap

**5. DVOL-RV Spread** (Volatility Risk Premium)
- Variance risk premium explains 15-20% of future variance (Bollerslev et al. 2009)
- Cross-sectional R² up to 30% (Carr & Wu 2009)
- Reduces HAR-RV RMSE by 10-12% (Andersen et al. 2003)
- Formula: DVOL - 30-day realized volatility

**6. Options Open Interest** *(tentative - data source pending)*
- Potential sources: CME Bitcoin options or Deribit (via OptionsDX)
- Aggregated total OI or put/call ratio
- Market depth indicator for volatility forecasting


## LSTM Architecture

- Input: Sequential windows of core features capturing temporal dependencies
- Regularization: Dropout layers and L2 penalties to prevent overfitting
- Feature engineering: Moving averages, differenced series, regime indicators

## Validation Framework

**Statistical Metrics:**
- MAPE, RMSE, directional accuracy
- Benchmarks: HAR-RV, GARCH (EGARCH), naive lag models

**Economic Validation:**
- Delta-neutral straddle strategy backtesting with transaction costs
- Performance: Sharpe ratio, maximum drawdown, P&L
- Regime analysis: High vs. low DVOL-RV spread

**Interpretability:**
- SHAP analysis for feature importance quantification
- Dynamic driver identification

## Data Sources

- **DVOL**: Deribit API (90% of Bitcoin options market)
- **On-chain metrics**: Bitcoin Researcher's Lab API
- Coverage: March 2021 - October 2025 (hourly)
- Observations: 38,789 overlapping hours


## Repository Structure

```
├── data/
│   ├── processed/
│   │   ├── bitcoin_dvol_hourly_complete.csv
│   │   ├── bitcoin_nvrv_hourly_20251015.csv
│   │   └── options_oi_clean.csv
│   └── raw/
├── docs/
│   ├── COVARIATE_MATH.md
│   ├── HISTORICAL_OI_INVESTIGATION.md
│   └── sources.md
├── scripts/
│   ├── analysis/
│   │   ├── covariate_regression.py
│   │   ├── feature_diagnostics.py
│   │   └── statistical_summary.py
│   ├── data_collection/
│   │   ├── deribit_dvol_hourly.py
│   │   └── researchbitcoin_data.py
│   └── preprocessing/
└── results/
```

## Key References

- Andersen et al. (2003): HAR-RV RMSE improvements
- Bollerslev et al. (2009): Variance risk premium and HAR-RV
- Carr & Wu (2009): Variance risk premiums
- Christensen & Prabhala (1998): Implied-realized volatility relation
- Fiaschetti et al. (2024): Active addresses and volatility
- Fleming et al. (2001): Volatility timing value
- Iraizoz Sánchez (2023): NVRV correlation analysis
- Yamak et al. (2019): Volume-volatility Granger causality
- Yang & Fantazzini (2022): NVRV vs. MVRV comparison