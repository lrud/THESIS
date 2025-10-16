# Bitcoin Volatility Analysis: LSTM Forecasting of Bitcoin Implied Volatility (DVOL)

## Project Overview

**Thesis Objective**: Develop and evaluate a parsimonious LSTM model to forecast next-day Bitcoin implied volatility (DVOL) using academically justified on-chain and derivatives-market features, validated by both statistical accuracy and economic significance.

This project implements a data collection and analysis pipeline for Bitcoin volatility research using Network Value to Realized Value (NVRV) and Deribit Volatility Index (DVOL) metrics, building toward an academically rigorous LSTM forecasting model.

## Status: Data Collection Complete ✅

### Datasets
**NVRV On-Chain Data** (`bitcoin_nvrv_hourly_20251015.csv`)
- Records: 39,864 hourly observations
- Coverage: 2021-03-24 to 2025-10-14
- Metrics: Price, Market Cap, Realized Cap, NVRV, NUPL, Volume, TX Count

**DVOL Volatility Data** (`bitcoin_dvol_hourly_complete.csv`)  
- Records: 38,789 hourly observations
- Coverage: 2021-03-24 to 2025-10-15
- Metrics: DVOL Open, High, Low, Close
- Note: DVOL represents Deribit 30-day implied volatility index (90% of Bitcoin options are traded on Deribit)

**Alignment:** 39,984 overlapping hours for econometric analysis

## Model Development Framework

### A-Priori Five-Feature LSTM Model

**Dependent Variable:**
- **DVOL**: Deribit 30-day implied volatility index, daily observations (June 2021–present)

**Core Predictors (Academically Justified):**

#### 1. Lagged DVOL (1-day, 7-day, 30-day lags)
**Academic Justification:**
- Lagged implied volatility explains 25% of future variance (Fleming et al. 2001)
- Daily autocorrelation ρ ≈ 0.80 (Christensen & Prabhala 1998)
- Boosts HAR-RV R² by 10–15% (Bollerslev et al. 2009)
- Strong persistence and mean reversion properties similar to VIX

#### 2. Transaction Volume (USD)
**Academic Justification:**
- Sequential information arrival causality with 89.02% rejection of no-causality null at 200-day window
- Strong volume→volatility Granger causality (Yamak et al. 2019)
- "The causal relationship from volume to price volatility is stronger than the causal relationship from volatility to volume"

#### 3. Active Addresses Count
**Academic Justification:**
- Statistically significant negative relationship between price volatility and active addresses (Fiaschetti et al. 2024)
- Fixed-effects panel regression on 58 blockchain tokens shows -3.96% to -5.88% address decrease per 10% volatility increase
- Volatility interlinked with active user count, user activity, holding behavior, and token retention
- Active user count is fundamental input for crypto asset volatility models

#### 4. Network Value to Realized Value (NVRV)
**Academic Justification:**
- Strongest correlation with Bitcoin price among on-chain metrics (Iraizoz Sánchez 2023)
- Measures aggregate unrealized profit/loss, capturing holder sentiment and hedging demand
- More precise measure of aggregate holder P&L than traditional MVRV (Yang & Fantazzini 2022)
- Profitable short-term trading performance with Sharpe ratio 0.41

#### 5. DVOL–RV Spread (Volatility Risk Premium)
**Academic Justification:**
- Variance risk premium explains 15–20% of future variance (Bollerslev et al. 2009)
- Cross-sectional R² up to 30% for forecasting realized volatility (Carr & Wu 2009)
- Captures independent risk factors not explained by standard asset pricing models
- Reduces HAR-RV RMSE by 10–12% (Andersen et al. 2003)

## NVRV Methodology

**Academic Formula** (validated against peer-reviewed literature):
```
NVRV = (Market Cap - Realized Cap) / Realized Cap
```
- Net premium format optimized for econometric analysis
- Chosen over NVT ratio due to:
  - NVT's high volatility and difficulty in practical application
  - Off-chain exchange volume reducing NVT signal effectiveness
  - NVRV's superior correlation with Bitcoin price movements
  - Better proxy for behavioral sentiment and hedging demand

**Why NVRV over Traditional MVRV:**
- Enhanced correlation with Bitcoin price movements
- Superior behavioral sentiment proxy
- Improved volatility modeling capabilities
- Accounts for lost coins and provides cleaner bubble detection

## Data Quality & Sources

**Enterprise-Grade APIs with Academic Citations:**
- **Laevitas API**: Enterprise-grade, cited in Fiaschetti et al. (2024)
- **Bitcoin Researcher's Lab API**: Used in Yamak et al. (2019) and other peer-reviewed studies
- **CoinGlass**: Funding rates data cited in Kim & Park (2025)

**Data Quality Note**: While Atzori (2019) raised concerns about cryptocurrency data standardization, this was prior to the explosion of institutional-grade data sources. Our chosen APIs have established academic track records and institutional usage.

## Model Architecture & Evaluation

### LSTM Network Design
- Input sequence windows capturing temporal dependencies across five features
- Feature engineering: Moving averages, differenced series, regime indicators
- Regularization: Dropout layers and L2 penalties to mitigate overfitting

### Benchmarks & Evaluation Metrics

**Statistical Validation:**
- MAPE, RMSE, and directional accuracy against:
  - HAR-RV models
  - GARCH (EGARCH) models  
  - Naïve lag models (DVOL autoregression)

**Economic Validation:**
- Delta-neutral straddle strategy backtesting with transaction costs
- Performance metrics: Sharpe ratio, maximum drawdown, P&L comparison
- Regime analysis: "High vs. low" DVOL–RV spread performance

**Interpretability:**
- SHAP analysis for feature importance quantification
- Dynamic driver identification for implied volatility
- Economic logic validation for each feature

## Academic Contributions

1. **First LSTM model** forecasting Bitcoin implied volatility using on-chain and derivatives data
2. **Empirical validation** of NVRV and DVOL–RV spread in implied volatility forecasting context
3. **Economic significance testing** via delta-neutral trading strategies on DVOL forecasts
4. **Explainable AI integration** identifying key market drivers of Bitcoin implied volatility

## Variables Considered but Excluded

### NVT (Network Value to Transactions) Ratio
**Excluded due to:**
- High volatility making it difficult to apply practically
- Off-chain exchange volume reducing signal effectiveness
- Lack of direct evidence in traditional finance literature
- Academic literature noting its limitations for crash prediction

### Perpetual Futures Funding Rates
**Excluded due to:**
- Weak academic support for inclusion in volatility models
- Low R² relationship with price movements over weekly periods
- Relationship degradation over longer time horizons
- Insufficient direct justification for implied volatility modeling

## Next Steps

### Immediate Development Phase
1. **Data preprocessing and feature engineering**
   - Moving averages and differenced series
   - Regime indicators and volatility clustering analysis
   - DVOL-RV spread calculations

2. **LSTM Model Implementation**
   - Architecture design with temporal dependency capture
   - Hyperparameter optimization
   - Regularization implementation (dropout, L2 penalties)

3. **Model Validation**
   - Statistical benchmarking against HAR-RV, GARCH models
   - Economic validation through delta-neutral strategies
   - SHAP analysis for interpretability

4. **Academic publication preparation**
   - Results documentation and academic writing
   - Peer review preparation

### Future Extensions
- Multi-venue DVOL indices as markets mature
- Intraday features and higher-frequency variants
- Alternative risk premium proxies exploration

## Implementation Timeline
- **Data Coverage**: June 2021–present (~1,500 observations)
- **API Costs**: $0–$50 monthly (within free tiers)
- **Development Phase**: Current focus on preprocessing and LSTM implementation

## File Structure
```
├── README.md
├── config/
├── data/
│   ├── features/
│   ├── processed/
│   └── raw/
│       ├── bitcoin_dvol_hourly_complete.csv
│       └── bitcoin_nvrv_hourly_20251015.csv
├── docs/
│   └── api_requirements_lean.md
├── models/
├── notebooks/
├── results/
└── scripts/
    ├── analysis/
    ├── data_collection/
    │   ├── deribit_dvol_hourly.py
    │   └── researchbitcoin_data.py
    ├── modeling/
    └── preprocessing/
```

## Usage

### Environment Setup
```bash
export RESEARCH_BITCOIN_API_TOKEN="your_api_token"
```

### Data Collection
```bash
python scripts/data_collection/researchbitcoin_data.py
```

## Academic References

### Core LSTM Model Support
- **Fleming, Kirby & Ostdiek (2001)**: "The Economic Value of Volatility Timing" - Demonstrates lagged VIX explains 25% of subsequent variance
- **Christensen & Prabhala (1998)**: "The Relation between Implied and Realized Volatility" - Shows daily autocorrelation ρ ≈ 0.80 for lag-1 VIX
- **Bollerslev, Tauchen & Zhou (2009)**: "Expected Stock Returns and Variance Risk Premia" - HAR-RV R² improvements of 10–15% with lagged VIX
- **Carr & Wu (2009)**: "Variance Risk Premiums" - Cross-sectional R² up to 30% using implied-realized volatility spread

### Bitcoin-Specific Variables
- **Fiaschetti et al. (2024)**: Fixed-effects panel regression showing statistically significant negative relationship between volatility and active addresses
- **Yamak et al. (2019)**: Volume→volatility Granger causality with 89.02% rejection of no-causality null
- **Iraizoz Sánchez (2023)**: NVRV shows strongest correlation with Bitcoin price among on-chain metrics
- **Yang & Fantazzini (2022)**: NVRV provides more precise measure of aggregate holder P&L than traditional MVRV

### Data Quality Validation
- **Atzori (2019)**: Early concerns about cryptocurrency data standardization (pre-institutional data sources)
- **Kim & Park (2025)**: CoinGlass funding rates academic citation
- **Enterprise APIs**: Laevitas, Bitcoin Researcher's Lab, CoinGlass all with established academic track records

The NVRV metric selection is based on academic literature demonstrating its superiority over traditional MVRV for Bitcoin analysis, while the LSTM framework leverages established volatility forecasting methodologies from traditional finance adapted for cryptocurrency markets.

## Technical Notes

- All timestamps in UTC
- CSV format with pandas-compatible datetime indexing
- Data validation performed on collection
- Missing values handled through dropna() operations