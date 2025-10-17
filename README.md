# LSTM Forecasting of Bitcoin Implied Volatility (DVOL)

## Objective

Develop a parsimonious LSTM model to forecast next-day Bitcoin implied volatility (DVOL) using on-chain metrics and historical volatility, validated by statistical accuracy.

## Current Status (October 16, 2025)

**Phase:** Model training and validation ✅

### ✅ Completed

**Data Collection & Preprocessing:**
- 37,927 hourly samples (April 2021 - October 2025)
- 5 core predictors engineered and validated
- Comprehensive statistical analysis confirmed LSTM suitability (all variables non-linear, non-stationary)
- No multicollinearity issues (all VIF < 5)

**Model Development:**
- Baseline LSTM trained but **failed catastrophically** (R² = -5.92, straight-line predictions)
- **Critical issue identified:** Non-stationary target with 32% downward trend from training to test period
- **Solution implemented:** First differences transformation (Δdvol = dvol_t - dvol_{t-1})
- **Differenced LSTM trained successfully:** R² = 0.997, MAPE = 0.54%, Directional Accuracy = 51.7%

### 🔍 Key Learnings

**Non-Stationarity Challenge:**
- DVOL decreased from mean=69.32 (train) to mean=47.40 (test) - a 32% drop
- Global normalization caused severe distribution shift in test set
- Model predictions appeared as straight lines near training mean
- **Solution:** Transformed to first differences, which are stationary (mean ≈ 0 across all splits)

**Model Performance (Differenced LSTM):**
- Test R²: 0.9970 (excellent fit)
- Test MAPE: 0.54% (highly accurate)
- Test Directional Accuracy: 51.73% (slightly better than random)
- Reconstruction successful: predictions map back to original DVOL scale correctly

### ⚠️ Concerns & Next Steps

**Potential Overfitting Risk:**
- R² = 0.997 is exceptionally high - may indicate overfitting to training data
- Directional accuracy only 51.7% suggests model captures magnitude better than direction
- Need to validate on completely unseen data (out-of-sample testing beyond Oct 2025)
- Consider ensemble methods or additional regularization

**Recommended Next Steps:**
1. Extended validation period with new data (Nov 2025+)
2. Benchmark against simpler models (HAR-RV, GARCH, naive persistence)
3. Hyperparameter tuning to balance accuracy vs. generalization
4. Explore attention mechanisms or bidirectional LSTM
5. Test different sequence lengths (48h, 72h windows)

### 🔄 Potential Enhancements

**Options Open Interest (6th predictor):**
- Investigated 8+ data sources - only viable option is OptionsDX (~$40-50 for historical data)
- Currently excluded from baseline model
- Could add market depth signal for volatility forecasting
- Decision pending on budget allocation

**Alternative Approaches:**
- Rolling window normalization (adapt to regime changes)
- Percentage changes instead of first differences
- Detrending methods for structural breaks

## Model Specification

### Target Variable
- **DVOL**: Deribit 30-day implied volatility index (24-hour ahead forecast)
- **Transformation**: First differences (Δdvol = dvol_t - dvol_{t-1}) to achieve stationarity

### Core Predictors (5 features)

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

**6. Options Open Interest** *(excluded from baseline - data acquisition pending)*
- **Status:** Evaluated 8+ data sources, only OptionsDX viable (~$40-50 for historical data)
- **Rationale for exclusion:** Building baseline model first with freely available features
- **Potential value:** Market depth indicator, informed trader positioning signal
- **Next steps:** Consider acquisition if baseline results warrant enhancement


## LSTM Architecture (Implemented)

- **Input:** Sequential windows (24h lookback) of 7 features
- **Architecture:** 2 LSTM layers, 128 hidden units each
- **Regularization:** 0.3 dropout, 1e-4 L2 penalty
- **Hardware:** 2x AMD Radeon RX 7900 XT GPUs (ROCm 7.0)
- **Training:** Early stopping (patience=15), learning rate 1e-4
- **Output:** Single value (Δdvol forecast), reconstructed to absolute DVOL

## Validation & Results

**Training Splits:**
- Train: 60% (April 2021 - December 2023)
- Validation: 20% (January 2024 - November 2024)  
- Test: 20% (November 2024 - October 2025)

**Test Performance (Differenced Model):**
- R² = 0.9970
- MAPE = 0.54%
- RMSE = 0.49
- MAE = 0.26
- Directional Accuracy = 51.7%

**Baseline Comparison (Absolute DVOL - Failed):**
- R² = -5.92 (worse than mean prediction)
- MAPE = 51.02%
- Directional Accuracy = 2.16%
- **Issue:** Non-stationary target caused catastrophic failure


## Documentation

**Key Documents:**
- `docs/CRITICAL_ISSUE_NON_STATIONARY_TARGET.md` - Complete analysis of non-stationarity problem and solution
- `docs/THESIS_METHODOLOGY_REFERENCE.md` - Comprehensive methodology for thesis writing
- `docs/MODEL_TRAINING_PROGRESS_LOG.md` - Session timeline and implementation details
- `docs/HISTORICAL_OI_INVESTIGATION.md` - Options OI data source research (8+ sources evaluated)
- `results/BASELINE_LSTM_SUMMARY.md` - Statistical analysis and baseline results

## Repository Structure

```
├── data/
│   ├── processed/bitcoin_lstm_features.csv (37,927 samples)
│   └── raw/ (DVOL, active addresses, NVRV)
├── docs/ (6 comprehensive documentation files)
├── models/ (baseline + differenced LSTM checkpoints)
├── scripts/
│   ├── analysis/ (statistical diagnostics)
│   ├── data_collection/ (API data fetching)
│   └── modeling/ (modular LSTM pipeline)
└── results/
    ├── csv/ (8 analysis outputs)
    └── visualizations/ (13 plots)
```

## References

Key literature supporting feature selection and methodology documented in `docs/COVARIATE_MATH.md` and `docs/sources.md`.

- Yang & Fantazzini (2022): NVRV vs. MVRV comparison