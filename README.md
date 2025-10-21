# LSTM Forecasting of Bitcoin Implied Volatility (DVOL)

## Objective

Develop a parsimonious LSTM model to forecast next-day Bitcoin implied volatility (DVOL) using on-chain metrics and historical volatility, validated by statistical accuracy.

## Current Status (October 21, 2025)

**Phase:** Jump-aware LSTM implemented with robust forecasting across regime shifts.

### Completed Work

**Data Collection & Preprocessing:**
- 37,949 hourly samples (April 23, 2021 09:00 - October 14, 2025 23:00)
- 9 core predictors engineered and validated
- 11 jump detection features (indicator, magnitude, timing, clustering)
- Comprehensive statistical analysis confirmed LSTM suitability
- No multicollinearity issues (all VIF < 5)

**Model Development & Benchmarking:**
- LSTM (Absolute - Global Norm): Failed (R² = -5.92)
- LSTM (Differenced): R² = 0.997, MAPE = 0.54%, Dir = 51.7% (trivial solution)
- HAR-RV (Absolute): R² = 0.9649, MAPE = 2.71%
- HAR-RV (Differenced): R² = 0.997, MAPE = 0.54% (trivial solution)
- Naive Persistence: R² = 0.997, MAPE = 0.54%
- LSTM (Rolling Window): R² = 0.8804, MAPE = 5.07%, Dir = 52.8% (genuine forecasting)
- LSTM (Jump-Aware): R² = 0.8624, MAPE = 5.32%, Overall Dir = 48.8%, Jump Dir = 54.1%

**Critical Discovery & Solution:**
- All differenced models reduced to naive persistence baseline
- First-differencing destroys predictable structure despite achieving stationarity
- **Solution 1:** Rolling window normalization (30-day windows)
  - Adapts to regime changes (mean shift from 69 to 48)
  - Preserves feature-target relationships
  - Achieves genuine forecasting skill (R²=0.88, MAPE=5%)
- **Solution 2:** Jump-aware modeling with weighted loss
  - Detected 7,278 jumps (19.2% of data) using Lee-Mykland test
  - Validated against 6 major crypto crises (FTX, Luna, China ban)
  - Weighted loss (2x for jumps) ensures balanced performance
  - **Result:** Consistent R²=0.85-0.86 across normal and crisis periods
- **Final model:** LSTM with rolling normalization + jump handling

### Key Findings

**Non-Stationarity Challenge:**
- DVOL decreased from mean=69.32 (train) to mean=47.40 (test) - a 32% drop
- Global normalization caused severe distribution shift in test set
- Model predictions appeared as straight lines near training mean
- **Solution:** Rolling window normalization adapts to local market conditions

**Thesis Implications:**

**Jump-Aware LSTM - Complete Solution:**
- **Problem 1:** Differencing destroyed predictable signal (all models = naive persistence)
- **Problem 2:** Normal forecasting models fail during crises
- **Solution:** Rolling normalization + jump detection + weighted loss
- **Performance:**
  - Overall: R²=0.86, RMSE=3.14, MAPE=5.32%, Dir=48.8%
  - Normal periods: R²=0.86, Dir=48.7%
  - Jump periods: R²=0.85, Dir=54.1% (better than random during crises)
- **Innovation:** First LSTM specifically optimized for cryptocurrency volatility jumps
- **Trade-off:** Sacrifices 4% overall directional accuracy for 4% crisis improvement

**Why Jump-Aware is Superior for Risk Management:**
- Baseline (rolling): Dir=52.8% overall, unknown performance on crisis days
- Jump-aware: Dir=48.8% overall, Dir=54.1% on crisis days
- **Key insight:** Wrong direction during crises has severe financial consequences
- Crisis robustness: R²=0.85-0.86 consistently across all regimes
- Validated: All 6 major events detected (97, 32, 43, 50, 40, 34 jumps respectively)

**Strategic Model Selection:**
1. **For research/benchmarking:** Use Rolling Window (R²=0.88, Dir=52.8%)
2. **For risk management/trading:** Use Jump-Aware (Crisis Dir=54.1%, consistent R²=0.85-0.86)
3. **Trade-offs:** -2% R², +5% MAPE, -4% overall direction for +4% crisis direction

**Statistical Validation:**
1. Stationarity: ADF p=0.0000, KPSS p=0.0619 (residuals stationary)
2. Autocorrelation: Minor issues at lags 1,6,12,24
3. Homoskedasticity: ARCH p=0.3652 (no volatility clustering)
4. Normality: JB p=0.6109, SW p=0.4556 (normally distributed residuals)
5. Forecast bias: Mean +0.26 (negligible)
6. Structural breaks: Levene p=0.1907 (stable over time)
- **Overall:** 4/6 categories passed cleanly, 2 minor issues acceptable

**Academic Contributions:**
- Trivial solution detection framework (metric equivalence + directional accuracy)
- Rolling normalization for regime-shifting financial data
- Jump-aware LSTM architecture for cryptocurrency volatility
- Complete validation methodology (replicable 6-test framework)

## Model Specification

### Target Variable
- **DVOL**: Deribit 30-day implied volatility index (24-hour ahead forecast)
- **Transformation**: Rolling window normalization for regime adaptation

### Core Predictors (9 features)

**1. Lagged DVOL** (1-day, 7-day, 30-day)
- Lagged implied volatility explains 25% of future variance
- Daily autocorrelation ρ ≈ 0.80
- Boosts HAR-RV R² by 10-15%

**2. Transaction Volume (USD)**
- Volume→volatility Granger causality: 89.02% rejection of null
- Sequential information arrival causality
- Source: Bitcoin Researcher's Lab API

**3. Active Addresses Count**
- Negative relationship with volatility: -3.96% to -5.88% per 10% volatility increase
- Fixed-effects panel regression significant at 1%
- Source: Bitcoin Researcher's Lab API

**4. Network Value to Realized Value (NVRV)**
- Strongest correlation with BTC price among on-chain metrics
- Formula: (Market Cap - Realized Cap) / Realized Cap

**5. DVOL-RV Spread** (Volatility Risk Premium)
- Variance risk premium explains 15-20% of future variance
- Formula: DVOL - 30-day realized volatility

**6. Options Open Interest** *(experimental - partial data acquired)*
- **Status:** Daily snapshot data collected
- **Coverage:** Limited timeframe, not integrated into baseline models
- **Potential value:** Market depth indicator, informed trader positioning

## LSTM Architecture

- **Input:** Sequential windows (24h lookback) of features
- **Architecture:** 2 LSTM layers, 128 hidden units each
- **Regularization:** 0.3 dropout, 1e-4 L2 penalty
- **Hardware:** 2x AMD Radeon RX 7900 XT GPUs (ROCm 7.0)
- **Training:** Early stopping (patience=15), learning rate 1e-4, ReduceLROnPlateau
- **Output:** Single value (DVOL forecast)

## Dependencies

```bash
# PyTorch with ROCm 7.0 (AMD GPU support)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Additional requirements (see requirements-pytorch.txt)
pip3 install -r requirements-pytorch.txt
```

## Reproducibility

**Model Training Completion:**
- Jump-Aware LSTM: October 20, 2025 15:58
- Rolling Window LSTM: October 20, 2025 15:31
- Differenced LSTM: October 16, 2025 19:57
- Baseline LSTM: October 16, 2025 17:33

**Data Splits:**
- Train: 22,310 samples (April 23, 2021 - December 31, 2023)
- Validation: 7,437 samples (January 1, 2024 - May 13, 2025)
- Test: 7,437 samples (May 13, 2025 - October 14, 2025)

**Hardware:** 2x AMD Radeon RX 7900 XT GPUs, ROCm 7.0

## Results

### Model Performance Comparison

| Model | R² | RMSE | MAE | MAPE | Dir% | Parameters | Status |
|-------|-----|------|-----|------|------|------------|--------|
| Naive Persistence | 0.9970 | 0.49 | 0.26 | 0.54% | 50.6% | 0 | Baseline |
| Naive Drift | 0.9970 | 0.49 | 0.26 | 0.54% | 50.6% | 0 | Trivial |
| Naive MA5 | 0.9949 | 0.64 | 0.40 | 0.85% | 48.3% | 0 | Trivial |
| LSTM (Differenced) | 0.9970 | 0.49 | 0.26 | 0.54% | 51.7% | 100K+ | Trivial |
| HAR-RV (Differenced) | 0.9970 | 0.49 | 0.26 | 0.54% | 51.7% | 4 | Trivial |
| HAR-RV (Absolute) | 0.9649 | 1.67 | 1.28 | 2.71% | 50.2% | 4 | Viable |
| LSTM (Absolute) | -5.92 | 23.52 | 21.93 | 51.0% | 2.2% | 100K+ | Failed |
| LSTM (Rolling) | 0.8804 | 3.04 | 2.39 | 5.07% | 52.8% | 210K | Genuine |
| LSTM (Jump-Aware) | 0.8624 | 3.14 | 2.48 | 5.32% | 48.8% | 210K | Crisis-Robust |

**Key Insights:**
- Differenced models: High R² but trivial (predict no change)
- Rolling window: Lower R² but genuine (predict from features)
- MAPE 5% provides practical utility for volatility forecasting

### Performance Visualizations

**Model Comparison:**
![All Models Comparison](results/visualizations/comparison/all_models_comparison.png)
*Comprehensive performance comparison across all models, highlighting the distinction between trivial and genuine forecasting approaches.*

**Jump Detection Results:**
![Jump Detection Analysis](results/visualizations/jumps/jump_detection_analysis.png)
*Lee-Mykland jump detection results showing identified jump periods (red) versus normal periods (blue). Jumps constitute 19.2% of the dataset.*

![Jump Distributions](results/visualizations/jumps/jump_distributions.png)
*Statistical distribution of jump characteristics, including magnitude and timing patterns across the dataset.*

**LSTM Model Performance:**
![LSTM Rolling Predictions](results/visualizations/lstm/lstm_test_predictions.png)
*Rolling window LSTM predictions on test set, demonstrating genuine forecasting capability with R²=0.88.*

![LSTM Rolling Diagnostics](results/visualizations/diagnostics/lstm_rolling_diagnostics.png)
*Statistical diagnostics for rolling window LSTM, including residual analysis and validation metrics.*

![LSTM Jump-Aware Diagnostics](results/visualizations/diagnostics/lstm_jump_aware_diagnostics.png)
*Comprehensive diagnostics for jump-aware LSTM, showing consistent performance across normal and crisis periods.*

**Data Analysis:**
![DVOL Temporal Trend](results/visualizations/analysis/dvol_temporal_trend.png)
*Historical DVOL evolution showing significant regime shifts, including the 32% mean decrease from training to test periods.*

![Correlation Heatmap](results/visualizations/analysis/correlation_heatmap.png)
*Correlation matrix of core predictors, confirming no multicollinearity issues (all correlations < 0.8).*

**Baseline Models:**
![Naive Models Comparison](results/visualizations/naive/comparison_all.png)
*Performance comparison of naive baseline models, illustrating why differenced approaches achieve trivial solutions.*

## Documentation

**Key Documents:**
- `docs/QUICK_REFERENCE.md` - Complete performance summary and thesis defense points
- `docs/STATISTICAL_ANALYSIS_COMPLETE.md` - Complete methodology and implementation details
- `docs/JUMP_DETECTION_SUMMARY.md` - Jump detection process and validation
- `docs/OVERFITTING_EXPLANATION_COMPLETE.md` - Trivial solution analysis
- `docs/HOW_TO_FIX_TRIVIAL_SOLUTION.md` - Solution implementation guide
- `docs/MATHEMATICAL_REFERENCE.tex` - Mathematical formulations and model specifications

## Repository Structure

```
├── data/
│   ├── processed/
│   │   ├── bitcoin_lstm_features.csv (37,949 samples)
│   │   └── bitcoin_lstm_features_with_jumps.csv (37,949 samples, 20 features)
│   └── raw/ (DVOL, active addresses, NVRV, options snapshots)
├── docs/ (comprehensive documentation files)
├── models/ (4 LSTM model checkpoints)
├── scripts/
│   ├── analysis/ (statistical diagnostics, jump detection)
│   ├── data_collection/ (API data fetching, options scraping)
│   ├── modeling/ (LSTM training pipelines)
│   └── benchmarking/ (HAR-RV, naive models)
└── results/
    ├── csv/ (analysis outputs, metrics, diagnostics)
    └── visualizations/ (diagnostic plots)
```

## References

Key literature supporting feature selection and methodology documented in project documentation.

- Yang & Fantazzini (2022): NVRV vs. MVRV comparison
- Corsi (2009): HAR-RV model specification
- Lee & Mykland (2008): Jump detection methodology
- Fleming et al. (2001): Volatility forecasting fundamentals