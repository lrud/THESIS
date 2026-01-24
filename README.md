# LSTM Forecasting of Bitcoin Implied Volatility (DVOL)

## Objective

Develop an LSTM neural network model to forecast Bitcoin implied volatility (DVOL) using on-chain metrics and historical volatility patterns, validated through statistical analysis.

## Current Status (January 23, 2026)

**Phase:** Model architecture optimization complete with v1.1 validated results.

### Recent Developments

**LSTM Architecture Optimization (January 2026):**
- Identified optimal architecture: 512 hidden units × 7 layers (13.8M parameters)
- R² = 0.800 on corrected v1.1 data with jump-aware features
- Depth scaling effective (3→5→7 layers), width scaling causes instability
- Rolling baseline (non-jump-aware): R² = 0.201, validates jump feature importance

**Statistical Validation Framework (January 2026):**
- NVRV confirmed non-stationary (ADF p=0.186), requires differencing for linear models
- VaR backtesting passes at 95% and 99% confidence (Kupiec test)
- LSTM achieves 78% improvement over persistence, 95% over historical mean
- Signal-to-noise analysis: Forecast error (RMSE=1.67) is 11x typical daily change (0.26)

**Unified Model Benchmarking (January 23, 2026):**
- Implemented 13 models: 5 linear (OLS, HAR-RV variants) + 8 tree-based (RF, XGBoost)
- All models use 720-hour rolling window normalization for fair LSTM comparison
- HAR-RV achieves R² = 0.9454 using only volatility persistence (canonical Corsi 2009)
- Tree-based models with lags match linear performance (RF: 0.9485, XGB: 0.9429)
- Jump features provide minimal or negative impact across all specifications
- **Critical finding:** All models achieve ~50% directional accuracy (statistically random)

**Dataset v1.1 Complete (December 2025):**
- Transaction volume data extended to December 28, 2025
- Dataset increased from 37,951 to 39,472 hourly samples
- **Critical fix**: `dvol_rv_spread` correlation corrected from 0.0485 (near-random) to 0.9905 (valid)
- All 9 core predictors now have 100% coverage with validated relationships

**Code Consolidation (November 2025):**
- 50% reduction in code duplication through systematic consolidation
- Unified utilities module (`scripts/utils/`) for metrics and HAR-RV models
- CLI training system replacing legacy script-based approach
- Documentation and backward compatibility preservation

### Completed Work

**Data Collection & Preprocessing:**
- 39,472 hourly samples (April 23, 2021 09:00 - December 28, 2025 23:00)
- 9 core predictors engineered and validated (100% complete)
- 11 jump detection features (indicator, magnitude, timing, clustering)
- Statistical analysis confirmed LSTM suitability
- No multicollinearity issues (all VIF < 5)

**Model Development & Benchmarking:**

- LSTM (Absolute - Global Norm): Failed (R² = -5.92)
- LSTM (Differenced): R² = 0.997, MAPE = 0.54%, Dir = 51.7% (trivial solution)
- Naive Persistence: R² = 0.997, MAPE = 0.54%
- LSTM (Rolling Window 512×7): R² = 0.201, MAE = 4.31, Dir = 49.7% (v1.1 baseline)
- **LSTM (Jump-Aware 512×7)**: R² = 0.800, MAE = 2.04, RMSE = 2.86, Dir = 49.7% (v1.1 optimal)

**Unified Model Benchmarks (January 23, 2026 - 13 Models):**

Using 720-hour rolling window normalization for fair comparison:

**Linear Models (5 specifications):**

| Model | Features | Test R² | Test RMSE | Test MAE | Dir% |
|------|----------|---------|-----------|----------|------|
| **HAR-RV (Canonical)** | 3 (daily+weekly+monthly lags) | 0.9454 | 1.71 | 1.25 | 50.6% |
| **OLS (With Lags + Jumps)** | 11 (lags + on-chain + jumps) | 0.9490 | 1.65 | 1.18 | 51.2% |
| **OLS (With Lags)** | 7 (lags + on-chain) | 0.9480 | 1.67 | 1.19 | 51.2% |
| **OLS (No Lags + Jumps)** | 8 (market + jumps) | 0.7393 | 3.74 | 2.85 | 51.3% |
| **OLS (No Lags)** | 4 (market features only) | 0.7363 | 3.76 | 2.89 | 51.3% |

**Tree-Based Models (8 specifications):**

| Model | Features | Test R² | Test RMSE | Test MAE | Dir% |
|------|----------|---------|-----------|----------|------|
| **RF (Lags + Jumps)** | 11 (lags + on-chain + jumps) | 0.9492 | 1.65 | 1.18 | 51.0% |
| **RF (Lags)** | 7 (lags + on-chain) | 0.9485 | 1.66 | 1.19 | 51.4% |
| **RF (No Lags + Jumps)** | 8 (market + jumps) | 0.7564 | 3.61 | 2.81 | 51.0% |
| **RF (No Lags)** | 4 (market features only) | 0.6914 | 4.06 | 2.99 | 50.8% |
| **XGB (Lags)** | 7 (lags + on-chain) | 0.9429 | 1.75 | 1.24 | 51.1% |
| **XGB (Lags + Jumps)** | 11 (lags + on-chain + jumps) | 0.9384 | 1.82 | 1.28 | 50.4% |
| **XGB (No Lags + Jumps)** | 8 (market + jumps) | 0.7304 | 3.80 | 2.87 | 51.1% |
| **XGB (No Lags)** | 4 (market features only) | 0.6989 | 4.02 | 2.98 | 50.2% |

**Key Findings (13 Models):**
- **Volatility persistence is king:** HAR-RV (3 features) achieves 94.5% R²
- **Tree models match linear:** RF and XGB with lags achieve comparable performance to OLS
- **Jump features have minimal impact:** Adding jumps improves R² by <0.5% across all specifications
- **Directional accuracy ~50%:** All 13 models are statistically random for direction prediction
- **Feature engineering trumps complexity:** Simple 3-feature HAR-RV outperforms complex 11-feature models

**Critical Discovery & Solution:**

- All differenced models reduced to naive persistence baseline
- First-differencing destroys predictable structure despite achieving stationarity
- **Solution 1:** Rolling window normalization (720-hour windows)
  - Adapts to regime changes (mean shift from 69 to 48)
  - Preserves feature-target relationships
  - Achieves genuine forecasting skill (R²=0.201 without jump features)
- **Solution 2:** Jump-aware modeling with weighted loss
  - Detected 7,278 jumps (19.2% of data) using Lee-Mykland test
  - Validated against 6 major crypto crises (FTX, Luna, China ban)
  - Weighted loss (2x for jumps) ensures balanced performance
  - **Result:** R² improvement from 0.201 to 0.800 with jump features
- **Final model:** LSTM 512×7 with rolling normalization + jump handling

**v1.0 vs v1.1 Performance:**

- v1.0 (incorrect dvol_rv_spread): R² = 0.8624 (inflated by feature error)
- v1.1 (corrected dvol_rv_spread): R² = 0.800 (valid performance)
- The v1.1 results represent genuine forecasting skill on accurate data

### Key Findings

**Non-Stationarity Challenge:**

- DVOL decreased from mean=69.32 (train) to mean=47.40 (test) - a 32% drop
- NVRV confirmed non-stationary (ADF p=0.186), requires differencing for linear models
- Global normalization caused severe distribution shift in test set
- **Solution:** Rolling window normalization adapts to local market conditions

**Magnitude vs Directional Prediction (v1.1 Key Insight):**

- **Magnitude forecasting (R²):** Jump-aware dramatically outperforms rolling (0.800 vs 0.201, +298%)
- **Directional forecasting:** Both models at ~49.7% (statistically indistinguishable from random)
- **Root cause:** Forecast error (RMSE=2.86) is 11x larger than typical daily change (0.26)
- **Interpretation:** Model excels at regime tracking (DVOL will be ~44-45) but cannot predict direction (up vs down)
- **Practical implication:** Suitable for risk management/option pricing, NOT for directional trading

**Thesis Implications:**

**Jump-Aware LSTM - Complete Solution:**
- **Problem 1:** Differencing destroyed predictable signal (all models = naive persistence)
- **Problem 2:** NVRV non-stationary breaks linear models
- **Solution:** Rolling normalization + jump detection + weighted loss
- **Performance (v1.1 validated):**
  - Overall: R²=0.800, RMSE=2.86, MAE=2.04, Dir=49.7%
  - Normal periods: R²=0.801, RMSE=2.85
  - Jump periods: R²=0.796, RMSE=2.93 (only 10% worse than normal)
- **Contribution:** First LSTM specifically optimized for cryptocurrency volatility jumps with 13.8M parameters
- **Trade-off:** Jump features provide 4x magnitude improvement (R² 0.201→0.800) but do not improve directional accuracy

**VaR Backtesting (Out-of-Sample Validation):**
- 95% VaR: 3.31 (5.01% exceedance vs 5.0% expected, Kupiec PASS)
- 99% VaR: 5.66 (1.01% exceedance vs 1.0% expected, Kupiec PASS)
- **Interpretation:** Model does NOT dangerously underestimate tail risk
- **Practical use:** VaR estimates are statistically valid for position sizing and risk limits

**Naive Benchmark Comparison (Out-of-Sample):**
- LSTM (MAE=1.20, R²=0.932) vs Persistence (MAE=5.41, R²=-0.018): +77.8% improvement
- LSTM vs Historical Mean (MAE=24.42, R²=-14.56): +95.1% improvement
- **Conclusion:** LSTM demonstrates genuine forecasting skill beyond simple baselines

**Academic Contributions:**

- Trivial solution detection framework (metric equivalence + directional accuracy)
- Rolling normalization for regime-shifting financial data
- Jump-aware LSTM architecture for cryptocurrency volatility (13.8M parameters, 512×7)
- VaR backtesting framework for financial model validation
- Signal-to-noise analysis explaining R² vs directional accuracy contradiction
- NVRV non-stationarity validation with ADF testing

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

- **Input:** Sequential windows (24h lookback) of 11 features (9 predictors + 2 jump features)
- **Optimal Architecture:** 7 LSTM layers, 512 hidden units each (13.8M parameters)
- **Regularization:** 0.5 dropout, 1e-4 L2 penalty
- **Hardware:** 2x AMD Radeon RX 7900 XT GPUs (ROCm 7.0)
- **Training:** Early stopping (patience=15), learning rate 1e-4, ReduceLROnPlateau
- **Jump-Aware Loss:** 2× weighting for jump periods (7,278 jumps, 19.2% of data)
- **Output:** Single value (DVOL forecast)

**Architecture Scaling Results (v1.1):**

| Architecture | Hidden | Layers | Params | R² | Status |
|--------------|--------|--------|--------|----|--------|
| Ultra-Large | 512 | 3 | 5.4M | 0.795 | Stable |
| Deep | 512 | 5 | 9.6M | 0.784 | Stable |
| **Optimal** | **512** | **7** | **13.8M** | **0.800** | **Stable** |
| Wide-Deep | 544 | 7 | 15.6M | 0.790 | Stable, worse |
| Wide | 1024 | 3 | 21.6M | 0.736 | Unstable (Val: inf) |

**Key Finding:** Depth scales (3→5→7 layers improves R²), width does not (512→544+ degrades performance)

## CLI Training System

The project implements a CLI training system that replaces the original script-based approach:

### Core Training Commands

```bash
# Optimal model (512×7, R² = 0.800 on v1.1)
.venv/bin/python cli/bin/train.py jump_aware \
  --hidden-size 512 --num-layers 7 --dropout 0.5 \
  --batch-size 32 --lr 0.0001 --epochs 100 \
  --use-multi-gpu --save-prefix deep_512x7

# Rolling baseline (non-jump-aware, R² = 0.201 on v1.1)
.venv/bin/python cli/bin/train.py rolling \
  --hidden-size 512 --num-layers 7 --dropout 0.5 \
  --batch-size 32 --lr 0.0001 --epochs 100 \
  --use-multi-gpu --save-prefix rolling_512x7

# Standard configurations
.venv/bin/python cli/bin/train.py jump_aware --epochs 50
.venv/bin/python cli/bin/train.py rolling --epochs 50
```

### Multi-GPU Training
```bash
# Automatic learning rate scaling for DataParallel stability
python cli/bin/train.py jump_aware --use-multi-gpu --lr 0.0001
# Internally scales to 0.00005 for dual GPU configuration
```

### Real-time Monitoring
```bash
# Monitor training progress and convergence metrics
tail -f results/logs/current_training.log
```

## Dependencies

### Core Requirements
```bash
# PyTorch with ROCm 7.0 (AMD GPU support)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Additional requirements (see requirements-pytorch.txt)
pip3 install -r requirements-pytorch.txt
```

### Hardware Requirements
- **Recommended**: Dual AMD Radeon RX 7900 XT GPUs (20GB VRAM each)
- **Minimum**: Single GPU with 8GB+ VRAM
- **CPU**: Multi-core processor for data preprocessing
- **RAM**: 16GB+ for large model training

## Reproducibility

**Model Training Completion:**
- Ultra-Large Jump-Aware (5.41M parameters): November 7, 2025 12:27
- Large Jump-Aware (1.36M parameters): November 7, 2025 11:13
- Jump-Aware LSTM: October 20, 2025 15:58
- Rolling Window LSTM: October 20, 2025 15:31
- Differenced LSTM: October 16, 2025 19:57
- Baseline LSTM: October 16, 2025 17:33

**Data Splits:**

- Train: 23,683 samples (60%, April 23, 2021 - February 9, 2024)
- Validation: 7,894 samples (20%, February 9, 2024 - January 14, 2025)
- Test: 7,895 samples (20%, January 14, 2025 - December 28, 2025)

**Hardware:** 2x AMD Radeon RX 7900 XT GPUs, ROCm 7.0

## Results

### Model Performance Comparison (v1.1 Complete Data - 15 Models)

**Linear and Tree-Based Models (13 specifications, 720-hour rolling normalization):**

| Model | R² | RMSE | MAE | Dir% | Parameters | Type | Status |
|-------|-----|------|-----|------|------------|------|--------|
| **RF (Lags + Jumps)** | 0.9492 | 1.65 | 1.18 | 51.0% | - | Tree | Best linear/tree |
| **OLS (With Lags + Jumps)** | 0.9490 | 1.65 | 1.18 | 51.2% | 11 | Linear | Best OLS |
| **RF (Lags)** | 0.9485 | 1.66 | 1.19 | 51.4% | - | Tree | - |
| **OLS (With Lags)** | 0.9480 | 1.67 | 1.19 | 51.2% | 7 | Linear | - |
| **HAR-RV (Canonical)** | 0.9454 | 1.71 | 1.25 | 50.6% | 3 | Linear | Baseline |
| **XGB (Lags)** | 0.9429 | 1.75 | 1.24 | 51.1% | - | Tree | - |
| **XGB (Lags + Jumps)** | 0.9384 | 1.82 | 1.28 | 50.4% | - | Tree | Jumps hurt |
| **RF (No Lags + Jumps)** | 0.7564 | 3.61 | 2.81 | 51.0% | - | Tree | - |
| **OLS (No Lags + Jumps)** | 0.7393 | 3.74 | 2.85 | 51.3% | 8 | Linear | - |
| **OLS (No Lags)** | 0.7363 | 3.76 | 2.89 | 51.3% | 4 | Linear | - |
| **XGB (No Lags + Jumps)** | 0.7304 | 3.80 | 2.87 | 51.1% | - | Tree | - |
| **XGB (No Lags)** | 0.6989 | 4.02 | 2.98 | 50.2% | - | Tree | - |
| **RF (No Lags)** | 0.6914 | 4.06 | 2.99 | 50.8% | - | Tree | Worst lag-free |

**LSTM and Baseline Models:**

| Model | R² | RMSE | MAE | MAPE | Dir% | Parameters | Dataset | Status |
|-------|-----|------|-----|------|------|------------|---------|--------|
| **LSTM Jump-Aware 512×7** | **0.800** | **2.86** | **2.04** | **~6.2%** | **49.7%** | **13.8M** | **Levels** | **Optimal (v1.1)** |
| **LSTM Rolling 512×7** | **0.201** | **6.20** | **4.31** | **9.45%** | **49.7%** | **13.8M** | **Levels** | **Non-jump baseline** |
| Naive Persistence | -0.018 | 6.46 | 5.41 | - | 50.0% | 0 | Levels | Out-of-sample |
| Historical Mean | -14.56 | 25.25 | 24.42 | - | - | 0 | Levels | Out-of-sample |

**v1.0 vs v1.1 Comparison (Feature Correction Impact):**

| Model | v1.0 R² | v1.1 R² | Change | Explanation |
|-------|---------|---------|--------|-------------|
| LSTM Jump-Aware | 0.8624 | 0.800 | -7.2% | dvol_rv_spread correlation fixed |
| LSTM Rolling | 0.8804 | 0.201 | -77.2% | Jump features provide 4x improvement |

**Key Insights (15 Models):**

- **Magnitude forecasting (R²):** Linear/tree models achieve 94-95% R² using volatility persistence
- **Directional forecasting:** ALL 15 models achieve ~50% accuracy (statistically random)
- **Volatility persistence dominates:** HAR-RV (3 features) achieves 94.5% R²
- **Tree models match linear:** RF and XGB perform comparably to OLS with same features
- **Jump features minimal impact:** Adding jumps improves R² by <0.5% across all specifications
- **LSTM underperforms linear:** Jump-aware LSTM (0.800) << best linear (0.9492)
- **Root cause:** Hourly DVOL autocorrelation = 0.9992 (extreme persistence)
- **Practical implication:** Models suitable for risk management, NOT directional trading

### Performance Visualizations

**Model Comparison:**
![All Models Comparison](results/visualizations/comparison/all_models_comparison.png)
*Visualization showing the distinction between statistical illusions (red) and genuine forecasting models (green). The plot reveals that high R² values (≈0.997) often indicate trivial solutions equivalent to naive persistence, while genuine forecasting models achieve lower R² (0.86-0.88) but demonstrate real directional accuracy (>50%).*

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
*Diagnostics for jump-aware LSTM, showing consistent performance across normal and crisis periods.*

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

- `CLAUDE.md` - Claude AI assistant guide and project context
- `docs/research/session_logs/THESIS_V2_SESSION_CONSOLIDATION_2026-01-02.md` - Complete research session log with v1.1 validated results, LSTM architecture optimization, and VaR backtesting
- `docs/QUICK_REFERENCE.md` - Performance summary and thesis defense points
- `scripts/thesis_v2/har_rv/` - Modular HAR-RV analysis package with statistical diagnostics
- `scripts/utils/README.md` - Consolidated utilities implementation guide

**Session Log Highlights (January 2, 2026):**

- LSTM architecture optimization: 512×7 (13.8M parameters) identified as optimal
- Jump-aware vs rolling comparison: R² 0.201 → 0.800 (+298%)
- NVRV non-stationarity confirmed (ADF p=0.186)
- VaR backtesting passes at 95% and 99% confidence (out-of-sample)
- Naive benchmarks: LSTM achieves 78% improvement over persistence, 95% over historical mean
- Signal-to-noise analysis explains R² vs directional accuracy contradiction
- v1.0 vs v1.1 comparison: Feature correction reduced R² from 0.8624 to 0.800 (valid performance)

## Repository Structure

```
├── cli/                          # Modern training interface
│   ├── bin/train.py             # Main CLI entry point
│   ├── config/config.py         # Configuration management system
│   └── scripts/trainers/        # Modular trainer implementations
├── scripts/                     # Analysis and modeling components
│   ├── thesis_v2/har_rv/        # Modular HAR-RV analysis package
│   │   ├── models.py            # Core HAR-RV model classes
│   │   ├── diagnostics.py       # Statistical testing framework
│   │   ├── baseline.py          # Baseline model runners
│   │   ├── visualization.py     # Plotting and visualization
│   │   ├── cli.py               # Command-line interface
│   │   └── __init__.py          # Package exports
│   ├── utils/                   # Consolidated shared utilities
│   │   ├── metrics.py           # Unified evaluation metrics
│   │   ├── har_rv.py            # Backward-compatible wrapper
│   │   └── __init__.py
│   ├── modeling/                # LSTM neural network components
│   ├── analysis/                # Statistical validation frameworks
│   ├── benchmarking/            # Benchmark utilities
│   └── data_collection/         # Data acquisition pipelines
├── deprecated/                  # Archived superseded implementations
│   └── har_rv_v1.0.py           # Original monolithic HAR-RV (2,480 lines)
├── data/
│   ├── processed/
│   │   └── bitcoin_lstm_features_v1.1_complete_with_jumps.csv (39,472 samples, 20 features)
│   └── raw/ (DVOL, active addresses, NVRV, options snapshots)
├── docs/ (documentation files)
├── models/ (LSTM model checkpoints, including large-scale models)
└── results/
    ├── cli_training/            # CLI training results with JSON metadata
    ├── csv/ (analysis outputs, metrics, diagnostics)
    └── visualizations/ (diagnostic plots)
```

## References

Key literature supporting feature selection and methodology documented in project documentation.

**Volatility Modeling and HAR-RV:**
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Fleming, J., Ostdiek, B., & Whaley, R. E. (2001). Predicting Stock Market Volatility: A New Measure. *Journal of Futures Markets*, 21(3), 267-287.

**Jump Detection:**
- Lee, S. S., & Mykland, P. A. (2008). Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics. *Review of Financial Studies*, 21(6), 2543-2577.

**On-Chain Metrics:**
- Yang, K., & Fantazzini, D. (2022). NVRV vs. MVRV Comparison for Cryptocurrency Analysis.

**Machine Learning for Volatility Forecasting:**
- Vrontos, I. et al. (2021). Forecasting VIX with Machine Learning. *Journal of Forecasting*.
- Balaneji, B., & Maringer, D. (2022). Implied Volatility Forecasting with XGBoost. *Quantitative Finance*.
- Zhang, L., & Hua, L. (2025). High-Frequency Financial Data Analysis: A Survey. *Mathematics*, 13(3), 347.