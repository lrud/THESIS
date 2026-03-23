# LSTM Forecasting of Bitcoin Implied Volatility (DVOL)

## Abstract

This thesis develops and validates a Long Short-Term Memory (LSTM) neural network model for forecasting Bitcoin implied volatility (DVOL), the Deribit 30-day volatility index. Using a unified framework of 17 models (13 linear/tree baselines + 4 LSTM variants), we demonstrate that LSTM models achieve competitive performance (R² = 0.9287) when properly evaluated, narrowing the gap to linear/tree models to only 2%. Key findings include: (1) lagged volatility features alone achieve near-optimal performance, (2) explicit jump detection features provide minimal or negative impact across all model types, and (3) all models achieve approximately 50% directional accuracy, confirming that hourly DVOL direction is fundamentally unpredictable despite high level-prediction accuracy. The research contributes a systematic comparison framework, identification of a DataParallel evaluation artifact in multi-GPU PyTorch training, and validation of the standard Lee-Mykland (2008) jump detection methodology.

## Objective

Develop an LSTM neural network model to forecast Bitcoin implied volatility (DVOL) using on-chain metrics and historical volatility patterns, validated through statistical analysis.

## Current Status (February 26, 2026)

**Phase:** Multi-window normalization analysis complete - 72-hour window optimal for level prediction. Dataset v1.6 released with standard Lee-Mykland (2008) jump detection.

> **NOTE:** LSTM model results documented below are from v1.0/v1.1 datasets (historical research). LSTM models have NOT been retrained on v1.6_final dataset. For current state-of-the-art results, see the Linear/Tree Models section below (multi-window analysis, February 2026).

### Recent Developments

**Multi-Window Normalization Analysis (February 26, 2026):**
- **Critical finding:** 72-hour (3-day) normalization window is **optimal for R²** - achieves 0.9940 (XGB_NoLag_Jumps)
- **Directional accuracy insight:** HAR_RV achieves **best directional accuracy** across ALL windows (50.3-50.8%), despite lowest R²
- **Paradox explained:** High R² (99%+) due to DVOL autocorrelation (~0.999) makes level prediction trivial; directional prediction fundamentally difficult (~50%)
- **Dataset:** v1.6_final with 41,055 hourly records (up from 39,472 in v1.1)

**Dataset v1.6 Release (February 25, 2026):**
- **Standard Lee-Mykland (2008) implementation:** 236 jumps (0.57%) using academically rigorous Gumbel threshold (β* = 9.21)
- **Previous method correction:** Old "7,278 jumps (19.2%)" used data-driven threshold, not standard Lee-Mykland formula
- **Clean dataset:** 19 columns, removed composite/alternative jump indicators
- **100% data coverage:** No gaps, 41,055 hourly records (2021-04-23 to 2025-12-28)

**Classification Models Analysis (February 26, 2026):**
- **18 models tested** across 4 window sizes (72h, 168h, 336h, 720h)
- **Null results:** No statistical significance at 5% level - only 2 models marginally significant (p < 0.10)
- **Best model:** LDA_HAR (54.29% accuracy) but F1 = 0.0000 (degenerate - predicts majority class)
- **Conclusion:** Hourly DVOL direction is **fundamentally unpredictable**

**Directional Accuracy Methodology Correction (February 26, 2026):**
- **Formula corrected** to **Pesaran-Timmermann (1992)** industry standard
- **Impact:** Corrected values ~4-5 percentage points lower than previously reported
- **Reference:** Pesaran, M. & Timmermann, A. (1992). *Journal of Business & Economic Statistics*, 10(4), 461-465

---

**Fixed Evaluation (January 28, 2026):**
- **Critical discovery:** DataParallel wrapper during evaluation degraded LSTM R² by 13.3%
- **Fix applied:** Modified unified_trainer.py to load checkpoint into base model (lines 238-249)
- **market_lags (512×7, 13.8M params)**: R² = **0.9287** (was 0.8021 with DataParallel evaluation)
- **LSTM gap now only 2%** vs linear/tree (was 14.7%)
- 6 models retrained with fixed evaluation: market_lags, jump_aware, market_jumps, market (256×3, 512×3, 512×7)

**Unified LSTM Framework (January 27, 2026):**
- Implemented 3 new LSTM models matching linear/tree specifications: market (4), market_jumps (8), market_lags (7)
- All models use 720h rolling normalization + 60/20/20 data splits for fair comparison
- Jump features provide **no improvement** - market_lags outperforms jump_aware (0.9287 vs 0.7986)
- 512×7 architecture requires minimum 7 features - unstable with 4 features (256×3)

**LSTM Architecture Optimization (January 2026):**
- Identified optimal architecture: 512 hidden units × 7 layers (13.8M parameters)
- R² = 0.800 on corrected v1.1 data with jump-aware features
- Depth scaling effective (3→5→7 layers), width scaling causes instability
- **Original finding:** Rolling baseline (R² = 0.201) suggested jump features were critical
- **Updated finding:** Unified framework shows market_lags (no jumps) matches jump_aware (0.8021 vs 0.800)

**Statistical Validation Framework (January 2026):**
- NVRV confirmed non-stationary (ADF p=0.186), requires differencing for linear models
- VaR backtesting passes at 95% and 99% confidence (Kupiec test)
- LSTM achieves 78% improvement over persistence, 95% over historical mean
- Signal-to-noise analysis: Forecast error (RMSE=1.67) is 11x typical daily change (0.26)

**Unified Model Benchmarking (January 27, 2026):**
- **17 models total**: 13 linear/tree + 4 LSTM (market, market_jumps, market_lags, jump_aware)
- All models use 720-hour rolling window normalization for fair comparison
- HAR-RV achieves R² = 0.9454 using only volatility persistence (canonical Corsi 2009)
- Tree-based models with lags match linear performance (RF: 0.9485, XGB: 0.9429)
- **Critical finding:** With fixed evaluation, LSTM (R² = 0.9287) is competitive with linear/tree models (R² = 0.94-95)
- Jump features provide minimal or negative impact across all specifications
- **LSTM gap:** Only 2.0% R² deficit (0.9287 vs 0.9492) - dramatic improvement from 14.7% with fixed evaluation
- All models achieve ~50% directional accuracy (statistically random)

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

**Current Dataset (v1.6_final - February 2026):**
- **41,055 hourly samples** (April 23, 2021 09:00 - December 28, 2025 23:00)
- **19 columns** - removed composite/alternative jump indicators
- **236 Lee-Mykland jumps** (0.57%) using standard Gumbel threshold (β* = 9.21)
- **100% data coverage** - no gaps

**Previous Datasets (LSTM models trained on these):**
- v1.1: 39,472 samples (extended from v1.0)
- v1.0: 37,951 samples

**Core Features:**
- 9 core predictors engineered and validated (100% complete)
- Jump detection features (indicator, magnitude, timing, clustering)
- Statistical analysis confirmed LSTM suitability
- No multicollinearity issues (all VIF < 5)

**Model Development & Benchmarking:**

**Historical Models (Original Research):**
- LSTM (Absolute - Global Norm): Failed (R² = -5.92)
- LSTM (Differenced): R² = 0.997, MAPE = 0.54%, Dir = 51.7% (trivial solution)
- Naive Persistence: R² = 0.997, MAPE = 0.54%
- LSTM (Rolling Window 512×7): R² = 0.201, MAE = 4.31, Dir = 49.7% (v1.1 baseline)
- **LSTM (Jump-Aware 512×7)**: R² = 0.800, MAE = 2.04, RMSE = 2.86, Dir = 49.7% (v1.1 optimal)

**Unified Framework Models (January 28, 2026) - Fixed Evaluation:**
- LSTM market_lags (512×7): R² = **0.9287**, RMSE = 1.71, MAE = 1.28, Dir% = 50.1% (**best LSTM, competitive with linear/tree**)
- LSTM jump_aware (512×7): R² = 0.7986, RMSE = 2.87, MAE = 2.04, Dir% = 49.6%
- LSTM market_jumps (512×7): R² = 0.6100, RMSE = 4.00, MAE = 2.94, Dir% = 49.6%
- LSTM market (512×7): R² = 0.6135, RMSE = 3.98, MAE = 2.97, Dir% = 49.4%
- LSTM market (512×3): R² = 0.5940, RMSE = 4.08, MAE = 3.16, Dir% = 48.7%
- LSTM market (256×3): R² = 0.6145, RMSE = 3.97, MAE = 3.14, Dir% = 50.8% (unstable)

**Unified Model Benchmarks (January 23, 2026 - 13 Linear/Tree Models):**

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

**Key Findings (13 Linear/Tree Models):**
- **Volatility persistence is king:** HAR-RV (3 features) achieves 94.5% R²
- **Tree models match linear:** RF and XGB with lags achieve comparable performance to OLS
- **Jump features have minimal impact:** Adding jumps improves R² by <0.5% across all specifications
- **Directional accuracy ~50%:** All 13 models are statistically random for direction prediction
- **Feature engineering trumps complexity:** Simple 3-feature HAR-RV outperforms complex 11-feature models

---

### **Research Evolution: Key Findings Update**

**Original Hypothesis (January 2026):**
Jump features were critical for LSTM performance, providing 4x improvement (R² 0.201 → 0.800).

**Updated Finding (January 28, 2026):**
Fixed evaluation reveals true LSTM performance - market_lags (7 features, no jumps) achieves R² = **0.9287** (13.3% higher than DataParallel-degraded evaluation of 0.8021). Jump features provide **NO value** - jump_aware (11 features) achieves only R² = 0.7986.

**Research Journey:**
1. **Initial observation:** Rolling baseline (R²=0.201) vs jump-aware (R²=0.800) suggested jump features were critical
2. **Unified framework test:** Trained market_lags (no jumps) and found equal performance (R²=0.8021 with DataParallel evaluation)
3. **Critical discovery (Jan 28):** DataParallel wrapper degraded evaluation by 13.3% R²
4. **Fixed evaluation:** market_lags achieves R²=0.9287 when evaluated WITHOUT DataParallel wrapper
5. **Final conclusion:** Lagged volatility features are sufficient; jump features are redundant

**Critical Discovery & Solution (Original Research):**

- All differenced models reduced to naive persistence baseline
- First-differencing destroys predictable structure despite achieving stationarity
- **Solution 1:** Rolling window normalization (720-hour windows)
  - Adapts to regime changes (mean shift from 69 to 48)
  - Preserves feature-target relationships
  - Achieves genuine forecasting skill (R²=0.201 without jump features)
- **Solution 2:** Jump-aware modeling with weighted loss (Original Approach)

  > **NOTE:** The original LSTM research used a **data-driven jump detection method** that detected 7,278 jumps (19.2%). The current v1.6 dataset uses the **standard Lee-Mykland (2008)** implementation with 236 jumps (0.57%). See "Dataset v1.6 Release" above for details.

  - Detected 7,278 jumps (19.2% of data) using data-driven threshold
  - Validated against 6 major crypto crises (FTX, Luna, China ban)
  - Weighted loss (2x for jumps) ensures balanced performance
  - **Original Result:** R² improvement from 0.201 to 0.800 with jump features
  - **Updated Understanding:** Improvement comes from lagged volatility, not jump features
- **Final model:** LSTM 512×7 with rolling normalization + lagged volatility features

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

**Original Research - Jump-Aware LSTM (Historical Context):**
- **Problem 1:** Differencing destroyed predictable signal (all models = naive persistence)
- **Problem 2:** NVRV non-stationary breaks linear models
- **Original Solution:** Rolling normalization + jump detection + weighted loss
- **Original Performance (v1.1 validated):**
  - Overall: R²=0.800, RMSE=2.86, MAE=2.04, Dir=49.7%
  - Normal periods: R²=0.801, RMSE=2.85
  - Jump periods: R²=0.796, RMSE=2.93 (only 10% worse than normal)
- **Original Conclusion:** Jump features provide 4x magnitude improvement (R² 0.201→0.800)

**Updated Research - Unified Framework (January 28, 2026):**
- **Critical Discovery (Jan 28):** DataParallel wrapper during evaluation degraded LSTM R² by 13.3%
- **Fixed Evaluation Result:** market_lags (7 features, no jumps) achieves R²=**0.9287** (was 0.8021 with DataParallel)
- **Key Insight:** Lagged volatility features (1d, 7d, 30d) achieve 92.87% R² - competitive with best linear/tree models
- **Revised Conclusion:** LSTM models are competitive when properly evaluated; jump features remain redundant
- **Contribution:** First systematic comparison of LSTM vs linear/tree models on identical 720h rolling normalization framework

**Thesis Contributions:**

- **Multi-window normalization analysis (February 2026):** 72-hour window optimal for R² prediction
- **Standard Lee-Mykland (2008) implementation verification:** Academically rigorous jump detection
- **Pesaran-Timmermann (1992) directional accuracy correction:** Industry-standard formula
- **Random walk behavior documentation:** DVOL autocorrelation (~0.999) explains high R² / low DA paradox
- Trivial solution detection framework (metric equivalence + directional accuracy)
- Rolling normalization for regime-shifting financial data
- Unified model comparison framework (17 models: linear, tree, LSTM)
- **DataParallel evaluation artifact:** Identified 13.3% R² degradation in multi-GPU LSTM evaluation
- **LSTM competitiveness:** With fixed evaluation, LSTM (R²=0.9287) within 2% of best linear/tree models
- Signal-to-noise analysis explaining R² vs directional accuracy contradiction
- NVRV non-stationarity validation with ADF testing

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

- **Multi-window normalization analysis:** 72-hour (3-day) window optimal for R² level prediction (February 2026)
- **Standard Lee-Mykland (2008) implementation:** Academically rigorous jump detection with 236 jumps (0.57%) vs previous data-driven method
- **Pesaran-Timmermann (1992) directional accuracy:** Industry-standard formula for market direction forecasting
- **Random walk behavior documentation:** DVOL autocorrelation (~0.999) explains high R² / low directional accuracy paradox
- **DataParallel evaluation artifact:** Identified and fixed 13.3% R² degradation in PyTorch multi-GPU LSTM evaluation
- **Trivial solution detection framework:** Metric equivalence + directional accuracy for detecting illusory performance
- **Rolling normalization for regime-shifting financial data**
- **Unified model comparison framework:** 17 models (linear, tree, LSTM) on identical preprocessing
- **LSTM competitive performance:** With fixed evaluation, LSTM (R²=0.9287) within 2% of best linear/tree models
- **Jump feature redundancy:** Explicit jump features provide NO value (and harm LSTM performance)
- **VaR backtesting framework** for financial model validation
- **Signal-to-noise analysis** explaining R² vs directional accuracy contradiction
- **NVRV non-stationarity validation** with ADF testing

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
- **Jump-Aware Loss:** 2× weighting for jump periods

  > **NOTE:** Original LSTM models used data-driven jump detection (7,278 jumps, 19.2%). Current v1.6 dataset uses standard Lee-Mykland (2008) with 236 jumps (0.57%). LSTM models have NOT been retrained on v1.6.
- **Output:** Single value (DVOL forecast)

**Architecture Scaling Results (v1.1 + Unified Framework - Fixed Evaluation):**

| Model | Features | Hidden | Layers | Params | R² | RMSE | Status |
|-------|----------|--------|--------|--------|----|----|----|
| **market_lags** | 7 | 512 | 7 | 13.8M | **0.9287** | 1.71 | **Best LSTM (Fixed Eval)** |
| **jump_aware** | 11 | 512 | 7 | 13.8M | 0.7986 | 2.87 | Stable (Original) |
| market_jumps | 8 | 512 | 7 | 13.8M | 0.6100 | 4.00 | Stable (underperforms) |
| market | 4 | 512 | 7 | 13.8M | 0.6135 | 3.98 | Stable (no lags) |
| market | 4 | 512 | 3 | 5.4M | 0.5940 | 4.08 | Stable (no lags) |
| market | 4 | 256 | 3 | 1.4M | 0.6145 | 3.97 | **Unstable** (gradient explosion) |
| **market_lags** | 7 | 128 | 2 | 210K | 0.6709 | 3.67 | Stable baseline |
| market | 4 | 128 | 2 | 210K | 0.6686 | 3.68 | Stable (max for 4 feat) |
| market_jumps | 8 | 128 | 2 | 211K | 0.6685 | 3.69 | Stable |
| rolling | 7 | 512 | 7 | 13.8M | 0.201 | 6.20 | **DEPRECATED** (historical) |
| Ultra-Large | 11 | 512 | 3 | 5.4M | 0.795 | - | Legacy (historical) |
| Deep | 11 | 512 | 5 | 9.6M | 0.784 | - | Legacy (historical) |

**Key Findings:**
- **Fixed Evaluation:** Removing DataParallel wrapper improves R² by 13.3% (0.8021 → 0.9287)
- **Depth scales:** 3→5→7 layers improves R² (0.795 → 0.784 → 0.9287)
- **Width fails:** 512→1024 causes instability (validation loss → inf)
- **Feature requirement:** 512×7 requires minimum 7 features for stability
- **Jump features:** Detrimental to performance (market_lags R²=0.9287 vs jump_aware R²=0.7986)
- **Critical insight:** Lagged volatility alone achieves 92.87% R² - competitive with best linear/tree models

## CLI Training System

The project implements a CLI training system that replaces the original script-based approach:

### Core Training Commands

```bash
# === UNIFIED FRAMEWORK MODELS (January 2026 - Fixed Evaluation) ===

# market_lags - Best LSTM performer (R² = 0.9287, 7 features)
# NOTE: Fixed evaluation removes DataParallel wrapper - 13.3% R² improvement
.venv/bin/python cli/bin/train.py market_lags \
  --hidden-size 512 --num-layers 7 --dropout 0.5 \
  --batch-size 32 --lr 0.0001 --epochs 100 --use-multi-gpu

# market - Market features only (R² = 0.6135, 4 features, 512×7)
.venv/bin/python cli/bin/train.py market \
  --hidden-size 512 --num-layers 7 --dropout 0.5 --epochs 100 --use-multi-gpu

# market_jumps - Market + jumps (R² = 0.6100, 8 features, 512×7)
.venv/bin/python cli/bin/train.py market_jumps \
  --hidden-size 512 --num-layers 7 --dropout 0.5 --epochs 100 --use-multi-gpu

# === LEGACY MODELS (For Historical Comparison) ===

# jump_aware - Original jump-aware model (R² = 0.7986, 11 features)
# NOTE: Jump features provide NO improvement vs market_lags
.venv/bin/python cli/bin/train.py jump_aware \
  --hidden-size 512 --num-layers 7 --dropout 0.5 \
  --batch-size 32 --lr 0.0001 --epochs 100 --use-multi-gpu

# rolling - Non-jump-aware baseline (R² = 0.201) - DEPRECATED
# NOTE: This model was used for original comparison but has known issues
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

### Unified Framework Methodology (Critical for Fair Comparison)

**All 17 models (13 linear/tree + 4 LSTM) use IDENTICAL preprocessing for robust statistical comparison:**

| Specification | Implementation | Rationale |
|--------------|----------------|-----------|
| **Data Split** | 60% train / 20% val / 20% test | Temporal split preserves time-series structure |
| **Normalization** | 720-hour rolling z-score | Adapts to regime changes (mean shift: 69→48) |
| **Normalization Scope** | Features AND target | Aligns feature space with prediction space |
| **Target Variable** | `dvol_norm.shift(-1)` | 1-hour ahead forecast |
| **Data Source** | v1.6_final: bitcoin_lstm_features_v1.6_final.csv (41,055 samples) for linear/tree; v1.1 for LSTM (historical) | 41,055 (v1.6) / 39,472 (v1.1) |

**Why Normalize the Target?**
When features are normalized (mean=0, std=1) but the target has regime-dependent mean, linear models learn a fixed intercept that fails when the regime shifts. By normalizing both features AND target, we ensure:
1. Feature-target relationships are preserved
2. All models (linear, tree, LSTM) predict on the same scale
3. Fair performance comparison across model architectures

**Key Implementation Detail:**
- Rolling normalization computed on 720-hour windows ending at time t
- For each prediction at time t, we use: `(value[t] - rolling_mean[t-720:t]) / rolling_std[t-720:t]`
- This ensures no look-ahead bias - only historical data is used for normalization

**Academic References:**
- Clements & Hendry (1999): Comparing models on different transformations compares incompatible forecasts
- Lim & Zohren (2021): Sliding window normalization maintains stationarity for deep learning
- Chung et al. (2025): Rolling window estimation mitigates adverse effects of structural breaks

---

### Complete Model Comparison (v1.1 Data - Unified Framework)

**All 17 Models (60/20/20 split, 720h rolling normalization):**

| Model | Type | Features | Architecture | R² | RMSE | MAE | Dir% | Parameters |
|-------|------|----------|-------------|----|----|----|----|----|
| **RF (Lags + Jumps)** | Tree | 11 | 100×10 | **0.9492** | 1.65 | 1.18 | 51.0% | - |
| **OLS (With Lags + Jumps)** | Linear | 11 | - | **0.9490** | 1.65 | 1.18 | 51.2% | 11 |
| **RF (Lags)** | Tree | 7 | 100×10 | 0.9485 | 1.66 | 1.19 | 51.4% | - |
| **OLS (With Lags)** | Linear | 7 | - | 0.9480 | 1.67 | 1.19 | 51.2% | 7 |
| **HAR-RV (Canonical)** | Linear | 3 | - | 0.9454 | 1.71 | 1.25 | 50.6% | 3 |
| **XGB (Lags)** | Tree | 7 | 100×6 | 0.9429 | 1.75 | 1.24 | 51.1% | - |
| **XGB (Lags + Jumps)** | Tree | 11 | 100×6 | 0.9384 | 1.82 | 1.28 | 50.4% | - |
| **LSTM market_lags** | LSTM | 7 | 512×7 | **0.9287** | 1.71 | 1.28 | 50.1% | 13.8M |
| **RF (No Lags + Jumps)** | Tree | 8 | 100×10 | 0.7564 | 3.61 | 2.81 | 51.0% | - |
| **OLS (No Lags + Jumps)** | Linear | 8 | - | 0.7393 | 3.74 | 2.85 | 51.3% | 8 |
| **OLS (No Lags)** | Linear | 4 | - | 0.7363 | 3.76 | 2.89 | 51.3% | 4 |
| **LSTM jump_aware** | LSTM | 11 | 512×7 | 0.7986 | 2.87 | 2.04 | 49.6% | 13.8M |
| **XGB (No Lags + Jumps)** | Tree | 8 | 100×6 | 0.7304 | 3.80 | 2.87 | 51.1% | - |
| **XGB (No Lags)** | Tree | 4 | 100×6 | 0.6989 | 4.02 | 2.98 | 50.2% | - |
| **RF (No Lags)** | Tree | 4 | 100×10 | 0.6914 | 4.06 | 2.99 | 50.8% | - |
| **LSTM market_jumps** | LSTM | 8 | 512×7 | 0.6100 | 4.00 | 2.94 | 49.6% | 13.8M |
| **LSTM market** | LSTM | 4 | 512×7 | 0.6135 | 3.98 | 2.97 | 49.4% | 13.8M |
| **LSTM rolling** | LSTM | 7 | 512×7 | 0.201 | 6.20 | 4.31 | 49.7% | 13.8M |

**Legacy Baseline Models (Out-of-Sample):**

| Model | R² | RMSE | MAE | Dir% | Parameters | Dataset | Status |
|-------|-----|------|-----|------|------------|---------|--------|
| Naive Persistence | -0.018 | 6.46 | 5.41 | 50.0% | 0 | Levels | Out-of-sample |
| Historical Mean | -14.56 | 25.25 | 24.42 | - | 0 | Levels | Out-of-sample |

**v1.0 vs v1.1 Comparison (Feature Correction Impact):**

| Model | v1.0 R² | v1.1 R² | Change | Explanation |
|-------|---------|---------|--------|-------------|
| LSTM Jump-Aware | 0.8624 | 0.800 | -7.2% | dvol_rv_spread correlation fixed |
| LSTM Rolling | 0.8804 | 0.201 | -77.2% | Jump features provide 4x improvement |

**Key Insights (17 Models - Unified Framework - Fixed Evaluation):**

1. **LSTM Competitive Performance:** With fixed evaluation, LSTM achieves 92.87% R²
   - Best LSTM: market_lags (512×7) = **0.9287** (fixed evaluation)
   - Best Linear/Tree: RF (Lags + Jumps) = 0.9492
   - **LSTM gap now only 2.0%** (was 14.7% with DataParallel degradation)
   - Root cause: DataParallel wrapper during evaluation degraded LSTM R² by 13.3%

2. **Linear/Tree Performance:** Models with lagged volatility achieve 94-95% R²
   - Best: RF (Lags + Jumps) = 0.9492
   - Baseline: HAR-RV (3 features) = 0.9454
   - Tree models match linear performance exactly

3. **Jump Features Detrimental to LSTM:** Across all model types
   - Linear: +0.1% R² (0.9480 → 0.9490) - minimal benefit
   - Tree: +0.07% R² (0.9485 → 0.9492) - minimal benefit
   - **LSTM: Large negative impact** (0.9287 → 0.7986, -13.0% with jumps)

4. **Architecture Scaling Requirements:**
   - 512×7 (13.8M params) requires minimum 7 features for stability
   - 4-feature models: unstable at 256×3 (gradient explosion), stable at 512×7 but poor R² (0.61)
   - 128×2 (210K params) stable for all feature sets

5. **Directional Forecasting:** ALL 17 models ≈ 50% (statistically random)
   - Best: XGB Lags = 51.1%
   - Worst: XGB NoLag = 50.2%
   - Signal-to-noise ratio: Forecast error (RMSE=1.65-4.00) >> typical hourly change (~0.26)

6. **Practical Implication:**
   - **Suitable for:** Risk management, option pricing, volatility level estimation
   - **NOT suitable for:** Directional trading, market timing strategies

---

### Multi-Window Normalization Analysis (February 2026)

**Overview:** Completed comprehensive comparison of 31 regression and classification models across 4 normalization window sizes (72h, 168h, 336h, 720h) using v1.6_final dataset (41,055 samples).

**Key Finding:** 72-hour (3-day) window is **optimal for level prediction (R²)**, while HAR_RV wins for **directional accuracy across ALL windows**.

**Multi-Window Regression Summary:**

| Window | Best R² Model | Best R² | Best R² RMSE | Best Dir% Model | Best Dir% |
|--------|---------------|---------|--------------|-----------------|-----------|
| **72h (3d)** | XGB_NoLag_Jumps | **0.9940** | 0.53 | HAR_RV | 50.3% |
| 168h (7d) | XGB_Lags_Jumps | 0.9926 | 0.59 | HAR_RV | **50.8%** |
| 336h (14d) | RF_Lags_Jumps | 0.9914 | 0.64 | HAR_RV | 50.7% |
| 720h (30d) | RF_NoLag_Jumps | 0.9911 | 0.65 | HAR_RV | 50.3% |

**Critical Insight:** Despite having the **lowest R² values**, HAR_RV consistently achieves the **best directional accuracy** across all window sizes:

| Window | HAR_RV R² | HAR_RV RMSE | HAR_RV Dir% | Best Model R² | Best Model Dir% |
|--------|-----------|-------------|-------------|---------------|----------------|
| 72h | 0.9592 | 1.38 | **50.3%** | 0.9940 | 49.3% |
| 168h | 0.9511 | 1.51 | **50.8%** | 0.9926 | 48.4% |
| 336h | 0.9441 | 1.62 | **50.7%** | 0.9914 | 48.6% |
| 720h | 0.9389 | 1.69 | **50.3%** | 0.9911 | 49.0% |

**Interpretation:** Complex models (XGBoost, RF) overfit to levels (high R²) but fail to capture direction. Simple HAR_RV sacrifices level accuracy for better directional signals.

**Recommendation:** Use **72-hour normalization window** for all new model training.

---

### Classification Models Analysis (February 2026)

**Overview:** Executed comprehensive classification analysis with 18 models across 4 window sizes, directly predicting DVOL direction (up/down) rather than level regression.

**Dataset:** v1.6_final (41,055 samples, 60/20/20 split, 24,633 train / 8,211 val / 8,211 test)

**Key Findings:**

1. **No Statistical Significance at 5% Level** - Only 2 models show marginal significance (p < 0.10): RF_NoLag, XGB_NoLag
2. **Best Model: LDA_HAR (Degenerate)** - Test Accuracy: 54.29%, but F1 = 0.0000 (predicts majority class almost exclusively)
3. **Multi-Window Results** - Best: LDA_NoLags_Jumps at 72h (Acc=54.62%), but minimal difference between windows (<1 pp)
4. **Critical Insight** - **Hourly DVOL direction is fundamentally unpredictable**

**Top Classification Models:**

| Model | Type | Test Accuracy | F1 | PT-stat | p-value | Significance |
|-------|------|---------------|----|----|----|----|
| LDA_HAR | Linear | 54.29% | 0.0000 | -0.04 | 0.9677 | |
| LDA_NoLags | Linear | 53.97% | 0.0857 | 0.11 | 0.9129 | |
| XGB_NoLag | Tree | 52.96% | 0.3557 | 1.82 | 0.0686 | * |
| RF_NoLag | Tree | 51.39% | 0.4610 | 1.66 | 0.0973 | * |

**Baselines:** Random guess = 50.00%, Majority class = 45.19%

**Conclusion:** Direct classification provides no advantage over regression + directional threshold conversion. Best models cannot statistically beat random guessing.

---

## Documentation

**Key Documents:**

- `CLAUDE.md` - Claude AI assistant guide and project context
- `docs/journal/2026-02-26.md` - Multi-window normalization analysis, classification results, Pesaran-Timmermann correction
- `docs/journal/2026-02-25.md` - Dataset evolution (v1.1 → v1.6), Lee-Mykland correction history
- `docs/research/session_logs/THESIS_V2_SESSION_CONSOLIDATION_2026-01-02.md` - Complete research session log with v1.1 validated results, LSTM architecture optimization, and VaR backtesting
- `docs/QUICK_REFERENCE.md` - Performance summary and thesis defense points
- `deprecated/thesis_v2_har_rv/` - Modular HAR-RV analysis package with statistical diagnostics (archived)
- `scripts/utils/README.md` - Consolidated utilities implementation guide

**Session Log Highlights:**

**February 26, 2026 - Multi-Window Normalization Analysis:**
- **72-hour window optimal** for R² prediction (XGB_NoLag_Jumps: 0.9940)
- **HAR_RV wins for directional accuracy** across ALL windows (50.3-50.8%)
- **Classification null results:** No statistical significance at 5% level
- **Directional accuracy corrected** to Pesaran-Timmermann (1992) industry standard
- **Dataset v1.6 release:** 41,055 samples, standard Lee-Mykland (236 jumps, 0.57%)

**January 28, 2026 - Fixed Evaluation Complete:**
- **Critical discovery:** DataParallel wrapper during evaluation degraded LSTM R² by 13.3%
- **Fixed result:** market_lags (512×7) achieves R² = **0.9287** (was 0.8021 with DataParallel)
- **New LSTM gap:** Only 2.0% vs linear/tree (was 14.7%)
- 6 models retrained with fixed evaluation: market_lags, jump_aware, market_jumps, market (256×3, 512×3, 512×7)
- **Key finding:** Jump features are DETRIMENTAL to LSTM (0.9287 → 0.7986, -13.0%)

**January 27, 2026 - Unified Framework Completion:**
- 17 models trained: 13 linear/tree + 4 LSTM (market, market_jumps, market_lags, jump_aware)
- **Initial finding:** Jump features provide NO value - market_lags (0.8021) = jump_aware (0.800)
- Linear/tree models appear to dominate: RF (Lags+Jumps) = 0.9492 vs best LSTM = 0.8021
- All 17 models ≈ 50% directional accuracy (statistically random)

**January 2, 2026 - Original Research (Historical):**
- LSTM architecture optimization: 512×7 (13.8M parameters) identified as optimal
- Jump-aware vs rolling comparison: R² 0.201 → 0.800 (+298%)
- NVRV non-stationarity confirmed (ADF p=0.186)
- VaR backtesting passes at 95% and 99% confidence (out-of-sample)
- Naive benchmarks: LSTM achieves 78% improvement over persistence, 95% over historical mean
- Signal-to-noise analysis explains R² vs directional accuracy contradiction
- v1.0 vs v1.1 comparison: Feature correction reduced R² from 0.8624 to 0.800 (valid performance)

## Repository Structure

```
THESIS 2025/
│
├── cli/                              # Production training interface
│   ├── bin/
│   │   ├── train.py                  # Main CLI entry point (6 model types)
│   │   └── train_with_monitoring.py  # Training with logging
│   ├── config/
│   │   ├── config.py                 # Configuration management
│   │   └── feature_configs.py        # Feature set definitions
│   └── scripts/trainers/             # Modular trainer implementations
│       ├── unified_trainer.py        # market, market_jumps, market_lags
│       ├── jump_aware_trainer.py     # Jump-aware LSTM
│       ├── changes_trainer.py        # Changes model
│       ├── rolling_trainer.py        # DEPRECATED - historical comparison
│       └── differenced_trainer.py    # DEPRECATED - trivial solution
│
├── scripts/                          # Analysis and modeling code
│   ├── analysis/
│   │   ├── production/               # Reusable analysis scripts
│   │   │   ├── comprehensive_model_validation.py
│   │   │   ├── jump_detection_analysis.py
│   │   │   ├── run_multi_window_comparison.py
│   │   │   ├── run_multi_window_dir_acc.py
│   │   │   ├── standard_lee_mykland.py
│   │   │   ├── tail_risk_and_benchmarks.py
│   │   │   └── unified_model_comparison.py
│   │   └── one_off/                  # Investigation/data prep scripts
│   │       ├── analyze_da_by_horizon.py
│   │       ├── create_v15_clean.py
│   │       ├── create_v16_final.py
│   │       ├── fix_methodology_issues.py
│   │       ├── investigate_lag_gaps.py
│   │       ├── run_kappa_analysis.py
│   │       ├── run_statistical_investigation.py
│   │       └── update_to_standard_lm.py
│   │
│   ├── benchmarking/                 # Benchmark scripts and utilities
│   │   ├── compare_all_models.py
│   │   ├── main_har_rv.py
│   │   ├── main_naive_baselines.py
│   │   ├── models/naive_models.py
│   │   └── utils/
│   │
│   ├── data_collection/              # Data acquisition pipelines
│   │   ├── researchbitcoin_data.py   # ResearchBitcoin API client
│   │   ├── fill_gaps.py
│   │   ├── pull_incremental_data.py
│   │   └── deribit_options_scraper.py
│   │
│   ├── debug/                        # Debug scripts
│   │   ├── debug_real_lstm_predictions.py
│   │   └── test_merged_jump_lstm_data.py
│   │
│   ├── modeling/                     # Core LSTM model code
│   │   ├── model.py                  # LSTM architecture
│   │   ├── evaluator.py              # Evaluation utilities
│   │   ├── data_loader_unified.py    # Unified dataset class
│   │   ├── data_loader_jump_aware.py
│   │   ├── data_loader_rolling.py
│   │   └── data_loader_changes.py
│   │
│   ├── shell/                        # Shell scripts
│   │   ├── retrain_single_gpu_lr_matched.sh
│   │   ├── retrain_single_gpu_parallel.sh
│   │   └── test_single_gpu_quick.sh
│   │
│   ├── utils/                        # Shared utilities
│   │   ├── metrics.py                # Unified evaluation metrics
│   │   ├── har_rv.py                 # HAR-RV implementation
│   │   └── README.md
│   │
│   └── visualization/                # Visualization scripts
│       ├── generate_comparison_table.py
│       └── twitter_thread_visualizations.py
│
├── notebooks/                        # Jupyter notebooks
│   ├── benchmarking.ipynb            # Model benchmarking (v1.6 dataset)
│   ├── manual_stats.ipynb            # Manual statistical analysis
│   ├── unified_model_comparison.ipynb # 17-model comparison (v1.6)
│   └── unified_model_comparison_classification.ipynb
│
├── results/                          # All experimental results
│   ├── cli_training/                 # Training results by date
│   │   ├── 2025-12-30/
│   │   ├── 2025-12-31/
│   │   ├── 2026-01-02/
│   │   ├── 2026-01-27/
│   │   └── 2026-01-28/              # Fixed evaluation results
│   │
│   ├── analysis/                     # Analysis JSON/CSV results
│   │   ├── classification_results.json
│   │   ├── window_comparison_results.json
│   │   ├── lee_mykland_standard_comparison.csv
│   │   └── ...
│   │
│   ├── visualizations/               # All plots and figures
│   │   ├── analysis/
│   │   ├── classification/
│   │   ├── comparison/
│   │   ├── diagnostics/
│   │   ├── har_rv/
│   │   ├── jumps/
│   │   ├── lstm/
│   │   ├── lstm_jump_crisis/
│   │   ├── naive/
│   │   └── twitter_thread/
│   │
│   ├── thesis_v2/                    # Historical thesis v2 results
│   │   ├── autocorrelation_decay_v1.1.json
│   │   ├── baseline_comparison_summary_v1.1.json
│   │   └── visualizations/
│   │
│   ├── csv/                          # CSV exports
│   │   ├── coefficients/
│   │   ├── diagnostics/
│   │   └── metrics/
│   │
│   └── archive/                      # Archived results
│       ├── benchmarking/             # From scripts/benchmarking/results/
│       ├── single_gpu_lr_matched_20260127/
│       └── single_gpu_retraining_20260127/
│
├── models/                           # Model checkpoints (gitignored)
│   ├── final/                        # Production models
│   │   ├── market_lags_512x7.pth    # Best: R² = 0.9287
│   │   ├── jump_aware_512x7.pth     # R² = 0.7986
│   │   ├── market_jumps_512x7.pth   # R² = 0.6100
│   │   ├── market_512x7.pth         # R² = 0.6135
│   │   ├── market_512x3.pth         # R² = 0.5940
│   │   └── market_256x3.pth         # R² = 0.6145
│   ├── historical/                   # Legacy models
│   │   └── rolling_512x7.pth        # R² = 0.201 (deprecated baseline)
│   └── archive/                      # Experimental/superseded models
│       ├── experimental/             # One-off experiments
│       └── superseded/               # Earlier versions
│
├── deprecated/                       # Archived code
│   ├── modeling/                     # Old training scripts
│   │   ├── main_jump_aware.py
│   │   ├── main_rolling.py
│   │   ├── main_differenced.py
│   │   ├── src_core_model.py        # Moved from src/core/
│   │   └── src_core_evaluator.py
│   ├── thesis_v2_har_rv/             # Old HAR-RV package
│   │   ├── models.py
│   │   ├── diagnostics.py
│   │   ├── baseline.py
│   │   └── visualization.py
│   └── har_rv_v1.0.py               # Original monolithic HAR-RV
│
├── docs/                             # Documentation
│   ├── data/
│   │   └── DATA_VERSIONING.md
│   ├── implementation/
│   │   ├── CLI_TRAINING.md
│   │   ├── code_consolidation_changes.md
│   │   └── final_code_consolidation_summary.md
│   ├── journal/                      # Research session logs
│   │   ├── 2026-02-25.md            # Dataset evolution, Lee-Mykland
│   │   └── 2026-02-26.md            # Multi-window, classification
│   ├── methodology/
│   │   ├── STATISTICAL_ANALYSIS_COMPLETE.md
│   │   ├── JUMP_DETECTION_SUMMARY.md
│   │   └── MATHEMATICAL_REFERENCE.tex
│   ├── project/
│   ├── research/
│   │   ├── THESIS_V2_IMPLEMENTATION_PLAN.md
│   │   ├── session_logs/
│   │   ├── model_comparison_methodology_resolution_2026-01-22.md
│   │   ├── single_gpu_retraining_plan.md
│   │   └── stationarity_cross_model_comparison_research.md
│   └── results/
│       ├── QUICK_REFERENCE.md
│       └── ultra_large_model_results.md
│
├── data/                             # Data files (gitignored)
│   ├── processed/                    # Feature-engineered datasets
│   │   ├── bitcoin_lstm_features_v1.6_final.csv  # RECOMMENDED (41,055 samples)
│   │   ├── bitcoin_lstm_features_v1.1_complete_with_jumps.csv
│   │   └── bitcoin_lstm_features_v1.0_*.csv
│   ├── raw/                          # Raw data from APIs
│   ├── archive/                      # Old dataset versions
│   └── deribit/                      # Deribit-specific data
│
└── deribit_data_collector/           # Data collection tools
    ├── btc_volatility_collector.py   # Custom volatility collector
    ├── historical_options_collector.py
    ├── deribit_data.py               # Deribit API wrapper
    └── examples/
```

## References

Key literature supporting feature selection and methodology documented in project documentation.

**Volatility Modeling and HAR-RV:**
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Fleming, J., Ostdiek, B., & Whaley, R. E. (2001). Predicting Stock Market Volatility: A New Measure. *Journal of Futures Markets*, 21(3), 267-287.

**Jump Detection:**
- Lee, S. S., & Mykland, P. A. (2008). Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics. *Review of Financial Studies*, 21(6), 2543-2577.

**Directional Accuracy:**
- Pesaran, M. & Timmermann, A. (1992). A Simple Nonparametric Test of Predictive Performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.

**On-Chain Metrics:**
- Yang, K., & Fantazzini, D. (2022). NVRV vs. MVRV Comparison for Cryptocurrency Analysis.

**Machine Learning for Volatility Forecasting:**
- Vrontos, I. et al. (2021). Forecasting VIX with Machine Learning. *Journal of Forecasting*.
- Balaneji, B., & Maringer, D. (2022). Implied Volatility Forecasting with XGBoost. *Quantitative Finance*.
- Zhang, L., & Hua, L. (2025). High-Frequency Financial Data Analysis: A Survey. *Mathematics*, 13(3), 347.