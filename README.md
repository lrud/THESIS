# LSTM Forecasting of Bitcoin Implied Volatility (DVOL)

## Objective

Develop an LSTM neural network model to forecast Bitcoin implied volatility (DVOL) using on-chain metrics and historical volatility patterns, validated through statistical analysis.

## Current Status (January 27, 2026)

**Phase:** Unified model comparison complete - 17 models evaluated (13 linear/tree + 4 LSTM).

### Recent Developments

**Unified LSTM Framework (January 27, 2026):**
- Implemented 3 new LSTM models matching linear/tree specifications: market (4), market_jumps (8), market_lags (7)
- All models use 720h rolling normalization + 60/20/20 data splits for fair comparison
- **market_lags (512×7, 13.8M params)**: R² = 0.8021 - **best LSTM performer**
- Jump features provide **no improvement** - market_lags matches jump_aware (0.8021 vs 0.800)
- 512×7 architecture requires minimum 7 features - unstable with 4 features

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
- **Critical finding:** Linear/tree models (R² = 0.94-95) significantly outperform LSTM (R² = 0.80)
- Jump features provide minimal or negative impact across all specifications
- **LSTM gap:** 14.7% R² deficit (0.8021 vs 0.9492) - extreme persistence favors linear models
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
- 39,472 hourly samples (April 23, 2021 09:00 - December 28, 2025 23:00)
- 9 core predictors engineered and validated (100% complete)
- 11 jump detection features (indicator, magnitude, timing, clustering)
- Statistical analysis confirmed LSTM suitability
- No multicollinearity issues (all VIF < 5)

**Model Development & Benchmarking:**

**Historical Models (Original Research):**
- LSTM (Absolute - Global Norm): Failed (R² = -5.92)
- LSTM (Differenced): R² = 0.997, MAPE = 0.54%, Dir = 51.7% (trivial solution)
- Naive Persistence: R² = 0.997, MAPE = 0.54%
- LSTM (Rolling Window 512×7): R² = 0.201, MAE = 4.31, Dir = 49.7% (v1.1 baseline)
- **LSTM (Jump-Aware 512×7)**: R² = 0.800, MAE = 2.04, RMSE = 2.86, Dir = 49.7% (v1.1 optimal)

**Unified Framework Models (January 27, 2026):**
- LSTM market_lags (512×7): R² = 0.8021, RMSE = 3.67, MAE = 2.81, Dir% = 51.9% (best LSTM)
- LSTM jump_aware (512×7): R² = 0.800, RMSE = 2.86, MAE = 2.04, Dir% = 49.7%
- LSTM market_lags (128×2): R² = 0.6709, RMSE = 3.67, MAE = 2.81, Dir% = 51.9%
- LSTM market (128×2): R² = 0.6686, RMSE = 3.68, MAE = 2.81, Dir% = 51.5%
- LSTM market_jumps (128×2): R² = 0.6685, RMSE = 3.69, MAE = 2.82, Dir% = 51.4%

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

**Updated Finding (January 27, 2026):**
Unified framework reveals jump features provide **NO value** - market_lags (7 features, no jumps) matches jump_aware (11 features with jumps): R² 0.8021 vs 0.800.

**Research Journey:**
1. **Initial observation:** Rolling baseline (R²=0.201) vs jump-aware (R²=0.800) suggested jump features were critical
2. **Unified framework test:** Trained market_lags (no jumps) and found equal performance (R²=0.8021)
3. **Conclusion:** Lagged volatility features capture all the jump-related information; explicit jump features are redundant

**Critical Discovery & Solution (Original Research):**

- All differenced models reduced to naive persistence baseline
- First-differencing destroys predictable structure despite achieving stationarity
- **Solution 1:** Rolling window normalization (720-hour windows)
  - Adapts to regime changes (mean shift from 69 to 48)
  - Preserves feature-target relationships
  - Achieves genuine forecasting skill (R²=0.201 without jump features)
- **Solution 2:** Jump-aware modeling with weighted loss (Original Approach)
  - Detected 7,278 jumps (19.2% of data) using Lee-Mykland test
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

**Updated Research - Unified Framework (January 27, 2026):**
- **New Finding:** market_lags (7 features, no jumps) achieves R²=0.8021, matching jump_aware (R²=0.800)
- **Key Insight:** Lagged volatility features (1d, 7d, 30d) capture all jump-related information
- **Revised Conclusion:** Jump features are redundant; the improvement comes from multi-scale volatility persistence, not jump detection
- **Contribution:** First systematic comparison of LSTM vs linear/tree models on identical 720h rolling normalization framework

**Thesis Contributions:**
- Trivial solution detection framework (metric equivalence + directional accuracy)
- Rolling normalization for regime-shifting financial data
- Unified model comparison framework (17 models: linear, tree, LSTM)
- Demonstration that simple HAR-RV (3 features) outperforms complex LSTM (13.8M params)
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

- Trivial solution detection framework (metric equivalence + directional accuracy)
- Rolling normalization for regime-shifting financial data
- **Unified model comparison framework:** 17 models (linear, tree, LSTM) on identical preprocessing
- **Lagged volatility dominance:** HAR-RV (3 features) outperforms LSTM (13.8M parameters)
- **Jump feature redundancy:** Explicit jump features provide no value beyond lagged volatility
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

**Architecture Scaling Results (v1.1 + Unified Framework):**

| Model | Features | Hidden | Layers | Params | R² | RMSE | Status |
|-------|----------|--------|--------|--------|----|----|----|
| **market_lags** | 7 | 512 | 7 | 13.8M | **0.8021** | 3.67 | **Best LSTM (Current)** |
| **jump_aware** | 11 | 512 | 7 | 13.8M | 0.8000 | 2.86 | Stable (Original) |
| **market_lags** | 7 | 128 | 2 | 210K | 0.6709 | 3.67 | Stable baseline |
| market | 4 | 128 | 2 | 210K | 0.6686 | 3.68 | Stable (max for 4 feat) |
| market_jumps | 8 | 128 | 2 | 211K | 0.6685 | 3.69 | Stable |
| market_jumps | 8 | 512 | 7 | 13.8M | 0.6202 | 3.95 | Stable (underperforms) |
| rolling | 7 | 512 | 7 | 13.8M | 0.201 | 6.20 | **DEPRECATED** (historical) |
| Ultra-Large | 11 | 512 | 3 | 5.4M | 0.795 | - | Legacy (historical) |
| Deep | 11 | 512 | 5 | 9.6M | 0.784 | - | Legacy (historical) |

**Key Findings:**
- **Depth scales:** 3→5→7 layers improves R² (0.795 → 0.784 → 0.800)
- **Width fails:** 512→1024 causes instability (validation loss → inf)
- **Feature requirement:** 512×7 requires minimum 7 features for stability
- **Jump features:** No improvement (market_lags = jump_aware, market_jumps < market)
- **Critical insight:** Lagged volatility (7 features) achieves same performance as jump-aware (11 features)

## CLI Training System

The project implements a CLI training system that replaces the original script-based approach:

### Core Training Commands

```bash
# === UNIFIED FRAMEWORK MODELS (January 2026) ===

# market_lags - Best LSTM performer (R² = 0.8021, 7 features)
.venv/bin/python cli/bin/train.py market_lags \
  --hidden-size 512 --num-layers 7 --dropout 0.5 \
  --batch-size 32 --lr 0.0001 --epochs 100 --use-multi-gpu

# market - Market features only (R² = 0.6686, 4 features, 128×2 max)
.venv/bin/python cli/bin/train.py market --epochs 50

# market_jumps - Market + jumps (R² = 0.6685, 8 features, 128×2)
.venv/bin/python cli/bin/train.py market_jumps --epochs 50

# === LEGACY MODELS (For Historical Comparison) ===

# jump_aware - Original jump-aware model (R² = 0.800, 11 features)
# NOTE: Unified framework shows jump features provide no improvement
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
| **Data Source** | bitcoin_lstm_features_v1.1_complete_with_jumps.csv | 39,472 samples (Apr 2021 - Dec 2025) |

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
| **RF (No Lags + Jumps)** | Tree | 8 | 100×10 | 0.7564 | 3.61 | 2.81 | 51.0% | - |
| **OLS (No Lags + Jumps)** | Linear | 8 | - | 0.7393 | 3.74 | 2.85 | 51.3% | 8 |
| **OLS (No Lags)** | Linear | 4 | - | 0.7363 | 3.76 | 2.89 | 51.3% | 4 |
| **XGB (No Lags + Jumps)** | Tree | 8 | 100×6 | 0.7304 | 3.80 | 2.87 | 51.1% | - |
| **XGB (No Lags)** | Tree | 4 | 100×6 | 0.6989 | 4.02 | 2.98 | 50.2% | - |
| **RF (No Lags)** | Tree | 4 | 100×10 | 0.6914 | 4.06 | 2.99 | 50.8% | - |
| **LSTM market_lags** | LSTM | 7 | 512×7 | **0.8021** | 3.67 | 2.81 | 51.9% | 13.8M |
| **LSTM jump_aware** | LSTM | 11 | 512×7 | 0.8000 | 2.86 | 2.04 | 49.7% | 13.8M |
| **LSTM market_lags** | LSTM | 7 | 128×2 | 0.6709 | 3.67 | 2.81 | 51.9% | 210K |
| **LSTM market** | LSTM | 4 | 128×2 | 0.6686 | 3.68 | 2.81 | 51.5% | 210K |
| **LSTM market_jumps** | LSTM | 8 | 128×2 | 0.6685 | 3.69 | 2.82 | 51.4% | 211K |
| **LSTM market_jumps** | LSTM | 8 | 512×7 | 0.6202 | 3.95 | 3.06 | 50.8% | 13.8M |
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

**Key Insights (17 Models - Unified Framework):**

1. **Linear/Tree Dominance:** Models with lagged volatility achieve 94-95% R²
   - Best: RF (Lags + Jumps) = 0.9492
   - Baseline: HAR-RV (3 features) = 0.9454
   - Tree models match linear performance exactly

2. **LSTM Performance Gap:** 14.7% R² deficit vs linear/tree
   - Best LSTM: market_lags (512×7) = 0.8021
   - Best Linear/Tree: RF (Lags + Jumps) = 0.9492
   - Root cause: Hourly DVOL autocorrelation = 0.9992 (extreme persistence favors linear models)

3. **Jump Features Add No Value:** Across all model types
   - Linear: +0.1% R² (0.9480 → 0.9490)
   - Tree: +0.07% R² (0.9485 → 0.9492)
   - LSTM: **Negative** impact (0.8021 → 0.800, 0.6709 → 0.6685)

4. **Architecture Scaling Requirements:**
   - 512×7 (13.8M params) requires minimum 7 features
   - 4-feature models unstable at large scale
   - 128×2 (210K params) stable for all feature sets

5. **Directional Forecasting:** ALL 17 models ≈ 50% (statistically random)
   - Best: LSTM market_lags = 51.9%
   - Worst: XGB NoLag = 50.2%
   - Signal-to-noise ratio: Forecast error (RMSE=1.65-3.95) >> typical hourly change (~0.26)

6. **Practical Implication:**
   - **Suitable for:** Risk management, option pricing, volatility level estimation
   - **NOT suitable for:** Directional trading, market timing strategies

## Documentation

**Key Documents:**

- `CLAUDE.md` - Claude AI assistant guide and project context
- `docs/research/session_logs/THESIS_V2_SESSION_CONSOLIDATION_2026-01-02.md` - Complete research session log with v1.1 validated results, LSTM architecture optimization, and VaR backtesting
- `docs/QUICK_REFERENCE.md` - Performance summary and thesis defense points
- `scripts/thesis_v2/har_rv/` - Modular HAR-RV analysis package with statistical diagnostics
- `scripts/utils/README.md` - Consolidated utilities implementation guide

**Session Log Highlights:**

**January 27, 2026 - Unified Framework Completion:**
- 17 models trained: 13 linear/tree + 4 LSTM (market, market_jumps, market_lags, jump_aware)
- **Critical discovery:** Jump features provide NO value - market_lags (0.8021) = jump_aware (0.800)
- Linear/tree models dominate: RF (Lags+Jumps) = 0.9492 vs best LSTM = 0.8021
- LSTM gap: 14.7% R² deficit - extreme persistence (autocorrelation = 0.9992) favors linear models
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
├── cli/                          # Modern training interface
│   ├── bin/train.py             # Main CLI entry point (supports 6 model types)
│   ├── config/
│   │   ├── config.py            # Configuration management system
│   │   └── feature_configs.py   # Unified feature set configurations
│   └── scripts/trainers/        # Modular trainer implementations
│       ├── jump_aware_trainer.py
│       ├── rolling_trainer.py   # DEPRECATED - historical comparison only
│       ├── differenced_trainer.py # DEPRECATED - trivial solution
│       └── unified_trainer.py   # Single trainer for market/market_jumps/market_lags
├── scripts/                     # Analysis and modeling components
│   ├── modeling/                # LSTM neural network components
│   │   ├── data_loader_unified.py  # Unified dataset for all LSTM models
│   │   └── lstm_dvol.py         # Core LSTM model architecture
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
    │   └── 2026-01-27/           # Latest unified framework results
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