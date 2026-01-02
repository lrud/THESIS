# LSTM Training Decision: Levels vs Changes Target

**Date:** December 30, 2025

## Context

Following completion of baseline benchmarking work, a decision was required regarding the target variable specification for LSTM model training on v1.1 data (39,472 samples, corrected `dvol_rv_spread`).

## Prior Work

### 1. Benchmarking Notebooks (December 29, 2025)

Two Jupyter notebooks were created to establish baseline models:

**`01_data_exploration.ipynb`:**
- Data quality verification on v1.1 data
- Statistical validation of stationarity properties
- Feature correlation analysis

**`benchmarking.ipynb`:**
- Established 5 baseline models predicting Δdvol (next-period DVOL change):
  - OLS (8 features): R² = 0.000, Dir = 50.2%
  - HAR-RV (3 features): R² = 0.001, Dir = 50.2%
  - Random Forest: R² = 0.117, Dir = 50.4%
  - XGBoost (baseline): R² = 0.405, Dir = 54.8%
  - XGBoost Spec D: R² = 0.490, Dir = 58.7% overall, R² = 0.696, Dir = 61.3% jumps

**Key Finding:** All linear models show near-zero R² on Δdvol, confirming DVOL changes are fundamentally difficult to predict with standard methods.

### 2. HAR-RV Module Deprecation

The monolithic HAR-RV implementation (2,480 lines) was moved to `deprecated/har_rv_v1.0.py`. Functionality is now preserved in:
- Modular `scripts/thesis_v2/har_rv/` package
- Backward-compatible wrapper at `scripts/utils/har_rv.py` (with deprecation warning)

## The Econometric Dilemma

### Problem Statement

**DVOL is non-stationary:** Mean decreased 32% from train (69.32) to test (47.40).

**Two standard solutions:**

| Approach | Method | README Result |
|----------|--------|---------------|
| **Differencing** | Predict Δdvol = dvol_t+1 - dvol_t | All models reduced to naive persistence (R²=0.997, trivial solution) |
| **Rolling Normalization** | Predict (dvol - local_mean) / local_std | R²=0.88, Dir=52.8% (genuine forecasting) |

**Core Question:** Can we compare R²=0.88 (rolling levels) to R²=0.49 (differenced changes)?

### Research Questions Investigated

1. **Time Series Econometrics:** Is rolling normalization methodologically sound for model comparison, or must all models use differencing?

2. **The Differencing Paradox:** Why did differencing make all baseline models reduce to naive persistence, despite achieving stationarity?

3. **Volatility Forecasting Literature:** How do HAR-RV and volatility models handle non-stationarity?

4. **Model Comparison Metrics:** How can we fairly compare models predicting different targets?

5. **README Core Claim:** Is "rolling normalization preserves feature relationships while differencing destroys them" a valid econometric justification?

## Literature Review Findings

### Key Source: Corsi (2009) - HAR-RV

**Paper:** "HAR-RV - Heterogeneous Autoregressive Model of Realized Volatility" (Journal of Financial Econometrics)

**Specification:**
$$RV_{t+1}^{(d)} = c + \beta^{(d)}RV_t^{(d)} + \beta^{(w)}RV_t^{(w)} + \beta^{(m)}RV_t^{(m)} + \omega_{t+1}$$

**Critical Finding:**
- **Target:** Daily realized volatility (LEVEL), not ΔRV
- **Transformation:** None - raw realized volatility
- **Results:** R² = 0.565 (USD/CHF), 0.707 (S&P 500), 0.236 (T-Bond)
- **Conclusion:** Realized volatility is not differenced, despite being non-stationary

### The Cointegration Explanation

**Source:** Boston College EC823 Lecture Notes on VECM
**Quote:** "If the series are cointegrated, they move together in the long run. A VAR in first differences, although properly specified in terms of covariance-stationary series, will not capture those long-run tendencies. This implies that the simple regression in first differences is **misspecified**."

**Implication:** When features (nvrv, dvol_rv_spread, etc.) have long-run relationships with dvol, differencing destroys those relationships, reducing R² to noise levels.

**Finding:** 2025 analysis showed VAR models on differenced cointegrated series performed **50% worse** (MSE 0.5105 vs 0.3463) than level models.

### Comparison Metrics Literature

**Diebold-Mariano Test:** Designed for comparing forecasts with different specifications. Requires loss differential stationarity.

**Directional Accuracy (DA):** Model-independent metric. Evaluates whether models correctly predict sign of next-period volatility, regardless of target transformation.

**Verdict:** R² values across different targets are NOT directly comparable. Use DA or Diebold-Mariano for fair comparison.

## Decision: Option C (Train Both)

Based on literature review, the decision is to train **two LSTM models**:

| Model | Target | Purpose | Benchmark |
|-------|--------|---------|-----------|
| **LSTM-Rolling** | dvol (level, rolling norm) | Validate HAR-RV literature precedent | Retrain XGBoost on levels |
| **LSTM-Differenced** | Δdvol (change) | Direct XGBoost comparison | XGBoost Spec D (R²=0.50) |

**Primary comparison metric:** Directional Accuracy (model-agnostic)

**Secondary metrics:** Diebold-Mariano test, jump-period evaluation

## Rationale

### For LSTM-Rolling (Levels)
1. **HAR-RV Precedent:** Corsi (2009) and all extensions predict levels, not changes
2. **Cointegration Principle:** Differencing destroys long-run feature relationships
3. **README Validation:** Previous R²=0.88 result is credible if literature precedent holds
4. **Trading Utility:** Level forecasts directly usable for option pricing

### For LSTM-Differenced (Changes)
1. **Econometric Rigor:** Standard practice for non-stationary series
2. **Direct Comparison:** Fair comparison to XGChange Spec D (R²=0.50)
3. **Honest Assessment:** R²≈0 on changes confirms task difficulty
4. **Sequential Learning Test:** Can LSTM capture dynamics trees miss on differenced data?

### Expected Outcomes (Based on Literature)
- **LSTM-Rolling:** R²=0.70-0.90 (HAR-RV precedent)
- **LSTM-Differenced:** R²=0.0-0.20 (cointegration information loss)

## Training Plan

### Step 1: LSTM-Rolling on v1.1
- **Data Loader:** `data_loader_rolling.py` or `data_loader_jump_aware.py`
- **Features:** 7 core (dvol_lag_1d, dvol_lag_7d, dvol_lag_30d, transaction_volume, network_activity, nvrv, dvol_rv_spread)
- **Target:** dvol level with rolling window normalization
- **Variants:**
  - Rolling: 7 features + rolling norm + standard loss
  - Jump-Aware: 7 features + 4 jump features + rolling norm + weighted loss

### Step 2: LSTM-Differenced on v1.1
- **Data Loader:** Create `data_loader_changes.py`
- **Features:** 7 core or Spec D features
- **Target:** Δdvol (dvol_change)
- **Purpose:** Direct comparison to XGBoost Spec D

### Step 3: Jump-Period Evaluation
- Train on all data
- Evaluate separately on jump vs normal periods
- Compare: LSTM-jumps vs XGBoost Spec D-jumps (Dir=61.3%)

### Step 4: Comparison
- Directional accuracy (primary metric)
- Diebold-Mariano test (statistical significance)
- Trading simulation (economic value)

## Literature References

### Primary Sources
1. **Corsi (2009)** - "HAR-RV - Heterogeneous Autoregressive Model of Realized Volatility" - Journal of Financial Econometrics
   - *Key finding:* Predict RV levels, R²=0.56-0.71

2. **Boston College VECM Lecture Notes** - EC823 Lecture 10
   - *Key finding:* VAR(Δy) misspecified if cointegration exists

3. **Andersen et al. (2003)** - "Modeling and Forecasting Realized Volatility"
   - *Key finding:* R²=35-45% on realized variance vs 2-4% on returns

### Comparison Methods
4. **Diebold & Mariano (1995)** - Forecast comparison test
5. **Directional Accuracy Literature (2024)** - DA as model-independent metric

### Modern Extensions
6. **HAR-GARCH, HARQ, Path-Dependent HAR (2023-2025)** - All use level targets
7. **Cointegration Analysis (2025)** - Differencing cointegrated series → 50% worse MSE

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Dec 29, 2025 | Created benchmarking notebooks | Establish baseline performance |
| Dec 29, 2025 | Moved HAR-RV to deprecated | Code consolidation |
| Dec 30, 2025 | Literature review on differencing vs rolling norm | Resolve target specification dilemma |
| Dec 30, 2025 | **Decision: Option C (Train both)** | HAR-RV precedent + econometric rigor |

## Next Steps

1. Create `data_loader_changes.py` for differenced target
2. Train LSTM-Rolling on v1.1 (validate README results)
3. Train LSTM-Differenced on v1.1 (compare to XGChange)
4. Evaluate both on jump periods
5. Compare via directional accuracy and Diebold-Mariano
6. Update README with validated v1.1 results

---

**Status:** Decision made, ready to proceed with implementation.
