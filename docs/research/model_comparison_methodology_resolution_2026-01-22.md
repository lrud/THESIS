# Model Comparison Methodology Resolution

**Date**: January 22, 2026
**Section**: 1.2 — Stationarity and Cross-Model Comparison Framework
**Status**: Resolved

---

## Issue Identified

Initial baseline model comparisons revealed a fundamental methodological problem: models were evaluated on inconsistent dependent variables and transformations. Traditional econometric models (OLS, HAR-RV, XGBoost) targeted ΔDVOL (first differences), while LSTM models targeted DVOL levels. Additionally, stationarity testing revealed conflicting statistical properties in the data.

### Statistical Diagnosis

| Test | DVOL Result | Interpretation |
|------|-------------|----------------|
| Augmented Dickey-Fuller (ADF) | p = 0.008 | Rejects unit root null hypothesis |
| KPSS | p = 0.01 | Rejects stationarity null hypothesis |
| Structural Analysis | 29% mean decline (72.6 → 51.3) | Indicates regime shift |

The conflicting test results (ADF suggests stationarity, KPSS suggests non-stationarity) indicated that DVOL exhibits complex non-stationarity characterized by structural breaks rather than a pure unit root process.

---

## Methodological Alternatives Considered

### Option 1: First-Differencing for All Models
Apply first-differencing to all predictors and target to achieve stationarity.

**Rejection**: Differenced LSTM achieved R²≈0.997, representing a trivial solution equivalent to naive persistence. First-differencing destroys the temporal predictive structure required for level-based forecasting.

### Option 2: Global Normalization
Apply global mean/standard deviation normalization computed from training set.

**Rejection**: Empirical testing showed R²=-5.92 under regime shift conditions, as global normalization could not adapt to the 32% mean decline from training to test periods.

### Option 3: Mixed Approaches
Allow each model class to use its native transformation (e.g., differencing for linear models, rolling normalization for sequential models).

**Rejection**: Clements and Hendry (1999) establish that comparing models on different transformations compares incompatible forecasts, rendering statistical inference invalid.

### Option 4: Two-Subset Comparison
Separate analyses for DVOL levels and ΔDVOL changes, with each model class evaluated on its theoretically appropriate transformation.

**Rejection**: While comprehensive, this approach would not address the core research question of identifying the optimal approach for volatility level forecasting required for option pricing and risk management applications.

---

## Selected Framework: Unified Experimental Setup

### Target Variable
All models predict **DVOL levels** (not first differences).

**Justification**: Cai et al. (2024) mandate that "all models are compared using the same target variable" to ensure fair comparison. Clements and Hendry (1999) demonstrate that comparing models on different transformations compares incompatible forecasting problems.

### Preprocessing Strategy
All models utilize **720-hour rolling window normalization** applied to all features and the target variable.

**Justification**: The *Risks* (2024) study finds that "rolling window GARCH" significantly outperforms expanding window models when structural breaks are present. Lim and Zohren (2021) identify normalization as critical for neural networks in time series forecasting, with standard practice involving sliding windows to maintain stationarity in inputs.

### Models Evaluated
- Econometric: HAR-RV
- Linear: OLS (with and without lagged features)
- Tree-based: Random Forest, XGBoost
- Sequential: LSTM (rolling baseline, jump-aware)

---

## Theoretical Rationale

### Structural Breaks vs. Unit Root Distinction

The conflicting ADF/KPSS results are resolved by recognizing DVOL exhibits **structural breaks** (regime shifts) rather than a pure unit root process.

- **Unit root process**: Series has no memory of a central tendency and follows a random walk
- **Structural break process**: Series is stationary within regimes, but the unconditional mean shifts between regimes

Empirical evidence supports the structural break interpretation: DVOL declined from a mean of 72.6 (first sample half) to 51.3 (second sample half), representing a 29% regime shift rather than gradual random walk drift.

### Why Rolling Normalization Addresses Structural Breaks

Rolling window normalization allows models to adapt to local market conditions by:
1. Using only recent historical data (720 hours) for normalization
2. Automatically adjusting to the current regime's mean and volatility
3. Avoiding contamination from historical regimes that no longer apply

This approach aligns with the structural break literature finding that "break-segmented" data handling improves forecast accuracy relative to expanding window methods (*Risks*, 2024).

---

## References

1.  **Cai, C., Ren, Y., & Yang, X. (2024)**. Forecasting realized volatility with a deep learning-based adaptive learning approach. *Finance Research Letters*, 60, 105081.
2.  **Clements, M. P., & Hendry, D. F. (1999)**. *Forecasting Non-stationary Economic Time Series*. The MIT Press.
3.  **Lim, B., & Zohren, S. (2021)**. Time-series forecasting with deep learning: a survey. *Philosophical Transactions of the Royal Society A*.
4.  **Forecasting Financial Volatility Under Structural Breaks** (2024). *Risks*, 18(9), 494.
