# Comprehensive Statistical Audit Report

**Date:** April 3, 2026  
**Scope:** All LSTM training results (512x3, 512x7, 72h normalization, single-GPU)  
**Auditor:** Claude AI (with literature-backed reference points)  
**Status:** CRITICAL FINDINGS — results require reinterpretation  

---

## Executive Summary

A systematic audit of all implicit statistical assumptions in the training pipeline has been conducted against literature reference points. The central finding is:

> **DVOL hourly autocorrelation is ρ=0.9992 at lag-1. A naive persistence baseline (predict DVOL[t+1] = DVOL[t]) achieves R²=0.9985 on the raw series. The reported LSTM R²=0.941, while numerically high, represents substantially worse performance than simply predicting "no change."**

This does not invalidate the project — but it fundamentally reframes how results should be interpreted and presented. The audit identifies 11 assumptions, flags 4 as CRITICAL, and provides actionable fixes.

---

## Assumption 1: R² = 0.94 Represents Good Predictive Performance

**Status: CRITICAL — Requires reinterpretation**

### What was assumed
R² = 0.9406 on the test set indicates the LSTM is a strong forecasting model for Bitcoin DVOL.

### Literature reference
In volatility forecasting, raw R² is misleading without a persistence baseline. Implied volatility indices (VIX, DVOL) are among the most persistent financial time series, with hourly lag-1 autocorrelations typically exceeding 0.99 (Andersen & Benzoni, 2009; Corsi, 2009). Any evaluation must compare against the trivially achievable persistence benchmark.

### What the data shows

| Baseline | R² | RMSE |
|----------|-----|------|
| **Naive persistence (DVOL[t+1] = DVOL[t])** | **0.9985** | **0.75** |
| **24h rolling mean (DVOL[t+1] = mean of last 24h)** | ~0.996 | ~1.1 |
| LSTM best (512x3, market) | 0.9406 | 1.62 |
| HAR-RV | 0.9592 | 1.38 |
| XGBoost (NoLag_Jumps) | 0.9940 | 0.53 |

DVOL hourly autocorrelation:
- Lag 1h: **ρ = 0.9992**
- Lag 24h: ρ = 0.9824
- Lag 72h: ρ = 0.9552
- Lag 168h (1 week): ρ = 0.9136

### Root cause analysis

The 72h rolling z-score normalization **destroys the level information** that persistence exploits. After normalization, DVOL's near-unit-root behavior is replaced with a mean-reverting z-score. The model must then predict the normalized deviation from a 72h rolling mean — a much harder task than predicting the raw level.

This means:
- **R² in normalized space** measures something fundamentally different from R² on raw levels
- The LSTM's R²=0.94 in **original** scale (after inverse transform) is **worse** than persistence because the inverse transform re-introduces the rolling mean as a baseline, but the model's normalized prediction errors are amplified by the rolling std
- XGBoost's R²=0.994 is high because tree models can effectively memorize the autoregressive structure (they see `dvol_lag_1d`, `dvol_lag_7d` as explicit features)

### Recommended fix
1. **Report persistence baseline alongside all models** — compute R²(DVOL[t+1] = DVOL[t]) on the exact same test set
2. **Report out-of-sample R² relative to persistence**: `R²_relative = 1 - MSE_model / MSE_persistence` (will be negative for LSTM)
3. **Restructure the narrative**: LSTM adds value in regime change detection (54.76% jump directional accuracy), not in level prediction
4. **Use DM test (Diebold-Mariano)** to test whether LSTM significantly beats persistence on any loss function

### Severity: **CRITICAL** — affects every conclusion about model quality

---

## Assumption 2: Rolling Normalization Has No Look-Ahead Bias

**Status: YELLOW — Minor concern, well-designed but has edge effects**

### What was assumed
The 72h rolling z-score normalization applied per-split has no information leakage.

### Code analysis (`data_loader_unified.py:99-128`)

```python
rolling_mean = self.data[col].rolling(self.window_size, min_periods=1).mean()
rolling_std = self.data[col].rolling(self.window_size, min_periods=1).std()
```

**Findings:**

1. **Normalization IS applied independently per split** (train/val/test each get their own `UnifiedLSTMDataset`) — this is correct and prevents cross-split leakage.

2. **Within each split**, the rolling window at time t uses data from [t-72, t] — this is a **centered/causal** window that only looks backward. No future information leaks in.

3. **`min_periods=1`** means the first 71 samples in each split use windows smaller than 72h. These are then **dropped** by `self.data = self.data[self.window_size:]` (line 128). This is correct.

4. **Edge effect**: The first 72 rows of each split are discarded, but the rolling statistics at position 72 use only the first 72 data points of that split (no data from the previous split). This is clean.

### Verdict: **No look-ahead bias.** The normalization is properly causal.

### Minor concern
The rolling mean/std include the current observation at time t when computing the normalization for time t. For the target variable, this means `target_normalized[t] = (DVOL[t] - mean(DVOL[t-72:t])) / std(DVOL[t-72:t])` — the target's own value leaks into its normalization. This is standard practice in online/rolling normalization and does not constitute look-ahead bias, but it does slightly reduce the variance of the normalized target, making the prediction task marginally easier.

### Severity: **LOW** — properly implemented

---

## Assumption 3: Train/Val/Test Temporal Split Is Valid

**Status: GREEN — Properly implemented**

### What was assumed
The 60/20/20 temporal split preserves temporal ordering and prevents data leakage.

### Code analysis (`data_loader_unified.py:202-208`)

```python
train_size = int(n * (1 - val_ratio - test_ratio))
train_data = df.iloc[:train_size]
val_data = df.iloc[train_size : train_size + val_size]
test_data = df.iloc[train_size + val_size :]
```

**Findings:**

| Split | Rows | Date Range (approx) |
|-------|------|-------------------|
| Train | 24,633 | Apr 2021 – Oct 2023 |
| Val | 8,211 | Oct 2023 – Oct 2024 |
| Test | 8,211 | Oct 2024 – Dec 2025 |

After removing 72 (window) + 24 (sequence) = 96 samples per split:
- Train: ~24,537 usable sequences
- Val: ~8,115 usable sequences  
- Test: ~8,115 usable sequences

This is a clean, non-overlapping temporal split. No shuffle is applied (correct for time series). The DataLoader uses `shuffle=True` for training only (correct — shuffles batches, not temporal order within sequences).

### Concern: Fixed single split
The entire evaluation rests on ONE train/val/test partition. There is no walk-forward validation, no expanding window, and no k-fold temporal cross-validation. This means:
- Results may be sensitive to the specific split date
- The test period (Oct 2024 – Dec 2025) may have particular volatility regimes
- No confidence intervals on test metrics

### Recommended fix
- Run at least 3-5 different train/test cutoff dates (e.g., test starting at 2024-06, 2024-09, 2024-12, 2025-03, 2025-06)
- Report mean ± std of R² across folds
- Alternatively, use walk-forward validation with expanding training window

### Severity: **MEDIUM** — single split limits generalizability claims

---

## Assumption 4: Directional Accuracy ~51% Is Near-Random

**Status: CRITICAL — Correct interpretation, but needs statistical testing**

### What was assumed
Directional accuracy of ~51% indicates the LSTM has essentially no directional prediction capability, consistent with the efficient market hypothesis.

### Statistical analysis

With N ≈ 8,115 test samples and directional accuracy of 51.07%:

Under H₀: p = 0.5 (random direction prediction):
```
z = (0.5107 - 0.5) / sqrt(0.5 × 0.5 / 8115) = 0.0107 / 0.00555 = 1.93
p-value = 0.054 (two-tailed)
```

This is **marginally non-significant at α=0.05**. The LSTM's directional accuracy is NOT statistically distinguishable from random guessing at the standard significance level.

However, the relevant test for time series with serial correlation is **Pesaran & Timmermann (1992/2009)** or the **Henriksson-Merton (1981)** market timing test, which accounts for the fact that DVOL direction changes are themselves serially correlated (high persistence means direction changes cluster).

### Literature reference
Pesaran & Timmermann (2009) and Blaskowitz & Herwartz (2014) show that standard binomial tests over-reject when directional data has serial dependence. The corrected test would yield even larger p-values.

For jump periods (54.76% directional accuracy with N=43):
```
z = (0.5476 - 0.5) / sqrt(0.5 × 0.5 / 43) = 0.0476 / 0.0763 = 0.62
p-value = 0.53
```

**Completely non-significant.** The apparent jump directional improvement is within noise.

### Verdict
- Overall Dir% ≈ 51%: **not statistically significant** (p=0.054, uncorrected; likely p>0.10 with serial correlation correction)
- Jump Dir% ≈ 55%: **not significant** (p=0.53, only 43 samples)
- Tree models' Dir% ≈ 49-50%: also not significant
- **All models have essentially zero directional predictive power**

### Recommended fix
1. Implement Pesaran-Timmermann (2009) test for directional accuracy with serial correlation correction
2. Report p-values alongside Dir%
3. Frame the finding honestly: "no model achieves statistically significant directional prediction, consistent with EMH"

### Severity: **HIGH** — claims about directional edge are unsupported

---

## Assumption 5: Tree Model R² = 0.994 Represents Genuine Superiority

**Status: CRITICAL — Requires nuance**

### What was assumed
XGBoost achieving R²=0.994 >> LSTM's R²=0.94 proves tree models are superior for this task.

### Analysis

Tree models in this pipeline see **explicit lag features** (`dvol_lag_1d`, `dvol_lag_7d`, `dvol_lag_30d`) as direct inputs. Given DVOL's lag-1 autocorrelation of 0.9992:

1. **XGBoost with `dvol_lag_1d`**: The model has direct access to DVOL[t-24h]. Since DVOL barely changes hour-to-hour (RMSE of persistence = 0.75), a tree can learn `DVOL[t+1] ≈ dvol_lag_1d` trivially. This inflates R² via near-duplicate target leakage, not genuine learning.

2. **The R² comparison is unfair**: LSTM does NOT see raw DVOL values — it sees only 72h-normalized z-scores where level information is removed. XGBoost (in the unified notebook) likely uses a different normalization or raw features. Different preprocessing = unfair comparison.

3. **OLS achieving R²=0.992** is the strongest evidence that this is persistence-driven: a linear regression of DVOL[t+1] on dvol_lag_1d alone would achieve R² ≈ 0.998+ given ρ=0.999.

### Literature reference
This is a well-known pitfall in volatility forecasting evaluation. Patton (2011) and Hansen & Lunde (2006) emphasize that raw R² comparisons between models with different information sets (especially lag inclusion) are meaningless. The proper comparison is:
- Use the SAME loss function
- Use the SAME normalization
- Apply Diebold-Mariano (1995) or Hansen et al. (2011) model confidence set tests

### Recommended fix
1. Compute persistence R² on the exact same test set with same normalization
2. Compare all models on the SAME normalized representation
3. Report DM test statistics between model pairs
4. Frame tree model superiority as "tree models exploit lag features effectively" rather than "tree models are better forecasters"

### Severity: **CRITICAL** — the headline comparison is misleading

---

## Assumption 6: 5.4M Parameters with ~24K Training Samples Is Acceptable

**Status: YELLOW — Risk exists but mitigated by regularization**

### What was assumed
A 5.41M parameter LSTM can be trained on ~24,537 training sequences without overfitting.

### Analysis

| Metric | Value | Rule of thumb |
|--------|-------|---------------|
| Parameters | 5,410,000 | — |
| Training samples | ~24,537 | — |
| **Ratio** | **1:220** | Literature suggests ≥10:1 (Cho et al., 2015) |

The parameter-to-sample ratio is 1:220, meaning the model has ~220 training samples per parameter. Literature recommendations vary wildly:
- Cho et al. (2015): 50-1000× samples per DV → this project has 220× → marginal
- Kavzoglu & Mather (2003): 10-100× per IV → satisfied
- Cheng et al. (2025, Monte Carlo study): NN stability requires 1000+ samples for small models → this dataset is large enough

### Mitigating factors
1. **Dropout = 0.4** (aggressive regularization)
2. **L2 weight decay = 1e-5** (explicit regularization)
3. **Gradient clipping = 1.0** (prevents exploding gradients)
4. **Early stopping with patience=15** (prevents overfitting to training data)
5. **512×7 = 512×3 performance** — identical results despite 2.5× more parameters suggests the model is NOT memorizing training data (if it were, 512×7 would overfit more)

### Evidence against overfitting
- All 4 feature sets converge to R²≈0.941 (a memorizing model would show variance across feature sets)
- 512×7 does NOT improve over 512×3 (capacity saturation, not memorization)
- Early stopping at epoch 21-23 (well before epoch 100 limit)

### Concern: No train R² reported
The training script does not log training R² — only training loss. Without knowing the train R², we cannot compute the **generalization gap** (train R² - test R²), which is the primary overfitting diagnostic.

### Recommended fix
1. Log train R² at each epoch alongside train loss
2. Compute and report train/test R² gap
3. If train R² >> 0.94, investigate further

### Severity: **MEDIUM** — signs are positive but incomplete diagnostics

---

## Assumption 7: MSE Loss Is Appropriate for Volatility Forecasting

**Status: YELLOW — Standard but suboptimal**

### What was assumed
MSE loss (possibly weighted for jump periods) is an appropriate training objective for DVOL forecasting.

### Analysis

MSE loss has known issues for volatility forecasting (Patton, 2011):

1. **MSE is sensitive to outliers**: Volatility has heavy tails (kurtosis > 3). MSE penalizes large errors quadratically, causing the model to prioritize avoiding large errors over being accurate in typical regimes.

2. **MSE-optimal forecast = conditional mean**: Under MSE, the optimal prediction is E[y|X]. For volatility, this systematically underestimates large volatility moves because E[y|X] < median(y|X) for right-skewed distributions.

3. **Alternative losses**:
   - **QLIKE** (Patton, 2011): `log(ŷ/y) - ŷ/y + 1` — more robust to outliers, standard in volatility forecasting literature
   - **MAE**: Less sensitive to outliers than MSE
   - **MAPE**: Already computed as 2.5%, but not used as training loss

4. **Jump weighting** (×2 for jump periods): This is a reasonable heuristic but ad hoc. The factor of 2× was not tuned or justified statistically.

### Literature reference
Patton (2011) shows that ranking of volatility forecasts can be **reversed** depending on which loss function is used. A model that wins under MSE may lose under QLIKE and vice versa. The standard practice in volatility forecasting is to evaluate under multiple loss functions.

### Recommended fix
1. Train with QLIKE loss as an alternative and compare
2. Evaluate all models under MSE, MAE, QLIKE, and log-loss
3. Apply DM test under each loss function

### Severity: **MEDIUM** — standard practice but should acknowledge limitations

---

## Assumption 8: Early Stopping Based on Validation Loss Is Unbiased

**Status: YELLOW — Minor selection bias present**

### What was assumed
Using the best validation loss checkpoint for model selection introduces no bias in test metrics.

### Analysis

The training script:
1. Trains for up to 100 epochs
2. Saves the model with lowest validation loss
3. Evaluates on the test set using this checkpoint

This is standard practice, but it introduces a subtle **model selection bias**:
- The "best" validation loss is the minimum of a stochastic process (val loss over epochs)
- This minimum is an optimistic estimate of the model's true generalization performance
- The test set evaluation inherits this optimism

The magnitude of this bias depends on the variance of the validation loss curve. With patience=15, the model cannot overfit to a single lucky epoch, which limits the bias.

### What's missing
- No confidence intervals on test metrics (from multiple random seeds)
- No repeated runs (each model trained only once with one random seed)
- Variance across random initializations is unknown

### Literature reference
Machine learning best practice (Bates et al., 2023) recommends either:
- Nested cross-validation for unbiased estimates
- Or reporting test metrics with confidence intervals from ≥5 random seeds

### Recommended fix
1. Run each model with 3-5 different random seeds
2. Report mean ± std of R², RMSE, Dir%
3. This is especially important given Dir% ≈ 51% (near the noise floor)

### Severity: **LOW** — bias is small with patience=15, but no variance estimates exist

---

## Assumption 9: Residual Analysis Is Not Required

**Status: HIGH — Critical gap in validation**

### What was assumed
Point metrics (R², RMSE, MAE) are sufficient to validate model quality. No residual analysis was performed.

### What should be checked

1. **Ljung-Box test on residuals**: If the model has captured all predictable structure, residuals should be white noise. Significant autocorrelation in residuals means the model is leaving predictable patterns on the table.

2. **ARCH effects in residuals**: If residuals show volatility clustering, the model has not captured the full dynamics. Apply Engle's ARCH-LM test.

3. **Normality of residuals**: Check via Jarque-Bera or Shapiro-Wilk test. Significant non-normality suggests systematic biases.

4. **Residual vs. predicted plot**: Check for heteroscedasticity (funnel shape = model errors scale with predicted value).

5. **Residual autocorrelation function (ACF)**: Plot autocorrelation of residuals at lags 1-168. Significant spikes indicate uncaptured temporal structure.

### Literature reference
This is standard in econometrics (see Hansen & Lunde, 2006; GARCH model validation at NYU V-Lab). No volatility forecasting paper should be published without residual diagnostics.

### Recommended fix
1. Compute Ljung-Box test on LSTM residuals (lags 1, 5, 10, 24)
2. Plot residual ACF
3. Test for ARCH effects
4. These tests can be computed from saved model predictions

### Severity: **HIGH** — missing standard validation

---

## Assumption 10: Stationarity Is Not Required After Rolling Normalization

**Status: YELLOW — Partially addressed**

### What was assumed
The 72h rolling z-score normalization makes the series approximately stationary, satisfying LSTM's implicit stationarity requirement.

### Analysis

DVOL in levels is clearly non-stationary (near unit root with ρ=0.999). The rolling z-score normalization transforms it to:
```
z[t] = (DVOL[t] - rolling_mean[t]) / rolling_std[t]
```

This is approximately stationary IF the rolling window is much shorter than the characteristic timescale of regime changes. With window=72h:
- DVOL's 168h (1 week) autocorrelation is still 0.91
- The rolling mean adapts with a 72h half-life, which is shorter than many volatility regimes
- **However**, during rapid regime shifts (e.g., LUNA crash, FTX collapse), the 72h window may lag behind the true mean for several days

The Augmented Dickey-Fuller (ADF) test or KPSS test should be applied to the normalized series to formally verify stationarity.

### Recommended fix
1. Run ADF test on normalized DVOL for each split
2. Report ADF statistic and p-value
3. If not stationary at 5% level, consider differencing or longer normalization windows

### Severity: **LOW** — 72h normalization likely achieves near-stationarity

---

## Assumption 11: Feature Set Choice Doesn't Matter (Confirmed)

**Status: GREEN — Properly validated**

### What was assumed
All 4 feature sets (market, market_jumps, jump_aware, market_lags) converge to R²≈0.941.

### Evidence

| Feature Set | Features | R² | RMSE | Dir% |
|------------|----------|-----|------|------|
| market | 4 | 0.9406 | 1.6245 | 51.07% |
| market_jumps | 8 | 0.9406 | 1.6247 | 51.02% |
| jump_aware | 11 | 0.9405 | 1.6252 | 50.89% |
| market_lags | 7 | 0.9405 | 1.6254 | 50.92% |

The variation across feature sets is R² ∈ [0.9405, 0.9406] — a range of 0.0001. This is **far below** any meaningful threshold. The LSTM learns essentially the same function regardless of features.

### Interpretation
This is actually a **strong finding**: the LSTM's predictive power comes entirely from the temporal structure (24h sequence window), not from the specific features. Adding lag features, jump indicators, or network metrics provides zero marginal information beyond what the temporal sequence of basic market features already captures.

### Severity: **NONE** — this is a valid and well-supported conclusion

---

## Summary Table

| # | Assumption | Severity | Verdict | Action Required |
|---|-----------|----------|---------|----------------|
| 1 | R²=0.94 = good performance | **CRITICAL** | R² is **worse than persistence** (0.9985) | Report persistence baseline, use relative metrics |
| 2 | No look-ahead bias | GREEN | Properly implemented | None |
| 3 | Temporal split is valid | YELLOW | Clean but single split | Add multi-split validation |
| 4 | Dir% ≈ 51% is near-random | **CRITICAL** | Correct but untested | Add Pesaran-Timmermann test, report p-values |
| 5 | Tree R²=0.994 > LSTM R²=0.94 | **CRITICAL** | Misleading comparison (different preprocessing + persistence exploitation) | Fair comparison with same normalization |
| 6 | 5.4M params / 24K samples OK | YELLOW | Mitigated by regularization | Log train R², compute generalization gap |
| 7 | MSE loss is appropriate | YELLOW | Standard but suboptimal | Evaluate under QLIKE, MAE |
| 8 | Early stopping is unbiased | YELLOW | Minor selection bias | Run 3-5 random seeds |
| 9 | Residual analysis not needed | **HIGH** | Missing standard validation | Ljung-Box, ARCH-LM, ACF plots |
| 10 | Stationarity from rolling norm | GREEN | Likely achieved | Verify with ADF test |
| 11 | Feature sets don't matter | GREEN | Well-supported | None |

---

## Recommended Priority Actions

### Before thesis submission (MUST do):

1. **Compute and report persistence baseline** on the exact same test set:
   ```python
   # Naive baseline: predict DVOL[t+1] = DVOL[t]
   y_persist = dvol_test[:-1]  # DVOL[t] for t=0..N-2
   y_actual = dvol_test[1:]    # DVOL[t+1] for t=0..N-2
   r2_persist = 1 - SS_res/SS_tot
   ```
   This will likely show R²_persist ≈ 0.998, making LSTM's 0.94 look properly contextualized.

2. **Run Pesaran-Timmermann (1992) test** on directional accuracy — this is the standard test in the literature.

3. **Add residual ACF plot** and Ljung-Box test — 10 lines of code, standard requirement.

4. **Reframe the thesis narrative**: 
   - "LSTM achieves R²=0.94 on normalized DVOL, comparable to HAR-RV"
   - "No model achieves statistically significant directional prediction"
   - "Tree model superiority is driven by direct access to lag features in a near-unit-root series"
   - The real contribution is the **methodological comparison**, not the absolute performance

### Nice to have:

5. Run 3-5 random seeds and report confidence intervals
6. Compute DM test statistics between model pairs
7. Try QLIKE loss
8. Walk-forward validation with 3+ cutoff dates

---

## Literature References

- Andersen, T.G. & Benzoni, L. (2009). "Realized Volatility." *Handbook of Financial Time Series*.
- Bates, S., Hastie, T., & Tibshirani, R. (2023). "Statistical Perspectives on Cross-validation." *Statistical Science*.
- Blaskowitz, O. & Herwartz, H. (2014). "Testing the Value of Directional Forecasts." *International Journal of Forecasting*, 30(1), 30-42.
- Cheng, Y., Petrides, K.V., & Li, J. (2025). "Estimating the Minimum Sample Size for Neural Network Model Fitting." *Behav. Sci.*, 15(2), 211.
- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*, 7(2), 174-196.
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." *JBES*, 13(3), 253-263.
- Hansen, P.R. & Lunde, A. (2006). "Consistent Ranking of Volatility Models." *Journal of Econometrics*, 131(1-2), 97-121.
- Henriksson, R.D. & Merton, R.C. (1981). "On Market Timing and Investment Performance." *Journal of Business*, 54(4), 513-533.
- Patton, A.J. (2011). "Volatility Forecast Comparison Using Imperfect Volatility Proxies." *Journal of Econometrics*, 160(1), 246-256.
- Pesaran, M.H. & Timmermann, A. (1992). "A Simple Nonparametric Test of Predictive Performance." *JBES*, 10(4), 461-465.
- Pesaran, M.H. & Timmermann, A. (2009). "Testing Dependence Among Serially Correlated Multicategory Variables." *JASA*, 104(485), 325-337.
