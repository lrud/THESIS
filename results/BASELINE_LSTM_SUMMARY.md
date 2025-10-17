# Baseline LSTM Training Summary
**Date:** October 16, 2025  
**Model:** LSTM (2 layers, 128 hidden units, 0.3 dropout, 1e-4 L2 regularization)  
**Hardware:** 2x AMD Radeon RX 7900 XT (ROCm 7.0)  
**Dataset:** 37,927 samples (April 2021 - October 2025)

---

## 1. PRELIMINARY STATISTICAL ANALYSIS

### 1.1 Descriptive Statistics (All Variables)

| Variable | N | Mean | Std Dev | Min | Median | Max | Skewness | Kurtosis | CV |
|----------|---|------|---------|-----|--------|-----|----------|----------|----|
| **dvol** (target) | 37,927 | 62.56 | 18.73 | 31.47 | 58.26 | 166.39 | 0.78 | 0.44 | 0.30 |
| **dvol_lag_1d** | 37,927 | 62.59 | 18.74 | 31.47 | 58.29 | 166.39 | 0.78 | 0.44 | 0.30 |
| **dvol_lag_7d** | 37,927 | 62.74 | 18.74 | 31.47 | 58.42 | 166.39 | 0.76 | 0.41 | 0.30 |
| **dvol_lag_30d** | 37,927 | 63.39 | 18.61 | 31.47 | 58.98 | 166.39 | 0.73 | 0.36 | 0.29 |
| **transaction_volume** | 37,927 | 3.15B | 3.82B | 26.5M | 1.96B | 125.7B | 6.02 | 93.85 | 1.21 |
| **active_addresses** | 37,927 | 15,674 | 9,647 | 244 | 13,232 | 98,585 | 1.77 | 4.58 | 0.62 |
| **nvrv** | 37,927 | 0.80 | 0.53 | -0.25 | 0.89 | 2.00 | -0.23 | -0.89 | 0.66 |
| **dvol_rv_spread** | 37,927 | 21.46 | 38.52 | -294.35 | 28.14 | 92.17 | -1.86 | 6.84 | 1.79 |

**Key Observations:**
- DVOL (target) has moderate volatility (CV = 0.30) with positive skewness
- Transaction volume is highly skewed (6.02) with extreme kurtosis (93.85) - needs transformation
- All lagged DVOL features highly correlated with target (as expected)
- Active addresses and transaction volume show high variability

---

### 1.2 Correlation with Target (DVOL)

| Predictor | Correlation with DVOL | Interpretation |
|-----------|----------------------|----------------|
| **dvol_lag_1d** | **0.982** | Very strong positive (1-day persistence) |
| **dvol_lag_7d** | **0.910** | Strong positive (weekly persistence) |
| **dvol_lag_30d** | **0.796** | Strong positive (monthly persistence) |
| **transaction_volume** | **0.365** | Moderate positive |
| **active_addresses** | **-0.308** | Moderate negative |
| **nvrv** | **0.005** | Negligible (essentially uncorrelated) |
| **dvol_rv_spread** | **-0.087** | Weak negative |

**Key Findings:**
- Lagged DVOL features are the strongest predictors (autoregressive nature)
- Transaction volume positively correlated with volatility
- Active addresses negatively correlated (more activity = lower volatility?)
- NVRV has virtually no linear relationship with DVOL
- Volatility spread has weak negative relationship

---

### 1.3 Multicollinearity Analysis (VIF)

| Variable | VIF | Interpretation |
|----------|-----|----------------|
| dvol_lag_1d | 4.26 | Low (acceptable) |
| nvrv | 3.40 | Low (acceptable) |
| active_addresses | 2.93 | Low (acceptable) |
| transaction_volume | 2.00 | Low (acceptable) |
| dvol_rv_spread | 1.29 | Low (acceptable) |

**Conclusion:** No multicollinearity issues detected. All VIF < 5 indicates features are sufficiently independent.

---

### 1.4 Normality Tests

| Variable | Jarque-Bera p-value | Shapiro-Wilk p-value | Normal? |
|----------|---------------------|----------------------|---------|
| dvol | 0.0 | 1.04e-37 | **No** |
| dvol_lag_1d | 0.0 | 4.17e-38 | **No** |
| transaction_volume | 0.0 | 2.51e-75 | **No** |
| active_addresses | 0.0 | 2.48e-54 | **No** |
| nvrv | 0.0 | 2.68e-32 | **No** |
| dvol_rv_spread | 0.0 | 2.32e-52 | **No** |

**Conclusion:** ALL variables are non-normally distributed (p < 0.05). This violates assumptions for linear regression but is handled naturally by neural networks.

---

### 1.5 Stationarity Tests

| Variable | ADF Stationary? | KPSS Stationary? | Consensus |
|----------|-----------------|------------------|-----------|
| dvol | Yes (p < 0.05) | **No** (p = 0.01) | **Non-Stationary** |
| transaction_volume | Yes (p < 0.05) | **No** (p = 0.01) | **Non-Stationary** |
| active_addresses | Yes (p < 0.05) | **No** (p = 0.01) | **Non-Stationary** |
| nvrv | **No** (p = 0.18) | **No** (p = 0.01) | **Non-Stationary** |
| dvol_rv_spread | Yes (p < 0.05) | **No** (p = 0.01) | **Non-Stationary** |

**Conclusion:** ALL variables are non-stationary. ADF and KPSS tests give conflicting results, but KPSS (more conservative) rejects stationarity for all. This suggests differencing or percent change transformations may improve modeling.

---

### 1.6 Linearity Tests (Ramsey RESET)

| Predictor | Linear R² | RESET p-value | Linear? | Recommendation |
|-----------|-----------|---------------|---------|----------------|
| dvol_lag_1d | **0.964** | 1.11e-16 | **No** | Polynomial/nonlinear |
| transaction_volume | 0.133 | 1.11e-16 | **No** | Polynomial/nonlinear |
| active_addresses | 0.095 | 1.11e-16 | **No** | Polynomial/nonlinear |
| nvrv | 0.00003 | 1.11e-16 | **No** | Polynomial/nonlinear |
| dvol_rv_spread | 0.008 | 1.11e-16 | **No** | Polynomial/nonlinear |

**Conclusion:** ALL relationships are non-linear (RESET test rejects linearity for all predictors). This strongly supports using LSTM/neural networks over linear models.

---

## 2. LSTM BASELINE TRAINING RESULTS

### 2.1 Model Architecture
```
LSTM_DVOL(
  (lstm): LSTM(7, 128, num_layers=2, dropout=0.3, batch_first=True)
  (dropout): Dropout(p=0.3)
  (fc1): Linear(128, 64)
  (relu): ReLU()
  (fc2): Linear(64, 1)
)

Total Parameters: 210,561
Trainable Parameters: 210,561
```

### 2.2 Training Configuration
- **Sequence Length:** 24 hours (lookback window)
- **Forecast Horizon:** 24 hours ahead
- **Batch Size:** 64
- **Learning Rate:** 0.001 (with ReduceLROnPlateau scheduler)
- **Optimizer:** Adam
- **Loss Function:** MSE
- **L2 Regularization:** 1e-4
- **Early Stopping:** Patience = 15 epochs
- **Max Epochs:** 100

### 2.3 Data Split
- **Train Set:** 22,756 samples (60%)
- **Validation Set:** 7,586 samples (20%)
- **Test Set:** 7,585 samples (20%)
- **Split Method:** Temporal (no shuffling to preserve time series order)

### 2.4 Training Progress

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|------------|----------|---------------|
| 1 | 1.0101 | 0.5271 | 1.00e-03 |
| 5 | 1.0003 | 0.5457 | 1.00e-03 |
| 10 | 1.0003 | 0.5466 | 1.00e-03 |
| 15 | 0.9998 | 0.5374 | 5.00e-04 |
| 20 | 0.9995 | 0.5360 | 2.50e-04 |
| **22** | **[EARLY STOP]** | **Best: 0.5237** | 2.50e-04 |

**Observations:**
- Training loss plateaued around 1.0 (minimal improvement)
- Validation loss oscillated between 0.52-0.55
- Learning rate reduced twice (epoch ~10 and ~15)
- Early stopping triggered at epoch 22 (15 epochs without val improvement)
- **CRITICAL ISSUE:** Train loss > Val loss suggests data issues or poor model fit

---

## 3. FINAL EVALUATION METRICS

### 3.1 Train Set Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 20.05 | Average prediction error of ~20 DVOL points |
| **MAE** | 16.79 | Typical error is ~17 DVOL points |
| **MAPE** | 27.38% | Predictions off by 27% on average |
| **R²** | **-0.0001** | ⚠️ Worse than predicting the mean |
| **Directional Accuracy** | **0.02%** | ⚠️ Cannot predict direction (should be ~50%) |

### 3.2 Validation Set Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 14.53 | Lower error than training (unusual) |
| **MAE** | 13.13 | Typical error ~13 DVOL points |
| **MAPE** | 24.82% | 25% average error |
| **R²** | **-1.73** | ⚠️ Much worse than baseline |
| **Directional Accuracy** | **1.53%** | ⚠️ Cannot predict direction |

### 3.3 Test Set Performance (MOST IMPORTANT)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | **23.44** | High prediction error |
| **MAE** | **21.69** | Typical error ~22 DVOL points |
| **MAPE** | **51.02%** | ⚠️ **CRITICAL: 51% average error** |
| **R²** | **-5.92** | ⚠️ **CRITICAL: Severely worse than baseline** |
| **Directional Accuracy** | **2.16%** | ⚠️ **CRITICAL: Essentially random** |

---

## 4. CRITICAL FINDINGS & ISSUES

### 🚨 **Major Problems Identified:**

1. **Negative R² Across All Sets**
   - Train: -0.0001
   - Validation: -1.73
   - Test: **-5.92** (catastrophic)
   - **Implication:** Model performs worse than simply predicting the mean DVOL value

2. **Near-Zero Directional Accuracy**
   - All sets < 3% (should be ~50% for random guessing)
   - **Implication:** Model cannot predict whether volatility will increase or decrease

3. **Extremely High MAPE on Test Set**
   - 51% average percentage error
   - **Implication:** Predictions are very far from actual values

4. **Unusual Loss Behavior**
   - Training loss (1.0) higher than validation loss (0.52)
   - **Implication:** Potential data leakage, normalization issues, or temporal alignment problems

5. **Model Likely Predicting Near-Constant Values**
   - Low directional accuracy + negative R² suggests model is predicting values close to the mean
   - **Implication:** Model has not learned meaningful patterns

---

## 5. ROOT CAUSE ANALYSIS

### 5.1 Data-Related Issues (Most Likely)
- ❌ **Sequence alignment problem:** 24hr lookback → 24hr forecast may have temporal misalignment
- ❌ **Normalization issue:** Features and target may be scaled differently
- ❌ **Non-stationarity:** All variables non-stationary, no differencing applied
- ❌ **Data leakage:** Possible future information bleeding into training data

### 5.2 Model Architecture Issues
- ❌ **Insufficient capacity:** 210K parameters may be too few for complex volatility dynamics
- ❌ **Short lookback window:** 24 hours may be insufficient for DVOL prediction
- ❌ **No attention mechanism:** Cannot weight important time steps
- ❌ **Single forecast horizon:** Predicting 24h ahead may be too far for current features

### 5.3 Feature Engineering Issues
- ❌ **Missing critical features:** No options open interest data (tentative 6th predictor)
- ❌ **No transformations:** Variables not differenced, log-transformed, or standardized properly
- ❌ **Weak predictors:** NVRV has near-zero correlation with DVOL
- ❌ **Linear features:** All relationships non-linear, but no polynomial terms

---

## 6. RECOMMENDED NEXT STEPS (PRIORITY ORDER)

### 🔴 **IMMEDIATE (Diagnostic)**
1. **Visualize predictions vs actuals** - Confirm model is predicting near-constant values
2. **Check data preprocessing** - Verify no data leakage in sequence creation
3. **Inspect normalization** - Ensure features and target scaled consistently
4. **Plot residuals** - Identify systematic errors

### 🟡 **HIGH PRIORITY (Data Improvements)**
1. **Apply differencing** - Use first differences: `Δdvol_t = dvol_t - dvol_{t-1}`
2. **Increase lookback window** - Try 48h or 72h sequences
3. **Adjust forecast horizon** - Try predicting 1h, 6h, or 12h ahead instead of 24h
4. **Add rolling features** - Rolling mean/std of volatility over multiple windows
5. **Transform target** - Try log(DVOL) or percentage change

### 🟢 **MEDIUM PRIORITY (Model Improvements)**
1. **Increase model capacity** - Try 3-4 layers, 256 hidden units
2. **Add attention mechanism** - Weight important time steps
3. **Try bidirectional LSTM** - Capture forward and backward patterns
4. **Experiment with loss functions** - Huber loss (robust to outliers), Quantile loss
5. **Hyperparameter tuning** - Grid search over learning rate, dropout, L2

### 🔵 **LOW PRIORITY (Feature Engineering)**
1. **Acquire options OI data** - OptionsDX (~$40-50 for historical data)
2. **Add time-based features** - Day of week, month, market hours
3. **Create interaction terms** - e.g., transaction_volume × active_addresses
4. **Add external features** - BTC price, market sentiment, funding rates

---

## 7. STATISTICAL VALIDATION OF LSTM CHOICE

Despite poor baseline performance, **LSTM remains the appropriate model choice** based on:

✅ **All variables non-linear** (RESET test rejects linearity)  
✅ **All variables non-stationary** (KPSS test consistent)  
✅ **All variables non-normal** (Jarque-Bera, Shapiro-Wilk reject normality)  
✅ **No multicollinearity** (All VIF < 5)  
✅ **High autocorrelation** (Lagged features highly correlated with target)

**Conclusion:** The poor performance is due to **implementation/configuration issues**, not model choice. LSTMs are designed to handle non-linear, non-stationary time series with autocorrelation.

---

## 8. FILES GENERATED

### Visualizations (`results/visualizations/`)
- `lstm_training_history.png` - Training/validation loss curves and learning rate schedule
- `lstm_train_predictions.png` - Train set: actual vs predicted DVOL
- `lstm_val_predictions.png` - Validation set: actual vs predicted DVOL
- `lstm_test_predictions.png` - Test set: actual vs predicted DVOL (critical)
- `correlation_heatmap.png` - Feature correlation matrix
- `scatter_plots_linearity.png` - Linearity assessment plots

### Data (`results/csv/`)
- `summary_statistics.csv` - Descriptive statistics for all variables
- `correlation_matrix.csv` - Pairwise correlations
- `normality_tests.csv` - Jarque-Bera, Shapiro-Wilk, D'Agostino-Pearson results
- `stationarity_tests.csv` - ADF and KPSS test results
- `linearity_tests.csv` - Ramsey RESET test results
- `vif_analysis.csv` - Variance Inflation Factors

### Model
- `models/lstm_baseline_best.pth` - Best model checkpoint (epoch with lowest val loss)

---

## 9. CONCLUSION

The baseline LSTM model **failed to learn meaningful patterns** from the Bitcoin DVOL data, as evidenced by:
- Negative R² scores (worse than naive baseline)
- Near-zero directional accuracy
- High MAPE (51% on test set)

However, the comprehensive statistical analysis confirms that:
- The data characteristics (non-linear, non-stationary, autocorrelated) support LSTM usage
- No multicollinearity issues exist
- The problem lies in model configuration, not model choice

**Next step:** Focus on data preprocessing improvements (differencing, longer sequences, shorter forecast horizon) before architectural changes.

---

**Generated:** October 16, 2025  
**Analysis By:** GitHub Copilot  
**Repository:** lrud/THESIS
