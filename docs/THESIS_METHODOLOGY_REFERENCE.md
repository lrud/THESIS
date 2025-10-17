# Thesis Writing Reference: Methodology and Implementation

**Purpose:** This document provides detailed records of all methodological decisions, implementations, and findings for thesis writing.

---

## Table of Contents
1. [Research Question](#research-question)
2. [Data Sources and Collection](#data-sources-and-collection)
3. [Feature Engineering](#feature-engineering)
4. [Preliminary Statistical Analysis](#preliminary-statistical-analysis)
5. [Model Selection Rationale](#model-selection-rationale)
6. [Critical Issue: Non-Stationarity](#critical-issue-non-stationarity)
7. [Solution Implementation](#solution-implementation)
8. [Model Architecture](#model-architecture)
9. [Training Procedure](#training-procedure)
10. [Evaluation Metrics](#evaluation-metrics)

---

## 1. Research Question

**Primary:** Can we forecast Bitcoin implied volatility (DVOL) 24 hours ahead using on-chain metrics and historical volatility data?

**Secondary:** What is the relative importance of different predictors in DVOL forecasting?

---

## 2. Data Sources and Collection

### 2.1 Data Sources

| Variable | Source | API/Method | Frequency | Coverage |
|----------|--------|------------|-----------|----------|
| DVOL (Deribit Volatility Index) | Deribit API | RESTful API | Hourly | May 2025 - Present |
| DVOL (Historical) | ResearchBitcoin.com | Paid API | Hourly | Apr 2021 - Oct 2025 |
| Transaction Volume | ResearchBitcoin.com | Paid API | Hourly | Apr 2021 - Oct 2025 |
| Active Addresses | ResearchBitcoin.com | Paid API | Hourly | Apr 2021 - Oct 2025 |
| NVRV Ratio | ResearchBitcoin.com | Paid API | Hourly | Apr 2021 - Oct 2025 |
| Realized Volatility | Calculated | From DVOL | Hourly | Apr 2021 - Oct 2025 |

### 2.2 Options Open Interest Investigation

**Challenge:** Historical options open interest data not freely available.

**Sources Investigated:**
1. Deribit API - Only current contracts, no historical OI
2. Deribit History API - Has expired contracts but no open_interest field
3. Databento - US equities only, no crypto coverage
4. Tardis.dev - Has OI data but prohibitively expensive (~$100s/month)
5. BarendPotijk GitHub - Wraps History API, no additional OI data
6. OptionsDX - Has historical OI, cost ~$40-50 for 2-3 years

**Decision:** Proceed without options OI for baseline model. Document as limitation and potential future enhancement.

**Documentation:** `docs/HISTORICAL_OI_INVESTIGATION.md`

---

## 3. Feature Engineering

### 3.1 Core Predictors (5 variables)

| Feature | Formula | Rationale | Literature Support |
|---------|---------|-----------|-------------------|
| `dvol_lag_1d` | DVOL at t-1 day | Volatility persistence | Autoregressive behavior of volatility |
| `dvol_lag_7d` | DVOL at t-7 days | Weekly patterns | Weekly seasonality in crypto markets |
| `dvol_lag_30d` | DVOL at t-30 days | Monthly trends | Longer-term momentum |
| `transaction_volume` | Daily transaction volume (USD) | Market activity proxy | Higher volume → higher uncertainty |
| `network_activity` | Active addresses per day | User engagement | Network growth correlates with volatility |

### 3.2 Derived Features (2 variables)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `nvrv` | Network Value to Realized Value | Market valuation metric, overvaluation → volatility |
| `dvol_rv_spread` | DVOL - RV_30d | Volatility risk premium, fear gauge |

### 3.3 Tentative 6th Predictor

**Options Open Interest:** Pending data acquisition (OptionsDX or current Deribit data only)

### 3.4 Feature Processing

```python
# Lag features created from DVOL
df['dvol_lag_1d'] = df['dvol'].shift(24)   # 24 hours = 1 day
df['dvol_lag_7d'] = df['dvol'].shift(168)  # 168 hours = 7 days
df['dvol_lag_30d'] = df['dvol'].shift(720) # 720 hours = 30 days

# Volatility risk premium
df['realized_volatility_30d'] = df['dvol'].rolling(720).std() * np.sqrt(365)
df['dvol_rv_spread'] = df['dvol'] - df['realized_volatility_30d']
```

---

## 4. Preliminary Statistical Analysis

### 4.1 Descriptive Statistics

**Dataset:** 37,927 hourly observations (April 2021 - October 2025)

| Variable | Mean | Std | Min | Max | Skewness | Kurtosis |
|----------|------|-----|-----|-----|----------|----------|
| dvol | 62.56 | 18.73 | 31.47 | 166.39 | 0.78 | 0.44 |
| transaction_volume | 3.15B | 3.82B | 26.5M | 125.7B | 6.02 | 93.85 |
| active_addresses | 15,674 | 9,647 | 244 | 98,585 | 1.77 | 4.58 |

**Key Observations:**
- DVOL shows moderate volatility (CV = 0.30)
- Transaction volume highly skewed (requires transformation)
- All variables show non-normal distributions

**Reference:** `results/csv/summary_statistics.csv`

### 4.2 Correlation Analysis

**Correlations with DVOL:**

| Predictor | Correlation | Interpretation |
|-----------|-------------|----------------|
| dvol_lag_1d | **0.982** | Very strong (1-day persistence) |
| dvol_lag_7d | **0.910** | Strong (weekly persistence) |
| dvol_lag_30d | **0.796** | Strong (monthly persistence) |
| transaction_volume | 0.365 | Moderate positive |
| active_addresses | -0.308 | Moderate negative |
| nvrv | 0.005 | Negligible |
| dvol_rv_spread | -0.087 | Weak negative |

**Finding:** Lagged DVOL features are strongest predictors, supporting autoregressive component in model.

**Reference:** `results/csv/correlation_matrix.csv`, `results/visualizations/correlation_heatmap.png`

### 4.3 Multicollinearity Assessment (VIF)

| Variable | VIF | Assessment |
|----------|-----|------------|
| dvol_lag_1d | 4.26 | Acceptable (< 5) |
| nvrv | 3.40 | Acceptable |
| active_addresses | 2.93 | Acceptable |
| transaction_volume | 2.00 | Acceptable |
| dvol_rv_spread | 1.29 | Acceptable |

**Conclusion:** No multicollinearity issues. All features can be included.

**Reference:** `results/csv/vif_analysis.csv`

### 4.4 Normality Tests (Jarque-Bera, Shapiro-Wilk, D'Agostino-Pearson)

**Result:** ALL variables reject normality (p < 0.05)

**Implication:** 
- Violates assumptions for linear regression
- Supports use of neural networks (distribution-agnostic)

**Reference:** `results/csv/normality_tests.csv`

### 4.5 Stationarity Tests (ADF, KPSS)

**Result:** ALL variables non-stationary (KPSS consistently rejects stationarity)

**Implication:** 
- Cannot use models requiring stationarity without transformation
- Led to critical issue discovery (see Section 6)

**Reference:** `results/csv/stationarity_tests.csv`

### 4.6 Linearity Tests (Ramsey RESET)

**Result:** ALL relationships non-linear (RESET test rejects linearity, p < 0.001)

| Predictor | Linear R² | RESET p-value | Conclusion |
|-----------|-----------|---------------|------------|
| dvol_lag_1d | 0.964 | 1.11e-16 | Non-linear |
| transaction_volume | 0.133 | 1.11e-16 | Non-linear |
| active_addresses | 0.095 | 1.11e-16 | Non-linear |

**Implication:** Strongly supports LSTM/neural network choice over linear models.

**Reference:** `results/csv/linearity_tests.csv`, `results/visualizations/scatter_plots_linearity.png`

---

## 5. Model Selection Rationale

### 5.1 Why LSTM?

Statistical analysis supports LSTM architecture:

| Requirement | LSTM Capability | Evidence |
|-------------|-----------------|----------|
| **Non-linear relationships** | Native non-linear activations (tanh, sigmoid) | All RESET tests reject linearity |
| **Non-stationary data** | Can learn time-varying patterns | All KPSS tests reject stationarity |
| **Non-normal distributions** | Distribution-agnostic | All normality tests reject |
| **Temporal dependencies** | Recurrent connections with memory | High autocorrelation in DVOL |
| **Long-term patterns** | Cell state preserves information | 30-day lag still predictive (r=0.80) |

### 5.2 Alternatives Considered

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| Linear Regression | Simple, interpretable | Assumes linearity (violated) | ❌ Not suitable |
| ARIMA | Classical time series | Requires stationarity | ❌ Data non-stationary |
| Random Forest | Handles non-linearity | No temporal structure | ❌ Ignores sequence |
| Transformer | State-of-art for sequences | Needs more data | ⚠️ Future work |
| GRU | Simpler than LSTM | Less memory capacity | ⚠️ Future comparison |

**Decision:** LSTM is theoretically and empirically justified for this problem.

---

## 6. Critical Issue: Non-Stationarity

### 6.1 Problem Discovery

**Observation:** Baseline LSTM model produced straight-line predictions despite training successfully.

**Investigation Steps:**
1. Visual inspection of prediction plots
2. Analysis of model outputs (constant values)
3. Data distribution analysis across time

**Root Cause:**
- DVOL has strong **downward trend** from 2021-2025
- Train period mean: 69.32
- Test period mean: 47.40 (32% decrease)
- Global normalization using train stats caused severe distribution shift

**Detailed Analysis:** `docs/CRITICAL_ISSUE_NON_STATIONARITY.md`

### 6.2 Why This Caused Failure

**Normalization Formula:**
```python
y_normalized = (y - train_mean) / train_std
             = (y - 69.32) / 20.06
```

**For test data (mean = 47.40):**
```python
y_test_normalized = (47.40 - 69.32) / 20.06
                  ≈ -1.09  # Heavily negative!
```

**Consequences:**
- Model trained on data centered at 0
- Test data centered at -1.09 (1 std deviation below training mean)
- Model never saw such systematically low values
- Defaulted to predicting near training mean
- Result: Flat predictions, negative R², zero directional accuracy

**Visual Evidence:** `results/visualizations/dvol_temporal_trend.png`

---

## 7. Solution Implementation

### 7.1 Transformation: First Differences

**Formula:**
```python
Δdvol_t = dvol_t - dvol_{t-1}
```

**Why This Works:**
1. **Removes trend:** Differencing eliminates deterministic trends
2. **Achieves stationarity:** Δdvol has mean ≈ 0 across all periods
3. **Eliminates distribution shift:** Train and test come from same distribution
4. **Preserves information:** Can reconstruct original via cumulative sum

**Verification (Post-Differencing):**
```
Train Δdvol mean: -0.0008 (vs. 69.32 absolute)
Val Δdvol mean:   -0.0004 (vs. 57.51 absolute)
Test Δdvol mean:  -0.0021 (vs. 47.40 absolute)
✅ All near zero = Stationary!
```

### 7.2 Reconstruction Formula

**After prediction:**
```python
dvol_t = dvol_{t-1} + Δdvol_t_predicted
```

**Cumulative reconstruction:**
```python
dvol_t = dvol_0 + Σ(Δdvol_i) for i=1 to t
```

### 7.3 Implementation

**New Module:** `scripts/modeling/data_loader_differenced.py`

**Key Functions:**
- `create_sequences_differenced()` - Creates sequences with Δdvol as target
- `reconstruct_from_diff()` - Converts predictions back to original scale
- `prepare_data_differenced()` - Complete pipeline

**Code Snippet:**
```python
# Create difference
df['dvol_diff'] = df['dvol'].diff()

# Store previous value for reconstruction
y_prev = df['dvol'].iloc[target_idx - 1]

# After prediction
dvol_predicted = y_prev + diff_predicted
```

---

## 8. Model Architecture

### 8.1 LSTM Specifications

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

### 8.2 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Input size** | 7 | Number of features |
| **Hidden size** | 128 | Balance capacity vs. overfitting |
| **Num layers** | 2 | Deep enough for complex patterns |
| **Dropout** | 0.3 | Regularization (prevent overfitting) |
| **L2 regularization** | 1e-4 | Weight decay penalty |
| **Sequence length** | 24 hours | 1 day lookback window |
| **Forecast horizon** | 24 hours | 1 day ahead prediction |
| **Batch size** | 64 | GPU memory efficiency |
| **Learning rate** | 0.001 | Adam optimizer default |
| **LR schedule** | ReduceLROnPlateau | Adaptive learning rate |
| **Early stopping** | Patience = 15 | Prevent overtraining |

### 8.3 Architecture Justification

**LSTM Layers:**
- Layer 1: Extract temporal features from input sequences
- Layer 2: Learn higher-level temporal abstractions

**Dropout:**
- Applied after LSTM and FC1 layers
- Prevents co-adaptation of neurons
- 30% rate based on standard practice

**Fully Connected Layers:**
- FC1 (128→64): Non-linear transformation
- ReLU: Introduce non-linearity
- FC2 (64→1): Final prediction

**L2 Regularization:**
- Applied to all weight matrices
- Penalizes large weights
- Prevents overfitting to noise

---

## 9. Training Procedure

### 9.1 Data Split

**Method:** Temporal split (no shuffling)

| Split | Proportion | Samples | Date Range |
|-------|------------|---------|------------|
| Train | 60% | 22,741 | Apr 2021 - Dec 2023 |
| Validation | 20% | 7,581 | Jan 2024 - Nov 2024 |
| Test | 20% | 7,581 | Nov 2024 - Oct 2025 |

**Rationale:**
- Temporal split preserves time series order
- No data leakage from future to past
- Test set represents most recent data (deployment scenario)

### 9.2 Normalization

**Method:** StandardScaler (Z-score normalization)

**Formula:**
```python
x_normalized = (x - μ_train) / σ_train
```

**Critical:** Fit only on training data, transform all splits using train statistics.

**For Differenced Model:**
- Features: Normalized as usual
- Target (Δdvol): Normalized using train Δdvol statistics (mean ≈ 0)

### 9.3 Training Loop

**Optimizer:** Adam
- Adaptive learning rates per parameter
- Momentum term for faster convergence
- Default β1=0.9, β2=0.999

**Loss Function:** MSE (Mean Squared Error)
```python
Loss = (1/n) * Σ(y_pred - y_actual)²
```

**Learning Rate Scheduler:** ReduceLROnPlateau
- Monitors validation loss
- Reduces LR by factor=0.5 when plateau detected
- Patience = 5 epochs

**Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Prevents exploding gradients
- Stabilizes training

**Early Stopping:**
- Monitors validation loss
- Stops if no improvement for 15 epochs
- Saves best model (lowest val loss)

### 9.4 Training Environment

**Hardware:**
- 2x AMD Radeon RX 7900 XT GPUs
- gfx1100 architecture
- ROCm 7.0 runtime

**Software:**
- PyTorch 2.10.0.dev20251015+rocm7.0
- Python 3.12.3
- CUDA API (ROCm provides CUDA compatibility)

**Training Time:**
- Baseline model: ~22 epochs (~15-20 minutes)
- Differenced model: [To be recorded]

---

## 10. Evaluation Metrics

### 10.1 Regression Metrics

**1. Root Mean Squared Error (RMSE)**
```python
RMSE = √[(1/n) * Σ(y_pred - y_actual)²]
```
- Penalizes large errors more than small errors
- Same units as target (DVOL percentage points)

**2. Mean Absolute Error (MAE)**
```python
MAE = (1/n) * Σ|y_pred - y_actual|
```
- Average absolute difference
- Less sensitive to outliers than RMSE

**3. Mean Absolute Percentage Error (MAPE)**
```python
MAPE = (100/n) * Σ|((y_actual - y_pred) / y_actual)|
```
- Scale-independent metric
- Interpretable as percentage error

**4. R² (Coefficient of Determination)**
```python
R² = 1 - (SS_residual / SS_total)
   = 1 - [Σ(y_actual - y_pred)² / Σ(y_actual - ȳ)²]
```
- Proportion of variance explained
- Range: (-∞, 1], where 1 is perfect fit
- **Negative R²** means worse than predicting mean

### 10.2 Time Series Specific Metrics

**5. Directional Accuracy**
```python
DA = (100/n) * Σ[sign(y_actual[t] - y_actual[t-1]) == 
                  sign(y_pred[t] - y_pred[t-1])]
```
- Percentage of correct direction predictions
- Critical for trading applications
- Random guessing = 50%

### 10.3 Baseline Model Results

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| RMSE | 20.05 | 14.53 | **23.44** |
| MAE | 16.79 | 13.13 | **21.69** |
| MAPE | 27.38% | 24.82% | **51.02%** |
| R² | -0.0001 | -1.73 | **-5.92** |
| Dir. Acc. | 0.02% | 1.53% | **2.16%** |

**Analysis:**
- Negative R² = predictions worse than mean baseline
- Near-zero directional accuracy = no predictive power
- Caused by non-stationarity issue

**Reference:** `results/BASELINE_LSTM_SUMMARY.md`

### 10.4 Differenced Model Results

[To be filled after training completes]

---

## 11. Key Methodological Decisions Summary

### What Went Right
✅ Comprehensive preliminary statistical analysis  
✅ Identified LSTM as theoretically justified choice  
✅ Proper temporal splitting (no data leakage)  
✅ Thorough documentation of process  
✅ Discovered and diagnosed critical non-stationarity issue  
✅ Implemented principled solution (differencing)  

### What Went Wrong (Initially)
❌ Did not check for temporal trends in target variable  
❌ Applied global normalization without verifying distribution consistency  
❌ Assumed stationarity without explicit testing  

### Lessons for Thesis
1. **Always visualize data over time** before modeling
2. **Check statistical properties by split** (train/val/test)
3. **Test assumptions explicitly** (don't assume stationarity)
4. **Document failures and fixes** (shows research process)
5. **Iterate and improve** (baseline → differenced model)

---

## 12. File Reference Index

### Documentation
- `docs/COVARIATE_MATH.md` - Mathematical formulation of features
- `docs/HISTORICAL_OI_INVESTIGATION.md` - Options OI data source investigation
- `docs/CRITICAL_ISSUE_NON_STATIONARY_TARGET.md` - Non-stationarity analysis
- `docs/MODEL_TRAINING_PROGRESS_LOG.md` - Training session log
- `docs/THESIS_METHODOLOGY_REFERENCE.md` - This file
- `results/BASELINE_LSTM_SUMMARY.md` - Baseline model comprehensive results

### Code
- `scripts/modeling/data_loader.py` - Baseline data pipeline
- `scripts/modeling/data_loader_differenced.py` - Differenced data pipeline
- `scripts/modeling/model.py` - LSTM architecture
- `scripts/modeling/trainer.py` - Training utilities
- `scripts/modeling/evaluator.py` - Evaluation and visualization
- `scripts/modeling/main.py` - Baseline training script
- `scripts/modeling/main_differenced.py` - Differenced training script
- `scripts/analysis/comprehensive_variable_analysis.py` - Statistical analysis
- `scripts/analysis/model_diagnostics.py` - Diagnostic tools

### Data
- `data/processed/bitcoin_lstm_features.csv` - Final feature set
- `results/csv/summary_statistics.csv` - Descriptive statistics
- `results/csv/correlation_matrix.csv` - Correlation analysis
- `results/csv/normality_tests.csv` - Distribution tests
- `results/csv/stationarity_tests.csv` - Stationarity tests
- `results/csv/linearity_tests.csv` - Linearity tests
- `results/csv/vif_analysis.csv` - Multicollinearity assessment

### Visualizations
- `results/visualizations/correlation_heatmap.png` - Feature correlations
- `results/visualizations/scatter_plots_linearity.png` - Linearity assessment
- `results/visualizations/dvol_temporal_trend.png` - Proof of trend issue
- `results/visualizations/lstm_training_history.png` - Baseline training curves
- `results/visualizations/lstm_*_predictions.png` - Baseline predictions
- `results/visualizations/lstm_differenced_*` - Differenced model results

### Models
- `models/lstm_baseline_best.pth` - Baseline model checkpoint
- `models/lstm_differenced_best.pth` - Differenced model checkpoint

---

**Document Status:** Living document, updated throughout research process  
**Last Updated:** October 16, 2025  
**For:** Master's Thesis in [Your Program]  
**Author:** [Your Name]
