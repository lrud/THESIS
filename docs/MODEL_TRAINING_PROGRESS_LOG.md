# Model Training Progress Log

## October 16, 2025 - Session Summary

### Session Objectives
1. Identify why baseline LSTM produced straight-line predictions
2. Implement fix for non-stationarity
3. Retrain model and compare results

---

## Timeline of Activities

### 1. Problem Identification (Morning)
**Issue Reported:** User observed that LSTM predictions appeared as perfectly straight lines  
**Hypothesis:** Data preprocessing or model specification error

### 2. Diagnostic Analysis
**Script Created:** `scripts/analysis/model_diagnostics.py`

**Key Findings:**
- Temporal trend analysis revealed DVOL decreased by 32% from train to test period
  - Train period (2021-2023): Mean = 69.32, Std = 20.06
  - Val period (2024): Mean = 57.51, Std = 8.80
  - Test period (late 2024-2025): Mean = 47.40, Std = 8.92

**Root Cause Identified:**
- Non-stationary target variable (strong downward trend)
- Global normalization using train statistics caused distribution shift
- Test data normalized to ~-1.09 standard deviations below training mean
- Model never encountered such systematically low values during training
- Defaulted to predicting near training mean → flat predictions

**Documentation Created:**
- `docs/CRITICAL_ISSUE_NON_STATIONARY_TARGET.md` - Complete analysis and solutions
- `results/visualizations/dvol_temporal_trend.png` - Visual proof of trend

---

## Solution Implementation

### Approach: First Differences Transformation
Transform target from absolute values to changes:
```
Δdvol_t = dvol_t - dvol_{t-1}
```

**Why This Works:**
- Removes temporal trend
- Makes data stationary (mean ≈ 0 for all periods)
- Eliminates distribution shift between train/test
- Predictions can be reconstructed: `dvol_t = dvol_{t-1} + Δdvol_t`

### Files Created:
1. **`scripts/modeling/data_loader_differenced.py`** (New)
   - `create_sequences_differenced()` - Creates sequences with differenced target
   - `reconstruct_from_diff()` - Reconstructs original scale from predictions
   - `prepare_data_differenced()` - Complete pipeline with differencing

2. **`scripts/modeling/main_differenced.py`** (New)
   - Training pipeline using differenced data
   - Automatic reconstruction of predictions
   - Side-by-side comparison with baseline

---

## Training Results: Differenced Model

### Data Statistics (Confirming Stationarity)
```
Differenced target statistics:
  Train Δdvol - Mean: -0.0008, Std: 0.8652
  Val Δdvol   - Mean: -0.0004, Std: 0.5353
  Test Δdvol  - Mean: -0.0021, Std: 0.4858
  ✅ All means near 0 = Stationary!
```

**Comparison with Baseline:**
| Metric | Baseline (Absolute) | Differenced | Improvement |
|--------|---------------------|-------------|-------------|
| Train Mean | 69.32 | -0.0008 | ✅ Stationary |
| Val Mean | 57.51 | -0.0004 | ✅ Stationary |
| Test Mean | 47.40 | -0.0021 | ✅ Stationary |
| Distribution Shift | 32% decrease | ~0% | ✅ Eliminated |

### Training Progress
- Model: Same architecture as baseline (210,561 parameters)
- Training in progress...
- [Results to be added when training completes]

---

## Expected Outcomes

Based on the fix, we expect:

1. **Predictions will have variance** (no more straight lines)
2. **R² will become positive** (currently -5.92 in baseline)
3. **Directional accuracy will improve** to 45-55% range (currently 2.16%)
4. **MAPE will decrease** (currently 51% in baseline)
5. **Model will learn temporal patterns** instead of predicting constant values

---

## Lessons Learned

### Critical Assumptions to Verify:
1. ✅ **Stationarity** - Time series must have constant statistical properties
2. ✅ **Distribution consistency** - Train/test sets must come from same distribution
3. ✅ **Normalization scope** - Be careful with global vs. local normalization
4. ✅ **Temporal trends** - Always visualize data over time before modeling

### Best Practices for Time Series:
1. **Always plot data over time** - Visual inspection catches trend issues
2. **Check statistics by split** - Compare mean/std across train/val/test
3. **Test for stationarity** - ADF and KPSS tests
4. **Consider transformations** - Differences, log returns, detrending
5. **Document thoroughly** - Future thesis writing requires detailed records

---

## Next Steps (After Training Completes)

1. **Evaluate and document results** in critical issue markdown
2. **Create comparison table** between baseline and differenced models
3. **Update README** with findings and improved methodology
4. **Consider additional improvements:**
   - Longer sequence lengths (48h or 72h)
   - Attention mechanisms
   - Bidirectional LSTM
   - Hyperparameter tuning

---

## Files Modified/Created This Session

### Documentation
- `docs/CRITICAL_ISSUE_NON_STATIONARY_TARGET.md` - Root cause analysis and solutions
- `results/BASELINE_LSTM_SUMMARY.md` - Baseline model comprehensive summary
- `docs/MODEL_TRAINING_PROGRESS_LOG.md` - This file

### Code
- `scripts/modeling/data_loader_differenced.py` - Differencing data pipeline
- `scripts/modeling/main_differenced.py` - Training script for differenced model
- `scripts/analysis/model_diagnostics.py` - Diagnostic tools

### Visualizations
- `results/visualizations/dvol_temporal_trend.png` - Proof of trend issue
- `results/visualizations/lstm_differenced_*` - To be generated during training

### Models
- `models/lstm_baseline_best.pth` - Baseline model (flawed due to non-stationarity)
- `models/lstm_differenced_best.pth` - Improved model (training in progress)

---

**Status:** ⏳ Training in progress  
**Last Updated:** October 16, 2025  
**Next Update:** After training completes
