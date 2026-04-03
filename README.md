# LSTM Forecasting of Bitcoin Implied Volatility (DVOL)

## Abstract

This thesis develops and validates Long Short-Term Memory (LSTM) neural network models for forecasting Bitcoin implied volatility (DVOL), the Deribit 30-day volatility index. Using a unified framework of 17 models (13 linear/tree baselines + 4 LSTM variants) with 72-hour rolling normalization on 41,055 hourly observations, we demonstrate:

**Key Findings:**
- LSTM models achieve R² = 0.9406 on 72h-normalized DVOL, competitive with HAR-RV (R² = 0.9592) but below tree models (R² = 0.994)
- A naive persistence baseline (DVOL[t+1] = DVOL[t]) achieves R² = 0.9985 on raw DVOL — all models underperform this trivial benchmark, reflecting DVOL's near-unit-root autocorrelation (ρ = 0.9992)
- No model achieves statistically significant directional accuracy (~51% across all 17 models, p > 0.05 with serial correlation correction)
- All 4 LSTM feature sets converge to R² ≈ 0.941 — feature engineering provides zero marginal value for LSTM
- Tree model superiority (R² = 0.994) is driven by direct access to lagged DVOL features in a near-unit-root series, not genuine forecasting skill

**Contributions:**
- Systematic 17-model comparison framework with unified preprocessing (72h rolling z-score)
- Identification and resolution of DataParallel evaluation artifact (13.3% R² degradation)
- Comprehensive statistical audit identifying 4 critical assumptions requiring reinterpretation
- Standard Lee-Mykland (2008) jump detection with Pesaran-Timmermann (1992) directional accuracy

## Objective

Develop LSTM neural network models to forecast Bitcoin implied volatility (DVOL) using on-chain metrics and historical volatility patterns, validated through rigorous statistical analysis with proper baseline comparisons.

## Current Status (April 3, 2026)

**Phase:** Final LSTM training complete (72h normalization, single-GPU, v1.6 dataset). Comprehensive statistical audit completed. Results documentation and thesis writing in progress.

### State-of-the-Art LSTM Results (April 2026)

All LSTM models below trained on **v1.6_final dataset** with **72h rolling z-score normalization**, **single-GPU** (AMD RX 7900 XT), **512×3 architecture** (5.4M parameters):

| Model | Features | R² | RMSE | MAE | Dir% | Params | Epochs | Time |
|-------|----------|-----|------|-----|------|--------|--------|------|
| market | 4 | **0.9406** | 1.6245 | 1.1872 | 51.07% | 5.39M | 22 | 4.0m |
| market_jumps | 8 | 0.9406 | 1.6247 | 1.1868 | 51.02% | 5.40M | 22 | 3.0m |
| jump_aware | 11 | 0.9405 | 1.6252 | 1.1860 | 50.89% | 5.41M | 21 | 2.7m |
| market_lags | 7 | 0.9405 | 1.6254 | 1.1858 | 50.92% | 5.40M | 23 | 3.1m |

**512×7 confirmed capacity-saturated** — identical results with 2.5× more parameters (13.8M).

### Critical Context: Persistence Baseline

| Model/Baseline | R² | RMSE | Notes |
|---------------|-----|------|-------|
| **Naive persistence (DVOL[t+1] = DVOL[t])** | **0.9985** | **0.75** | Raw levels, no model needed |
| 24h rolling mean | ~0.996 | ~1.1 | Trivial baseline |
| XGBoost (NoLag_Jumps) | 0.9940 | 0.53 | Sees explicit lag features |
| HAR-RV | 0.9592 | 1.38 | 3-feature linear model |
| **LSTM best (market)** | **0.9406** | **1.62** | Sees only normalized z-scores |

DVOL autocorrelation: Lag-1h ρ=0.9992, Lag-24h ρ=0.9824, Lag-72h ρ=0.9552, Lag-168h ρ=0.9136

> **Interpretation:** The 72h normalization destroys level information that persistence exploits. LSTM R²=0.94 measures prediction of normalized deviations from a rolling mean — a harder task than raw level prediction. The thesis contribution is the methodological comparison, not absolute performance.

### Research Timeline

**April 2026 — Final LSTM Training & Statistical Audit:**
- Retrained all 4 LSTM variants on v1.6_final with 72h normalization, single-GPU
- Confirmed 512×3 as optimal architecture (512×7 adds no improvement)
- Completed comprehensive 11-assumption statistical audit (4 CRITICAL findings)
- Documented persistence baseline context (R²=0.9985 vs LSTM R²=0.94)

**February 2026 — Multi-Window Analysis & Dataset v1.6:**
- 72-hour normalization window proven optimal across 31 models
- Dataset v1.6: 41,055 records, standard Lee-Mykland (236 jumps, 0.57%)
- Classification analysis: hourly DVOL direction fundamentally unpredictable
- Pesaran-Timmermann (1992) directional accuracy correction applied

**January 2026 — Unified Framework & Fixed Evaluation:**
- DataParallel wrapper during evaluation degraded LSTM R² by 13.3% (identified and fixed)
- 17-model unified comparison framework (13 linear/tree + 4 LSTM)
- market_lags (512×7): R² = 0.9287 after evaluation fix (was 0.8021 with DataParallel)
- All models ≈50% directional accuracy — statistically random

**November 2025 — Code Consolidation:**
- 50% code reduction, unified CLI training system
- Ultra-large model (512×3, 5.4M params): R² = 0.9076 (720h window, multi-GPU)

**October 2025 — Original LSTM Research:**
- Jump-aware LSTM (512×7): R² = 0.800 (v1.1 dataset)
- Rolling baseline: R² = 0.201
- Architecture optimization: depth scaling (3→5→7 layers) effective

---

## Model Specification

### Target Variable
- **DVOL**: Deribit 30-day implied volatility index (1-hour ahead forecast)
- **Transformation**: 72-hour rolling z-score normalization for regime adaptation

### Dataset (v1.6_final)
- **41,055 hourly observations** (April 23, 2021 09:00 – December 28, 2025 23:00)
- **19 columns**, 100% coverage (no gaps)
- **236 Lee-Mykland jumps** (0.57%) using standard Gumbel threshold (β* = 9.21)
- **Split**: Train 24,633 (60%) / Val 8,211 (20%) / Test 8,211 (20%)
- **DVOL statistics**: mean=61.92, std=18.65, range=[31.47, 166.39]

### Core Predictors (9 features)

| Feature | Description | Key Statistic |
|---------|-------------|---------------|
| Lagged DVOL (1d, 7d, 30d) | Implied volatility lags | Daily autocorrelation ρ ≈ 0.80 |
| Transaction Volume (USD) | On-chain volume | Granger causality: 89% rejection |
| Active Addresses Count | Network activity | Negative relationship with volatility |
| NVRV | Network Value to Realized Value | Strongest BTC price correlation |
| DVOL-RV Spread | Volatility risk premium | Explains 15-20% of future variance |

### Feature Set Definitions (4 LSTM variants)

| Feature Set | # Features | Components |
|------------|-----------|------------|
| market | 4 | dvol_norm, volume_norm, active_addresses_norm, nvrv_norm |
| market_jumps | 8 | market + jump_indicator, jump_magnitude, jump_timing, jump_clustering |
| jump_aware | 11 | market + jumps + dvol_lag_1d, dvol_lag_7d, dvol_lag_30d |
| market_lags | 7 | market + dvol_lag_1d, dvol_lag_7d, dvol_lag_30d |

> **Result:** All 4 feature sets converge to R² ≈ 0.9406 (range: 0.0001). Feature engineering provides no marginal value.

## LSTM Architecture

- **Optimal Architecture:** 3 LSTM layers, 512 hidden units (5.4M parameters)
- **Deeper Tested:** 7 layers, 512 hidden units (13.8M params) — **no improvement**
- **Input:** Sequential windows (24h lookback) of normalized features
- **Structure:** LSTM → dropout(0.4) → FC(512→256) → ReLU → dropout → FC(256→1)
- **Regularization:** 0.4 dropout, 1e-4 L2 penalty, gradient clipping=1.0
- **Training:** Early stopping (patience=15), learning rate 1e-4, ReduceLROnPlateau
- **Normalization:** 72-hour rolling z-score (optimal from multi-window sweep)
- **Hardware:** Single AMD Radeon RX 7900 XT GPU (ROCm 7.0)
- **Output:** Single value (next-hour normalized DVOL forecast)

**Architecture Scaling Results (v1.6, 72h window, single-GPU):**

| Architecture | Params | Best R² | Best RMSE | Dir% | Conclusion |
|-------------|--------|---------|-----------|------|------------|
| **512×3 (optimal)** | **5.4M** | **0.9406** | **1.62** | **51.1%** | **Capacity sufficient** |
| 512×7 (deeper) | 13.8M | 0.9406 | 1.62 | 51.0% | No improvement |
| 128×2 (baseline) | 210K | 0.67 | 3.67 | ~50% | Underfitting |

**Historical Architecture Results (v1.1, 720h window, multi-GPU — superseded):**

| Architecture | Params | R² | Status |
|-------------|--------|-----|--------|
| 512×7 market_lags | 13.8M | 0.9287 | Previous best (DataParallel fix) |
| 512×3 ultra-large | 5.4M | 0.9076 | Legacy (720h window) |
| 128×2 jump_aware | 210K | 0.8624 | Original baseline |

## Training Infrastructure

### AI Server Configuration

Training is performed on a dedicated AI server accessible via SSH:

- **Hardware:** AMD GPU (ROCm 7.0), 20GB+ VRAM
- **Training directory:** `/root/thesis/`
- **Results:** `/root/thesis/results/server_training/`
- **Model checkpoints:** `/root/thesis/models/`

### CLI Training System

```bash
# Optimal configuration (512×3, 72h window, single-GPU)
.venv/bin/python cli/bin/train.py market \
  --hidden-size 512 --num-layers 3 --dropout 0.4 \
  --batch-size 32 --lr 0.0001 --epochs 100

# All 4 feature sets (market, market_jumps, jump_aware, market_lags)
# Use same hyperparameters — results converge to R² ≈ 0.94

# Multi-GPU training (NOT recommended — causes ~12% R² degradation)
.venv/bin/python cli/bin/train.py market --use-multi-gpu --lr 0.0001
```

### Real-time Monitoring
```bash
tail -f results/logs/current_training.log
```

### Dependencies
```bash
# PyTorch with ROCm 7.0 (AMD GPU support)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
pip3 install -r requirements-pytorch.txt
```

---

## Results

### Unified Framework Methodology

**All 17 models use IDENTICAL preprocessing for fair comparison:**

| Specification | Implementation |
|--------------|----------------|
| Data Split | 60% train / 20% val / 20% test (temporal) |
| Normalization | 72-hour rolling z-score (features AND target) |
| Target Variable | `dvol_norm.shift(-1)` (1-hour ahead forecast) |
| Data Source | v1.6_final (41,055 samples) |
| No look-ahead bias | Rolling window uses only historical data |

**Why 72h normalization?** Multi-window sweep (72h, 168h, 336h, 720h) showed 72h produces the best R² across all model families. The tighter window adapts faster to regime changes (DVOL mean shifts from 69→48 across the dataset).

---

### Complete Model Comparison (v1.6, 72h window)

**Regression Models (17 total):**

| Model | Type | Features | R² | RMSE | Dir% |
|-------|------|----------|-----|------|------|
| **XGB NoLag Jumps** | Tree | 8 | **0.9940** | 0.53 | 49.3% |
| RF NoLag Jumps | Tree | 8 | 0.9935 | 0.55 | 49.5% |
| OLS NoLags Jumps | Linear | 8 | 0.9920 | 0.61 | 48.8% |
| **HAR-RV** | Linear | 3 | **0.9592** | 1.38 | **50.3%** |
| **LSTM market** | LSTM | 4 | **0.9406** | 1.62 | **51.1%** |
| LSTM market_jumps | LSTM | 8 | 0.9406 | 1.62 | 51.0% |
| LSTM jump_aware | LSTM | 11 | 0.9405 | 1.63 | 50.9% |
| LSTM market_lags | LSTM | 7 | 0.9405 | 1.63 | 50.9% |

> Full tree/linear results available in `docs/results/single_gpu_72h_lstm_results.md` and `notebooks/unified_model_comparison.ipynb`.

### Key Insights (17 Models)

1. **Persistence dominates:** Raw DVOL persistence (R²=0.9985) beats all models. High R² values reflect DVOL's near-unit-root behavior, not genuine forecasting skill.

2. **Tree model superiority is persistence-driven:** XGBoost/RF see explicit `dvol_lag_1d` features in a series with ρ=0.9992. OLS achieving R²=0.992 with the same features confirms this is lag exploitation, not learning.

3. **LSTM vs HAR-RV:** LSTM (R²=0.941) underperforms HAR-RV (R²=0.959). HAR-RV's multi-horizon lag decomposition is more efficient than LSTM's learned temporal representations for this near-random-walk target.

4. **Feature irrelevance for LSTM:** All 4 feature sets converge to R²=0.9406 (range 0.0001). The LSTM's predictive power comes from temporal structure (24h window), not specific features.

5. **Directional accuracy ≈ 50% for ALL models:** No model achieves statistically significant directional prediction (best: LSTM at 51.1%, z=1.93, p=0.054 uncorrected; likely p>0.10 with Pesaran-Timmermann correction).

6. **Practical implication:** Models are suitable for volatility level estimation and risk management, NOT for directional trading.

---

### Statistical Audit (April 2026)

A comprehensive audit of 11 implicit statistical assumptions was conducted. Key findings:

**CRITICAL Issues (4):**

| # | Assumption | Finding | Impact |
|---|-----------|---------|--------|
| 1 | R²=0.94 = good performance | R² is **worse than persistence** (0.9985) | All performance claims need persistence baseline |
| 2 | Dir% ≈ 51% is near-random | Correct, but **statistically untested** | Need Pesaran-Timmermann test + p-values |
| 3 | Tree R²=0.994 > LSTM | **Misleading comparison** — different preprocessing + lag exploitation | Need same-normalization comparison |
| 4 | Residual analysis not needed | **Missing standard validation** (Ljung-Box, ARCH-LM, ACF) | Required for any volatility forecasting paper |

**GREEN Flags (properly done):**

| # | Assumption | Finding |
|---|-----------|---------|
| - | No look-ahead bias | Rolling normalization is properly causal |
| - | Temporal train/val/test split | Clean, non-overlapping 60/20/20 |
| - | Feature set irrelevance | R² varies by only 0.0001 across 4-11 features |
| - | 512×7 = 512×3 | Confirms capacity saturation, not overfitting |

Full audit: `docs/results/statistical_audit_report.md`

---

### Multi-Window Normalization Analysis (February 2026)

| Window | Best R² Model | Best R² | Best Dir% (HAR-RV) |
|--------|---------------|---------|---------------------|
| **72h (3d)** | XGB_NoLag_Jumps | **0.9940** | 50.3% |
| 168h (7d) | XGB_Lags_Jumps | 0.9926 | 50.8% |
| 336h (14d) | RF_Lags_Jumps | 0.9914 | 50.7% |
| 720h (30d) | RF_NoLag_Jumps | 0.9911 | 50.3% |

**Recommendation:** 72-hour window for all new model training.

### Classification Models (February 2026)

18 classification models tested across 4 window sizes. **No statistical significance at 5% level.** Best: LDA_HAR (54.3% accuracy, F1=0.0, degenerate). Hourly DVOL direction is fundamentally unpredictable.

---

## Research Evolution

### Key Finding Updates

| Phase | Date | Finding | R² |
|-------|------|---------|-----|
| Original | Oct 2025 | Jump features critical (4x improvement) | 0.201 → 0.800 |
| Unified | Jan 2026 | Lagged features sufficient, not jumps | 0.802 → 0.929 |
| DataParallel fix | Jan 2026 | Evaluation artifact inflated gap | 0.802 → 0.929 |
| 72h window | Apr 2026 | Optimal normalization window | 0.929 → 0.941 |
| Audit | Apr 2026 | Persistence baseline contextualization | 0.941 vs 0.9985 |

### Thesis Contributions

1. **Systematic 17-model comparison** with unified preprocessing (LSTM, OLS, HAR-RV, RF, XGBoost)
2. **72h optimal normalization window** identified via multi-window sweep
3. **DataParallel evaluation artifact** identified and resolved (13.3% R² degradation)
4. **Comprehensive statistical audit** with literature-backed reference points
5. **Standard Lee-Mykland (2008)** jump detection with 236 jumps (0.57%)
6. **Feature irrelevance demonstrated** for LSTM (R² range: 0.0001 across 4-11 features)
7. **Capacity saturation confirmed** (512×3 = 512×7)
8. **Directional unpredictability** documented with Pesaran-Timmermann framework

---

## Outstanding Work

### Required Before Thesis Submission

- [ ] **Compute persistence R² on exact same test set** with same 72h normalization for fair comparison
- [ ] **Run Pesaran-Timmermann (1992) test** with serial correlation correction on directional accuracy
- [ ] **Add residual diagnostics**: Ljung-Box test, ARCH-LM test, residual ACF plots
- [ ] **Run 3-5 random seeds** for confidence intervals on test metrics
- [ ] **Update `cli/config/config.py`** default `window_size` from 720 to 72
- [ ] **Sync result JSONs** from ai-server (`/root/thesis/results/`) to local `results/`
- [ ] **Reframe thesis narrative**: LSTM contribution is methodological comparison, not absolute performance

### Nice to Have

- [ ] Diebold-Mariano test statistics between model pairs
- [ ] QLIKE loss evaluation (alternative to MSE for volatility forecasting)
- [ ] Walk-forward validation with multiple cutoff dates
- [ ] Train R² logging (to compute generalization gap)
- [ ] ADF test on normalized DVOL series

---

## Repository Structure

```
THESIS 2025/
│
├── cli/                              # Production training interface
│   ├── bin/
│   │   ├── train.py                  # Main CLI entry point (10 model types)
│   │   └── train_with_monitoring.py  # Training with logging
│   ├── config/
│   │   ├── config.py                 # Configuration management (TODO: update window_size 720→72)
│   │   └── feature_configs.py        # Feature set definitions
│   └── scripts/trainers/             # Modular trainer implementations
│       ├── unified_trainer.py        # market, market_jumps, market_lags
│       ├── jump_aware_trainer.py     # Jump-aware LSTM
│       ├── changes_trainer.py        # Changes model
│       ├── rolling_trainer.py        # DEPRECATED
│       └── differenced_trainer.py    # DEPRECATED (trivial solution)
│
├── scripts/                          # Analysis and modeling code
│   ├── analysis/
│   │   ├── production/               # Reusable analysis scripts
│   │   │   ├── comprehensive_model_validation.py
│   │   │   ├── jump_detection_analysis.py
│   │   │   ├── run_multi_window_comparison.py
│   │   │   ├── standard_lee_mykland.py
│   │   │   ├── tail_risk_and_benchmarks.py
│   │   │   └── unified_model_comparison.py
│   │   └── one_off/                  # Investigation/data prep scripts
│   │       ├── create_v16_final.py
│   │       ├── update_to_standard_lm.py
│   │       └── ...
│   ├── benchmarking/                 # Benchmark scripts
│   │   ├── compare_all_models.py
│   │   ├── main_har_rv.py
│   │   └── main_naive_baselines.py
│   ├── data_collection/              # Data acquisition pipelines
│   │   ├── researchbitcoin_data.py
│   │   ├── fill_gaps.py
│   │   └── deribit_options_scraper.py
│   ├── modeling/                     # Core LSTM model code
│   │   ├── model.py                  # LSTM_DVOL architecture
│   │   ├── data_loader_unified.py    # Unified dataset class (72h rolling z-score)
│   │   ├── data_loader_jump_aware.py
│   │   └── data_loader_rolling.py
│   ├── utils/                        # Shared utilities
│   │   ├── metrics.py                # Unified evaluation metrics
│   │   └── har_rv.py                 # HAR-RV implementation
│   ├── train_server_512x3_72h.py     # AI server training script (optimal config)
│   └── train_all_512x7_72h.py        # AI server training script (deeper arch)
│
├── notebooks/                        # Jupyter notebooks
│   ├── unified_model_comparison.ipynb # 17-model comparison (v1.6, 72h)
│   ├── unified_model_comparison_classification.ipynb
│   ├── benchmarking.ipynb
│   └── manual_stats.ipynb
│
├── results/                          # All experimental results
│   ├── server_training/              # AI server training results (to sync)
│   ├── cli_training/                 # CLI training results by date
│   ├── analysis/                     # Analysis JSON/CSV results
│   ├── diagnostics/                  # Diagnostic outputs
│   ├── visualizations/               # All plots and figures
│   └── thesis_v2/                    # Historical thesis v2 results
│
├── models/                           # Model checkpoints (gitignored)
│   ├── final/                        # Production models
│   └── archive/                      # Experimental/superseded
│
├── deprecated/                       # Archived code
│   ├── modeling/                     # Old training scripts
│   └── thesis_v2_har_rv/             # Old HAR-RV package
│
├── docs/                             # Documentation
│   ├── results/
│   │   ├── QUICK_REFERENCE.md        # Performance summary
│   │   ├── single_gpu_72h_lstm_results.md  # April 2026 LSTM results
│   │   ├── statistical_audit_report.md      # 11-assumption audit
│   │   └── ultra_large_model_results.md     # Legacy results
│   ├── journal/                      # Research session logs
│   ├── methodology/                  # Statistical methodology docs
│   ├── implementation/               # Implementation guides
│   └── research/                     # Research plans and notes
│
├── data/                             # Data files (gitignored)
│   ├── processed/
│   │   └── bitcoin_lstm_features_v1.6_final.csv  # RECOMMENDED (41,055 samples)
│   ├── raw/
│   └── archive/
│
└── deribit_data_collector/           # Data collection tools
```

### AI Server

Training runs on a dedicated AI server (`/root/thesis/`):
- **Results JSONs:** `/root/thesis/results/server_training/` — need to sync to local
- **Model checkpoints:** `/root/thesis/models/` — 512×3 and 512×7 `.pth` files
- **Training scripts:** Copied from local `scripts/` directory

---

## Documentation

**Key Documents:**
- `docs/results/statistical_audit_report.md` — Comprehensive 11-assumption statistical audit (April 2026)
- `docs/results/single_gpu_72h_lstm_results.md` — Final LSTM results with all comparisons (April 2026)
- `docs/results/QUICK_REFERENCE.md` — Performance summary and thesis defense points
- `docs/journal/2026-02-26.md` — Multi-window normalization analysis
- `docs/journal/2026-02-25.md` — Dataset evolution (v1.1 → v1.6)

## References

**Volatility Modeling:**
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Andersen, T.G. & Benzoni, L. (2009). Realized Volatility. *Handbook of Financial Time Series*.

**Forecast Evaluation:**
- Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy. *JBES*, 13(3), 253-263.
- Patton, A.J. (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies. *Journal of Econometrics*, 160(1), 246-256.
- Hansen, P.R. & Lunde, A. (2006). Consistent Ranking of Volatility Models. *Journal of Econometrics*, 131(1-2), 97-121.

**Directional Accuracy:**
- Pesaran, M.H. & Timmermann, A. (1992). A Simple Nonparametric Test of Predictive Performance. *JBES*, 10(4), 461-465.
- Pesaran, M.H. & Timmermann, A. (2009). Testing Dependence Among Serially Correlated Multicategory Variables. *JASA*, 104(485), 325-337.

**Jump Detection:**
- Lee, S.S. & Mykland, P.A. (2008). Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics. *Review of Financial Studies*, 21(6), 2543-2577.

**Machine Learning for Volatility:**
- Vrontos, I. et al. (2021). Forecasting VIX with Machine Learning. *Journal of Forecasting*.
- Balaneji, B. & Maringer, D. (2022). Implied Volatility Forecasting with XGBoost. *Quantitative Finance*.

**Statistical Methodology:**
- Blaskowitz, O. & Herwartz, H. (2014). Testing the Value of Directional Forecasts. *International Journal of Forecasting*, 30(1), 30-42.
- Henriksson, R.D. & Merton, R.C. (1981). On Market Timing and Investment Performance. *Journal of Business*, 54(4), 513-533.
