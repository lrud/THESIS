# Single-GPU LSTM Training Results (72h Normalization)

**Date:** April 3, 2026
**Hardware:** AMD Radeon RX 7900 XT (20GB VRAM), single GPU (CUDA_VISIBLE_DEVICES=1)
**Dataset:** `bitcoin_lstm_features_v1.6_final.csv` (37,949 hourly observations)

---

## Training Configuration

| Parameter | 512x3 (Optimal) | 512x7 (Tested) |
|-----------|-----------------|-----------------|
| Hidden size | 512 | 512 |
| LSTM layers | 3 | 7 |
| Dropout | 0.4 | 0.5 |
| Learning rate | 0.0001 | 0.0001 |
| Batch size | 32 | 32 |
| Sequence length | 24h | 24h |
| Normalization | 72h rolling z-score | 72h rolling z-score |
| Early stopping | patience=15 | patience=15 |
| Multi-GPU | No (single GPU) | No (single GPU) |

**Why 72h?** Multi-window sweep in `unified_model_comparison.ipynb` showed 72h rolling normalization produces the best R² across all model families. Previous LSTM training used 720h, which was suboptimal.

---

## LSTM Results: 512x3 (Optimal Architecture)

| Model | Features | R2 | RMSE | MAE | MAPE | Dir% | Params | Epochs | Time |
|-------|----------|-----|------|-----|------|------|--------|--------|------|
| market | 4 | **0.9406** | 1.6245 | 1.1872 | 2.544% | 51.07% | 5.39M | 22 | 4.0m |
| market_jumps | 8 | 0.9406 | 1.6247 | 1.1868 | 2.543% | 51.02% | 5.40M | 22 | 3.0m |
| jump_aware | 11 | 0.9405 | 1.6252 | 1.1860 | 2.541% | 50.89% | 5.41M | 21 | 2.7m |
| market_lags | 7 | 0.9405 | 1.6254 | 1.1858 | 2.540% | 50.92% | 5.40M | 23 | 3.1m |

### Jump Decomposition (models with jump features)

| Model | Normal R2 | Jump R2 | Normal Dir% | Jump Dir% | Jump Samples |
|-------|-----------|---------|-------------|-----------|--------------|
| jump_aware (512x3) | 0.9425 | 0.6908 | 50.84% | **54.76%** | 43 |
| market_jumps (512x3) | 0.9426 | 0.6920 | 50.97% | **54.76%** | 43 |

---

## LSTM Results: 512x7 (Deeper Architecture — No Improvement)

| Model | R2 | RMSE | Dir% | Params | Time |
|-------|-----|------|------|--------|------|
| jump_aware | 0.9406 | 1.6239 | 50.98% | 13.81M | 4.6m |
| market | 0.9406 | 1.6245 | 51.04% | 13.80M | 5.6m |
| market_jumps | 0.9406 | 1.6247 | 51.02% | 13.81M | 6.8m |
| market_lags | 0.9405 | 1.6260 | 50.96% | 13.81M | 7.5m |

**Conclusion:** 512x7 (13.8M params) produces identical results to 512x3 (5.4M params). The task is capacity-saturated at 512x3 — deeper models provide no benefit.

---

## Comparison: LSTM vs Unified Trainer (Linear & Tree Models)

All models below use the **72h normalization window** on the same dataset.

### Overall R² Comparison

| Model | Family | Features | R2 | RMSE | Dir% |
|-------|--------|----------|-----|------|------|
| **XGB NoLag Jumps** | Tree | 8 | **0.9940** | 0.53 | 49.3% |
| XGB Lags Jumps | Tree | 11 | 0.9936 | 0.55 | 49.1% |
| RF NoLag Jumps | Tree | 8 | 0.9935 | 0.55 | 49.5% |
| RF Lags Jumps | Tree | 11 | 0.9935 | 0.55 | 49.3% |
| RF NoLag | Tree | 4 | 0.9927 | 0.58 | 49.7% |
| XGB Lags | Tree | 7 | 0.9927 | 0.58 | 49.5% |
| XGB NoLag | Tree | 4 | 0.9927 | 0.58 | 49.5% |
| RF Lags | Tree | 7 | 0.9927 | 0.59 | 49.4% |
| OLS NoLags Jumps | Linear | 8 | 0.9920 | 0.61 | 48.8% |
| OLS WithLags Jumps | Linear | 11 | 0.9920 | 0.61 | 48.9% |
| OLS NoLags | Linear | 4 | 0.9919 | 0.62 | 48.8% |
| OLS WithLags | Linear | 7 | 0.9919 | 0.62 | 48.9% |
| **LSTM market** | **LSTM** | **4** | **0.9406** | **1.62** | **51.1%** |
| **LSTM market_jumps** | **LSTM** | **8** | **0.9406** | **1.62** | **51.0%** |
| **LSTM jump_aware** | **LSTM** | **11** | **0.9405** | **1.63** | **50.9%** |
| **LSTM market_lags** | **LSTM** | **7** | **0.9405** | **1.63** | **50.9%** |
| HAR-RV | Linear | 3 | 0.9592 | 1.38 | 50.3% |

### Key Takeaways

1. **Tree models dominate R²** — XGBoost/RF achieve R²=0.994 vs LSTM's R²=0.941. This is expected: tree models can overfit the near-random-walk DVOL series via direct feature-to-target mapping, while LSTMs must learn temporal patterns through backpropagation through time.

2. **LSTM slightly edges directional accuracy** — LSTM achieves 50.9-51.1% vs tree models at 48.8-49.7%. Neither is meaningfully above the 50% random baseline.

3. **Feature sets don't matter for LSTM** — All 4 feature sets converge to R²≈0.941 and Dir%≈51%. The LSTM learns essentially the same mapping regardless of whether you give it 4 features or 11.

4. **LSTM vs HAR-RV** — LSTM (R²=0.941) underperforms HAR-RV (R²=0.959), suggesting the temporal structure captured by HAR-RV's multi-horizon lag decomposition is more efficient for this task.

---

## Comparison: Single-GPU vs Previous Multi-GPU (DataParallel) Results

Previous results used 720h normalization window and multi-GPU DataParallel training.

| Model | Config | R2 | RMSE | Key Difference |
|-------|--------|-----|------|----------------|
| LSTM jump_aware | 512x3, 72h, single-GPU | **0.9405** | 1.6252 | Current (optimal) |
| LSTM ultra-large | 512x3, 720h, multi-GPU | 0.9076 | 2.57 | Previous best |
| LSTM jump_aware | 128x2, 720h, multi-GPU | 0.8624 | 3.14 | Original baseline |

**The single-GPU + 72h window combo improves R² from 0.908 to 0.941** — a 3.7 percentage point gain attributable to:
- **72h normalization** (primary): tighter rolling window adapts faster to regime changes
- **Single-GPU training** (secondary): eliminates DataParallel R² degradation artifact (~12% documented in prior experiments)

---

## Research Implications

- **Capacity saturation**: 512x3 is sufficient. 512x7 adds 2.5x parameters with zero gain.
- **Feature saturation**: 4 features (market-only) perform identically to 11 features (jump_aware). LSTM doesn't leverage additional features.
- **Directional accuracy**: ~51% across all configurations — consistent with efficient market hypothesis. DVOL next-hour direction is not predictable.
- **Tree models are stronger for this task**: XGBoost R²=0.994 >> LSTM R²=0.941 on identical data/window. The sequential modeling overhead of LSTM doesn't pay off for a near-random-walk target.

---

## Files

- Results JSONs: `results/server_training/*_512x3_72h_*.json` and `*_512x7_72h_*.json`
- Model checkpoints: `models/server_*_512x3_72h_best.pth` and `*_512x7_72h_best.pth`
- Training scripts: `scripts/train_server_512x3_72h.py`, `scripts/train_all_512x7_72h.py`
- Unified comparison notebook: `notebooks/unified_model_comparison.ipynb`
