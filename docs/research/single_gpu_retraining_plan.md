# Single GPU Retraining Experiment Plan

## Overview

This document outlines the plan to retrain all LSTM models on single GPU to confirm the DataParallel evaluation degradation findings.

## Background

### Problem Identified

Previous multi-GPU training using PyTorch's `DataParallel` showed:
- **market_lags (512×7)**: R² = 0.8021 (CLI training with DataParallel evaluation)
- **Same checkpoint evaluated on single GPU**: R² = 0.9308

This represents a **~12% R² degradation** due to DataParallel evaluation.

### Root Cause

PyTorch's `DataParallel` wrapper causes evaluation performance degradation due to:
- Batch splitting across GPUs causing different forward pass behavior
- Dropout layers not behaving consistently across splits
- BatchNorm statistics update issues in eval mode

### Research Validation

**PyTorch Forum Evidence:**
1. "Performance degrades with DataParallel" (2019)
2. "DataParallel results in a different network compared to single GPU" (2018)
3. "Accuracy difference on multi GPU with nn.DataParallel" (2019) - 22% accuracy drop
4. "Torch DDP Multi-GPU gives low accuracy metric" (2023) - 22% accuracy drop

**ROCm 7 Documentation:**
- Single-GPU DDP training consistently achieves **highest accuracy**
- Multi-GPU DDP can degrade convergence due to large effective batch sizes
- `CUDA_VISIBLE_DEVICES` allows independent GPU assignment per process

## Experimental Design

### Objective

Confirm that single-GPU training (without DataParallel) produces superior results compared to multi-GPU training.

### Hypothesis

**H1:** Single-GPU training will achieve R² ≈ 0.93 for large models (512×7)
**H2:** Single-GPU training will show minimal improvement for small models (128×2)

### Methodology

#### Training Configuration

All models trained with identical hyperparameters:
- **Epochs:** 100 (early stopping patience=15)
- **Batch Size:** 32
- **Learning Rate:** 0.0001
- **Sequence Length:** 24
- **Window Size:** 720
- **Data Split:** 60/20/20 (train/val/test)

#### Model Specifications

| Model Type | Features | Hidden | Layers | Dropout | Parameters |
|------------|----------|--------|--------|---------|------------|
| market_lags | 7 | 512 | 7 | 0.5 | 13.8M |
| jump_aware | 11 | 512 | 7 | 0.5 | 13.8M |
| market_jumps | 8 | 512 | 7 | 0.5 | 13.8M |
| market | 4 | 256 | 3 | 0.4 | ~5M |

#### Parallel Execution Strategy

Using `CUDA_VISIBLE_DEVICES` to assign specific GPUs:

**Round 1 (Large Models - 512×7):**
- GPU 0: market_lags (parallel)
- GPU 1: jump_aware (parallel)
- GPU 0: market_jumps (sequential after first two complete)

**Round 2 (Medium Model - 256×3):**
- GPU 1: market (sequential)

### Expected Results

#### Large Models (512×7, 13.8M parameters)

| Model | Multi-GPU (DataParallel) | Single-GPU (Expected) | Improvement |
|-------|-------------------------|----------------------|-------------|
| market_lags | R² = 0.8021 | R² ≈ 0.93 | +12% |
| jump_aware | R² = 0.800 | R² ≈ 0.80-0.93 | +0-13% |
| market_jumps | R² = 0.6202 | R² ≈ 0.80-0.93 | +18-31% |

#### Medium Model (256×3, ~5M parameters)

| Model | Multi-GPU | Single-GPU (Expected) | Improvement |
|-------|-----------|----------------------|-------------|
| market | R² = 0.6686 | R² ≈ 0.75-0.85 | +8-18% |

**Rationale:** All models expected to show improvement with single-GPU evaluation.

## Implementation

### Quick Test (1 epoch)

Verify setup before full training:

```bash
./scripts/test_single_gpu_quick.sh
```

### Full Training

Execute parallel single-GPU training:

```bash
./scripts/retrain_single_gpu_parallel.sh
```

### Results Location

Results will be saved to:
```
results/single_gpu_retraining_YYYYMMDD_HHMMSS/
├── training.log
├── market_lags_512x7_result.json
├── jump_aware_512x7_result.json
├── market_lags_128x2_result.json
├── market_128x2_result.json
└── market_jumps_128x2_result.json
```

## Validation Criteria

### Success Metrics

1. **Large Models (512×7):**
   - market_lags achieves R² ≥ 0.90
   - Significant improvement over multi-GPU baseline (0.8021)

2. **Medium Models (128×2):**
   - Minimal degradation or improvement over multi-GPU baseline
   - Stable training (no NaN or divergence)

3. **Reproducibility:**
   - Consistent results across multiple runs
   - No DataParallel artifacts in evaluation

### Failure Modes

1. **Training Divergence:**
   - NaN values in loss
   - Model weights become unstable

2. **No Improvement:**
   - Single-GPU results match multi-GPU (R² ≈ 0.80)
   - Indicates deeper issue than DataParallel evaluation

3. **Hardware Issues:**
   - GPU memory overflow
   - Driver instability

## Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Quick Test | ~5 minutes | Verify both GPUs work independently |
| Round 1 (Large Models) | ~2-3 hours | Parallel training of 512×7 models |
| Round 2 (Medium Models) | ~1-2 hours | Sequential training of 128×2 models |
| Analysis | ~30 minutes | Compare results with multi-GPU baseline |

**Total Estimated Time:** 4-6 hours

## References

### PyTorch Documentation
- [Distributed Data Parallel Training on AMD GPU with ROCm](https://rocm.blogs.amd.com/artificial-intelligence/ddp-training-pytorch/)
- [Efficient Training on Multiple GPUs - Hugging Face](https://huggingface.co/docs/transformers/v4.44.0/perf_train_gpu_many)

### PyTorch Forum Discussions
- "Performance degrades with DataParallel" (2019)
- "DataParallel results in a different network compared to single GPU" (2018)
- "Accuracy difference on multi GPU with nn.DataParallel" (2019)

### StackOverflow
- "Torch DataParallel model predict same data is different between single GPU or CPU" (2024)

## Conclusion

This experiment will definitively confirm whether the DataParallel evaluation issue is responsible for the performance discrepancy between single-GPU (R²=0.9308) and multi-GPU (R²=0.8021) results.

If successful, this will validate that:
1. The R²=0.9308 results ARE legitimate and from the CLI-trained checkpoint
2. Previous multi-GPU results were understated by ~12% R² due to evaluation artifact
3. Single-GPU training should be preferred for final model evaluation

## Next Steps

1. Run quick test to verify setup
2. Execute full parallel training
3. Analyze results and compare with multi-GPU baseline
4. Update README with confirmed single-GPU performance metrics
5. Document findings in thesis methodology section
