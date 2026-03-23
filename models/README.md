# LSTM Model Checkpoints

This directory contains trained LSTM model checkpoints organized by status.

## Directory Structure

```
models/
├── final/           # Production-ready models (January 28, 2026)
├── historical/      # Deprecated baseline models
└── archive/         # Experimental and superseded models
    ├── superseded/  # Earlier versions replaced by final models
    └── experimental/ # One-off experiments and tests
```

---

## Final Models (Production)

Trained on January 28, 2026 with fixed evaluation (no DataParallel wrapper).

| File | Architecture | Features | R² | RMSE | MAE | Parameters |
|------|-------------|----------|-----|------|-----|------------|
| `market_lags_512x7.pth` | 512×7 | 7 (lags + on-chain) | **0.9287** | 1.71 | 1.28 | 13.8M |
| `jump_aware_512x7.pth` | 512×7 | 11 (lags + on-chain + jumps) | 0.7986 | 2.87 | 2.04 | 13.8M |
| `market_jumps_512x7.pth` | 512×7 | 8 (market + jumps) | 0.6100 | 4.00 | 2.94 | 13.8M |
| `market_512x7.pth` | 512×7 | 4 (market only) | 0.6135 | 3.98 | 2.97 | 13.8M |
| `market_512x3.pth` | 512×3 | 4 (market only) | 0.5940 | 4.08 | 3.16 | 5.4M |
| `market_256x3.pth` | 256×3 | 4 (market only) | 0.6145 | 3.97 | 3.14 | 1.4M |

**Best Model:** `market_lags_512x7.pth` (R² = 0.9287)

### Key Findings:
- Lagged volatility features (1d, 7d, 30d) are critical for performance
- Jump features provide NO benefit and harm LSTM performance
- 512×7 architecture requires minimum 7 features for stability

---

## Historical Models

Legacy models kept for research documentation.

| File | Architecture | R² | Notes |
|------|-------------|-----|-------|
| `rolling_512x7.pth` | 512×7 | 0.201 | Original rolling baseline (DEPRECATED) |

This model was used for the original jump-aware comparison showing 4x improvement (R² 0.201 → 0.800). However, the unified framework later showed this was due to feature engineering, not jump detection.

---

## Archive

### Superseded Models

Earlier versions from January 27, 2026, replaced by the January 28 final models.

| File | Notes |
|------|-------|
| `market_jumps_jan27.pth` | Replaced by `final/market_jumps_512x7.pth` |
| `market_lags_jan27.pth` | Replaced by `final/market_lags_512x7.pth` |
| `market_256x3_jan27.pth` | Replaced by `final/market_256x3.pth` |

### Experimental Models

One-off experiments and debugging runs. Not recommended for use.

| Pattern | Description |
|---------|-------------|
| `single_gpu_*.pth` | Single GPU training experiments |
| `lr_matched_*.pth` | Learning rate scaling experiments |
| `cli_*.pth` | Early CLI framework experiments |
| `deep_*.pth` | 5-layer architecture experiments |
| `test_*.pth` | Stability tests |

---

## Usage

### Loading a Model

```python
import torch
from scripts.modeling.model import LSTM_DVOL

# Load best model
model = LSTM_DVOL(
    input_size=7,
    hidden_size=512,
    num_layers=7,
    dropout=0.5
)
model.load_state_dict(torch.load('models/final/market_lags_512x7.pth'))
model.eval()
```

### Training New Models

See `cli/bin/train.py` for the current training interface:

```bash
.venv/bin/python cli/bin/train.py market_lags \
  --hidden-size 512 --num-layers 7 --dropout 0.5 \
  --batch-size 32 --lr 0.0001 --epochs 100 --use-multi-gpu
```

---

## Model History

| Date | Event |
|------|-------|
| 2026-01-28 | Fixed evaluation (removed DataParallel wrapper), R² improved 13.3% |
| 2026-01-27 | Unified framework completion, 17 models trained |
| 2026-01-02 | Architecture optimization, 512×7 identified as optimal |
| 2025-12-30 | Deep architecture experiments |
| 2025-10-20 | Original jump-aware LSTM training |
