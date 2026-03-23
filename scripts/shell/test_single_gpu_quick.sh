#!/bin/bash
# =============================================================================
# Quick Test: Single GPU Training (1 epoch only)
# Purpose: Verify setup before full retraining
# =============================================================================

set -e

PROJECT_ROOT="/home/lrud1314/PROJECTS_WORKING/THESIS 2025"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Quick Test: Single GPU Training"
echo "========================================"
echo "Date: $(date)"
echo ""

# Test both GPUs with a quick 1-epoch training
echo "=== Testing GPU 0: market_lags (512x7) - 1 epoch ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python cli/bin/train.py market_lags \
    --hidden-size 512 --num-layers 7 --dropout 0.5 \
    --batch-size 32 --lr 0.0001 --epochs 1 --patience 100 \
    --save-prefix "test_gpu0"

echo ""
echo "=== Testing GPU 1: market (256x3) - 1 epoch ==="
CUDA_VISIBLE_DEVICES=1 .venv/bin/python cli/bin/train.py market \
    --hidden-size 256 --num-layers 3 --dropout 0.4 \
    --batch-size 32 --lr 0.0001 --epochs 1 --patience 100 \
    --save-prefix "test_gpu1"

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"
echo ""
echo "If both tests succeeded, the setup is ready for full training."
echo "Run the full training with:"
echo "  ./scripts/retrain_single_gpu_parallel.sh"
echo ""
