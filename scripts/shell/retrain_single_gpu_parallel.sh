#!/bin/bash
# =============================================================================
# Single GPU Retraining Experiment - Parallel Execution
# Purpose: Confirm DataParallel evaluation degradation findings
# Strategy: Train all LSTM models on single GPU, parallel across 2 GPUs
# =============================================================================

set -e  # Exit on error

PROJECT_ROOT="/home/lrud1314/PROJECTS_WORKING/THESIS 2025"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Single GPU LSTM Retraining Experiment"
echo "========================================"
echo "Date: $(date)"
echo "Purpose: Confirm DataParallel evaluation issue"
echo ""

# Create results directory
RESULTS_DIR="results/single_gpu_retraining_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/training.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "Results directory: $RESULTS_DIR"
echo ""

# =============================================================================
# Training Configuration
# =============================================================================

EPOCHS=100
BATCH_SIZE=32
LR=0.0001
PATIENCE=15

# Model configurations
declare -A MODELS=(
    ["market_lags_512x7"]="market_lags:512:7:0.5"
    ["jump_aware_512x7"]="jump_aware:512:7:0.5"
    ["market_jumps_512x7"]="market_jumps:512:7:0.5"
    ["market_256x3"]="market:256:3:0.4"
)

# =============================================================================
# Function: Train single model on specified GPU
# =============================================================================

train_model() {
    local MODEL_NAME=$1
    local MODEL_TYPE=$2
    local HIDDEN_SIZE=$3
    local NUM_LAYERS=$4
    local DROPOUT=$5
    local GPU_ID=$6

    echo "----------------------------------------"
    echo "Training: $MODEL_NAME on GPU $GPU_ID"
    echo "----------------------------------------"

    CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python cli/bin/train.py "$MODEL_TYPE" \
        --hidden-size "$HIDDEN_SIZE" \
        --num-layers "$NUM_LAYERS" \
        --dropout "$DROPOUT" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --save-prefix "single_gpu_${MODEL_NAME}"

    # Copy result to our results directory
    LATEST_RESULT=$(ls -t results/cli_training/*_${MODEL_TYPE}_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        cp "$LATEST_RESULT" "$RESULTS_DIR/${MODEL_NAME}_result.json"
        echo "Result saved: $RESULTS_DIR/${MODEL_NAME}_result.json"
    fi

    echo "Completed: $MODEL_NAME on GPU $GPU_ID"
    echo ""
}

# =============================================================================
# Training Schedule - Parallel Execution on 2 GPUs
# =============================================================================

echo "========================================"
echo "Starting Parallel Training Schedule"
echo "========================================"
echo ""

# Round 1: Large models (512x7) - Two parallel, then third
echo "=== ROUND 1: Large Models (512x7) ==="
echo "Starting: $(date)"

train_model "market_lags_512x7" "market_lags" 512 7 0.5 0 &
PID1=$!

train_model "jump_aware_512x7" "jump_aware" 512 7 0.5 1 &
PID2=$!

# Wait for both to complete
wait $PID1
wait $PID2

echo "First two 512x7 models completed: $(date)"
echo ""

# Third 512x7 model
train_model "market_jumps_512x7" "market_jumps" 512 7 0.5 0

echo "Round 1 completed: $(date)"
echo ""

# Round 2: Medium model (256x3)
echo "=== ROUND 2: Medium Model (256x3) ==="
echo "Starting: $(date)"

train_model "market_256x3" "market" 256 3 0.4 1

echo "Round 2 completed: $(date)"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "========================================"
echo "Training Complete - Summary"
echo "========================================"
echo "Results directory: $RESULTS_DIR"
echo ""

# Extract and display key metrics
echo "Performance Summary (Single GPU):"
echo "-----------------------------------"

for model in "${!MODELS[@]}"; do
    if [ -f "$RESULTS_DIR/${model}_result.json" ]; then
        R2=$(jq -r '.evaluation.overall."R²" // "N/A"' "$RESULTS_DIR/${model}_result.json")
        RMSE=$(jq -r '.evaluation.overall.RMSE // "N/A"' "$RESULTS_DIR/${model}_result.json")
        MAE=$(jq -r '.evaluation.overall.MAE // "N/A"' "$RESULTS_DIR/${model}_result.json")
        printf "%-25s R²=%-8s RMSE=%-8s MAE=%-8s\n" "$model" "$R2" "$RMSE" "$MAE"
    else
        echo "$model: FAILED - No result file"
    fi
done

echo ""
echo "========================================"
echo "Comparison with Previous Multi-GPU Results"
echo "========================================"
echo ""
echo "Expected improvements (DataParallel evaluation degradation):"
echo "  - market_lags (512x7): Should improve from R²=0.8021 to R²≈0.93"
echo "  - jump_aware (512x7): Should improve from R²=0.800 to R²≈0.93"
echo "  - Smaller models: Minimal change (less affected by DataParallel)"
echo ""

echo "Full log saved to: $LOG_FILE"
echo "========================================"
