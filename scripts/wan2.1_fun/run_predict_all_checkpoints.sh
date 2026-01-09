#!/bin/bash
# ==============================================================================
# Run PSI Control inference with multiple checkpoints
#
# This script runs predict_v2v_psi_control.py with:
# 1. No checkpoint (baseline with random PSI projection)
# 2. All available checkpoints in the specified folder
#
# Usage:
#   ./scripts/wan2.1_fun/run_predict_all_checkpoints.sh
#   CUDA_VISIBLE_DEVICES=7 ./scripts/wan2.1_fun/run_predict_all_checkpoints.sh
# ==============================================================================

set -e  # Exit on error

# Configuration
CHECKPOINT_DIR="output_psi_control_test"  # Directory containing checkpoints
VIDEO_PATH="/ccn2/dataset/kinetics400/Kinetics400/k400/val/--07WQ2iBlw_000001_000011.mp4"
SEEDS="42,123,456,789,1024"  # 5 different seeds (comma-separated)
SAVE_PATH="samples/psi-control-output"

# Change to project root
cd "$(dirname "$0")/../.."
echo "Working directory: $(pwd)"

# ==============================================================================
# 1. Run baseline (no checkpoint) with multiple seeds
# ==============================================================================
echo ""
echo "=============================================================="
echo "Running BASELINE (no checkpoint - random PSI projection)"
echo "  Seeds: $SEEDS"
echo "=============================================================="

python examples/wan2.1_fun/predict_v2v_psi_control.py \
    --video_path "$VIDEO_PATH" \
    --seeds "$SEEDS" \
    --checkpoint_name "${CHECKPOINT_DIR}-0000" \
    --save_path "$SAVE_PATH"

# ==============================================================================
# 2. Find and run all checkpoints
# ==============================================================================
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "WARNING: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    echo "Please copy your checkpoints to this directory first."
    exit 1
fi

# Find all checkpoint files and extract iteration numbers
echo ""
echo "=============================================================="
echo "Searching for checkpoints in: $CHECKPOINT_DIR"
echo "=============================================================="

# Get unique iteration numbers from checkpoint files
# Pattern: checkpoint-XXXX.safetensors (not the comfyui compatible ones)
ITERATIONS=$(ls "$CHECKPOINT_DIR"/checkpoint-*.safetensors 2>/dev/null | \
    grep -v "compatible_with_comfyui" | \
    sed 's/.*checkpoint-\([0-9]*\)\.safetensors/\1/' | \
    sort -n | uniq)

if [ -z "$ITERATIONS" ]; then
    echo "No checkpoint files found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found checkpoints for iterations: $ITERATIONS"
echo ""

# Run for each checkpoint (models loaded once, all seeds processed together)
for ITER in $ITERATIONS; do
    LORA_PATH="${CHECKPOINT_DIR}/checkpoint-${ITER}.safetensors"
    PSI_PATH="${CHECKPOINT_DIR}/psi_projection-${ITER}.safetensors"
    CHECKPOINT_NAME="${CHECKPOINT_DIR}-${ITER}"
    
    echo "=============================================================="
    echo "Running checkpoint: $ITER"
    echo "  LoRA: $LORA_PATH"
    echo "  PSI:  $PSI_PATH"
    echo "  Output: $SAVE_PATH/$CHECKPOINT_NAME"
    echo "  Seeds: $SEEDS"
    echo "=============================================================="
    
    # Check if files exist
    if [ ! -f "$LORA_PATH" ]; then
        echo "WARNING: LoRA checkpoint not found: $LORA_PATH"
        continue
    fi
    
    # PSI projection might not exist for all iterations
    PSI_ARG=""
    if [ -f "$PSI_PATH" ]; then
        PSI_ARG="--psi_projection_path $PSI_PATH"
    else
        echo "NOTE: PSI projection not found for iteration $ITER, using random init"
    fi
    
    python examples/wan2.1_fun/predict_v2v_psi_control.py \
        --video_path "$VIDEO_PATH" \
        --seeds "$SEEDS" \
        --lora_path "$LORA_PATH" \
        $PSI_ARG \
        --checkpoint_name "$CHECKPOINT_NAME" \
        --save_path "$SAVE_PATH"
    
    echo ""
done

echo "=============================================================="
echo "All checkpoints processed!"
echo "Results saved to: $SAVE_PATH"
echo "=============================================================="
ls -la "$SAVE_PATH"/

