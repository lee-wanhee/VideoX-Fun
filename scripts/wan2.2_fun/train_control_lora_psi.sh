#!/bin/bash

# ============================================================================
# Training Script for VideoX-Fun with PSI Control Feature Extractor
# ============================================================================
# This script trains the model using PSIPredictor to extract control features
# on-the-fly during training, instead of using VAE-encoded control signals.
# ============================================================================

# Model and data paths - Updated with your actual paths
export MODEL_PATH="/scratch/m000063/users/wanhee/VideoX-Fun/models/Wan2.2-Fun-5B-Control"
export DATA_DIR="/scratch/m000063/data/bvd2/handpicked"
export DATA_META="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/handpicked_debug.csv"
export CONFIG_PATH="${MODEL_PATH}/config.json"

# PSI Model paths - Update these to point to your actual PSI model locations
# These paths should be relative to where your PSI models are stored
# Based on parallel_feature_test.py, these might be in a models directory
export PSI_MODELS_DIR="/path/to/psi/models"  # UPDATE THIS PATH!
export PSI_MODEL_NAME="${PSI_MODELS_DIR}/PSI_7B_RGBCDF_bvd_4frame_Unified_Vocab_Balanced_Task_V2_continue_ctx_8192/model_01400000.pt"
export PSI_QUANTIZER_NAME="${PSI_MODELS_DIR}/PLPQ-ImageNetOpenImages-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2/model_best.pt"
export PSI_FLOW_QUANTIZER_NAME="${PSI_MODELS_DIR}/HLQ-flow-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l2-coarsel21e-2-fg_v1_5/model_best.pt"
export PSI_DEPTH_QUANTIZER_NAME="${PSI_MODELS_DIR}/HLQ-depth-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2-200k_ft500k_3/model_best.pt"

# PSI Feature Extraction Settings
export PSI_MASK_RATIO=0.0  # 0.0 = fully visible, increase to mask more patches
export PSI_TEMPERATURE=1.0
export PSI_TOP_P=0.9
export PSI_TOP_K=1000

# Output settings
export OUTPUT_DIR="outputs/psi_control_lora_$(date +%Y%m%d_%H%M%S)"

# LoRA settings
export RANK=128
export NETWORK_ALPHA=64

# Training hyperparameters
export LEARNING_RATE=1e-4
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export MAX_TRAIN_STEPS=10000

# Validation settings
export VALIDATION_STEPS=1000
export VALIDATION_PROMPTS=(
    ""
    ""
    ""
    ""
    ""
)
export VALIDATION_PATHS=(
    "/scratch/m000063/data/bvd2/handpicked/vYAUIF61MCo_part0000.mp4"
    "/scratch/m000063/data/bvd2/handpicked/yqDBM7LyNZQ_part0001.mp4"
    "/scratch/m000063/data/bvd2/handpicked/nXpp2e6cv20_part0028.mp4"
    "/scratch/m000063/data/bvd2/handpicked/1lYYXkLMhbY_part0016.mp4"
    "/scratch/m000063/data/bvd2/handpicked/QYiddvAhffI_part0002.mp4"
)


# Video settings
export VIDEO_SAMPLE_SIZE=512
export VIDEO_SAMPLE_N_FRAMES=17
export VIDEO_SAMPLE_STRIDE=4

# Other settings
export CHECKPOINTING_STEPS=500
export SEED=42

# ============================================================================
# Pre-flight Checks
# ============================================================================
echo "============================================================================"
echo "PSI Control LoRA Training - Pre-flight Checks"
echo "============================================================================"

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi
echo "✓ Data directory: $DATA_DIR"

if [ ! -f "$DATA_META" ]; then
    echo "ERROR: Data metadata CSV not found: $DATA_META"
    exit 1
fi
echo "✓ Data metadata: $DATA_META"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi
echo "✓ Model path: $MODEL_PATH"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "WARNING: Config not found at $CONFIG_PATH"
    echo "  Trying configuration.json..."
    export CONFIG_PATH="${MODEL_PATH}/configuration.json"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: No config file found in $MODEL_PATH"
        exit 1
    fi
fi
echo "✓ Config path: $CONFIG_PATH"

# Check PSI model paths
echo ""
echo "PSI Model Paths:"
echo "  Model: $PSI_MODEL_NAME"
echo "  Quantizer: $PSI_QUANTIZER_NAME"
echo "  Flow Quantizer: $PSI_FLOW_QUANTIZER_NAME"
echo "  Depth Quantizer: $PSI_DEPTH_QUANTIZER_NAME"

if [ "$PSI_MODELS_DIR" = "/path/to/psi/models" ]; then
    echo ""
    echo "WARNING: PSI_MODELS_DIR is set to default placeholder!"
    echo "  Please update PSI_MODELS_DIR in this script to point to your actual PSI model directory."
    echo "  You can check where your PSI models are by looking at the parallel_feature_test.py paths."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Training Configuration:"
echo "  Batch Size: $TRAIN_BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective Batch Size: $((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Steps: $MAX_TRAIN_STEPS"
echo "  LoRA Rank: $RANK"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# ============================================================================
# Launch Training
# ============================================================================
echo "Starting training..."
echo "============================================================================"

accelerate launch \
  train_control_lora.py \
  --pretrained_model_name_or_path "${MODEL_PATH}" \
  --train_data_dir "${DATA_DIR}" \
  --train_data_meta "${DATA_META}" \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  \
  `# PSI Control Extractor Settings` \
  --use_psi_control_extractor \
  --psi_model_name "${PSI_MODEL_NAME}" \
  --psi_quantizer_name "${PSI_QUANTIZER_NAME}" \
  --psi_flow_quantizer_name "${PSI_FLOW_QUANTIZER_NAME}" \
  --psi_depth_quantizer_name "${PSI_DEPTH_QUANTIZER_NAME}" \
  --psi_mask_ratio ${PSI_MASK_RATIO} \
  --psi_temperature ${PSI_TEMPERATURE} \
  --psi_top_p ${PSI_TOP_P} \
  --psi_top_k ${PSI_TOP_K} \
  \
  `# LoRA Settings` \
  --rank ${RANK} \
  --network_alpha ${NETWORK_ALPHA} \
  --target_name "attn1,attn2" \
  \
  `# Training Settings` \
  --learning_rate ${LEARNING_RATE} \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --max_train_steps ${MAX_TRAIN_STEPS} \
  --lr_scheduler "constant_with_warmup" \
  --lr_warmup_steps 500 \
  \
  `# Validation Settings` \
  --validation_steps ${VALIDATION_STEPS} \
  --validation_prompts "${VALIDATION_PROMPTS[@]}" \
  --validation_paths "${VALIDATION_PATHS[@]}" \
  \
  `# Video Settings` \
  --video_sample_size ${VIDEO_SAMPLE_SIZE} \
  --video_sample_n_frames ${VIDEO_SAMPLE_N_FRAMES} \
  --video_sample_stride ${VIDEO_SAMPLE_STRIDE} \
  --image_sample_size ${VIDEO_SAMPLE_SIZE} \
  \
  `# Data Processing` \
  --enable_bucket \
  --random_hw_adapt \
  --train_mode "control" \
  \
  `# Optimization` \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --use_8bit_adam \
  \
  `# Checkpointing` \
  --checkpointing_steps ${CHECKPOINTING_STEPS} \
  --checkpoints_total_limit 5 \
  \
  `# Other` \
  --seed ${SEED} \
  --dataloader_num_workers 4 \
  --report_to tensorboard

# ============================================================================
# Post-Training
# ============================================================================
echo ""
echo "============================================================================"
echo "Training completed!"
echo "============================================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "To resume training from a checkpoint:"
echo "  Add --resume_from_checkpoint <checkpoint_path> to the accelerate launch command"
echo ""

# ============================================================================
# Notes:
# ============================================================================
# 
# 1. IMPORTANT: Update PSI_MODELS_DIR to point to your actual PSI model directory
#    The PSI models (model_name, quantizer_name, etc.) should be accessible paths
#
# 2. The PSI control extractor will be used to extract features on-the-fly
#    during training instead of using VAE-encoded control signals
#
# 3. GPU Memory Tips:
#    - For 24GB GPU: batch_size=1, gradient_accumulation=4-8
#    - For 40GB GPU: batch_size=2, gradient_accumulation=2-4
#    - For 80GB GPU: batch_size=4, gradient_accumulation=2
#
# 4. If you run out of memory:
#    - Add --low_vram flag
#    - Reduce --video_sample_size (e.g., 256 or 384)
#    - Reduce --video_sample_n_frames (e.g., 9 or 13)
#    - Increase --gradient_accumulation_steps
#    - Add --vae_mini_batch 8 (or smaller)
#
# 5. PSI Feature Extraction Settings:
#    - psi_mask_ratio: Controls how much of the input to mask (0.0 = fully visible)
#    - psi_temperature/top_p/top_k: Control PSI sampling behavior
#    - Adjust these based on your control signal requirements
#
# 6. Training time estimates (rough):
#    - 10k steps with batch_size=1: ~15-25 hours on A100 (slower due to PSI extraction)
#    - PSI feature extraction adds overhead compared to VAE encoding
#    - Validation adds ~10-15 minutes per validation run with PSI
#
# 7. Monitor training:
#    - Tensorboard: tensorboard --logdir outputs/psi_control_lora_*/logs
#    - Check sanity_check folder for first batch visualizations
#    - Check sample folder for validation outputs
#
# 8. Debugging:
#    - If PSI fails to load, check that ccwm package is accessible
#    - Check PSI model paths are correct and files exist
#    - Look for PSI-related errors in the log output
#
# ============================================================================

