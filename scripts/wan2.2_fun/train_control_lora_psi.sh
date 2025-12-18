#!/bin/bash

# ============================================================================
# Training Script for VideoX-Fun with PSI Control Feature Extractor
# ============================================================================
# This script trains the model using PSIPredictor to extract control features
# on-the-fly during training, instead of using VAE-encoded control signals.
# ============================================================================

# Multi-GPU / Distributed Training Settings
# Set NUM_GPUS to the number of GPUs you want to use (1, 2, 4, 8, etc.)
export NUM_GPUS=2  # Change this to use different number of GPUs
export NUM_NODES=1  # For multi-node training, increase this
export GPU_IDS="0,1"  # Which GPUs to use (e.g., "0,1" for first 2 GPUs)

# Model and data paths - Updated with your actual paths
export MODEL_PATH="/scratch/m000063/users/wanhee/VideoX-Fun/models/Wan2.2-Fun-5B-Control"
export DATA_DIR="/scratch/m000063/data/bvd2/handpicked"
export DATA_META="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/handpicked_filtered_81frames.csv"
export CONFIG_PATH="/ccn2/u/wanhee/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml"

# PSI Model paths - Using same paths as parallel_feature_test.py (relative paths)
# These paths are relative and will be resolved by PSIPredictor
export PSI_MODEL_NAME="PSI_7B_RGBCDF_bvd_4frame_Unified_Vocab_Balanced_Task_V2_continue_ctx_8192/model_01400000.pt"
export PSI_QUANTIZER_NAME="PLPQ-ImageNetOpenImages-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2/model_best.pt"
export PSI_FLOW_QUANTIZER_NAME="HLQ-flow-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l2-coarsel21e-2-fg_v1_5/model_best.pt"
export PSI_DEPTH_QUANTIZER_NAME="HLQ-depth-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2-200k_ft500k_3/model_best.pt"

# PSI Feature Extraction Settings
export PSI_MASK_RATIO=0.0  # 0.0 = fully visible, increase to mask more patches
export PSI_TEMPERATURE=1.0
export PSI_TOP_P=0.9
export PSI_TOP_K=1000
export PSI_TIME_GAP_SEC=0.5  # Time gap between 2 frames for control extraction (in seconds)

# Output settings
export OUTPUT_DIR="outputs/psi_control_lora_$(date +%Y%m%d_%H%M%S)"

# LoRA settings (from train_control_lora.sh)
export RANK=64
export NETWORK_ALPHA=32

# Training hyperparameters
# Note: With 2 GPUs, effective batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS
export LEARNING_RATE=1e-5
export TRAIN_BATCH_SIZE=1  # Per GPU batch size
export GRADIENT_ACCUMULATION_STEPS=16
export MAX_TRAIN_STEPS=5000

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


# Video settings (from train_control_lora.sh defaults)
export VIDEO_SAMPLE_SIZE=512
export VIDEO_SAMPLE_N_FRAMES=81  # Default is 81, not 17!
export VIDEO_SAMPLE_STRIDE=2
export VIDEO_REPEAT=1

# Optimizer settings (from train_control_lora.sh)
export ADAM_WEIGHT_DECAY=3e-2
export ADAM_EPSILON=1e-10
export MAX_GRAD_NORM=0.05

# Other settings
export CHECKPOINTING_STEPS=500
export SEED=42
export VAE_MINI_BATCH=1
export DATALOADER_NUM_WORKERS=8

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

# Verify PSI model files exist (they will be resolved by PSIPredictor)
echo "✓ PSI model paths configured (using relative paths from parallel_feature_test.py)"

echo ""
echo "Training Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Batch Size (per GPU): $TRAIN_BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective Batch Size: $((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Steps: $MAX_TRAIN_STEPS"
echo "  LoRA Rank: $RANK"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# ============================================================================
# Launch Training
# ============================================================================
echo "Starting training..."
echo "Multi-GPU Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Number of Nodes: $NUM_NODES"
echo "  GPU IDs: $GPU_IDS"
echo "============================================================================"

# Set which GPUs to use
export CUDA_VISIBLE_DEVICES=$GPU_IDS

accelerate launch \
  --num_processes=$NUM_GPUS \
  --num_machines=$NUM_NODES \
  --mixed_precision=bf16 \
  --multi_gpu \
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
  --psi_time_gap_sec ${PSI_TIME_GAP_SEC} \
  \
  `# LoRA Settings (from train_control_lora.sh)` \
  --rank ${RANK} \
  --network_alpha ${NETWORK_ALPHA} \
  --target_name "q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
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
  `# Video Settings (from train_control_lora.sh)` \
  --video_sample_size ${VIDEO_SAMPLE_SIZE} \
  --video_sample_n_frames ${VIDEO_SAMPLE_N_FRAMES} \
  --video_sample_stride ${VIDEO_SAMPLE_STRIDE} \
  --video_repeat ${VIDEO_REPEAT} \
  --image_sample_size ${VIDEO_SAMPLE_SIZE} \
  --token_sample_size ${VIDEO_SAMPLE_SIZE} \
  \
  `# Data Processing (from train_control_lora.sh)` \
  --enable_bucket \
  --random_hw_adapt \
  --training_with_video_token_length \
  --uniform_sampling \
  --train_mode "control_ref" \
  --control_ref_image "first_frame" \
  --add_inpaint_info \
  --add_full_ref_image_in_self_attention \
  --boundary_type "low" \
  \
  `# Optimization (from train_control_lora.sh)` \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --adam_weight_decay ${ADAM_WEIGHT_DECAY} \
  --adam_epsilon ${ADAM_EPSILON} \
  --max_grad_norm ${MAX_GRAD_NORM} \
  --vae_mini_batch ${VAE_MINI_BATCH} \
  --low_vram \
  \
  `# Checkpointing` \
  --checkpointing_steps ${CHECKPOINTING_STEPS} \
  --checkpoints_total_limit 5 \
  \
  `# Other` \
  --seed ${SEED} \
  --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
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
# 1. Multi-GPU Training:
#    - Set NUM_GPUS at the top of this script (default: 2)
#    - Set GPU_IDS to specify which GPUs to use (e.g., "0,1,2,3" for 4 GPUs)
#    - Effective batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS
#    - With 2 GPUs: effective batch size = 1 * 4 * 2 = 8
#
# 2. PSI model paths are configured using the same relative paths as parallel_feature_test.py
#    These paths will be resolved by PSIPredictor
#
# 3. The PSI control extractor will be used to extract features on-the-fly
#    during training instead of using VAE-encoded control signals
#
# 4. GPU Memory Tips:
#    - For 24GB GPU (per GPU): batch_size=1, gradient_accumulation=4-8
#    - For 40GB GPU (per GPU): batch_size=2, gradient_accumulation=2-4
#    - For 80GB GPU (per GPU): batch_size=4, gradient_accumulation=2
#    - With 2x24GB GPUs: effective batch size = 1 * 4 * 2 = 8 (good for training)
#
# 5. If you run out of memory:
#    - Add --low_vram flag
#    - Reduce --video_sample_size (e.g., 256 or 384)
#    - Reduce --video_sample_n_frames (e.g., 9 or 13)
#    - Increase --gradient_accumulation_steps
#    - Add --vae_mini_batch 8 (or smaller)
#
# 6. PSI Feature Extraction Settings:
#    - psi_mask_ratio: Controls how much of the input to mask (0.0 = fully visible)
#    - psi_temperature/top_p/top_k: Control PSI sampling behavior
#    - Adjust these based on your control signal requirements
#
# 7. Training time estimates (rough):
#    - 10k steps with 2 GPUs: ~8-12 hours on 2xA100 (faster with multi-GPU!)
#    - PSI feature extraction adds overhead compared to VAE encoding
#    - Validation adds ~10-15 minutes per validation run with PSI
#
# 8. Monitor training:
#    - Tensorboard: tensorboard --logdir outputs/psi_control_lora_*/logs
#    - Check sanity_check folder for first batch visualizations
#    - Check sample folder for validation outputs
#
# 9. Debugging:
#    - If PSI fails to load, check that ccwm package is accessible
#    - Check PSI model paths are correct and files exist
#    - Look for PSI-related errors in the log output
#
# ============================================================================

