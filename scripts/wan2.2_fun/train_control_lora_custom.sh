#!/bin/bash

# ============================================================================
# Training Script for VideoX-Fun with Custom Control Extractor
# ============================================================================
# This script demonstrates how to train the model with a custom control
# feature extractor instead of using VAE-encoded control signals.
# ============================================================================

# Model and data paths
export MODEL_PATH="/path/to/VideoX-Fun-pretrained-model"
export DATA_DIR="/path/to/your/training/videos"
export DATA_META="/path/to/your/training/metadata.csv"
export CONFIG_PATH="${MODEL_PATH}/config.yaml"

# Custom control extractor (your model)
export CONTROL_EXTRACTOR_PATH="/path/to/your/control_extractor.pth"  # Set to "none" if using dummy/random init

# Output settings
export OUTPUT_DIR="outputs/custom_control_lora_$(date +%Y%m%d_%H%M%S)"

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
    "A person dancing gracefully"
    "A person walking forward"
)
export VALIDATION_PATHS=(
    "/path/to/validation_video1.mp4"
    "/path/to/validation_video2.mp4"
)

# Video settings
export VIDEO_SAMPLE_SIZE=512
export VIDEO_SAMPLE_N_FRAMES=17
export VIDEO_SAMPLE_STRIDE=4

# Other settings
export CHECKPOINTING_STEPS=500
export SEED=42

# ============================================================================
# Accelerate Configuration
# ============================================================================
# Make sure you have run: accelerate config
# Or create a custom accelerate config file

# ============================================================================
# Launch Training
# ============================================================================

accelerate launch \
  train_control_lora.py \
  --pretrained_model_name_or_path "${MODEL_PATH}" \
  --train_data_dir "${DATA_DIR}" \
  --train_data_meta "${DATA_META}" \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  \
  `# Custom Control Extractor Settings` \
  --use_custom_control_extractor \
  --control_extractor_path "${CONTROL_EXTRACTOR_PATH}" \
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
# Notes:
# ============================================================================
# 
# 1. Make sure you have implemented your control extractor in 
#    custom_control_extractor.py before running this script
#
# 2. If you haven't trained your control extractor yet, you can start with
#    a dummy model by setting CONTROL_EXTRACTOR_PATH to "none" or removing
#    the --control_extractor_path argument
#
# 3. Adjust batch size and gradient accumulation based on your GPU memory:
#    - For 24GB GPU: batch_size=1, gradient_accumulation=4-8
#    - For 40GB GPU: batch_size=2, gradient_accumulation=2-4
#    - For 80GB GPU: batch_size=4, gradient_accumulation=2
#
# 4. If you run out of memory, try:
#    - Add --low_vram flag
#    - Reduce --video_sample_size (e.g., 256 or 384)
#    - Reduce --video_sample_n_frames (e.g., 9 or 13)
#    - Increase --gradient_accumulation_steps
#    - Add --vae_mini_batch 8 (or smaller)
#
# 5. Training time estimates (rough):
#    - 10k steps with batch_size=1: ~10-20 hours on A100
#    - Validation adds ~5-10 minutes per validation run
#
# 6. Monitor training:
#    - Tensorboard: tensorboard --logdir outputs/custom_control_lora_*/logs
#    - Check sanity_check folder for first batch visualizations
#    - Check sample folder for validation outputs
#
# ============================================================================

