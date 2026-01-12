#!/bin/bash
# ==============================================================================
# Experiment 6: Temporal-Aware PSI Propagation
# - Random masking (different masks each batch)
# - Random order (different token order each batch)
# - PSI projection enabled (trains semantic feature projection)
# - use_all_tokens=True (4x more features per patch - 32768 vs 8192)
# - psi_temporal_propagation=True (NEW: propagate PSI frame 1 to intermediate latents)
#
# Key difference from exp4:
# Instead of only giving PSI info at latent 0 and second_latent_idx,
# this experiment propagates PSI frame 1 info to ALL intermediate latents
# (from 1 to second_latent_idx) with temporal offset embeddings.
#
# This gives the model "advance notice" of where the video is heading,
# with a countdown signal indicating "frames until target".
#
# Control signal structure:
#   Latent 0: PSI frame 0 + VAE + temporal_offset(0)
#   Latent 1: PSI frame 1 + VAE + temporal_offset(3)  <- "target is 3 frames ahead"
#   Latent 2: PSI frame 1 + VAE + temporal_offset(2)  <- "target is 2 frames ahead"
#   Latent 3: PSI frame 1 + VAE + temporal_offset(1)  <- "target is 1 frame ahead"
#   Latent 4: PSI frame 1 + VAE + temporal_offset(0)  <- "I'm at my target"
#   Latent 5+: mask_embedding (no PSI info)
#
# Usage: bash exp6_temporal_propagation.sh <host_address> <node_rank>
# Example: bash exp6_temporal_propagation.sh n12 0
# ==============================================================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <host_address> <node_rank>"
    echo "  host_address: Master node hostname (e.g., n12)"
    echo "  node_rank: Current node rank (0 for master, 1-3 for workers)"
    exit 1
fi

MASTER_ADDR=$1
NODE_RANK=$2

# Load required modules
module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

# Activate conda environment
source /scratch/m000063/users/wanhee/miniconda3/etc/profile.d/conda.sh
conda activate videox-fun

# WandB API key
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    export WANDB_ENTITY="long-range-prediction"
    export WANDB_RUN_NAME="exp6_temporal_propagation"
    echo "WandB API key loaded from ~/.wandb_api_key"
else
    echo "WARNING: ~/.wandb_api_key not found. WandB logging may fail."
fi

# Change to working directory
cd /scratch/m000063/users/wanhee/VideoX-Fun

# Create logs directory if it doesn't exist
mkdir -p /scratch/m000063/users/wanhee/VideoX-Fun/logs

# Model and data paths
export MODEL_NAME="/scratch/m000063/users/wanhee/VideoX-Fun/models/Wan2.1-Fun-1.3B-Control"
export DATASET_NAME="/scratch/m000063/data/bvd2/kinetics700"
export DATASET_META_NAME="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/kinetics700_49f.csv"

# Output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR="/scratch/m000063/users/wanhee/VideoX-Fun/output_exp6_temporal_propagation_${TIMESTAMP}"
echo "Output directory: $OUTPUT_DIR"

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Multi-node configuration: 4 nodes x 8 GPUs = 32 GPUs total
export MASTER_PORT=45682
NNODES=4
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# Batch size calculation: 128 = 1 * 4 grad_accum * 32 GPUs
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8

echo "=============================================="
echo "Experiment 6: Temporal-Aware PSI Propagation"
echo "=============================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE (4 nodes × 8 GPUs)"
echo "NODE_RANK: $NODE_RANK"
echo "Effective batch size: 128 (${TRAIN_BATCH_SIZE} × ${GRADIENT_ACCUMULATION_STEPS} grad_accum × ${WORLD_SIZE} GPUs)"
echo "PSI Settings:"
echo "  - psi_fix_masking_seed: None (RANDOM)"
echo "  - psi_fix_order_seed: None (RANDOM)"
echo "  - psi_vae_only: False (WITH PSI semantic projection)"
echo "  - psi_use_all_tokens: True (4x features - 32768 dim)"
echo "  - psi_temporal_propagation: True (NEW - propagate with offset embeddings)"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

accelerate launch \
    --multi_gpu \
    --num_processes $WORLD_SIZE \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision="bf16" \
    scripts/wan2.1_fun/train_control_lora_psi.py \
    --config_path="config/wan2.1/wan_civitai.yaml" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_NAME \
    --train_data_meta=$DATASET_META_NAME \
    --image_sample_size=512 \
    --video_sample_size=512 \
    --token_sample_size=512 \
    --fix_sample_size 512 512 \
    --video_sample_stride=2 \
    --video_sample_n_frames=49 \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --video_repeat=1 \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --dataloader_num_workers=8 \
    --max_train_steps=20000 \
    --checkpointing_steps=1000 \
    --learning_rate=1e-05 \
    --seed=123 \
    --output_dir=$OUTPUT_DIR \
    --gradient_checkpointing \
    --mixed_precision="bf16" \
    --adam_weight_decay=3e-2 \
    --adam_epsilon=1e-10 \
    --vae_mini_batch=1 \
    --max_grad_norm=0.05 \
    --training_with_video_token_length \
    --uniform_sampling \
    --enable_bucket \
    --train_mode="control_ref" \
    --control_ref_image="random" \
    --add_full_ref_image_in_self_attention \
    --rank=64 \
    --network_alpha=32 \
    --target_name="q,k,v,ffn.0,ffn.2" \
    --use_peft_lora \
    --enable_psi_control \
    --psi_use_all_tokens \
    --psi_temporal_propagation \
    --save_state \
    --report_to="wandb" \
    --tracker_project_name="wan-psi-control"
    # To resume training, uncomment the line below and set OUTPUT_DIR to existing checkpoint dir:
    # --resume_from_checkpoint="latest"



