#!/bin/bash
# ==============================================================================
# Multi-Node Training - Node 0 (Master)
# Run this on n07
# ==============================================================================

# Load required modules
module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

# Activate conda environment
source /scratch/m000063/users/wanhee/miniconda3/etc/profile.d/conda.sh
conda activate videox-fun

# Change to working directory
cd /scratch/m000063/users/wanhee/VideoX-Fun

# Create logs directory if it doesn't exist
mkdir -p /scratch/m000063/users/wanhee/VideoX-Fun/logs

# Model and data paths
export MODEL_NAME="/scratch/m000063/users/wanhee/VideoX-Fun/models/Wan2.1-Fun-1.3B-Control"
export DATASET_NAME="/scratch/m000063/data/bvd2/kinetics700"
export DATASET_META_NAME="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/kinetics700_49f.csv"

# Output directory with timestamp
export OUTPUT_DIR="/scratch/m000063/users/wanhee/VideoX-Fun/output_psi_control_test"

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Multi-node configuration
export MASTER_ADDR=n07
export MASTER_PORT=29500
export NODE_RANK=0

NNODES=2
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "=============================================="
echo "Training Configuration:"
echo "=============================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE (2 nodes × 8 GPUs)"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "Effective batch size: 64 (1 × 4 grad_accum × 16 GPUs)"
echo "Learning rate: 1e-5 (Stage 1)"
echo "Max steps: 5000"
echo "Checkpointing: every 1000 steps"
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
    --train_batch_size=1 \
    --video_repeat=1 \
    --gradient_accumulation_steps=4 \
    --dataloader_num_workers=8 \
    --max_train_steps=5000 \
    --checkpointing_steps=1000 \
    --learning_rate=1e-05 \
    --seed=42 \
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
    --resume_from_checkpoint="latest"

