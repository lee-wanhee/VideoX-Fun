#!/bin/bash
#SBATCH --job-name=train_psi_stage2
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000063-pm01
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1000GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000063/users/wanhee/VideoX-Fun/logs/train_psi_stage2_%j.out
#SBATCH --error=/scratch/m000063/users/wanhee/VideoX-Fun/logs/train_psi_stage2_%j.err

# ==============================================================================
# Stage 2: Fine-tuning on cleaner synthetic data
# ==============================================================================
# Paper: "fine-tuning on cleaner synthetic data for 800 steps (Wan 2.1)"
# Learning rate: 1e-6 (10× lower than Stage 1)
# Max steps: 800 (approximately one epoch)
#
# With 4 nodes × 8 GPUs = 32 GPUs total
# Effective batch size = 1 × 4 × 32 = 128
# ==============================================================================

# Load required modules
module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

# Activate conda environment
source ~/.bashrc
conda activate videox-fun

# Change to working directory
cd /scratch/m000063/users/wanhee/VideoX-Fun

# Create logs directory if it doesn't exist
mkdir -p /scratch/m000063/users/wanhee/VideoX-Fun/logs

# ==============================================================================
# IMPORTANT: Set this to your Stage 1 output directory!
# ==============================================================================
export STAGE1_CHECKPOINT="/scratch/m000063/users/wanhee/VideoX-Fun/output_psi_control_YYYYMMDD_HHMMSS"

# Model and data paths (use cleaner/synthetic data for Stage 2)
export MODEL_NAME="/scratch/m000063/users/wanhee/VideoX-Fun/models/Wan2.1-Fun-1.3B-Control"
export DATASET_NAME="/scratch/m000063/data/bvd2/kinetics700"  # TODO: Replace with cleaner synthetic data
export DATASET_META_NAME="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/kinetics700_49f.csv"  # TODO: Replace with cleaner data meta

# Output directory for Stage 2
export OUTPUT_DIR="/scratch/m000063/users/wanhee/VideoX-Fun/output_psi_control_stage2_$(date +%Y%m%d_%H%M%S)"

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Get master node info
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Calculate total number of processes
NNODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "=============================================="
echo "Stage 2: Fine-tuning Configuration"
echo "=============================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE (4 nodes × 8 GPUs)"
echo "NNODES: $NNODES"
echo "Effective batch size: 128 (1 × 4 grad_accum × 32 GPUs)"
echo "Learning rate: 1e-6 (Stage 2 - 10× lower)"
echo "Max steps: 800 (Stage 2)"
echo "Checkpointing: every 200 steps"
echo "Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "Validation: DISABLED (run separately after training)"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# Run with srun + accelerate
srun --jobid=$SLURM_JOB_ID bash -c '
    source ~/.bashrc
    conda activate videox-fun
    cd /scratch/m000063/users/wanhee/VideoX-Fun
    export NODE_RANK=$SLURM_NODEID
    accelerate launch \
        --multi_gpu \
        --num_processes '$WORLD_SIZE' \
        --num_machines '$NNODES' \
        --machine_rank $NODE_RANK \
        --main_process_ip '$MASTER_ADDR' \
        --main_process_port '$MASTER_PORT' \
        --mixed_precision="bf16" \
        scripts/wan2.1_fun/train_control_lora_psi.py \
        --config_path="config/wan2.1/wan_civitai.yaml" \
        --pretrained_model_name_or_path='$MODEL_NAME' \
        --train_data_dir='$DATASET_NAME' \
        --train_data_meta='$DATASET_META_NAME' \
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
        --max_train_steps=800 \
        --checkpointing_steps=200 \
        --learning_rate=1e-06 \
        --seed=42 \
        --output_dir='$OUTPUT_DIR' \
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
        --resume_from_checkpoint='$STAGE1_CHECKPOINT'
'

