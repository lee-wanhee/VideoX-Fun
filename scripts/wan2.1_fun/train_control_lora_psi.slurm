#!/bin/bash
#SBATCH --job-name=train_psi_control
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000063-pm04
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1000GB
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/m000063/users/wanhee/VideoX-Fun/logs/train_psi_%j.out
#SBATCH --error=/scratch/m000063/users/wanhee/VideoX-Fun/logs/train_psi_%j.err

# ==============================================================================
# Training Configuration (based on paper settings)
# ==============================================================================
# Paper: "batch size 128 with learning rates of 1×10−5 and 1×10−6 for stages 1 and 2"
# Stage 1: 4.8K steps on OpenVid-1M (filtered 0.6M videos)
# Stage 2: 800 steps on cleaner synthetic data (Wan 2.1)
#
# With 4 nodes × 8 GPUs = 32 GPUs total
# Effective batch size = train_batch_size × gradient_accumulation_steps × num_gpus
# 128 = 1 × 4 × 32
# ==============================================================================

# Load required modules
module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

# Activate conda environment
source ~/.bashrc
conda activate videox-fun

# WandB API key (read from secure file, not in version control)
# To set up: echo "your-api-key" > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    export WANDB_RUN_NAME="stage1_lora64_lr1e5"
    echo "WandB API key loaded from ~/.wandb_api_key"
else
    echo "WARNING: ~/.wandb_api_key not found. WandB logging may fail."
    echo "To fix: echo 'your-api-key' > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key"
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
export OUTPUT_DIR="/scratch/m000063/users/wanhee/VideoX-Fun/output_psi_control_$(date +%Y%m%d_%H%M%S)"

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Get master node info
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=45678

# Calculate total number of processes
NNODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "=============================================="
echo "Training Configuration:"
echo "=============================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE (4 nodes × 8 GPUs)"
echo "NNODES: $NNODES"
echo "Effective batch size: 128 (1 × 4 grad_accum × 32 GPUs)"
echo "Learning rate: 1e-5 (Stage 1)"
echo "Max steps: 4800 (Stage 1)"
echo "Checkpointing: every 1000 steps (~3 hours)"
echo "Validation: DISABLED (run separately after training)"
echo "Logging: WandB (long-range-prediction/wan-psi-control)"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# Run with srun + accelerate
srun --jobid=$SLURM_JOB_ID bash -c '
    source /scratch/m000063/users/wanhee/miniconda3/etc/profile.d/conda.sh
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
        --max_train_steps=5000 \
        --checkpointing_steps=1000 \
        --learning_rate=1e-05 \
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
        --report_to="wandb" \
        --tracker_project_name="wan-psi-control" \
        --tracker_entity="long-range-prediction" \
        --resume_from_checkpoint="latest"
'

# started the first training at 10pm on 2026-01-07