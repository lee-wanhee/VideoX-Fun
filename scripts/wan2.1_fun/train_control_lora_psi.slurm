#!/bin/bash
#SBATCH --job-name=train_psi_control
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000063-pm01
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1000GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_psi_%j.out
#SBATCH --error=logs/train_psi_%j.err

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
mkdir -p logs

# Model and data paths
export MODEL_NAME="/scratch/m000063/users/wanhee/VideoX-Fun/models/Wan2.1-Fun-1.3B-Control"
export DATASET_NAME="/scratch/m000063/data/bvd2/kinetics700"
export DATASET_META_NAME="/scratch/m000063/users/wanhee/VideoX-Fun/datasets/kinetics700_49f.csv"

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

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NNODES: $NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NODEID: $SLURM_NODEID"

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
        --video_sample_stride=2 \
        --video_sample_n_frames=49 \
        --train_batch_size=1 \
        --video_repeat=1 \
        --gradient_accumulation_steps=1 \
        --dataloader_num_workers=8 \
        --num_train_epochs=100 \
        --checkpointing_steps=50 \
        --learning_rate=1e-05 \
        --seed=42 \
        --output_dir="output_dir" \
        --gradient_checkpointing \
        --mixed_precision="bf16" \
        --adam_weight_decay=3e-2 \
        --adam_epsilon=1e-10 \
        --vae_mini_batch=1 \
        --max_grad_norm=0.05 \
        --training_with_video_token_length \
        --uniform_sampling \
        --train_mode="control_ref" \
        --control_ref_image="random" \
        --add_full_ref_image_in_self_attention \
        --rank=64 \
        --network_alpha=32 \
        --target_name="q,k,v,ffn.0,ffn.2" \
        --use_peft_lora \
        --enable_psi_control
'

