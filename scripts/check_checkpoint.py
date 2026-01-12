#!/usr/bin/env python3
"""
Script to inspect checkpoint contents and verify what's being saved.
Usage: python check_checkpoint.py <checkpoint_folder_path>
Example: python check_checkpoint.py /scratch/m000063/users/wanhee/VideoX-Fun/output_exp4_all_tokens_with_projection_20260111_202638/checkpoint-1000
"""

import os
import sys
import pickle
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def check_checkpoint(checkpoint_path):
    print("=" * 80)
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        return
    
    if not os.path.isdir(checkpoint_path):
        print(f"ERROR: Checkpoint path is not a directory: {checkpoint_path}")
        return
    
    # List all files in checkpoint
    print("\n📁 Files in checkpoint directory:")
    files = sorted(os.listdir(checkpoint_path))
    for f in files:
        fpath = os.path.join(checkpoint_path, f)
        size = os.path.getsize(fpath)
        size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.2f} KB"
        print(f"  - {f} ({size_str})")
    
    print("\n" + "=" * 80)
    
    # Check sampler_pos_start.pkl (contains step/epoch info)
    pkl_path = os.path.join(checkpoint_path, "sampler_pos_start.pkl")
    if os.path.exists(pkl_path):
        print("\n📊 sampler_pos_start.pkl contents:")
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            print(f"  Type: {type(data)}")
            if isinstance(data, tuple):
                print(f"  Tuple length: {len(data)}")
                for i, item in enumerate(data):
                    print(f"  Item {i}: {type(item).__name__} = {item}")
            else:
                print(f"  Value: {data}")
        except Exception as e:
            print(f"  ERROR loading: {e}")
    else:
        print("\n⚠️  sampler_pos_start.pkl NOT FOUND")
    
    # Check random_states pkl files
    random_states = [f for f in files if f.startswith("random_states") and f.endswith(".pkl")]
    if random_states:
        print(f"\n🎲 Random state files found: {random_states}")
        for rs_file in random_states[:1]:  # Check first one
            rs_path = os.path.join(checkpoint_path, rs_file)
            try:
                with open(rs_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"  {rs_file}: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            except Exception as e:
                print(f"  ERROR loading {rs_file}: {e}")
    
    # Check optimizer state
    optimizer_files = [f for f in files if f.startswith("optimizer")]
    if optimizer_files:
        print(f"\n⚙️  Optimizer files: {optimizer_files}")
        for opt_file in optimizer_files:
            opt_path = os.path.join(checkpoint_path, opt_file)
            try:
                if opt_file.endswith('.bin') or opt_file.endswith('.pt'):
                    state = torch.load(opt_path, map_location='cpu')
                    if isinstance(state, dict):
                        print(f"  {opt_file} keys: {list(state.keys())}")
                        if 'state' in state:
                            print(f"    - 'state' has {len(state['state'])} parameter groups")
                        if 'param_groups' in state:
                            for i, pg in enumerate(state['param_groups']):
                                print(f"    - param_group {i}: lr={pg.get('lr', 'N/A')}, step info in state")
            except Exception as e:
                print(f"  ERROR loading {opt_file}: {e}")
    
    # Check scheduler state
    scheduler_files = [f for f in files if f.startswith("scheduler")]
    if scheduler_files:
        print(f"\n📅 Scheduler files: {scheduler_files}")
        for sched_file in scheduler_files:
            sched_path = os.path.join(checkpoint_path, sched_file)
            try:
                if sched_file.endswith('.bin') or sched_file.endswith('.pt'):
                    state = torch.load(sched_path, map_location='cpu')
                    print(f"  {sched_file}: {type(state)}")
                    if isinstance(state, dict):
                        print(f"    Keys: {list(state.keys())}")
                        if 'last_epoch' in state:
                            print(f"    last_epoch: {state['last_epoch']}")
                        if '_step_count' in state:
                            print(f"    _step_count: {state['_step_count']}")
            except Exception as e:
                print(f"  ERROR loading {sched_file}: {e}")
    
    # Check LoRA model weights
    lora_files = [f for f in files if 'lora' in f.lower() and f.endswith('.safetensors')]
    if lora_files:
        print(f"\n🔧 LoRA model files: {lora_files}")
        for lora_file in lora_files:
            lora_path = os.path.join(checkpoint_path, lora_file)
            try:
                state_dict = load_file(lora_path)
                print(f"  {lora_file}: {len(state_dict)} tensors")
                # Show first few keys
                keys = list(state_dict.keys())[:5]
                for k in keys:
                    print(f"    - {k}: {state_dict[k].shape}")
                if len(state_dict) > 5:
                    print(f"    ... and {len(state_dict) - 5} more tensors")
            except Exception as e:
                print(f"  ERROR loading {lora_file}: {e}")
    
    # Check PSI projection weights
    psi_files = [f for f in files if 'psi' in f.lower() and f.endswith('.safetensors')]
    if psi_files:
        print(f"\n🧠 PSI projection files: {psi_files}")
        for psi_file in psi_files:
            psi_path = os.path.join(checkpoint_path, psi_file)
            try:
                state_dict = load_file(psi_path)
                print(f"  {psi_file}: {len(state_dict)} tensors")
                for k, v in state_dict.items():
                    print(f"    - {k}: {v.shape}")
            except Exception as e:
                print(f"  ERROR loading {psi_file}: {e}")
    
    # Check for accelerate state files
    accel_files = [f for f in files if 'accelerator' in f.lower() or 'scaler' in f.lower()]
    if accel_files:
        print(f"\n🚀 Accelerator state files: {accel_files}")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    
    # Try to infer step from checkpoint name
    checkpoint_name = os.path.basename(checkpoint_path)
    if checkpoint_name.startswith("checkpoint-"):
        try:
            step = int(checkpoint_name.split("-")[1])
            print(f"  Step (from folder name): {step}")
        except:
            pass
    
    # Check if this is a complete checkpoint
    required_files = ["lora_diffusion_pytorch_model.safetensors", "optimizer.bin", "scheduler.bin", "sampler_pos_start.pkl"]
    missing = [f for f in required_files if f not in files and f.replace('.bin', '.pt') not in files]
    if missing:
        print(f"  ⚠️  Missing files for full resume: {missing}")
    else:
        print(f"  ✅ All required files present for resume")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_checkpoint.py <checkpoint_folder_path>")
        print("Example: python check_checkpoint.py /path/to/output/checkpoint-1000")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    check_checkpoint(checkpoint_path)

