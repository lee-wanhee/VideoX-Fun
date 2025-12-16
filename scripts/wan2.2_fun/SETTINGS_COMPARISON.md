# Settings Comparison: Updated to Match train_control_lora.sh

## What Changed

Your `train_control_lora_psi.sh` has been updated to match the default codebase settings from `train_control_lora.sh`, while keeping the PSI control extractor functionality.

## Key Changes Made

### ❌ OLD Settings → ✅ NEW Settings (from train_control_lora.sh)

| Setting | OLD (Custom) | NEW (Codebase Default) |
|---------|--------------|----------------------|
| **LoRA Rank** | 128 | **64** |
| **LoRA Network Alpha** | 64 | **32** |
| **LoRA Target Layers** | "attn1,attn2" | **"q,k,v,ffn.0,ffn.2"** |
| **Use PEFT LoRA** | Not included | **✓ Added** |
| **Video Frames** | 17 frames | **81 frames** (!!) |
| **Video Stride** | 4 | **2** |
| **Video Resolution** | 512 | **640** |
| **Train Mode** | "control" | **"control_ref"** |
| **Max Grad Norm** | Not set | **0.05** |
| **Adam Weight Decay** | Not set | **3e-2** |
| **Adam Epsilon** | Not set | **1e-10** |
| **VAE Mini Batch** | Not set | **1** |
| **Dataloader Workers** | 4 | **8** |

### Additional Flags Added from train_control_lora.sh

```bash
--training_with_video_token_length
--uniform_sampling
--control_ref_image "random"
--add_inpaint_info
--add_full_ref_image_in_self_attention
--boundary_type "low"
--low_vram
--video_repeat 1
--token_sample_size 640
```

## What Stayed the Same (Your Custom Settings)

✅ **PSI Control Extractor:** All PSI-specific settings preserved
✅ **Multi-GPU Setup:** 2-GPU configuration (NUM_GPUS=2)
✅ **Learning Rate:** 1e-5 (your choice based on paper)
✅ **Gradient Accumulation:** 16 (your choice)
✅ **Max Steps:** 5,000 (your choice)
✅ **Validation:** Your validation videos and prompts

## 🚨 IMPORTANT: Video Settings Impact

### Before (Custom):
- 17 frames × 512×512 = Lower memory, shorter clips
- Good for quick testing

### After (Codebase Default):
- **81 frames × 640×640 = MUCH higher memory!**
- More frames, higher resolution
- **This will use significantly more GPU memory**

## Memory Implications

With the new settings (81 frames, 640 resolution):

| GPU | Per-GPU Batch Size | Gradient Accum | Feasible? |
|-----|-------------------|----------------|-----------|
| 24GB (RTX 3090) | 1 | 16 | ⚠️ Tight fit, may OOM |
| 40GB (A100) | 1 | 16 | ✓ Should work |
| 80GB (A100) | 1-2 | 8-16 | ✓ Comfortable |

**If you get OOM (Out of Memory) errors**, you can:

### Option 1: Reduce Frames (Recommended for Testing)
```bash
export VIDEO_SAMPLE_N_FRAMES=49  # Reduce from 81
export VIDEO_SAMPLE_STRIDE=3     # Increase stride
```

### Option 2: Reduce Resolution
```bash
export VIDEO_SAMPLE_SIZE=512  # Reduce from 640
```

### Option 3: Use More Gradient Accumulation
```bash
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=32  # Increase from 16
# Note: This will slow down training but reduce memory
```

## Current Configuration Summary

```bash
# Multi-GPU
NUM_GPUS=2
GPU_IDS="0,1"

# Training
LEARNING_RATE=1e-5
TRAIN_BATCH_SIZE=1 (per GPU)
GRADIENT_ACCUMULATION_STEPS=16
Effective Batch Size = 1 × 16 × 2 = 32

# LoRA (from train_control_lora.sh)
RANK=64
NETWORK_ALPHA=32
TARGET_LAYERS="q,k,v,ffn.0,ffn.2"

# Video (from train_control_lora.sh)
FRAMES=81
RESOLUTION=640×640
STRIDE=2

# Optimizer (from train_control_lora.sh)
ADAM_WEIGHT_DECAY=3e-2
ADAM_EPSILON=1e-10
MAX_GRAD_NORM=0.05
```

## Comparison with Paper Settings

| Parameter | Paper (Stage 1) | Your Setup (Updated) |
|-----------|-----------------|---------------------|
| Learning Rate | 1×10⁻⁵ | 1×10⁻⁵ ✓ |
| Batch Size | 128 | 32 |
| LoRA Rank | N/A (full model) | 64 (codebase default) |
| Target Layers | N/A | q,k,v,ffn.0,ffn.2 |
| Train Mode | N/A | control_ref |

## Why These Settings?

The `train_control_lora.sh` settings are:
1. **Tested and validated** by the VideoX-Fun team
2. **Optimized for control-based training** (control_ref mode)
3. **Properly configured** for the Wan 2.2 architecture
4. **Include all necessary flags** for stable training

Your previous custom settings were a good starting point, but following the codebase defaults ensures compatibility with the model architecture and training pipeline.

## Testing Recommendation

### Step 1: Test with Smaller Video Settings First
Before running the full training, test with reduced settings:

```bash
# Edit train_control_lora_psi.sh temporarily:
export VIDEO_SAMPLE_N_FRAMES=33  # ~40% of 81
export VIDEO_SAMPLE_SIZE=512     # Lower resolution
export MAX_TRAIN_STEPS=100       # Just test run
```

Run training and watch GPU memory usage:
```bash
watch -n 1 nvidia-smi
```

### Step 2: If It Works, Gradually Increase
```bash
# Try 49 frames (60% of 81)
export VIDEO_SAMPLE_N_FRAMES=49

# Then try 65 frames (80% of 81)
export VIDEO_SAMPLE_N_FRAMES=65

# Finally try full 81 frames
export VIDEO_SAMPLE_N_FRAMES=81
```

### Step 3: Full Training
Once you find the sweet spot for your GPUs, run the full 5,000 steps.

## Quick Reference

**Files Updated:**
- ✅ `train_control_lora_psi.sh` - Updated to match train_control_lora.sh

**New Documentation:**
- 📄 `SETTINGS_COMPARISON.md` (this file)
- 📄 `TRAINING_NOTES.md` (paper comparison)
- 📄 `README.md` (quick start)

**To run training:**
```bash
cd /ccn2/u/wanhee/VideoX-Fun/scripts/wan2.2_fun
./train_control_lora_psi.sh
```

---

**Last Updated:** Dec 16, 2025

