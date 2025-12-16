# Training Notes and Reference

## Reference: Teacher Model Training (from Paper)

### Overview
Teacher models are initialized from partial weights of VideoXFun's Wan variants (AIGC-Apps & Alibaba PAI Team, 2024), which extend Wan I2V models with additional control channels. This initialization accelerates convergence compared to training from scratch.

### Two-Stage Training Process

Both Wan 2.1 and Wan 2.2 undergo two-stage training:

#### Stage 1: Initial Training
- **Dataset:** Filtered OpenVid-1M (0.6M videos)
- **Steps:** 4,800 steps
- **Batch Size:** 128
- **Learning Rate:** 1×10⁻⁵

#### Stage 2: Fine-tuning
- **Dataset:** Cleaner synthetic data
- **Steps:** 
  - Wan 2.1: 800 steps (~1 epoch)
  - Wan 2.2: 400 steps (~1 epoch)
- **Batch Size:** 128
- **Learning Rate:** 1×10⁻⁶

### Training Details

**Track Sampling:**
- Randomly sample 1,000-2,500 tracks per training sample
- Sinusoidal positional embedding: d = 64 dimensions
- Stochastic track masking applied during fine-tuning stage (Stage 2)

**Model Architecture:**
- Track head remains frozen after initial training (Stage 1)
- Track head already operates chunk-wise

---

## Your Current Training Configuration

### Hardware Setup
- **GPUs:** 2 (configurable via `NUM_GPUS`)
- **GPU IDs:** 0,1
- **Nodes:** 1

### Training Hyperparameters
- **Learning Rate:** 1×10⁻⁵ (Stage 1 equivalent)
- **Batch Size (per GPU):** 1
- **Gradient Accumulation:** 16
- **Number of GPUs:** 2
- **Effective Batch Size:** 1 × 16 × 2 = **32**
- **Max Steps:** 5,000
- **Validation Steps:** 1,000

### LoRA Settings
- **Rank:** 128
- **Network Alpha:** 64
- **Target Layers:** attn1, attn2

### Video Settings
- **Resolution:** 512×512
- **Frames:** 17
- **Stride:** 4

### PSI Control Settings
- **Mask Ratio:** 0.0 (fully visible)
- **Temperature:** 1.0
- **Top-p:** 0.9
- **Top-k:** 1000

### Optimization
- **Mixed Precision:** bf16
- **Gradient Checkpointing:** Enabled
- **8-bit Adam:** Enabled

---

## Comparison: Paper vs Your Setup

| Parameter | Paper (Stage 1) | Paper (Stage 2) | Your Setup |
|-----------|-----------------|-----------------|------------|
| Learning Rate | 1×10⁻⁵ | 1×10⁻⁶ | **1×10⁻⁵** |
| Batch Size | 128 | 128 | **32** (effective) |
| Steps | 4,800 | 400-800 | **5,000** |
| Dataset | OpenVid-1M (0.6M) | Synthetic | BVD2 handpicked |
| Training Type | Full model | Full model | **LoRA fine-tuning** |

### Notes on Differences

1. **Batch Size:** Your effective batch size (32) is smaller than the paper (128)
   - This is reasonable for LoRA training with limited data
   - Larger gradient accumulation (16) helps stabilize training with small per-GPU batch size

2. **Learning Rate:** You're using 1×10⁻⁵ (Stage 1 rate)
   - Appropriate for initial LoRA training
   - Consider reducing to 1×10⁻⁶ if you do a second fine-tuning stage

3. **Training Method:** You're using LoRA instead of full fine-tuning
   - More parameter efficient
   - Faster training
   - Lower memory requirements
   - Rank 128 is a good balance between capacity and efficiency

4. **Steps:** 5,000 steps is reasonable
   - More than Stage 2 (400-800), less than Stage 1 (4,800)
   - Good for targeted fine-tuning on specific data

---

## Recommended Training Strategy

### Option 1: Single-Stage Training (Current)
```bash
# Your current setup - good for quick iteration
Learning Rate: 1×10⁻⁵
Steps: 5,000
Batch Size: 32 (effective)
```

### Option 2: Two-Stage Training (Following Paper)
```bash
# Stage 1: Initial LoRA training
export LEARNING_RATE=1e-5
export MAX_TRAIN_STEPS=4000
export GRADIENT_ACCUMULATION_STEPS=16  # Effective batch size: 32

# Stage 2: Fine-tuning (run after Stage 1 completes)
export LEARNING_RATE=1e-6
export MAX_TRAIN_STEPS=1000
export GRADIENT_ACCUMULATION_STEPS=16
# Add: --resume_from_checkpoint outputs/psi_control_lora_<timestamp>/checkpoint-4000
```

### Option 3: Longer Training with Smaller Batch
```bash
# Good for limited GPU memory or smaller datasets
Learning Rate: 5×10⁻⁶ (between Stage 1 and 2)
Steps: 8,000
Batch Size: 16 (effective) - reduce gradient accumulation to 8
```

---

## Training Tips Based on Paper

1. **Track Masking:** The paper applies stochastic track masking in Stage 2
   - Your PSI mask ratio is 0.0 (no masking)
   - Consider increasing `PSI_MASK_RATIO` to 0.1-0.3 for regularization

2. **Freezing Layers:** Paper freezes track head after Stage 1
   - Your LoRA only trains attention layers (attn1, attn2)
   - Consider freezing additional layers if needed

3. **Batch Size Scaling:** If you can increase effective batch size:
   ```bash
   # For 2×40GB GPUs:
   TRAIN_BATCH_SIZE=2
   GRADIENT_ACCUMULATION_STEPS=8
   # Effective: 2 × 8 × 2 = 32 ✓ (same as current)
   
   # For 2×80GB GPUs:
   TRAIN_BATCH_SIZE=2
   GRADIENT_ACCUMULATION_STEPS=16
   # Effective: 2 × 16 × 2 = 64
   ```

4. **Learning Rate Scheduling:** Paper uses constant LR, but you have warmup
   ```bash
   --lr_scheduler "constant_with_warmup"
   --lr_warmup_steps 500
   ```
   This is good! Helps with training stability.

---

## Monitoring Training Progress

### Key Metrics to Watch

1. **Loss Curve:** Should decrease steadily
   - Check tensorboard: `tensorboard --logdir outputs/psi_control_lora_*/logs`
   
2. **Validation Outputs:** Every 1,000 steps
   - Compare to reference videos
   - Look for control signal adherence
   
3. **Gradient Norms:** Should be stable, not exploding
   
4. **PSI Feature Statistics:** Check that PSI extraction is working
   - Similar ranges to parallel_feature_test.py output

### Expected Training Time

- **2×A100 GPUs:** ~10-15 hours for 5,000 steps
- **2×A6000 GPUs:** ~15-20 hours for 5,000 steps
- **2×RTX3090 GPUs:** ~20-30 hours for 5,000 steps

PSI feature extraction adds ~20-30% overhead compared to VAE encoding.

---

## Quick Reference: Adjusting for Different Scenarios

### If Training is Too Slow
```bash
MAX_TRAIN_STEPS=3000  # Reduce steps
VALIDATION_STEPS=1500  # Less frequent validation
```

### If Running Out of Memory
```bash
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32  # Keep effective batch size
VIDEO_SAMPLE_N_FRAMES=13  # Reduce from 17
# Add: --vae_mini_batch 8
# Add: --low_vram
```

### If Loss is Unstable
```bash
LEARNING_RATE=5e-6  # Lower learning rate
GRADIENT_ACCUMULATION_STEPS=32  # More smoothing
# Or add gradient clipping in train_control_lora.py
```

### If Quality is Poor After Training
```bash
# Try Stage 2 fine-tuning:
LEARNING_RATE=1e-6
MAX_TRAIN_STEPS=1000
# Resume from best checkpoint
```

---

## File Locations

- **Training Script:** `train_control_lora_psi.sh`
- **Training Code:** `train_control_lora.py`
- **PSI Extractor:** `psi_control_extractor.py`
- **Test Script:** `parallel_feature_test.py`
- **Output Directory:** `outputs/psi_control_lora_<timestamp>/`

## Related Documentation

- `README.md` - Quick start guide
- `create_csv_from_videos.py` - Dataset preparation
- Paper reference: VideoXFun Wan variants (AIGC-Apps & Alibaba PAI Team, 2024)

---

**Last Updated:** Dec 16, 2025

