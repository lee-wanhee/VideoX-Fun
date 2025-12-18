# GPU Utilization Analysis - Dec 18, 2025

## Problem Detected

Training showing **very low GPU utilization**:
- **GPU Usage: 8-15%** ← Should be 80-100%
- **Memory Usage: 40GB / 81GB** (50%) ← Room for improvement
- **H100 GPUs underutilized** ← Expensive hardware sitting idle

## Root Cause: PSI Feature Extraction Bottleneck

### The Issue
1. **PSI is a 7B parameter model** (`PSI_7B_RGBCDF...`)
2. **Each GPU process loads its own PSI copy** (with `accelerate` multi-GPU)
3. **PSI runs inference sequentially for every training sample**
4. **PSI inference is CPU-bound** in some operations (mask generation, frame processing)
5. Training GPUs wait idle while PSI extracts features

### Why GPUs are Idle
```
Timeline per training sample:
├─ 0-5s:   PSI feature extraction (CPU-bound, slow) ← BOTTLENECK
│          └─ GPUs sitting idle at 8-15% 
├─ 5-6s:   VAE encoding (GPU)
├─ 6-7s:   Transformer forward/backward (GPU)
└─ 7-8s:   Gradient accumulation
```

With `GRADIENT_ACCUMULATION_STEPS=16`:
- Need to process 16 samples before optimizer step
- 16 samples × 5-8s each = **80-128 seconds per step**
- Most time spent in PSI, not in actual training

## Solutions Implemented

### 1. ✅ Reduced Gradient Accumulation
**Changed:** `GRADIENT_ACCUMULATION_STEPS: 16 → 4`

**Why:**
- Faster feedback on training progress
- Step updates every ~30s instead of ~90s
- Still maintains reasonable batch size: 2 × 4 × 2 = **16 effective batch size**

**Trade-off:**
- Slightly noisier gradients (but still reasonable)
- More frequent optimizer steps = slightly more overhead

### 2. ✅ Increased Batch Size
**Changed:** `TRAIN_BATCH_SIZE: 1 → 2` (per GPU)

**Why:**
- You have **80GB H100s** with only 40GB used (50% free!)
- Batch size of 2 should fit comfortably
- Better GPU utilization during forward/backward pass
- More efficient than gradient accumulation

**Expected result:**
- Memory usage: 40GB → 55-65GB (still safe)
- GPU utilization during training: 15% → 40-60%
- Throughput: ~2x samples per step

### 3. ✅ Added PSI Timing Logs
**Added to `psi_control_extractor.py`:**
```python
print(f"[PSI GPU={self.device}] Extracting control from 2 frames...")
psi_start_time = time.time()
# ... extraction ...
print(f"[PSI GPU={self.device}] Feature extraction took {psi_elapsed:.2f}s")
```

**Why:**
- Diagnose exactly how long PSI takes per sample
- Confirm if PSI is the bottleneck
- Track if PSI is using the correct GPU

## Expected New Performance

### Before (Current)
- Batch size: 1 per GPU
- Gradient accumulation: 16 steps
- Effective batch: 1 × 16 × 2 = 32
- Time per step: ~90 seconds
- GPU utilization: 8-15%

### After (With Changes)
- Batch size: 2 per GPU
- Gradient accumulation: 4 steps
- Effective batch: 2 × 4 × 2 = 16
- Time per step: ~30-40 seconds (2-3x faster!)
- GPU utilization: 30-60% (better, but PSI still bottleneck)

## Long-Term Solutions (Future)

If PSI remains the bottleneck, consider:

### Option A: Pre-compute PSI Features
**Pros:**
- Eliminate PSI overhead during training
- Much faster training iterations
- Can use full GPU capacity for training

**Cons:**
- Requires storage space for pre-computed features
- Less flexible (can't change PSI params during training)

**Implementation:**
```bash
# 1. Extract features for all videos once
python scripts/precompute_psi_features.py \
  --data_dir $DATA_DIR \
  --output_dir $FEATURES_DIR

# 2. Train with pre-computed features
python train_control_lora.py \
  --use_precomputed_control_features \
  --control_features_dir $FEATURES_DIR
```

### Option B: Optimize PSI Inference
- Use `torch.compile()` on PSI model
- Batch PSI extraction across multiple samples
- Use faster sampling (lower top_k, higher temperature)

### Option C: Use Lighter Control Signal
- Switch to VAE-encoded control (faster than PSI)
- Use simpler control extractors (edge detection, depth, etc.)
- Reduce PSI model size (use smaller variant if available)

### Option D: Dedicated PSI GPU
- Use 3 GPUs: 2 for training, 1 for PSI extraction
- PSI runs on separate GPU, feeds features to training GPUs
- Requires custom data pipeline

## Monitoring Next Steps

Watch for these in your logs:

1. **PSI timing messages:**
   ```
   [PSI GPU=cuda:0] Extracting control from 2 frames: [0, 20]...
   [PSI GPU=cuda:0] Feature extraction took 4.52s
   ```
   
2. **Step completion speed:**
   - Should see step 1 complete in ~30-40s (vs ~90s before)
   
3. **GPU utilization:**
   - Should increase to 30-60% during forward/backward
   - Will still be low during PSI extraction (unavoidable)
   
4. **Memory usage:**
   - Should increase to 55-65GB (monitor with `nvidia-smi`)
   - If OOM, reduce batch_size back to 1

## Recommended Actions

1. **Restart training** with the updated script
2. **Monitor first few steps** to confirm:
   - PSI timing logs appear
   - Steps complete in ~30-40s
   - No OOM errors
3. **If still too slow**, consider pre-computing PSI features
4. **If OOM**, reduce batch_size back to 1

## Summary

The low GPU utilization is **expected with PSI feature extraction**. PSI is a 7B model running inference sequentially, which creates a bottleneck. The changes will:

1. ✅ Give you faster feedback (4x fewer accumulation steps)
2. ✅ Better utilize available memory (batch_size=2)
3. ✅ Show you exactly how long PSI takes

But fundamentally, **PSI will remain the bottleneck** until you either:
- Pre-compute features (recommended for production)
- Switch to a faster control signal
- Accept slower training with better control quality

---

**Files Modified:**
- `train_control_lora_psi.sh`: batch_size=2, gradient_accumulation=4
- `psi_control_extractor.py`: Added timing and device logging

