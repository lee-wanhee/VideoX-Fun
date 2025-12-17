# Code Flow: Dataloader → PSI Control → Model

This document explains the complete data flow from dataloader to model when training with `train_control_lora_psi.sh`.

## Overview

```
CSV Metadata → Dataset → DataLoader → Collate → VAE/PSI → Transformer
```

---

## 1. Data Loading (Dataset)

### Dataset Class: `ImageVideoControlDataset`
**Location:** `videox_fun/data/dataset_image_video.py`

```python
train_dataset = ImageVideoControlDataset(
    args.train_data_meta,  # CSV file path
    args.train_data_dir,   # Video directory
    video_sample_size=512,
    video_sample_n_frames=81,
    video_sample_stride=2,
    enable_bucket=True,
    enable_camera_info=False  # For control_ref mode
)
```

### What the Dataset Returns (`__getitem__`):

```python
sample = {
    "pixel_values": tensor,         # Target video frames (F, C, H, W)
    "control_pixel_values": tensor, # Control video frames (F, C, H, W)
    "subject_image": tensor,        # Reference image (for control_ref mode)
    "text": str,                    # Text prompt
    "data_type": str,              # "video" or "image"
    "idx": int                     # Sample index
}
```

**Frame Loading Process:**
1. Reads video path from CSV metadata
2. Uses `VideoReader` to extract frames
3. Samples frames with stride (e.g., every 2nd frame)
4. Resizes frames to target size
5. Loads control video from separate path (specified in CSV)
6. Returns both target and control frames

---

## 2. DataLoader & Collate Function

### DataLoader Creation
**Location:** `train_control_lora.py` lines 1324-1331, 1677-1683

```python
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler=batch_sampler,  # Aspect ratio batching
    collate_fn=collate_fn,        # Custom collation
    num_workers=8,
    persistent_workers=True
)
```

### Collate Function (`collate_fn`)
**Location:** `train_control_lora.py` lines 1350-1662

**Purpose:** Batch multiple samples and apply transformations

**Key Steps:**

```python
def collate_fn(examples):
    # 1. Extract first sample to determine if image or video
    pixel_value = examples[0]["pixel_values"]  # (F, H, W, C) numpy
    data_type = examples[0]["data_type"]
    
    # 2. Calculate aspect ratio and target size
    closest_size, closest_ratio = get_closest_ratio(h, w, ratios)
    
    # 3. Process each sample
    for example in examples:
        # Convert to tensor and normalize to [-1, 1]
        pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2)
        pixel_values = pixel_values / 255.0  # [0, 1]
        
        # Apply transforms (resize, center crop)
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(closest_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])  # → [-1, 1]
        ])
        pixel_values = transform(pixel_values)
        
        # Same for control frames
        control_pixel_values = transform(control_pixel_values)
        
        new_examples["pixel_values"].append(pixel_values)
        new_examples["control_pixel_values"].append(control_pixel_values)
        new_examples["text"].append(example["text"])
    
    # 4. Stack to create batch
    # Limit frames to batch_video_length (e.g., 81 frames)
    new_examples["pixel_values"] = torch.stack([
        example[:batch_video_length] for example in new_examples["pixel_values"]
    ])  # Shape: (B, F, C, H, W)
    
    new_examples["control_pixel_values"] = torch.stack([
        example[:batch_video_length] for example in new_examples["control_pixel_values"]
    ])  # Shape: (B, F, C, H, W)
    
    return new_examples
```

**Output Shapes:**
```python
batch = {
    "pixel_values": (B, F, C, H, W),         # e.g., (1, 81, 3, 512, 512)
    "control_pixel_values": (B, F, C, H, W), # e.g., (1, 81, 3, 512, 512)
    "text": [str, str, ...]                  # List of B prompts
}
```
where values are in **[-1, 1]** range.

---

## 3. Training Loop - Processing Control Signals

### Location: `train_control_lora.py` lines 1900-2098

```python
for step, batch in enumerate(train_dataloader):
    # ============= Step 1: Extract from batch =============
    pixel_values = batch["pixel_values"].to(weight_dtype)
    # Shape: (B, F, C, H, W) in [-1, 1]
    
    control_pixel_values = batch["control_pixel_values"].to(weight_dtype)
    # Shape: (B, F, C, H, W) in [-1, 1]
    
    # ============= Step 2: Encode with VAE =============
    with torch.no_grad():
        # Target latents (for ground truth)
        latents = _batch_encode_vae(pixel_values)
        # Shape: (B, C_latent, F_latent, H_latent, W_latent)
        # e.g., (1, 16, 21, 64, 64) for 81 frames at 512x512
        latents = latents * vae.config.scaling_factor
        
        # ============= Step 3: PSI Control Feature Extraction =============
        if args.use_custom_control_extractor:
            # Use PSI control extractor
            control_latents = control_extractor(control_pixel_values)
            # Shape: (B, C_latent, F_latent, H_latent, W_latent)
            
            # Apply dropout (90% chance to keep, 10% to zero)
            for bs_index in range(control_latents.size()[0]):
                zero_init = np.random.choice([0, 1], p=[0.90, 0.10])
                if zero_init:
                    control_latents[bs_index] = 0
        else:
            # Original: Use VAE encoding
            control_latents = _batch_encode_vae(control_pixel_values)
        
        # ============= Step 4: Prepare for Transformer =============
        # For control_ref mode, concatenate with ref latents
        if args.train_mode == "control_ref":
            ref_latents = _batch_encode_vae(ref_pixel_values)
            ref_latents_conv_in = torch.zeros_like(latents)
            ref_latents_conv_in[:, :, :1] = ref_latents  # First frame
            
            # Concatenate: [control, ref]
            control_latents = torch.cat([control_latents, ref_latents_conv_in], dim=1)
            # Shape: (B, C_latent*2, F_latent, H_latent, W_latent)
        
        # For inpaint mode, also concatenate inpaint info
        if args.add_inpaint_info:
            inpaint_latents = torch.cat([mask, mask_latents], dim=1)
            control_latents = torch.cat([control_latents, inpaint_latents], dim=1)
```

---

## 4. PSI Control Feature Extractor

### Class: `PSIControlFeatureExtractor`
**Location:** `scripts/wan2.2_fun/psi_control_extractor.py`

### Initialization (`__init__`)
**Location:** lines 44-110

```python
control_extractor = PSIControlFeatureExtractor(
    model_name="PSI_7B_RGBCDF_bvd_4frame_Unified_Vocab_Balanced_Task_V2_continue_ctx_8192/model_01400000.pt",
    quantizer_name="PLPQ-ImageNetOpenImages-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2/model_best.pt",
    flow_quantizer_name="HLQ-flow-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l2-coarsel21e-2-fg_v1_5/model_best.pt",
    depth_quantizer_name="HLQ-depth-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2-200k_ft500k_3/model_best.pt",
    latent_channels=16,              # VAE latent channels
    temporal_compression=4,          # VAE temporal compression ratio
    spatial_compression=8,           # VAE spatial compression ratio
    time_gap_sec=0.5,               # Time between 2 frames
    mask_ratio=0.0,                 # No masking
    temperature=1.0,
    top_p=0.9,
    top_k=1000,
)
```

### Forward Pass (`forward`)
**Location:** lines 146-376

```python
def forward(self, control_pixel_values, fps=None):
    """
    Args:
        control_pixel_values: (B, F, C, H, W) in [-1, 1]
        
    Returns:
        control_embeddings: (B, C_latent, F, H_latent, W_latent)
    """
    B, F, C, H, W = control_pixel_values.shape
    
    # ========== Step 1: Convert to [0, 1] ==========
    rgb_frames_01 = (control_pixel_values + 1.0) / 2.0
    
    # ========== Step 2: Select 2 frames ==========
    # Estimate FPS (assumes ~2 second videos)
    estimated_duration = 2.0
    estimated_fps = F / estimated_duration
    
    # Calculate frame indices
    frame_0_idx = 0
    frame_1_idx = int(estimated_fps * time_gap_sec)  # e.g., frame 20
    frame_1_idx = min(frame_1_idx, F - 1)
    
    # Extract 2 frames for PSI
    rgb_frame_0 = rgb_frames_01[b, frame_0_idx]  # (C, H, W)
    rgb_frame_1 = rgb_frames_01[b, frame_1_idx]  # (C, H, W)
    
    # ========== Step 3: PSI Feature Extraction ==========
    outputs = predictor.parallel_extract_features(
        prompt="rgb0,rgb1->rgb1",
        rgb_frames=[rgb_frame_0, rgb_frame_1],  # Only 2 frames!
        time_codes=[0, int(time_gap_sec * 1000)],  # [0ms, 500ms]
        mask_ratio_cond=0.0,      # Condition frame fully visible
        mask_ratio_pred=0.9,      # Prediction frame 90% masked
        temperature=1.0,
        top_p=0.9,
        top_k=1000,
        seed=42
    )
    
    # Extract features for both frames
    features_0 = outputs["rgb"]["F0_pred"]  # Features for frame 0
    features_1 = outputs["rgb"]["F20_pred"] # Features for frame 20
    
    # ========== Step 4: Assign Features to All Frames ==========
    for frame_idx in range(F):
        if frame_idx == 0:
            # Frame 0: Use F0 features
            embeddings[:, :, frame_idx] = features_0
        else:
            # All other frames: Use F20 features
            embeddings[:, :, frame_idx] = features_1
    
    # ========== Step 5: Projection (if needed) ==========
    if self.projection is None:
        # First time: Create projection layer
        # PSI output might have different channels than VAE
        C_psi = embeddings.shape[1]
        if C_psi != latent_channels:
            self.projection = nn.Conv2d(C_psi, latent_channels, 1)
    
    if self.projection is not None:
        # Project to match VAE latent dimensions
        B, C, F, H, W = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3, 4)  # (B, F, C, H, W)
        embeddings = embeddings.reshape(B*F, C, H, W)
        embeddings = self.projection(embeddings)
        embeddings = embeddings.reshape(B, F, latent_channels, H, W)
        embeddings = embeddings.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
    
    return embeddings  # (B, C_latent, F, H_latent, W_latent)
```

### Key Insight: 2-Frame Extraction

**Only 2 frames are passed to PSI:**
- Frame 0 (first frame)
- Frame N (at `time_gap_sec` seconds later, e.g., frame 20 for 0.5s gap)

**Feature assignment:**
```
Video Frames:  [0]  [1]  [2]  [3] ... [20] [21] ... [80]
PSI Features:  [F0] [F20][F20][F20]...[F20][F20]...[F20]
                ↑                       ↑
             Extract               Extract
```

This is much more efficient than processing all 81 frames!

---

## 5. Transformer Forward Pass

### Location: `train_control_lora.py` lines 2256-2266

```python
# ========== Prepare inputs ==========
# Add noise to latents
noise = torch.randn_like(latents)
timesteps = torch.randint(0, 1000, (B,))
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

# Encode text prompts
prompt_embeds = text_encoder(prompt_ids)  # (B, SeqLen, D_text)

# Calculate sequence length for transformer
target_shape = (C_latent, F_latent, H_latent, W_latent)
seq_len = math.ceil(
    (H_latent * W_latent) / (patch_size_h * patch_size_w) * F_latent
)

# ========== Transformer forward ==========
noise_pred = transformer3d(
    x=noisy_latents,          # Noisy target latents (B, C, F, H, W)
    context=prompt_embeds,    # Text embeddings (B, SeqLen, D)
    t=timesteps,              # Timesteps (B,) or (B, SeqLen)
    seq_len=seq_len,          # Sequence length
    y=control_latents,        # ← CONTROL SIGNAL from PSI! (B, C*2, F, H, W)
    y_camera=None,            # Camera control (optional)
    full_ref=full_ref,        # Full reference image (optional)
)
# Output: (B, C, F, H, W) - predicted noise

# ========== Loss calculation ==========
loss = F.mse_loss(noise_pred, noise)
```

### Control Signal Flow Inside Transformer

The `y=control_latents` parameter is processed by the transformer:

1. **Control Encoder** (inside transformer):
   - Takes `y` (control_latents) as input
   - Projects to transformer dimensions
   - Processes through control-specific layers

2. **Cross-Attention**:
   - Queries: From noisy latents `x`
   - Keys/Values: From control signal `y`
   - Allows the model to attend to control information

3. **Addition/Concatenation**:
   - Control features are combined with main latent features
   - Influences the denoising process

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. CSV METADATA                                                     │
│    ┌──────────────────────────────────────────┐                    │
│    │ video_path, control_path, text, ...      │                    │
│    └──────────────────┬───────────────────────┘                    │
└───────────────────────┼──────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. DATASET (ImageVideoControlDataset)                              │
│    ┌──────────────────────────────────────────┐                    │
│    │ - Load video frames with VideoReader     │                    │
│    │ - Sample frames (stride=2, n_frames=81)  │                    │
│    │ - Load control video                     │                    │
│    │ - Returns: pixel_values (F,C,H,W)        │                    │
│    │           control_pixel_values (F,C,H,W) │                    │
│    └──────────────────┬───────────────────────┘                    │
└───────────────────────┼──────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. DATALOADER + COLLATE                                             │
│    ┌──────────────────────────────────────────┐                    │
│    │ - Batch samples                          │                    │
│    │ - Resize/Crop to closest aspect ratio    │                    │
│    │ - Normalize to [-1, 1]                   │                    │
│    │ - Stack: (B, F, C, H, W)                 │                    │
│    │   e.g., (1, 81, 3, 512, 512)             │                    │
│    └──────────────────┬───────────────────────┘                    │
└───────────────────────┼──────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. TRAINING LOOP                                                    │
│                                                                     │
│    ┌────────────────┐         ┌──────────────────┐                │
│    │ pixel_values   │         │control_pixel_vals│                │
│    │ (B,F,C,H,W)    │         │ (B,F,C,H,W)      │                │
│    │ [-1,1]         │         │ [-1,1]           │                │
│    └───────┬────────┘         └────────┬─────────┘                │
│            ▼                           ▼                           │
│    ┌───────────────┐          ┌────────────────────────┐          │
│    │  VAE Encode   │          │ PSI Control Extractor  │          │
│    │               │          │                        │          │
│    │ Output:       │          │ 1. Convert to [0,1]    │          │
│    │ latents       │          │ 2. Select 2 frames:    │          │
│    │ (B,C,F,H,W)   │          │    - Frame 0           │          │
│    │               │          │    - Frame 20 (0.5s)   │          │
│    └───────┬───────┘          │ 3. PSI Feature Extract │          │
│            │                  │    parallel_extract... │          │
│            │                  │ 4. Assign to all:      │          │
│            │                  │    F0→frame0, F20→rest │          │
│            │                  │ 5. Project to C_latent │          │
│            │                  │                        │          │
│            │                  │ Output: control_latents│          │
│            │                  │ (B,C,F,H,W)            │          │
│            │                  └────────┬───────────────┘          │
│            │                           │                           │
│            ▼                           ▼                           │
│    ┌──────────────────────────────────────────────┐               │
│    │ Concatenate with ref_latents (control_ref)   │               │
│    │ control_latents = cat([control, ref], dim=1) │               │
│    │ Shape: (B, C*2, F, H, W)                     │               │
│    └──────────────────┬───────────────────────────┘               │
│                       │                                            │
│                       │    ┌──────────────┐                       │
│                       │    │ Text Encoder │                       │
│                       │    └──────┬───────┘                       │
│                       │           │ prompt_embeds                 │
│                       ▼           ▼                                │
│    ┌─────────────────────────────────────────────┐               │
│    │  TRANSFORMER3D                              │               │
│    │                                             │               │
│    │  transformer3d(                             │               │
│    │    x = noisy_latents,      ← Target latents│               │
│    │    context = prompt_embeds, ← Text cond    │               │
│    │    t = timesteps,           ← Noise level  │               │
│    │    seq_len = ...,                           │               │
│    │    y = control_latents, ← PSI CONTROL! ◄───┼─── KEY INPUT  │
│    │    y_camera = None,                         │               │
│    │    full_ref = full_ref                      │               │
│    │  )                                          │               │
│    │                                             │               │
│    │  → noise_pred (B,C,F,H,W)                  │               │
│    └─────────────────┬───────────────────────────┘               │
│                      ▼                                            │
│    ┌─────────────────────────────────────┐                       │
│    │  Loss = MSE(noise_pred, target)     │                       │
│    │  Backward + Optimize LoRA params    │                       │
│    └─────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary

### Key Data Shapes

| Stage | Variable | Shape | Range |
|-------|----------|-------|-------|
| Dataset | pixel_values | (F, C, H, W) | [0, 255] |
| Dataset | control_pixel_values | (F, C, H, W) | [0, 255] |
| Collate | pixel_values | (B, F, C, H, W) | [-1, 1] |
| Collate | control_pixel_values | (B, F, C, H, W) | [-1, 1] |
| VAE | latents | (B, 16, F/4, H/8, W/8) | scaled |
| PSI | control_latents | (B, 16, F/4, H/8, W/8) | scaled |
| Concat | control_latents | (B, 32, F/4, H/8, W/8) | scaled |
| Transformer | noise_pred | (B, 16, F/4, H/8, W/8) | predicted |

### Key Settings from `train_control_lora_psi.sh`

- **Input frames:** 81 frames
- **PSI extraction:** Only 2 frames (0, 20)
- **Time gap:** 0.5 seconds between control frames
- **Resolution:** 512×512
- **Train mode:** `control_ref`
- **Control type:** PSI features (not VAE encoded)
- **Dropout:** 10% chance to zero control signal

### Critical Code Locations

1. **Dataset loading:** `videox_fun/data/dataset_image_video.py:263-596`
2. **Collate function:** `train_control_lora.py:1350-1662`
3. **PSI extractor init:** `train_control_lora.py:1059-1091`
4. **Control extraction:** `train_control_lora.py:2060-2083`
5. **PSI forward:** `psi_control_extractor.py:146-376`
6. **Transformer forward:** `train_control_lora.py:2258-2266`

---

## FAQ

**Q: Why only 2 frames for PSI?**
A: PSI is computationally expensive. Using only 2 frames (first and one at 0.5s) captures temporal information while being efficient. Features are reused across all frames.

**Q: What does the control signal do?**
A: It guides the video generation by providing semantic control. The transformer uses cross-attention to incorporate control features into the denoising process.

**Q: Why concatenate with ref_latents?**
A: In `control_ref` mode, the reference image (first frame) provides additional guidance for maintaining consistency.

**Q: What's the difference from VAE encoding?**
A: PSI extracts high-level semantic features (objects, motion), while VAE encodes low-level visual features. PSI provides richer control signals.

**Q: Can I change time_gap_sec?**
A: Yes! Adjust `PSI_TIME_GAP_SEC` in the training script. Larger gaps capture longer-term motion.

