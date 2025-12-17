# Quick Reference: Data Flow

## Simple Flow Diagram

```
CSV → Dataset → DataLoader → VAE/PSI → Transformer → Loss
```

## Step-by-Step

### 1. **CSV Metadata** (`handpicked.csv`)
```csv
video_path,control_path,text
/path/to/video.mp4,/path/to/control.mp4,"description"
```

### 2. **Dataset** (`ImageVideoControlDataset`)
```python
# Reads video files
sample = {
    "pixel_values": (81, 3, 512, 512),       # Target video
    "control_pixel_values": (81, 3, 512, 512), # Control video
    "text": "video description"
}
```

### 3. **DataLoader Collate**
```python
# Batches and normalizes
batch = {
    "pixel_values": (1, 81, 3, 512, 512),        # [-1, 1]
    "control_pixel_values": (1, 81, 3, 512, 512), # [-1, 1]
    "text": ["description"]
}
```

### 4. **VAE Encoding** (Target)
```python
latents = vae.encode(pixel_values)
# Shape: (1, 16, 21, 64, 64)
# 81 frames → 21 latent frames (compression 4x)
# 512x512 → 64x64 (compression 8x)
```

### 5. **PSI Control Extraction** ⭐
```python
# Only processes 2 frames!
control_latents = psi_extractor(control_pixel_values)

# Internal process:
#   1. Select frame 0 and frame 20 (0.5s gap)
#   2. Extract PSI features for these 2 frames
#   3. Assign F0 → frame 0, F20 → all other frames
#   4. Project to match VAE dimensions

# Output shape: (1, 16, 21, 64, 64)
```

### 6. **Concatenate** (for control_ref mode)
```python
control_latents = torch.cat([
    control_latents,    # (1, 16, 21, 64, 64) - PSI features
    ref_latents_conv_in # (1, 16, 21, 64, 64) - First frame
], dim=1)
# Output: (1, 32, 21, 64, 64)
```

### 7. **Transformer**
```python
noise_pred = transformer3d(
    x=noisy_latents,       # (1, 16, 21, 64, 64) - Noisy target
    context=prompt_embeds, # (1, seq_len, 4096) - Text embedding
    t=timesteps,           # Noise level
    y=control_latents,     # (1, 32, 21, 64, 64) ← PSI CONTROL!
)
# Output: (1, 16, 21, 64, 64) - Predicted noise
```

### 8. **Loss & Training**
```python
loss = MSE(noise_pred, target_noise)
loss.backward()  # Update LoRA parameters
```

---

## Key Insight: 2-Frame PSI Extraction

**Your videos have 81 frames, but PSI only processes 2 frames!**

```
Video:     [F0] [F1] [F2] ... [F20] [F21] ... [F80]
                                ↑                
PSI Input: [F0] ─────────────► [F20]
           Extract              Extract
           features             features
                ↓                   ↓
PSI Output:[F0] [F20][F20]...[F20][F20]...[F20]
           Used Reused for all remaining frames
           once
```

**Why?**
- Efficient: 2 frames instead of 81
- Temporal info: 0.5s gap captures motion
- Reuse: F20 features work for all middle/end frames

**Configuration:**
```bash
export PSI_TIME_GAP_SEC=0.5  # Time between F0 and F20
```

---

## Code Locations

| Component | File | Function/Class | Lines |
|-----------|------|----------------|-------|
| Dataset | `videox_fun/data/dataset_image_video.py` | `ImageVideoControlDataset` | 263-596 |
| DataLoader | `train_control_lora.py` | `collate_fn` | 1350-1662 |
| PSI Init | `train_control_lora.py` | `main()` | 1059-1091 |
| PSI Extract | `train_control_lora.py` | training loop | 2060-2083 |
| PSI Class | `psi_control_extractor.py` | `PSIControlFeatureExtractor` | 31-398 |
| PSI Forward | `psi_control_extractor.py` | `forward()` | 146-376 |
| Transformer | `train_control_lora.py` | training loop | 2258-2266 |

---

## Training Script Arguments

From `train_control_lora_psi.sh`:

```bash
# Core settings
--train_mode "control_ref"              # Use control + reference
--control_ref_image "first_frame"       # Reference is first frame
--add_full_ref_image_in_self_attention # Include ref in attention

# PSI settings  
--use_psi_control_extractor            # Use PSI instead of VAE
--psi_model_name "PSI_7B_..."          # PSI model checkpoint
--psi_quantizer_name "PLPQ-..."        # Quantizer checkpoint
--psi_time_gap_sec 0.5                 # 0.5s between control frames
--psi_mask_ratio 0.0                   # No masking (fully visible)

# Video settings
--video_sample_n_frames 81             # 81 frames per video
--video_sample_size 512                # 512x512 resolution
--video_sample_stride 2                # Sample every 2nd frame
```

---

## Data Shape Reference

| Stage | Tensor | Shape | Channels | Frames | Spatial |
|-------|--------|-------|----------|--------|---------|
| Raw Video | - | - | RGB | 81 | 512×512 |
| Dataset Output | `pixel_values` | (81, 3, 512, 512) | RGB | 81 | 512×512 |
| Batch | `pixel_values` | (B, 81, 3, 512, 512) | RGB | 81 | 512×512 |
| VAE Latent | `latents` | (B, 16, 21, 64, 64) | Latent | 21 | 64×64 |
| PSI Latent | `control_latents` | (B, 16, 21, 64, 64) | Latent | 21 | 64×64 |
| Combined | `control_latents` | (B, 32, 21, 64, 64) | Latent | 21 | 64×64 |

**Compression Ratios:**
- **Temporal:** 81 frames → 21 latent frames (4x compression)
- **Spatial:** 512×512 → 64×64 (8x compression)
- **Channels:** 3 (RGB) → 16 (latent)

---

## Common Questions

**Q: Where does PSI get called?**
```python
# train_control_lora.py, line 2062
control_latents = control_extractor(control_pixel_values)
```

**Q: What happens in control_extractor()?**
```python
# psi_control_extractor.py, line 146
def forward(self, control_pixel_values, fps=None):
    # 1. Convert [-1,1] → [0,1]
    # 2. Select 2 frames (0 and ~20)
    # 3. Call PSIPredictor
    # 4. Assign features to all frames
    # 5. Project to latent dimensions
    return control_embeddings
```

**Q: How is control passed to the model?**
```python
# train_control_lora.py, line 2263
noise_pred = transformer3d(
    x=noisy_latents,
    y=control_latents,  # ← HERE!
    ...
)
```

**Q: Can I visualize the control?**
Yes! The control latents are in the same space as VAE latents, so you can decode them:
```python
control_decoded = vae.decode(control_latents / vae.config.scaling_factor)
```

