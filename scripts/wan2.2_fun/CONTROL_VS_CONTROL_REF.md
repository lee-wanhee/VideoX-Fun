# Control vs Control_Ref Training Modes

## Quick Answer

| Mode | What It Does |
|------|--------------|
| **`control`** | Uses only control signals (depth, edges, pose, etc.) for conditioning |
| **`control_ref`** | Uses control signals **+ reference frame** from the video for better consistency |

**TL;DR:** `control_ref` gives the model an actual frame from your video as extra context, which helps maintain visual style, colors, and consistency.

---

## Detailed Explanation

### Mode 1: `control` (Basic Control)

```bash
--train_mode "control"
```

**Input to the model:**
- Text prompt
- Control signal (depth/edge/pose maps)
- Noise

**Process:**
1. Takes control signals (e.g., depth maps extracted by PSI)
2. Encodes them through VAE or custom extractor
3. Uses them to condition the diffusion process
4. Generates video based on control + text

**Use Case:**
- Pure control-based generation
- No style/appearance reference needed
- Model decides all visual details

**Example:**
```
Input: Depth map + "a person walking"
Output: Video matching depth but style varies
```

---

### Mode 2: `control_ref` (Control + Reference Frame)

```bash
--train_mode "control_ref"
--control_ref_image "random"  # or "first_frame"
--add_full_ref_image_in_self_attention  # optional
```

**Input to the model:**
- Text prompt
- Control signal (depth/edge/pose maps)
- **Reference frame** (a frame from the target video)
- Noise

**Process:**
1. Randomly selects a frame from the video (or uses first frame)
2. Encodes this reference frame through VAE
3. Concatenates reference latents with control latents
4. Optionally adds full reference in self-attention layers
5. Model uses both control structure AND reference appearance

**Use Case:**
- Better visual consistency
- Preserve style/colors from reference
- More stable training with appearance anchor

**Example:**
```
Input: Depth map + reference frame showing sunset colors + "a person walking"
Output: Video matching depth AND preserving sunset aesthetic from reference
```

---

## Key Differences in the Code

### Data Loading

**`control` mode:**
```python
# Only loads:
- pixel_values (target video)
- control_pixel_values (control signals)
```

**`control_ref` mode:**
```python
# Loads everything above PLUS:
- ref_pixel_values (reference frame)
- clip_pixel_values (CLIP features of reference)
- clip_idx (which frame was selected)
```

### Reference Frame Selection

```python
if args.control_ref_image == "first_frame":
    clip_index = 0  # Always use first frame
else:  # "random"
    # 40% probability for first frame, 60% distributed among others
    clip_index = random.choice([0, 1, 2, ..., num_frames])
```

**Why random?**
- Helps model learn to use reference from any position
- More robust during inference
- Prevents overfitting to first-frame references

### Model Conditioning

**`control` mode:**
```python
model_input = {
    'latents': noisy_video_latents,
    'y': control_latents,  # Just control signal
}
```

**`control_ref` mode:**
```python
model_input = {
    'latents': noisy_video_latents,
    'y': control_latents,  # Control signal
    'ref_latents_conv_in': ref_latents,  # Reference frame latents!
    'full_ref': full_ref if add_full_ref else None,  # For self-attention
}
```

### Dropout/Conditioning Tricks

Both modes use **conditioning dropout** (10% chance to zero out):
- `control`: Drops control signals with 10% probability
- `control_ref`: Drops BOTH control AND reference with 10% probability each

This teaches the model to work without conditioning (like classifier-free guidance).

---

## Additional Flags in `control_ref`

### 1. `--control_ref_image`

```bash
--control_ref_image "first_frame"  # Always use frame 0
--control_ref_image "random"       # Randomly sample (recommended)
```

**Recommendation:** Use `"random"` for better generalization.

### 2. `--add_full_ref_image_in_self_attention`

```bash
--add_full_ref_image_in_self_attention
```

**What it does:**
- Injects reference frame features into self-attention layers
- Allows model to attend to reference throughout the network
- Better style/detail preservation

**When to use:**
- For stronger reference conditioning
- When appearance consistency is critical

### 3. `--add_inpaint_info`

```bash
--add_inpaint_info
```

**What it does:**
- Adds mask information for inpainting tasks
- Creates masked versions of frames
- Teaches model to fill in missing regions

**When to use:**
- For models that support inpainting
- Training on partially masked videos

---

## Performance Comparison

| Aspect | `control` | `control_ref` |
|--------|-----------|---------------|
| **Training Stability** | Good | **Better** (reference anchor) |
| **Visual Consistency** | Variable | **More consistent** |
| **Color/Style Match** | Random | **Preserved from reference** |
| **Memory Usage** | Lower | Slightly higher (+1 reference frame) |
| **Training Speed** | Faster | ~5% slower (extra VAE encode) |
| **Flexibility at Inference** | High | **Very High** (can provide custom ref) |

---

## Why Use `control_ref`?

### ✅ Advantages

1. **Better Visual Consistency**
   - Reference frame anchors the visual style
   - Colors, lighting, and aesthetics preserved

2. **More Stable Training**
   - Model has two sources of information (control + reference)
   - Easier to learn correct mappings

3. **Better for Real Videos**
   - Real videos have consistent appearance across frames
   - Reference helps model learn this consistency

4. **Flexible at Inference**
   - Can provide different reference frames for different styles
   - Mix and match control + reference for creative control

### ⚠️ Considerations

1. **Slightly More Complex**
   - Need to handle reference frame selection
   - More data processing

2. **Reference Dependency**
   - Model learns to rely on reference
   - May perform worse if reference quality is poor

---

## Your Current Setup

Your `train_control_lora_psi.sh` uses:

```bash
--train_mode "control_ref"  ✓ Better than control
--control_ref_image "random"  ✓ Good for generalization
--add_full_ref_image_in_self_attention  ✓ Stronger conditioning
--add_inpaint_info  ✓ Inpainting support
```

**This is the recommended configuration!** 🎉

It matches the codebase defaults and provides:
- Control signals from PSI extractor
- Random reference frame for consistency
- Full reference in self-attention for detail preservation
- Inpaint support for future flexibility

---

## Visualization

### Control Mode
```
Input Video Frames:  [F0][F1][F2][F3][F4]
                       ↓   ↓   ↓   ↓   ↓
Control Signals:     [C0][C1][C2][C3][C4]  → Model → Generated Video
                       (depth/pose/etc)
```

### Control_Ref Mode
```
Input Video Frames:  [F0][F1][F2][F3][F4]
                       ↓   ↓   ↓   ↓   ↓
Control Signals:     [C0][C1][C2][C3][C4] ─┐
                       (depth/pose/etc)    │
                                           ├→ Model → Generated Video
Reference Frame:     [F2] ─────────────────┘
                     (random selection)      (uses structure + style)
```

---

## Example Use Cases

### Use `control` when:
- Testing pure control conditioning
- Don't care about specific visual style
- Want maximum generation diversity
- Experimenting with new control types

### Use `control_ref` when:
- Training on real videos (your case!)
- Need visual consistency across frames
- Want to preserve specific styles/colors
- Building production models (better quality)

---

## Summary

**Your question: "What's the difference between control and control_ref?"**

**Answer:** `control_ref` = `control` + **reference frame**

The reference frame gives the model:
1. Visual appearance to match
2. Style/color/lighting context
3. Better training stability
4. More consistent outputs

**You made the right choice** using `control_ref` in your training script! It's the recommended mode for VideoX-Fun training with real video data.

---

**Last Updated:** Dec 16, 2025

