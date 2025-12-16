# Quick Start: Custom Control Training

## What Was Added

✅ Support for custom control feature extraction models  
✅ On-the-fly control embedding generation  
✅ Dummy implementation to get you started  
✅ Complete documentation and examples  

## 3-Step Quick Start

### Step 1: Edit the Control Extractor

Open `custom_control_extractor.py` and implement your model in the `forward()` method:

```python
class CustomControlFeatureExtractor(nn.Module):
    def forward(self, control_pixel_values):
        # INPUT:  (B, F, 3, H, W) - video frames in [-1, 1]
        # OUTPUT: (B, 16, F, H//8, W//8) - control embeddings
        
        # TODO: Replace this with your actual model
        # Example: pose estimation, depth estimation, edge detection, etc.
        
        embeddings = your_model(control_pixel_values)
        return embeddings
```

### Step 2: Test It

```bash
# Test your implementation
python custom_control_extractor.py

# Test integration
python test_custom_control.py
```

### Step 3: Train

```bash
# Use the example script
bash train_control_lora_custom.sh

# Or run directly
accelerate launch train_control_lora.py \
  --pretrained_model_name_or_path /path/to/model \
  --train_data_dir /path/to/data \
  --train_data_meta /path/to/meta.csv \
  --config_path /path/to/config.yaml \
  --use_custom_control_extractor \
  --control_extractor_path /path/to/your_model.pth \
  --output_dir outputs/custom_control \
  --rank 128 \
  --network_alpha 64 \
  --learning_rate 1e-4 \
  --train_batch_size 1 \
  --max_train_steps 10000
```

## For Testing (Before Implementing Your Model)

You can start training immediately with dummy embeddings:

```bash
# Just add the flag, omit the model path
accelerate launch train_control_lora.py \
  --use_custom_control_extractor \
  --pretrained_model_name_or_path /path/to/model \
  --train_data_dir /path/to/data \
  --train_data_meta /path/to/meta.csv \
  --config_path /path/to/config.yaml \
  --output_dir outputs/test_custom_control \
  # ... other args
```

This will use zero/dummy embeddings so you can verify the pipeline works.

## Key Points

| Aspect | Details |
|--------|---------|
| **Input Format** | `(B, F, 3, H, W)` video frames in range `[-1, 1]` |
| **Output Format** | `(B, 16, F, H//8, W//8)` latent embeddings |
| **Model Status** | Frozen during training (no gradients) |
| **When It Runs** | Every training batch, on-the-fly |
| **Memory** | Similar to VAE encoding, supports `--low_vram` |

## What You Need to Implement

In `custom_control_extractor.py`:

1. **`__init__`**: Load your model (pose estimator, depth model, etc.)
2. **`forward`**: Extract features and convert to latent embeddings
3. **`from_pretrained`** (optional): Load your model checkpoint

## Example: Pose Control

```python
class PoseControlExtractor(nn.Module):
    def __init__(self, latent_channels=16, spatial_compression=8, **kwargs):
        super().__init__()
        self.pose_model = load_pose_estimator()  # Your pose model
        self.encoder = build_encoder(latent_channels)  # Encode to latents
        
    def forward(self, control_pixel_values):
        B, F, C, H, W = control_pixel_values.shape
        
        # Extract pose for each frame
        poses = []
        for i in range(F):
            frame = control_pixel_values[:, i]
            pose = self.pose_model(frame)  # (B, keypoints, H, W)
            poses.append(pose)
        
        # Encode poses to latent space
        embeddings = []
        for pose in poses:
            emb = self.encoder(pose)  # (B, 16, H//8, W//8)
            embeddings.append(emb)
        
        # Stack: (B, 16, F, H//8, W//8)
        control_embeddings = torch.stack(embeddings, dim=2)
        return control_embeddings
```

## Files Reference

| File | Purpose |
|------|---------|
| `train_control_lora.py` | Modified training script (main) |
| `custom_control_extractor.py` | **Implement your model here** |
| `train_control_lora_custom.sh` | Example training script |
| `test_custom_control.py` | Test integration |
| `README_CUSTOM_CONTROL.md` | Full documentation |
| `CHANGES_SUMMARY.md` | Technical details of changes |

## Debugging

### Check First Batch
Training saves visualizations in `{output_dir}/sanity_check/`:
- `*.gif` - Training videos
- `*_control.gif` - Control videos
- `*_ref.gif` - Reference frames

### Monitor Training
```bash
# Start tensorboard
tensorboard --logdir outputs/custom_control/logs

# Watch for:
# - train_loss should decrease
# - No NaN or Inf values
# - Validation samples improve over time
```

### Common Fixes

**Out of Memory:**
```bash
--low_vram \
--gradient_checkpointing \
--train_batch_size 1 \
--gradient_accumulation_steps 8
```

**Shape Errors:**
```python
# Output MUST be: (B, 16, F, H//8, W//8)
# If H=512, W=512, F=17:
# Output shape: (B, 16, 17, 64, 64)
```

**Model Not Loading:**
```python
# Check your from_pretrained implementation
# Make sure file exists and format is correct (.pth, .safetensors, etc.)
```

## Next Steps

1. ✅ Read this guide
2. ⬜ Implement your control extractor in `custom_control_extractor.py`
3. ⬜ Run tests: `python custom_control_extractor.py`
4. ⬜ Start training: `bash train_control_lora_custom.sh`
5. ⬜ Monitor progress and iterate

## Need Help?

- 📖 Full docs: `README_CUSTOM_CONTROL.md`
- 🔧 Technical details: `CHANGES_SUMMARY.md`
- 💡 Examples: `custom_control_extractor.py` (has 2 example implementations)

## Summary

You now have a training pipeline that:
- ✅ Uses your custom control model
- ✅ Extracts control on-the-fly
- ✅ Works with all existing features
- ✅ Has dummy implementation to get started

**Just implement the `forward()` method in `custom_control_extractor.py` and you're ready to train!** 🚀

