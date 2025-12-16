# Custom Control Feature Extractor for LoRA Training

This guide explains how to train the VideoX-Fun model with a custom control feature extractor instead of using VAE-encoded control signals.

## Overview

By default, the training script encodes control videos using the VAE encoder. However, you may want to use a custom feature extraction model (e.g., pose estimator, depth estimator, edge detector, etc.) to create control embeddings on-the-fly during training.

This modification allows you to:
- Use any custom model to extract control features from input videos
- Process control signals on-the-fly without precomputation
- Train the model to follow your specific control signal format

## Quick Start

### 1. Implement Your Control Extractor

Edit `custom_control_extractor.py` and replace the dummy implementation with your actual model:

```python
class CustomControlFeatureExtractor(nn.Module):
    def __init__(self, latent_channels=16, temporal_compression=4, spatial_compression=8, **kwargs):
        super().__init__()
        # TODO: Initialize your model here
        # Example: self.pose_estimator = load_pose_model()
        #          self.encoder = YourEncoder()
        
    def forward(self, control_pixel_values):
        # TODO: Implement your feature extraction
        # Input: (B, F, 3, H, W) video frames in [-1, 1]
        # Output: (B, C_latent, F, H//8, W//8) control embeddings
        
        # Example:
        # pose_maps = self.pose_estimator(control_pixel_values)
        # control_embeddings = self.encoder(pose_maps)
        # return control_embeddings
```

### 2. Train with Custom Control

Use the `--use_custom_control_extractor` flag:

```bash
accelerate launch train_control_lora.py \
  --pretrained_model_name_or_path /path/to/model \
  --train_data_dir /path/to/data \
  --train_data_meta /path/to/meta.csv \
  --config_path /path/to/config.yaml \
  --use_custom_control_extractor \
  --control_extractor_path /path/to/your/control_model.pth \
  --output_dir output/custom_control_lora \
  # ... other training arguments ...
```

## Implementation Details

### Input Format

Your control extractor receives `control_pixel_values` with shape `(B, F, C, H, W)`:
- `B`: Batch size
- `F`: Number of frames (e.g., 17)
- `C`: RGB channels (3)
- `H, W`: Height and width (e.g., 512x512)
- Values are normalized to `[-1, 1]`

### Output Format

Your control extractor must output embeddings with shape `(B, C_latent, F, H_latent, W_latent)`:
- `B`: Same batch size
- `C_latent`: Latent channels (typically 16, matches VAE)
- `F`: Same number of frames
- `H_latent`: Height // spatial_compression (typically H // 8)
- `W_latent`: Width // spatial_compression (typically W // 8)

### Example: Pose-Based Control

```python
import torch
import torch.nn as nn

class PoseControlExtractor(nn.Module):
    def __init__(self, latent_channels=16, spatial_compression=8, **kwargs):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_compression = spatial_compression
        
        # Load your pose estimation model
        from your_pose_model import PoseEstimator
        self.pose_estimator = PoseEstimator()
        
        # Encoder to convert pose maps to latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(17, 64, 3, 2, 1),  # 17 keypoint channels
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, latent_channels, 3, 2, 1),
        )
        
    def forward(self, control_pixel_values):
        B, F, C, H, W = control_pixel_values.shape
        
        # Extract pose for each frame
        pose_maps = []
        for i in range(F):
            frame = control_pixel_values[:, i]  # (B, 3, H, W)
            # Convert from [-1, 1] to [0, 1] for pose estimator
            frame_01 = (frame + 1.0) / 2.0
            
            # Extract pose (B, 17, H, W) - 17 keypoint heatmaps
            pose = self.pose_estimator(frame_01)
            pose_maps.append(pose)
        
        # Process each pose map through encoder
        embeddings = []
        for pose in pose_maps:
            emb = self.encoder(pose)  # (B, latent_channels, H//8, W//8)
            embeddings.append(emb)
        
        # Stack: (B, latent_channels, F, H//8, W//8)
        control_embeddings = torch.stack(embeddings, dim=2)
        
        return control_embeddings
```

## Training Arguments

### Required Arguments
- `--use_custom_control_extractor`: Enable custom control extraction
- `--config_path`: Path to model config file
- `--pretrained_model_name_or_path`: Path to pretrained model

### Optional Arguments
- `--control_extractor_path`: Path to your control model checkpoint (if None, uses random initialization)

### Other Important Arguments
- `--train_data_dir`: Directory containing training videos
- `--train_data_meta`: CSV file with video metadata
- `--output_dir`: Where to save checkpoints
- `--rank`: LoRA rank (default: 128)
- `--network_alpha`: LoRA alpha (default: 64)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--checkpointing_steps`: Save checkpoint every N steps (default: 500)
- `--validation_steps`: Run validation every N steps (default: 2000)

## Complete Training Example

```bash
#!/bin/bash

# Set paths
MODEL_PATH="/path/to/VideoX-Fun-pretrained"
DATA_DIR="/path/to/training_videos"
DATA_META="/path/to/metadata.csv"
CONFIG_PATH="${MODEL_PATH}/config.yaml"
CONTROL_MODEL="/path/to/your_control_extractor.pth"
OUTPUT_DIR="outputs/custom_control_training"

# Training settings
RANK=128
ALPHA=64
LR=1e-4
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_STEPS=10000

# Launch training
accelerate launch --config_file accelerate_config.yaml \
  train_control_lora.py \
  --pretrained_model_name_or_path "${MODEL_PATH}" \
  --train_data_dir "${DATA_DIR}" \
  --train_data_meta "${DATA_META}" \
  --config_path "${CONFIG_PATH}" \
  --use_custom_control_extractor \
  --control_extractor_path "${CONTROL_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --rank ${RANK} \
  --network_alpha ${ALPHA} \
  --learning_rate ${LR} \
  --train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --max_train_steps ${MAX_STEPS} \
  --checkpointing_steps 500 \
  --validation_steps 1000 \
  --validation_prompts "A person dancing" "A person walking" \
  --validation_paths "/path/to/val_video1.mp4" "/path/to/val_video2.mp4" \
  --video_sample_size 512 \
  --video_sample_n_frames 17 \
  --enable_bucket \
  --random_hw_adapt \
  --mixed_precision bf16 \
  --seed 42
```

## Debugging Tips

### 1. Test Your Control Extractor

Before training, test your control extractor:

```bash
python custom_control_extractor.py
```

This will run the built-in test to verify input/output shapes.

### 2. Check First Batch

The training script saves visualizations of the first batch in `{output_dir}/sanity_check/`. Check these to ensure:
- Control videos are loaded correctly
- Frames are properly normalized

### 3. Monitor Loss

Watch the training loss. If using custom control:
- Loss might be higher initially as the model learns your control format
- Should decrease steadily over time
- If loss doesn't decrease, check control embedding dimensions and values

### 4. Low VRAM Mode

If you run out of memory, use `--low_vram` flag. This moves models to CPU when not in use:

```bash
--low_vram
```

### 5. Gradient Checkpointing

Enable gradient checkpointing to save memory:

```bash
--gradient_checkpointing
```

## Advanced Usage

### Mixed Training (VAE + Custom Control)

You can switch between VAE and custom control by changing the flag. This allows you to:
1. Start with VAE control (more stable)
2. Fine-tune with custom control

### Conditional Dropout

The training script randomly zeros out control embeddings 10% of the time (classifier-free guidance training). This is controlled in the training loop and helps the model learn to generate without control.

### Multi-GPU Training

Use accelerate for distributed training:

```bash
accelerate config  # Configure multi-GPU setup
accelerate launch train_control_lora.py ...
```

## Common Issues

### Issue: Shape Mismatch Error
**Solution**: Ensure your control extractor outputs the correct shape `(B, 16, F, H//8, W//8)`. The latent channels and spatial dimensions must match the VAE configuration.

### Issue: Out of Memory
**Solutions**:
- Reduce `--train_batch_size`
- Increase `--gradient_accumulation_steps`
- Use `--low_vram` flag
- Enable `--gradient_checkpointing`
- Reduce `--video_sample_n_frames`
- Reduce `--video_sample_size`

### Issue: Control Not Working
**Solutions**:
- Verify control embeddings have reasonable values (not NaN or Inf)
- Check control dropout is working (should be zero 10% of time)
- Ensure control model is in eval mode (it should be frozen)
- Visualize control embeddings to verify they contain information

## File Structure

```
scripts/wan2.2_fun/
├── train_control_lora.py          # Main training script (modified)
├── custom_control_extractor.py    # Your control extractor implementation
├── README_CUSTOM_CONTROL.md       # This file
└── train_control_lora.sh          # Example training script
```

## Next Steps

1. Implement your control extraction model in `custom_control_extractor.py`
2. Test it with `python custom_control_extractor.py`
3. Prepare your training data (videos + metadata CSV)
4. Run training with `--use_custom_control_extractor`
5. Monitor training logs and visualizations
6. Evaluate generated videos with validation

Good luck with your training!

