"""
Simple test script for parallel_extract_features function.

This script:
1. Loads a PSIPredictor model
2. Loads two frames from a video (or creates synthetic frames)
3. Creates masks: first frame fully visible (0% masked), second frame 90% masked
4. Calls parallel_extract_features
5. Saves visualization of outputs

Usage:
    python inv/parallel_feature/parallel_feature_test.py

Configuration:
    - Edit the model_name, quantizer_name paths in main() to match your model locations
    - Set video_path to load frames from a real video, or leave as None for synthetic frames
    - Adjust mask_ratio, temperature, seed, etc. as needed for testing

This is designed to test and debug the parallel_extract_features implementation.
"""

import os
import sys
import torch
import numpy as np
import random
from PIL import Image
from pathlib import Path

# Add parent directory to path to import ccwm modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ccwm.predictor.psi_predictor import PSIPredictor
from ccwm.utils.image_processing import video_to_frames, load_image


def create_mask_indices(mask_ratio: float, seed: int = 0) -> list:
    """
    Create unmask indices for a 32x32 patch grid.
    
    Args:
        mask_ratio: Ratio of patches to mask (0.0 = all visible, 1.0 = all masked)
        seed: Random seed
        
    Returns:
        List of patch indices to unmask (keep visible)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Total patches in 32x32 grid
    total_patches = 1024
    
    # Number of patches to keep visible (unmask)
    n_unmask = int(total_patches * (1 - mask_ratio))
    
    # Randomly select patches to unmask
    unmask_indices = np.random.choice(total_patches, n_unmask, replace=False).tolist()
    random.shuffle(unmask_indices)
    
    print(f"Mask ratio {mask_ratio}: Unmasking {len(unmask_indices)}/1024 patches")
    
    return unmask_indices


def main():
    # Configuration - using same paths as Gradio demo
    model_name = "PSI_7B_RGBCDF_bvd_4frame_Unified_Vocab_Balanced_Task_V2_continue_ctx_8192/model_01500000.pt"
    quantizer_name = "PLPQ-ImageNetOpenImages-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2/model_best.pt"
    flow_quantizer_name = "HLQ-flow-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l2-coarsel21e-2-fg_v1_5/model_best.pt"
    depth_quantizer_name = "HLQ-depth-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2-200k_ft500k_3/model_best.pt"
    
    # Video/image input - you can change this to your video path
    # For testing, we'll create synthetic frames if no video path is provided
    video_path = None  # Set to your video path, e.g., "/path/to/video.mp4"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    out_dir = "./parallel_feature_test_output"
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("Parallel Extract Features Test")
    print("=" * 80)
    
    # Load predictor
    print(f"\nLoading PSIPredictor...")
    print(f"  Model: {model_name}")
    print(f"  Quantizer: {quantizer_name}")
    print(f"  Device: {device}")
    
    try:
        predictor = PSIPredictor(
            model_name=model_name,
            quantizer_name=quantizer_name,
            flow_quantizer_name=flow_quantizer_name,
            depth_quantizer_name=depth_quantizer_name,
            device=device
        )
        print("✓ Predictor loaded successfully")
    except Exception as e:
        print(f"✗ Error loading predictor: {e}")
        print("\nPlease update the model paths in the script to point to your actual model files.")
        return
    
    # Load frames
    print(f"\nLoading frames...")
    if video_path and os.path.exists(video_path):
        print(f"  Loading from video: {video_path}")
        frames = video_to_frames(video_path, frame_skip=0, img_size=(512, 512), center_crop=True)
        if len(frames) < 2:
            print("✗ Video has less than 2 frames. Need at least 2 frames.")
            return
        # Take first two frames
        frame0_np = frames[0] / 255.0  # Normalize to [0, 1]
        frame1_np = frames[1] / 255.0
        print(f"✓ Loaded 2 frames from video")
    else:
        # Create synthetic test frames (gradient patterns)
        print("  No video path provided, creating synthetic test frames...")
        frame0_np = np.zeros((512, 512, 3), dtype=np.float32)
        frame1_np = np.zeros((512, 512, 3), dtype=np.float32)
        
        # Frame 0: Red gradient
        for i in range(512):
            frame0_np[i, :, 0] = i / 512.0  # Red channel gradient
        
        # Frame 1: Blue gradient
        for i in range(512):
            frame1_np[:, i, 2] = i / 512.0  # Blue channel gradient
        
        print("✓ Created synthetic test frames")
    
    print(f"  Frame 0 shape: {frame0_np.shape}, range: [{frame0_np.min():.3f}, {frame0_np.max():.3f}]")
    print(f"  Frame 1 shape: {frame1_np.shape}, range: [{frame1_np.min():.3f}, {frame1_np.max():.3f}]")
    
    # Save input frames for reference
    frame0_pil = Image.fromarray((frame0_np * 255).astype(np.uint8))
    frame1_pil = Image.fromarray((frame1_np * 255).astype(np.uint8))
    frame0_pil.save(os.path.join(out_dir, "input_frame0.png"))
    frame1_pil.save(os.path.join(out_dir, "input_frame1.png"))
    print(f"  Saved input frames to {out_dir}")
    
    # Prepare RGB frames list
    rgb_frames = [frame0_np, frame1_np]
    
    # Create masks
    print(f"\nCreating masks...")
    # Frame 0: Fully visible (mask_ratio=0.0)
    unmask_indices_frame0 = create_mask_indices(mask_ratio=0.0, seed=seed)
    # Frame 1: 90% masked (mask_ratio=0.9)
    unmask_indices_frame1 = create_mask_indices(mask_ratio=0.9, seed=seed + 1)
    
    unmask_indices_rgb = [unmask_indices_frame0, unmask_indices_frame1]
    
    # Prepare prompt
    prompt = "rgb0,rgb1->rgb1"
    print(f"\nPrompt: '{prompt}'")
    
    # Time codes
    time_codes = [0, 200]  # 0ms and 200ms
    print(f"Time codes: {time_codes}")
    
    # Call parallel_extract_features
    print(f"\nCalling parallel_extract_features...")
    print(f"  Temperature: 1.0")
    print(f"  Top-p: 0.9")
    print(f"  Top-k: 1000")
    print(f"  Seed: {seed}")
    
    outputs = predictor.parallel_extract_features(
        prompt=prompt,
        rgb_frames=rgb_frames,
        flow_frames=None,
        depth_frames=None,
        unmask_indices_rgb=unmask_indices_rgb,
        unmask_indices_flow=None,
        unmask_indices_depth=None,
        camposes=None,
        use_campose_scale=True,
        time_codes=time_codes,
        index_order=None,
        conditioning_order=None,
        poke_vectors=None,
        seed=seed,
        num_seq_patches=32,
        temp=1.0,
        top_p=0.9,
        top_k=1000,
        out_dir=out_dir,
    )
    
    print("✓ parallel_extract_features completed successfully")
    print(f"\nOutputs type: {type(outputs)}")
    
    def print_tensor_stats(name, tensor):
        """Print tensor statistics: shape, min, max, mean, std"""
        if tensor.numel() == 0:
            print(f"    {name}: empty tensor, shape={tensor.shape}")
            return
        tensor_float = tensor.float()
        print(f"    {name}: shape={tensor.shape}, "
              f"min={tensor_float.min().item():.4f}, max={tensor_float.max().item():.4f}, "
              f"mean={tensor_float.mean().item():.4f}, std={tensor_float.std().item():.4f}")
    
    # Try to visualize/save outputs
    if outputs is not None:
        if isinstance(outputs, dict):
            print(f"Output keys: {list(outputs.keys())}")
            
            for key, value in outputs.items():
                print(f"\n  {key}:")
                if isinstance(value, torch.Tensor):
                    print_tensor_stats("tensor", value)
                elif isinstance(value, list):
                    print(f"    list of {len(value)} items")
                    for i, item in enumerate(value):
                        if isinstance(item, torch.Tensor):
                            print_tensor_stats(f"  [{i}]", item)
                        elif isinstance(item, Image.Image):
                            print(f"      [{i}]: PIL Image, size={item.size}, mode={item.mode}")
                            # Save the image
                            output_path = os.path.join(out_dir, f"{key}_{i}.png")
                            item.save(output_path)
                            print(f"           → Saved to {output_path}")
                        elif item is None:
                            print(f"      [{i}]: None")
                        else:
                            print(f"      [{i}]: {type(item)}")
                elif isinstance(value, Image.Image):
                    print(f"    PIL Image, size={value.size}, mode={value.mode}")
                    output_path = os.path.join(out_dir, f"{key}.png")
                    value.save(output_path)
                    print(f"    → Saved to {output_path}")
                elif isinstance(value, np.ndarray):
                    print(f"    ndarray: shape={value.shape}, "
                          f"min={value.min():.4f}, max={value.max():.4f}, "
                          f"mean={value.mean():.4f}, std={value.std():.4f}")
                elif value is None:
                    print(f"    None")
                else:
                    print(f"    type={type(value)}")
        
        elif isinstance(outputs, (list, tuple)):
            print(f"Number of outputs: {len(outputs)}")
            for i, item in enumerate(outputs):
                print(f"  Output {i}:", end="")
                if isinstance(item, torch.Tensor):
                    print_tensor_stats("", item)
                elif isinstance(item, Image.Image):
                    print(f" PIL Image, size={item.size}")
                    output_path = os.path.join(out_dir, f"output_{i}.png")
                    item.save(output_path)
                    print(f"    → Saved to {output_path}")
                else:
                    print(f" {type(item)}")
        
        elif isinstance(outputs, torch.Tensor):
            print_tensor_stats("Output tensor", outputs)
        
        else:
            print(f"Unknown output format: {type(outputs)}")
    else:
        print("✗ No outputs returned (None)")
    
    print("\n" + "=" * 80)
    print("Test completed")
    print(f"Outputs saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

