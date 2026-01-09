"""
Inference script for Wan2.1-Fun Control with PSI Control

This script loads a trained LoRA + PSI projection model and generates videos
using WanFunControlPipeline with PSI control features.

Usage:
    python examples/wan2.1_fun/predict_v2v_psi_control.py
    python examples/wan2.1_fun/predict_v2v_psi_control.py --video_path /path/to/video.mp4
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, WanTransformer3DModel)
from videox_fun.models.psi_projection import PSIProjectionSwiGLU
from videox_fun.pipeline import WanFunControlPipeline
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)

# Add path for PSI control extractor
sys.path.insert(0, os.path.join(project_root, "scripts/wan2.1_fun"))
from psi_control_extractor import PSIControlFeatureExtractor

# ==============================================================================
# Configuration - MODIFY THESE FOR YOUR SETUP
# ==============================================================================

# GPU memory mode: "model_cpu_offload" or "model_full_load"
GPU_memory_mode = "model_cpu_offload"

# Config and model path
config_path = "config/wan2.1/wan_civitai.yaml"
model_name = "models/Wan2.1-Fun-1.3B-Control"  # Base model path

# Trained checkpoint paths - can be overridden by command line args
default_lora_path = None  # Path to trained LoRA checkpoint
default_psi_projection_path = None  # Path to trained PSI projection
default_checkpoint_name = "none"  # Name for output folder (e.g., "output_psi_control_test-5000")

# PSI Control Extractor config (same as training)
psi_control_extractor_config = {
    'model_name': "PSI_7B_RGBCDF_bvd_4frame_Unified_Vocab_Balanced_Task_V2_continue_ctx_8192/model_01400000.pt",
    'quantizer_name': "PLPQ-ImageNetOpenImages-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2/model_best.pt",
    'flow_quantizer_name': "HLQ-flow-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l2-coarsel21e-2-fg_v1_5/model_best.pt",
    'depth_quantizer_name': "HLQ-depth-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2-200k_ft500k_3/model_best.pt",
    'spatial_compression': 8,
}

# Input/Output settings
default_video_path = "/ccn2/dataset/kinetics400/Kinetics400/k400/val/--07WQ2iBlw_000001_000011.mp4"
sample_size = [512, 512]  # [height, width]
video_length = 49
output_fps = 16  # Output video fps

# PSI Control settings (matching training)
# The model was trained with frames sampled at (1+4*n) offsets within 0.2-1.0s
# For inference, pick specific frame times (e.g., 1.0s and 1.5s)
psi_source_fps = 30.0  # FPS of the source control video
psi_start_time_sec = 0.5  # Start time for first PSI frame (seconds)
psi_time_gap_sec_config = 0.5  # Time gap between two PSI frames (seconds)
# Computed frame indices
psi_start_frame = int(psi_start_time_sec * psi_source_fps)  # Frame 30 at 30fps for 1.0s
psi_control_frame_offset = int(psi_time_gap_sec_config * psi_source_fps)  # Frame offset: 15 for 0.5s gap

# Generation settings
prompt = "A video"
negative_prompt = "blurry, low quality, distorted, deformed"
guidance_scale = 6.0
seed = 42  # Fixed random seed
num_inference_steps = 50
lora_weight = 1.0

# Output - will be set based on checkpoint_name arg
default_save_path = "samples/psi-control-output"

# Use torch.float16 if GPU does not support torch.bfloat16
weight_dtype = torch.bfloat16

# ==============================================================================
# Helper Functions
# ==============================================================================

def load_video_frames(video_path, num_frames, sample_size, fps=None):
    """Load video frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to load
        sample_size: [height, width] for resizing
        fps: Target FPS for frame sampling (None = use all frames)
        
    Returns:
        frames: (B, C, F, H, W) tensor in [0, 1] range
        original_fps: Original FPS of the video
    """
    cap = cv2.VideoCapture(video_path)
    input_frames = []
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = 1 if fps is None else max(1, int(original_fps // fps))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
            input_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frame_count += 1
    
    cap.release()
    
    # Convert to tensor
    input_frames = torch.from_numpy(np.array(input_frames))[:num_frames]
    input_frames = input_frames.permute([3, 0, 1, 2]).unsqueeze(0) / 255.0
    
    return input_frames, original_fps


def tensor_to_pil(tensor):
    """Convert a tensor frame to PIL Image.
    
    Args:
        tensor: (C, H, W) tensor in [-1, 1] or [0, 1] range
        
    Returns:
        PIL.Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Handle [-1, 1] range
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    # Convert to float32 first (numpy doesn't support bfloat16)
    tensor = (tensor.float() * 255).cpu().numpy().astype(np.uint8)
    return Image.fromarray(tensor)


def save_video_from_tensor(tensor, output_path, fps):
    """Save a video tensor as mp4 file.
    
    Args:
        tensor: (B, C, F, H, W) tensor in [0, 1] range
        output_path: Path to save the video
        fps: Output FPS
    """
    if tensor.dim() == 5:
        tensor = tensor[0]  # Remove batch dimension
    
    # tensor: (C, F, H, W)
    C, F, H, W = tensor.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for frame_idx in range(F):
        frame = tensor[:, frame_idx]  # (C, H, W)
        frame = frame.permute(1, 2, 0)  # (H, W, C)
        frame = (frame.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()


def visualize_psi_masking(frame0, frame1, mask_ratio_cond=0.0, mask_ratio_pred=0.9, 
                          seed=42, output_path=None):
    """Visualize PSI masking on two frames.
    
    Creates a 2x2 plot showing:
    - Row 1: Original frame0, Original frame1
    - Row 2: Masked frame0, Masked frame1
    
    Args:
        frame0: First frame tensor (C, H, W) in [0, 1] range
        frame1: Second frame tensor (C, H, W) in [0, 1] range
        mask_ratio_cond: Mask ratio for conditioning frame (default 0.0)
        mask_ratio_pred: Mask ratio for prediction frame (default 0.9)
        seed: Random seed for mask generation
        output_path: Path to save the figure
    """
    # Convert tensors to numpy for visualization
    if isinstance(frame0, torch.Tensor):
        frame0_np = frame0.permute(1, 2, 0).float().cpu().numpy()
    else:
        frame0_np = frame0
    if isinstance(frame1, torch.Tensor):
        frame1_np = frame1.permute(1, 2, 0).float().cpu().numpy()
    else:
        frame1_np = frame1
    
    # PSI uses 32x32 patch grid (each patch is 16x16 pixels for 512x512 images)
    patch_grid_size = 32
    total_patches = patch_grid_size * patch_grid_size  # 1024
    
    H, W = frame0_np.shape[:2]
    patch_h = H // patch_grid_size
    patch_w = W // patch_grid_size
    
    def create_mask_indices(mask_ratio, seed_offset):
        """Create unmask indices for a 32x32 patch grid."""
        np.random.seed(seed + seed_offset)
        n_unmask = int(total_patches * (1 - mask_ratio))
        unmask_indices = np.random.choice(total_patches, n_unmask, replace=False)
        return set(unmask_indices)
    
    def apply_patch_mask(frame, unmask_indices):
        """Apply patch-level masking to a frame."""
        masked_frame = np.zeros_like(frame)
        # Create a gray background for masked regions
        masked_frame[:, :] = [0.5, 0.5, 0.5]  # Gray
        
        for patch_idx in unmask_indices:
            row = patch_idx // patch_grid_size
            col = patch_idx % patch_grid_size
            y_start = row * patch_h
            y_end = (row + 1) * patch_h
            x_start = col * patch_w
            x_end = (col + 1) * patch_w
            masked_frame[y_start:y_end, x_start:x_end] = frame[y_start:y_end, x_start:x_end]
        
        return masked_frame
    
    def create_mask_overlay(frame, unmask_indices, alpha=0.5):
        """Create an overlay showing which patches are masked."""
        overlay = frame.copy()
        mask_color = np.array([1.0, 0.0, 0.0])  # Red for masked patches
        
        for patch_idx in range(total_patches):
            if patch_idx not in unmask_indices:
                row = patch_idx // patch_grid_size
                col = patch_idx % patch_grid_size
                y_start = row * patch_h
                y_end = (row + 1) * patch_h
                x_start = col * patch_w
                x_end = (col + 1) * patch_w
                overlay[y_start:y_end, x_start:x_end] = (
                    alpha * mask_color + (1 - alpha) * overlay[y_start:y_end, x_start:x_end]
                )
        
        return overlay
    
    # Generate mask indices
    unmask_indices_0 = create_mask_indices(mask_ratio_cond, seed_offset=0)
    unmask_indices_1 = create_mask_indices(mask_ratio_pred, seed_offset=1)
    
    # Apply masks
    masked_frame0 = apply_patch_mask(frame0_np, unmask_indices_0)
    masked_frame1 = apply_patch_mask(frame1_np, unmask_indices_1)
    
    # Create overlay views (red = masked)
    overlay_frame0 = create_mask_overlay(frame0_np, unmask_indices_0)
    overlay_frame1 = create_mask_overlay(frame1_np, unmask_indices_1)
    
    # Calculate statistics
    n_visible_0 = len(unmask_indices_0)
    n_visible_1 = len(unmask_indices_1)
    pct_visible_0 = 100 * n_visible_0 / total_patches
    pct_visible_1 = 100 * n_visible_1 / total_patches
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Row 1: Original frames
    axes[0, 0].imshow(np.clip(frame0_np, 0, 1))
    axes[0, 0].set_title(f'Frame 0 (Original)\nFirst PSI frame', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.clip(frame1_np, 0, 1))
    axes[0, 1].set_title(f'Frame 1 (Original)\nSecond PSI frame', fontsize=12)
    axes[0, 1].axis('off')
    
    # Row 2: Masked frames (with overlay showing masked regions in red)
    axes[1, 0].imshow(np.clip(overlay_frame0, 0, 1))
    axes[1, 0].set_title(f'Frame 0 Masked (mask_ratio={mask_ratio_cond})\n'
                         f'{n_visible_0}/{total_patches} visible ({pct_visible_0:.1f}%)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.clip(overlay_frame1, 0, 1))
    axes[1, 1].set_title(f'Frame 1 Masked (mask_ratio={mask_ratio_pred})\n'
                         f'{n_visible_1}/{total_patches} visible ({pct_visible_1:.1f}%)', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.suptitle('PSI Control Frame Masking Visualization\n(Red = masked patches)', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved masking visualization: {output_path}")
    
    plt.close()
    
    return fig


def visualize_gt_vs_pred(gt_frame0, gt_frame1, pred_frame0, pred_frame1, 
                         frame0_info="", frame1_info="", output_path=None):
    """Visualize GT vs Predicted frames comparison.
    
    Creates a 2x2 plot showing:
    - Row 1: GT_frame0, GT_frame1
    - Row 2: pred_frame0, pred_frame1
    
    Args:
        gt_frame0: Ground truth first frame tensor (C, H, W) in [0, 1] range
        gt_frame1: Ground truth second frame tensor (C, H, W) in [0, 1] range
        pred_frame0: Predicted first frame tensor (C, H, W) in [0, 1] range
        pred_frame1: Predicted second frame tensor (C, H, W) in [0, 1] range
        frame0_info: Additional info string for frame 0
        frame1_info: Additional info string for frame 1
        output_path: Path to save the figure
    """
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.float().cpu()
            if tensor.dim() == 3 and tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            return tensor.numpy()
        return tensor
    
    gt0_np = to_numpy(gt_frame0)
    gt1_np = to_numpy(gt_frame1)
    pred0_np = to_numpy(pred_frame0)
    pred1_np = to_numpy(pred_frame1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Row 1: GT frames
    axes[0, 0].imshow(np.clip(gt0_np, 0, 1))
    axes[0, 0].set_title(f'GT Frame 0\n{frame0_info}', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.clip(gt1_np, 0, 1))
    axes[0, 1].set_title(f'GT Frame 1\n{frame1_info}', fontsize=12)
    axes[0, 1].axis('off')
    
    # Row 2: Predicted frames
    axes[1, 0].imshow(np.clip(pred0_np, 0, 1))
    axes[1, 0].set_title(f'Predicted Frame 0\n(Generated video frame 0)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.clip(pred1_np, 0, 1))
    axes[1, 1].set_title(f'Predicted Frame 1\n{frame1_info}', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.suptitle('Ground Truth vs Predicted Frames Comparison', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved GT vs Pred visualization: {output_path}")
    
    plt.close()
    
    return fig


def get_video_frame_from_latent_idx(latent_idx):
    """Convert latent frame index to video frame index.
    
    With VAE temporal compression ratio of 4:
    - Latent 0 -> Video frame 0
    - Latent 1 -> Video frame 1
    - Latent 2 -> Video frame 5
    - Latent 3 -> Video frame 9
    - Latent L (L >= 1) -> Video frame 1 + 4*(L-1) = 4*L - 3
    
    Args:
        latent_idx: Index in latent space
        
    Returns:
        video_frame_idx: Corresponding video frame index
    """
    if latent_idx == 0:
        return 0
    else:
        return 1 + 4 * (latent_idx - 1)


# ==============================================================================
# Main Script
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with PSI Control for video generation")
    parser.add_argument(
        "--video_path",
        type=str,
        default=default_video_path,
        help=f"Path to input video file (default: {default_video_path})"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=str(seed),
        help=f"Comma-separated seeds for multiple runs (default: {seed}). E.g., '42,123,456'"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=default_lora_path,
        help="Path to trained LoRA checkpoint (.safetensors). None for no LoRA."
    )
    parser.add_argument(
        "--psi_projection_path",
        type=str,
        default=default_psi_projection_path,
        help="Path to trained PSI projection (.safetensors). None for random init."
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=default_checkpoint_name,
        help="Name for output folder (e.g., 'output_psi_control_test-5000' or 'none' for baseline)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help=f"Base path for saving outputs (default: {default_save_path})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video_path
    
    # Parse seeds (comma-separated)
    run_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    run_lora_path = args.lora_path
    run_psi_projection_path = args.psi_projection_path
    run_checkpoint_name = args.checkpoint_name
    run_save_path = os.path.join(args.save_path, run_checkpoint_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(config_path)
    
    print("=" * 60)
    print("Checkpoint Configuration:")
    print(f"  LoRA path: {run_lora_path}")
    print(f"  PSI projection path: {run_psi_projection_path}")
    print(f"  Checkpoint name: {run_checkpoint_name}")
    print(f"  Output path: {run_save_path}")
    print(f"  Seeds: {run_seeds} ({len(run_seeds)} runs)")
    print("=" * 60)

    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    # Load Transformer
    print("Loading Transformer...")
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    # Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Load Text Encoder
    print("Loading Text Encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Load CLIP Image Encoder
    print("Loading CLIP Image Encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype)
    clip_image_encoder = clip_image_encoder.eval()

    # Load PSI Control Extractor (frozen)
    print("Loading PSI Control Extractor...")
    psi_control_extractor = PSIControlFeatureExtractor.from_pretrained(psi_control_extractor_config)
    psi_control_extractor = psi_control_extractor.eval()
    psi_control_extractor.requires_grad_(False)

    # Load PSI Projection
    print("Loading PSI Projection...")
    psi_projection = PSIProjectionSwiGLU(
        n_input_channels=8192, 
        n_hidden_channels=256, 
        n_output_channels=16
    ).to(weight_dtype)
    
    if run_psi_projection_path is not None and os.path.exists(run_psi_projection_path):
        print(f"Loading trained PSI projection from: {run_psi_projection_path}")
        psi_state_dict = load_file(run_psi_projection_path)
        psi_projection.load_state_dict(psi_state_dict)
    else:
        print("INFO: No trained PSI projection path provided. Using random initialization.")
    psi_projection = psi_projection.eval()

    # Load Scheduler
    print("Loading Scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Create Pipeline with PSI control
    print("Creating Pipeline...")
    pipeline = WanFunControlPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
        psi_control_extractor=psi_control_extractor,
        psi_projection=psi_projection,
    )

    # Apply GPU memory mode
    if GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    # Merge LoRA if provided
    if run_lora_path is not None and os.path.exists(run_lora_path):
        print(f"Merging LoRA from: {run_lora_path}")
        pipeline = merge_lora(pipeline, run_lora_path, lora_weight, device=device, dtype=weight_dtype)
    else:
        print("INFO: No LoRA path provided. Using base model weights.")

    print("=" * 60)
    print("Loading input video...")
    print("=" * 60)
    print(f"Video path: {video_path}")
    print(f"Output size: {sample_size}")
    print(f"Video length: {video_length} frames")
    print(f"Seeds: {run_seeds}")
    print("=" * 60)

    with torch.no_grad():
        # Adjust video length for VAE temporal compression
        actual_video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

        # Load video frames - need enough frames to reach psi_start_frame + psi_control_frame_offset
        num_frames_to_load = max(psi_start_frame + psi_control_frame_offset + 1, actual_video_length)
        full_control_video, original_fps = load_video_frames(
            video_path, 
            num_frames=num_frames_to_load,
            sample_size=sample_size, 
            fps=psi_source_fps
        )
        
        print(f"Loaded video shape: {full_control_video.shape}")
        print(f"Original video FPS: {original_fps}")
        
        # Extract exactly 2 frames for PSI control at specified times
        # Frame 1: at psi_start_frame (e.g., 1.0s = frame 30)
        # Frame 2: at psi_start_frame + psi_control_frame_offset (e.g., 1.5s = frame 45)
        first_frame_idx = min(psi_start_frame, full_control_video.shape[2] - 1)
        second_frame_idx = min(psi_start_frame + psi_control_frame_offset, full_control_video.shape[2] - 1)
        
        frame_0 = full_control_video[:, :, first_frame_idx:first_frame_idx+1, :, :]  # First PSI frame
        frame_1 = full_control_video[:, :, second_frame_idx:second_frame_idx+1, :, :]  # Second PSI frame
        psi_control_video = torch.cat([frame_0, frame_1], dim=2)  # (B, C, 2, H, W)
        
        # Compute PSI control parameters
        # Time gap is between the two PSI frames (not from frame 0 of video)
        psi_time_gap_sec = (second_frame_idx - first_frame_idx) / psi_source_fps
        psi_second_latent_idx = 1 + (psi_control_frame_offset - 1) // 4 if psi_control_frame_offset > 0 else 0
        
        print(f"PSI Control: frame {first_frame_idx} ({first_frame_idx/psi_source_fps:.2f}s) + frame {second_frame_idx} ({second_frame_idx/psi_source_fps:.2f}s)")
        print(f"  Time gap: {psi_time_gap_sec:.3f}s, latent idx: {psi_second_latent_idx}")

        # Create output directory
        if not os.path.exists(run_save_path):
            os.makedirs(run_save_path, exist_ok=True)
        
        # Get the two input frames for visualization (in [0, 1] range)
        vis_frame0 = full_control_video[0, :, first_frame_idx]  # (C, H, W)
        vis_frame1 = full_control_video[0, :, second_frame_idx]  # (C, H, W)
        
        # Prepare input for PSI extractor: (B, F, C, H, W) in [-1, 1]
        psi_input = psi_control_video.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W) -> (B, F, C, H, W)
        psi_input = psi_input * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        psi_input = psi_input.to(device, weight_dtype)
        
        # =====================================================================
        # Use first PSI frame as reference image for CLIP and start_image
        # =====================================================================
        # Convert first PSI frame to PIL for CLIP (at first_frame_idx, e.g., 1.0s)
        first_psi_frame_01 = full_control_video[0, :, first_frame_idx]  # (C, H, W) in [0, 1]
        clip_image = tensor_to_pil(first_psi_frame_01)
        
        # Prepare start_image tensor for the pipeline (first PSI frame as reference)
        # This matches training's control_ref mode where ref_latents_conv_in is added
        # In the pipeline, start_image becomes start_image_latentes_conv_in (same purpose)
        start_image_tensor = full_control_video[:, :, first_frame_idx:first_frame_idx+1, :, :]  # (B, C, 1, H, W)

        # =====================================================================
        # Loop through seeds for generation
        # =====================================================================
        for seed_idx, run_seed in enumerate(run_seeds):
            print("=" * 60)
            print(f"Generating video [{seed_idx + 1}/{len(run_seeds)}] with seed={run_seed}")
            print("=" * 60)
            print(f"Prompt: {prompt}")
            print("=" * 60)
            
            # Set seed for this run
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            generator = torch.Generator(device=device).manual_seed(run_seed)
            
            # =====================================================================
            # Extract PSI features with this seed (different mask for each seed)
            # =====================================================================
            print(f"Extracting PSI features with seed={run_seed}...")
            psi_outputs = psi_control_extractor(psi_input, time_gap_sec=psi_time_gap_sec, seed=run_seed)
            psi_decoded_frames = psi_outputs['decoded_frames']  # (B, 2, C, H, W) in [-1, 1]
            print(f"PSI decoded frames shape: {psi_decoded_frames.shape}")

            # Generate video
            # start_image is concatenated with PSI control latents on channel dimension (matching training)
            # PSI control: 16ch + start_image: 16ch = 32ch total (matches training)
            # clip_image is used for CLIP conditioning
            sample = pipeline(
                prompt,
                num_frames=actual_video_length,
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                control_video=psi_control_video,  # 2-frame control video
                psi_time_gap_sec=psi_time_gap_sec,  # Time gap between frames
                psi_second_latent_idx=psi_second_latent_idx,  # Latent index for second frame
                psi_seed=run_seed,  # Seed for PSI mask generation
                start_image=start_image_tensor,  # First frame as reference (matching training's ref_latents_conv_in)
                clip_image=clip_image,  # First frame used for CLIP conditioning
            ).videos

            # =========================================================================
            # Save all outputs for this seed
            # =========================================================================
            prefix = f"seed{run_seed:05d}"
            
            print(f"Saving outputs with prefix: {prefix}")
            
            # 1. Save generated video
            if actual_video_length == 1:
                output_path = os.path.join(run_save_path, f"{prefix}_generated.png")
                image = sample[0, :, 0]
                image = image.transpose(0, 1).transpose(1, 2)
                image = (image * 255).numpy().astype(np.uint8)
                image = Image.fromarray(image)
                image.save(output_path)
                print(f"Saved generated image: {output_path}")
            else:
                output_path = os.path.join(run_save_path, f"{prefix}_generated.mp4")
                save_videos_grid(sample, output_path, fps=output_fps)
                print(f"Saved generated video: {output_path}")
            
            # 2. Save control video
            control_video_path = os.path.join(run_save_path, f"{prefix}_control_video.mp4")
            save_video_from_tensor(full_control_video, control_video_path, fps=output_fps)
            print(f"Saved control video: {control_video_path}")

            # 3. Save reference image (first frame)
            ref_image_path = os.path.join(run_save_path, f"{prefix}_ref_image.png")
            clip_image.save(ref_image_path)
            print(f"Saved reference image: {ref_image_path}")

            # 4. Save PSI decoded frames
            for i in range(psi_decoded_frames.shape[1]):
                psi_frame = psi_decoded_frames[0, i]  # (C, H, W) in [-1, 1]
                psi_frame_pil = tensor_to_pil(psi_frame)
                psi_frame_path = os.path.join(run_save_path, f"{prefix}_psi_decoded_frame{i}.png")
                psi_frame_pil.save(psi_frame_path)
                print(f"Saved PSI decoded frame {i}: {psi_frame_path}")

            # 5. Save input frames (the two PSI control frames)
            input_frame0 = full_control_video[0, :, first_frame_idx]  # (C, H, W) in [0, 1]
            input_frame0_pil = tensor_to_pil(input_frame0)
            input_frame0_path = os.path.join(run_save_path, f"{prefix}_input_frame0.png")
            input_frame0_pil.save(input_frame0_path)
            print(f"Saved input frame 0: {input_frame0_path}")
            
            input_frame1 = full_control_video[0, :, second_frame_idx]  # (C, H, W) in [0, 1]
            input_frame1_pil = tensor_to_pil(input_frame1)
            input_frame1_path = os.path.join(run_save_path, f"{prefix}_input_frame1.png")
            input_frame1_pil.save(input_frame1_path)
            print(f"Saved input frame 1: {input_frame1_path}")
            
            # 6. Extract predicted frames from generated video and save comparison
            # sample shape: (B, C, F, H, W)
            # pred_frame0 corresponds to video frame 0 (latent idx 0)
            # pred_frame1 corresponds to the video frame at psi_second_latent_idx
            pred_video_frame1_idx = get_video_frame_from_latent_idx(psi_second_latent_idx)
            pred_video_frame1_idx = min(pred_video_frame1_idx, sample.shape[2] - 1)  # Clamp to valid range
            
            pred_frame0 = sample[0, :, 0]  # (C, H, W) - first frame of generated video
            pred_frame1 = sample[0, :, pred_video_frame1_idx]  # (C, H, W) - frame at psi_second_latent_idx
            
            # Save predicted frame 1 (corresponding to GT frame 1)
            pred_frame1_pil = tensor_to_pil(pred_frame1)
            pred_frame1_path = os.path.join(run_save_path, f"{prefix}_pred_frame1.png")
            pred_frame1_pil.save(pred_frame1_path)
            print(f"Saved pred frame 1 (video frame {pred_video_frame1_idx}): {pred_frame1_path}")
            
            # 7. Create GT vs Pred comparison visualization
            gt_vs_pred_path = os.path.join(run_save_path, f"{prefix}_gt_vs_pred.png")
            visualize_gt_vs_pred(
                gt_frame0=input_frame0,
                gt_frame1=input_frame1,
                pred_frame0=pred_frame0,
                pred_frame1=pred_frame1,
                frame0_info=f"(source frame {first_frame_idx})",
                frame1_info=f"(video frame {pred_video_frame1_idx}, latent {psi_second_latent_idx})",
                output_path=gt_vs_pred_path
            )
            
            # 8. Save PSI masking visualization
            # Masking is now seed-dependent (different mask pattern for each seed)
            mask_viz_path = os.path.join(run_save_path, f"{prefix}_psi_masking.png")
            visualize_psi_masking(
                vis_frame0, vis_frame1,
                mask_ratio_cond=0.0,  # Frame 0: fully visible
                mask_ratio_pred=0.9,  # Frame 1: 90% masked
                seed=run_seed,  # Each seed gets different mask pattern
                output_path=mask_viz_path
            )
            
            print("")

    print("=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"\nOutput files in {run_save_path}:")
    print(f"  seed*_generated.mp4          - Generated videos")
    print(f"  seed*_control_video.mp4      - Input control video")
    print(f"  seed*_ref_image.png          - Reference image")
    print(f"  seed*_psi_decoded_frame0.png - PSI decoded first frame")
    print(f"  seed*_psi_decoded_frame1.png - PSI decoded second frame")
    print(f"  seed*_input_frame0.png       - Original input frame 0 (GT)")
    print(f"  seed*_input_frame1.png       - Original input frame 1 (GT)")
    print(f"  seed*_pred_frame1.png        - Predicted frame 1 from generated video")
    print(f"  seed*_gt_vs_pred.png         - GT vs Predicted comparison plot")
    print(f"  seed*_psi_masking.png        - PSI masking visualization")
    print("=" * 60)


if __name__ == "__main__":
    main()
