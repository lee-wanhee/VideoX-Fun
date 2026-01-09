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

# Trained checkpoint paths - UPDATE THESE
lora_path = None  # Path to trained LoRA checkpoint, e.g., "output_psi_control/checkpoint-5000.safetensors"
psi_projection_path = None  # Path to trained PSI projection, e.g., "output_psi_control/psi_projection-5000.safetensors"

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
# For inference, pick a specific offset (e.g., frame 13 at 30fps = 0.43s, latent idx 4)
psi_control_frame_offset = 13  # Second frame index relative to first frame
psi_source_fps = 30.0  # FPS of the source control video

# Generation settings
prompt = "A video"
negative_prompt = "blurry, low quality, distorted, deformed"
guidance_scale = 6.0
seed = 42  # Fixed random seed
num_inference_steps = 50
lora_weight = 1.0

# Output
save_path = "samples/psi-control-output"

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
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video_path
    
    # Set fixed random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(config_path)

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
    
    if psi_projection_path is not None and os.path.exists(psi_projection_path):
        print(f"Loading trained PSI projection from: {psi_projection_path}")
        psi_state_dict = load_file(psi_projection_path)
        psi_projection.load_state_dict(psi_state_dict)
    else:
        print("WARNING: No trained PSI projection path provided. Using random initialization.")
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
    if lora_path is not None and os.path.exists(lora_path):
        print(f"Merging LoRA from: {lora_path}")
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
    else:
        print("WARNING: No LoRA path provided. Using base model weights.")

    print("=" * 60)
    print("Loading input video...")
    print("=" * 60)
    print(f"Video path: {video_path}")
    print(f"Output size: {sample_size}")
    print(f"Video length: {video_length} frames")
    print(f"Seed: {seed}")
    print("=" * 60)

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        # Adjust video length for VAE temporal compression
        actual_video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

        # Load video frames
        num_frames_to_load = max(psi_control_frame_offset + 1, actual_video_length)
        full_control_video, original_fps = load_video_frames(
            video_path, 
            num_frames=num_frames_to_load,
            sample_size=sample_size, 
            fps=psi_source_fps
        )
        
        print(f"Loaded video shape: {full_control_video.shape}")
        print(f"Original video FPS: {original_fps}")
        
        # Extract exactly 2 frames for PSI control: frame 0 and frame at offset
        # full_control_video shape: (B, C, F, H, W)
        frame_0 = full_control_video[:, :, 0:1, :, :]  # First frame
        frame_offset = min(psi_control_frame_offset, full_control_video.shape[2] - 1)
        frame_1 = full_control_video[:, :, frame_offset:frame_offset+1, :, :]  # Second frame
        psi_control_video = torch.cat([frame_0, frame_1], dim=2)  # (B, C, 2, H, W)
        
        # Compute PSI control parameters
        psi_time_gap_sec = frame_offset / psi_source_fps
        psi_second_latent_idx = 1 + (frame_offset - 1) // 4 if frame_offset > 0 else 0
        
        print(f"PSI Control: frame 0 + frame {frame_offset} (time gap: {psi_time_gap_sec:.3f}s, latent idx: {psi_second_latent_idx})")

        # =====================================================================
        # Extract PSI decoded frames before pipeline for saving
        # =====================================================================
        print("Extracting PSI features for visualization...")
        
        # Prepare input for PSI extractor: (B, F, C, H, W) in [-1, 1]
        psi_input = psi_control_video.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W) -> (B, F, C, H, W)
        psi_input = psi_input * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        psi_input = psi_input.to(device, weight_dtype)
        
        # Extract PSI features
        psi_outputs = psi_control_extractor(psi_input, time_gap_sec=psi_time_gap_sec)
        psi_decoded_frames = psi_outputs['decoded_frames']  # (B, 2, C, H, W) in [-1, 1]
        
        print(f"PSI decoded frames shape: {psi_decoded_frames.shape}")
        
        # =====================================================================
        # Use first frame as reference image for CLIP
        # =====================================================================
        # Convert first frame to PIL for CLIP
        first_frame_01 = full_control_video[0, :, 0]  # (C, H, W) in [0, 1]
        clip_image = tensor_to_pil(first_frame_01)
        
        # Prepare start_image tensor for the pipeline (first frame as reference)
        # This matches training's control_ref mode where ref_latents_conv_in is added
        # In the pipeline, start_image becomes start_image_latentes_conv_in (same purpose)
        start_image_tensor = full_control_video[:, :, 0:1, :, :]  # (B, C, 1, H, W)

        print("=" * 60)
        print("Generating video...")
        print("=" * 60)
        print(f"Prompt: {prompt}")
        print("=" * 60)

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
            start_image=start_image_tensor,  # First frame as reference (matching training's ref_latents_conv_in)
            clip_image=clip_image,  # First frame used for CLIP conditioning
        ).videos

    # =========================================================================
    # Save all outputs
    # =========================================================================
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Count existing outputs to get next index
    existing_files = [f for f in os.listdir(save_path) if f.endswith('_generated.mp4')]
    index = len(existing_files) + 1
    prefix = str(index).zfill(4)
    
    print("=" * 60)
    print(f"Saving outputs with prefix: {prefix}")
    print("=" * 60)
    
    # 1. Save generated video
    if actual_video_length == 1:
        output_path = os.path.join(save_path, f"{prefix}_generated.png")
        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(output_path)
        print(f"Saved generated image: {output_path}")
    else:
        output_path = os.path.join(save_path, f"{prefix}_generated.mp4")
        save_videos_grid(sample, output_path, fps=output_fps)
        print(f"Saved generated video: {output_path}")

    # 2. Save control video
    control_video_path = os.path.join(save_path, f"{prefix}_control_video.mp4")
    save_video_from_tensor(full_control_video, control_video_path, fps=output_fps)
    print(f"Saved control video: {control_video_path}")

    # 3. Save reference image (first frame)
    ref_image_path = os.path.join(save_path, f"{prefix}_ref_image.png")
    clip_image.save(ref_image_path)
    print(f"Saved reference image: {ref_image_path}")

    # 4. Save PSI decoded frames
    for i in range(psi_decoded_frames.shape[1]):
        psi_frame = psi_decoded_frames[0, i]  # (C, H, W) in [-1, 1]
        psi_frame_pil = tensor_to_pil(psi_frame)
        psi_frame_path = os.path.join(save_path, f"{prefix}_psi_decoded_frame{i}.png")
        psi_frame_pil.save(psi_frame_path)
        print(f"Saved PSI decoded frame {i}: {psi_frame_path}")

    # 5. Save input frames (frame 0 and frame at offset)
    input_frame0 = full_control_video[0, :, 0]  # (C, H, W) in [0, 1]
    input_frame0_pil = tensor_to_pil(input_frame0)
    input_frame0_path = os.path.join(save_path, f"{prefix}_input_frame0.png")
    input_frame0_pil.save(input_frame0_path)
    print(f"Saved input frame 0: {input_frame0_path}")
    
    input_frame1 = full_control_video[0, :, frame_offset]  # (C, H, W) in [0, 1]
    input_frame1_pil = tensor_to_pil(input_frame1)
    input_frame1_path = os.path.join(save_path, f"{prefix}_input_frame1.png")
    input_frame1_pil.save(input_frame1_path)
    print(f"Saved input frame 1: {input_frame1_path}")

    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nOutput files in {save_path}:")
    print(f"  {prefix}_generated.mp4       - Generated video")
    print(f"  {prefix}_control_video.mp4   - Input control video")
    print(f"  {prefix}_ref_image.png       - Reference image (first frame)")
    print(f"  {prefix}_psi_decoded_frame0.png - PSI decoded first frame")
    print(f"  {prefix}_psi_decoded_frame1.png - PSI decoded second frame")
    print(f"  {prefix}_input_frame0.png    - Original input frame 0")
    print(f"  {prefix}_input_frame1.png    - Original input frame 1")
    print("=" * 60)


if __name__ == "__main__":
    main()
