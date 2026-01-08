"""
Inference script for Wan2.1-Fun Control with PSI Control

This script loads a trained LoRA + PSI projection model and generates videos
using WanFunControlPipeline with PSI control features.

Usage:
    python examples/wan2.1_fun/predict_v2v_psi_control.py
"""

import os
import sys

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
control_video = "asset/pose.mp4"  # Path to control video (pose/edge/depth)
ref_image = None  # Path to reference image (optional, for appearance)
sample_size = [512, 512]  # [height, width]
video_length = 49
fps = 16

# Generation settings
prompt = "A person dancing gracefully in a garden."
negative_prompt = "blurry, low quality, distorted, deformed"
guidance_scale = 6.0
seed = 42
num_inference_steps = 50
lora_weight = 1.0

# Output
save_path = "samples/psi-control-output"

# Use torch.float16 if GPU does not support torch.bfloat16
weight_dtype = torch.bfloat16

# ==============================================================================
# Main Script
# ==============================================================================

def main():
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
    print("Generating video...")
    print("=" * 60)
    print(f"Control video: {control_video}")
    print(f"Reference image: {ref_image}")
    print(f"Prompt: {prompt}")
    print(f"Output size: {sample_size}")
    print(f"Video length: {video_length} frames")
    print("=" * 60)

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        # Adjust video length for VAE temporal compression
        actual_video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

        # Load control video
        input_video, input_video_mask, _, _ = get_video_to_video_latent(
            control_video, 
            video_length=actual_video_length, 
            sample_size=sample_size, 
            fps=fps, 
            ref_image=None
        )

        # Load reference image for CLIP (optional)
        clip_image = None
        ref_image_tensor = None
        if ref_image is not None and os.path.exists(ref_image):
            clip_image = Image.open(ref_image).convert("RGB")
            ref_image_tensor = get_image_latent(ref_image, sample_size=sample_size)

        # Generate video
        # The pipeline will automatically use PSI control since psi_control_extractor and psi_projection are provided
        sample = pipeline(
            prompt,
            num_frames=actual_video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            control_video=input_video,
            ref_image=ref_image_tensor,
            clip_image=clip_image,
        ).videos

    # Save output
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path) if path.endswith('.mp4')]) + 1
    prefix = str(index).zfill(4)
    
    if actual_video_length == 1:
        output_path = os.path.join(save_path, f"{prefix}.png")
        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(output_path)
        print(f"Saved image to: {output_path}")
    else:
        output_path = os.path.join(save_path, f"{prefix}.mp4")
        save_videos_grid(sample, output_path, fps=fps)
        print(f"Saved video to: {output_path}")

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

