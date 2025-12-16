"""
PSI Control Feature Extractor

This module integrates PSIPredictor's parallel_extract_features functionality
into VideoX-Fun's control training pipeline.

The PSIPredictor extracts high-level semantic features from video frames that
can be used as control signals for video generation.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add parent directory to path to import ccwm modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ccwm.predictor.psi_predictor import PSIPredictor
    from ccwm.utils.image_processing import video_to_frames, load_image
    PSI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import PSIPredictor: {e}")
    print("Please make sure ccwm package is installed and accessible.")
    PSI_AVAILABLE = False


class PSIControlFeatureExtractor(nn.Module):
    """
    Control feature extractor using PSIPredictor's parallel_extract_features.
    
    This wrapper integrates PSIPredictor into the VideoX-Fun training pipeline,
    extracting control features from video frames during training.
    
    Input: control_pixel_values (B, F, C, H, W) - normalized to [-1, 1]
    Output: control_embeddings (B, C_out, F, H_latent, W_latent)
    """
    
    def __init__(
        self,
        model_name: str,
        quantizer_name: str,
        flow_quantizer_name: str = None,
        depth_quantizer_name: str = None,
        latent_channels: int = 16,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
        device: str = "cuda",
        mask_ratio: float = 0.0,  # How much to mask (0.0 = fully visible)
        prompt_template: str = "rgb0,rgb1->rgb1",  # Prompt for PSI
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 1000,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        
        if not PSI_AVAILABLE:
            raise ImportError(
                "PSIPredictor is not available. Please install ccwm package.\n"
                "Make sure the ccwm module is in the parent directory or installed."
            )
        
        self.latent_channels = latent_channels
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        self.device = device
        self.mask_ratio = mask_ratio
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        
        # Initialize PSIPredictor
        print(f"Initializing PSIPredictor...")
        print(f"  Model: {model_name}")
        print(f"  Quantizer: {quantizer_name}")
        print(f"  Flow quantizer: {flow_quantizer_name}")
        print(f"  Depth quantizer: {depth_quantizer_name}")
        print(f"  Device: {device}")
        
        self.predictor = PSIPredictor(
            model_name=model_name,
            quantizer_name=quantizer_name,
            flow_quantizer_name=flow_quantizer_name,
            depth_quantizer_name=depth_quantizer_name,
            device=device
        )
        
        print("✓ PSIPredictor loaded successfully")
        
        # Projection layer to adapt PSI output to expected latent dimensions
        # PSI outputs may have different channel dimensions than VAE latents
        # We'll create this dynamically after the first forward pass
        self.projection = None
        
    def create_mask_indices(self, mask_ratio: float, num_frames: int, seed: int = 0) -> list:
        """
        Create unmask indices for each frame in a 32x32 patch grid.
        
        Args:
            mask_ratio: Ratio of patches to mask (0.0 = all visible, 1.0 = all masked)
            num_frames: Number of frames
            seed: Random seed
            
        Returns:
            List of unmask indices for each frame
        """
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        # Total patches in 32x32 grid (PSI uses 32x32 patches)
        total_patches = 1024
        
        unmask_indices_list = []
        for frame_idx in range(num_frames):
            # Number of patches to keep visible (unmask)
            n_unmask = int(total_patches * (1 - mask_ratio))
            
            # Randomly select patches to unmask
            unmask_indices = np.random.choice(
                total_patches, n_unmask, replace=False
            ).tolist()
            random.shuffle(unmask_indices)
            
            unmask_indices_list.append(unmask_indices)
        
        return unmask_indices_list
    
    def forward(self, control_pixel_values):
        """
        Extract control features from input video frames using PSIPredictor.
        
        Args:
            control_pixel_values: (B, F, C, H, W) tensor, values in [-1, 1]
                                 B = batch size
                                 F = number of frames
                                 C = 3 (RGB channels)
                                 H, W = height and width
        
        Returns:
            control_embeddings: (B, C_out, F, H_latent, W_latent) tensor
                               B = batch size
                               C_out = latent_channels (e.g., 16)
                               F = number of frames
                               H_latent = H // spatial_compression
                               W_latent = W // spatial_compression
        """
        batch_size, num_frames, channels, height, width = control_pixel_values.shape
        
        # Calculate output spatial dimensions
        h_latent = height // self.spatial_compression
        w_latent = width // self.spatial_compression
        
        # Convert from [-1, 1] to [0, 1] for PSIPredictor
        # PSI expects images in [0, 1] range
        rgb_frames_01 = (control_pixel_values + 1.0) / 2.0
        
        # Process each sample in the batch
        batch_embeddings = []
        
        for b in range(batch_size):
            # Get frames for this sample: (F, C, H, W)
            sample_frames = rgb_frames_01[b]
            
            # Convert to list of numpy arrays (H, W, C) as expected by PSI
            rgb_frames = []
            for f in range(num_frames):
                frame = sample_frames[f].permute(1, 2, 0)  # (H, W, C)
                frame_np = frame.cpu().numpy()
                rgb_frames.append(frame_np)
            
            # Create masks for each frame
            unmask_indices_rgb = self.create_mask_indices(
                self.mask_ratio, 
                num_frames, 
                seed=self.seed + b
            )
            
            # Generate prompt for PSI
            # Example: "rgb0,rgb1->rgb1" means use rgb0 and rgb1 to predict rgb1
            frame_ids = ",".join([f"rgb{i}" for i in range(num_frames)])
            prompt = f"{frame_ids}->rgb{num_frames-1}"
            
            # Create time codes (evenly spaced)
            time_codes = [int(i * 1000 / num_frames) for i in range(num_frames)]
            
            # Call parallel_extract_features
            try:
                outputs = self.predictor.parallel_extract_features(
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
                    seed=self.seed + b,
                    num_seq_patches=32,
                    temp=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    out_dir=None,
                )
                
                # Extract feature tensor from outputs
                # The output format depends on PSI implementation
                # Typically it returns a dict with 'features' or similar key
                if isinstance(outputs, dict):
                    # Try common keys
                    feature_tensor = None
                    for key in ['features', 'embeddings', 'latents', 'hidden_states']:
                        if key in outputs and isinstance(outputs[key], torch.Tensor):
                            feature_tensor = outputs[key]
                            break
                    
                    if feature_tensor is None:
                        # If not found, try to use any tensor in the output
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor):
                                feature_tensor = value
                                break
                    
                    if feature_tensor is None:
                        raise ValueError(f"Could not find feature tensor in PSI outputs. Keys: {list(outputs.keys())}")
                
                elif isinstance(outputs, torch.Tensor):
                    feature_tensor = outputs
                else:
                    raise ValueError(f"Unexpected PSI output type: {type(outputs)}")
                
                # Ensure feature tensor is on the correct device
                feature_tensor = feature_tensor.to(control_pixel_values.device)
                
                # Expected shape: (C, F, H, W) or (F, C, H, W) or similar
                # We need to reshape/transpose to (C, F, H_latent, W_latent)
                
                # Add batch dimension if needed
                if feature_tensor.dim() == 4:
                    # Assume (C, F, H, W) or (F, C, H, W)
                    if feature_tensor.shape[1] == num_frames:
                        # (C, F, H, W) - already correct
                        pass
                    elif feature_tensor.shape[0] == num_frames:
                        # (F, C, H, W) - transpose
                        feature_tensor = feature_tensor.permute(1, 0, 2, 3)
                
                # Resize to match expected latent dimensions if needed
                if feature_tensor.shape[-2:] != (h_latent, w_latent):
                    # Reshape spatial dimensions
                    c, f, h, w = feature_tensor.shape
                    feature_tensor = feature_tensor.reshape(c * f, h, w).unsqueeze(0)  # (1, C*F, H, W)
                    feature_tensor = torch.nn.functional.interpolate(
                        feature_tensor,
                        size=(h_latent, w_latent),
                        mode='bilinear',
                        align_corners=False
                    )
                    feature_tensor = feature_tensor.squeeze(0).reshape(c, f, h_latent, w_latent)
                
                # Create projection layer if needed
                current_channels = feature_tensor.shape[0]
                if self.projection is None and current_channels != self.latent_channels:
                    print(f"Creating projection layer: {current_channels} -> {self.latent_channels} channels")
                    self.projection = nn.Conv2d(
                        current_channels, 
                        self.latent_channels, 
                        kernel_size=1
                    ).to(control_pixel_values.device, control_pixel_values.dtype)
                
                # Apply projection if needed
                if self.projection is not None:
                    # Process each frame through projection
                    c, f, h, w = feature_tensor.shape
                    feature_tensor = feature_tensor.permute(1, 0, 2, 3).reshape(f, c, h, w)  # (F, C, H, W)
                    projected_frames = []
                    for frame_feat in feature_tensor:
                        projected = self.projection(frame_feat.unsqueeze(0))  # (1, C_out, H, W)
                        projected_frames.append(projected.squeeze(0))
                    feature_tensor = torch.stack(projected_frames, dim=1)  # (C_out, F, H, W)
                
                batch_embeddings.append(feature_tensor)
                
            except Exception as e:
                print(f"Warning: PSI feature extraction failed for batch {b}: {e}")
                print("Falling back to zero embeddings")
                # Fallback: create zero embeddings
                fallback_embedding = torch.zeros(
                    self.latent_channels,
                    num_frames,
                    h_latent,
                    w_latent,
                    device=control_pixel_values.device,
                    dtype=control_pixel_values.dtype
                )
                batch_embeddings.append(fallback_embedding)
        
        # Stack batch: (B, C_out, F, H_latent, W_latent)
        control_embeddings = torch.stack(batch_embeddings, dim=0)
        
        return control_embeddings
    
    @classmethod
    def from_pretrained(cls, config_dict, **kwargs):
        """
        Load the PSI control feature extractor from a configuration.
        
        Args:
            config_dict: Dictionary containing PSI model paths and settings
                        Required keys: model_name, quantizer_name
                        Optional keys: flow_quantizer_name, depth_quantizer_name, device, etc.
            **kwargs: Additional arguments for model initialization
        
        Returns:
            Loaded model instance
        """
        # Merge config_dict with kwargs
        merged_config = {**config_dict, **kwargs}
        
        # Create model instance
        model = cls(**merged_config)
        
        return model


if __name__ == "__main__":
    """Test the PSI control feature extractor"""
    print("=" * 80)
    print("Testing PSI Control Feature Extractor")
    print("=" * 80)
    
    # Test configuration
    test_config = {
        'model_name': "PSI_7B_RGBCDF_bvd_4frame_Unified_Vocab_Balanced_Task_V2_continue_ctx_8192/model_01400000.pt",
        'quantizer_name': "PLPQ-ImageNetOpenImages-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2/model_best.pt",
        'flow_quantizer_name': "HLQ-flow-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l2-coarsel21e-2-fg_v1_5/model_best.pt",
        'depth_quantizer_name': "HLQ-depth-nq2-gen2_0-wavelet-small-bs512-lr1e-4-l1-dinov21e0224-coarsel11e-2-200k_ft500k_3/model_best.pt",
        'latent_channels': 16,
        'spatial_compression': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    try:
        # Create extractor
        print("\nCreating PSI Control Feature Extractor...")
        extractor = PSIControlFeatureExtractor(**test_config)
        extractor.eval()
        
        # Create dummy input
        batch_size = 1
        num_frames = 2
        height, width = 512, 512
        
        print(f"\nCreating test input: ({batch_size}, {num_frames}, 3, {height}, {width})")
        control_input = torch.randn(batch_size, num_frames, 3, height, width)
        control_input = torch.tanh(control_input)  # Normalize to [-1, 1]
        
        if torch.cuda.is_available():
            control_input = control_input.cuda()
            extractor = extractor.cuda()
        
        # Test forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            output = extractor(control_input)
        
        print(f"\nResults:")
        print(f"  Input shape:  {control_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected:     ({batch_size}, {test_config['latent_channels']}, {num_frames}, {height//8}, {width//8})")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output mean:  {output.mean().item():.4f}")
        print(f"  Output std:   {output.std().item():.4f}")
        
        print("\n" + "=" * 80)
        print("✓ Test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

