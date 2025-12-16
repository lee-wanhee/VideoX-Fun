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
    
    Extracts control from 2 frames spaced 0.5 seconds apart.
    
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
        time_gap_sec: float = 0.5,  # Time gap between the 2 frames (in seconds)
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
        self.time_gap_sec = time_gap_sec
        
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
        
        # Set PSI predictor to eval mode
        self.predictor.model.eval()
        
        print("✓ PSIPredictor loaded successfully (inference mode)")
        
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
    
    def forward(self, control_pixel_values, fps=None):
        """
        Extract control features from input video frames using PSIPredictor.
        
        Extracts features from only 2 frames spaced time_gap_sec (default 0.5s) apart.
        
        Args:
            control_pixel_values: (B, F, C, H, W) tensor, values in [-1, 1]
                                 B = batch size
                                 F = number of frames
                                 C = 3 (RGB channels)
                                 H, W = height and width
            fps: Frames per second of the video (optional, defaults to estimating from num_frames)
        
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
            
            # ==================================================================
            # Sample 2 frames that are time_gap_sec apart
            # ==================================================================
            # Estimate FPS if not provided (assume video is ~2-3 seconds)
            if fps is None:
                estimated_duration = 2.0  # seconds
                fps = num_frames / estimated_duration
            
            # Calculate frame indices for 2 frames with time_gap_sec apart
            frames_per_gap = int(fps * self.time_gap_sec)
            
            # Always sample: frame 0 and frame at +time_gap_sec
            frame_idx_0 = 0
            frame_idx_1 = min(frames_per_gap, num_frames - 1)
            
            # Handle edge case: if video is too short, use first and last frame
            if frame_idx_1 == frame_idx_0:
                frame_idx_1 = num_frames - 1
            
            sampled_indices = [frame_idx_0, frame_idx_1]
            
            print(f"[PSI] Extracting control from 2 frames: {sampled_indices} "
                  f"(out of {num_frames} total, time gap: {self.time_gap_sec}s, fps: {fps:.1f})")
            
            # Convert ONLY the 2 sampled frames to numpy arrays
            rgb_frames = []
            for idx in sampled_indices:
                frame = sample_frames[idx].permute(1, 2, 0)  # (H, W, C)
                frame_np = frame.cpu().numpy()
                rgb_frames.append(frame_np)
            
            # Create masks for the 2 frames
            unmask_indices_rgb = self.create_mask_indices(
                self.mask_ratio, 
                2,  # Only 2 frames
                seed=self.seed + b
            )
            
            # Generate prompt for PSI: "rgb0,rgb1->rgb1"
            prompt = "rgb0,rgb1->rgb1"
            
            # Time codes: 0ms and time_gap_sec*1000 ms
            time_codes = [0, int(self.time_gap_sec * 1000)]
            
            # Call parallel_extract_features (no gradients needed)
            with torch.no_grad():
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
            
            # Extract features from PSI predictor outputs
            # Output structure:
            # - hidden_state_coarse_2d_list: list of (1, 32, 32, 1, 4096) per frame
            # - embeddings_coarse_2d_list: list of (1, 32, 32, 1, 4096) per frame
            # We extracted features from 2 frames only
            
            if not isinstance(outputs, dict):
                raise ValueError(f"Expected dict output from PSI, got {type(outputs)}")
            
            # Extract the feature lists (contains features for 2 frames)
            hidden_states_list = outputs.get('hidden_state_coarse_2d_list', None)
            embeddings_list = outputs.get('embeddings_coarse_2d_list', None)
            
            if hidden_states_list is None or embeddings_list is None:
                raise ValueError(
                    f"PSI output missing required keys. "
                    f"Expected: 'hidden_state_coarse_2d_list', 'embeddings_coarse_2d_list'. "
                    f"Got: {list(outputs.keys())}"
                )
            
            if len(hidden_states_list) != 2 or len(embeddings_list) != 2:
                raise ValueError(
                    f"Expected 2 frames from PSI output, got {len(hidden_states_list)}"
                )
            
            # Process features from the 2 extracted frames
            extracted_frame_features = []
            for psi_frame_idx in range(2):  # Only 2 frames
                # Get features for this frame
                # Shape: (1, 32, 32, 1, 4096)
                hidden_state = hidden_states_list[psi_frame_idx]
                embeddings = embeddings_list[psi_frame_idx]
                
                # Remove extra dimensions: (1, 32, 32, 1, 4096) -> (32, 32, 4096)
                hidden_state = hidden_state.squeeze(0).squeeze(-2)  # (32, 32, 4096)
                embeddings = embeddings.squeeze(0).squeeze(-2)  # (32, 32, 4096)
                
                # Use ALL hidden states and ALL embeddings
                hidden_features = hidden_state  # (32, 32, 4096)
                embedding_features = embeddings  # (32, 32, 4096)
                
                # Combine hidden states and embeddings
                combined = torch.cat([hidden_features, embedding_features], dim=-1)  # (32, 32, 8192)
                
                extracted_frame_features.append(combined)
            
            # Stack the 2 extracted frames: (2, 32, 32, D)
            extracted_features = torch.stack(extracted_frame_features, dim=0)  # (2, H, W, D)
            
            # ==================================================================
            # Assign features to all frames in the video
            # ==================================================================
            # We extracted features from 2 frames, but need features for all num_frames
            # Strategy: Use F0 features for frame 0, F1 features for all other frames
            
            frame_features = []
            for target_frame_idx in range(num_frames):
                if target_frame_idx == 0:
                    # Frame 0: Use features from first extracted frame (F0)
                    frame_features.append(extracted_features[0])
                else:
                    # Frames 1-N: Use features from second extracted frame (F20)
                    frame_features.append(extracted_features[1])
            
            # Stack frames: (F, 32, 32, D)
            feature_tensor = torch.stack(frame_features, dim=0)
            F, H_psi, W_psi, D = feature_tensor.shape
            
            # Reshape to (D, F, 32, 32) for consistency with control embedding format
            feature_tensor = feature_tensor.permute(3, 0, 1, 2)  # (D, F, 32, 32)
            
            # Ensure correct device
            feature_tensor = feature_tensor.to(control_pixel_values.device)
            
            # Resize spatial dimensions to match VAE latent size if needed
            if (H_psi, W_psi) != (h_latent, w_latent):
                # Reshape for interpolation: (D, F, 32, 32) -> (D*F, 32, 32) -> (1, D*F, 32, 32)
                D, F, H_psi, W_psi = feature_tensor.shape
                feature_tensor = feature_tensor.reshape(D * F, H_psi, W_psi).unsqueeze(0)
                feature_tensor = torch.nn.functional.interpolate(
                    feature_tensor,
                    size=(h_latent, w_latent),
                    mode='bilinear',
                    align_corners=False
                )
                feature_tensor = feature_tensor.squeeze(0).reshape(D, F, h_latent, w_latent)
            
            # Create projection layer if needed (project from D=8192 to latent_channels=16)
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
                D, F, H, W = feature_tensor.shape
                feature_tensor = feature_tensor.permute(1, 0, 2, 3).reshape(F, D, H, W)  # (F, D, H, W)
                projected_frames = []
                for frame_feat in feature_tensor:
                    projected = self.projection(frame_feat.unsqueeze(0))  # (1, C_out, H, W)
                    projected_frames.append(projected.squeeze(0))
                feature_tensor = torch.stack(projected_frames, dim=1)  # (C_out, F, H, W)
            
            batch_embeddings.append(feature_tensor)
        
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

