"""
PSI Control Feature Extractor

This module integrates PSIPredictor's parallel_extract_features functionality
into VideoX-Fun's control training pipeline.

The PSIPredictor extracts high-level semantic features from video frames that
can be used as control signals for video generation.
"""

import os
import sys
import time
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
    
    This is a PURE feature extractor with NO learnable parameters.
    It returns raw outputs that are processed downstream:
    - decoded_frames: PSI's predicted RGB frames (to be VAE-encoded externally)
    - semantic_features: Hidden states + embeddings (to be projected in transformer)
    
    Extracts control from 2 frames spaced 0.5 seconds apart.
    
    Input: control_pixel_values (B, F, C, H, W) - normalized to [-1, 1]
    Output: dict with:
        - 'decoded_frames': (B, C, H, W) - single RGB frame in [-1, 1] for VAE encoding
                            (VAE encode once, then repeat latent for efficiency)
        - 'semantic_features': (B, 8192, H_psi, W_psi) - raw PSI features for projection
                               H_psi, W_psi = 32, 32 (PSI patch grid)
    """
    
    # Class constant for PSI feature dimension
    PSI_FEATURE_DIM = 8192  # hidden_states (4096) + embeddings (4096)
    
    def __init__(
        self,
        model_name: str,
        quantizer_name: str,
        flow_quantizer_name: str = None,
        depth_quantizer_name: str = None,
        spatial_compression: int = 8,
        device: str = "cuda",
        mask_ratio_cond: float = 0.0,  # Mask ratio for conditioning frame (0.0 = fully visible)
        mask_ratio_pred: float = 0.9,  # Mask ratio for prediction frame (0.9 = 90% masked)
        prompt_template: str = "rgb0,rgb1->rgb1",  # Prompt for PSI
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 1000,
        **kwargs
    ):
        super().__init__()
        
        if not PSI_AVAILABLE:
            raise ImportError(
                "PSIPredictor is not available. Please install ccwm package.\n"
                "Make sure the ccwm module is in the parent directory or installed."
            )
        
        self.spatial_compression = spatial_compression
        self.device = device
        self.mask_ratio_cond = mask_ratio_cond
        self.mask_ratio_pred = mask_ratio_pred
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
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
        
        # Set PSI predictor to eval mode and freeze
        self.predictor.model.eval()
        for param in self.predictor.model.parameters():
            param.requires_grad = False
        
        print("✓ PSIPredictor loaded successfully (frozen, no learnable params)")
        
        # Register a dummy parameter for diffusers pipeline compatibility
        # This is needed because accelerate hooks require at least one parameter to determine device
        self._dummy_param = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)
    
    @property
    def dtype(self):
        """Return the dtype of the model (required by diffusers pipeline)."""
        return torch.bfloat16
        
    def create_mask_indices(self, mask_ratio: float, num_frames: int, seed: int = None) -> list:
        """
        Create unmask indices for each frame in a 32x32 patch grid.
        
        Args:
            mask_ratio: Ratio of patches to mask (0.0 = all visible, 1.0 = all masked)
            num_frames: Number of frames
            seed: Random seed. If None, uses current random state (for training diversity).
                  If provided, sets the seed for reproducible masks (for inference).
            
        Returns:
            List of unmask indices for each frame
        """
        import random
        
        # Only set seed if explicitly provided (for reproducible inference)
        # During training, seed=None uses current random state for diverse masks
        if seed is not None:
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
    
    def forward(self, control_pixel_values, fps=30.0, time_gap_sec=0.5, seed=None):
        """
        Extract raw features from input video frames using PSIPredictor.
        
        This is a PURE feature extractor - no learnable parameters, no VAE encoding.
        Returns raw outputs to be processed downstream (VAE encoding + projection).
        
        Now expects EXACTLY 2 frames (pre-sampled by dataloader) with the actual time gap.
        
        Args:
            control_pixel_values: (B, F, C, H, W) tensor, values in [-1, 1]
                                 B = batch size
                                 F = number of frames (should be 2: frame 0 and sampled second frame)
                                 C = 3 (RGB channels)
                                 H, W = height and width
            fps: Frames per second of the video (default: 30.0, for logging)
            time_gap_sec: Actual time gap between the two frames in seconds (default: 0.5s).
                         This is used for PSI's inter-frame feature extraction.
            seed: Random seed for reproducible masks (inference). If None, uses current
                  random state for diverse masks (training).
        
        Returns:
            dict with:
                'decoded_frames': (B, 2, C, H, W) - PSI decoded RGB frames in [-1, 1]
                                  One for each input frame (both frames decoded)
                'semantic_features': (B, 2, 8192, H_psi, W_psi) - raw PSI features per frame
                                     H_psi, W_psi = 32, 32 (PSI patch grid)
        """
        # Use provided seed for reproducible inference, or None for training diversity
        # When seed=None, masks will vary each forward pass (good for training)
        # When seed is provided, masks are reproducible (good for inference)
        current_seed = seed  # Keep as None if not provided
        batch_size, num_frames, channels, height, width = control_pixel_values.shape
        
        # Expect exactly 2 frames from dataloader
        assert num_frames == 2, f"Expected 2 frames from dataloader, got {num_frames}"
        
        # Convert from [-1, 1] to [0, 1] for PSIPredictor
        # PSI expects images in [0, 1] range
        rgb_frames_01 = (control_pixel_values + 1.0) / 2.0
        
        # Process each sample in the batch
        batch_decoded_frames = []
        batch_semantic_features = []
        
        for b in range(batch_size):
            # Get frames for this sample: (F, C, H, W) where F=2
            sample_frames = rgb_frames_01[b]
            
            # ==================================================================
            # Process both frames (pre-sampled by dataloader)
            # ==================================================================
            # Frame indices are 0 and 1 (corresponding to frame 0 and sampled second frame)
            sampled_indices = [0, 1]
            
            print(f"[PSI GPU={self.device}] Extracting from 2 frames "
                  f"(time gap: {time_gap_sec:.3f}s, fps: {fps})")
            
            psi_start_time = time.time()
            
            # Convert ONLY the 2 sampled frames to numpy arrays
            # PSI expects 512x512 images (32x32 patch grid with 16x16 patches)
            PSI_IMAGE_SIZE = 512
            rgb_frames = []
            for idx in sampled_indices:
                frame = sample_frames[idx]  # (C, H, W)
                # Resize to PSI's expected resolution (512x512)
                if frame.shape[1] != PSI_IMAGE_SIZE or frame.shape[2] != PSI_IMAGE_SIZE:
                    frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0),  # Add batch dim
                        size=(PSI_IMAGE_SIZE, PSI_IMAGE_SIZE),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dim
                frame = frame.permute(1, 2, 0)  # (H, W, C)
                # Convert to float32 first since numpy doesn't support bfloat16
                frame_np = frame.float().cpu().numpy()
                rgb_frames.append(frame_np)
            
            # Create masks for the 2 frames
            # If seed is provided (inference), use it for reproducible masks
            # If seed is None (training), use current random state for diverse masks
            frame0_seed = (current_seed + b) if current_seed is not None else None
            frame1_seed = (current_seed + b + 1) if current_seed is not None else None
            
            unmask_indices_frame0 = self.create_mask_indices(
                mask_ratio=self.mask_ratio_cond,
                num_frames=1,
                seed=frame0_seed
            )
            unmask_indices_frame1 = self.create_mask_indices(
                mask_ratio=self.mask_ratio_pred,
                num_frames=1, 
                seed=frame1_seed
            )
            unmask_indices_rgb = unmask_indices_frame0 + unmask_indices_frame1
            
            # Generate prompt for PSI: "rgb0,rgb1->rgb1"
            prompt = "rgb0,rgb1->rgb1"
            
            # Time codes: 0ms and time_gap_sec*1000 ms
            time_codes = [0, int(time_gap_sec * 1000)]
            
            # Call parallel_extract_features (no gradients needed)
            # For PSI predictor seed: use provided seed if available, otherwise None for training diversity
            # When seed=None, PSI predictor uses different random outputs each time (good for training)
            psi_predictor_seed = (current_seed + b) if current_seed is not None else None
            
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
                    seed=psi_predictor_seed,
                    num_seq_patches=32,
                    temp=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    out_dir=None,
                )
            
            psi_elapsed = time.time() - psi_start_time
            print(f"[PSI GPU={self.device}] Feature extraction took {psi_elapsed:.2f}s")
            
            if not isinstance(outputs, dict):
                raise ValueError(f"Expected dict output from PSI, got {type(outputs)}")
            
            # Extract decoded frames and semantic features from PSI output
            decoded_frames_list = outputs.get('decoded_frames_list', None)
            hidden_states_list = outputs.get('hidden_state_coarse_2d_list', None)
            embeddings_list = outputs.get('embeddings_coarse_2d_list', None)
            
            if decoded_frames_list is None:
                raise ValueError(
                    f"PSI output missing 'decoded_frames_list'. Got: {list(outputs.keys())}"
                )
            
            # ==================================================================
            # Part 1: Process BOTH decoded frames (frame 0 and sampled second frame)
            # ==================================================================
            if len(decoded_frames_list) < 2:
                # Debug logging only on failure
                print(f"[PSI FAIL] Expected 2 decoded frames, got {len(decoded_frames_list)}")
                print(f"[PSI FAIL] Output keys: {list(outputs.keys())}")
                print(f"[PSI FAIL] Input: rgb_frames={len(rgb_frames)}, time_codes={time_codes}, prompt={prompt}")
                for i, frame in enumerate(decoded_frames_list):
                    frame_info = f"type={type(frame).__name__}"
                    if hasattr(frame, 'shape'):
                        frame_info += f", shape={frame.shape}"
                    elif hasattr(frame, 'size'):
                        frame_info += f", size={frame.size}"
                    print(f"[PSI FAIL] decoded_frames_list[{i}]: {frame_info}")
                return None
            
            # Process both frames
            from PIL import Image as PILImage
            sample_decoded_frames = []
            for frame_idx in range(2):
                decoded_frame = decoded_frames_list[frame_idx]
                
                # Handle different input types (PIL Image, numpy array, or tensor)
                if isinstance(decoded_frame, PILImage.Image):
                    # Convert PIL Image to numpy array
                    decoded_frame = np.array(decoded_frame).astype(np.float32) / 255.0  # [0, 255] -> [0, 1]
                
                if isinstance(decoded_frame, np.ndarray):
                    decoded_frame = torch.from_numpy(decoded_frame)
                
                # Convert to (C, H, W) and normalize to [-1, 1] for VAE
                if decoded_frame.dim() == 3 and decoded_frame.shape[-1] == 3:
                    decoded_frame = decoded_frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                decoded_frame = decoded_frame.float()
                if decoded_frame.max() <= 1.0:
                    decoded_frame = decoded_frame * 2.0 - 1.0  # [0,1] -> [-1,1]
                
                # Resize to match input size
                if decoded_frame.shape[1] != height or decoded_frame.shape[2] != width:
                    decoded_frame = torch.nn.functional.interpolate(
                        decoded_frame.unsqueeze(0),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                decoded_frame = decoded_frame.to(control_pixel_values.device, control_pixel_values.dtype)
                sample_decoded_frames.append(decoded_frame)
            
            # Stack frames: (2, C, H, W)
            sample_decoded_frames = torch.stack(sample_decoded_frames, dim=0)
            batch_decoded_frames.append(sample_decoded_frames)
            
            # ==================================================================
            # Part 2: Extract raw semantic features for BOTH frames
            # ==================================================================
            sample_semantic_features = []
            if hidden_states_list is not None and embeddings_list is not None:
                for frame_idx in range(2):
                    hidden_state = hidden_states_list[frame_idx]
                    embeddings = embeddings_list[frame_idx]
                    
                    # Remove extra dimensions: (1, 32, 32, 1, 4096) -> (32, 32, 4096)
                    hidden_state = hidden_state.squeeze(0).squeeze(-2)
                    embeddings = embeddings.squeeze(0).squeeze(-2)
                    
                    # Combine hidden states and embeddings: (32, 32, 8192)
                    combined = torch.cat([hidden_state, embeddings], dim=-1)
                    
                    # Reshape: (32, 32, 8192) -> (8192, 32, 32)
                    semantic_features = combined.permute(2, 0, 1)
                    semantic_features = semantic_features.to(control_pixel_values.device, control_pixel_values.dtype)
                    sample_semantic_features.append(semantic_features)
            else:
                # No semantic features available, create zeros for both frames
                for _ in range(2):
                    semantic_features = torch.zeros(
                        self.PSI_FEATURE_DIM, 32, 32,
                        device=control_pixel_values.device,
                        dtype=control_pixel_values.dtype
                    )
                    sample_semantic_features.append(semantic_features)
            
            # Stack: (2, 8192, 32, 32)
            sample_semantic_features = torch.stack(sample_semantic_features, dim=0)
            batch_semantic_features.append(sample_semantic_features)
        
        # Stack batch
        # decoded_frames: (B, 2, C, H, W) - both frames per sample
        decoded_frames = torch.stack(batch_decoded_frames, dim=0)
        # semantic_features: (B, 2, 8192, 32, 32) - features for both frames
        semantic_features = torch.stack(batch_semantic_features, dim=0)
        
        print(f"[PSI] Output decoded_frames: {decoded_frames.shape} (2 frames)")
        print(f"[PSI] Output semantic_features: {semantic_features.shape} (2 frames)")
        
        return {
            'decoded_frames': decoded_frames,  # (B, 2, C, H, W) - both frames
            'semantic_features': semantic_features,  # (B, 2, 8192, 32, 32) - features for both frames
        }
    
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
            outputs = extractor(control_input)
        
        decoded_frames = outputs['decoded_frames']
        semantic_features = outputs['semantic_features']
        
        print(f"\nResults:")
        print(f"  Input shape:           {control_input.shape}")
        print(f"  decoded_frames shape:  {decoded_frames.shape}")
        print(f"  Expected:              ({batch_size}, 2, 3, {height}, {width})")
        print(f"  decoded_frames range:  [{decoded_frames.min().item():.4f}, {decoded_frames.max().item():.4f}]")
        print(f"  semantic_features shape: {semantic_features.shape}")
        print(f"  Expected:                ({batch_size}, 2, {PSIControlFeatureExtractor.PSI_FEATURE_DIM}, 32, 32)")
        
        print("\n" + "=" * 80)
        print("✓ Test completed successfully!")
        print("=" * 80)
        print("\nNote: decoded_frames should be VAE-encoded and semantic_features should be")
        print("projected in WanTransformer3DModel to create the final control signal.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


