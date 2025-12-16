"""
Custom Control Feature Extractor Template

This file provides a template for implementing your custom control feature extraction model.
Replace the dummy implementation with your actual model.

The control extractor takes video frames as input and outputs latent embeddings
that will be used to condition the video generation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomControlFeatureExtractor(nn.Module):
    """
    Custom control feature extractor for video generation.
    
    This is a template - replace the implementation with your actual model.
    
    Input: control_pixel_values (B, F, C, H, W) - normalized to [-1, 1]
           where B=batch, F=frames, C=channels(3), H=height, W=width
    
    Output: control_embeddings (B, C_out, F, H_latent, W_latent)
            where C_out=latent_channels (typically 16)
                  H_latent = H // spatial_compression (typically H // 8)
                  W_latent = W // spatial_compression (typically W // 8)
    
    The output dimensions should match the VAE latent space dimensions.
    """
    
    def __init__(
        self, 
        latent_channels=16,
        temporal_compression=4,
        spatial_compression=8,
        **kwargs
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        
        # ====================================================================
        # TODO: Initialize your actual control feature extraction model here
        # ====================================================================
        
        # Example: Simple CNN-based encoder (replace with your model)
        self.encoder = nn.Sequential(
            # Initial conv: 3 channels -> 64
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # H/2, W/2
            nn.ReLU(inplace=True),
            
            # Downsample further
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/4, W/4
            nn.ReLU(inplace=True),
            
            # Final layer to latent space
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=2, padding=1),  # H/8, W/8
        )
        
        # Example: You might want to add temporal processing
        # self.temporal_encoder = nn.LSTM(...)
        # self.temporal_conv = nn.Conv3d(...)
        
    def forward(self, control_pixel_values):
        """
        Extract control features from input video frames.
        
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
                               F = number of frames (same as input)
                               H_latent = H // spatial_compression
                               W_latent = W // spatial_compression
        """
        batch_size, num_frames, channels, height, width = control_pixel_values.shape
        
        # ====================================================================
        # TODO: Implement your actual feature extraction logic here
        # ====================================================================
        
        # Calculate output spatial dimensions
        h_latent = height // self.spatial_compression
        w_latent = width // self.spatial_compression
        
        # Process each frame independently (example implementation)
        frame_embeddings = []
        for i in range(num_frames):
            frame = control_pixel_values[:, i]  # (B, C, H, W)
            embedding = self.encoder(frame)  # (B, C_out, H_latent, W_latent)
            frame_embeddings.append(embedding)
        
        # Stack frame embeddings: (B, C_out, F, H_latent, W_latent)
        control_embeddings = torch.stack(frame_embeddings, dim=2)
        
        # Alternative: Process spatially then temporally
        # Reshape: (B*F, C, H, W)
        # frames_flat = control_pixel_values.reshape(-1, channels, height, width)
        # embeddings_flat = self.encoder(frames_flat)  # (B*F, C_out, H_latent, W_latent)
        # control_embeddings = embeddings_flat.reshape(
        #     batch_size, num_frames, self.latent_channels, h_latent, w_latent
        # ).permute(0, 2, 1, 3, 4)  # (B, C_out, F, H_latent, W_latent)
        
        return control_embeddings
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """
        Load the control feature extractor from a pretrained checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
            **kwargs: Additional arguments for model initialization
        
        Returns:
            Loaded model instance
        """
        import os
        
        # Create model instance
        model = cls(**kwargs)
        
        # Load weights if checkpoint exists
        if model_path is not None and os.path.exists(model_path):
            print(f"Loading control extractor from: {model_path}")
            
            if model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
            
            # Handle state_dict wrapped in 'state_dict' key
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load weights
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
            print("Control extractor loaded successfully!")
        else:
            print(f"No checkpoint found at {model_path}, using randomly initialized model")
        
        return model


# ============================================================================
# Example: Advanced Control Extractor with Temporal Modeling
# ============================================================================

class AdvancedControlExtractor(nn.Module):
    """
    Example of a more advanced control extractor with temporal modeling.
    """
    
    def __init__(
        self, 
        latent_channels=16,
        temporal_compression=4,
        spatial_compression=8,
        hidden_dim=256,
        **kwargs
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        
        # Spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, hidden_dim, 3, 2, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Temporal encoder (3D convolutions)
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Final projection to latent channels
        self.output_proj = nn.Conv3d(hidden_dim, latent_channels, kernel_size=1)
        
    def forward(self, control_pixel_values):
        B, F, C, H, W = control_pixel_values.shape
        
        # Process spatially: (B*F, C, H, W) -> (B*F, hidden_dim, H/8, W/8)
        frames_flat = control_pixel_values.reshape(-1, C, H, W)
        spatial_features = self.spatial_encoder(frames_flat)
        
        # Reshape for temporal processing: (B, hidden_dim, F, H/8, W/8)
        _, hidden_dim, h, w = spatial_features.shape
        temporal_input = spatial_features.reshape(B, F, hidden_dim, h, w)
        temporal_input = temporal_input.permute(0, 2, 1, 3, 4)
        
        # Process temporally
        temporal_features = self.temporal_encoder(temporal_input)
        
        # Project to latent channels
        control_embeddings = self.output_proj(temporal_features)
        
        return control_embeddings
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        return CustomControlFeatureExtractor.from_pretrained.__func__(cls, model_path, **kwargs)


if __name__ == "__main__":
    # Test the custom control extractor
    print("Testing Custom Control Feature Extractor...")
    
    # Create dummy input
    batch_size = 2
    num_frames = 17
    height, width = 512, 512
    control_input = torch.randn(batch_size, num_frames, 3, height, width)
    control_input = (control_input - 0.5) * 2  # Normalize to [-1, 1]
    
    # Test basic extractor
    model = CustomControlFeatureExtractor(
        latent_channels=16,
        spatial_compression=8,
    )
    
    with torch.no_grad():
        output = model(control_input)
    
    print(f"Input shape: {control_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 16, {num_frames}, {height//8}, {width//8})")
    
    assert output.shape == (batch_size, 16, num_frames, height//8, width//8), \
        f"Output shape mismatch! Got {output.shape}"
    
    print("✓ Basic extractor test passed!")
    
    # Test advanced extractor
    model_advanced = AdvancedControlExtractor(
        latent_channels=16,
        spatial_compression=8,
    )
    
    with torch.no_grad():
        output_advanced = model_advanced(control_input)
    
    print(f"\nAdvanced extractor output shape: {output_advanced.shape}")
    assert output_advanced.shape == (batch_size, 16, num_frames, height//8, width//8), \
        f"Advanced output shape mismatch! Got {output_advanced.shape}"
    
    print("✓ Advanced extractor test passed!")
    print("\nAll tests passed! You can now implement your actual control extraction logic.")

