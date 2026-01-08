import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelRMSNorm2d(nn.Module):
    """
    RMSNorm over channel dimension for (B, C, H, W).
    x_norm = x / rms(x, dim=C)
    optional affine: weight (C,), bias (C,)
    """
    def __init__(self, num_channels: int, eps: float = 1e-6, scale: bool = True):
        super().__init__()
        self.eps = eps
        self.scale = scale
        self.weight = nn.Parameter(torch.ones(num_channels)) if scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # rms over channel: sqrt(mean(x^2, dim=1))
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.weight is not None:
            w = self.weight.view(1, -1, 1, 1)
            x = x * w
        return x


class MaskEmbeddingProjection(nn.Module):
    """
    Learnable mask embedding for frames without PSI control.
    
    This embedding is added to masked (non-PSI-controlled) latent frames.
    Initialized to zero so training starts with no effect, gradually learning
    what information to add for masked frames.
    
    Uses the same SwiGLU architecture as PSI projection for consistency,
    but with a learnable input embedding instead of PSI features.
    """
    def __init__(self, n_hidden_channels, n_output_channels, pdrop=0.0):
        super().__init__()
        
        self.n_output_channels = n_output_channels
        self.n_hidden_channels = n_hidden_channels
        
        # Learnable embedding (will be expanded spatially)
        # Shape: (1, n_hidden_channels, 1, 1) - broadcast to any spatial size
        self.mask_embedding = nn.Parameter(torch.zeros(1, n_hidden_channels, 1, 1))
        
        # Projection from hidden to output (same as PSI projection's down-proj)
        self.proj_down = nn.Conv2d(n_hidden_channels, n_output_channels, kernel_size=1)
        self.drop = nn.Dropout(pdrop)
        
        self.init_weights()
    
    def init_weights(self):
        # Zero init for gradual learning
        nn.init.zeros_(self.mask_embedding)
        nn.init.zeros_(self.proj_down.weight)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)
    
    def forward(self, spatial_size):
        """
        Generate mask embedding for given spatial size.
        
        Args:
            spatial_size: tuple (H, W) for the spatial dimensions
            
        Returns:
            Mask embedding of shape (1, n_output_channels, H, W)
        """
        h, w = spatial_size
        
        # Expand embedding to spatial size
        # (1, n_hidden_channels, 1, 1) -> (1, n_hidden_channels, H, W)
        expanded = self.mask_embedding.expand(1, self.n_hidden_channels, h, w)
        
        # Project to output channels
        out = self.proj_down(expanded)  # (1, n_output_channels, H, W)
        out = self.drop(out)
        
        return out


class PSIProjectionSwiGLU(nn.Module):
    """
    PSI feature projection with optional mask embedding.
    
    Projects PSI semantic features (8192 dim) to latent space (16 dim).
    Also includes a learnable mask embedding for non-PSI-controlled frames.
    Both are zero-initialized for gradual learning.
    """
    def __init__(self, n_input_channels, n_hidden_channels, n_output_channels, pdrop=0.0, eps=1e-6):
        super().__init__()
        
        self.n_output_channels = n_output_channels

        # PSI feature projection (pre-norm on input channels)
        self.norm = ChannelRMSNorm2d(n_input_channels, eps=eps, scale=True)
        # up-proj: Cin -> 2*H (value, gate)
        self.proj_up = nn.Conv2d(n_input_channels, 2 * n_hidden_channels, kernel_size=1)
        # down-proj: H -> Cout
        self.proj_down = nn.Conv2d(n_hidden_channels, n_output_channels, kernel_size=1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(pdrop)
        
        # Mask embedding for non-PSI-controlled frames
        self.mask_projection = MaskEmbeddingProjection(
            n_hidden_channels=n_hidden_channels,
            n_output_channels=n_output_channels,
            pdrop=pdrop
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.proj_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_down.weight)  # Zero init for gradual learning

        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        # norm weight already ones by default

    def forward(self, x):
        """
        Project PSI features to latent space.
        
        Args:
            x: PSI features of shape (B, n_input_channels, H, W)
            
        Returns:
            Projected features of shape (B, n_output_channels, H, W)
        """
        x = self.norm(x)                 # (B, Cin, H, W)

        uv = self.proj_up(x)             # (B, 2H, H, W)
        u, v = uv.chunk(2, dim=1)        # each (B, H, H, W)

        x = u * self.act(v)              # SwiGLU

        x = self.proj_down(x)            # (B, Cout, H, W)
        x = self.drop(x)
        return x
    
    def get_mask_embedding(self, spatial_size):
        """
        Get mask embedding for non-PSI-controlled frames.
        
        Args:
            spatial_size: tuple (H, W) for the spatial dimensions
            
        Returns:
            Mask embedding of shape (1, n_output_channels, H, W)
        """
        return self.mask_projection(spatial_size)
