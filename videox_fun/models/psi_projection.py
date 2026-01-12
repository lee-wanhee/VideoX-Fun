import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalOffsetEmbedding(nn.Module):
    """
    Temporal offset embedding using sinusoidal encoding → MLP.
    
    Encodes the "frames until target" offset using sinusoidal positional encoding,
    then projects to latent channels. This allows the model to understand temporal
    position when receiving propagated PSI control signals.
    
    For autoregressive rollout:
    - offset=0 means "I'm at the target frame"
    - offset=3 means "target frame is 3 latent frames ahead"
    """
    def __init__(self, n_output_channels, sin_dim=32, hidden_dim=64):
        super().__init__()
        self.n_output_channels = n_output_channels
        self.sin_dim = sin_dim
        
        # Project sinusoidal features to output channels
        self.offset_proj = nn.Sequential(
            nn.Linear(sin_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_output_channels),
        )
        
        self.init_weights()
    
    def init_weights(self):
        # Zero-init final layer for gradual learning
        nn.init.zeros_(self.offset_proj[-1].weight)
        nn.init.zeros_(self.offset_proj[-1].bias)
    
    def get_sinusoidal_emb(self, offset, device, dtype):
        """
        Generate sinusoidal embedding for offset value.
        
        Args:
            offset: int, the temporal offset (frames until target)
            device: torch device
            dtype: torch dtype
            
        Returns:
            Sinusoidal embedding of shape (1, sin_dim)
        """
        half_dim = self.sin_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb_scale)
        
        offset_tensor = torch.tensor([offset], device=device, dtype=dtype)
        emb = offset_tensor[:, None] * emb[None, :]  # (1, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (1, sin_dim)
        
        return emb
    
    def forward(self, offset, spatial_size, device, dtype):
        """
        Generate temporal offset embedding.
        
        Args:
            offset: int, frames until target (0 = at target)
            spatial_size: tuple (H, W) for spatial dimensions
            device: torch device
            dtype: torch dtype
            
        Returns:
            Temporal embedding of shape (1, n_output_channels, H, W)
        """
        # Get sinusoidal embedding
        sin_emb = self.get_sinusoidal_emb(offset, device, dtype)  # (1, sin_dim)
        
        # Project to output channels
        emb = self.offset_proj(sin_emb)  # (1, n_output_channels)
        
        # Reshape for spatial broadcasting: (1, C) -> (1, C, 1, 1) -> (1, C, H, W)
        H, W = spatial_size
        emb = emb.view(1, self.n_output_channels, 1, 1).expand(1, -1, H, W)
        
        return emb


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
    PSI feature projection with optional mask embedding and temporal offset encoding.
    
    Projects PSI semantic features to latent space (16 dim).
    - Coarse mode (use_all_tokens=False): 8192 input channels (4096 hidden + 4096 embeddings)
    - All tokens mode (use_all_tokens=True): 32768 input channels (4 * (4096 hidden + 4096 embeddings))
    
    Also includes:
    - Learnable mask embedding for non-PSI-controlled frames
    - Temporal offset embedding for propagated PSI control signals (optional)
    
    All are zero-initialized for gradual learning.
    """
    def __init__(self, n_input_channels, n_hidden_channels, n_output_channels, pdrop=0.0, eps=1e-6,
                 enable_temporal_embedding=False):
        super().__init__()
        
        self.n_output_channels = n_output_channels
        self.enable_temporal_embedding = enable_temporal_embedding

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
        
        # Temporal offset embedding for propagated PSI control (optional)
        # Only created when enable_temporal_embedding=True to avoid unused parameters in DDP
        if enable_temporal_embedding:
            self.temporal_offset_embedding = TemporalOffsetEmbedding(
                n_output_channels=n_output_channels,
                sin_dim=32,
                hidden_dim=64
            )
        else:
            self.temporal_offset_embedding = None

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.proj_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_down.weight)  # Zero init for gradual learning

        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        # norm weight already ones by default

    @property
    def dtype(self):
        """Return the dtype of the model (required by diffusers pipeline)."""
        return self.proj_up.weight.dtype
    
    @property
    def device(self):
        """Return the device of the model."""
        return self.proj_up.weight.device

    def get_temporal_offset_embedding(self, offset, spatial_size):
        """
        Get temporal offset embedding for a given offset value.
        
        Args:
            offset: int, frames until target (0 = at target frame)
            spatial_size: tuple (H, W) for spatial dimensions
            
        Returns:
            Temporal embedding of shape (1, n_output_channels, H, W)
            
        Raises:
            RuntimeError: If temporal embedding is not enabled
        """
        if self.temporal_offset_embedding is None:
            raise RuntimeError(
                "Temporal offset embedding is not enabled. "
                "Initialize PSIProjectionSwiGLU with enable_temporal_embedding=True."
            )
        return self.temporal_offset_embedding(
            offset=offset,
            spatial_size=spatial_size,
            device=self.device,
            dtype=self.dtype
        )

    def forward(self, x, return_mask_embedding=False, mask_spatial_size=None,
                temporal_offsets=None, temporal_spatial_size=None):
        """
        Project PSI features to latent space.
        
        Args:
            x: PSI features of shape (B, n_input_channels, H, W)
            return_mask_embedding: If True, also return the mask embedding for non-PSI frames
            mask_spatial_size: Optional tuple (H, W) for mask embedding size. 
                              If None, uses same size as output.
            temporal_offsets: Optional list of int offsets to compute temporal embeddings for.
                             Used for temporal-aware PSI propagation.
            temporal_spatial_size: Optional tuple (H, W) for temporal embeddings.
                                  Required if temporal_offsets is provided.
            
        Returns:
            If return_mask_embedding=False and temporal_offsets=None:
                Projected features of shape (B, n_output_channels, H, W)
            If return_mask_embedding=True or temporal_offsets is not None:
                Dict containing:
                - 'projected': Projected features (B, n_output_channels, H, W)
                - 'mask_embedding': (1, n_output_channels, H, W) if return_mask_embedding=True
                - 'temporal_embeddings': dict mapping offset -> (1, C, H, W) if temporal_offsets provided
        """
        x = self.norm(x)                 # (B, Cin, H, W)

        uv = self.proj_up(x)             # (B, 2H, H, W)
        u, v = uv.chunk(2, dim=1)        # each (B, H, H, W)

        x = u * self.act(v)              # SwiGLU

        x = self.proj_down(x)            # (B, Cout, H, W)
        x = self.drop(x)
        
        # If neither optional output is requested, return simple tensor (backward compatible)
        if not return_mask_embedding and temporal_offsets is None:
            return x
        
        # Build output dict
        output = {'projected': x}
        
        if return_mask_embedding:
            # Get spatial size for mask embedding
            if mask_spatial_size is not None:
                h, w = mask_spatial_size
            else:
                h, w = x.shape[-2], x.shape[-1]
            output['mask_embedding'] = self.mask_projection((h, w))
        
        if temporal_offsets is not None:
            # Compute temporal embeddings for all requested offsets
            # This goes through forward pass so DDP handles gradients correctly
            if self.temporal_offset_embedding is None:
                raise RuntimeError(
                    "Temporal offset embedding is not enabled but temporal_offsets was requested. "
                    "Initialize PSIProjectionSwiGLU with enable_temporal_embedding=True."
                )
            assert temporal_spatial_size is not None, "temporal_spatial_size required when temporal_offsets provided"
            temporal_embs = {}
            for offset in temporal_offsets:
                temporal_embs[offset] = self.temporal_offset_embedding(
                    offset=offset,
                    spatial_size=temporal_spatial_size,
                    device=self.device,
                    dtype=self.dtype
                )
            output['temporal_embeddings'] = temporal_embs
        
        return output
