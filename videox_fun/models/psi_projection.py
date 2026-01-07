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

class PSIProjectionSwiGLU(nn.Module):
    def __init__(self, n_input_channels, hidden_channels, n_output_channels, pdrop=0.0, eps=1e-6):
        super().__init__()

        # pre-norm on input channels
        self.norm = ChannelRMSNorm2d(n_input_channels, eps=eps, scale=True)
        # up-proj: Cin -> 2*H (value, gate)
        self.proj_up = nn.Conv2d(n_input_channels, 2 * hidden_channels, kernel_size=1)
        # down-proj: H -> Cout
        self.proj_down = nn.Conv2d(hidden_channels, n_output_channels, kernel_size=1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(pdrop)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.proj_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proj_down.weight, mean=0.0, std=0.0)

        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        # norm weight already ones by default

    def forward(self, x):
        x = self.norm(x)                 # (B, Cin, H, W)

        uv = self.proj_up(x)             # (B, 2H, H, W)
        u, v = uv.chunk(2, dim=1)        # each (B, H, H, W)

        x = u * self.act(v)              # SwiGLU

        x = self.proj_down(x)            # (B, Cout, H, W)
        x = self.drop(x)
        return x
