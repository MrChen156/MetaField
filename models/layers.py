"""Low-level layers shared across training, GA, and benchmarks."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConv2d(nn.Module):
    """
    Mixed padding strategy matching the original implementation:
    - X / width: circular padding for periodic boundary conditions
    - Z / height: replicate padding for top-bottom boundary behavior
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )
        self.pad_size = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_size, self.pad_size, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad_size, self.pad_size), mode="replicate")
        return self.conv(x)


class FourierFeature(nn.Module):
    """Fourier feature mapping for the condition vector."""

    def __init__(self, input_dim: int = 3, mapping_size: int = 64, scale: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * float(scale))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        # `register_buffer` is runtime-correct here; cast keeps static type checkers
        # from treating the buffer as `Tensor | Module`.
        projection_matrix = cast(torch.Tensor, self.B)
        x_proj = 2 * math.pi * torch.matmul(v, projection_matrix)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation conditioned on the global design vector."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.scale(cond).unsqueeze(2).unsqueeze(3)
        beta = self.shift(cond).unsqueeze(2).unsqueeze(3)
        return x * (gamma + 1.0) + beta
