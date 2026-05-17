"""Higher-level building blocks used by MetaField."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FiLM, PhysicsConv2d


class ResBlock(nn.Module):
    """CNN residual block with FiLM conditioning and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = PhysicsConv2d(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(min(16, out_channels), out_channels)
        self.film1 = FiLM(cond_dim, out_channels)

        self.conv2 = PhysicsConv2d(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(min(16, out_channels), out_channels)
        self.film2 = FiLM(cond_dim, out_channels)

        self.act = nn.SiLU()
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(min(16, out_channels), out_channels),
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.film1(out, cond)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.film2(out, cond)
        return self.act(out + residual)


class DownBlock(nn.Module):
    """Downsampling block kept behavior-compatible with the original script."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            PhysicsConv2d(channels, channels, stride=2),
            nn.GroupNorm(min(16, channels), channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GlobalAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, max_dist: int = 64, qk_scale: float | None = None, drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.max_dist = max_dist
        self.table_size = 2 * self.max_dist - 1
        self.rel_pos_table = nn.Parameter(torch.zeros(self.table_size, self.table_size, num_heads))
        nn.init.trunc_normal_(self.rel_pos_table, std=0.02)

    def get_rel_pos_bias(self, height: int, width: int) -> torch.Tensor:
        coords_h = torch.arange(height, device=self.rel_pos_table.device)
        coords_w = torch.arange(width, device=self.rel_pos_table.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.max_dist - 1
        relative_coords[:, :, 1] += self.max_dist - 1
        relative_coords[:, :, 0] = relative_coords[:, :, 0].clamp(0, self.table_size - 1)
        relative_coords[:, :, 1] = relative_coords[:, :, 1].clamp(0, self.table_size - 1)
        bias = self.rel_pos_table[relative_coords[:, :, 0], relative_coords[:, :, 1]]
        return bias.permute(2, 0, 1).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        num_tokens = height * width
        x_flat = x.view(batch, num_tokens, channels)
        qkv = self.qkv(x_flat).reshape(
            batch, num_tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.get_rel_pos_bias(height, width)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x_flat = (attn @ v).transpose(1, 2).reshape(batch, num_tokens, channels)
        x_flat = self.proj(x_flat)
        x_flat = self.drop(x_flat)
        return x_flat.view(batch, height, width, channels)


class TransformerLayer(nn.Module):
    """Transformer layer with GELU MLP, unchanged from the original behavior."""

    def __init__(self, dim: int, max_dist: int = 64, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GlobalAttentionBlock(dim, max_dist=max_dist, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x_in = x.permute(0, 2, 3, 1)
        x_in = x_in + self.attn(self.norm1(x_in))
        x_out = x_in + self.mlp(self.norm2(x_in))
        return x_out.permute(0, 3, 1, 2)
