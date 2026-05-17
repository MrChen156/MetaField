"""MetaField model definition shared across the whole project."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DownBlock, ResBlock, TransformerLayer
from .layers import FourierFeature


class MetaField(nn.Module):
    """
    MetaField keeps the original MetaGlobalUNet architecture unchanged.
    The class is renamed for the cleaned repository layout, while a
    backward-compatible alias is kept below for old checkpoints/scripts.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 6,
        cond_channels: int = 3,
        base_channels: int = 96,
        cond_embed_dim: int = 256,
        heads: int = 8,
        max_dist: int = 48,
        transformer_depth: int = 8,
    ):
        super().__init__()
        self.fourier = FourierFeature(input_dim=cond_channels, mapping_size=64)
        self.cond_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, cond_embed_dim),
            nn.GELU(),
        )

        self.inc = ResBlock(in_channels, base_channels, cond_embed_dim)

        self.down1 = DownBlock(base_channels)
        self.enc1 = ResBlock(base_channels, base_channels * 2, cond_embed_dim)

        self.down2 = DownBlock(base_channels * 2)
        self.enc2 = ResBlock(base_channels * 2, base_channels * 4, cond_embed_dim)

        self.down3 = DownBlock(base_channels * 4)
        self.bottle_conv = nn.Conv2d(base_channels * 4, base_channels * 8, 1)
        self.transformer = nn.Sequential(
            *[
                TransformerLayer(dim=base_channels * 8, max_dist=max_dist, num_heads=heads)
                for _ in range(transformer_depth)
            ]
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ResBlock(base_channels * 8 + base_channels * 4, base_channels * 4, cond_embed_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, cond_embed_dim)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ResBlock(base_channels * 2 + base_channels, base_channels, cond_embed_dim)

        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        cond_embed = self.cond_mlp(self.fourier(cond_vec))

        x0 = self.inc(x, cond_embed)
        x1 = self.enc1(self.down1(x0), cond_embed)
        x2 = self.enc2(self.down2(x1), cond_embed)

        bottleneck = self.transformer(self.bottle_conv(self.down3(x2)))

        d1 = self.dec1(torch.cat([self.up1(bottleneck), x2], dim=1), cond_embed)
        d2 = self.dec2(torch.cat([self.up2(d1), x1], dim=1), cond_embed)
        d3 = self.dec3(torch.cat([self.up3(d2), x0], dim=1), cond_embed)
        return self.outc(d3)


MetaGlobalUNet = MetaField
