from __future__ import annotations

import torch

from models import MetaField


def test_metafield_output_shape():
    model = MetaField(
        in_channels=5,
        out_channels=6,
        cond_channels=3,
        base_channels=16,
        cond_embed_dim=64,
        heads=4,
        max_dist=32,
        transformer_depth=2,
    )
    x = torch.randn(2, 5, 64, 64)
    cond = torch.randn(2, 3)
    y = model(x, cond)
    assert y.shape == (2, 6, 64, 64)
