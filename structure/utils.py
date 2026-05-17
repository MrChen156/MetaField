"""Geometry/grid utilities shared by search and benchmarks."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

PAD_MULTIPLE = 32
GRID_RES = 3.0
N_Z_CELLS = 268
N_MAT_SLOTS = 70
FLOW_CODE = 0
SUB_CODE = 1


def get_padded_size(height: int, width: int, multiple: int = PAD_MULTIPLE) -> tuple[int, int]:
    return ((height - 1) // multiple + 1) * multiple, ((width - 1) // multiple + 1) * multiple


def torch_pad_center_circX_repZ(data, target_height: int, target_width: int):
    tensor = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    shape = list(tensor.shape)
    height, width = shape[-2], shape[-1]
    diff_h, diff_w = target_height - height, target_width - width
    pad_t, pad_l = diff_h // 2, diff_w // 2
    pad_b, pad_r = diff_h - pad_t, diff_w - pad_l
    top, bottom = pad_t, pad_t + height
    left, right = pad_l, pad_l + width

    output_shape = shape[:-2] + [target_height, target_width]
    out = torch.zeros(output_shape, dtype=tensor.dtype)
    out[..., top:bottom, left:right] = tensor

    if pad_t > 0:
        src = out[..., top : top + 1, left:right]
        expand_shape = list(src.shape)
        expand_shape[-2] = pad_t
        out[..., :top, left:right] = src.expand(expand_shape)
    if pad_b > 0:
        src = out[..., bottom - 1 : bottom, left:right]
        expand_shape = list(src.shape)
        expand_shape[-2] = pad_b
        out[..., bottom:, left:right] = src.expand(expand_shape)

    strip = out[..., :, left:right]
    if pad_l > 0:
        out[..., :, :left] = strip[..., :, -pad_l:]
    if pad_r > 0:
        out[..., :, right:] = strip[..., :, :pad_r]

    return out.numpy() if isinstance(data, np.ndarray) else out


def pad_coords_linear(arr: np.ndarray, target_len: int) -> np.ndarray:
    length = len(arr)
    diff = target_len - length
    pad_l = diff // 2
    pad_r = diff - pad_l
    dx = (arr[1] - arr[0]) if len(arr) > 1 else 1.0
    prefix = arr[0] - dx * np.arange(pad_l, 0, -1)
    suffix = arr[-1] + dx * np.arange(1, pad_r + 1)
    return np.concatenate([prefix, arr, suffix]).astype(np.float32)


def compute_sdf(geom: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(geom.astype(np.float32))
    grad_mag = np.sqrt(dx**2 + dy**2)
    is_edge = grad_mag > 1e-3
    if is_edge.sum() == 0:
        return np.ones_like(geom, dtype=np.float32)
    dist = distance_transform_edt(~is_edge)
    return (dist / (geom.shape[1] / 4.0)).astype(np.float32)
