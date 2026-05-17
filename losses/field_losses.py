"""Field-domain losses kept behavior-compatible with the original training code."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_sobel_kernels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).view(1, 1, 3, 3).float()
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).view(1, 1, 3, 3).float()
    return kx, ky


def compute_training_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    field_norm: float,
    grad_weight: float,
    kx: torch.Tensor,
    ky: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target_norm = target / field_norm
    diff = pred - target_norm

    amp = torch.sqrt(target_norm[:, 0] ** 2 + target_norm[:, 1] ** 2 + 1e-8).unsqueeze(1)
    pix_weight = 1.0 + 5.0 * amp
    loss_mse = ((diff**2) * pix_weight * mask).sum() / (mask.sum() * 6 + 1e-8)

    channels = pred.shape[1]
    g_pred_x = F.conv2d(pred, kx.expand(channels, -1, -1, -1), padding=1, groups=channels)
    g_pred_y = F.conv2d(pred, ky.expand(channels, -1, -1, -1), padding=1, groups=channels)
    g_gt_x = F.conv2d(target_norm, kx.expand(channels, -1, -1, -1), padding=1, groups=channels)
    g_gt_y = F.conv2d(target_norm, ky.expand(channels, -1, -1, -1), padding=1, groups=channels)
    loss_grad = (torch.abs(g_pred_x - g_gt_x) + torch.abs(g_pred_y - g_gt_y)) * mask
    loss_grad = loss_grad.sum() / (mask.sum() * channels + 1e-8)

    loss_total = loss_mse + grad_weight * loss_grad
    return {"mse": loss_mse, "grad": loss_grad, "total": loss_total}


def compute_validation_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, field_norm: float) -> torch.Tensor:
    target_norm = target / field_norm
    return ((pred - target_norm) ** 2 * mask).sum() / (mask.sum() * 6 + 1e-8)
