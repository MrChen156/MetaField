"""Training loop and validation loop."""

from __future__ import annotations

import contextlib
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast

from losses.field_losses import build_sobel_kernels, compute_training_losses, compute_validation_mse

from .checkpoint import resume_or_warmstart, save_training_checkpoint
from .ddp import is_master
from .logging_utils import append_history_row, get_total_memory_gb, plot_training_history


@dataclass
class TrainerConfig:
    save_dir: str
    pretrain_path: str = ""
    epochs: int = 2000
    batch_size: int = 36
    grad_accum_steps: int = 4
    lr: float = 6e-4
    final_div_factor: float = 1e2
    cache_clear_interval: int = 100
    field_norm: float = 10.0
    grad_weight: float = 1e-4
    plot_interval: int = 20


def _all_reduce_if_needed(stats: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats


def run_training(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    val_loader,
    train_sampler,
    config: TrainerConfig,
    device: torch.device,
    world_size: int = 1,
) -> None:
    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
    start_epoch, best_loss = resume_or_warmstart(
        save_dir=config.save_dir,
        pretrain_path=config.pretrain_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        steps_per_epoch=steps_per_epoch,
    )

    if is_master():
        current_lr = scheduler.get_last_lr()[0]
        print(f"Optimizer steps/epoch: {steps_per_epoch}")
        print(f"Total optimizer steps: {steps_per_epoch * config.epochs}")
        print(f"Effective batch size: {config.batch_size * config.grad_accum_steps * world_size}")
        if start_epoch > 1:
            print(f"Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.4e}, lr={current_lr:.2e}")
        elif config.pretrain_path and Path(config.pretrain_path).exists():
            print(f"Warm-start from pretrained weights: {config.pretrain_path}")
        else:
            print("Training from scratch.")

    kx, ky = build_sobel_kernels(device)
    autocast_ctx = "cuda" if device.type == "cuda" else device.type

    for epoch in range(start_epoch, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        stats = torch.zeros(3, device=device)

        if device.type == "cuda" and epoch % config.cache_clear_interval == 1:
            torch.cuda.empty_cache()
            gc.collect()
            if is_master():
                print(f"Cache cleared at epoch {epoch}")

        optimizer.zero_grad()

        for batch_idx, (x, y, cond, mask) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            is_sync_step = ((batch_idx + 1) % config.grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader))

            if (batch_idx + 1) == len(train_loader) and (len(train_loader) % config.grad_accum_steps != 0):
                current_accum_steps = len(train_loader) % config.grad_accum_steps
            else:
                accum_start = (batch_idx // config.grad_accum_steps) * config.grad_accum_steps
                remaining = len(train_loader) - accum_start
                current_accum_steps = min(config.grad_accum_steps, remaining)

            ddp_ctx = contextlib.nullcontext() if is_sync_step or not hasattr(model, "no_sync") else model.no_sync()
            amp_enabled = device.type == "cuda"

            with ddp_ctx:
                with autocast(device_type=autocast_ctx, enabled=amp_enabled):
                    pred = model(x, cond)
                    losses = compute_training_losses(
                        pred=pred,
                        target=y,
                        mask=mask,
                        field_norm=config.field_norm,
                        grad_weight=config.grad_weight,
                        kx=kx,
                        ky=ky,
                    )
                    loss_scaled = losses["total"] / current_accum_steps
                scaler.scale(loss_scaled).backward()

            stats[0] += losses["mse"].detach()
            stats[1] += losses["grad"].detach()
            stats[2] += losses["total"].detach()

            if is_sync_step:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        stats = _all_reduce_if_needed(stats)
        stats /= (world_size * len(train_loader))
        avg_mse, avg_grad, avg_total = stats[0].item(), stats[1].item(), stats[2].item()

        model.eval()
        val_loss = torch.zeros(1, device=device)
        with torch.no_grad():
            for x, y, cond, mask in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with autocast(device_type=autocast_ctx, enabled=amp_enabled):
                    pred = model(x, cond)
                    val_loss += compute_validation_mse(pred, y, mask, config.field_norm)

        val_loss = _all_reduce_if_needed(val_loss)
        val_loss /= (world_size * len(val_loader))
        avg_val = val_loss.item()

        if is_master():
            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[Ep {epoch:04d}/{config.epochs}] "
                f"T_MSE: {avg_mse:.4e} | T_Grad: {avg_grad:.4e} | "
                f"Val: {avg_val:.4e} | RAM: {get_total_memory_gb():.1f}GB | {epoch_time:.1f}s",
                flush=True,
            )
            append_history_row(
                Path(config.save_dir) / "history.csv",
                [epoch, avg_mse, avg_grad, avg_total, avg_val, current_lr, epoch_time],
            )
            is_best = avg_val < best_loss
            if is_best:
                best_loss = avg_val
            save_training_checkpoint(
                save_dir=config.save_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_loss=best_loss,
                is_best=is_best,
            )
            if is_best:
                print(f"New best val loss: {best_loss:.4e}", flush=True)
            if epoch % config.plot_interval == 0:
                plot_training_history(Path(config.save_dir) / "history.csv", config.save_dir)
