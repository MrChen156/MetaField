"""Checkpoint save/load helpers for training."""

from __future__ import annotations

import gc
from pathlib import Path

import torch


def save_training_checkpoint(
    save_dir: str | Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    best_loss: float,
    is_best: bool = False,
) -> None:
    save_dir = Path(save_dir)
    state_dict = {
        "epoch": epoch,
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(state_dict, save_dir / "last_model.pth")
    if is_best:
        torch.save(state_dict, save_dir / "best_model.pth")


def resume_or_warmstart(
    save_dir: str | Path,
    pretrain_path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    steps_per_epoch: int,
) -> tuple[int, float]:
    save_dir = Path(save_dir)
    resume_path = save_dir / "last_model.pth"
    map_location = device

    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=map_location)
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))

        steps_to_skip = (start_epoch - 1) * steps_per_epoch
        for _ in range(steps_to_skip):
            scheduler.step()

        del checkpoint
        gc.collect()
        return start_epoch, best_loss

    if pretrain_path and Path(pretrain_path).exists():
        checkpoint = torch.load(pretrain_path, map_location=map_location)
        target_model = model.module if hasattr(model, "module") else model
        state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        state = {key.replace("module.", ""): value for key, value in state.items()}
        target_model.load_state_dict(state)
        del checkpoint
        gc.collect()
        return 1, float("inf")

    return 1, float("inf")
