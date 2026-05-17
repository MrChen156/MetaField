"""Surrogate-backed fitness evaluation."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from losses.metrics import top_percent_mean_energy
from models import MetaField

from .genome import GASearchConfig


class SurrogateEvaluator:
    def __init__(self, config: GASearchConfig):
        self.config = config
        self.models: list[tuple[torch.device, MetaField]] = []
        for device in self._select_devices():
            model = MetaField(
                in_channels=5,
                out_channels=6,
                cond_channels=3,
                base_channels=config.base_channels,
                cond_embed_dim=config.cond_embed_dim,
                heads=config.heads,
                max_dist=config.max_dist,
                transformer_depth=config.transformer_depth,
            ).to(device)
            checkpoint = torch.load(config.checkpoint_path, map_location=device)
            state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
            state = self._remap_state_dict_keys(state)
            model.load_state_dict(state)
            model.eval()
            self.models.append((device, model))

        if not self.models:
            raise RuntimeError(f"No surrogate model could be loaded from {config.checkpoint_path}")

    @staticmethod
    def _remap_state_dict_keys(state: dict) -> dict:
        mapped = {}
        for key, value in state.items():
            new_key = key.replace("module.", "")
            new_key = new_key.replace(".n1.", ".norm1.")
            new_key = new_key.replace(".n2.", ".norm2.")
            for block_name in ("down1", "down2", "down3"):
                new_key = new_key.replace(f"{block_name}.0.", f"{block_name}.block.0.")
                new_key = new_key.replace(f"{block_name}.1.", f"{block_name}.block.1.")
                new_key = new_key.replace(f"{block_name}.2.", f"{block_name}.block.2.")
            mapped[new_key] = value
        return mapped

    def _select_devices(self) -> list[torch.device]:
        if self.config.devices:
            return [torch.device(device) for device in self.config.devices[: self.config.max_models]]
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{idx}") for idx in range(min(torch.cuda.device_count(), self.config.max_models))]
        if torch.backends.mps.is_available():
            return [torch.device("mps")]
        return [torch.device("cpu")]

    @torch.no_grad()
    def _infer_on_device(self, device_idx: int, x_np: np.ndarray, c_np: np.ndarray) -> np.ndarray:
        device, model = self.models[device_idx]
        x_t = torch.from_numpy(x_np).to(device, non_blocking=True)
        c_t = torch.from_numpy(c_np).to(device, non_blocking=True)
        amp_enabled = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            pred = model(x_t, c_t)
        return (pred * self.config.field_norm).cpu().numpy()

    def evaluate_batch(self, x_list: list[np.ndarray], cond_list: list[np.ndarray]) -> list[np.ndarray]:
        if not x_list:
            return []
        x_all = np.stack(x_list)
        c_all = np.stack(cond_list)

        if len(self.models) == 1:
            return list(self._infer_on_device(0, x_all, c_all))

        midpoint = len(x_list) // 2
        results: list[np.ndarray | None] = [None, None]

        def run_slot(slot: int, device_idx: int, x_chunk: np.ndarray, c_chunk: np.ndarray) -> None:
            results[slot] = self._infer_on_device(device_idx, x_chunk, c_chunk)

        threads = [
            threading.Thread(target=run_slot, args=(0, 0, x_all[:midpoint], c_all[:midpoint])),
            threading.Thread(target=run_slot, args=(1, 1, x_all[midpoint:], c_all[midpoint:])),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return list(np.concatenate([results[0], results[1]], axis=0))


def compute_fitness(y_pred: np.ndarray, mask: np.ndarray, top_percent: float = 0.05) -> float:
    return top_percent_mean_energy(y_pred, mask, top_percent=top_percent)
