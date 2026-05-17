"""Logging, config dump, and training history helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path

import matplotlib
import psutil
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_total_memory_gb() -> float:
    """容器级已用内存 / Container-level memory usage."""
    try:
        with open("/sys/fs/cgroup/memory.current", "r", encoding="utf-8") as handle:
            return int(handle.read().strip()) / 1024**3
    except OSError:
        try:
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", "r", encoding="utf-8") as handle:
                return int(handle.read().strip()) / 1024**3
        except OSError:
            return psutil.virtual_memory().used / 1024**3


def save_yaml_config(config, save_path: str | Path) -> None:
    payload = asdict(config) if is_dataclass(config) else dict(config)
    with open(save_path, "w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)


def prepare_history_file(csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse", "train_grad", "train_total", "val_mse", "lr", "time"])


def append_history_row(csv_path: str | Path, row: list[object]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def plot_training_history(csv_path: str | Path, save_dir: str | Path) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    epochs, mses, grads, val_mses = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                epochs.append(int(row["epoch"]))
                mses.append(float(row["train_mse"]))
                grads.append(float(row["train_grad"]))
                val_mses.append(float(row["val_mse"]))
            except (KeyError, TypeError, ValueError):
                continue

    if len(epochs) < 2:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss", color="tab:blue")
    ax1.plot(epochs, mses, color="tab:blue", alpha=0.6, label="Train MSE")
    ax1.plot(epochs, val_mses, color="tab:orange", linewidth=2, label="Val MSE")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Gradient Loss", color="tab:red")
    ax2.plot(epochs, grads, color="tab:red", linestyle="--", alpha=0.5, label="Train Grad")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_yscale("log")

    fig.tight_layout()
    plt.savefig(Path(save_dir) / "loss_history.png", dpi=150)
    plt.close()
