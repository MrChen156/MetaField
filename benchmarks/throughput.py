"""Throughput benchmark retained from the current benchmark logic."""

from __future__ import annotations

import contextlib
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from models import MetaField
from structure.utils import get_padded_size


@dataclass
class ThroughputBenchmarkConfig:
    input_shape: tuple[int, int] = (268, 160)
    batch_start: int = 2
    batch_limit: int = 4096
    warmup_iters: int = 3
    timed_iters: int = 20
    dtype: str = "float32"
    device: str = ""
    min_safety_gb: float = 4.0
    max_stall: int = 3
    fdtd_seconds: int = 150
    ga_population: int = 2048
    ga_generations: int = 2000
    output_path: str = "results/benchmark/throughput.json"
    base_channels: int = 96
    heads: int = 8
    max_dist: int = 48
    cond_embed_dim: int = 256
    transformer_depth: int = 8

    @classmethod
    def from_dict(cls, payload: dict) -> "ThroughputBenchmarkConfig":
        payload = dict(payload)
        if "input_shape" in payload:
            payload["input_shape"] = tuple(payload["input_shape"])
        return cls(**payload)


def clear_memory(device_type: str) -> None:
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device_type == "mps":
        torch.mps.empty_cache()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_mem_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**3)
    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / (1024**3)
        except RuntimeError:
            return -1.0
    return -1.0


def get_free_mem_gb(device: torch.device) -> float:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return free / (1024**3)
    if device.type == "mps":
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    return 999.0


def estimate_next_batch_gb(current_mem: float) -> float:
    return current_mem * 1.8


def _resolve_device(preferred: str) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return contextlib.nullcontext()


def benchmark_model(config: ThroughputBenchmarkConfig) -> dict[str, object]:
    device = _resolve_device(config.device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    elif device.type == "mps":
        import psutil

        device_name = "Apple Silicon (MPS)"
        total_mem = psutil.virtual_memory().total / (1024**3)
    else:
        device_name = "CPU"
        total_mem = 0.0

    dtype = getattr(torch, config.dtype)
    model = MetaField(
        in_channels=5,
        out_channels=6,
        cond_channels=3,
        base_channels=config.base_channels,
        cond_embed_dim=config.cond_embed_dim,
        heads=config.heads,
        max_dist=config.max_dist,
        transformer_depth=config.transformer_depth,
    ).to(device).eval()

    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    target_h, target_w = get_padded_size(config.input_shape[0], config.input_shape[1], 32)

    current_batch = config.batch_start
    stall_count = 0
    best_throughput = 0.0
    optimal_batch = 0
    prev_throughput = 0.0
    last_mem_gb = 0.0
    results: list[dict[str, float | int | str]] = []

    while current_batch <= config.batch_limit:
        free_gb = get_free_mem_gb(device)
        estimated_need = estimate_next_batch_gb(max(last_mem_gb, 0.5))
        if free_gb < config.min_safety_gb:
            results.append({"batch": current_batch, "status": f"unsafe_free_lt_{config.min_safety_gb}gb"})
            break
        if estimated_need > free_gb * 0.85:
            results.append({"batch": current_batch, "status": "unsafe_projected_usage"})
            break

        try:
            clear_memory(device.type)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            x = torch.randn(current_batch, 5, target_h, target_w, dtype=dtype, device=device)
            cond = torch.randn(current_batch, 3, dtype=dtype, device=device)

            with _autocast_context(device), torch.no_grad():
                for _ in range(config.warmup_iters):
                    _ = model(x, cond)
            sync_device(device)

            mem_gb = get_mem_gb(device)
            last_mem_gb = mem_gb

            start_time = time.time()
            with _autocast_context(device), torch.no_grad():
                for _ in range(config.timed_iters):
                    _ = model(x, cond)
            sync_device(device)
            end_time = time.time()

            avg_ms = (end_time - start_time) * 1000 / config.timed_iters
            throughput = current_batch / avg_ms * 1000
            ms_per_sample = avg_ms / current_batch

            if throughput <= prev_throughput * 1.02:
                stall_count += 1
                status = f"stall_{stall_count}"
            else:
                stall_count = 0
                status = "ok"

            if throughput > best_throughput:
                best_throughput = throughput
                optimal_batch = current_batch
                status = "best"

            results.append(
                {
                    "batch": current_batch,
                    "mem_gb": mem_gb,
                    "free_gb": get_free_mem_gb(device),
                    "ms_per_batch": avg_ms,
                    "throughput": throughput,
                    "ms_per_sample": ms_per_sample,
                    "status": status,
                }
            )

            prev_throughput = throughput
            del x, cond

            if stall_count >= config.max_stall:
                break
            current_batch *= 2
        except RuntimeError as exc:
            err = str(exc).lower()
            if any(token in err for token in ["out of memory", "oom", "32-bit", "mps backend"]):
                results.append({"batch": current_batch, "status": "oom"})
                clear_memory(device.type)
                break
            raise

    total_evals = config.ga_population * config.ga_generations
    surrogate_hours = total_evals / max(best_throughput, 1.0) / 3600
    fdtd_hours = total_evals * config.fdtd_seconds / 3600

    summary = {
        "device": str(device),
        "device_name": device_name,
        "device_total_memory_gb": total_mem,
        "model_params_million": n_params,
        "model_size_mb": param_mb,
        "input_shape": [target_h, target_w],
        "optimal_batch": optimal_batch,
        "peak_throughput": best_throughput,
        "latency_ms_per_struct": 1000 / max(best_throughput, 1.0),
        "fdtd_speedup": best_throughput * config.fdtd_seconds,
        "surrogate_hours_for_ga": surrogate_hours,
        "fdtd_hours_for_ga": fdtd_hours,
        "ga_speedup": fdtd_hours / max(surrogate_hours, 0.01),
        "results": results,
        "config": asdict(config),
    }

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
