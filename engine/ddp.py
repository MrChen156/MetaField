"""DDP setup and cleanup helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_ddp() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return gpu, rank, world_size
    return 0, 0, 1


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
