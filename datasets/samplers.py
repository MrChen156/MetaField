"""Samplers shared by the training entrypoints."""

from __future__ import annotations

import random

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedBucketSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

        self.group_indices: dict[str, list[int]] = {}
        for idx, key in enumerate(dataset.keys):
            group_name = dataset.group_map[key]
            self.group_indices.setdefault(group_name, []).append(idx)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        batches: list[list[int]] = []
        global_batch_size = self.batch_size * self.num_replicas

        for indices in self.group_indices.values():
            index_tensor = torch.tensor(indices)
            if self.shuffle:
                index_tensor = index_tensor[torch.randperm(len(index_tensor), generator=generator)]
                shard_indices = index_tensor.tolist()
            else:
                shard_indices = index_tensor.tolist()

            total_needed = (len(shard_indices) // global_batch_size) * global_batch_size
            shard_indices = shard_indices[:total_needed]

            for start in range(0, len(shard_indices), global_batch_size):
                global_batch = shard_indices[start : start + global_batch_size]
                local_batch = global_batch[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]
                if len(local_batch) == self.batch_size:
                    batches.append(local_batch)

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return sum(len(indices) // (self.batch_size * self.num_replicas) for indices in self.group_indices.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
