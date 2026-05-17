"""LMDB-backed dataset implementation."""

from __future__ import annotations

import pickle
from pathlib import Path

import lmdb
import torch
from torch.utils.data import Dataset

from .utils import expand_split_entries, load_split_entries


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path: str | Path, split_json: str | Path, mode: str = "train"):
        self.lmdb_path = str(lmdb_path)
        self.env: lmdb.Environment | None = None
        self.txn = None
        split_entries = load_split_entries(split_json, mode)
        self.keys, self.group_map = expand_split_entries(split_entries)

    def __len__(self) -> int:
        return len(self.keys)

    def _open(self) -> None:
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._open()
        assert self.txn is not None
        key = self.keys[idx]
        raw = self.txn.get(key)
        if raw is None:
            raise KeyError(f"Missing LMDB sample for key {key!r}")
        sample = pickle.loads(raw)
        return (
            torch.from_numpy(sample["x"]),
            torch.from_numpy(sample["y"]),
            torch.from_numpy(sample["cond"]),
            torch.from_numpy(sample["mask"]),
        )


LMDB_Dataset = LMDBDataset
