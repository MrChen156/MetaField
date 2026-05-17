from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

lmdb = pytest.importorskip("lmdb")

from datasets import LMDBDataset


def test_lmdb_dataset_smoke(tmp_path):
    lmdb_path = tmp_path / "sample.lmdb"
    split_path = tmp_path / "split.json"

    env = lmdb.open(str(lmdb_path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        sample = {
            "x": np.zeros((5, 32, 32), dtype=np.float32),
            "y": np.zeros((6, 32, 32), dtype=np.float32),
            "cond": np.zeros((3,), dtype=np.float32),
            "mask": np.ones((1, 32, 32), dtype=np.float32),
        }
        txn.put(b"group_a/0", pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
    env.close()

    split = {"train": [{"group": "group_a", "indices": [0]}], "val": [], "test": []}
    split_path.write_text(json.dumps(split), encoding="utf-8")

    dataset = LMDBDataset(lmdb_path, split_path, mode="train")
    x, y, cond, mask = dataset[0]
    assert x.shape == (5, 32, 32)
    assert y.shape == (6, 32, 32)
    assert cond.shape == (3,)
    assert mask.shape == (1, 32, 32)
