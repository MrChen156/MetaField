#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate train/val/test split JSON for MetaField unified H5 files.

Strategy: spectral interpolation / 光谱插值
- Iterate through every design ID present in the H5 database.
- For each design, split its wavelength samples into train/val/test.
- The model sees every geometry during training, but only sparse spectral points.
- The output JSON is consumed by datasets/samplers.py and training configs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def split_indices(indices: list[int], ratios: tuple[float, float, float]) -> tuple[list[int], list[int], list[int]]:
    """Split a list of indices based on train/val/test ratios."""
    n = len(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def generate_interpolation_splits(
    h5_path: Path,
    output_json: Path,
    ratios: tuple[float, float, float],
    seed: int,
) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    print(f"Generating spectral-interpolation splits from {h5_path}")
    random.seed(seed)
    np.random.seed(seed)

    splits_out = {"train": [], "val": [], "test": []}
    buffer = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    total_samples = 0
    total_designs = 0

    with h5py.File(h5_path, "r") as f:
        for grp_key in f.keys():
            grp = f[grp_key]
            if "id" not in grp:
                continue

            all_ids = grp["id"][()].flatten()
            unique_ids = np.unique(all_ids)
            total_designs += len(unique_ids)

            print(f"Group {grp_key}: {len(unique_ids)} unique designs")

            for uid in unique_ids:
                design_indices = np.where(all_ids == uid)[0]
                design_indices = design_indices.tolist()
                random.shuffle(design_indices)

                tr_idx, va_idx, te_idx = split_indices(design_indices, ratios)

                buffer["train"][grp_key].extend(tr_idx)
                buffer["val"][grp_key].extend(va_idx)
                buffer["test"][grp_key].extend(te_idx)

                total_samples += len(design_indices)

    print("Formatting split JSON")
    for split_type in ["train", "val", "test"]:
        for grp_key, indices in buffer[split_type].items():
            if indices:
                indices.sort()
                splits_out[split_type].append({
                    "group": grp_key,
                    "indices": indices,
                })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(splits_out, handle, indent=2)

    n_tr = sum(len(item["indices"]) for item in splits_out["train"])
    n_va = sum(len(item["indices"]) for item in splits_out["val"])
    n_te = sum(len(item["indices"]) for item in splits_out["test"])

    print(f"Saved to {output_json}")
    print("Split summary:")
    print(f"Total designs processed: {total_designs}")
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {n_tr} ({n_tr / total_samples:.1%})")
    print(f"Val samples: {n_va} ({n_va / total_samples:.1%})")
    print(f"Test samples: {n_te} ({n_te / total_samples:.1%})")
    print("Strategy: interpolation, each design is split across spectrum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/val/test split JSON from a MetaField H5 file.")
    parser.add_argument("--h5", type=Path, required=True, help="Input unified H5 file.")
    parser.add_argument("--output", type=Path, required=True, help="Output split JSON path.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_interpolation_splits(
        h5_path=args.h5,
        output_json=args.output,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )
