#!/usr/bin/env python3
"""Memory-safe HDF5 -> LMDB converter for MetaField datasets.

中文说明:
- 分批读取 HDF5，避免一次性加载大文件。
- 分批提交 LMDB transaction，降低内存峰值。
- 输出 key 形如 ``size_288_192/1003``，与 split JSON 中的 group/index 对应。

English:
- Read large H5 files in batches.
- Commit LMDB transactions periodically to control memory usage.
- Store keys as ``group/index`` so they match the generated split JSON.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import shutil
from pathlib import Path

import h5py
import lmdb
import numpy as np
from tqdm import tqdm


def convert(
    h5_path: Path,
    lmdb_path: Path,
    map_size_gb: int,
    batch_size: int,
    commit_every: int,
    overwrite: bool,
) -> None:
    print(f"Source H5: {h5_path}")
    print(f"Target LMDB: {lmdb_path}")

    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    if lmdb_path.exists():
        if not overwrite:
            raise FileExistsError(f"LMDB path already exists: {lmdb_path}. Pass --overwrite to replace it.")
        shutil.rmtree(lmdb_path)
        print("Removed existing LMDB directory")

    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    map_size = int(map_size_gb * 1024**3)

    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size,
        writemap=True,
        map_async=True,
        meminit=False,
    )

    total_samples = 0
    all_keys = []  # 存储所有key用于后续索引

    with h5py.File(h5_path, "r") as f:
        group_names = list(f.keys())
        print(f"Found {len(group_names)} groups")

        total_count = 0
        for g_name in group_names:
            grp = f[g_name]
            if "x" in grp:
                total_count += grp["x"].shape[0]
        print(f"Total samples: {total_count}")

        pbar = tqdm(total=total_count, desc="Converting")

        txn = env.begin(write=True)
        pending_count = 0

        for g_name in group_names:
            grp = f[g_name]
            if "x" not in grp:
                continue

            n_samples = grp["x"].shape[0]

            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                x_batch = grp["x"][batch_start:batch_end]
                y_batch = grp["y"][batch_start:batch_end]
                cond_batch = grp["cond"][batch_start:batch_end]
                mask_batch = grp["mask"][batch_start:batch_end]

                for i in range(batch_end - batch_start):
                    local_idx = batch_start + i

                    key_str = f"{g_name}/{local_idx}"
                    key_bytes = key_str.encode("ascii")
                    all_keys.append(key_str)

                    sample = {
                        "x": x_batch[i].astype(np.float32),
                        "y": y_batch[i].astype(np.float32),
                        "cond": cond_batch[i].astype(np.float32),
                        "mask": mask_batch[i].astype(np.float32),
                    }
                    value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)

                    txn.put(key_bytes, value)
                    pending_count += 1
                    total_samples += 1

                    if pending_count >= commit_every:
                        txn.commit()
                        txn = env.begin(write=True)
                        pending_count = 0
                        gc.collect()

                pbar.update(batch_end - batch_start)

                del x_batch, y_batch, cond_batch, mask_batch
                gc.collect()

        pbar.close()

        meta = {
            "total_samples": total_samples,
            "keys": all_keys,
        }
        txn.put(b"__meta__", pickle.dumps(meta))
        txn.commit()

    env.sync()
    env.close()

    # 统计结果
    lmdb_size = sum(
        os.path.getsize(os.path.join(lmdb_path, item))
        for item in os.listdir(lmdb_path)
    )
    print("\nConversion complete")
    print(f"Total samples: {total_samples}")
    print(f"LMDB size: {lmdb_size / 1024**3:.2f} GB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MetaField unified H5 dataset to LMDB.")
    parser.add_argument("--h5", type=Path, required=True, help="Input unified H5 file.")
    parser.add_argument("--lmdb", type=Path, required=True, help="Output LMDB directory.")
    parser.add_argument("--map-size-gb", type=int, default=180, help="LMDB map size in GB.")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of H5 samples to read per batch.")
    parser.add_argument("--commit-every", type=int, default=2000, help="Commit LMDB transaction every N samples.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing LMDB output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.h5, args.lmdb, args.map_size_gb, args.batch_size, args.commit_every, args.overwrite)
