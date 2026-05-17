"""Helpers for dataset split handling."""

from __future__ import annotations

import json
from pathlib import Path


def load_split_entries(split_json: str | Path, mode: str) -> list[dict]:
    with open(split_json, "r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    return split_data[mode]


def expand_split_entries(split_entries: list[dict]) -> tuple[list[bytes], dict[bytes, str]]:
    keys: list[bytes] = []
    group_map: dict[bytes, str] = {}
    for item in split_entries:
        group_name = item["group"]
        for idx in item["indices"]:
            key = f"{group_name}/{idx}".encode("ascii")
            keys.append(key)
            group_map[key] = group_name
    return keys, group_map
