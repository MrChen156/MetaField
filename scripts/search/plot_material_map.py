"""Plot material maps from GA best_structure.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from structure.encoding import StructureEncoder


MATERIAL_COLORS = {
    0: "#1f77b4",   # flow
    1: "#ffbb78",   # sub
    2: "#d62728",   # Au
    3: "#bdbdbd",   # Ag
    4: "#17becf",   # Ti
    5: "#9467bd",   # Pt
    6: "#e7a7c8",   # TiO2
    7: "#f7f7f7",   # SiO2
    8: "#7f7f7f",   # Pd
    9: "#ff7f0e",   # Al
    10: "#c7c7c7",  # Al2O3
    11: "#8c564b",  # Cu
    12: "#2ca02c",  # Ge
    13: "#bcbd22",  # Nb2O5
    14: "#1b9e77",  # VO2
    15: "#6a3d9a",  # MoS2
}


def load_code_to_name(mapping_json: Path) -> dict[int, str]:
    with open(mapping_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    name_to_code = payload.get("material_name_to_code", {})
    return {int(code): str(name) for name, code in name_to_code.items()}


def parse_structure(payload: dict) -> tuple[float, float, float, float, list[int]]:
    if {"r_top_nm", "r_bot_nm", "height_nm", "period_nm"}.issubset(payload):
        r_top = float(payload["r_top_nm"])
        r_bot = float(payload["r_bot_nm"])
        height = float(payload["height_nm"])
        period = float(payload["period_nm"])
    else:
        r_top = float(payload["Rtop"])
        r_bot = float(payload["Rbot"])
        height = float(payload["H"])
        period = float(payload["P"])

    materials = [int(item) for item in payload["materials"]]
    if len(materials) < 70:
        materials.extend([0] * (70 - len(materials)))
    return r_top, r_bot, height, period, materials[:70]


def encode_for_plot(mat_map: np.ndarray, codes: list[int]) -> np.ndarray:
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    encoded = np.zeros_like(mat_map, dtype=np.int32)
    for code, idx in code_to_idx.items():
        encoded[mat_map == code] = idx
    return encoded


def save_colorbar(output_path: Path, codes: list[int], code_to_name: dict[int, str]) -> None:
    patches = [
        Patch(facecolor=MATERIAL_COLORS.get(code, "#000000"), edgecolor="black", label=f"{code}: {code_to_name.get(code, f'Material {code}')}")
        for code in codes
    ]
    fig_height = max(1.5, 0.38 * len(patches))
    fig, ax = plt.subplots(figsize=(3.2, fig_height))
    ax.axis("off")
    ax.legend(handles=patches, loc="center left", frameon=False, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_material_map(json_path: Path, mapping_json: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    code_to_name = load_code_to_name(mapping_json)
    r_top, r_bot, height, period, materials = parse_structure(payload)
    mat_map = StructureEncoder.build_material_map(r_top, r_bot, height, period, np.asarray(materials, dtype=np.int32))

    used_codes = sorted(int(code) for code in np.unique(mat_map))
    colors = [MATERIAL_COLORS.get(code, "#000000") for code in used_codes]
    encoded = encode_for_plot(mat_map, used_codes)
    cmap = ListedColormap(colors)

    output_dir = json_path.parent
    map_path = output_dir / "material_map.png"
    colorbar_path = output_dir / "material_colorbar.png"
    stats_path = output_dir / "material_usage.json"

    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    ax.imshow(encoded, cmap=cmap, interpolation="nearest", origin="upper", aspect="equal")
    ax.set_title(f"Material Map (P={period:.1f}nm)")
    ax.set_xlabel("x cell")
    ax.set_ylabel("z cell")
    fig.tight_layout()
    fig.savefig(map_path, dpi=220)
    plt.close(fig)

    save_colorbar(colorbar_path, used_codes, code_to_name)

    active_stack_codes = sorted({int(code) for code in materials if int(code) != 0})
    stats = {
        "source_json": str(json_path),
        "material_map": str(map_path),
        "colorbar": str(colorbar_path),
        "geometry": {"r_top_nm": r_top, "r_bot_nm": r_bot, "height_nm": height, "period_nm": period},
        "displayed_codes": [
            {"code": code, "name": code_to_name.get(code, f"Material {code}"), "color": MATERIAL_COLORS.get(code, "#000000")}
            for code in used_codes
        ],
        "active_stack_materials": [
            {"code": code, "name": code_to_name.get(code, f"Material {code}"), "color": MATERIAL_COLORS.get(code, "#000000")}
            for code in active_stack_codes
        ],
    }
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument("--mapping-json", type=Path, default=Path("material_ri_mapping.json"))
    args = parser.parse_args()

    for json_path in args.json_paths:
        stats = plot_material_map(json_path, args.mapping_json)
        print(stats["material_map"])
        print(stats["colorbar"])


if __name__ == "__main__":
    main()
