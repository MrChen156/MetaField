"""Structure-to-tensor encoding for surrogate model inputs."""

from __future__ import annotations

import math

import numpy as np

from .materials import MaterialDatabase
from .utils import (
    FLOW_CODE,
    GRID_RES,
    N_Z_CELLS,
    PAD_MULTIPLE,
    SUB_CODE,
    compute_sdf,
    get_padded_size,
    pad_coords_linear,
    torch_pad_center_circX_repZ,
)


class StructureEncoder:
    def __init__(self, material_db: MaterialDatabase):
        self.material_db = material_db

    @staticmethod
    def build_material_map(r_top: float, r_bot: float, height_nm: float, period_nm: float, mat_layers: np.ndarray) -> np.ndarray:
        width = math.ceil(period_nm / GRID_RES) + 1
        mat_map = np.full((N_Z_CELLS, width), FLOW_CODE, dtype=np.int32)
        sub_start = 84 if height_nm < 0 else 251
        mat_map[sub_start:, :] = SUB_CODE

        h_cells = max(1, round(abs(height_nm) / GRID_RES))
        d_top = round(2 * r_top / GRID_RES)
        d_bot = round(2 * r_bot / GRID_RES)
        center_x = width // 2
        x_idx = np.arange(width)

        valid_layers: list[int] = []
        for mat in mat_layers:
            if mat == 0:
                break
            valid_layers.append(int(mat))

        for idx in range(h_cells):
            frac = idx / max(h_cells - 1, 1)
            diameter = d_bot + (d_top - d_bot) * frac
            inside = np.abs(x_idx - center_x) <= (diameter / 2.0)

            if height_nm < 0:
                row = sub_start + h_cells - 1 - idx
                if row < N_Z_CELLS:
                    mat_map[row, inside] = FLOW_CODE
            else:
                row = sub_start - 1 - idx
                if row >= 0:
                    mat_map[row, inside] = SUB_CODE

        for mat in valid_layers:
            is_solid = mat_map != FLOW_CODE
            top_profile = np.argmax(is_solid, axis=0)
            for x_pos in range(width):
                row = top_profile[x_pos] - 1
                if 0 <= row < N_Z_CELLS:
                    mat_map[row, x_pos] = mat

        return mat_map

    def genome_to_input(self, genome: np.ndarray, z_source_m: float = 580e-9):
        freq_thz = genome[0]
        wavelength_nm = 299792.458 / freq_thz

        r_top, r_bot, height_nm, period_nm = genome[1:5]
        mat_layers = genome[5:].astype(np.int32)

        mat_map = self.build_material_map(r_top, r_bot, height_nm, period_nm, mat_layers)
        num_z, num_x = mat_map.shape

        if height_nm < 0:
            z_nm = np.linspace(-50, 750, num_z)
        else:
            z_nm = np.linspace(-550, 250, num_z)
        x_nm = np.linspace(-period_nm / 2, period_nm / 2, num_x)
        x_m = (x_nm * 1e-9).astype(np.float32)
        z_m = (z_nm * 1e-9).astype(np.float32)

        n_map, k_map = self.material_db.batch_nk_maps(mat_map, wavelength_nm)
        target_h, target_w = get_padded_size(num_z, num_x, PAD_MULTIPLE)

        n_pad = torch_pad_center_circX_repZ(n_map[None, ...], target_h, target_w)[0]
        k_pad = torch_pad_center_circX_repZ(k_map[None, ...], target_h, target_w)[0]
        eps_r = n_pad**2 - k_pad**2
        eps_i = 2.0 * n_pad * k_pad

        x_pad_m = pad_coords_linear(x_m, target_w)
        z_pad_m = pad_coords_linear(z_m, target_h)
        xx, zz = np.meshgrid(x_pad_m, z_pad_m)

        lam_m = wavelength_nm * 1e-9
        k0 = np.float32(2 * np.pi / lam_m)
        k0x = (xx * k0).astype(np.float32)
        k0z = (zz * k0).astype(np.float32)

        geom_pad = torch_pad_center_circX_repZ(mat_map.astype(np.float32)[None, ...], target_h, target_w)[0]
        sdf = compute_sdf(geom_pad)

        x_tensor = np.stack([eps_r, eps_i, k0x, k0z, sdf], axis=0).astype(np.float32)
        mask = (geom_pad == FLOW_CODE).astype(np.uint8)[None, ...]

        period_m = float(x_m.max() - x_m.min() + (x_m[1] - x_m[0]))
        cond = np.array([k0 * 1e-7 - 1.0, k0 * period_m * 0.5, k0 * z_source_m], dtype=np.float32)

        return x_tensor, cond, mask, (target_h, target_w)
