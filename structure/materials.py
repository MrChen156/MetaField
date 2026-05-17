"""Material database and wavelength-dependent lookup."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import FLOW_CODE, SUB_CODE


class MaterialDatabase:
    def __init__(self, materials_json: str | Path, mapping_json: str | Path):
        self.dl_params: dict[int, dict] = {}
        self.const_nk: dict[int, tuple[float, float]] = {}
        self.code_to_name: dict[int, str] = {}
        self.name_to_code: dict[str, int] = {}

        with open(mapping_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.name_to_code = data.get("material_name_to_code", {})
        self.code_to_name = {value: key for key, value in self.name_to_code.items()}
        for name, ri_val in data.get("material_mappings", {}).items():
            code = self.name_to_code.get(name)
            if code is not None:
                self.const_nk[code] = (float(ri_val), 0.0)

        materials_json = Path(materials_json)
        if materials_json.exists():
            with open(materials_json, "r", encoding="utf-8") as handle:
                materials = json.load(handle)
            for item in materials.get("materials", []):
                name = item.get("material_name")
                code = self.name_to_code.get(name)
                if code is not None:
                    self.dl_params[code] = item

    def _drude_lorentz_eps(self, wavelength_nm: float, params_dict: dict) -> complex:
        energy_ev = 1239.84193 / wavelength_nm
        params = params_dict["fitted_params"]
        eps = params["eps_inf"] + 0j
        wp, g_d = params["wp"], params["gD"]
        eps += -wp**2 / (energy_ev**2 + 1j * g_d * energy_ev)
        for osc in params.get("lorentz_oscillators", []):
            strength, energy_0, gamma = osc["strength"], osc["energy"], osc["gamma"]
            eps += strength / (energy_0**2 - energy_ev**2 - 1j * gamma * energy_ev)
        return eps

    def get_nk(self, code: int, wavelength_nm: float) -> tuple[float, float]:
        if code == FLOW_CODE:
            return 1.33, 0.0
        if code == SUB_CODE:
            return 1.6, 0.0
        if code in self.dl_params:
            params = self.dl_params[code]
            wl_range = params.get("wavelength_range", {})
            wl_clamped = np.clip(wavelength_nm, wl_range.get("min", 300), wl_range.get("max", 1000))
            eps = self._drude_lorentz_eps(wl_clamped, params)
            ri = np.sqrt(eps)
            return float(np.real(ri)), float(np.abs(np.imag(ri)))
        if code in self.const_nk:
            return self.const_nk[code]
        return 1.0, 0.0

    def batch_nk_maps(self, mat_map: np.ndarray, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
        n_map = np.zeros_like(mat_map, dtype=np.float32)
        k_map = np.zeros_like(mat_map, dtype=np.float32)
        for code in np.unique(mat_map):
            n_val, k_val = self.get_nk(int(code), wavelength_nm)
            mask = mat_map == code
            n_map[mask] = n_val
            k_map[mask] = k_val
        return n_map, k_map
