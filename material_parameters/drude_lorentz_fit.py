"""Drude-Lorentz material fitting utilities for maintaining materials.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class DrudeLorentzFitConfig:
    input_file: str
    material_name: str
    num_oscillators: int = 16
    k_weight: float = 8.0
    min_wavelength_nm: float = 300.0
    max_wavelength_nm: float = 1000.0
    input_wavelength_unit: str = "um"
    skiprows: int = 1
    default_refractive_index: float = 1.5
    materials_json: str = "material_parameters/materials.json"
    material_mapping_json: str = "material_ri_mapping.json"
    max_iterations: int = 3000
    ftol: float = 1e-9
    plot_output: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "DrudeLorentzFitConfig":
        return cls(**payload)


def load_material_mappings(mapping_json_path: str | Path) -> tuple[dict[str, float], dict[str, int]]:
    mapping_path = Path(mapping_json_path)
    if not mapping_path.exists():
        return {}, {}
    with open(mapping_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("material_mappings", {}), data.get("material_name_to_code", {})


def save_material_mappings(
    material_ri_mapping: dict[str, float],
    material_name_to_code_mapping: dict[str, int],
    mapping_json_path: str | Path,
) -> None:
    payload = {
        "material_mappings": material_ri_mapping,
        "material_name_to_code": material_name_to_code_mapping,
    }
    with open(mapping_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def get_or_create_material_code(
    material_name: str,
    mapping_json_path: str | Path,
    default_ri: float = 1.5,
) -> int:
    material_ri_mapping, material_name_to_code_mapping = load_material_mappings(mapping_json_path)
    if material_name in material_name_to_code_mapping:
        return int(material_name_to_code_mapping[material_name])

    max_code = max(material_name_to_code_mapping.values()) if material_name_to_code_mapping else 0
    new_code = int(max_code) + 1
    material_name_to_code_mapping[material_name] = new_code
    material_ri_mapping[material_name] = float(default_ri)
    save_material_mappings(material_ri_mapping, material_name_to_code_mapping, mapping_json_path)
    return new_code


def load_optical_table(
    filename: str | Path,
    wavelength_unit: str = "um",
    skiprows: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(filename, skiprows=skiprows)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Optical table must contain at least three columns: wavelength, n, k")

    wavelength = data[:, 0].astype(np.float64)
    if wavelength_unit == "um":
        wavelength_nm = wavelength * 1000.0
    elif wavelength_unit == "nm":
        wavelength_nm = wavelength
    else:
        raise ValueError(f"Unsupported wavelength unit: {wavelength_unit}")

    n_data = data[:, 1].astype(np.float64)
    k_data = data[:, 2].astype(np.float64)
    sort_idx = np.argsort(wavelength_nm)
    return wavelength_nm[sort_idx], n_data[sort_idx], k_data[sort_idx]


def drude_lorentz(energy_ev: np.ndarray, params: np.ndarray, num_oscillators: int) -> np.ndarray:
    eps_inf = params[0]
    wp = params[1]
    g_d = params[2]

    drude = -(wp**2) / (energy_ev**2 + 1j * g_d * energy_ev)
    lorentz = 0.0j
    for idx in range(num_oscillators):
        offset = 3 + idx * 3
        strength = params[offset]
        energy = params[offset + 1]
        gamma = params[offset + 2]
        lorentz += strength / (energy**2 - energy_ev**2 - 1j * gamma * energy_ev)
    return eps_inf + drude + lorentz


def objective(
    params: np.ndarray,
    energy_ev: np.ndarray,
    n_exp: np.ndarray,
    k_exp: np.ndarray,
    k_weight: float,
    num_oscillators: int,
) -> float:
    eps = drude_lorentz(energy_ev, params, num_oscillators)
    sqrt_eps = np.sqrt(eps)
    n_calc = np.real(sqrt_eps)
    k_calc = np.imag(sqrt_eps)
    n_error = n_exp - n_calc
    k_error = k_exp - k_calc
    return float(np.sum(n_error**2) + k_weight * np.sum(k_error**2))


def get_drude_lorentz_function(params: np.ndarray, num_oscillators: int) -> Callable[[float | np.ndarray], np.ndarray]:
    def dielectric_function(wavelength_nm: float | np.ndarray) -> np.ndarray:
        energy_ev = 1239.84193 / np.asarray(wavelength_nm)
        return drude_lorentz(energy_ev, params, num_oscillators)

    return dielectric_function


def get_drude_lorentz_equation(params: np.ndarray, num_oscillators: int) -> str:
    eps_inf = params[0]
    wp = params[1]
    g_d = params[2]

    equation = f"ε(λ) = {eps_inf:.4f} - {wp:.4f}² / (ω² + j{g_d:.4f}ω) "
    for idx in range(num_oscillators):
        offset = 3 + idx * 3
        strength = params[offset]
        energy = params[offset + 1]
        gamma = params[offset + 2]
        equation += f"+ {strength:.4f} / ({energy:.4f}² - ω² - j{gamma:.4f}ω) "
    return equation


def build_material_record(
    fitted_params: np.ndarray,
    num_oscillators: int,
    min_wavelength_nm: float,
    max_wavelength_nm: float,
    material_name: str,
    material_code: int,
) -> dict:
    record = {
        "material_name": material_name,
        "material_code": int(material_code),
        "wavelength_range": {"min": float(min_wavelength_nm), "max": float(max_wavelength_nm)},
        "num_oscillators": int(num_oscillators),
        "fitted_params": {
            "eps_inf": float(fitted_params[0]),
            "wp": float(fitted_params[1]),
            "gD": float(fitted_params[2]),
            "lorentz_oscillators": [],
        },
        "equation": get_drude_lorentz_equation(fitted_params, num_oscillators),
    }
    for idx in range(num_oscillators):
        offset = 3 + idx * 3
        record["fitted_params"]["lorentz_oscillators"].append(
            {
                "strength": float(fitted_params[offset]),
                "energy": float(fitted_params[offset + 1]),
                "gamma": float(fitted_params[offset + 2]),
            }
        )
    return record


def upsert_material_record(materials_json_path: str | Path, material_record: dict) -> Path:
    materials_path = Path(materials_json_path)
    materials_path.parent.mkdir(parents=True, exist_ok=True)

    if materials_path.exists():
        with open(materials_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, list):
            existing = {"materials": existing}
        elif not isinstance(existing, dict) or "materials" not in existing:
            existing = {"materials": []}
    else:
        existing = {"materials": []}

    replaced = False
    for idx, item in enumerate(existing["materials"]):
        if isinstance(item, dict) and item.get("material_name") == material_record["material_name"]:
            existing["materials"][idx] = material_record
            replaced = True
            break
    if not replaced:
        existing["materials"].append(material_record)

    with open(materials_path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, ensure_ascii=False)
    return materials_path


def save_fit_result(
    fitted_params: np.ndarray,
    num_oscillators: int,
    min_wavelength_nm: float,
    max_wavelength_nm: float,
    material_name: str,
    materials_json_path: str | Path,
    mapping_json_path: str | Path,
    default_ri: float = 1.5,
) -> tuple[int, Path]:
    material_code = get_or_create_material_code(material_name, mapping_json_path, default_ri=default_ri)
    material_record = build_material_record(
        fitted_params=fitted_params,
        num_oscillators=num_oscillators,
        min_wavelength_nm=min_wavelength_nm,
        max_wavelength_nm=max_wavelength_nm,
        material_name=material_name,
        material_code=material_code,
    )
    output_path = upsert_material_record(materials_json_path, material_record)
    return material_code, output_path


def plot_fit_result(
    wavelength_nm_all: np.ndarray,
    n_exp_all: np.ndarray,
    k_exp_all: np.ndarray,
    fitted_params: np.ndarray,
    num_oscillators: int,
    min_wavelength_nm: float,
    max_wavelength_nm: float,
    material_name: str,
    output_path: str | Path,
) -> Path:
    eps_fit_all = drude_lorentz(1239.84193 / wavelength_nm_all, fitted_params, num_oscillators)
    sqrt_eps = np.sqrt(eps_fit_all)
    n_fit_all = np.real(sqrt_eps)
    k_fit_all = np.imag(sqrt_eps)

    mask = (wavelength_nm_all >= min_wavelength_nm) & (wavelength_nm_all <= max_wavelength_nm)
    wl_nm_filtered = wavelength_nm_all[mask]
    n_exp_filtered = n_exp_all[mask]
    k_exp_filtered = k_exp_all[mask]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(wavelength_nm_all, n_exp_all, "k.", markersize=4, alpha=0.3, label="Original Data (All)")
    plt.plot(wl_nm_filtered, n_exp_filtered, "b.", markersize=5, label="Fitted Range Data")
    plt.plot(wavelength_nm_all, n_fit_all, "r-", linewidth=2, label="Fit Curve (L-BFGS-B)")
    plt.axvline(min_wavelength_nm, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(max_wavelength_nm, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Refractive Index (n)")
    plt.title(f"{material_name} Refractive Index Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(wavelength_nm_all, k_exp_all, "k.", markersize=4, alpha=0.3, label="Original Data (All)")
    plt.plot(wl_nm_filtered, k_exp_filtered, "b.", markersize=5, label="Fitted Range Data")
    plt.plot(wavelength_nm_all, k_fit_all, "r-", linewidth=2, label="Fit Curve (L-BFGS-B)")
    plt.axvline(min_wavelength_nm, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(max_wavelength_nm, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Extinction Coefficient (k)")
    plt.title(f"{material_name} Extinction Coefficient Fit")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def run_fit(config: DrudeLorentzFitConfig) -> tuple[np.ndarray, Callable[[float | np.ndarray], np.ndarray], Path]:
    wl_nm_all, n_exp_all, k_exp_all = load_optical_table(
        config.input_file,
        wavelength_unit=config.input_wavelength_unit,
        skiprows=config.skiprows,
    )

    mask = (wl_nm_all >= config.min_wavelength_nm) & (wl_nm_all <= config.max_wavelength_nm)
    wl_nm = wl_nm_all[mask]
    n_exp = n_exp_all[mask]
    k_exp = k_exp_all[mask]
    if len(wl_nm) == 0:
        raise ValueError(
            f"No optical data points found in [{config.min_wavelength_nm}, {config.max_wavelength_nm}] nm"
        )

    energy_ev = 1239.84193 / wl_nm
    x0 = [2.3, 1.0, 1e6]
    for idx in range(config.num_oscillators):
        strength = max(0.1, 4.0 / (idx + 1))
        energy = 0.1 + (10.0 - 0.1) * (idx / max(1, config.num_oscillators - 1))
        gamma = 0.1 + 0.9 * (idx / max(1, config.num_oscillators - 1))
        x0.extend([strength, energy, gamma])

    bounds = [(1, 10)] + [(0, 100), (50000.0, np.inf)] + [(0, 30), (0.01, 20), (0.0001, 10)] * config.num_oscillators
    result = minimize(
        objective,
        x0=np.asarray(x0, dtype=np.float64),
        args=(energy_ev, n_exp, k_exp, config.k_weight, config.num_oscillators),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": config.max_iterations, "ftol": config.ftol},
    )

    fitted_params = result.x
    dielectric_function = get_drude_lorentz_function(fitted_params, config.num_oscillators)
    _, output_path = save_fit_result(
        fitted_params=fitted_params,
        num_oscillators=config.num_oscillators,
        min_wavelength_nm=config.min_wavelength_nm,
        max_wavelength_nm=config.max_wavelength_nm,
        material_name=config.material_name,
        materials_json_path=config.materials_json,
        mapping_json_path=config.material_mapping_json,
        default_ri=config.default_refractive_index,
    )
    if config.plot_output:
        plot_fit_result(
            wavelength_nm_all=wl_nm_all,
            n_exp_all=n_exp_all,
            k_exp_all=k_exp_all,
            fitted_params=fitted_params,
            num_oscillators=config.num_oscillators,
            min_wavelength_nm=config.min_wavelength_nm,
            max_wavelength_nm=config.max_wavelength_nm,
            material_name=config.material_name,
            output_path=config.plot_output,
        )
    return fitted_params, dielectric_function, output_path


def fit_and_save_material(config: DrudeLorentzFitConfig) -> tuple[np.ndarray, Callable[[float | np.ndarray], np.ndarray], Path]:
    return run_fit(config)
