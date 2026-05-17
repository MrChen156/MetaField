"""Fit a material optical table and update materials.json / material_ri_mapping.json."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from material_parameters import DrudeLorentzFitConfig, fit_and_save_material


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/materials/drude_lorentz_fit.yaml")
    args = parser.parse_args()

    with open(Path(args.config), "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    config = DrudeLorentzFitConfig.from_dict(payload)
    fitted_params, dielectric_function, output_path = fit_and_save_material(config)

    print(f"Material: {config.material_name}")
    print(f"Oscillators: {config.num_oscillators}")
    print(f"Output: {output_path}")
    print(f"eps(500nm): {dielectric_function(500.0)}")
    print(f"Fitted params length: {len(fitted_params)}")


if __name__ == "__main__":
    main()
