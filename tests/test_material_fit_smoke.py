from __future__ import annotations

import json

import numpy as np

from material_parameters.drude_lorentz_fit import save_fit_result


def test_save_fit_result_updates_mapping_and_materials(tmp_path):
    mapping_path = tmp_path / "material_ri_mapping.json"
    materials_path = tmp_path / "materials.json"

    mapping_path.write_text(
        json.dumps(
            {
                "material_mappings": {"flow": 1.33, "sub": 1.6},
                "material_name_to_code": {"flow": 0, "sub": 1},
            }
        ),
        encoding="utf-8",
    )
    materials_path.write_text(json.dumps({"materials": []}), encoding="utf-8")

    fitted_params = np.array([2.3, 1.0, 1e6, 0.5, 2.0, 0.1], dtype=np.float64)
    material_code, output_path = save_fit_result(
        fitted_params=fitted_params,
        num_oscillators=1,
        min_wavelength_nm=300.0,
        max_wavelength_nm=1000.0,
        material_name="TestMat",
        materials_json_path=materials_path,
        mapping_json_path=mapping_path,
        default_ri=1.7,
    )

    assert material_code == 2
    assert output_path == materials_path

    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping_payload["material_name_to_code"]["TestMat"] == 2
    assert mapping_payload["material_mappings"]["TestMat"] == 1.7

    materials_payload = json.loads(materials_path.read_text(encoding="utf-8"))
    assert materials_payload["materials"][0]["material_name"] == "TestMat"
    assert materials_payload["materials"][0]["material_code"] == 2
    assert materials_payload["materials"][0]["num_oscillators"] == 1
