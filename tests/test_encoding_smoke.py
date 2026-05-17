from __future__ import annotations

import json

import numpy as np

from structure import MaterialDatabase, StructureEncoder
from structure.utils import N_MAT_SLOTS


def test_structure_encoding_smoke(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    materials_path = tmp_path / "materials.json"

    mapping_path.write_text(
        json.dumps(
            {
                "material_name_to_code": {"MatA": 4},
                "material_mappings": {"MatA": 1.8},
            }
        ),
        encoding="utf-8",
    )
    materials_path.write_text(json.dumps({"materials": []}), encoding="utf-8")

    material_db = MaterialDatabase(materials_path, mapping_path)
    encoder = StructureEncoder(material_db)

    genome = np.zeros(5 + N_MAT_SLOTS, dtype=np.float64)
    genome[0] = 500.0
    genome[1] = 140.0
    genome[2] = 110.0
    genome[3] = 300.0
    genome[4] = 420.0
    genome[5:10] = 4

    x_tensor, cond, mask, padded_size = encoder.genome_to_input(genome)
    assert x_tensor.shape[0] == 5
    assert cond.shape == (3,)
    assert mask.shape[0] == 1
    assert padded_size[0] >= 268
    assert padded_size[1] >= x_tensor.shape[-1]
