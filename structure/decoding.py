"""Genome decoding helpers for reporting and serialization."""

from __future__ import annotations

import numpy as np


def genome_to_summary(genome: np.ndarray) -> dict[str, object]:
    wavelength_nm = float(299792.458 / genome[0])
    materials = genome[5:].astype(int).tolist()
    return {
        "wavelength_nm": wavelength_nm,
        "r_top_nm": float(genome[1]),
        "r_bot_nm": float(genome[2]),
        "height_nm": float(genome[3]),
        "period_nm": float(genome[4]),
        "materials": materials,
        "combined_vector": genome.tolist(),
    }
