"""Geometry and material constraints kept separate from GA orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import N_MAT_SLOTS


@dataclass
class StructureConstraintRanges:
    freq_range: tuple[float, float]
    r_top_range: tuple[float, float]
    r_bot_range: tuple[float, float]
    height_range: tuple[float, float]
    period_range: tuple[float, float]
    min_gap_nm: float
    min_block_cells: int
    max_material_transitions: int
    adhesion_materials: list[int]


class StructureConstraints:
    def __init__(self, ranges: StructureConstraintRanges):
        self.ranges = ranges

    def repair(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        repaired = genome.copy()
        repaired[0] = np.clip(repaired[0], *self.ranges.freq_range)
        repaired[1] = np.clip(repaired[1], *self.ranges.r_top_range)
        repaired[2] = np.clip(repaired[2], *self.ranges.r_bot_range)
        repaired[3] = np.clip(repaired[3], *self.ranges.height_range)
        repaired[4] = np.clip(repaired[4], *self.ranges.period_range)

        height_val = repaired[3]
        if height_val > 0 and repaired[1] > repaired[2]:
            repaired[1], repaired[2] = repaired[2], repaired[1]
        elif height_val < 0 and repaired[2] > repaired[1]:
            repaired[1], repaired[2] = repaired[2], repaired[1]

        max_diameter = 2 * max(repaired[1], repaired[2])
        if max_diameter > repaired[4] - self.ranges.min_gap_nm:
            scale = (repaired[4] - self.ranges.min_gap_nm) / (max_diameter + 1e-6)
            repaired[1] = max(0, repaired[1] * scale)
            repaired[2] = max(0, repaired[2] * scale)

        mat_vec = repaired[5 : 5 + N_MAT_SLOTS].astype(int)

        if mat_vec[0] not in self.ranges.adhesion_materials:
            mat_vec[0] = int(rng.choice(self.ranges.adhesion_materials))

        trunc_idx = len(mat_vec)
        for idx, value in enumerate(mat_vec):
            if value == 0:
                trunc_idx = idx
                break
        if trunc_idx < len(mat_vec):
            mat_vec[trunc_idx:] = 0

        if trunc_idx > 0:
            active_mats = mat_vec[:trunc_idx]
            transitions = 0
            last_mat = active_mats[0]
            current_block = 1

            for idx in range(1, len(active_mats)):
                if active_mats[idx] != last_mat:
                    if current_block < self.ranges.min_block_cells:
                        active_mats[idx] = last_mat
                        current_block += 1
                    else:
                        transitions += 1
                        if transitions > self.ranges.max_material_transitions:
                            active_mats[idx:] = last_mat
                            break
                        last_mat = active_mats[idx]
                        current_block = 1
                else:
                    current_block += 1
            mat_vec[:trunc_idx] = active_mats

        repaired[5 : 5 + N_MAT_SLOTS] = mat_vec
        return repaired
