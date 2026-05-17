"""GA configuration and genome-related defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from structure.constraints import StructureConstraintRanges


@dataclass
class GASearchConfig:
    surrogate_checkpoint: str = "checkpoints/best_model.pth"
    materials_json: str = "material_parameters/materials.json"
    material_mapping_json: str = "material_ri_mapping.json"
    training_vectors_json: str = "combined_vectors.json"
    output_dir: str = "results/ga"

    freq_range: tuple[float, float] = (374.7405725, 749.481145)
    z_source_m: float = 580e-9
    field_norm: float = 10.0

    base_channels: int = 96
    heads: int = 8
    max_dist: int = 48
    cond_embed_dim: int = 256
    transformer_depth: int = 8

    population_size: int = 256
    generations: int = 500
    elite_count: int = 16
    tournament_size: int = 3
    crossover_rate: float = 0.85
    mutation_rate_geo: float = 0.40
    mutation_rate_mat: float = 0.30
    batch_size: int = 80
    seed_ratio: float = 0.50

    r_top_range: tuple[float, float] = (100.0, 180.0)
    r_bot_range: tuple[float, float] = (80.0, 150.0)
    height_range: tuple[float, float] = (-500.0, 500.0)
    period_range: tuple[float, float] = (300.0, 600.0)
    min_gap_nm: float = 12.0

    min_block_cells: int = 3
    max_material_transitions: int = 5
    manifold_threshold: float = 3.0
    manifold_penalty_weight: float = 10000.0

    allowed_materials: list[int] = field(default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    adhesion_materials: list[int] = field(default_factory=lambda: [4, 6, 13])
    material_stack: dict | None = None
    devices: list[str] = field(default_factory=list)
    max_models: int = 2

    @classmethod
    def from_dict(cls, payload: dict) -> "GASearchConfig":
        payload = dict(payload)
        tuple_fields = {"freq_range", "r_top_range", "r_bot_range", "height_range", "period_range"}
        for key in tuple_fields:
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    def to_constraint_ranges(self) -> StructureConstraintRanges:
        return StructureConstraintRanges(
            freq_range=self.freq_range,
            r_top_range=self.r_top_range,
            r_bot_range=self.r_bot_range,
            height_range=self.height_range,
            period_range=self.period_range,
            min_gap_nm=self.min_gap_nm,
            min_block_cells=self.min_block_cells,
            max_material_transitions=self.max_material_transitions,
            adhesion_materials=self.adhesion_materials,
        )

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.surrogate_checkpoint)
