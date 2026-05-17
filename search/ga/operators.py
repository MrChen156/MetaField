"""Genome initialization and genetic operators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from structure.constraints import StructureConstraints
from structure.utils import N_MAT_SLOTS

from .genome import GASearchConfig


class GeneticOperators:
    def __init__(self, config: GASearchConfig, constraints: StructureConstraints):
        self.config = config
        self.constraints = constraints
        self.rng = np.random.default_rng()
        self.training_genomes = np.empty((0, 4 + N_MAT_SLOTS))
        self.geo_mean = None
        self.geo_cov_inv = None

    def load_bounds_and_manifold(self) -> None:
        if not self.config.training_vectors_json:
            self.training_genomes = np.empty((0, 4 + N_MAT_SLOTS))
            self.geo_mean = None
            self.geo_cov_inv = None
            return

        vectors_path = Path(self.config.training_vectors_json)
        if not vectors_path.exists():
            self.training_genomes = np.empty((0, 4 + N_MAT_SLOTS))
            self.geo_mean = None
            self.geo_cov_inv = None
            return

        with open(vectors_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        all_vecs = np.array(list(data.values()), dtype=np.float64)
        geo = all_vecs[:, :4]

        self.geo_mean = np.mean(geo, axis=0)
        cov = np.cov(geo, rowvar=False) + np.eye(4) * 1e-6
        self.geo_cov_inv = np.linalg.inv(cov)

        margin = 0.1
        for idx, bounds_name in enumerate(["r_top_range", "r_bot_range", "height_range", "period_range"]):
            current_lo, current_hi = getattr(self.config, bounds_name)
            if current_lo == current_hi:
                continue

            lo, hi = geo[:, idx].min(), geo[:, idx].max()
            span = hi - lo + 1e-6
            lo_new, hi_new = lo - margin * span, hi + margin * span

            if bounds_name == "r_top_range":
                lo_new, hi_new = max(100.0, lo_new), min(180.0, hi_new)
            elif bounds_name == "r_bot_range":
                lo_new, hi_new = max(80.0, lo_new), min(150.0, hi_new)
            elif bounds_name == "height_range":
                lo_new, hi_new = max(-500.0, lo_new), min(500.0, hi_new)
            elif bounds_name == "period_range":
                lo_new, hi_new = max(300.0, lo_new), min(600.0, hi_new)
            setattr(self.config, bounds_name, (lo_new, hi_new))

        self.constraints.ranges.r_top_range = self.config.r_top_range
        self.constraints.ranges.r_bot_range = self.config.r_bot_range
        self.constraints.ranges.height_range = self.config.height_range
        self.constraints.ranges.period_range = self.config.period_range
        self.training_genomes = all_vecs

    def _stack_config(self) -> dict | None:
        stack_cfg = self.config.material_stack
        if not stack_cfg or not stack_cfg.get("enabled", False):
            return None
        return stack_cfg

    def _random_material_stack(self) -> np.ndarray:
        stack_cfg = self._stack_config()
        if not stack_cfg:
            return np.zeros(N_MAT_SLOTS, dtype=np.float64)

        max_layers = int(stack_cfg.get("max_layers", 3))
        min_layer_cells = int(stack_cfg.get("min_layer_cells", self.config.min_block_cells))
        max_total_cells = int(stack_cfg.get("max_total_cells", N_MAT_SLOTS))
        max_total_cells = min(max_total_cells, N_MAT_SLOTS)

        bottom_choices = [int(item) for item in stack_cfg.get("bottom_material_choices", self.config.adhesion_materials)]
        middle_choices = [int(item) for item in stack_cfg.get("middle_material_choices", bottom_choices)]
        top_material = int(stack_cfg["top_material"])
        allow_single_top_only = bool(stack_cfg.get("allow_single_top_only", False))

        min_layers = 1 if allow_single_top_only or top_material in bottom_choices else 2
        n_layers = int(self.rng.integers(min_layers, max_layers + 1))
        n_layers = max(1, min(n_layers, max_layers))

        if n_layers == 1:
            materials = [top_material]
        else:
            materials = [int(self.rng.choice(bottom_choices))]
            for _ in range(max(0, n_layers - 2)):
                materials.append(int(self.rng.choice(middle_choices)))
            materials.append(top_material)

        min_total = min_layer_cells * len(materials)
        if min_total > max_total_cells:
            raise ValueError("material_stack min_layer_cells * max_layers exceeds max_total_cells")
        total_cells = int(self.rng.integers(min_total, max_total_cells + 1))
        remaining = total_cells - min_total
        extras = self.rng.multinomial(remaining, np.ones(len(materials)) / len(materials))
        thicknesses = [min_layer_cells + int(extra) for extra in extras]

        mat_vec = np.zeros(N_MAT_SLOTS, dtype=np.float64)
        cursor = 0
        for material, thickness in zip(materials, thicknesses):
            mat_vec[cursor : cursor + thickness] = material
            cursor += thickness
        return mat_vec

    def _compress_material_blocks(self, mat_vec: np.ndarray) -> list[tuple[int, int]]:
        active = [int(item) for item in mat_vec.astype(int).tolist() if int(item) != 0]
        if not active:
            return []

        blocks: list[tuple[int, int]] = []
        current = active[0]
        length = 1
        for item in active[1:]:
            if item == current:
                length += 1
            else:
                blocks.append((current, length))
                current = item
                length = 1
        blocks.append((current, length))
        return blocks

    def _apply_material_stack_constraint(self, genome: np.ndarray) -> np.ndarray:
        stack_cfg = self._stack_config()
        if not stack_cfg:
            return genome

        max_layers = int(stack_cfg.get("max_layers", 3))
        min_layer_cells = int(stack_cfg.get("min_layer_cells", self.config.min_block_cells))
        bottom_choices = {int(item) for item in stack_cfg.get("bottom_material_choices", self.config.adhesion_materials)}
        middle_choices = {int(item) for item in stack_cfg.get("middle_material_choices", bottom_choices)}
        top_material = int(stack_cfg["top_material"])
        allow_single_top_only = bool(stack_cfg.get("allow_single_top_only", False))

        blocks = self._compress_material_blocks(genome[5:])
        if allow_single_top_only and len(blocks) == 1 and blocks[0][0] == top_material and blocks[0][1] >= min_layer_cells:
            return genome

        valid = bool(blocks)
        valid = valid and len(blocks) <= max_layers
        valid = valid and blocks[-1][0] == top_material
        valid = valid and blocks[0][0] in bottom_choices
        valid = valid and all(length >= min_layer_cells for _, length in blocks)
        if len(blocks) > 2:
            valid = valid and all(material in middle_choices for material, _ in blocks[1:-1])

        if not valid:
            genome[5:] = self._random_material_stack()
        return genome

    def random_genome(self) -> np.ndarray:
        cfg = self.config
        genome = np.zeros(5 + N_MAT_SLOTS, dtype=np.float64)
        genome[0] = self.rng.uniform(*cfg.freq_range)
        genome[1] = self.rng.uniform(*cfg.r_top_range)
        genome[2] = self.rng.uniform(*cfg.r_bot_range)
        genome[3] = self.rng.uniform(*cfg.height_range)
        genome[4] = self.rng.uniform(*cfg.period_range)

        pos = 0
        while pos < N_MAT_SLOTS:
            mat = self.rng.choice(cfg.allowed_materials)
            block_len = self.rng.integers(cfg.min_block_cells, 15)
            end = min(pos + block_len, N_MAT_SLOTS)
            genome[5 + pos : 5 + end] = mat
            pos = end
        if self._stack_config():
            genome[5:] = self._random_material_stack()
        genome = self.constraints.repair(genome, self.rng)
        return self._apply_material_stack_constraint(genome)

    def init_population(self) -> list[np.ndarray]:
        pop: list[np.ndarray] = []
        seed_count = min(int(self.config.population_size * self.config.seed_ratio), len(self.training_genomes))
        if seed_count > 0:
            indices = self.rng.choice(len(self.training_genomes), seed_count, replace=False)
            for idx in indices:
                base_struct = self.training_genomes[idx].copy()
                full_gene = np.zeros(5 + N_MAT_SLOTS, dtype=np.float64)
                full_gene[0] = self.rng.uniform(*self.config.freq_range)
                full_gene[1:] = base_struct
                if self._stack_config():
                    full_gene[5:] = self._random_material_stack()
                full_gene = self.constraints.repair(full_gene, self.rng)
                pop.append(self._apply_material_stack_constraint(full_gene))

        while len(pop) < self.config.population_size:
            pop.append(self.random_genome())
        return pop

    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        child = parent_a.copy()
        if self.rng.random() < self.config.crossover_rate:
            for idx in range(5):
                if self.rng.random() < 0.5:
                    child[idx] = parent_b[idx]
            left, right = sorted(self.rng.choice(N_MAT_SLOTS, 2, replace=False))
            child[5 + left : 5 + right] = parent_b[5 + left : 5 + right]
        child = self.constraints.repair(child, self.rng)
        return self._apply_material_stack_constraint(child)

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        cfg = self.config
        child = genome.copy()
        if self.rng.random() < cfg.mutation_rate_geo:
            idx = self.rng.integers(0, 5)
            ranges = [cfg.freq_range, cfg.r_top_range, cfg.r_bot_range, cfg.height_range, cfg.period_range]
            lo, hi = ranges[idx]
            child[idx] += self.rng.normal(0, (hi - lo) * 0.1)
        if self.rng.random() < cfg.mutation_rate_mat:
            if self._stack_config():
                child[5:] = self._random_material_stack()
            else:
                start = self.rng.integers(0, N_MAT_SLOTS)
                block_len = self.rng.integers(cfg.min_block_cells, 12)
                end = min(start + block_len, N_MAT_SLOTS)
                child[5 + start : 5 + end] = self.rng.choice(cfg.allowed_materials)
        child = self.constraints.repair(child, self.rng)
        return self._apply_material_stack_constraint(child)

    def select(self, pop: list[np.ndarray], fitness: np.ndarray) -> np.ndarray:
        idx = self.rng.choice(len(pop), self.config.tournament_size, replace=False)
        return pop[idx[np.argmax(fitness[idx])]]

    def manifold_penalty(self, genome: np.ndarray) -> float:
        if self.geo_mean is None or self.geo_cov_inv is None:
            return 0.0
        geo_gene = genome[1:5]
        diff = geo_gene - self.geo_mean
        m_dist_sq = diff @ self.geo_cov_inv @ diff.T
        m_dist = np.sqrt(max(0.0, m_dist_sq))
        if m_dist <= self.config.manifold_threshold:
            return 0.0
        return self.config.manifold_penalty_weight * (m_dist - self.config.manifold_threshold)
