"""Main GA runner separated from structure encoding and model definition."""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structure.decoding import genome_to_summary
from structure.encoding import StructureEncoder
from structure.utils import N_MAT_SLOTS

from .fitness import SurrogateEvaluator, compute_fitness
from .genome import GASearchConfig
from .operators import GeneticOperators


def setup_ga_logger(output_dir: str | Path) -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("MetaFieldGA")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(output_dir / "ga_run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    return logger


class GARunner:
    def __init__(
        self,
        config: GASearchConfig,
        evaluator: SurrogateEvaluator,
        encoder: StructureEncoder,
        operators: GeneticOperators,
        logger: logging.Logger,
    ):
        self.config = config
        self.evaluator = evaluator
        self.encoder = encoder
        self.operators = operators
        self.log = logger
        self.operators.load_bounds_and_manifold()

    def evaluate_population(self, pop: list[np.ndarray]) -> np.ndarray:
        cfg = self.config
        n = len(pop)

        def encode_one(i: int):
            return i, *self.encoder.genome_to_input(pop[i], cfg.z_source_m)

        inputs = [None] * n
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
            futures = {pool.submit(encode_one, i): i for i in range(n)}
            for future in as_completed(futures):
                result = future.result()
                inputs[result[0]] = result

        groups: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {}
        for idx, x, cond, mask, padded_size in inputs:
            groups.setdefault(padded_size, []).append((idx, x, cond, mask))

        fitness = np.zeros(n, dtype=np.float64)
        for items in groups.values():
            indices = [item[0] for item in items]
            xs = [item[1] for item in items]
            conds = [item[2] for item in items]
            masks = [item[3] for item in items]
            for start in range(0, len(indices), cfg.batch_size):
                end = min(start + cfg.batch_size, len(indices))
                preds = self.evaluator.evaluate_batch(xs[start:end], conds[start:end])
                for offset, pred in enumerate(preds):
                    idx = indices[start + offset]
                    base_fit = compute_fitness(pred, masks[start + offset])
                    penalty = self.operators.manifold_penalty(pop[idx])
                    fitness[idx] = max(0.0, base_fit - penalty)
        return fitness

    def _save_checkpoint(self, generation: int, pop: list[np.ndarray], fitness: np.ndarray, best_genome: np.ndarray, best_fitness: float, history: list[dict]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "ga_log.csv"
        write_header = not csv_path.exists()

        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    [
                        "gen",
                        "best_fitness",
                        "mean_fitness",
                        "median_fitness",
                        "std_fitness",
                        "best_ever",
                        "best_wavelength_nm",
                        "best_r_top",
                        "best_r_bot",
                        "best_height",
                        "best_period",
                        "unique_materials",
                        "time_sec",
                    ]
                )
            row = history[-1]
            best_wavelength_nm = 299792.458 / best_genome[0]
            unique_materials = len(set(best_genome[5 : 5 + N_MAT_SLOTS].astype(int)) - {0})
            writer.writerow(
                [
                    row["gen"],
                    f"{row['best']:.6e}",
                    f"{row['mean']:.6e}",
                    f"{row['median']:.6e}",
                    f"{row['std']:.6e}",
                    f"{row['best_ever']:.6e}",
                    f"{best_wavelength_nm:.1f}",
                    f"{best_genome[1]:.1f}",
                    f"{best_genome[2]:.1f}",
                    f"{best_genome[3]:.1f}",
                    f"{best_genome[4]:.1f}",
                    unique_materials,
                    f"{row['time']:.2f}",
                ]
            )

        summary = genome_to_summary(best_genome)
        summary["fitness"] = float(best_fitness)
        summary["generation"] = generation
        with open(output_dir / "best_structure.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        state = {
            "gen": generation,
            "best_genome": best_genome.tolist(),
            "best_fitness": float(best_fitness),
            "population": [genome.tolist() for genome in pop],
            "rng_state": self.operators.rng.bit_generator.state,
        }
        with open(output_dir / "ga_state.json", "w", encoding="utf-8") as handle:
            json.dump(state, handle)

    def _save_plots(self, generation: int, best_genome: np.ndarray, best_fitness: float) -> None:
        output_dir = Path(self.config.output_dir)
        csv_path = output_dir / "ga_log.csv"
        if csv_path.exists():
            gens, bests, means, evers = [], [], [], []
            with open(csv_path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        gens.append(int(row["gen"]))
                        bests.append(float(row["best_fitness"]))
                        means.append(float(row["mean_fitness"]))
                        evers.append(float(row["best_ever"]))
                    except (TypeError, ValueError):
                        continue

            if len(gens) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(gens, evers, "r-", lw=2, label="Best Ever")
                ax.plot(gens, bests, color="orange", alpha=0.6, label="Gen Best")
                ax.plot(gens, means, "b-", alpha=0.4, label="Gen Mean")
                ax.set_xlabel("Generation")
                ax.set_ylabel("|E|^2 Top 5% Volumetric")
                ax.set_title(f"GA Convergence @ Generation {generation}")
                ax.legend()
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "convergence.png", dpi=150)
                plt.close()

        try:
            x, cond, mask, padded_size = self.encoder.genome_to_input(best_genome, self.config.z_source_m)
            pred = self.evaluator.evaluate_batch([x], [cond])[0]
            mat_map = StructureEncoder.build_material_map(best_genome[1], best_genome[2], best_genome[3], best_genome[4], best_genome[5:].astype(int))
            num_z, num_x = mat_map.shape
            target_h, target_w = padded_size
            pad_top = (target_h - num_z) // 2
            pad_left = (target_w - num_x) // 2

            e_sq = pred[0] ** 2 + pred[1] ** 2 + pred[2] ** 2 + pred[3] ** 2
            e_sq_crop = e_sq[pad_top : pad_top + num_z, pad_left : pad_left + num_x]
            mask_crop = mask[0, pad_top : pad_top + num_z, pad_left : pad_left + num_x] > 0
            ex_crop = np.sqrt(pred[0] ** 2 + pred[1] ** 2)[pad_top : pad_top + num_z, pad_left : pad_left + num_x]
            ez_crop = np.sqrt(pred[2] ** 2 + pred[3] ** 2)[pad_top : pad_top + num_z, pad_left : pad_left + num_x]

            period_nm = best_genome[4]
            height_nm = best_genome[3]
            z_max = 750 if height_nm < 0 else 250
            z_min = -50 if height_nm < 0 else -550
            extent_nm = [-period_nm / 2, period_nm / 2, z_max, z_min]

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes[0, 0].imshow(mat_map, cmap="tab20", extent=extent_nm, aspect="equal")
            axes[0, 0].set_title(f"Material Map (P={period_nm:.1f}nm)")

            e_sq_vis = np.where(mask_crop, e_sq_crop, np.nan)
            axes[0, 1].imshow(np.log10(e_sq_vis + 1e-20), cmap="hot", extent=extent_nm, aspect="equal")
            axes[0, 1].set_title("log10|E|^2 (Volumetric Fluid)")

            axes[1, 0].imshow(ex_crop, cmap="inferno", extent=extent_nm, aspect="equal")
            axes[1, 0].set_title("|Ex| Full Field")

            axes[1, 1].imshow(ez_crop, cmap="inferno", extent=extent_nm, aspect="equal")
            axes[1, 1].set_title("|Ez| Full Field")

            plt.suptitle(
                f"Gen {generation} | WL={299792.458 / best_genome[0]:.1f}nm | "
                f"Rtop={best_genome[1]:.0f} Rbot={best_genome[2]:.0f} "
                f"H={best_genome[3]:.0f} P={best_genome[4]:.0f}nm | "
                f"Fit={best_fitness:.4e}",
                fontsize=12,
            )
            plt.tight_layout()
            plt.savefig(output_dir / "best_field.png", dpi=150)
            plt.close()
        except Exception as exc:  # pragma: no cover - visualization shouldn't break the run
            self.log.warning("Plot failed: %s", exc)

    def try_resume(self):
        state_path = Path(self.config.output_dir) / "ga_state.json"
        if not state_path.exists():
            return None
        try:
            with open(state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            pop = [np.array(item) for item in state["population"]]
            best_genome = np.array(state["best_genome"])
            best_fitness = state["best_fitness"]
            start_gen = state["gen"] + 1
            self.operators.rng.bit_generator.state = state["rng_state"]

            history: list[dict] = []
            csv_path = Path(self.config.output_dir) / "ga_log.csv"
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        history.append(
                            {
                                "gen": int(row["gen"]),
                                "best": float(row["best_fitness"]),
                                "mean": float(row["mean_fitness"]),
                                "median": float(row["median_fitness"]),
                                "std": float(row["std_fitness"]),
                                "best_ever": float(row["best_ever"]),
                                "time": float(row["time_sec"]),
                            }
                        )
            self.log.info("Resumed from generation %s", start_gen - 1)
            return pop, start_gen, best_genome, best_fitness, history
        except Exception:
            return None

    def run(self) -> tuple[np.ndarray, float, list[dict]]:
        cfg = self.config
        self.log.info("GA Config: pop=%s, generations=%s", cfg.population_size, cfg.generations)
        resumed = self.try_resume()
        if resumed:
            pop, start_gen, best_ever_genome, best_ever_fitness, history = resumed
        else:
            pop = self.operators.init_population()
            start_gen, best_ever_fitness, best_ever_genome, history = 0, -np.inf, None, []

        for generation in range(start_gen, cfg.generations):
            start_time = time.time()
            fitness = self.evaluate_population(pop)
            gen_best_idx = np.argmax(fitness)
            gen_best_fit = fitness[gen_best_idx]
            gen_mean, gen_median, gen_std = np.mean(fitness), np.median(fitness), np.std(fitness)
            improved = False

            if gen_best_fit > best_ever_fitness:
                best_ever_fitness = gen_best_fit
                best_ever_genome = pop[gen_best_idx].copy()
                improved = True

            elapsed = time.time() - start_time
            history.append(
                {
                    "gen": generation,
                    "best": gen_best_fit,
                    "mean": gen_mean,
                    "median": gen_median,
                    "std": gen_std,
                    "best_ever": best_ever_fitness,
                    "time": elapsed,
                }
            )
            marker = "*" if improved else " "
            self.log.info(
                "%s[Gen %04d/%04d] Best:%0.4e Ever:%0.4e | %0.1fs",
                marker,
                generation,
                cfg.generations,
                gen_best_fit,
                best_ever_fitness,
                elapsed,
            )

            self._save_checkpoint(generation, pop, fitness, best_ever_genome, best_ever_fitness, history)
            if generation % 25 == 0 or improved or generation == cfg.generations - 1:
                self._save_plots(generation, best_ever_genome, best_ever_fitness)

            ranked = np.argsort(-fitness)
            new_pop = [pop[ranked[idx]].copy() for idx in range(cfg.elite_count)]
            while len(new_pop) < cfg.population_size:
                parent_a = self.operators.select(pop, fitness)
                parent_b = self.operators.select(pop, fitness)
                child = self.operators.mutate(self.operators.crossover(parent_a, parent_b))
                new_pop.append(child)
            pop = new_pop

        return best_ever_genome, float(best_ever_fitness), history

    def save_final_results(self, best_genome: np.ndarray, best_fitness: float, history: list[dict]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(asdict(self.config), handle, indent=2, default=str)

        total_time = sum(item["time"] for item in history)
        summary = genome_to_summary(best_genome)
        summary["best_fitness"] = float(best_fitness)
        summary["total_time_sec"] = total_time
        with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
