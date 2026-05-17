"""Lightweight GA entrypoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from search.ga.fitness import SurrogateEvaluator
from search.ga.genome import GASearchConfig
from search.ga.operators import GeneticOperators
from search.ga.runner import GARunner, setup_ga_logger
from structure import MaterialDatabase, StructureConstraints, StructureEncoder


def load_config(config_path: str | Path) -> GASearchConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return GASearchConfig.from_dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/search/ga.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_ga_logger(config.output_dir)
    start_time = time.time()

    material_db = MaterialDatabase(config.materials_json, config.material_mapping_json)
    encoder = StructureEncoder(material_db)
    constraints = StructureConstraints(config.to_constraint_ranges())
    operators = GeneticOperators(config, constraints)
    evaluator = SurrogateEvaluator(config)
    runner = GARunner(config, evaluator, encoder, operators, logger)

    best_genome, best_fitness, history = runner.run()
    runner.save_final_results(best_genome, best_fitness, history)
    logger.info("Done. Total time: %.1fs", time.time() - start_time)


if __name__ == "__main__":
    main()
