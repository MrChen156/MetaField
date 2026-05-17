"""Lightweight throughput benchmark entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from benchmarks import ThroughputBenchmarkConfig, benchmark_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bench/throughput.yaml")
    args = parser.parse_args()

    with open(Path(args.config), "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    config = ThroughputBenchmarkConfig.from_dict(payload)
    summary = benchmark_model(config)

    print(f"Device: {summary['device_name']}")
    print(f"Model params: {summary['model_params_million']:.1f}M")
    print(f"Optimal batch: {summary['optimal_batch']}")
    print(f"Peak throughput: {summary['peak_throughput']:.0f} structures/sec")
    print(f"Latency: {summary['latency_ms_per_struct']:.2f} ms/struct")
    print(f"Result saved to: {config.output_path}")


if __name__ == "__main__":
    main()
