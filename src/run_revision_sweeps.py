#!/usr/bin/env python3
"""Run prespecified matrix-misspecification and discrete-crew checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


CONDITIONS = [
    ("matrix_scale_075", ["--policy-matrix-scale", "0.75"]),
    ("matrix_scale_125", ["--policy-matrix-scale", "1.25"]),
    ("matrix_noise_015", ["--policy-matrix-noise", "0.15"]),
    ("integer_crews_12", ["--integer-crews", "12"]),
    ("integer_crews_24", ["--integer-crews", "24"]),
]


def read_primary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["primary_family"] == "True" and row["group"] in {"all", "cascade", "ood"}
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/revision_sensitivity"))
    parser.add_argument("--scenarios", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    simulator = Path(__file__).with_name("simulate_rollout_revised.py")
    statistics_script = Path(__file__).with_name("paired_statistics.py")
    combined = []
    for condition, extra in CONDITIONS:
        destination = args.results_root / condition
        scenario_file = destination / "rollout_scenarios.csv"
        if args.force or not scenario_file.exists():
            command = [
                sys.executable,
                "-u",
                str(simulator),
                "--results",
                str(destination),
                "--scenarios",
                str(args.scenarios),
                "--workers",
                str(args.workers),
                *extra,
            ]
            subprocess.run(command, check=True)
        stats_file = destination / "paired_statistics.csv"
        subprocess.run(
            [
                sys.executable,
                str(statistics_script),
                "--scenarios",
                str(scenario_file),
                "--output",
                str(stats_file),
                "--bootstrap-replicates",
                str(args.bootstrap_replicates),
                "--policy",
                "robust_pc_rollout",
            ],
            check=True,
        )
        for row in read_primary(stats_file):
            combined.append({"condition": condition, **row})
    output = args.results_root / "sensitivity_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
