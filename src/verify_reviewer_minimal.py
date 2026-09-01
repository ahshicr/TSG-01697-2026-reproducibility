#!/usr/bin/env python3
"""Verify the self-contained reviewer-minimal reproduction package."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(value: float, target: float, tolerance: float = 5e-7) -> bool:
    return abs(value - target) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.package_root.resolve()
    failures: list[str] = []

    manifest = root / "SHA256SUMS.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"size: {row['path']}")
            continue
        if sha256(path) != row["sha256"]:
            failures.append(f"sha256: {row['path']}")

    scenario_path = (
        root
        / "results/operational/boulder_robust_rollout/rollout_scenarios.csv"
    )
    scenarios = pd.read_csv(scenario_path, usecols=["scenario_id", "policy"])
    if scenarios["scenario_id"].nunique() != 4096 or len(scenarios) != 24576:
        failures.append("primary scenario count is not 4,096 x 6")

    paired = pd.read_csv(
        root
        / "results/operational/boulder_robust_rollout/paired_statistics_central.csv"
    )
    all_row = paired[(paired["group"] == "all") & (paired["metric"] == "cost")].iloc[0]
    if not close(float(all_row["relative_reduction_percent"]), 0.6327148):
        failures.append("central-PC principal reduction differs from 0.6327%")

    risk = pd.read_csv(
        root
        / "results/operational/boulder_robust_rollout/robust_vs_central_risk.csv"
    )
    primary = risk[(risk["condition"] == "primary") & (risk["group"] == "all")].iloc[0]
    if not close(float(primary["robust_minus_central_mean"]), 1.6830252):
        failures.append("robust-minus-central mean differs from 1.6830")
    if float(primary["robust_cvar95_cost"]) <= float(primary["central_cvar95_cost"]):
        failures.append("stored CVaR95 no longer supports the reported negative audit")

    if failures:
        print(f"FAIL ({len(failures)}):")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        f"PASS: {len(rows)} hashed files; 4,096 paired scenarios; "
        "central-PC and robust-positioning statistics verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
