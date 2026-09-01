#!/usr/bin/env python3
"""Combine independently seeded crew-routing batches into one audit table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = []
    source = []
    offset = 0
    for batch, directory in enumerate(args.inputs):
        path = directory / "crew_routing_scenarios.csv"
        frame = pd.read_csv(path)
        local_count = int(frame.scenario_id.nunique())
        frame["batch_id"] = batch
        frame["local_scenario_id"] = frame["scenario_id"]
        frame["scenario_id"] = frame["scenario_id"] + offset
        frames.append(frame)
        source.append({"path": path.as_posix(), "sha256": sha256(path), "scenarios": local_count})
        offset += local_count
    full = pd.concat(frames, ignore_index=True)
    scenario_path = args.out / "crew_routing_scenarios.csv"
    full.to_csv(scenario_path, index=False)

    rows = []
    for keys, group in full.groupby(["crews", "service_scale", "policy"]):
        crews, service_scale, policy = keys
        for metric in (
            "integrated_risk_6h",
            "integer_unrouted_integrated_risk_6h",
            "routing_risk_penalty_6h",
            "risk_weighted_completion_h",
            "risk_fraction_completed_within_6h",
            "total_travel_h",
        ):
            values = group[metric].to_numpy(dtype=float)
            sd = float(values.std(ddof=1))
            rows.append(
                {
                    "crews": crews,
                    "service_scale": service_scale,
                    "policy": policy,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": sd,
                    "ci95_low": float(values.mean() - 1.96 * sd / np.sqrt(len(values))),
                    "ci95_high": float(values.mean() + 1.96 * sd / np.sqrt(len(values))),
                    "n": len(values),
                }
            )
    summary_path = args.out / "crew_routing_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    manifest = {
        "scenario_count": int(full.scenario_id.nunique()),
        "rows": len(full),
        "independent_seed_batches": source,
        "outputs": {
            scenario_path.name: {"sha256": sha256(scenario_path), "bytes": scenario_path.stat().st_size},
            summary_path.name: {"sha256": sha256(summary_path), "bytes": summary_path.stat().st_size},
        },
    }
    (args.out / "crew_routing_combined_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
