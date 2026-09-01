#!/usr/bin/env python3
"""Paired inference for integer, routed restoration experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    samples = rng.integers(0, len(delta), size=(replicates, len(delta)))
    return tuple(np.quantile(delta[samples].mean(axis=1), [0.025, 0.975]).tolist())


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("results/operational/crew_routing/crew_routing_scenarios.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/operational/crew_routing/crew_routing_paired_statistics.csv"),
    )
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    frame = pd.read_csv(args.scenarios)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    comparisons = ("route_aware_pc", "pc_rollout")
    for policy in comparisons:
        for crews in sorted(frame.crews.unique()):
            for service_scale in sorted(frame.service_scale.unique()):
                subset = frame[(frame.crews == crews) & (frame.service_scale == service_scale)]
                wide = subset.pivot(index="scenario_id", columns="policy", values="integrated_risk_6h")
                delta = (wide[policy] - wide.forecast_matched).to_numpy(dtype=float)
                low, high = bootstrap_ci(delta, rng, args.replicates)
                nonzero = delta[np.abs(delta) > 1e-12]
                wilcoxon = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
                rows.append(
                    {
                    "comparison": f"{policy}_minus_forecast_matched",
                    "primary_family": policy == "route_aware_pc",
                    "crews": int(crews),
                    "service_scale": float(service_scale),
                    "n_pairs": len(delta),
                    "policy_mean": float(wide[policy].mean()),
                    "comparator_mean": float(wide.forecast_matched.mean()),
                    "mean_difference": float(delta.mean()),
                    "relative_reduction_percent": float(-100 * delta.mean() / wide.forecast_matched.mean()),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "paired_t_p": float(stats.ttest_rel(wide[policy], wide.forecast_matched).pvalue),
                    "wilcoxon_p": wilcoxon,
                    }
                )

    for name in ("paired_t_p", "wilcoxon_p"):
        primary_indices = [index for index, row in enumerate(rows) if row["primary_family"]]
        adjusted = holm([float(rows[index][name]) for index in primary_indices])
        for row in rows:
            row[f"holm_{name}"] = ""
        for index, value in zip(primary_indices, adjusted):
            rows[index][f"holm_{name}"] = value

    # The routing penalty is evaluated separately because its null value is
    # exactly zero and it quantifies the cost of making a plan executable.
    penalties = []
    for keys, group in frame.groupby(["crews", "service_scale", "policy"]):
        crews, service_scale, policy = keys
        values = group.routing_risk_penalty_6h.to_numpy(dtype=float)
        low, high = bootstrap_ci(values, rng, args.replicates)
        penalties.append(
            {
                "crews": int(crews),
                "service_scale": float(service_scale),
                "policy": policy,
                "n": len(values),
                "mean_routing_penalty": float(values.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "fraction_positive": float(np.mean(values > 1e-12)),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    penalty_path = args.output.with_name("crew_routing_penalty_statistics.csv")
    pd.DataFrame(penalties).to_csv(penalty_path, index=False)
    manifest = {
        "primary_family": "Route-aware PC versus forecast-matched integrated routed risk over 3 crew counts x 3 service-time scales",
        "multiplicity": "Holm correction across all nine prespecified crew/service comparisons",
        "bootstrap_replicates": args.replicates,
        "seed": args.seed,
        "outputs": {
            args.output.name: {"sha256": sha256(args.output), "bytes": args.output.stat().st_size},
            penalty_path.name: {"sha256": sha256(penalty_path), "bytes": penalty_path.stat().st_size},
        },
    }
    args.output.with_name("crew_routing_statistics_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
