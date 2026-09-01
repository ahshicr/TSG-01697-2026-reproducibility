#!/usr/bin/env python3
"""Evaluate policy rankings under prespecified alternative objective weights.

The simulator stores every additive cost primitive except the cascade term.
The latter is recovered algebraically from the reference objective.  No policy
is re-optimized under the alternative weights: this is deliberately an
evaluation-objective sensitivity analysis on the same paired trajectories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = {
    "mobility_delay": 1.8,
    "unserved_energy": 2.6,
    "comm_loss": 1.7,
    "power_service_loss": 1.2,
    "cascade": 12.0,
}

SCHEMES = {
    "reference": REFERENCE,
    "mobility_x2": {**REFERENCE, "mobility_delay": 3.6},
    "energy_x2": {**REFERENCE, "unserved_energy": 5.2},
    "communication_x2": {**REFERENCE, "comm_loss": 3.4},
    "power_service_x2": {**REFERENCE, "power_service_loss": 2.4},
    "cascade_half": {**REFERENCE, "cascade": 6.0},
}


def bootstrap_mean_ci(delta: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    chunk = 250
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        index = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start : start + size] = delta[index].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def add_primitives(frame: pd.DataFrame) -> pd.DataFrame:
    known = sum(
        REFERENCE[column] * frame[column]
        for column in ("mobility_delay", "unserved_energy", "comm_loss", "power_service_loss")
    )
    frame = frame.copy()
    frame["cascade"] = (frame["cost"] - known) / REFERENCE["cascade"]
    if (frame["cascade"] < -1e-4).any():
        raise ValueError("Recovered cascade primitive is negative; objective definition changed")
    frame["cascade"] = frame["cascade"].clip(lower=0.0)
    return frame


def score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(weights[column] * frame[column] for column in weights)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("results/operational/boulder_robust_rollout/rollout_scenarios.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/operational/boulder_robust_rollout/objective_weight_sensitivity_central.csv"),
    )
    parser.add_argument("--policy", default="pc_rollout")
    parser.add_argument("--comparators", nargs="+", default=["forecast_matched"])
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=41073)
    args = parser.parse_args()

    data = add_primitives(pd.read_csv(args.scenarios))
    rows: list[dict[str, float | int | str]] = []
    for scheme_index, (scheme, weights) in enumerate(SCHEMES.items()):
        subset = data.copy()
        subset["evaluation_cost"] = score(subset, weights)
        pivot = subset.pivot(index="scenario_id", columns="policy", values="evaluation_cost")
        for comparator_index, comparator in enumerate(args.comparators):
            paired = pivot[[args.policy, comparator]].dropna()
            delta = paired[args.policy].to_numpy() - paired[comparator].to_numpy()
            low, high = bootstrap_mean_ci(
                delta,
                args.seed + 101 * scheme_index + 1009 * comparator_index,
                args.bootstrap_replicates,
            )
            reference_mean = float(paired[comparator].mean())
            rows.append(
                {
                    "scheme": scheme,
                    "policy": args.policy,
                    "comparator": comparator,
                    "n_pairs": len(delta),
                    "policy_mean": float(paired[args.policy].mean()),
                    "comparator_mean": reference_mean,
                    "mean_difference": float(delta.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "relative_reduction_percent": float(-100.0 * delta.mean() / reference_mean),
                    "mobility_weight": weights["mobility_delay"],
                    "energy_weight": weights["unserved_energy"],
                    "communication_weight": weights["comm_loss"],
                    "power_service_weight": weights["power_service_loss"],
                    "cascade_weight": weights["cascade"],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
