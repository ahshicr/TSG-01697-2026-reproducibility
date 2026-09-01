#!/usr/bin/env python3
"""Audit whether worst-set robust PC improves tail risk over central PC.

The analysis is deliberately paired at scenario level.  Upper-tail risk is
reported for cost (larger is worse), and the direct comparison is always
``robust_pc_rollout - pc_rollout``.  Negative differences therefore favor the
robust extension.  The script evaluates the primary run, prespecified threat
groups, and every integrated misspecification/crew sensitivity run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POLICIES = ("forecast_matched", "pc_rollout", "robust_pc_rollout")
SENSITIVITY_CONDITIONS = (
    "matrix_scale_075",
    "matrix_scale_125",
    "matrix_noise_015",
    "integer_crews_12",
    "integer_crews_24",
)


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 250
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        index = rng.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = values[index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def upper_cvar(values: np.ndarray, probability: float) -> float:
    threshold = float(np.quantile(values, probability))
    return float(values[values >= threshold].mean())


def analyse_subset(
    data: pd.DataFrame,
    condition: str,
    group: str,
    seed: int,
    replicates: int,
) -> dict[str, float | int | str]:
    pivot = data.pivot(index="scenario_id", columns="policy", values="cost")
    pivot = pivot[list(POLICIES)].dropna()
    robust = pivot["robust_pc_rollout"].to_numpy(dtype=np.float64)
    central = pivot["pc_rollout"].to_numpy(dtype=np.float64)
    matched = pivot["forecast_matched"].to_numpy(dtype=np.float64)
    difference = robust - central
    low, high = bootstrap_mean_ci(difference, seed, replicates)
    tolerance = 1e-10
    return {
        "condition": condition,
        "group": group,
        "n_pairs": len(pivot),
        "matched_mean_cost": float(matched.mean()),
        "central_mean_cost": float(central.mean()),
        "robust_mean_cost": float(robust.mean()),
        "central_reduction_vs_matched_percent": float(100 * (matched.mean() - central.mean()) / matched.mean()),
        "robust_reduction_vs_matched_percent": float(100 * (matched.mean() - robust.mean()) / matched.mean()),
        "robust_minus_central_mean": float(difference.mean()),
        "robust_minus_central_ci95_low": low,
        "robust_minus_central_ci95_high": high,
        "robust_win_fraction": float(np.mean(difference < -tolerance)),
        "tie_fraction": float(np.mean(np.abs(difference) <= tolerance)),
        "robust_loss_fraction": float(np.mean(difference > tolerance)),
        "central_q95_cost": float(np.quantile(central, 0.95)),
        "robust_q95_cost": float(np.quantile(robust, 0.95)),
        "central_q99_cost": float(np.quantile(central, 0.99)),
        "robust_q99_cost": float(np.quantile(robust, 0.99)),
        "central_cvar95_cost": upper_cvar(central, 0.95),
        "robust_cvar95_cost": upper_cvar(robust, 0.95),
        "central_cvar99_cost": upper_cvar(central, 0.99),
        "robust_cvar99_cost": upper_cvar(robust, 0.99),
        "central_max_cost": float(central.max()),
        "robust_max_cost": float(robust.max()),
        "robust_minus_central_q95": float(np.quantile(difference, 0.95)),
        "robust_minus_central_q99": float(np.quantile(difference, 0.99)),
        "robust_minus_central_max": float(difference.max()),
        "robust_minus_central_min": float(difference.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/operational/boulder_robust_rollout/robust_vs_central_risk.csv"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    rows: list[dict[str, float | int | str]] = []

    primary = pd.read_csv(
        root / "results/operational/boulder_robust_rollout/rollout_scenarios.csv"
    )
    rows.append(
        analyse_subset(primary, "primary", "all", args.seed, args.bootstrap_replicates)
    )
    for group_index, group in enumerate(("nominal", "single_domain", "cascade", "ood"), start=1):
        rows.append(
            analyse_subset(
                primary[primary["group"] == group],
                "primary",
                group,
                args.seed + 1009 * group_index,
                args.bootstrap_replicates,
            )
        )

    for condition_index, condition in enumerate(SENSITIVITY_CONDITIONS, start=1):
        frame = pd.read_csv(
            root / f"results/revision_sensitivity/{condition}/rollout_scenarios.csv"
        )
        rows.append(
            analyse_subset(
                frame,
                condition,
                "all",
                args.seed + 100_003 * condition_index,
                args.bootstrap_replicates,
            )
        )
        rows.append(
            analyse_subset(
                frame[frame["group"] == "ood"],
                condition,
                "ood",
                args.seed + 100_003 * condition_index + 401,
                args.bootstrap_replicates,
            )
        )

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
