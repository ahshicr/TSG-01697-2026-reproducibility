#!/usr/bin/env python3
"""Paired inference for the closed-loop restoration experiments.

The primary family is the PC-rollout versus forecast-matched comparison for
total cost, evaluated overall and in the four prespecified threat classes.
Holm correction is applied only to this five-test family.  Other endpoints and
comparators are explicitly labeled exploratory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = [
    ("cost", "lower"),
    ("mobility_delay", "lower"),
    ("unserved_energy", "lower"),
    ("comm_loss", "lower"),
    ("power_service_loss", "lower"),
    ("min_voltage_pu", "higher"),
    ("voltage_violation_pu_hours", "lower"),
    ("thermal_overload_pu_hours", "lower"),
    ("losses_mwh", "lower"),
    ("mean_smartds_projection_fraction", "higher"),
    ("smartds_curtailed_energy_kwh", "lower"),
    ("crew_completion_fraction", "higher"),
    ("crew_mean_completion_h", "lower"),
    ("crew_total_travel_h", "lower"),
    ("latency_ms", "lower"),
]
COMPARATORS = ["forecast_matched", "pc_rollout", "static", "greedy", "oracle"]
GROUPS = ["all", "nominal", "single_domain", "cascade", "ood"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_values(rows, group: str, policy: str, comparator: str, metric: str):
    selected = rows if group == "all" else [row for row in rows if row["group"] == group]
    by_key = {}
    for row in selected:
        by_key.setdefault(int(row["scenario_id"]), {})[row["policy"]] = float(row[metric])
    keys = sorted(key for key, values in by_key.items() if policy in values and comparator in values)
    proposed = np.asarray([by_key[key][policy] for key in keys], dtype=np.float64)
    reference = np.asarray([by_key[key][comparator] for key in keys], dtype=np.float64)
    return proposed, reference


def bootstrap_mean_ci(delta: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    if len(delta) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 250
    written = 0
    while written < replicates:
        size = min(chunk, replicates - written)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        means[written : written + size] = delta[indices].mean(axis=1)
        written += size
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def wilcoxon_p(delta: np.ndarray) -> float:
    nonzero = delta[np.abs(delta) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    return float(stats.wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def analyse(
    rows,
    seed: int,
    replicates: int,
    policy: str = "pc_rollout",
    comparators: list[str] | None = None,
    primary_comparator: str = "forecast_matched",
):
    comparators = COMPARATORS if comparators is None else comparators
    output = []
    for comparator in comparators:
        for group in GROUPS:
            for metric, direction in METRICS:
                proposed, ref = paired_values(rows, group, policy, comparator, metric)
                # Negative always means the selected policy is numerically smaller.
                delta = proposed - ref
                low, high = bootstrap_mean_ci(
                    delta,
                    seed + 1009 * GROUPS.index(group) + 9176 * METRICS.index((metric, direction)),
                    replicates,
                )
                if len(delta) > 1 and np.std(delta, ddof=1) > 0:
                    t_p = float(stats.ttest_rel(proposed, ref).pvalue)
                    dz = float(delta.mean() / np.std(delta, ddof=1))
                else:
                    t_p = 1.0
                    dz = 0.0
                ref_mean = float(ref.mean()) if len(ref) else float("nan")
                percent = float(-100.0 * delta.mean() / ref_mean) if abs(ref_mean) > 1e-12 else float("nan")
                output.append(
                    {
                        "policy": policy,
                        "comparator": comparator,
                        "group": group,
                        "metric": metric,
                        "direction": direction,
                        "n_pairs": len(delta),
                        "policy_mean": float(proposed.mean()),
                        "comparator_mean": ref_mean,
                        "mean_difference_pc_minus_comparator": float(delta.mean()),
                        "mean_difference_policy_minus_comparator": float(delta.mean()),
                        "mean_difference_bootstrap_ci95_low": low,
                        "mean_difference_bootstrap_ci95_high": high,
                        "relative_reduction_percent": percent,
                        "paired_cohens_dz": dz,
                        "paired_t_p": t_p,
                        "wilcoxon_p": wilcoxon_p(delta),
                        "primary_family": comparator == primary_comparator and metric == "cost",
                        "holm_p_primary_paired_t": "",
                        "holm_p_primary_wilcoxon": "",
                    }
                )

    primary_indices = [index for index, row in enumerate(output) if row["primary_family"]]
    t_adjusted = holm_adjust([float(output[index]["paired_t_p"]) for index in primary_indices])
    w_adjusted = holm_adjust([float(output[index]["wilcoxon_p"]) for index in primary_indices])
    for index, t_value, w_value in zip(primary_indices, t_adjusted, w_adjusted):
        output[index]["holm_p_primary_paired_t"] = t_value
        output[index]["holm_p_primary_wilcoxon"] = w_value
    return output


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("results/revision/rollout_scenarios.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/revision/paired_statistics.csv"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=41073)
    parser.add_argument("--policy", default="pc_rollout")
    parser.add_argument("--comparators", nargs="+", default=COMPARATORS)
    parser.add_argument("--primary-comparator", default="forecast_matched")
    args = parser.parse_args()
    rows = analyse(
        read_rows(args.scenarios),
        args.seed,
        args.bootstrap_replicates,
        policy=args.policy,
        comparators=args.comparators,
        primary_comparator=args.primary_comparator,
    )
    write_csv(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
