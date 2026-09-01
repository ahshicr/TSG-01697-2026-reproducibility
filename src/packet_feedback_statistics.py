#!/usr/bin/env python3
"""Paired control-loop effects of communication backup duration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CONDITIONS = {
    "no_backup": Path("results/operational/boulder_packet_feedback_no_backup/rollout_scenarios.csv"),
    "backup_60s": Path("results/operational/boulder_packet_feedback_backup_60s/rollout_scenarios.csv"),
    "backup_300s": Path("results/operational/boulder_packet_feedback_backup_300s/rollout_scenarios.csv"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    samples = rng.integers(0, len(delta), size=(replicates, len(delta)))
    return tuple(np.quantile(delta[samples].mean(axis=1), [0.025, 0.975]).tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/operational/packet_feedback_statistics_central.csv"),
    )
    parser.add_argument("--policy", default="pc_rollout")
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=41073)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    frames = []
    for condition, path in CONDITIONS.items():
        frame = pd.read_csv(path)
        frame = frame.loc[frame.policy.eq(args.policy)].copy()
        frame["condition"] = condition
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)

    summaries = []
    for (condition, group), part in full.groupby(["condition", "group"]):
        summaries.append(
            {
                "row_type": "summary",
                "condition": condition,
                "comparator": "",
                "group": group,
                "n_pairs": len(part),
                "cost_mean": float(part.cost.mean()),
                "mean_control_action_fraction": float(part.mean_control_action_fraction.mean()),
                "executed_to_requested_restoration": float(
                    part.executed_restoration.sum() / part.requested_restoration.sum()
                ),
                "service_continuity_mean": float(part.service_continuity.mean()),
                "cost_difference": "",
                "cost_bootstrap_ci95_low": "",
                "cost_bootstrap_ci95_high": "",
                "relative_cost_reduction_percent": "",
                "paired_t_p": "",
                "wilcoxon_p": "",
            }
        )

    comparisons = []
    reference = full.loc[full.condition.eq("no_backup")]
    for condition in ("backup_60s", "backup_300s"):
        candidate = full.loc[full.condition.eq(condition)]
        for group in ("all", "nominal", "single_domain", "cascade", "ood"):
            left = candidate if group == "all" else candidate.loc[candidate.group.eq(group)]
            right = reference if group == "all" else reference.loc[reference.group.eq(group)]
            paired = left.merge(right, on="scenario_id", suffixes=("_candidate", "_reference"))
            delta = (paired.cost_candidate - paired.cost_reference).to_numpy(dtype=float)
            low, high = bootstrap_ci(delta, rng, args.replicates)
            nonzero = delta[np.abs(delta) > 1e-12]
            comparisons.append(
                {
                    "row_type": "paired_comparison",
                    "condition": condition,
                    "comparator": "no_backup",
                    "group": group,
                    "n_pairs": len(delta),
                    "cost_mean": float(paired.cost_candidate.mean()),
                    "mean_control_action_fraction": float(
                        paired.mean_control_action_fraction_candidate.mean()
                    ),
                    "executed_to_requested_restoration": float(
                        paired.executed_restoration_candidate.sum()
                        / paired.requested_restoration_candidate.sum()
                    ),
                    "service_continuity_mean": float(paired.service_continuity_candidate.mean()),
                    "cost_difference": float(delta.mean()),
                    "cost_bootstrap_ci95_low": low,
                    "cost_bootstrap_ci95_high": high,
                    "relative_cost_reduction_percent": float(
                        -100 * delta.mean() / paired.cost_reference.mean()
                    ),
                    "paired_t_p": float(stats.ttest_rel(paired.cost_candidate, paired.cost_reference).pvalue),
                    "wilcoxon_p": float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries + comparisons).to_csv(args.output, index=False)
    manifest = {
        "policy": args.policy,
        "traffic_multiplier": 2.0,
        "packet_feedback": "effective on-time control delivery fraction derates requested automated actions before service and threat transitions",
        "paired_scenarios": 1024,
        "bootstrap_replicates": args.replicates,
        "seed": args.seed,
        "source_sha256": {name: sha256(path) for name, path in CONDITIONS.items()},
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
