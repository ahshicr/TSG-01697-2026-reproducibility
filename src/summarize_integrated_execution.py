"""Summarize the online SMART-DS and crew-event evidence used by the revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(label: str, path: Path, horizon: int, load_multiplier: float) -> dict:
    frame = pd.read_csv(path)
    return {
        "condition": label,
        "load_multiplier": load_multiplier,
        "paired_scenarios": int(frame["scenario_id"].nunique()),
        "policy_scenario_rows": int(len(frame)),
        "online_smartds_policy_hours": int(len(frame) * horizon),
        "raw_infeasible_policy_hours": int(frame["smartds_raw_infeasible_hours"].sum()),
        "projected_infeasible_policy_hours": int(frame["smartds_projected_infeasible_hours"].sum()),
        "mean_projection_fraction": float(frame["mean_smartds_projection_fraction"].mean()),
        "minimum_policy_scenario_projection_fraction": float(
            frame["mean_smartds_projection_fraction"].min()
        ),
        "curtailed_energy_kwh": float(frame["smartds_curtailed_energy_kwh"].sum()),
        "mean_crew_jobs": float(frame["crew_job_count"].mean()),
        "mean_crew_jobs_dispatched": float(frame["crew_jobs_dispatched"].mean()),
        "mean_crew_jobs_completed_within_horizon": float(frame["crew_jobs_completed"].mean()),
        "mean_crew_completion_fraction": float(frame["crew_completion_fraction"].mean()),
        "mean_crew_completion_h": float(frame["crew_mean_completion_h"].mean()),
        "mean_crew_travel_h": float(frame["crew_total_travel_h"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "results" / "operational" / "boulder_robust_rollout" / "rollout_scenarios.csv",
    )
    parser.add_argument(
        "--stress",
        type=Path,
        default=ROOT
        / "results"
        / "operational"
        / "boulder_integrated_execution_stress"
        / "rollout_scenarios.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "operational" / "integrated_execution_summary.csv",
    )
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()
    rows = [
        summarize("measured_load_primary", args.primary, args.horizon, 1.0),
        summarize("20x_ev_execution_stress", args.stress, args.horizon, 20.0),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    manifest = {
        "purpose": (
            "Online execution evidence: every hourly charging action is solved by SMART-DS/OpenDSS; "
            "curtailment is applied before unserved energy and backlog; only routed crew completion events repair state."
        ),
        "horizon_h": args.horizon,
        "inputs": {
            str(args.primary.relative_to(ROOT)): sha256(args.primary),
            str(args.stress.relative_to(ROOT)): sha256(args.stress),
        },
        "output": {
            "path": str(args.out.relative_to(ROOT)),
            "bytes": args.out.stat().st_size,
            "sha256": sha256(args.out),
        },
    }
    manifest_path = args.out.with_name("integrated_execution_summary_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
