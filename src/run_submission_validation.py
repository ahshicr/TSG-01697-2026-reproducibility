"""Rebuild the integrated evidence with training-only policy normalization.

Conditions are fixed in this file before any new outcomes are read. Primary,
stress and legacy sensitivities retain all six policies. New ablations compare
central PC with the same-information baseline and retain the shared portfolio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

import simulate_rollout_revised as sim


UNSUPPORTED = ("comm_persistence", "pc_coupling", "rc_coupling", "cr_coupling", "cp_coupling")
CONDITIONS = {
    "primary": {"scenarios": 4096},
    "electrical_stress": {"energy_scale": 20.0},
    "backup_0": {"packet_traffic_multiplier": 2.0, "packet_backup_duration_s": 0.0},
    "backup_60": {"packet_traffic_multiplier": 2.0, "packet_backup_duration_s": 60.0},
    "backup_300": {"packet_traffic_multiplier": 2.0, "packet_backup_duration_s": 300.0},
    "matrix_scale_075": {"policy_matrix_scale": 0.75},
    "matrix_scale_125": {"policy_matrix_scale": 1.25},
    "matrix_noise_015": {"policy_matrix_noise": 0.15},
    "crews_12": {"integer_crews": 12},
    "crews_24": {"integer_crews": 24},
    **{f"zero_{name}": {"realized_zero": [name]} for name in UNSUPPORTED},
    "zero_all_unsupported": {"realized_zero": list(UNSUPPORTED)},
    "expanded_unsupported": {"realized_range_factor": 2.0},
    "nonlinear_ood": {"realized_transition_mode": "nonlinear_saturation"},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate_condition(task):
    """Identical exogenous random draws are retained across condition variants."""
    scenario_id, group = task
    rng = np.random.default_rng(sim.G["seed"] + scenario_id * 7919)
    candidates = sim.G["valid_first_hours"]
    first_hour = int(candidates[int(rng.integers(0, len(candidates)))])
    initial, innovations = sim.scenario_inputs(rng, group)
    coefficients = {
        name: float(rng.uniform(bounds[0], bounds[1]))
        for name, bounds in sim.G["transition_uncertainty_bounds"].items()
    }
    for name in sim.G.get("realized_zero", []):
        coefficients[name] = 0.0
    factor = sim.G.get("realized_range_factor", 1.0)
    for name in UNSUPPORTED:
        coefficients[name] *= factor
    crew = sim.prepare_crew_scenario(rng, initial)
    mapping = scenario_id % len(sim.G["smartds_mappings"])
    rows = []
    for policy in sim.G["policies"]:
        metrics = sim.evaluate_policy(policy, first_hour, initial, innovations, crew, mapping, coefficients)
        rows.append({
            "scenario_id": scenario_id, "group": group, "first_hour": first_hour,
            "policy": policy, "smartds_mapping_index": mapping,
            **{f"true_{name}": value for name, value in coefficients.items()}, **metrics,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/submission_20260905"))
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scenarios", type=int, default=None, help="Override counts for smoke tests only")
    args = parser.parse_args()
    sources = {p.name: digest(p) for p in [Path(__file__), Path(sim.__file__)]}
    for name in args.conditions:
        changes = CONDITIONS[name]
        out = args.output / name
        marker = out / "completion.json"
        count = args.scenarios or changes.get("scenarios", 1024)
        if marker.exists():
            old = json.loads(marker.read_text())
            if old["source_sha256"] == sources and old["scenario_count"] == count:
                assert digest(out / "rollout_scenarios.csv") == old["scenario_sha256"]
                print(f"Already complete {name}: {count} scenarios", flush=True)
                continue
            raise RuntimeError(f"Existing output differs from this specification: {out}")
        config = sim.build_parser().parse_args([])
        custom = {}
        for key, value in changes.items():
            if key.startswith("realized_"):
                custom[key] = value
            elif key != "scenarios":
                setattr(config, key, value)
        payload = sim.payload_from_args(config)
        payload.update(custom)
        if custom:
            payload["policies"] = ["forecast_matched", "pc_rollout"]
        out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        specification = {
            "condition": name, "changes": changes, "scenario_count": count,
            "source_sha256": sources, "policies": payload["policies"],
            "normalization_rule": payload["normalization_rule"],
            "normalization_train_end": payload["normalization_train_end"],
            "crew_prior_cutoff": payload["crew_prior_cutoff"],
            "crew_prior_events": payload["crew_prior_events"],
            "random_state": payload["seed"], "horizon": payload["horizon"],
            "central_policy_coefficients": payload["central_policy_coefficients"],
            "realized_reference_bounds": payload["transition_uncertainty_bounds"],
            "budgets": {k: payload[k] for k in ("total_charge", "total_comm", "total_restore")},
            "started_unix": started,
        }
        (out / "specification.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
        print(f"Starting {name}: {count} scenarios, {len(payload['policies'])} policies", flush=True)
        tasks = [(i, sim.GROUPS[i % len(sim.GROUPS)]) for i in range(count)]
        rows = []
        with mp.Pool(args.workers, initializer=sim.init_worker, initargs=(payload,)) as pool:
            for index, chunk in enumerate(pool.imap_unordered(simulate_condition, tasks, chunksize=4), 1):
                rows.extend(chunk)
                if index % 128 == 0 or index == count:
                    print(f"{name}: {index}/{count}, {time.time()-started:.1f} s", flush=True)
        rows.sort(key=lambda row: (row["scenario_id"], row["policy"]))
        assert len(rows) == count * len(payload["policies"])
        assert all(row["smartds_projected_infeasible_hours"] == 0 for row in rows)
        sim.write_csv(out / "rollout_scenarios.csv", rows)
        sim.write_csv(out / "rollout_summary.csv", sim.aggregate(rows))
        sim.write_csv(out / "rollout_manifest.csv", [{
            "condition": name, "scenarios": count, "horizon": payload["horizon"],
            "workers": args.workers, "seconds": time.time()-started,
            "policies": " ".join(payload["policies"]), "source_sha256": sources["simulate_rollout_revised.py"],
            "normalization_train_end": payload["normalization_train_end"],
            **{k: payload[k] for k in ("total_charge", "total_comm", "total_restore")},
        }])
        marker.write_text(json.dumps({**specification, "elapsed_s": time.time()-started,
            "scenario_sha256": digest(out / "rollout_scenarios.csv"), "rows": len(rows)}, indent=2), encoding="utf-8")
        print(f"Completed {name}", flush=True)


if __name__ == "__main__":
    main()
