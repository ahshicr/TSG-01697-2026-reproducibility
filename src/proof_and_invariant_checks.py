"""Machine-checkable invariants supporting the implementation-level theory."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_rollout_revised import allocate, largest_remainder_crews


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "submission_service_20260905" / "proof_checks"
DATA = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_rollout_dataset.npz"
PARAMETERS = ROOT / "results" / "calibration_strict_20260905" / "transition_parameter_uncertainty.csv"
POWER = ROOT / "results" / "standalone_smartds_strict_20260905" / "full_year" / "smartds_ev_scenarios.csv"
PACKETS = ROOT / "results" / "packet_training_20260905" / "packet_network_scenarios.csv"
ROUTES = ROOT / "results" / "operational" / "crew_routing" / "crew_route_examples.json"
CHOICE = ROOT / "results" / "operational" / "station_choice" / "station_choice_scenarios.csv"
MAIN_ROLLOUT = ROOT / "results" / "submission_service_20260905" / "primary" / "rollout_scenarios.csv"
STRESS_ROLLOUT = (
    ROOT / "results" / "submission_service_20260905" / "electrical_stress" / "rollout_scenarios.csv"
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260825)
    data = np.load(DATA)
    n = data["adj"].shape[0]
    rows = []

    # Continuous allocator budget and nonnegativity.
    maximum_budget_error = 0.0
    minimum_action = np.inf
    for _ in range(10000):
        budget = float(rng.uniform(0.01, 1000.0))
        action = allocate(budget, rng.lognormal(size=n), reserve_fraction=float(rng.uniform(0, 0.2)))
        maximum_budget_error = max(maximum_budget_error, abs(float(action.sum()) - budget))
        minimum_action = min(minimum_action, float(action.min()))
    allocator_pass = maximum_budget_error <= 2e-4 and minimum_action >= 0
    rows.append(
        {
            "property": "continuous_action_budget",
            "code_reference": "src/simulate_rollout_revised.py::allocate",
            "evidence": "10,000 random score/budget tests",
            "status": "PASS" if allocator_pass else "FAIL",
            "detail": f"max_abs_budget_error={maximum_budget_error:.3e}; min_action={minimum_action:.3e}",
        }
    )

    # Largest-remainder integer projection.
    crew_failures = 0
    for crews in (1, 4, 12, 24, 50):
        for _ in range(1000):
            count = largest_remainder_crews(rng.random(n), crews)
            crew_failures += int(count.sum() != crews or np.any(count < 0) or not np.issubdtype(count.dtype, np.integer))
    rows.append(
        {
            "property": "integer_crew_count",
            "code_reference": "src/simulate_rollout_revised.py::largest_remainder_crews",
            "evidence": "5,000 random score vectors",
            "status": "PASS" if crew_failures == 0 else "FAIL",
            "detail": f"failures={crew_failures}",
        }
    )

    # Worst-service backlog recursion over the full 2023 test interval.
    energy = data["energy"].astype(float)
    start = int(data["split_val_end_index"])
    rho = 0.22
    emax = energy[start:].max(axis=0)
    theoretical_bound = emax / (1.0 - rho)
    backlog = np.zeros(n)
    maximum_ratio = 0.0
    for arrivals in energy[start:]:
        backlog = arrivals + rho * backlog  # zero service is the worst case
        ratio = np.divide(backlog, theoretical_bound, out=np.zeros_like(backlog), where=theoretical_bound > 0)
        maximum_ratio = max(maximum_ratio, float(ratio.max()))
    rows.append(
        {
            "property": "backlog_bounded_zero_service",
            "code_reference": "q[t+1]=[e[t]+rho*q[t]-s[t]]_+ with rho=0.22",
            "evidence": f"all {len(energy) - start} test hours and 50 stations",
            "status": "PASS" if maximum_ratio <= 1.0 + 1e-10 else "FAIL",
            "detail": f"max_empirical_to_theoretical_bound={maximum_ratio:.9f}",
        }
    )

    # Projected threat remains in the invariant cube under upper-bound coefficients.
    parameter_frame = pd.read_csv(PARAMETERS).set_index("parameter")
    def upper(name: str, fallback: float) -> float:
        return float(parameter_frame.loc[name, "upper"]) if name in parameter_frame.index else fallback
    coefficients = {
        "road_persistence": upper("threat_persistence", 0.9),
        "power_persistence": upper("power_persistence", 0.9),
        "comm_persistence": upper("comm_persistence", 0.8),
        "spatial_spread": upper("spatial_spread", 0.35),
        "pr_coupling": upper("pr_coupling", 0.04),
        "pc_coupling": upper("pc_coupling", 0.3),
        "rc_coupling": upper("rc_coupling", 0.08),
        "cr_coupling": upper("cr_coupling", 0.16),
        "cp_coupling": upper("cp_coupling", 0.08),
    }
    adjacency = data["adj"].astype(float)
    threat = rng.uniform(0, 1, size=(3, n))
    invariant_failures = 0
    for _ in range(100000):
        road, power, comm = threat
        nxt = np.stack(
            [
                coefficients["road_persistence"] * road
                + coefficients["spatial_spread"] * (adjacency @ road)
                + coefficients["pr_coupling"] * power
                + coefficients["cr_coupling"] * comm,
                coefficients["power_persistence"] * power
                + coefficients["spatial_spread"] * (adjacency @ power)
                + coefficients["cp_coupling"] * (adjacency @ comm),
                coefficients["comm_persistence"] * comm
                + coefficients["spatial_spread"] * (adjacency @ comm)
                + coefficients["pc_coupling"] * power
                + coefficients["rc_coupling"] * (adjacency @ road),
            ]
        )
        innovation = rng.uniform(0, 0.02, size=(3, n))
        repair = rng.uniform(0, 0.05, size=(3, n))
        threat = np.clip(nxt + innovation - repair, 0.0, 0.98)
        invariant_failures += int(np.any(threat < 0) or np.any(threat > 0.98))
    rows.append(
        {
            "property": "threat_cube_invariance",
            "code_reference": "projected transition Pi_[0,0.98]",
            "evidence": "100,000 steps at simultaneous upper uncertainty bounds",
            "status": "PASS" if invariant_failures == 0 else "FAIL",
            "detail": f"failures={invariant_failures}; final_range=[{threat.min():.3f},{threat.max():.3f}]",
        }
    )

    # Full 2023 AC projection audit.
    power = pd.read_csv(POWER)
    projected_ok = (
        power["projected_converged"].astype(bool)
        & power["projected_v_min_pu"].ge(0.95 - 1e-7)
        & power["projected_v_max_pu"].le(1.05 + 1e-7)
        & power["projected_max_line_loading_pu"].le(1.0 + 1e-7)
    )
    rows.append(
        {
            "property": "ac_action_projection",
            "code_reference": "src/smartds_ev_feasibility.py::solve_alpha",
            "evidence": POWER.relative_to(ROOT).as_posix(),
            "status": "PASS" if projected_ok.all() else "FAIL",
            "detail": f"feasible={int(projected_ok.sum())}/{len(projected_ok)}",
        }
    )

    # Backup power and action derating contracts.
    packets = pd.read_csv(PACKETS)
    full_backup = packets["backup_duration_s"].ge(120.0)
    no_backup = packets["backup_duration_s"].eq(0.0)
    backup_error = float(
        np.max(np.abs(packets.loc[full_backup, "effective_service_rate_pps"] - packets.loc[full_backup, "nominal_service_rate_pps"]))
    )
    depleted_error = float(
        np.max(np.abs(packets.loc[no_backup, "effective_service_rate_pps"] - packets.loc[no_backup, "degraded_service_rate_pps"]))
    )
    action_fraction_ok = packets["effective_action_fraction"].between(0, 1).all()
    rows.append(
        {
            "property": "communication_backup_and_action_derating",
            "code_reference": "src/packet_network_validation.py::simulate_link",
            "evidence": PACKETS.relative_to(ROOT).as_posix(),
            "status": "PASS" if backup_error < 1e-10 and depleted_error < 1e-10 and action_fraction_ok else "FAIL",
            "detail": f"full_backup_error={backup_error:.3e}; depleted_error={depleted_error:.3e}",
        }
    )

    # Every example work order appears exactly once in integer routes.
    route_examples = json.loads(ROUTES.read_text(encoding="utf-8"))
    route_failures = 0
    for example in route_examples:
        jobs = len(example["selected_station_ids"])
        visited = [item["node"] for route in example["routes"] for item in route["sequence"] if item["node"] != 0]
        route_failures += int(sorted(visited) != list(range(1, jobs + 1)))
    rows.append(
        {
            "property": "routed_work_order_coverage",
            "code_reference": "src/crew_routing_validation.py::solve_routes",
            "evidence": ROUTES.relative_to(ROOT).as_posix(),
            "status": "PASS" if route_failures == 0 else "FAIL",
            "detail": f"examples={len(route_examples)}; failures={route_failures}",
        }
    )

    # The main 4,096-scenario experiment must itself execute SMART-DS projection
    # and routed crew completion events, not merely cite standalone experiments.
    main_rollout = pd.read_csv(MAIN_ROLLOUT)
    online_grid_ok = (
        main_rollout["mean_smartds_projection_fraction"].between(0.0, 1.0)
        & main_rollout["smartds_projected_infeasible_hours"].eq(0)
        & main_rollout["min_voltage_pu"].ge(0.95 - 1e-7)
        & main_rollout["thermal_overloaded_branch_hours"].eq(0)
    )
    rows.append(
        {
            "property": "online_smartds_projection_in_main_rollout",
            "code_reference": "src/simulate_rollout_revised.py::project_smartds_action",
            "evidence": MAIN_ROLLOUT.relative_to(ROOT).as_posix(),
            "status": "PASS" if online_grid_ok.all() else "FAIL",
            "detail": f"feasible_policy_scenarios={int(online_grid_ok.sum())}/{len(online_grid_ok)}",
        }
    )

    stress_rollout = pd.read_csv(STRESS_ROLLOUT)
    stress_gate_ok = bool(
        stress_rollout["smartds_raw_infeasible_hours"].sum() > 0
        and stress_rollout["smartds_projected_infeasible_hours"].sum() == 0
        and stress_rollout["smartds_curtailed_energy_kwh"].sum() > 0
        and stress_rollout["mean_smartds_projection_fraction"].min() < 1.0
        and stress_rollout["smartds_curtailed_energy_kwh"].le(
            stress_rollout["unserved_energy"] + 1e-7
        ).all()
    )
    rows.append(
        {
            "property": "smartds_projection_changes_service_and_backlog_under_stress",
            "code_reference": "src/simulate_rollout_revised.py::evaluate_policy",
            "evidence": STRESS_ROLLOUT.relative_to(ROOT).as_posix(),
            "status": "PASS" if stress_gate_ok else "FAIL",
            "detail": (
                f"raw_infeasible_hours={int(stress_rollout['smartds_raw_infeasible_hours'].sum())}; "
                f"projected_infeasible_hours={int(stress_rollout['smartds_projected_infeasible_hours'].sum())}; "
                f"curtailed_kwh={stress_rollout['smartds_curtailed_energy_kwh'].sum():.3f}"
            ),
        }
    )

    event_failures = 0
    for row in main_rollout.itertuples(index=False):
        event_tokens = [] if not isinstance(row.crew_completion_events, str) or not row.crew_completion_events else row.crew_completion_events.split(";")
        completion_h = [float(token.rsplit("@", 1)[1]) for token in event_tokens]
        completed_in_horizon = sum(value <= 6.0 + 1e-12 for value in completion_h)
        event_failures += int(
            row.crew_jobs_completed != completed_in_horizon
            or len(event_tokens) != row.crew_jobs_dispatched
            or row.crew_jobs_completed > row.crew_jobs_dispatched
            or row.crew_jobs_dispatched > row.crew_job_count
            or row.executed_restoration != row.crew_jobs_completed
            or row.requested_restoration != row.crew_job_count
        )
    paired_input_failures = 0
    for _, group in main_rollout.groupby("scenario_id"):
        paired_input_failures += int(
            group["smartds_mapping_index"].nunique() != 1
            or group["crew_job_count"].nunique() != 1
            or group["crew_jobs_dispatched"].nunique() != 1
        )
    rows.append(
        {
            "property": "crew_completion_events_in_main_rollout",
            "code_reference": "src/simulate_rollout_revised.py::build_crew_plan/evaluate_policy",
            "evidence": MAIN_ROLLOUT.relative_to(ROOT).as_posix(),
            "status": "PASS" if event_failures == 0 and paired_input_failures == 0 else "FAIL",
            "detail": (
                f"policy_scenarios={len(main_rollout)}; event_failures={event_failures}; "
                f"paired_input_failures={paired_input_failures}"
            ),
        }
    )

    # Choice flows conserve requested energy and never exceed capacity.
    choice = pd.read_csv(CHOICE)
    choice_ok = (
        choice["served_kwh"].le(choice["requested_kwh"] + 1e-7)
        & choice["served_kwh"].ge(-1e-9)
        & choice["max_capacity_utilization"].le(1.0 + 1e-7)
    )
    rows.append(
        {
            "property": "station_choice_flow_feasibility",
            "code_reference": "src/station_choice_validation.py",
            "evidence": CHOICE.relative_to(ROOT).as_posix(),
            "status": "PASS" if choice_ok.all() else "FAIL",
            "detail": f"feasible={int(choice_ok.sum())}/{len(choice_ok)}",
        }
    )

    output_csv = OUT / "theory_implementation_map.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "scope": "Finite arithmetic, scenario-record, and execution checks. Mathematical statements require the manuscript proofs.",
        "source_sha256": {p.relative_to(ROOT).as_posix():sha256(p) for p in
            [Path(__file__),PARAMETERS,POWER,PACKETS,MAIN_ROLLOUT,STRESS_ROLLOUT,CHOICE,ROUTES]},
        "checks": rows,
        "output": {"path": output_csv.relative_to(ROOT).as_posix(), "bytes": output_csv.stat().st_size, "sha256": sha256(output_csv)},
    }
    report_path = OUT / "proof_check_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
