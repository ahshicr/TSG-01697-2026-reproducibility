"""Measure score error and regret on fully evaluated route alternatives.

Small instances enumerate every ordered assignment to two interchangeable crews.
Primary-size instances evaluate every member of the unchanged common portfolio.
No fitted calibration or future information enters the selection score.
"""
from __future__ import annotations

import argparse
import itertools
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

import simulate_rollout_revised as sim
import service_route_scoring as service


def enumerate_routes(crew, packet):
    requested = crew["jobs"]
    delivered = crew["dispatch_uniform"] < packet[requested]
    jobs = requested[delivered]
    duration = np.rint(crew["service_h"][delivered] * 60.0).astype(int)
    depot = int(sim.G["crew_depot_index"])
    travel = sim.G["crew_travel_minutes"]
    seen = set()
    for ordering in itertools.permutations(range(len(jobs))):
        for cut in range(len(jobs) + 1):
            routes = tuple(sorted((ordering[:cut], ordering[cut:])))
            if routes in seen:
                continue
            seen.add(routes)
            completion = np.full(sim.G["adj"].shape[0], np.inf)
            travel_total = 0
            route_log = []
            for vehicle, route in enumerate(routes):
                clock = 0
                origin = depot
                visits = []
                for idx in route:
                    zone = int(jobs[idx])
                    move = int(travel[origin, zone])
                    clock += move + int(duration[idx])
                    travel_total += move
                    completion[zone] = clock / 60.0
                    visits.append(f"{zone}@{clock / 60.0:.3f}")
                    origin = zone
                travel_total += int(travel[origin, depot])
                route_log.append(f"v{vehicle}:" + "|".join(visits))
            key = repr(routes)
            yield key, {
                "requested_jobs": requested, "dispatched_jobs": jobs,
                "completion_by_zone": completion, "total_travel_h": travel_total / 60.0,
                "route_log": ";".join(route_log), "event_log": ";".join(
                    f"{z}@{completion[z]:.3f}" for z in jobs),
                "source_policy": "enumerated", "route_score": 0.0,
            }


def evaluate_task(task):
    scenario_id, group, mode = task
    rng = np.random.default_rng(sim.G["seed"] + scenario_id * 7919)
    firsts = sim.G["valid_first_hours"]
    first = int(firsts[int(rng.integers(0, len(firsts)))])
    initial, innovations = sim.scenario_inputs(rng, group)
    coefficients = {n: float(rng.uniform(*b)) for n, b in sim.G["transition_uncertainty_bounds"].items()}
    crew = sim.prepare_crew_scenario(rng, initial)
    packet = sim.packet_action_fraction(initial)
    if mode == "exhaustive":
        subset = np.arange(min(4, len(crew["jobs"])))
        crew = {key: value[subset] for key, value in crew.items()}
        plans = dict(enumerate_routes(crew, packet))
    else:
        plans = sim.route_candidate_plans(first, initial, np.zeros(initial.shape[1]), packet, crew)
    forecast = sim.forecast_at(first, sim.G["horizon"])
    rows = []
    for name, plan in plans.items():
        exposure_score = sim.projected_route_loss(initial, forecast, plan, sim.G["central_policy_coefficients"])
        score = service.projected_service_cost(initial, forecast, plan, sim.G["central_policy_coefficients"])
        plan = {**plan, "source_policy": name, "route_score": score}
        metrics = sim.evaluate_policy("forecast_matched", first, initial, innovations, crew,
            scenario_id % len(sim.G["smartds_mappings"]), coefficients, crew_plan_override=plan)
        rows.append({"mode": mode, "scenario_id": scenario_id, "group": group,
            "candidate": name, "predicted_score": score, "exposure_score": exposure_score,
            "realized_cost": metrics["cost"],
            "travel_h": metrics["crew_total_travel_h"], "completed": metrics["crew_jobs_completed"],
            "unserved_kwh": metrics["unserved_energy"], "grid_infeasible_hours": metrics["smartds_projected_infeasible_hours"]})
    best = min(range(len(rows)), key=lambda i: (rows[i]["predicted_score"], rows[i]["candidate"]))
    predicted = np.array([r["predicted_score"] for r in rows])
    realized = np.array([r["realized_cost"] for r in rows])
    residual = realized - predicted
    regret = float(realized[best] - realized.min())
    exposure_best = min(range(len(rows)), key=lambda i:(rows[i]['exposure_score'],rows[i]['candidate']))
    exposure_regret = float(realized[exposure_best]-realized.min())
    epsilon = float(np.abs(residual).max())
    span = float(np.ptp(residual))
    assert regret <= 2 * epsilon + 1e-7
    assert regret <= span + 1e-7
    assert all(r["grid_infeasible_hours"] == 0 for r in rows)
    # Since these service actions are identical, route overrides must give the
    # same physical cost for central and matched labels.
    if scenario_id == 0:
        chosen = {**plans[rows[best]["candidate"]], "source_policy": "check", "route_score": float(predicted[best])}
        check = sim.evaluate_policy("pc_rollout", first, initial, innovations, crew,
            scenario_id % len(sim.G["smartds_mappings"]), coefficients, crew_plan_override=chosen)
        assert check["cost"] == realized[best]
    return rows, {"mode": mode, "scenario_id": scenario_id, "group": group,
        "candidates": len(rows), "requested_jobs": len(crew["jobs"]),
        "selected_score": float(predicted[best]), "selected_cost": float(realized[best]),
        "minimum_cost": float(realized.min()), "realized_regret": regret,
        "exposure_regret": exposure_regret,
        "relative_regret_percent": 100 * regret / max(float(realized.min()), 1e-12),
        "maximum_absolute_score_error": epsilon, "twice_error_bound": 2 * epsilon,
        "centered_error_bound": span, "exact_optimum": bool(regret <= 1e-7),
        "bound_pass": bool(regret <= span + 1e-7)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/submission_service_20260905/route_error"))
    parser.add_argument("--forecast-results", type=Path, default=Path("results/real_ev_strict_20260905"))
    parser.add_argument("--calibration", type=Path, default=Path("results/calibration_strict_20260905/transition_parameter_uncertainty.csv"))
    parser.add_argument("--packets", type=Path, default=Path("results/packet_training_20260905/packet_network_scenarios.csv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exhaustive-scenarios", type=int, default=64)
    parser.add_argument("--portfolio-scenarios", type=int, default=128)
    args = parser.parse_args()
    if (args.output / "completion.json").exists():
        raise RuntimeError("Route error output already exists. Use a new explicit output path.")
    args.output.mkdir(parents=True, exist_ok=True)
    detailed, summary = [], []
    start = time.time()
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(sim.__file__),Path(service.__file__),Path(sim.solve_alpha.__code__.co_filename),
         args.calibration,args.packets,sim.read_best_forecast(args.forecast_results/'prediction_metrics.csv')]}
    (args.output/'specification.json').write_text(json.dumps(dict(
        exhaustive_scenarios=args.exhaustive_scenarios,portfolio_scenarios=args.portfolio_scenarios,
        source_sha256=sources,selected_score='continuous service cost',
        comparison_score='historical threat exposure',random_state=2026),indent=2),encoding='utf-8')
    for mode, count, crews in [("exhaustive", args.exhaustive_scenarios, 2), ("portfolio", args.portfolio_scenarios, 4)]:
        config = sim.build_parser().parse_args([])
        config.integer_crews = crews
        config.forecast_results = args.forecast_results
        config.robust_uncertainty_file = args.calibration
        config.packet_results = args.packets
        payload = sim.payload_from_args(config)
        hours = set(payload['forecast_index'])
        payload['valid_first_hours'] = np.asarray([hour for hour in payload['valid_first_hours']
            if all(hour+h in hours for h in range(payload['horizon']))])
        tasks = [(i, sim.GROUPS[i % 4], mode) for i in range(count)]
        with mp.Pool(args.workers, initializer=sim.init_worker, initargs=(payload,)) as pool:
            for index, (records, row) in enumerate(pool.imap_unordered(evaluate_task, tasks), 1):
                detailed.extend(records)
                summary.append(row)
                if index % 8 == 0 or index == count:
                    print(f"{mode}: {index}/{count}, elapsed {time.time()-start:.1f} s", flush=True)
    sim.write_csv(args.output / "candidate_results.csv", sorted(detailed, key=lambda r: (r["mode"], r["scenario_id"], r["candidate"])))
    sim.write_csv(args.output / "scenario_results.csv", sorted(summary, key=lambda r: (r["mode"], r["scenario_id"])))
    (args.output / "completion.json").write_text(json.dumps({
        "exhaustive_scenarios": args.exhaustive_scenarios, "portfolio_scenarios": args.portfolio_scenarios,
        "candidate_evaluations": len(detailed), "all_bounds_pass": all(r["bound_pass"] for r in summary),
        "source_sha256": sources,
        "seconds": time.time()-start,
        "interpretation": "Observed finite-instance error and regret, not a population uniform guarantee.",
        "enumeration": "All ordered assignments of at most four dispatched jobs to two interchangeable crews.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
