"""Crew-routing validation on geocoded Boulder charging-service locations.

The model uses integer crews, road-circuity travel-time estimates, explicit
service times, and OR-Tools vehicle routing. EAGLE-I county restoration traces
inform a scenario prior; they are not mislabelled as utility crew work orders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


ROOT = Path(__file__).resolve().parents[1]
EV_DATA = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_forecast_dataset.npz"
COORDS = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_station_coordinates.csv"
EVENTS = ROOT / "data" / "external" / "processed" / "eaglei_boulder" / "boulder_outage_events_threshold_sensitivity.csv"
FORECAST_DIR = ROOT / "results" / "real_ev"
BASE_POLICIES = ("static", "greedy", "forecast_matched", "pc_rollout", "oracle")
POLICIES = ("static", "greedy", "forecast_matched", "pc_rollout", "route_aware_pc", "oracle")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            value.update(chunk)
    return value.hexdigest()


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def best_forecast(directory: Path) -> tuple[Path, dict]:
    metrics = pd.read_csv(directory / "prediction_metrics.csv")
    metrics = metrics.loc[metrics["mode"].eq("plain")].copy()
    metrics["selection_score"] = metrics["mae_demand"] + 0.25 * metrics["mae_energy"]
    row = metrics.sort_values("selection_score").iloc[0].to_dict()
    path = directory / f"forecast_plain_seed{int(row['seed'])}.npz"
    return path, row


def solve_routes(
    travel_minutes: np.ndarray,
    service_minutes: np.ndarray,
    priority: np.ndarray,
    crews: int,
    time_limit_ms: int,
    *,
    deterministic: bool = False,
) -> dict:
    """Minimize travel plus priority-weighted task arrival time."""
    n = len(service_minutes)
    manager = pywrapcp.RoutingIndexManager(n, crews, 0)
    routing = pywrapcp.RoutingModel(manager)

    def travel_callback(from_index: int, to_index: int) -> int:
        a, b = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(travel_minutes[a, b])

    def time_callback(from_index: int, to_index: int) -> int:
        a, b = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(service_minutes[a] + travel_minutes[a, b])

    travel_index = routing.RegisterTransitCallback(travel_callback)
    time_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(travel_index)
    routing.AddDimension(time_index, 0, 72 * 60, True, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")
    scaled_priority = np.maximum(1, np.rint(200 * priority / max(float(priority.mean()), 1e-9))).astype(int)
    for node in range(1, n):
        index = manager.NodeToIndex(node)
        time_dimension.SetCumulVarSoftUpperBound(index, 0, int(scaled_priority[node]))

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        if deterministic
        else routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    if deterministic:
        # The integrated 4,096-scenario experiment must be bitwise rerunnable.
        # Returning the first feasible insertion solution avoids a wall-clock
        # stopping rule changing the selected route between otherwise identical
        # runs.  The standalone route-quality experiment retains local search.
        parameters.solution_limit = 1
    parameters.time_limit.FromMilliseconds(time_limit_ms)
    parameters.log_search = False
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("OR-Tools failed to produce a crew route")

    arrival = np.full(n, np.nan)
    completion = np.full(n, np.nan)
    route_rows = []
    total_travel = 0.0
    for vehicle in range(crews):
        index = routing.Start(vehicle)
        sequence = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            start_minute = int(solution.Value(time_dimension.CumulVar(index)))
            arrival[node] = min(arrival[node], start_minute) if np.isfinite(arrival[node]) else start_minute
            completion[node] = start_minute + service_minutes[node]
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            total_travel += float(travel_minutes[node, next_node])
            sequence.append({"node": int(node), "arrival_min": start_minute, "completion_min": int(completion[node])})
            index = next_index
        route_rows.append({"vehicle": vehicle, "sequence": sequence})
    return {
        "arrival_min": arrival,
        "completion_min": completion,
        "total_travel_min": total_travel,
        "routes": route_rows,
        "objective": int(solution.ObjectiveValue()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "operational" / "crew_routing")
    parser.add_argument("--forecast-dir", type=Path, default=FORECAST_DIR)
    parser.add_argument("--scenarios", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=36)
    parser.add_argument("--crews", nargs="+", type=int, default=[4, 12, 24])
    parser.add_argument("--service-scale", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--road-circuity", type=float, default=1.25)
    parser.add_argument("--speed-kmh", type=float, default=30.0)
    parser.add_argument("--horizon-h", type=float, default=6.0)
    parser.add_argument("--time-limit-ms", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = np.load(EV_DATA)
    station_ids = data["station_id"].astype(str)
    coords = pd.read_csv(COORDS).set_index("station_id").loc[station_ids]
    distance = haversine_km(coords["latitude"].to_numpy(), coords["longitude"].to_numpy())
    medoid = int(np.argmin(distance.sum(axis=1)))
    forecast_path, forecast_metrics = best_forecast(args.forecast_dir)
    forecast = np.load(forecast_path)
    pred, truth, forecast_indices = forecast["pred"], forecast["truth"], forecast["indices"]
    if pred.shape != truth.shape or pred.shape[2] != len(station_ids):
        raise RuntimeError("Forecast tensor is incompatible with Boulder stations")

    event_frame = pd.read_csv(EVENTS, parse_dates=["start", "end_exclusive"])
    event_frame = event_frame.loc[event_frame["threshold_customers"].eq(50)].copy()
    recovery = event_frame["post_peak_half_recovery_h"].dropna().to_numpy(dtype=float)
    if not len(recovery):
        recovery = (event_frame["duration_h"] / 2).to_numpy(dtype=float)
    recovery = np.clip(recovery, 0.25, 8.0)

    adjacency = data["adj"].astype(float)
    mean_energy = data["energy"][: int(data["split_train_end_index"])].mean(axis=0).astype(float)
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    route_examples = []
    jobs = min(args.jobs, len(station_ids))
    sampling_weight = mean_energy + 0.05 * max(float(mean_energy.mean()), 1e-6)
    sampling_weight /= sampling_weight.sum()

    for scenario in range(args.scenarios):
        row_index = int(rng.integers(0, len(forecast_indices)))
        first_hour = int(forecast_indices[row_index])
        selected = np.sort(rng.choice(len(station_ids), size=jobs, replace=False, p=sampling_weight))
        base_severity = float(
            np.clip(
                np.log1p(rng.choice(event_frame["peak_customers_out"].to_numpy(dtype=float)))
                / np.log1p(event_frame["peak_customers_out"].quantile(0.99)),
                0.2,
                1.0,
            )
        )
        threat = np.zeros(len(station_ids), dtype=float)
        threat[selected] = base_severity * rng.uniform(0.65, 1.0, size=jobs)
        propagated = threat + 0.60 * (adjacency @ threat) + 0.30 * (adjacency @ (adjacency @ threat))
        pred_exposure = pred[row_index, :, :, 1].sum(axis=0) + 0.25 * pred[row_index, :, :, 0].sum(axis=0)
        true_exposure = truth[row_index, :, :, 1].sum(axis=0) + 0.25 * truth[row_index, :, :, 0].sum(axis=0)
        pred_norm = pred_exposure / max(float(np.mean(pred_exposure[selected])), 1e-6)
        true_norm = true_exposure / max(float(np.mean(true_exposure[selected])), 1e-6)
        true_risk_full = propagated * (1.0 + true_norm)

        # Node zero is the staging depot; jobs follow at nodes 1..jobs.
        job_distance = distance[np.ix_(selected, selected)]
        depot_to_job = distance[medoid, selected]
        route_distance = np.zeros((jobs + 1, jobs + 1), dtype=float)
        route_distance[0, 1:] = depot_to_job
        route_distance[1:, 0] = depot_to_job
        route_distance[1:, 1:] = job_distance
        travel_minutes = np.rint(route_distance * args.road_circuity / args.speed_kmh * 60).astype(int)

        base_service = rng.choice(recovery, size=jobs, replace=True)
        score_full = {
            "static": mean_energy + mean_energy.mean(),
            "greedy": threat + 1e-6,
            "forecast_matched": threat * (1.0 + pred_norm) + 1e-6,
            "pc_rollout": propagated * (1.0 + pred_norm) + 1e-6,
            "oracle": true_risk_full + 1e-6,
        }
        for service_scale in args.service_scale:
            service_minutes = np.concatenate(
                [[0], np.rint(np.clip(base_service * service_scale, 0.25, 16.0) * 60).astype(int)]
            )
            for crews in args.crews:
                routed_candidates = {}
                unrouted_candidates = {}
                for policy in BASE_POLICIES:
                    priority = np.concatenate([[0.0], score_full[policy][selected]])
                    routed_candidates[policy] = solve_routes(
                        travel_minutes, service_minutes, priority, crews, args.time_limit_ms
                    )
                    unrouted_candidates[policy] = solve_routes(
                        np.zeros_like(travel_minutes),
                        service_minutes,
                        priority,
                        crews,
                        max(40, args.time_limit_ms // 2),
                    )

                # Route-aware PC is an information-feasible candidate-portfolio
                # policy.  It evaluates routes produced by all non-oracle
                # heuristics under the PC-predicted integrated-risk objective
                # and executes the lowest-risk candidate.  Thus routing is
                # inside the decision, rather than assessed only afterwards.
                predicted_risk = score_full["pc_rollout"][selected]
                eligible = ("static", "greedy", "forecast_matched", "pc_rollout")
                route_source = min(
                    eligible,
                    key=lambda candidate: float(
                        np.sum(
                            predicted_risk
                            * np.minimum(
                                routed_candidates[candidate]["completion_min"][1:] / 60.0,
                                args.horizon_h,
                            )
                        )
                    ),
                )
                unrouted_source = min(
                    eligible,
                    key=lambda candidate: float(
                        np.sum(
                            predicted_risk
                            * np.minimum(
                                unrouted_candidates[candidate]["completion_min"][1:] / 60.0,
                                args.horizon_h,
                            )
                        )
                    ),
                )
                routed_candidates["route_aware_pc"] = routed_candidates[route_source]
                unrouted_candidates["route_aware_pc"] = unrouted_candidates[unrouted_source]

                for policy in POLICIES:
                    solution = routed_candidates[policy]
                    unrouted_solution = unrouted_candidates[policy]
                    completion_h = solution["completion_min"][1:] / 60.0
                    unrouted_completion_h = unrouted_solution["completion_min"][1:] / 60.0
                    true_risk = true_risk_full[selected]
                    integrated_risk = float(np.sum(true_risk * np.minimum(completion_h, args.horizon_h)))
                    unrouted_integrated_risk = float(
                        np.sum(true_risk * np.minimum(unrouted_completion_h, args.horizon_h))
                    )
                    weighted_completion = float(np.sum(true_risk * completion_h) / max(true_risk.sum(), 1e-9))
                    rows.append(
                        {
                            "scenario_id": scenario,
                            "first_hour": first_hour,
                            "timestamp_local": str(data["timestamp_local"][first_hour]),
                            "jobs": jobs,
                            "crews": crews,
                            "service_scale": service_scale,
                            "policy": policy,
                            "route_source_policy": route_source if policy == "route_aware_pc" else policy,
                            "integrated_risk_6h": integrated_risk,
                            "continuous_instant_integrated_risk_6h": 0.0,
                            "integer_unrouted_integrated_risk_6h": unrouted_integrated_risk,
                            "routing_risk_penalty_6h": integrated_risk - unrouted_integrated_risk,
                            "risk_weighted_completion_h": weighted_completion,
                            "jobs_completed_within_6h": int(np.count_nonzero(completion_h <= args.horizon_h)),
                            "risk_fraction_completed_within_6h": float(
                                true_risk[completion_h <= args.horizon_h].sum() / max(true_risk.sum(), 1e-9)
                            ),
                            "max_completion_h": float(completion_h.max()),
                            "total_travel_h": float(solution["total_travel_min"] / 60.0),
                            "solver_objective": solution["objective"],
                        }
                    )
                    if scenario == 0 and service_scale == 1.0 and crews in (12, 24):
                        route_examples.append(
                            {
                                "scenario_id": scenario,
                                "policy": policy,
                                "crews": crews,
                                "selected_station_ids": station_ids[selected].tolist(),
                                "routes": solution["routes"],
                            }
                        )
        print(f"scenario {scenario + 1}/{args.scenarios} complete")

    scenarios_path = args.out / "crew_routing_scenarios.csv"
    write_csv(scenarios_path, rows)
    frame = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in frame.groupby(["crews", "service_scale", "policy"]):
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
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(
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
    write_csv(summary_path, summary_rows)
    examples_path = args.out / "crew_route_examples.json"
    examples_path.write_text(json.dumps(route_examples, indent=2), encoding="utf-8")
    manifest = {
        "solver": "Google OR-Tools vehicle routing 9.15.6755",
        "routing": {
            "integer_crews": args.crews,
            "road_distance": f"haversine distance x {args.road_circuity} circuity",
            "speed_kmh": args.speed_kmh,
            "depot": f"station-coordinate medoid ({station_ids[medoid]})",
            "objective": "road travel plus priority-weighted task arrival time",
            "time_limit_ms_per_solve": args.time_limit_ms,
            "comparators": {
                "continuous_instant": "original divisible/immediate restoration idealization (risk removed at t=0)",
                "integer_unrouted": "same integer crews and service times with zero travel matrix",
                "integer_routed": "integer crews, service times, and geocoded travel matrix",
            },
            "state_update": (
                "A job's true risk accumulates only until its repair completion event, then is set to zero; "
                "integrated risk therefore changes causally with the routed completion schedule."
            ),
        },
        "service_time_prior": {
            "source": EVENTS.relative_to(ROOT).as_posix(),
            "construction": (
                "Empirical EAGLE-I Boulder event post-peak half-recovery times, clipped to 0.25-8 h, "
                "used as a scenario prior rather than asserted crew work-order durations."
            ),
            "sensitivity_multipliers": args.service_scale,
        },
        "forecast": {"path": forecast_path.relative_to(ROOT).as_posix(), "selection": forecast_metrics},
        "scenarios": {"count": args.scenarios, "jobs": jobs, "policies": POLICIES, "rows": len(rows)},
        "outputs": {
            scenarios_path.name: {"bytes": scenarios_path.stat().st_size, "sha256": sha256(scenarios_path)},
            summary_path.name: {"bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
            examples_path.name: {"bytes": examples_path.stat().st_size, "sha256": sha256(examples_path)},
        },
    }
    manifest_path = args.out / "crew_routing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
