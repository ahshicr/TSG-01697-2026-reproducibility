"""Station-choice and coordinated rerouting validation for Boulder EV demand."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_forecast_dataset.npz"
COORDS = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_station_coordinates.csv"


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


def no_choice(demand: np.ndarray, capacity: np.ndarray, open_station: np.ndarray) -> np.ndarray:
    flow = np.zeros((len(demand), len(demand)))
    served = np.minimum(demand, capacity) * open_station
    np.fill_diagonal(flow, served)
    return flow


def nearest_choice(
    demand: np.ndarray, capacity: np.ndarray, open_station: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    flow = np.zeros((len(demand), len(demand)))
    residual_capacity = capacity * open_station
    # Larger origins choose first; tie-breaking is stable by station index.
    for origin in np.argsort(-demand, kind="stable"):
        remaining = float(demand[origin])
        for destination in np.argsort(distance[origin], kind="stable"):
            if not open_station[destination] or remaining <= 1e-10:
                continue
            served = min(remaining, residual_capacity[destination])
            flow[origin, destination] += served
            residual_capacity[destination] -= served
            remaining -= served
    return flow


def logit_choice(
    demand: np.ndarray,
    capacity: np.ndarray,
    open_station: np.ndarray,
    distance: np.ndarray,
    distance_beta: float,
    loyalty: float,
    rounds: int = 12,
) -> np.ndarray:
    n = len(demand)
    flow = np.zeros((n, n))
    residual_demand = demand.astype(float).copy()
    residual_capacity = capacity * open_station
    attractiveness = np.log1p(capacity / max(float(np.median(capacity)), 1e-9))
    for _ in range(rounds):
        active_dest = residual_capacity > 1e-9
        if residual_demand.sum() <= 1e-9 or not active_dest.any():
            break
        proposed = np.zeros((n, n))
        for origin in np.flatnonzero(residual_demand > 1e-9):
            utility = -distance_beta * distance[origin] + attractiveness
            utility[origin] += loyalty
            utility[~active_dest] = -np.inf
            utility -= np.max(utility[active_dest])
            probability = np.zeros(n)
            probability[active_dest] = np.exp(utility[active_dest])
            probability /= probability.sum()
            proposed[origin] = residual_demand[origin] * probability
        destination_total = proposed.sum(axis=0)
        scale = np.minimum(1.0, residual_capacity / np.maximum(destination_total, 1e-12))
        accepted = proposed * scale[None, :]
        flow += accepted
        residual_demand = np.maximum(residual_demand - accepted.sum(axis=1), 0.0)
        residual_capacity = np.maximum(residual_capacity - accepted.sum(axis=0), 0.0)
    return flow


def coordinated_choice(
    demand: np.ndarray,
    capacity: np.ndarray,
    open_station: np.ndarray,
    distance: np.ndarray,
    distance_beta: float,
    loyalty: float,
) -> np.ndarray:
    """Solve a capacity-constrained origin-to-station minimum-cost flow."""
    origins = np.flatnonzero(demand > 1e-10)
    destinations = np.flatnonzero(open_station & (capacity > 1e-10))
    n_flow = len(origins) * len(destinations)
    n_vars = n_flow + len(origins)
    if not len(origins) or not len(destinations):
        return np.zeros((len(demand), len(demand)))
    cost = np.empty(n_vars, dtype=float)
    pairs = []
    cursor = 0
    for origin in origins:
        for destination in destinations:
            pairs.append((origin, destination))
            cost[cursor] = distance_beta * distance[origin, destination] + loyalty * (origin != destination)
            cursor += 1
    # Unserved energy receives a penalty larger than any cross-city trip.
    cost[n_flow:] = max(100.0, float(np.max(cost[:n_flow]) + 20.0))

    equality = lil_matrix((len(origins), n_vars), dtype=float)
    for i in range(len(origins)):
        equality[i, i * len(destinations) : (i + 1) * len(destinations)] = 1.0
        equality[i, n_flow + i] = 1.0
    inequality = lil_matrix((len(destinations), n_vars), dtype=float)
    for j in range(len(destinations)):
        inequality[j, j:n_flow:len(destinations)] = 1.0
    result = linprog(
        cost,
        A_ub=inequality.tocsr(),
        b_ub=capacity[destinations],
        A_eq=equality.tocsr(),
        b_eq=demand[origins],
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Station-choice flow optimization failed: {result.message}")
    flow = np.zeros((len(demand), len(demand)))
    for value, (origin, destination) in zip(result.x[:n_flow], pairs):
        flow[origin, destination] = value
    return flow


def metrics(flow: np.ndarray, demand: np.ndarray, capacity: np.ndarray, distance: np.ndarray) -> dict:
    served_by_origin = flow.sum(axis=1)
    served_by_station = flow.sum(axis=0)
    total = float(demand.sum())
    rerouted = flow.copy()
    np.fill_diagonal(rerouted, 0.0)
    return {
        "requested_kwh": total,
        "served_kwh": float(flow.sum()),
        "unserved_kwh": float(np.maximum(demand - served_by_origin, 0).sum()),
        "served_fraction": float(flow.sum() / max(total, 1e-9)),
        "rerouted_kwh": float(rerouted.sum()),
        "rerouted_fraction": float(rerouted.sum() / max(flow.sum(), 1e-9)),
        "energy_weighted_extra_distance_km": float((flow * distance).sum() / max(flow.sum(), 1e-9)),
        "max_capacity_utilization": float(np.max(served_by_station / np.maximum(capacity, 1e-9))),
        "stations_at_capacity": int(np.count_nonzero(served_by_station >= capacity - 1e-7)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "operational" / "station_choice")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--closure-fraction", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--demand-multiplier", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--distance-beta", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--loyalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA)
    timestamps = data["timestamp_local"]
    energy = data["energy"].astype(float)
    coords = pd.read_csv(COORDS).set_index("station_id").loc[data["station_id"].astype(str)]
    distance = haversine_km(coords["latitude"].to_numpy(), coords["longitude"].to_numpy())
    train_end = int(data["split_train_end_index"])
    # This is a conservative empirical service-capacity proxy, not a charger-nameplate claim.
    capacity = np.maximum(np.quantile(energy[:train_end], 0.995, axis=0) * 1.20, 7.2)
    test = np.flatnonzero((timestamps >= np.datetime64("2023-01-01")) & (energy.sum(axis=1) > 0))
    n_peak = args.hours // 2
    peak = test[np.argsort(energy[test].sum(axis=1))[-n_peak:]]
    rng = np.random.default_rng(args.seed)
    random = rng.choice(np.setdiff1d(test, peak), size=args.hours - n_peak, replace=False)
    hours = np.sort(np.concatenate([peak, random]))

    rows: list[dict] = []
    methods = ("no_choice", "nearest", "logit", "coordinated")
    mean_activity = energy[:train_end].mean(axis=0)
    closure_weight = mean_activity + 0.05 * max(float(mean_activity.mean()), 1e-6)
    closure_weight /= closure_weight.sum()
    for h_position, hour in enumerate(hours):
        for closure_fraction in args.closure_fraction:
            closed_count = max(1, int(round(len(capacity) * closure_fraction)))
            closed = rng.choice(len(capacity), closed_count, replace=False, p=closure_weight)
            open_station = np.ones(len(capacity), dtype=bool)
            open_station[closed] = False
            for demand_multiplier in args.demand_multiplier:
                demand = energy[hour] * demand_multiplier
                for distance_beta in args.distance_beta:
                    flows = {
                        "no_choice": no_choice(demand, capacity, open_station),
                        "nearest": nearest_choice(demand, capacity, open_station, distance),
                        "logit": logit_choice(
                            demand, capacity, open_station, distance, distance_beta, args.loyalty
                        ),
                        "coordinated": coordinated_choice(
                            demand, capacity, open_station, distance, distance_beta, args.loyalty
                        ),
                    }
                    for method in methods:
                        rows.append(
                            {
                                "hour_index": int(hour),
                                "timestamp_local": str(timestamps[hour]),
                                "closure_fraction": closure_fraction,
                                "closed_stations": closed_count,
                                "demand_multiplier": demand_multiplier,
                                "distance_beta_per_km": distance_beta,
                                "loyalty_penalty": args.loyalty,
                                "method": method,
                                **metrics(flows[method], demand, capacity, distance),
                            }
                        )
        print(f"hour {h_position + 1}/{len(hours)} complete")

    scenarios_path = args.out / "station_choice_scenarios.csv"
    write_csv(scenarios_path, rows)
    frame = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in frame.groupby(
        ["closure_fraction", "demand_multiplier", "distance_beta_per_km", "method"]
    ):
        closure, demand_multiplier, beta, method = keys
        for metric in (
            "served_fraction",
            "rerouted_fraction",
            "energy_weighted_extra_distance_km",
            "max_capacity_utilization",
        ):
            values = group[metric].to_numpy(dtype=float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(
                {
                    "closure_fraction": closure,
                    "demand_multiplier": demand_multiplier,
                    "distance_beta_per_km": beta,
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": sd,
                    "ci95_low": float(values.mean() - 1.96 * sd / np.sqrt(len(values))),
                    "ci95_high": float(values.mean() + 1.96 * sd / np.sqrt(len(values))),
                    "n": len(values),
                }
            )
    summary_path = args.out / "station_choice_summary.csv"
    write_csv(summary_path, summary_rows)
    manifest = {
        "choice_models": {
            "no_choice": "original station only",
            "nearest": "capacity-aware sequential nearest available station",
            "logit": "iterative capacity-aware distance/loyalty multinomial logit",
            "coordinated": "global capacity-constrained minimum-cost flow (HiGHS)",
        },
        "capacity_proxy": (
            "max(7.2 kW, 1.2 x station training-period 99.5th percentile reconstructed load); "
            "stress proxy, not reported charger nameplate capacity"
        ),
        "sensitivity": {
            "closure_fraction": args.closure_fraction,
            "demand_multiplier": args.demand_multiplier,
            "distance_beta_per_km": args.distance_beta,
            "loyalty_penalty": args.loyalty,
        },
        "hours": len(hours),
        "rows": len(rows),
        "outputs": {
            scenarios_path.name: {"bytes": scenarios_path.stat().st_size, "sha256": sha256(scenarios_path)},
            summary_path.name: {"bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
        },
    }
    manifest_path = args.out / "station_choice_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
