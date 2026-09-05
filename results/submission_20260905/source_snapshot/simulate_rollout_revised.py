#!/usr/bin/env python3
"""Closed-loop, information-matched rollout evaluation for the revision.

Key differences from the submitted experiment:

* actions are recomputed after every hourly observation;
* PC rollout never reads the realized future threat sequence;
* forecast-matched, central-PC, and robust-PC receive the same demand forecast,
  current threat observation, and finite route portfolio; static and greedy
  retain their separately declared information and route rules;
* SMART-DS/OpenDSS projects every hourly charging action before service is
  credited;
* integer crew routes are committed at the first decision and only completed
  repair events affect the subsequent threat trajectory;
* unmet charging is counted once in the stage objective; and
* electrical feasibility is evaluated after scoring and route outcomes are
  recorded separately from the model-based policy cost.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd

from crew_routing_validation import haversine_km
from smartds_ev_feasibility import COORDS as SMARTDS_COORDS
from smartds_ev_feasibility import MASTER as SMARTDS_MASTER
from smartds_ev_feasibility import apply_ev_load, compile_feeder, feasible, mapping_for_seed, solve_alpha
from smartds_ev_feasibility import dss as smartds_dss


ROOT = Path(__file__).resolve().parents[1]
CREW_EVENTS = (
    ROOT
    / "data"
    / "external"
    / "processed"
    / "eaglei_boulder"
    / "boulder_outage_events_threshold_sensitivity.csv"
)


G = {}

POLICIES = ["static", "greedy", "forecast_matched", "pc_rollout", "oracle"]
GROUPS = ["nominal", "single_domain", "cascade", "ood"]


def read_best_forecast(metrics_path: Path, mode: str = "plain") -> Path:
    rows = []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == mode:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no forecast metrics for mode={mode}")
    best = min(rows, key=lambda row: float(row["best_val_score"]))
    return metrics_path.parent / f"forecast_{mode}_seed{best['seed']}.npz"


def init_worker(payload):
    G.clear()
    G.update(payload)
    if G.get("smartds_online", False):
        smartds_dss.Text.Command("Clear")
        names, base_kw, base_kvar, phases = compile_feeder(Path(G["smartds_master"]))
        if names != G["smartds_load_names"]:
            raise RuntimeError("SMART-DS load order changed between payload construction and worker initialization")
        G["smartds_base_kw"] = base_kw
        G["smartds_base_kvar"] = base_kvar
        G["smartds_load_phases"] = phases


def prepare_crew_scenario(rng: np.random.Generator, initial_threat: np.ndarray) -> dict:
    """Create one policy-independent set of incident jobs and service times."""

    seed_risk = initial_threat.sum(axis=0).astype(float)
    seed_risk = seed_risk + 0.60 * (G["adj"] @ seed_risk) + 0.30 * (G["adj"] @ (G["adj"] @ seed_risk))
    active = np.flatnonzero(seed_risk > 1e-10)
    if len(active) > G["crew_jobs"]:
        order = np.argsort(-seed_risk[active], kind="stable")
        active = np.sort(active[order[: G["crew_jobs"]]])
    recovery = G["crew_recovery_h"]
    service_h = (
        rng.choice(recovery, size=len(active), replace=True) * G["crew_service_scale"]
        if len(active)
        else np.empty(0, dtype=float)
    )
    return {
        "jobs": active.astype(np.int64),
        "service_h": np.clip(service_h, 0.25, 16.0),
        "dispatch_uniform": rng.random(len(active)),
    }


def solve_priority_routes(
    travel_minutes: np.ndarray,
    service_minutes: np.ndarray,
    priority: np.ndarray,
    crews: int,
) -> dict:
    """Deterministic integer list routing with priority and completion time.

    Each assignment selects the crew--job pair with the largest priority per
    incremental completion minute.  The rule uses explicit integer vehicles,
    locations, travel, service, and completion times, while avoiding a
    wall-clock stopping criterion that would change paired results across runs.
    """

    n = len(service_minutes)
    remaining = set(range(1, n))
    crew_time = np.zeros(crews, dtype=np.int64)
    crew_location = np.zeros(crews, dtype=np.int64)
    arrival = np.full(n, np.nan)
    completion = np.full(n, np.nan)
    route_rows = [{"vehicle": vehicle, "sequence": []} for vehicle in range(crews)]
    total_travel = 0
    while remaining:
        best_key = None
        best = None
        for vehicle in range(crews):
            origin = int(crew_location[vehicle])
            for job in sorted(remaining):
                travel = int(travel_minutes[origin, job])
                start = int(crew_time[vehicle] + travel)
                finish = int(start + service_minutes[job])
                benefit_rate = float(priority[job]) / max(finish, 1)
                key = (-benefit_rate, finish, travel, job, vehicle)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (vehicle, job, travel, start, finish)
        vehicle, job, travel, start, finish = best
        remaining.remove(job)
        arrival[job] = start
        completion[job] = finish
        crew_time[vehicle] = finish
        crew_location[vehicle] = job
        total_travel += travel
        route_rows[vehicle]["sequence"].append(
            {"node": int(job), "arrival_min": int(start), "completion_min": int(finish)}
        )
    for vehicle in range(crews):
        total_travel += int(travel_minutes[int(crew_location[vehicle]), 0])
    objective = int(round(float(np.nansum(priority * completion)))) + total_travel
    return {
        "arrival_min": arrival,
        "completion_min": completion,
        "total_travel_min": float(total_travel),
        "routes": route_rows,
        "objective": objective,
    }


def build_crew_plan(
    restoration_priority: np.ndarray,
    packet_fraction: np.ndarray,
    scenario: dict,
) -> dict:
    """Translate first-step priorities into integer routes and completion events."""

    requested_jobs = scenario["jobs"]
    delivered = scenario["dispatch_uniform"] < packet_fraction[requested_jobs]
    jobs = requested_jobs[delivered]
    service_h = scenario["service_h"][delivered]
    completion_by_zone = np.full(restoration_priority.size, np.inf, dtype=float)
    if not len(jobs) or G["integer_crews"] <= 0:
        return {
            "requested_jobs": requested_jobs,
            "dispatched_jobs": jobs,
            "completion_by_zone": completion_by_zone,
            "total_travel_h": 0.0,
            "route_log": "",
            "event_log": "",
        }

    depot = int(G["crew_depot_index"])
    distance_min = G["crew_travel_minutes"]
    n_jobs = len(jobs)
    travel = np.zeros((n_jobs + 1, n_jobs + 1), dtype=np.int64)
    travel[0, 1:] = distance_min[depot, jobs]
    travel[1:, 0] = distance_min[jobs, depot]
    travel[1:, 1:] = distance_min[np.ix_(jobs, jobs)]
    service_minutes = np.concatenate([[0], np.rint(service_h * 60.0).astype(np.int64)])
    priority = np.concatenate([[0.0], np.maximum(restoration_priority[jobs], 1e-9)])
    solution = solve_priority_routes(
        travel,
        service_minutes,
        priority,
        min(G["integer_crews"], max(n_jobs, 1)),
    )
    completion_h = solution["completion_min"][1:] / 60.0
    completion_by_zone[jobs] = completion_h
    station_ids = G["station_ids"]
    event_log = ";".join(
        f"{station_ids[zone]}@{finish:.3f}" for zone, finish in sorted(zip(jobs, completion_h), key=lambda x: x[1])
    )
    route_parts = []
    for route in solution["routes"]:
        visits = []
        for visit in route["sequence"]:
            node = int(visit["node"])
            if node == 0:
                continue
            zone = int(jobs[node - 1])
            visits.append(f"{station_ids[zone]}@{visit['completion_min'] / 60.0:.3f}")
        route_parts.append(f"v{route['vehicle']}:" + "|".join(visits))
    return {
        "requested_jobs": requested_jobs,
        "dispatched_jobs": jobs,
        "completion_by_zone": completion_by_zone,
        "total_travel_h": float(solution["total_travel_min"] / 60.0),
        "route_log": ";".join(route_parts),
        "event_log": event_log,
    }


def projected_route_loss(
    current_threat: np.ndarray,
    forecast: np.ndarray,
    plan: dict,
    policy_coefficients: dict[str, float],
) -> float:
    """Score a routed completion schedule under one declared transition matrix."""

    z = current_threat.astype(np.float32).copy()
    repaired = np.zeros(z.shape[1], dtype=bool)
    domain_weight = np.asarray(
        [G["restore_road_weight"], G["restore_power_weight"], G["restore_comm_weight"]],
        dtype=float,
    )[:, None]
    mean_demand = np.maximum(G["mean_demand"], 1e-6)
    mean_energy = np.maximum(G["mean_energy"], 1e-6)
    total = 0.0
    zero = np.zeros(z.shape[1], dtype=np.float32)
    for step in range(G["horizon"]):
        row = forecast[min(step, len(forecast) - 1)]
        exposure = 1.0 + 0.25 * row[:, 0] / mean_demand + 0.25 * row[:, 1] / mean_energy
        exposure = np.clip(exposure, 1.0, 6.0)
        total += float(np.sum(domain_weight * z * exposure[None, :]))
        completed_now = (plan["completion_by_zone"] <= step + 1.0) & ~repaired
        repaired |= completed_now
        if step + 1 < G["horizon"]:
            z = threat_step(
                z,
                np.zeros_like(z),
                zero,
                policy_model=True,
                coefficient_override=policy_coefficients,
            )
            z[:, repaired] = 0.0
    return total


def route_candidate_plans(
    first_hour: int,
    threat: np.ndarray,
    backlog: np.ndarray,
    packet_fraction: np.ndarray,
    crew_scenario: dict,
) -> dict:
    """Construct the common candidate portfolio without selecting a route."""
    priorities = {}
    for candidate in ("static", "greedy", "forecast_matched", "pc_rollout", "robust_pc_rollout"):
        _, _, priority = policy_action(candidate, first_hour, G["horizon"], threat, backlog, None)
        priorities[candidate] = priority
    priorities["uniform_route"] = np.ones(threat.shape[1], dtype=np.float32)
    return {
        candidate: build_crew_plan(priority, packet_fraction, crew_scenario)
        for candidate, priority in priorities.items()
    }


def route_portfolio_plan(
    selector_policy: str,
    first_hour: int,
    threat: np.ndarray,
    backlog: np.ndarray,
    packet_fraction: np.ndarray,
    crew_scenario: dict,
    selector_priority: np.ndarray,
) -> dict:
    """Select from the same executable route candidates under each policy model.

    The robust, central-PC, and forecast-matched policies receive an identical
    candidate set.  Forecast-matched uses current exposure only; central PC
    rolls the calibrated central transition forward; robust PC uses the
    worst score over the declared transition-matrix set.
    """

    plans = route_candidate_plans(first_hour, threat, backlog, packet_fraction, crew_scenario)
    forecast = forecast_at(first_hour, G["horizon"])
    scores = {}
    if selector_policy == "robust_pc_rollout":
        for candidate, plan in plans.items():
            scores[candidate] = max(
                projected_route_loss(threat, forecast, plan, coefficients)
                for coefficients in G["robust_policy_variants"]
            )
    elif selector_policy == "pc_rollout":
        central = G["central_policy_coefficients"]
        for candidate, plan in plans.items():
            scores[candidate] = projected_route_loss(threat, forecast, plan, central)
    elif selector_policy == "forecast_matched":
        jobs = crew_scenario["jobs"]
        for candidate, plan in plans.items():
            completion = np.minimum(plan["completion_by_zone"][jobs], G["horizon"])
            scores[candidate] = float(np.sum(selector_priority[jobs] * completion))
    else:
        raise ValueError(f"route portfolio is not defined for {selector_policy}")
    source = min(scores, key=lambda candidate: (scores[candidate], candidate))
    selected = plans[source]
    selected["source_policy"] = source
    selected["route_score"] = float(scores[source])
    return selected


def project_smartds_action(requested_ev_kw: np.ndarray, mapping_index: int) -> tuple[float, dict, dict]:
    """Project one station-level hourly charging action with the full OpenDSS feeder."""

    if not G.get("smartds_online", False):
        neutral = {
            "converged": True,
            "v_min_pu": 1.0,
            "v_max_pu": 1.0,
            "nodes_below_0_95": 0,
            "nodes_above_1_05": 0,
            "deenergized_nodes": 0,
            "max_line_loading_pu": 0.0,
            "limiting_line": "",
            "losses_kw": 0.0,
            "source_kw": 0.0,
        }
        return 1.0, neutral, neutral
    mapping = G["smartds_mappings"][mapping_index]
    ev_by_load = np.zeros(len(G["smartds_load_names"]), dtype=float)
    np.add.at(ev_by_load, mapping, np.asarray(requested_ev_kw, dtype=float))
    return solve_alpha(
        G["smartds_load_names"],
        G["smartds_base_kw"],
        G["smartds_base_kvar"],
        ev_by_load,
        G["smartds_power_factor"],
        G["smartds_voltage_low"],
        G["smartds_voltage_high"],
        G["smartds_line_limit"],
        iterations=G["smartds_bisection_iterations"],
    )


def allocate(total: float, score: np.ndarray, reserve_fraction: float = 0.04) -> np.ndarray:
    score = np.maximum(np.asarray(score, dtype=np.float64), 0.0)
    n = score.size
    reserve = float(total) * reserve_fraction / n
    distributable = max(float(total) - reserve * n, 0.0)
    weights = np.sqrt(score + 1e-9)
    if weights.sum() <= 1e-12:
        weights = np.full(n, 1.0 / n)
    else:
        weights /= weights.sum()
    return (reserve + distributable * weights).astype(np.float32)


def largest_remainder_crews(score: np.ndarray, crews: int) -> np.ndarray:
    """Convert continuous zone priorities to auditable integer crew counts."""

    score = np.maximum(np.asarray(score, dtype=np.float64), 0.0)
    if crews <= 0:
        return np.zeros(score.size, dtype=np.int64)
    if score.sum() <= 1e-12:
        score = np.ones_like(score)
    quota = crews * score / score.sum()
    count = np.floor(quota).astype(np.int64)
    remainder = crews - int(count.sum())
    if remainder:
        order = np.argsort(-(quota - count), kind="stable")
        count[order[:remainder]] += 1
    return count


def packet_action_fraction(threat: np.ndarray) -> np.ndarray:
    """Map current power/communication threat to measured packet enactment.

    The response surface is the nearest observed finite-buffer packet
    simulation at the same station, traffic multiplier, and backup duration.
    No latency model is fitted inside the rollout and the lookup is therefore
    auditable row-by-row against ``packet_network_scenarios.csv``.
    """

    surface = G.get("packet_response_surface")
    if surface is None:
        return np.ones(threat.shape[1], dtype=np.float32)
    power, comm = threat[1], threat[2]
    points = surface["points"]
    actions = surface["actions"]
    valid = surface["valid"]
    result = np.ones(threat.shape[1], dtype=np.float32)
    for zone in range(threat.shape[1]):
        mask = valid[zone]
        if not np.any(mask):
            continue
        distance = (
            ((points[zone, mask, 0] - power[zone]) / 0.75) ** 2
            + ((points[zone, mask, 1] - comm[zone]) / 0.375) ** 2
        )
        nearest = int(np.argmin(distance))
        result[zone] = float(actions[zone, mask][nearest])
    return np.clip(result, 0.0, 1.0)


def scenario_inputs(rng: np.random.Generator, group: str):
    """Return current threat and policy-independent future innovations."""

    n = G["adj"].shape[0]
    horizon = G["horizon"]
    mean_demand = G["mean_demand"]
    probs = mean_demand + 0.05 * mean_demand.mean()
    probs /= probs.sum()
    if group == "nominal":
        k, severity, domains, innovation_scale = 2, 0.08, [0], 0.003
    elif group == "single_domain":
        k, severity, domains, innovation_scale = 6, 0.35, [int(rng.integers(0, 3))], 0.008
    elif group == "cascade":
        k, severity, domains, innovation_scale = 8, 0.45, [0, 1, 2], 0.012
    else:
        k, severity, domains, innovation_scale = 12, 0.62, [0, 1, 2], 0.022
    zones = rng.choice(n, size=k, replace=False, p=probs)
    initial = np.zeros((3, n), dtype=np.float32)
    for domain in domains:
        initial[domain, zones] = severity * rng.uniform(0.75, 1.25, size=k)
    innovations = np.zeros((horizon, 3, n), dtype=np.float32)
    for step in range(horizon):
        active = zones if group != "nominal" else zones[:1]
        for domain in domains:
            innovations[step, domain, active] = innovation_scale * rng.uniform(0.6, 1.4, size=len(active))
    return np.clip(initial, 0.0, 0.95), innovations


def threat_step(
    threat: np.ndarray,
    innovation: np.ndarray,
    restoration: np.ndarray,
    *,
    policy_model: bool = False,
    coefficient_override: dict[str, float] | None = None,
) -> np.ndarray:
    """Action-dependent road/power/communication threat transition."""

    road, power, comm = threat
    adj = G["adj"]
    prefix = "policy_" if policy_model else ""
    def coefficient(name: str) -> float:
        if coefficient_override is not None and name in coefficient_override:
            return float(coefficient_override[name])
        return float(G[f"{prefix}{name}"])

    spread = coefficient("spatial_spread")
    road_next = (
        coefficient("road_persistence") * road
        + spread * (adj @ road)
        + coefficient("pr_coupling") * power
        + coefficient("cr_coupling") * comm
    )
    power_next = coefficient("power_persistence") * power + spread * (adj @ power) + coefficient("cp_coupling") * (adj @ comm)
    comm_next = (
        coefficient("comm_persistence") * comm
        + spread * (adj @ comm)
        + coefficient("pc_coupling") * power
        + coefficient("rc_coupling") * (adj @ road)
    )
    propagated = np.stack([road_next, power_next, comm_next], axis=0)
    if not policy_model and G.get("realized_transition_mode", "linear") == "nonlinear_saturation":
        spatial_state = np.stack([adj @ row for row in threat], axis=0)
        propagated = 0.98 * (-np.expm1(-(np.maximum(propagated, 0.0) + 0.15 * spatial_state**2) / 0.98))
    next_threat = propagated + innovation
    per_zone_reference = G["base_total_restore"] / threat.shape[1]
    repair = np.clip(restoration / max(per_zone_reference, 1e-8), 0.0, 3.0)
    effects = np.asarray(G["restoration_effect"], dtype=np.float64)[:, None] * repair[None, :]
    return np.clip(next_threat - effects, 0.0, 0.98).astype(np.float32)


def projected_threat_loss(
    current_threat: np.ndarray,
    forecast: np.ndarray,
    horizon: int,
    *,
    first_restoration: np.ndarray | None = None,
    known_innovations: np.ndarray | None = None,
    policy_coefficients: dict[str, float] | None = None,
) -> float:
    """Evaluate a candidate first action with the information available now.

    Non-oracle policies set ``known_innovations`` to ``None``.  Forecast demand
    is used only as an exposure weight, while the cyber--physical state follows
    the explicit transition in :func:`threat_step`.
    """

    z = current_threat.astype(np.float32).copy()
    domain_weight = np.asarray(
        [G["restore_road_weight"], G["restore_power_weight"], G["restore_comm_weight"]],
        dtype=np.float64,
    )[:, None]
    mean_demand = np.maximum(G["mean_demand"], 1e-6)
    mean_energy = np.maximum(G["mean_energy"], 1e-6)
    total = 0.0
    zero_restore = np.zeros(z.shape[1], dtype=np.float32)
    for step in range(max(horizon, 1)):
        row = forecast[min(step, len(forecast) - 1)]
        exposure = 1.0 + 0.25 * row[:, 0] / mean_demand + 0.25 * row[:, 1] / mean_energy
        exposure = np.clip(exposure, 1.0, 6.0)
        total += float(np.sum(domain_weight * z * exposure[None, :]))
        if step + 1 < horizon:
            innovation = (
                known_innovations[step]
                if known_innovations is not None and step < len(known_innovations)
                else np.zeros_like(z)
            )
            action = first_restoration if step == 0 and first_restoration is not None else zero_restore
            z = threat_step(
                z,
                innovation,
                action,
                policy_model=True,
                coefficient_override=policy_coefficients,
            )
    return total


def rollout_restoration_score(
    current_threat: np.ndarray,
    forecast: np.ndarray,
    horizon: int,
    *,
    known_innovations: np.ndarray | None = None,
    policy_coefficients: dict[str, float] | None = None,
) -> np.ndarray:
    """Return the marginal multi-step benefit of repairing each zone now."""

    base = projected_threat_loss(
        current_threat,
        forecast,
        horizon,
        known_innovations=known_innovations,
        policy_coefficients=policy_coefficients,
    )
    n = current_threat.shape[1]
    quantum = G["base_total_restore"] / n
    benefit = np.zeros(n, dtype=np.float64)
    for zone in range(n):
        candidate = np.zeros(n, dtype=np.float32)
        candidate[zone] = quantum
        with_action = projected_threat_loss(
            current_threat,
            forecast,
            horizon,
            first_restoration=candidate,
            known_innovations=known_innovations,
            policy_coefficients=policy_coefficients,
        )
        benefit[zone] = max(base - with_action, 0.0)
    return benefit


def forecast_at(first_hour: int, remaining: int) -> np.ndarray:
    """Use the forecast issued from the current observed hour, with padding."""

    row = G["forecast_index"].get(first_hour)
    if row is None:
        raw = G["raw"]
        end = min(first_hour + remaining, raw.shape[0])
        values = raw[first_hour:end]
    else:
        values = G["forecast_pred"][row, :remaining]
    if len(values) == 0:
        values = G["raw"][first_hour - 1 : first_hour]
    if len(values) < remaining:
        values = np.concatenate([values, np.repeat(values[-1:], remaining - len(values), axis=0)], axis=0)
    return values.astype(np.float32)


def truth_at(first_hour: int, remaining: int) -> np.ndarray:
    raw = G["raw"]
    values = raw[first_hour : min(first_hour + remaining, raw.shape[0])]
    if len(values) < remaining:
        values = np.concatenate([values, np.repeat(values[-1:], remaining - len(values), axis=0)], axis=0)
    return values.astype(np.float32)


def policy_action(
    policy: str,
    first_hour: int,
    remaining: int,
    threat: np.ndarray,
    backlog: np.ndarray,
    oracle_innovations: np.ndarray | None,
    *,
    discretize_restoration: bool = True,
):
    mean_demand = G["mean_demand"]
    mean_energy = G["mean_energy"]
    threat_now = threat.sum(axis=0)
    if policy == "static":
        forecast = np.stack([mean_demand, mean_energy], axis=-1)[None, ...]
        demand_score = mean_demand
        energy_score = mean_energy + G["backlog_score"] * backlog
        comm_score = mean_demand + G["comm_energy_weight"] * mean_energy
        restore_score = mean_demand
    else:
        forecast = truth_at(first_hour, remaining) if policy == "oracle" else forecast_at(first_hour, remaining)
        demand_score = forecast[..., 0].sum(axis=0)
        energy_score = forecast[..., 1].sum(axis=0) + G["backlog_score"] * backlog
        comm_score = demand_score + G["comm_energy_weight"] * energy_score
        demand_score += G["service_threat_weight"] * threat_now
        energy_score += G["service_threat_weight"] * threat_now
        comm_score += G["comm_threat_weight"] * threat_now
        if policy == "greedy":
            restore_score = threat_now
        elif policy == "forecast_matched":
            # Same demand forecast and current threat observation as PC rollout;
            # only the cross-domain transition rollout is ablated.
            exposure = 1.0 + 0.25 * forecast[0, :, 0] / np.maximum(mean_demand, 1e-6)
            exposure += 0.25 * forecast[0, :, 1] / np.maximum(mean_energy, 1e-6)
            exposure = np.clip(exposure, 1.0, 6.0)
            restore_score = (
                G["restore_power_weight"] * threat[1]
                + G["restore_comm_weight"] * threat[2]
                + G["restore_road_weight"] * threat[0]
            ) * exposure
        elif policy == "pc_rollout":
            direct = (
                G["restore_power_weight"] * threat[1]
                + G["restore_comm_weight"] * threat[2]
                + G["restore_road_weight"] * threat[0]
            )
            marginal_benefit = rollout_restoration_score(threat, forecast, remaining)
            restore_score = direct + G["restore_envelope_weight"] * marginal_benefit
        elif policy == "robust_pc_rollout":
            direct = (
                G["restore_power_weight"] * threat[1]
                + G["restore_comm_weight"] * threat[2]
                + G["restore_road_weight"] * threat[0]
            )
            variant_benefits = [
                rollout_restoration_score(
                    threat,
                    forecast,
                    remaining,
                    policy_coefficients=coefficients,
                )
                for coefficients in G["robust_policy_variants"]
            ]
            # Maximin score: allocate according to the smallest action benefit
            # attained over the prespecified transition uncertainty set.
            marginal_benefit = np.min(np.stack(variant_benefits, axis=0), axis=0)
            restore_score = direct + G["restore_envelope_weight"] * marginal_benefit
        elif policy == "oracle":
            restore_score = rollout_restoration_score(
                threat,
                forecast,
                remaining,
                known_innovations=oracle_innovations,
            )
        else:
            raise ValueError(policy)
    charge = allocate(G["total_charge"], energy_score, G["service_reserve_fraction"])
    communication = allocate(G["total_comm"], comm_score, G["service_reserve_fraction"])
    restoration = allocate(G["total_restore"], restore_score, G["restore_reserve_fraction"])
    if discretize_restoration and G["integer_crews"] > 0:
        crews = largest_remainder_crews(restoration, G["integer_crews"])
        restoration = crews.astype(np.float32) * (G["total_restore"] / G["integer_crews"])
    return charge, communication, restoration


def evaluate_policy(
    policy: str,
    first_hour: int,
    initial_threat: np.ndarray,
    innovations: np.ndarray,
    crew_scenario: dict,
    smartds_mapping_index: int,
    true_coefficients: dict[str, float] | None = None,
    crew_plan_override: dict | None = None,
):
    raw = G["raw"]
    threat = initial_threat.copy()
    backlog = np.zeros_like(G["mean_energy"])
    repaired = np.zeros(threat.shape[1], dtype=bool)
    crew_plan = None
    decision_seconds = 0.0
    metrics = {
        "cost": 0.0,
        "mobility_delay": 0.0,
        "unserved_energy": 0.0,
        "comm_loss": 0.0,
        "feeder_margin_exceedance": 0.0,
        "power_service_loss": 0.0,
        "peak_risk": 0.0,
        "served_mobility": 0.0,
        "total_mobility": 0.0,
        "control_action_fraction_sum": 0.0,
        "requested_restoration": 0.0,
        "executed_restoration": 0.0,
        "requested_charge_capacity": 0.0,
        "executed_charge_capacity": 0.0,
        "smartds_projection_fraction_sum": 0.0,
        "smartds_curtailed_energy_kwh": 0.0,
        "smartds_raw_infeasible_hours": 0,
        "smartds_projected_infeasible_hours": 0,
        "min_voltage_pu": float("inf"),
        "voltage_violation_pu_hours": 0.0,
        "voltage_violating_bus_hours": 0,
        "thermal_overload_pu_hours": 0.0,
        "thermal_overloaded_branch_hours": 0,
        "losses_mwh": 0.0,
    }
    for step in range(G["horizon"]):
        hour = first_hour + step
        remaining = G["horizon"] - step
        oracle_future = innovations[step : step + remaining]
        started = time.perf_counter()
        charge_cap, comm_cap, restoration_priority = policy_action(
            policy,
            hour,
            remaining,
            threat,
            backlog,
            oracle_future,
        )
        action_fraction = packet_action_fraction(threat)
        if crew_plan is None:
            if crew_plan_override is not None:
                crew_plan = dict(crew_plan_override)
            elif policy in {"forecast_matched", "pc_rollout", "robust_pc_rollout"}:
                crew_plan = route_portfolio_plan(
                    policy,
                    first_hour,
                    threat,
                    backlog,
                    action_fraction,
                    crew_scenario,
                    restoration_priority,
                )
            else:
                crew_plan = build_crew_plan(restoration_priority, action_fraction, crew_scenario)
                crew_plan["source_policy"] = policy
                crew_plan["route_score"] = float("nan")
            metrics["requested_restoration"] = float(len(crew_plan["requested_jobs"]))
        decision_seconds += time.perf_counter() - started

        # Packet deadlines gate the remote charging and communication actions.
        # The first dispatch command also determines which crew jobs enter the
        # committed integer route.  Repair benefit is credited only when the
        # corresponding routed completion event occurs below.
        metrics["control_action_fraction_sum"] += float(action_fraction.mean())
        metrics["requested_charge_capacity"] += float(charge_cap.sum())
        charge_cap = charge_cap * action_fraction
        comm_cap = comm_cap * action_fraction

        demand = raw[hour, :, 0]
        new_energy = raw[hour, :, 1] * G["energy_scale"]
        energy_required = new_energy + G["backlog_carryover"] * backlog
        road, power, comm = threat
        power_available = 1.0 - power
        comm_load = (
            G["comm_mobility_load"] * demand
            + G["comm_energy_load"] * energy_required
            + G["comm_road_load"] * road
            + G["comm_power_load"] * power
        )
        served_comm = np.minimum(
            comm_load,
            comm_cap * power_available * (1.0 - G["comm_direct_derate"] * comm),
        )
        comm_loss = np.maximum(comm_load - served_comm, 0.0)
        comm_support = 1.0 - np.clip(comm_loss / (comm_load + 1.0), 0.0, 0.9)
        requested_served_energy = np.minimum(
            energy_required,
            charge_cap * power_available * comm_support,
        )
        alpha, raw_grid, projected_grid = project_smartds_action(
            requested_served_energy,
            smartds_mapping_index,
        )
        projected_ok = feasible(
            projected_grid,
            G["smartds_voltage_low"],
            G["smartds_voltage_high"],
            G["smartds_line_limit"],
        )
        if not projected_ok:
            raise RuntimeError(f"SMART-DS projection failed its execution contract: {projected_grid}")
        raw_ok = feasible(
            raw_grid,
            G["smartds_voltage_low"],
            G["smartds_voltage_high"],
            G["smartds_line_limit"],
        )
        served_energy = requested_served_energy * alpha
        unserved_energy = np.maximum(energy_required - served_energy, 0.0)
        backlog = unserved_energy
        metrics["smartds_projection_fraction_sum"] += float(alpha)
        metrics["smartds_curtailed_energy_kwh"] += float((requested_served_energy - served_energy).sum())
        metrics["smartds_raw_infeasible_hours"] += int(not raw_ok)
        metrics["smartds_projected_infeasible_hours"] += int(not projected_ok)
        metrics["min_voltage_pu"] = min(metrics["min_voltage_pu"], float(projected_grid["v_min_pu"]))
        metrics["voltage_violation_pu_hours"] += max(
            G["smartds_voltage_low"] - float(projected_grid["v_min_pu"]), 0.0
        ) + max(float(projected_grid["v_max_pu"]) - G["smartds_voltage_high"], 0.0)
        metrics["voltage_violating_bus_hours"] += int(projected_grid["nodes_below_0_95"]) + int(
            projected_grid["nodes_above_1_05"]
        )
        metrics["thermal_overload_pu_hours"] += max(
            float(projected_grid["max_line_loading_pu"]) - G["smartds_line_limit"], 0.0
        )
        metrics["thermal_overloaded_branch_hours"] += int(
            float(projected_grid["max_line_loading_pu"]) > G["smartds_line_limit"]
        )
        metrics["losses_mwh"] += float(projected_grid["losses_kw"]) / 1000.0
        metrics["executed_charge_capacity"] += float(charge_cap.sum()) * float(alpha)

        mobility_delay = demand * (road + G["mobility_comm_weight"] * (1.0 - comm_support))
        served_mobility = np.maximum(demand - mobility_delay, 0.0)
        power_service_loss = power * energy_required
        cascade = power * comm + road * (1.0 - comm_support)
        risk = (
            G["cost_mobility"] * mobility_delay
            + G["cost_unserved"] * unserved_energy
            + G["cost_comm"] * comm_loss
            + G["cost_power_service"] * power_service_loss
            + G["cost_cascade"] * cascade
        )
        metrics["cost"] += float(risk.sum())
        metrics["mobility_delay"] += float(mobility_delay.sum())
        metrics["unserved_energy"] += float(unserved_energy.sum())
        metrics["comm_loss"] += float(comm_loss.sum())
        metrics["power_service_loss"] += float(power_service_loss.sum())
        metrics["served_mobility"] += float(served_mobility.sum())
        metrics["total_mobility"] += float(demand.sum())
        metrics["peak_risk"] = max(metrics["peak_risk"], float(risk.max()))
        completed_now = (crew_plan["completion_by_zone"] <= step + 1.0) & ~repaired
        repaired |= completed_now
        metrics["executed_restoration"] += float(np.count_nonzero(completed_now))
        threat = threat_step(
            threat,
            innovations[step],
            np.zeros(threat.shape[1], dtype=np.float32),
            coefficient_override=true_coefficients,
        )
        # Completed repairs remain restored over the remainder of the finite
        # horizon; later propagation or innovations cannot silently reactivate
        # a repaired job without a newly declared incident.
        threat[:, repaired] = 0.0

    metrics["service_continuity"] = metrics["served_mobility"] / max(metrics["total_mobility"], 1e-8)
    metrics["mean_control_action_fraction"] = metrics.pop("control_action_fraction_sum") / G["horizon"]
    metrics["mean_smartds_projection_fraction"] = metrics.pop("smartds_projection_fraction_sum") / G["horizon"]
    metrics["crew_job_count"] = int(len(crew_plan["requested_jobs"]))
    metrics["crew_jobs_dispatched"] = int(len(crew_plan["dispatched_jobs"]))
    metrics["crew_jobs_completed"] = int(np.count_nonzero(repaired))
    metrics["crew_completion_fraction"] = float(
        np.count_nonzero(repaired) / max(len(crew_plan["requested_jobs"]), 1)
    )
    finite_completion = crew_plan["completion_by_zone"][np.isfinite(crew_plan["completion_by_zone"])]
    metrics["crew_mean_completion_h"] = float(finite_completion.mean()) if len(finite_completion) else 0.0
    metrics["crew_total_travel_h"] = float(crew_plan["total_travel_h"])
    metrics["crew_completion_events"] = crew_plan["event_log"]
    metrics["crew_routes"] = crew_plan["route_log"]
    metrics["crew_route_source_policy"] = crew_plan["source_policy"]
    metrics["route_score"] = crew_plan["route_score"]
    metrics["latency_ms"] = 1000.0 * decision_seconds / G["horizon"]
    return metrics


def simulate_one(task):
    scenario_id, group = task
    rng = np.random.default_rng(G["seed"] + scenario_id * 7919)
    candidates = G["valid_first_hours"]
    first_hour = int(candidates[int(rng.integers(0, len(candidates)))])
    initial, innovations = scenario_inputs(rng, group)
    true_coefficients = None
    if G.get("sample_transition_uncertainty", False):
        true_coefficients = {
            name: float(rng.uniform(bounds[0], bounds[1]))
            for name, bounds in G["transition_uncertainty_bounds"].items()
        }
    crew_scenario = prepare_crew_scenario(rng, initial)
    smartds_mapping_index = scenario_id % len(G["smartds_mappings"])
    rows = []
    for policy in G.get("policies", POLICIES):
        metrics = evaluate_policy(
            policy,
            first_hour,
            initial,
            innovations,
            crew_scenario,
            smartds_mapping_index,
            true_coefficients=true_coefficients,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "group": group,
                "first_hour": first_hour,
                "policy": policy,
                "smartds_mapping_index": smartds_mapping_index,
                **(
                    {f"true_{name}": value for name, value in true_coefficients.items()}
                    if true_coefficients is not None
                    else {}
                ),
                **metrics,
            }
        )
    return rows


def mean_ci(values):
    values = list(map(float, values))
    mean = statistics.fmean(values) if values else float("nan")
    if len(values) <= 1:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def aggregate(rows):
    metrics = [
        "cost",
        "mobility_delay",
        "unserved_energy",
        "comm_loss",
        "feeder_margin_exceedance",
        "power_service_loss",
        "peak_risk",
        "service_continuity",
        "min_voltage_pu",
        "voltage_violation_pu_hours",
        "voltage_violating_bus_hours",
        "thermal_overload_pu_hours",
        "thermal_overloaded_branch_hours",
        "losses_mwh",
        "latency_ms",
        "mean_control_action_fraction",
        "requested_restoration",
        "executed_restoration",
        "requested_charge_capacity",
        "executed_charge_capacity",
        "mean_smartds_projection_fraction",
        "smartds_curtailed_energy_kwh",
        "smartds_raw_infeasible_hours",
        "smartds_projected_infeasible_hours",
        "crew_job_count",
        "crew_jobs_dispatched",
        "crew_jobs_completed",
        "crew_completion_fraction",
        "crew_mean_completion_h",
        "crew_total_travel_h",
    ]
    out = []
    for policy, group in sorted({(row["policy"], row["group"]) for row in rows}):
        subset = [row for row in rows if row["policy"] == policy and row["group"] == group]
        record = {"policy": policy, "group": group, "n": len(subset)}
        for metric in metrics:
            mean, ci = mean_ci(row[metric] for row in subset)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95"] = ci
        out.append(record)
    return out


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def estimate_transition_spectral_radius(adj: np.ndarray, args, iterations: int = 100) -> float:
    default_persistence = float(getattr(args, "threat_persistence", 0.48))
    road_persistence = getattr(args, "road_persistence", None)
    power_persistence = getattr(args, "power_persistence", None)
    comm_persistence = getattr(args, "comm_persistence", None)
    road_persistence = default_persistence if road_persistence is None else float(road_persistence)
    power_persistence = default_persistence if power_persistence is None else float(power_persistence)
    comm_persistence = default_persistence if comm_persistence is None else float(comm_persistence)
    pr_coupling = float(getattr(args, "pr_coupling", 0.0))
    rng = np.random.default_rng(11)
    value = rng.normal(size=(3, adj.shape[0]))
    value /= np.linalg.norm(value)
    estimate = 0.0
    for _ in range(iterations):
        road, power, comm = value
        nxt = np.stack(
            [
                road_persistence * road + args.spatial_spread * (adj @ road) + pr_coupling * power + args.cr_coupling * comm,
                power_persistence * power + args.spatial_spread * (adj @ power) + args.cp_coupling * (adj @ comm),
                comm_persistence * comm + args.spatial_spread * (adj @ comm) + args.pc_coupling * power + args.rc_coupling * (adj @ road),
            ],
            axis=0,
        )
        estimate = float(np.linalg.norm(nxt))
        value = nxt / max(estimate, 1e-12)
    return estimate


def load_packet_response_surface(
    path: Path | None,
    zones: int,
    traffic_multiplier: float,
    backup_duration_s: float,
) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    selected: dict[int, list[tuple[float, float, float]]] = {zone: [] for zone in range(zones)}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not math.isclose(float(row["traffic_multiplier"]), traffic_multiplier, abs_tol=1e-9):
                continue
            if not math.isclose(float(row["backup_duration_s"]), backup_duration_s, abs_tol=1e-9):
                continue
            zone = int(row["station_index"])
            if 0 <= zone < zones:
                selected[zone].append(
                    (
                        float(row["power_threat"]),
                        float(row["communication_threat"]),
                        float(row["effective_action_fraction"]),
                    )
                )
    counts = [len(selected[zone]) for zone in range(zones)]
    if not counts or min(counts) == 0:
        raise ValueError(
            f"Packet response surface is incomplete for traffic={traffic_multiplier}, "
            f"backup={backup_duration_s}: min station rows={min(counts, default=0)}"
        )
    width = max(counts)
    points = np.zeros((zones, width, 2), dtype=np.float32)
    actions = np.ones((zones, width), dtype=np.float32)
    valid = np.zeros((zones, width), dtype=bool)
    for zone, records in selected.items():
        for index, (power, comm, action) in enumerate(records):
            points[zone, index] = (power, comm)
            actions[zone, index] = action
            valid[zone, index] = True
    return {"points": points, "actions": actions, "valid": valid}


def payload_from_args(args):
    data = np.load(args.data, allow_pickle=True)
    metrics_path = args.forecast_results / "prediction_metrics.csv"
    forecast_path = read_best_forecast(metrics_path, "plain")
    forecast = np.load(forecast_path)
    raw = np.stack([data["pickup"], data["energy"]], axis=-1).astype(np.float32)
    indices = forecast["indices"].astype(np.int64)
    training_end = int(data["split_train_end_index"])
    if not 0 < training_end < int(data["split_val_end_index"]) <= int(indices.min()):
        raise ValueError("Training statistics and test forecast periods must be chronologically disjoint")
    mean_demand = data["pickup"][:training_end].mean(axis=0).astype(np.float32)
    mean_energy = data["energy"][:training_end].mean(axis=0).astype(np.float32)
    station_ids = data["station_id"].astype(str)

    coordinates = pd.read_csv(args.smartds_coordinates).set_index("station_id").loc[station_ids]
    lat = coordinates["latitude"].to_numpy(dtype=float)
    lon = coordinates["longitude"].to_numpy(dtype=float)
    distance_km = haversine_km(lat, lon)
    crew_travel_minutes = np.rint(
        distance_km * args.crew_road_circuity / args.crew_speed_kmh * 60.0
    ).astype(np.int64)
    crew_depot_index = int(np.argmin(distance_km.sum(axis=1)))
    event_frame = pd.read_csv(args.crew_events)
    event_frame = event_frame.loc[event_frame["threshold_customers"].eq(50)].copy()
    training_cutoff = pd.Timestamp(str(data["timestamp_local"][training_end]))
    event_end = pd.to_datetime(event_frame["end_exclusive"])
    event_frame = event_frame.loc[event_end < training_cutoff].copy()
    recovery_h = event_frame["post_peak_half_recovery_h"].dropna().to_numpy(dtype=float)
    if not len(recovery_h):
        recovery_h = (event_frame["duration_h"] / 2.0).dropna().to_numpy(dtype=float)
    recovery_h = np.clip(recovery_h, 0.25, 8.0)
    if not len(recovery_h):
        raise RuntimeError("No EAGLE-I recovery durations are available for integrated crew execution")

    smartds_online = not args.disable_smartds_online
    if smartds_online:
        smartds_dss.Text.Command("Clear")
        load_names, base_kw, base_kvar, load_phases = compile_feeder(args.smartds_master)
        eligible_loads = np.flatnonzero(load_phases == 3)
        if not len(eligible_loads):
            raise RuntimeError("No three-phase SMART-DS loads are available for station embedding")
        coordinate_rows = coordinates.reset_index()
        smartds_mappings = np.stack(
            [
                mapping_for_seed(coordinate_rows, eligible_loads, args.smartds_mapping_seed + index)
                for index in range(args.smartds_mappings)
            ],
            axis=0,
        ).astype(np.int64)
        _, baseline_grid, _ = solve_alpha(
            load_names,
            base_kw,
            base_kvar,
            np.zeros(len(load_names), dtype=float),
            args.smartds_power_factor,
            args.smartds_voltage_low,
            args.smartds_voltage_high,
            args.smartds_line_limit,
            iterations=args.smartds_bisection_iterations,
        )
        if not feasible(
            baseline_grid,
            args.smartds_voltage_low,
            args.smartds_voltage_high,
            args.smartds_line_limit,
        ):
            raise RuntimeError(f"SMART-DS base case violates the execution contract: {baseline_grid}")
    else:
        load_names = ["disabled"]
        smartds_mappings = np.zeros((1, len(station_ids)), dtype=np.int64)
        baseline_grid = {
            "v_min_pu": 1.0,
            "v_max_pu": 1.0,
            "max_line_loading_pu": 0.0,
            "losses_kw": 0.0,
        }
    packet_response_surface = load_packet_response_surface(
        args.packet_results,
        raw.shape[1],
        args.packet_traffic_multiplier,
        args.packet_backup_duration_s,
    )
    base_total_restore = float(max(10.0, mean_demand.sum() * 0.018))
    resolved_persistence = {
        "road_persistence": args.threat_persistence if args.road_persistence is None else args.road_persistence,
        "power_persistence": args.threat_persistence if args.power_persistence is None else args.power_persistence,
        "comm_persistence": args.threat_persistence if args.comm_persistence is None else args.comm_persistence,
    }
    coefficient_values = {
        **resolved_persistence,
        "spatial_spread": args.spatial_spread,
        "pr_coupling": args.pr_coupling,
        "pc_coupling": args.pc_coupling,
        "rc_coupling": args.rc_coupling,
        "cr_coupling": args.cr_coupling,
        "cp_coupling": args.cp_coupling,
    }
    coefficient_names = [
        "road_persistence",
        "power_persistence",
        "comm_persistence",
        "spatial_spread",
        "pr_coupling",
        "pc_coupling",
        "rc_coupling",
        "cr_coupling",
        "cp_coupling",
    ]
    uncertainty_bounds = {}
    robust_variants = []
    if args.robust_uncertainty_file is not None:
        alias = {"threat_persistence": "road_persistence"}
        with args.robust_uncertainty_file.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = alias.get(row["parameter"], row["parameter"])
                if name in coefficient_values:
                    coefficient_values[name] = float(row["central"])
                    uncertainty_bounds[name] = (float(row["lower"]), float(row["upper"]))
        if not uncertainty_bounds:
            raise ValueError(f"No usable transition bounds in {args.robust_uncertainty_file}")
    elif args.sample_transition_uncertainty:
        raise ValueError("--sample-transition-uncertainty requires --robust-uncertainty-file")

    policy_rng = np.random.default_rng(args.policy_matrix_seed)
    central_policy_coefficients = {}
    policy_coefficients = {}
    policy_multipliers = {}
    for name in coefficient_names:
        multiplier = args.policy_matrix_scale
        if args.policy_matrix_noise > 0:
            multiplier *= max(0.0, 1.0 + float(policy_rng.normal(0.0, args.policy_matrix_noise)))
        policy_multipliers[name] = float(multiplier)
        value = float(coefficient_values[name] * multiplier)
        central_policy_coefficients[name] = value
        policy_coefficients[f"policy_{name}"] = value

    if uncertainty_bounds:
        # Apply the same policy-side perturbation to the central matrix and to
        # every member of its uncertainty portfolio.  Realized dynamics remain
        # calibrated and are not modified by this sensitivity parameter.
        central = {
            name: float(value * policy_multipliers[name])
            for name, value in coefficient_values.items()
        }
        scaled_bounds = {
            name: (
                float(bounds[0] * policy_multipliers[name]),
                float(bounds[1] * policy_multipliers[name]),
            )
            for name, bounds in uncertainty_bounds.items()
        }
        robust_variants.append(central)
        robust_variants.append(
            {**central, **{name: bounds[0] for name, bounds in scaled_bounds.items()}}
        )
        robust_variants.append(
            {**central, **{name: bounds[1] for name, bounds in scaled_bounds.items()}}
        )
        for name, bounds in sorted(scaled_bounds.items()):
            robust_variants.append({**central, name: bounds[0]})
            robust_variants.append({**central, name: bounds[1]})
    policies = list(POLICIES)
    if robust_variants:
        policies.insert(policies.index("oracle"), "robust_pc_rollout")
    payload = {
        "raw": raw,
        "adj": data["adj"].astype(np.float32),
        "mean_demand": mean_demand,
        "mean_energy": mean_energy,
        "station_ids": station_ids,
        "forecast_pred": forecast["pred"].astype(np.float32),
        "forecast_index": {int(hour): row for row, hour in enumerate(indices)},
        "valid_first_hours": indices[indices + args.horizon < raw.shape[0]],
        "horizon": args.horizon,
        "seed": args.seed,
        "total_charge": float(mean_energy.sum() * args.charge_capacity_factor),
        "total_comm": float(data["comm_capacity"].sum() * args.comm_capacity_factor),
        "total_restore": base_total_restore * args.restore_scale,
        "base_total_restore": base_total_restore,
        "energy_scale": args.energy_scale,
        **coefficient_values,
        **policy_coefficients,
        "restoration_effect": [args.road_restore_effect, args.power_restore_effect, args.comm_restore_effect],
        "backlog_carryover": args.backlog_carryover,
        "backlog_score": args.backlog_score,
        "comm_energy_weight": args.comm_energy_weight,
        "service_threat_weight": args.service_threat_weight,
        "comm_threat_weight": args.comm_threat_weight,
        "restore_power_weight": args.restore_power_weight,
        "restore_comm_weight": args.restore_comm_weight,
        "restore_road_weight": args.restore_road_weight,
        "restore_envelope_weight": args.restore_envelope_weight,
        "service_reserve_fraction": args.service_reserve_fraction,
        "restore_reserve_fraction": args.restore_reserve_fraction,
        "integer_crews": args.integer_crews,
        "crew_jobs": args.crew_jobs,
        "crew_service_scale": args.crew_service_scale,
        "crew_travel_minutes": crew_travel_minutes,
        "crew_depot_index": crew_depot_index,
        "crew_recovery_h": recovery_h,
        "comm_mobility_load": args.comm_mobility_load,
        "comm_energy_load": args.comm_energy_load,
        "comm_road_load": args.comm_road_load,
        "comm_power_load": args.comm_power_load,
        "comm_direct_derate": args.comm_direct_derate,
        "mobility_comm_weight": args.mobility_comm_weight,
        "cost_mobility": args.cost_mobility,
        "cost_unserved": args.cost_unserved,
        "cost_comm": args.cost_comm,
        "cost_power_service": args.cost_power_service,
        "cost_cascade": args.cost_cascade,
        "smartds_online": smartds_online,
        "smartds_master": str(args.smartds_master.resolve()),
        "smartds_load_names": load_names,
        "smartds_mappings": smartds_mappings,
        "smartds_power_factor": args.smartds_power_factor,
        "smartds_voltage_low": args.smartds_voltage_low,
        "smartds_voltage_high": args.smartds_voltage_high,
        "smartds_line_limit": args.smartds_line_limit,
        "smartds_bisection_iterations": args.smartds_bisection_iterations,
        "smartds_baseline": baseline_grid,
        "forecast_file": forecast_path.name,
        "robust_policy_variants": robust_variants,
        "central_policy_coefficients": central_policy_coefficients,
        "transition_coefficient_names": coefficient_names,
        "transition_uncertainty_bounds": uncertainty_bounds,
        "sample_transition_uncertainty": args.sample_transition_uncertainty,
        "normalization_train_end": training_end,
        "normalization_rule": "station means and resource budgets use only the forecasting training period",
        "crew_prior_cutoff": str(training_cutoff),
        "crew_prior_events": int(len(event_frame)),
        "packet_response_surface": packet_response_surface,
        "policies": policies,
    }
    return payload


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/external/processed/boulder_ev/boulder_ev_rollout_dataset.npz"),
    )
    parser.add_argument("--forecast-results", type=Path, default=Path("results/real_ev"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/operational/boulder_robust_rollout"),
    )
    parser.add_argument("--feeder-case", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenarios", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    parser.add_argument("--charge-capacity-factor", type=float, default=1.22)
    parser.add_argument("--comm-capacity-factor", type=float, default=0.72)
    parser.add_argument("--restore-scale", type=float, default=1.0)
    parser.add_argument("--integer-crews", type=int, default=4)
    parser.add_argument("--crew-jobs", type=int, default=36)
    parser.add_argument("--crew-events", type=Path, default=CREW_EVENTS)
    parser.add_argument("--crew-road-circuity", type=float, default=1.25)
    parser.add_argument("--crew-speed-kmh", type=float, default=30.0)
    parser.add_argument("--crew-service-scale", type=float, default=1.0)
    parser.add_argument("--threat-persistence", type=float, default=0.48)
    parser.add_argument("--road-persistence", type=float, default=None)
    parser.add_argument("--power-persistence", type=float, default=None)
    parser.add_argument("--comm-persistence", type=float, default=None)
    parser.add_argument("--spatial-spread", type=float, default=0.20)
    parser.add_argument("--pr-coupling", type=float, default=0.0)
    parser.add_argument("--pc-coupling", type=float, default=0.14)
    parser.add_argument("--rc-coupling", type=float, default=0.04)
    parser.add_argument("--cr-coupling", type=float, default=0.08)
    parser.add_argument("--cp-coupling", type=float, default=0.04)
    parser.add_argument("--policy-matrix-scale", type=float, default=1.0)
    parser.add_argument("--policy-matrix-noise", type=float, default=0.0)
    parser.add_argument("--policy-matrix-seed", type=int, default=31991)
    parser.add_argument(
        "--robust-uncertainty-file",
        type=Path,
        default=Path("results/operational/transition_calibration/transition_parameter_uncertainty.csv"),
    )
    parser.add_argument(
        "--sample-transition-uncertainty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample realized transition coefficients from the calibrated intervals "
            "for every scenario (default: enabled). Use "
            "--no-sample-transition-uncertainty only for a fixed-central ablation."
        ),
    )
    parser.add_argument(
        "--packet-results",
        type=Path,
        default=Path("results/operational/packet_network/packet_network_scenarios.csv"),
    )
    parser.add_argument("--packet-traffic-multiplier", type=float, default=1.0)
    parser.add_argument("--packet-backup-duration-s", type=float, default=60.0)
    parser.add_argument("--road-restore-effect", type=float, default=0.08)
    parser.add_argument("--power-restore-effect", type=float, default=0.12)
    parser.add_argument("--comm-restore-effect", type=float, default=0.10)
    parser.add_argument("--backlog-carryover", type=float, default=0.22)
    parser.add_argument("--backlog-score", type=float, default=1.0)
    parser.add_argument("--comm-energy-weight", type=float, default=0.70)
    parser.add_argument("--service-threat-weight", type=float, default=1.50)
    parser.add_argument("--comm-threat-weight", type=float, default=1.50)
    parser.add_argument("--restore-power-weight", type=float, default=1.60)
    parser.add_argument("--restore-comm-weight", type=float, default=1.30)
    parser.add_argument("--restore-road-weight", type=float, default=1.10)
    parser.add_argument("--restore-envelope-weight", type=float, default=0.80)
    parser.add_argument("--service-reserve-fraction", type=float, default=0.04)
    parser.add_argument("--restore-reserve-fraction", type=float, default=0.02)
    parser.add_argument("--comm-mobility-load", type=float, default=0.13)
    parser.add_argument("--comm-energy-load", type=float, default=0.19)
    parser.add_argument("--comm-road-load", type=float, default=6.0)
    parser.add_argument("--comm-power-load", type=float, default=3.5)
    parser.add_argument("--comm-direct-derate", type=float, default=0.40)
    parser.add_argument("--mobility-comm-weight", type=float, default=0.45)
    parser.add_argument("--cost-mobility", type=float, default=1.8)
    parser.add_argument("--cost-unserved", type=float, default=2.6)
    parser.add_argument("--cost-comm", type=float, default=1.7)
    parser.add_argument("--cost-power-service", type=float, default=1.2)
    parser.add_argument("--cost-cascade", type=float, default=12.0)
    parser.add_argument("--smartds-master", type=Path, default=SMARTDS_MASTER)
    parser.add_argument("--smartds-coordinates", type=Path, default=SMARTDS_COORDS)
    parser.add_argument("--smartds-mappings", type=int, default=20)
    parser.add_argument("--smartds-mapping-seed", type=int, default=20260825)
    parser.add_argument("--smartds-power-factor", type=float, default=0.98)
    parser.add_argument("--smartds-voltage-low", type=float, default=0.95)
    parser.add_argument("--smartds-voltage-high", type=float, default=1.05)
    parser.add_argument("--smartds-line-limit", type=float, default=1.0)
    parser.add_argument("--smartds-bisection-iterations", type=int, default=14)
    parser.add_argument("--disable-smartds-online", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = payload_from_args(args)
    central_args = argparse.Namespace(
        threat_persistence=args.threat_persistence,
        road_persistence=payload["road_persistence"],
        power_persistence=payload["power_persistence"],
        comm_persistence=payload["comm_persistence"],
        spatial_spread=payload["spatial_spread"],
        pr_coupling=payload["pr_coupling"],
        pc_coupling=payload["pc_coupling"],
        rc_coupling=payload["rc_coupling"],
        cr_coupling=payload["cr_coupling"],
        cp_coupling=payload["cp_coupling"],
    )
    spectral_radius = estimate_transition_spectral_radius(payload["adj"], central_args)
    policy_args = argparse.Namespace(
        threat_persistence=args.threat_persistence,
        road_persistence=payload["policy_road_persistence"],
        power_persistence=payload["policy_power_persistence"],
        comm_persistence=payload["policy_comm_persistence"],
        spatial_spread=payload["policy_spatial_spread"],
        pr_coupling=payload["policy_pr_coupling"],
        pc_coupling=payload["policy_pc_coupling"],
        rc_coupling=payload["policy_rc_coupling"],
        cr_coupling=payload["policy_cr_coupling"],
        cp_coupling=payload["policy_cp_coupling"],
    )
    policy_spectral_radius = estimate_transition_spectral_radius(payload["adj"], policy_args)
    tasks = [(index, GROUPS[index % len(GROUPS)]) for index in range(args.scenarios)]
    started = time.time()
    rows = []
    if args.workers <= 1:
        init_worker(payload)
        for task in tasks:
            rows.extend(simulate_one(task))
    else:
        with mp.Pool(args.workers, initializer=init_worker, initargs=(payload,)) as pool:
            for chunk in pool.imap_unordered(simulate_one, tasks, chunksize=4):
                rows.extend(chunk)
                if len(rows) % 2000 == 0:
                    print(f"simulated policy rows={len(rows)} elapsed={time.time() - started:.1f}s", flush=True)
    summary = aggregate(rows)
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "rollout_scenarios.csv", rows)
    write_csv(args.results / "rollout_summary.csv", summary)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update(
        {
            # The parser retains legacy transition defaults for backward
            # compatibility, but the calibrated ledger overwrites them before
            # any policy or realized transition is evaluated.  Record the
            # coefficients actually used here so the manifest cannot appear
            # to contradict Table III.
            "threat_persistence": "deprecated; see domain-specific calibrated persistence columns",
            **{
                name: payload[name]
                for name in payload["transition_coefficient_names"]
            },
            "seconds": time.time() - started,
            "forecast_file": payload["forecast_file"],
            "normalization_train_end": payload["normalization_train_end"],
            "normalization_rule": payload["normalization_rule"],
            "transition_spectral_radius_estimate": spectral_radius,
            "policy_transition_spectral_radius_estimate": policy_spectral_radius,
            "policies": " ".join(payload["policies"]),
            "robust_policy_variant_count": len(payload["robust_policy_variants"]),
            "information_rule": (
                "forecast_matched, pc_rollout, and robust_pc_rollout use the same demand "
                "forecast, current threat, backlog, and finite route portfolio; static uses "
                "historical station means, greedy uses the common forecast with a direct "
                "current-threat route, and oracle alone receives future innovations"
            ),
            "execution_rule": (
                "packet delivery gates remote actions; SMART-DS projects every hourly charging action; "
                "only routed crew completion events clear repaired threat states"
            ),
            "route_candidate_set": (
                "static greedy forecast_matched pc_rollout robust_pc_rollout uniform_route"
            ),
            "route_selection_rule": (
                "forecast_matched minimizes current-risk-weighted completion; pc_rollout minimizes "
                "calibrated-central transition loss; robust_pc_rollout minimizes worst predicted "
                "transition loss over the same route candidates"
            ),
            **{
                f"effective_{name}": payload["central_policy_coefficients"][name]
                for name in payload["transition_coefficient_names"]
            },
            "smartds_baseline_v_min_pu": payload["smartds_baseline"]["v_min_pu"],
            "smartds_baseline_v_max_pu": payload["smartds_baseline"]["v_max_pu"],
            "smartds_baseline_max_line_loading_pu": payload["smartds_baseline"]["max_line_loading_pu"],
        }
    )
    write_csv(args.results / "rollout_manifest.csv", [manifest])
    mapping_rows = []
    for mapping_index, mapping in enumerate(payload["smartds_mappings"]):
        for station_index, load_index in enumerate(mapping):
            mapping_rows.append(
                {
                    "mapping_index": mapping_index,
                    "station_index": station_index,
                    "station_id": payload["station_ids"][station_index],
                    "smartds_load_index": int(load_index),
                    "smartds_load_name": payload["smartds_load_names"][int(load_index)],
                }
            )
    write_csv(args.results / "station_to_smartds_load_mappings.csv", mapping_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
