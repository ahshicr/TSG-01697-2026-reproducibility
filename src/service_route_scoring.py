"""Candidate service-cost evaluation using only the forecast issued at dispatch.

This development module leaves the historical simulator unchanged while its
reference experiments finish. The score contains no OpenDSS call, packet lookup,
future observation, innovation, or sampled realized propagation coefficient.
The no-transition control holds each unrepaired threat at its current value.
Both controls honor exactly the same routed repair completion schedule.
"""
from __future__ import annotations

import numpy as np

import simulate_rollout_revised as sim


def continuous_service_cost(threat, backlog, forecast, step):
    """One ungated service step, with allocations from the remaining issued forecast."""
    remaining = forecast[step:]
    demand_score = remaining[..., 0].sum(axis=0)
    energy_score = remaining[..., 1].sum(axis=0) + sim.G["backlog_score"] * backlog
    comm_score = demand_score + sim.G["comm_energy_weight"] * energy_score
    energy_score = energy_score + sim.G["service_threat_weight"] * threat.sum(axis=0)
    comm_score = comm_score + sim.G["comm_threat_weight"] * threat.sum(axis=0)
    charge = sim.allocate(sim.G["total_charge"], energy_score, sim.G["service_reserve_fraction"])
    communication = sim.allocate(sim.G["total_comm"], comm_score, sim.G["service_reserve_fraction"])
    road, power, comm = threat
    demand = forecast[step, :, 0]
    required = forecast[step, :, 1] * sim.G["energy_scale"] + sim.G["backlog_carryover"] * backlog
    load = (sim.G["comm_mobility_load"] * demand + sim.G["comm_energy_load"] * required
            + sim.G["comm_road_load"] * road + sim.G["comm_power_load"] * power)
    comm_loss = np.maximum(load - communication * (1 - power)
                           * (1 - sim.G["comm_direct_derate"] * comm), 0)
    support = 1 - np.clip(comm_loss / (load + 1), 0, .9)
    served = np.minimum(required, charge * (1 - power) * support)
    unserved = np.maximum(required - served, 0)
    mobility_loss = demand * (road + sim.G["mobility_comm_weight"] * (1 - support))
    risk = (sim.G["cost_mobility"] * mobility_loss + sim.G["cost_unserved"] * unserved
            + sim.G["cost_comm"] * comm_loss + sim.G["cost_power_service"] * power * required
            + sim.G["cost_cascade"] * (power * comm + road * (1 - support)))
    return float(risk.sum()), unserved


def projected_service_cost(current_threat, forecast, plan, policy_coefficients, *,
                           propagate=True, initial_backlog=None):
    """Predict service cost without consulting any realized evaluation variables."""
    threat = current_threat.copy().astype(np.float32)
    backlog = (np.zeros(threat.shape[1], dtype=np.float32) if initial_backlog is None
               else np.asarray(initial_backlog, dtype=np.float32).copy())
    total = 0.0
    zero = np.zeros(threat.shape[1], dtype=np.float32)
    for step in range(sim.G["horizon"]):
        stage, backlog = continuous_service_cost(threat, backlog, forecast, step)
        total += stage
        if step + 1 < sim.G["horizon"]:
            if propagate:
                threat = sim.threat_step(threat, np.zeros_like(threat), zero,
                                         policy_model=True, coefficient_override=policy_coefficients)
            repaired = plan["completion_by_zone"] <= step + 1.0
            threat[:, repaired] = 0.0
    return total


def select_service_route(selector_policy, first_hour, threat, backlog, packet_fraction,
                         crew_scenario, selector_priority):
    plans = sim.route_candidate_plans(first_hour, threat, backlog, packet_fraction, crew_scenario)
    forecast = sim.forecast_at(first_hour, sim.G["horizon"])
    scores = {}
    for name, plan in plans.items():
        if selector_policy == "robust_pc_rollout":
            scores[name] = max(projected_service_cost(threat, forecast, plan, matrix,
                                                      initial_backlog=backlog)
                               for matrix in sim.G["robust_policy_variants"])
        else:
            scores[name] = projected_service_cost(
                threat, forecast, plan, sim.G["central_policy_coefficients"],
                propagate=selector_policy != "forecast_matched", initial_backlog=backlog)
    selected = min(scores, key=lambda name: (scores[name], name))
    result = dict(plans[selected])
    result.update(source_policy=selected, route_score=float(scores[selected]))
    return result


def causal_forecast_at(first_hour, remaining):
    row = sim.G["forecast_index"].get(first_hour)
    if row is None:
        # A missing forecast must never expose realized future demand.
        values = sim.G["raw"][first_hour - 1:first_hour]
    else:
        values = sim.G["forecast_pred"][row, :remaining]
    if len(values) < remaining:
        values = np.concatenate([values, np.repeat(values[-1:], remaining - len(values), axis=0)])
    return values.astype(np.float32)


def install():
    sim.route_portfolio_plan = select_service_route
    sim.forecast_at = causal_forecast_at


def init_worker(payload):
    sim.init_worker(payload)
    install()
