#!/usr/bin/env python3
"""Auditable LinDistFlow validation on the MATPOWER case33bw feeder.

The rollout controller does not use these metrics when constructing its action.
They are therefore an external electrical-feasibility check rather than another
term in the policy score.  Zone-level hourly charging energy is mapped
deterministically to the 32 load buses and interpreted as average MW over the
one-hour decision interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class RadialFeeder:
    base_mva: float
    base_kv: float
    p_base_mw: np.ndarray
    q_base_mvar: np.ndarray
    parent: np.ndarray
    child: np.ndarray
    r_pu: np.ndarray
    x_pu: np.ndarray
    v_min: float = 0.90
    v_max: float = 1.10

    @property
    def n_buses(self) -> int:
        return int(self.p_base_mw.size)


def _matrix(text: str, name: str) -> np.ndarray:
    match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, flags=re.S)
    if match is None:
        raise ValueError(f"matrix mpc.{name} not found")
    rows = []
    for raw in match.group(1).splitlines():
        raw = raw.split("%", 1)[0].strip().rstrip(";")
        if raw:
            rows.append([float(value) for value in raw.split()])
    return np.asarray(rows, dtype=np.float64)


def load_case33bw(path: Path) -> RadialFeeder:
    """Load the official MATPOWER case33bw file without a MATLAB dependency."""

    text = path.read_text(encoding="utf-8")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9.]+)", text)
    if base_match is None:
        raise ValueError("baseMVA not found")
    base_mva = float(base_match.group(1))
    bus = _matrix(text, "bus")
    branch = _matrix(text, "branch")
    active = branch[:, 10] > 0.5
    branch = branch[active]
    base_kv = float(bus[0, 9])
    z_base_ohm = base_kv**2 / base_mva
    parent = branch[:, 0].astype(np.int64) - 1
    child = branch[:, 1].astype(np.int64) - 1
    if branch.shape[0] != bus.shape[0] - 1:
        raise ValueError("case33bw active branches are not radial")
    return RadialFeeder(
        base_mva=base_mva,
        base_kv=base_kv,
        p_base_mw=bus[:, 2] / 1000.0,
        q_base_mvar=bus[:, 3] / 1000.0,
        parent=parent,
        child=child,
        r_pu=branch[:, 2] / z_base_ohm,
        x_pu=branch[:, 3] / z_base_ohm,
        v_min=float(np.min(bus[:, 12])),
        v_max=float(np.max(bus[:, 11])),
    )


def demand_balanced_zone_mapping(mean_zone_energy: np.ndarray, feeder: RadialFeeder) -> np.ndarray:
    """Map zones to load buses using a deterministic demand-balancing rule.

    This is a benchmark mapping, not a claim that a TLC taxi zone corresponds to
    a particular physical bus.  The mapping is stable and uses only historical
    mean charging energy and the feeder's base-load proportions.
    """

    energy = np.maximum(np.asarray(mean_zone_energy, dtype=np.float64), 0.0)
    load_buses = np.flatnonzero(feeder.p_base_mw > 0.0)
    target = feeder.p_base_mw[load_buses]
    target = target / target.sum()
    assigned = np.zeros(load_buses.size, dtype=np.float64)
    mapping = np.empty(energy.size, dtype=np.int64)
    for zone in np.argsort(-energy, kind="stable"):
        score = assigned / np.maximum(target, 1e-12)
        slot = int(np.argmin(score))
        mapping[zone] = load_buses[slot]
        assigned[slot] += max(float(energy[zone]), 1e-9)
    return mapping


def aggregate_zones(values: np.ndarray, zone_to_bus: np.ndarray, n_buses: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != zone_to_bus.size:
        raise ValueError("last dimension must match zone mapping")
    out = np.zeros((*values.shape[:-1], n_buses), dtype=np.float64)
    for zone, bus in enumerate(zone_to_bus):
        out[..., bus] += values[..., zone]
    return out


def _radial_flows(feeder: RadialFeeder, p_pu: np.ndarray, q_pu: np.ndarray):
    p_sub = p_pu.copy()
    q_sub = q_pu.copy()
    p_flow = np.zeros(feeder.parent.size, dtype=np.float64)
    q_flow = np.zeros_like(p_flow)
    for edge in range(feeder.parent.size - 1, -1, -1):
        child = feeder.child[edge]
        parent = feeder.parent[edge]
        p_flow[edge] = p_sub[child]
        q_flow[edge] = q_sub[child]
        p_sub[parent] += p_sub[child]
        q_sub[parent] += q_sub[child]
    return p_flow, q_flow


def solve_lindistflow(
    feeder: RadialFeeder,
    ev_bus_mw: np.ndarray,
    *,
    ev_power_factor: float = 0.97,
    thermal_margin: float = 1.25,
):
    """Run a loss-aware diagnostic LinDistFlow approximation for one hour."""

    ev_bus_mw = np.maximum(np.asarray(ev_bus_mw, dtype=np.float64), 0.0)
    if ev_bus_mw.shape != feeder.p_base_mw.shape:
        raise ValueError("EV bus load shape does not match feeder")
    pf = float(np.clip(ev_power_factor, 1e-3, 1.0))
    q_ratio = np.tan(np.arccos(pf))
    p_pu = (feeder.p_base_mw + ev_bus_mw) / feeder.base_mva
    q_pu = (feeder.q_base_mvar + q_ratio * ev_bus_mw) / feeder.base_mva
    p_flow, q_flow = _radial_flows(feeder, p_pu, q_pu)

    base_p, base_q = _radial_flows(
        feeder,
        feeder.p_base_mw / feeder.base_mva,
        feeder.q_base_mvar / feeder.base_mva,
    )
    base_s = np.sqrt(base_p**2 + base_q**2)
    thermal_limit = np.maximum(thermal_margin * base_s, 0.01)

    v_sq = np.ones(feeder.n_buses, dtype=np.float64)
    losses_pu = 0.0
    for edge, (parent, child) in enumerate(zip(feeder.parent, feeder.child)):
        current_sq = (p_flow[edge] ** 2 + q_flow[edge] ** 2) / max(v_sq[parent], 1e-8)
        losses_pu += feeder.r_pu[edge] * current_sq
        v_sq[child] = max(
            v_sq[parent]
            - 2.0 * (feeder.r_pu[edge] * p_flow[edge] + feeder.x_pu[edge] * q_flow[edge])
            + (feeder.r_pu[edge] ** 2 + feeder.x_pu[edge] ** 2) * current_sq,
            0.0,
        )
    voltage = np.sqrt(v_sq)
    apparent = np.sqrt(p_flow**2 + q_flow**2)
    low_violation = np.maximum(feeder.v_min - voltage, 0.0)
    high_violation = np.maximum(voltage - feeder.v_max, 0.0)
    thermal_violation = np.maximum(apparent / thermal_limit - 1.0, 0.0)
    return {
        "min_voltage_pu": float(voltage.min()),
        "voltage_violation_pu": float((low_violation + high_violation).sum()),
        "voltage_violating_buses": int(np.count_nonzero(low_violation + high_violation)),
        "thermal_overload_pu": float(thermal_violation.sum()),
        "thermal_overloaded_branches": int(np.count_nonzero(thermal_violation)),
        "losses_mw": float(losses_pu * feeder.base_mva),
    }


def evaluate_hourly_charging(
    feeder: RadialFeeder,
    served_energy_kwh: np.ndarray,
    zone_to_bus: np.ndarray,
    *,
    ev_power_factor: float = 0.97,
    thermal_margin: float = 1.25,
):
    """Evaluate an H-by-N zone charging tensor on the benchmark feeder."""

    served = np.asarray(served_energy_kwh, dtype=np.float64)
    if served.ndim != 2:
        raise ValueError("served_energy_kwh must have shape [hours, zones]")
    # kWh delivered during a one-hour interval equals average kW; divide by
    # 1000 to obtain average MW.
    bus_mw = aggregate_zones(served / 1000.0, zone_to_bus, feeder.n_buses)
    rows = [
        solve_lindistflow(
            feeder,
            hour,
            ev_power_factor=ev_power_factor,
            thermal_margin=thermal_margin,
        )
        for hour in bus_mw
    ]
    return {
        "min_voltage_pu": min(row["min_voltage_pu"] for row in rows),
        "voltage_violation_pu_hours": sum(row["voltage_violation_pu"] for row in rows),
        "voltage_violating_bus_hours": sum(row["voltage_violating_buses"] for row in rows),
        "thermal_overload_pu_hours": sum(row["thermal_overload_pu"] for row in rows),
        "thermal_overloaded_branch_hours": sum(row["thermal_overloaded_branches"] for row in rows),
        "losses_mwh": sum(row["losses_mw"] for row in rows),
    }

