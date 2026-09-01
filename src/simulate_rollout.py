#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import time

import numpy as np


G = {}


POLICIES = ["static", "greedy", "plain_rollout", "pc_rollout", "oracle"]
GROUPS = ["nominal", "single_domain", "cascade", "ood"]


def read_best_forecast(metrics_path: Path, mode: str) -> Path:
    rows = []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == mode:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no metrics for mode={mode}")
    best = min(rows, key=lambda r: float(r["mae_demand"]) + 0.25 * float(r["mae_energy"]))
    return metrics_path.parent / f"forecast_{mode}_seed{best['seed']}.npz"


def init_worker(payload):
    G.update(payload)


def allocate(total: float, score: np.ndarray, floor_fraction: float = 0.04) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    score = np.maximum(score, 0.0)
    n = score.size
    base = total * floor_fraction / n
    remaining = max(total - base * n, 1e-9)
    if score.sum() <= 1e-12:
        return np.full(n, total / n, dtype=np.float32)
    weights = np.sqrt(score + 1e-9)
    weights /= weights.sum()
    return (base + remaining * weights).astype(np.float32)


def threat_field(rng: np.random.Generator, group: str, sample: int):
    adj = G["adj"]
    n = adj.shape[0]
    h = G["horizon"]
    seed = np.zeros(n, dtype=np.float32)
    mean_demand = G["mean_demand"]
    probs = mean_demand + 0.05 * mean_demand.mean()
    probs = probs / probs.sum()
    if group == "nominal":
        k, sev, rho = 2, 0.08, 0.10
        domains = ["road"]
    elif group == "single_domain":
        k, sev, rho = 6, 0.35, 0.22
        domains = [rng.choice(["road", "power", "comm"])]
    elif group == "cascade":
        k, sev, rho = 8, 0.45, 0.32
        domains = ["road", "power", "comm"]
    else:
        k, sev, rho = 12, 0.62, 0.42
        domains = ["road", "power", "comm"]
    zones = rng.choice(n, size=k, replace=False, p=probs)
    seed[zones] = sev * rng.uniform(0.75, 1.25, size=k)
    road = np.zeros((h, n), dtype=np.float32)
    power = np.zeros_like(road)
    comm = np.zeros_like(road)
    active = seed
    for t in range(h):
        active = np.clip(active, 0.0, 0.95)
        if "road" in domains:
            road[t] = active * rng.uniform(0.85, 1.10)
        if "power" in domains:
            power[t] = active * rng.uniform(0.80, 1.05)
        if "comm" in domains:
            comm[t] = active * rng.uniform(0.75, 1.15)
        spread = adj @ active
        active = (0.62 * active + rho * spread) * rng.uniform(0.92, 1.03)
    if group in {"cascade", "ood"}:
        comm = np.clip(
            comm + G["pc_coupling"] * power + G["rc_coupling"] * (adj @ road.T).T,
            0,
            0.98,
        )
        road = np.clip(road + G["cr_coupling"] * comm, 0, 0.98)
        power = np.clip(power + G["cp_coupling"] * (adj @ comm.T).T, 0, 0.98)
    return {"road": road, "power": power, "comm": comm, "group": group, "sample": sample}


def policy_scores(policy: str, sample: int, threat, current):
    mean_demand = G["mean_demand"]
    mean_energy = G["mean_energy"]
    adj = G["adj"]
    plain = G["plain_pred"][sample]
    physics = G["physics_pred"][sample]
    truth = G["truth"][sample]
    threat_now = threat["road"][0] + threat["power"][0] + threat["comm"][0]
    future_threat = threat["road"].sum(axis=0) + threat["power"].sum(axis=0) + threat["comm"].sum(axis=0)
    if policy == "static":
        demand_score = mean_demand
        energy_score = mean_energy
        comm_score = mean_demand + 0.8 * mean_energy
        restore_score = mean_demand
    elif policy == "greedy":
        demand_score = current[:, 0] + 4.0 * threat_now
        energy_score = current[:, 1] + 3.5 * threat_now
        comm_score = current[:, 0] + current[:, 1] + 4.0 * threat_now
        restore_score = threat_now
    elif policy == "plain_rollout":
        demand_score = plain[..., 0].sum(axis=0) + 1.5 * threat_now
        energy_score = plain[..., 1].sum(axis=0) + 1.5 * threat_now
        comm_score = demand_score + 0.7 * energy_score
        restore_score = threat_now + 0.2 * adj @ threat_now
    elif policy == "pc_rollout":
        forecast = plain
        demand_score = forecast[..., 0].sum(axis=0) + 1.5 * threat_now
        energy_score = forecast[..., 1].sum(axis=0) + 1.5 * threat_now
        comm_score = demand_score + 0.7 * energy_score
        vulnerability = future_threat + 0.8 * adj @ future_threat
        restore_score = 1.6 * threat["power"].sum(axis=0)
        restore_score += 1.3 * threat["comm"].sum(axis=0) + 1.1 * threat["road"].sum(axis=0)
        restore_score += 0.8 * vulnerability
    elif policy == "oracle":
        demand_score = truth[..., 0].sum(axis=0) + 2.0 * future_threat
        energy_score = truth[..., 1].sum(axis=0) + 2.0 * threat["power"].sum(axis=0)
        comm_score = demand_score + energy_score + 1.5 * threat["comm"].sum(axis=0)
        restore_score = future_threat
    else:
        raise ValueError(policy)
    return demand_score, energy_score, comm_score, restore_score


def evaluate_policy(policy: str, sample: int, threat, current):
    truth = G["truth"][sample]
    base_power = G["base_power"]
    feeder_limit = G["feeder_limit"]
    comm_capacity = G["comm_capacity"]
    total_charge = G["total_charge"]
    total_comm = G["total_comm"]
    total_restore = G["total_restore"]
    demand_score, energy_score, comm_score, restore_score = policy_scores(policy, sample, threat, current)
    t0 = time.perf_counter()
    charge_cap = allocate(total_charge, energy_score)
    comm_cap = allocate(total_comm, comm_score)
    restore = allocate(total_restore, restore_score, floor_fraction=0.02)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    budget_factor = total_restore / max(G["base_total_restore"], 1e-6)
    restore_norm = np.clip(restore / (restore.max() + 1e-6) * budget_factor, 0.0, 1.2)
    metrics = {
        "cost": 0.0,
        "mobility_delay": 0.0,
        "unserved_energy": 0.0,
        "comm_loss": 0.0,
        "voltage_violation": 0.0,
        "peak_risk": 0.0,
        "served_mobility": 0.0,
        "total_mobility": 0.0,
        "latency_ms": latency_ms,
    }
    backlog = np.zeros_like(base_power)
    for h in range(truth.shape[0]):
        demand = truth[h, :, 0]
        energy = truth[h, :, 1] + 0.22 * backlog
        road = np.clip(threat["road"][h] * (1.0 - 0.35 * restore_norm), 0.0, 0.98)
        power = np.clip(threat["power"][h] * (1.0 - 0.55 * restore_norm), 0.0, 0.98)
        comm = np.clip(threat["comm"][h] * (1.0 - 0.45 * restore_norm), 0.0, 0.98)
        power_avail = 1.0 - power
        comm_load = 0.13 * demand + 0.19 * energy + 6.0 * road + 3.5 * power
        served_comm = np.minimum(comm_load, comm_cap * power_avail * (1.0 - 0.4 * comm))
        comm_loss = np.maximum(comm_load - served_comm, 0.0)
        comm_factor = 1.0 - np.clip(comm_loss / (comm_load + 1.0), 0.0, 0.9)
        served_energy = np.minimum(energy, charge_cap * power_avail * comm_factor)
        unserved_energy = np.maximum(energy - served_energy, 0.0)
        effective_load = base_power + served_energy
        limit = feeder_limit * (1.0 - G["power_derate"] * power)
        voltage_violation = np.maximum(effective_load - limit, 0.0)
        signal_support = np.clip(served_comm / (comm_load + 1.0), 0.0, 1.0)
        mobility_delay = demand * (road + 0.45 * (1.0 - signal_support)) + 0.03 * unserved_energy
        served_mobility = np.maximum(demand - mobility_delay, 0.0)
        cascade = power * comm + road * (1.0 - signal_support)
        risk = (
            1.8 * mobility_delay
            + 2.6 * unserved_energy
            + 1.7 * comm_loss
            + 4.2 * voltage_violation
            + 12.0 * cascade
        )
        metrics["cost"] += float(risk.sum())
        metrics["mobility_delay"] += float(mobility_delay.sum())
        metrics["unserved_energy"] += float(unserved_energy.sum())
        metrics["comm_loss"] += float(comm_loss.sum())
        metrics["voltage_violation"] += float(voltage_violation.sum())
        metrics["served_mobility"] += float(served_mobility.sum())
        metrics["total_mobility"] += float(demand.sum())
        metrics["peak_risk"] = max(metrics["peak_risk"], float(risk.max()))
        backlog = np.maximum(unserved_energy - 0.25 * charge_cap, 0.0)
    metrics["service_continuity"] = metrics["served_mobility"] / max(metrics["total_mobility"], 1e-6)
    return metrics


def simulate_one(args_tuple):
    scenario_id, group = args_tuple
    rng = np.random.default_rng(G["seed"] + scenario_id * 7919)
    n_samples = G["truth"].shape[0]
    sample = int(rng.integers(0, n_samples))
    first_hour = int(G["indices"][sample])
    raw = G["raw"]
    current = raw[first_hour - 1]
    threat = threat_field(rng, group, sample)
    rows = []
    for policy in POLICIES:
        metrics = evaluate_policy(policy, sample, threat, current)
        row = {
            "scenario_id": scenario_id,
            "group": group,
            "sample": sample,
            "first_hour": first_hour,
            "policy": policy,
            **metrics,
        }
        rows.append(row)
    return rows


def mean_ci(values):
    values = list(map(float, values))
    n = len(values)
    mean = statistics.fmean(values) if values else float("nan")
    if n <= 1:
        return mean, 0.0
    stdev = statistics.stdev(values)
    return mean, 1.96 * stdev / math.sqrt(n)


def aggregate(rows):
    keys = sorted({(r["policy"], r["group"]) for r in rows})
    out = []
    metrics = [
        "cost",
        "mobility_delay",
        "unserved_energy",
        "comm_loss",
        "voltage_violation",
        "peak_risk",
        "service_continuity",
        "latency_ms",
    ]
    for policy, group in keys:
        subset = [r for r in rows if r["policy"] == policy and r["group"] == group]
        record = {"policy": policy, "group": group, "n": len(subset)}
        for metric in metrics:
            mean, ci = mean_ci([r[metric] for r in subset])
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/nyc_tlc_2023_hourly.npz"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--scenarios", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    parser.add_argument("--feeder-margin-scale", type=float, default=1.0)
    parser.add_argument("--charge-capacity-factor", type=float, default=1.22)
    parser.add_argument("--restore-scale", type=float, default=1.0)
    parser.add_argument("--pc-coupling", type=float, default=0.28)
    parser.add_argument("--rc-coupling", type=float, default=0.08)
    parser.add_argument("--cr-coupling", type=float, default=0.18)
    parser.add_argument("--cp-coupling", type=float, default=0.08)
    parser.add_argument("--power-derate", type=float, default=0.35)
    parser.add_argument("--restore-reference-multiplier", type=float, default=4.0)
    args = parser.parse_args()

    metrics_path = args.results / "prediction_metrics.csv"
    physics_path = read_best_forecast(metrics_path, "physics")
    plain_path = read_best_forecast(metrics_path, "plain")
    data = np.load(args.data, allow_pickle=True)
    physics = np.load(physics_path)
    plain = np.load(plain_path)
    truth = physics["truth"].astype(np.float32).copy()
    physics_pred = physics["pred"].astype(np.float32).copy()
    plain_pred = plain["pred"].astype(np.float32).copy()
    raw = np.stack([data["pickup"], data["energy"]], axis=-1).astype(np.float32)
    raw[..., 1] *= args.energy_scale
    truth[..., 1] *= args.energy_scale
    physics_pred[..., 1] *= args.energy_scale
    plain_pred[..., 1] *= args.energy_scale
    base_power = data["base_power"].astype(np.float32)
    feeder_limit_base = data["feeder_limit"].astype(np.float32)
    feeder_limit = base_power + (feeder_limit_base - base_power) * args.feeder_margin_scale
    payload = {
        "raw": raw,
        "adj": data["adj"].astype(np.float32),
        "base_power": base_power,
        "feeder_limit": feeder_limit.astype(np.float32),
        "comm_capacity": data["comm_capacity"].astype(np.float32),
        "mean_demand": data["pickup"].mean(axis=0).astype(np.float32),
        "mean_energy": raw[..., 1].mean(axis=0).astype(np.float32),
        "base_mean_energy": data["energy"].mean(axis=0).astype(np.float32),
        "truth": truth,
        "physics_pred": physics_pred,
        "plain_pred": plain_pred,
        "indices": physics["indices"].astype(np.int64),
        "horizon": truth.shape[1],
        "seed": args.seed,
        "pc_coupling": args.pc_coupling,
        "rc_coupling": args.rc_coupling,
        "cr_coupling": args.cr_coupling,
        "cp_coupling": args.cp_coupling,
        "power_derate": args.power_derate,
    }
    payload["total_charge"] = float(payload["base_mean_energy"].sum() * args.charge_capacity_factor)
    payload["total_comm"] = float(payload["comm_capacity"].sum() * 0.72)
    base_total_restore = float(max(10.0, payload["mean_demand"].sum() * 0.018))
    payload["total_restore"] = float(base_total_restore * args.restore_scale)
    payload["base_total_restore"] = float(base_total_restore)
    payload["restore_reference"] = float(base_total_restore / payload["adj"].shape[0] * args.restore_reference_multiplier)
    tasks = [(i, GROUPS[i % len(GROUPS)]) for i in range(args.scenarios)]
    started = time.time()
    rows = []
    if args.workers <= 1:
        init_worker(payload)
        for task in tasks:
            rows.extend(simulate_one(task))
    else:
        with mp.Pool(args.workers, initializer=init_worker, initargs=(payload,)) as pool:
            for chunk in pool.imap_unordered(simulate_one, tasks, chunksize=8):
                rows.extend(chunk)
                if len(rows) % 2000 == 0:
                    print(f"simulated policy rows={len(rows)} elapsed={time.time() - started:.1f}s", flush=True)
    summary = aggregate(rows)
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "rollout_scenarios.csv", rows)
    write_csv(args.results / "rollout_summary.csv", summary)
    write_csv(args.results / "rollout_manifest.csv", [{
        "scenarios": args.scenarios,
        "workers": args.workers,
        "seconds": time.time() - started,
        "physics_forecast": physics_path.name,
        "plain_forecast": plain_path.name,
        "total_charge": payload["total_charge"],
        "total_comm": payload["total_comm"],
        "total_restore": payload["total_restore"],
        "energy_scale": args.energy_scale,
        "feeder_margin_scale": args.feeder_margin_scale,
        "charge_capacity_factor": args.charge_capacity_factor,
        "restore_scale": args.restore_scale,
        "pc_coupling": args.pc_coupling,
        "rc_coupling": args.rc_coupling,
        "cr_coupling": args.cr_coupling,
        "cp_coupling": args.cp_coupling,
        "power_derate": args.power_derate,
        "restore_reference_multiplier": args.restore_reference_multiplier,
    }])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
