"""Packet-level validation of the hourly communication-service proxy.

This is a reproducible finite-buffer discrete-event queue with packet erasures,
retransmission backoff, and control deadlines. It does not claim to reproduce a
specific utility protocol; all traffic assumptions are exposed to sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EV_DATA = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_forecast_dataset.npz"
EVENTS = ROOT / "data" / "external" / "processed" / "eaglei_boulder" / "boulder_outage_events_threshold_sensitivity.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            value.update(chunk)
    return value.hexdigest()


def simulate_link(
    rng: np.random.Generator,
    arrival_rate_pps: float,
    service_rate_pps: float,
    duration_s: float,
    buffer_packets: int,
    control_fraction: float,
    erasure_probability: float,
    max_retries: int,
    backoff_mean_s: float,
    deadline_s: float,
    backup_duration_s: float = 0.0,
    degraded_service_rate_pps: float | None = None,
) -> dict:
    """Run an M/M/1/K-like link with erasures and retransmission events."""
    if arrival_rate_pps <= 0 or service_rate_pps <= 0:
        raise ValueError("Arrival and service rates must be positive")
    events = []
    sequence = 0
    time = 0.0
    packet_id = 0
    control_generated = 0
    while True:
        time += float(rng.exponential(1.0 / arrival_rate_pps))
        if time > duration_s:
            break
        control = bool(rng.random() < control_fraction)
        control_generated += int(control)
        packet = (packet_id, control, 0, time)
        heapq.heappush(events, (time, sequence, "arrival", packet))
        sequence += 1
        packet_id += 1

    waiting: deque[tuple] = deque()
    current = None
    generated = packet_id
    arrivals_attempted = 0
    buffer_drops = 0
    erased = 0
    delivered = 0
    control_delivered = 0
    control_failed = 0
    control_latency = []

    def schedule_service(now: float, packet: tuple) -> None:
        nonlocal sequence, current
        current = packet
        active_rate = (
            degraded_service_rate_pps
            if degraded_service_rate_pps is not None and now >= backup_duration_s
            else service_rate_pps
        )
        finish = now + float(rng.exponential(1.0 / max(active_rate, 1e-9)))
        heapq.heappush(events, (finish, sequence, "departure", packet))
        sequence += 1

    def retry_or_fail(now: float, packet: tuple) -> None:
        nonlocal sequence, control_failed
        pid, control, attempt, origin = packet
        if attempt < max_retries:
            retry = (pid, control, attempt + 1, origin)
            retry_time = now + float(rng.exponential(backoff_mean_s))
            heapq.heappush(events, (retry_time, sequence, "arrival", retry))
            sequence += 1
        elif control:
            control_failed += 1

    drain_limit = duration_s + max(10.0, 4.0 * deadline_s)
    while events:
        now, _, kind, packet = heapq.heappop(events)
        if now > drain_limit:
            break
        if kind == "arrival":
            arrivals_attempted += 1
            in_system = len(waiting) + int(current is not None)
            if in_system >= buffer_packets:
                buffer_drops += 1
                retry_or_fail(now, packet)
            elif current is None:
                schedule_service(now, packet)
            else:
                waiting.append(packet)
        else:
            if current is None or current[0] != packet[0] or current[2] != packet[2]:
                raise RuntimeError("Discrete-event service state became inconsistent")
            current = None
            if rng.random() < erasure_probability:
                erased += 1
                retry_or_fail(now, packet)
            else:
                delivered += 1
                if packet[1]:
                    control_delivered += 1
                    control_latency.append(now - packet[3])
            if waiting:
                schedule_service(now, waiting.popleft())

    # Control packets not delivered or explicitly failed by the drain limit are failures.
    unresolved_control = max(control_generated - control_delivered - control_failed, 0)
    control_failed += unresolved_control
    latencies = np.asarray(control_latency, dtype=float)
    on_time = int(np.count_nonzero(latencies <= deadline_s))
    return {
        "generated_packets": generated,
        "arrival_attempts_including_retries": arrivals_attempted,
        "delivered_packets": delivered,
        "buffer_drops": buffer_drops,
        "erasures": erased,
        "control_generated": control_generated,
        "control_delivered": control_delivered,
        "control_failed": control_failed,
        "control_delivery_fraction": control_delivered / max(control_generated, 1),
        "control_on_time_fraction": on_time / max(control_generated, 1),
        "control_latency_mean_ms": float(latencies.mean() * 1000) if latencies.size else float("nan"),
        "control_latency_p95_ms": float(np.quantile(latencies, 0.95) * 1000) if latencies.size else float("nan"),
        "control_latency_p99_ms": float(np.quantile(latencies, 0.99) * 1000) if latencies.size else float("nan"),
        "attempt_drop_fraction": buffer_drops / max(arrivals_attempted, 1),
    }


def analytic_validation(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    arrival, service = 10.0, 20.0
    result = simulate_link(
        rng,
        arrival,
        service,
        duration_s=4000.0,
        buffer_packets=10000,
        control_fraction=1.0,
        erasure_probability=0.0,
        max_retries=0,
        backoff_mean_s=0.05,
        deadline_s=10.0,
    )
    theoretical_ms = 1000.0 / (service - arrival)
    relative_error = abs(result["control_latency_mean_ms"] - theoretical_ms) / theoretical_ms
    if relative_error > 0.08:
        raise RuntimeError(f"Packet simulator failed M/M/1 validation: relative error={relative_error:.3f}")
    return {"lambda_pps": arrival, "mu_pps": service, "theoretical_mean_ms": theoretical_ms, "relative_error": relative_error, **result}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "operational" / "packet_network")
    parser.add_argument("--hours", type=int, default=20)
    parser.add_argument("--traffic-multiplier", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--base-arrival-pps", type=float, default=8.0)
    parser.add_argument("--ev-session-arrival-pps", type=float, default=0.8)
    parser.add_argument("--base-service-pps", type=float, default=30.0)
    parser.add_argument("--buffer-packets", type=int, default=100)
    parser.add_argument("--control-fraction", type=float, default=0.05)
    parser.add_argument("--deadline-ms", type=float, default=500.0)
    parser.add_argument("--backup-s", nargs="+", type=float, default=[0.0, 60.0, 300.0])
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    validation = analytic_validation(args.seed)
    data = np.load(EV_DATA)
    timestamps = data["timestamp_local"]
    connected = data["connected_session_hours"].astype(float)
    arrivals = data["pickup"].astype(float)
    test = np.flatnonzero(timestamps >= np.datetime64("2023-01-01"))
    total_activity = connected.sum(axis=1) + arrivals.sum(axis=1)
    n_peak = args.hours // 2
    peak = test[np.argsort(total_activity[test])[-n_peak:]]
    rng = np.random.default_rng(args.seed)
    remaining = np.setdiff1d(test, peak)
    random = rng.choice(remaining, size=args.hours - n_peak, replace=False)
    chosen_hours = np.sort(np.concatenate([peak, random]))

    events = pd.read_csv(EVENTS)
    events = events.loc[events["threshold_customers"].eq(50)]
    peak_outages = events["peak_customers_out"].to_numpy(dtype=float)
    p99 = max(float(np.quantile(peak_outages, 0.99)), 1.0)
    severity_samples = np.clip(np.log1p(peak_outages) / np.log1p(p99), 0.0, 1.0)

    rows: list[dict] = []
    for hour_position, hour in enumerate(chosen_hours):
        power_threat = float(rng.choice(severity_samples)) if hour_position % 2 else 0.0
        comm_threat = 0.5 * power_threat
        for multiplier in args.traffic_multiplier:
            for backup_s in args.backup_s:
                for station in range(connected.shape[1]):
                    arrival_rate = multiplier * (
                        args.base_arrival_pps
                        + args.ev_session_arrival_pps * (connected[hour, station] + arrivals[hour, station])
                    )
                    nominal_service_rate = args.base_service_pps * (1.0 - 0.70 * comm_threat)
                    degraded_service_rate = nominal_service_rate * (1.0 - 0.75 * power_threat)
                    erasure = min(0.25, 0.002 + 0.08 * comm_threat)
                    result = simulate_link(
                        rng,
                        arrival_rate,
                        max(nominal_service_rate, 0.5),
                        args.duration_s,
                        args.buffer_packets,
                        args.control_fraction,
                        erasure,
                        max_retries=2,
                        backoff_mean_s=0.05,
                        deadline_s=args.deadline_ms / 1000.0,
                        backup_duration_s=backup_s,
                        degraded_service_rate_pps=max(degraded_service_rate, 0.5),
                    )
                    protected_fraction = min(max(backup_s / args.duration_s, 0.0), 1.0)
                    effective_service_rate = (
                        protected_fraction * nominal_service_rate
                        + (1.0 - protected_fraction) * degraded_service_rate
                    )
                    algebraic_support = min(1.0, effective_service_rate / max(arrival_rate, 1e-9))
                    rows.append(
                        {
                        "hour_index": int(hour),
                        "timestamp_local": str(timestamps[hour]),
                        "station_index": station,
                        "station_id": str(data["station_id"][station]),
                        "traffic_multiplier": multiplier,
                        "power_threat": power_threat,
                        "communication_threat": comm_threat,
                        "arrival_rate_pps": arrival_rate,
                        "backup_duration_s": backup_s,
                        "nominal_service_rate_pps": nominal_service_rate,
                        "degraded_service_rate_pps": degraded_service_rate,
                        "effective_service_rate_pps": effective_service_rate,
                        "utilization": arrival_rate / max(effective_service_rate, 1e-9),
                        "erasure_probability": erasure,
                        "algebraic_support": algebraic_support,
                        "effective_action_fraction": result["control_on_time_fraction"],
                            **result,
                        }
                    )
        print(f"hour {hour_position + 1}/{len(chosen_hours)} complete")

    scenarios_path = args.out / "packet_network_scenarios.csv"
    write_csv(scenarios_path, rows)
    frame = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in frame.groupby(["traffic_multiplier", "power_threat", "backup_duration_s"], sort=True):
        multiplier, threat, backup_s = keys
        for metric in (
            "control_on_time_fraction",
            "control_delivery_fraction",
            "control_latency_p95_ms",
            "attempt_drop_fraction",
            "algebraic_support",
        ):
            values = group[metric].dropna().to_numpy(dtype=float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(
                {
                    "traffic_multiplier": multiplier,
                    "power_threat": threat,
                    "backup_duration_s": backup_s,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": sd,
                    "ci95_low": float(values.mean() - 1.96 * sd / np.sqrt(len(values))),
                    "ci95_high": float(values.mean() + 1.96 * sd / np.sqrt(len(values))),
                    "n": len(values),
                }
            )
    summary_path = args.out / "packet_network_summary.csv"
    write_csv(summary_path, summary_rows)

    finite = frame[["algebraic_support", "control_on_time_fraction"]].dropna()
    proxy_correlation = float(finite.corr(method="spearman").iloc[0, 1])
    proxy_mae = float(np.mean(np.abs(finite["algebraic_support"] - finite["control_on_time_fraction"])))
    manifest = {
        "simulator": "finite-buffer discrete-event FCFS packet queue with erasures and retransmission backoff",
        "analytic_validation": validation,
        "packet_contract": {
            "duration_s": args.duration_s,
            "buffer_packets": args.buffer_packets,
            "control_fraction": args.control_fraction,
            "deadline_ms": args.deadline_ms,
            "max_retries": 2,
            "base_arrival_pps": args.base_arrival_pps,
            "ev_session_arrival_pps": args.ev_session_arrival_pps,
            "base_service_pps": args.base_service_pps,
            "traffic_multipliers": args.traffic_multiplier,
            "backup_duration_s": args.backup_s,
        },
        "backup_logic_check": (
            "For service starts before backup_duration_s, service rate excludes power derating; after depletion, "
            "the power-threat factor is applied. A backup longer than the simulated outage window therefore "
            "prevents direct power-induced service-rate loss for that window."
        ),
        "interpretation": (
            "Traffic rates are transparent stress assumptions linked to observed station activity, not captured "
            "utility packet traces. The effective action fraction is the share of control packets delivered "
            "within deadline and is used to derate executable rollout actions."
        ),
        "hourly_proxy_comparison": {"spearman": proxy_correlation, "mae": proxy_mae},
        "rows": len(rows),
        "outputs": {
            scenarios_path.name: {"bytes": scenarios_path.stat().st_size, "sha256": sha256(scenarios_path)},
            summary_path.name: {"bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
        },
    }
    manifest_path = args.out / "packet_network_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
