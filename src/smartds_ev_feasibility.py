"""Independent OpenDSS validation with real Boulder EV load profiles.

The Boulder stations and the SMART-DS feeder are not geographically co-located.
Accordingly, this experiment never presents a station-to-bus assignment as an
observation. It repeats the complete AC power-flow calculation over randomized
address-group embeddings and reports the distribution across mappings.

Candidate charging service is projected into the AC-feasible set by bisection.
This makes network constraints action-dependent rather than a post-hoc label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER = (
    ROOT
    / "data"
    / "external"
    / "raw"
    / "smartds_v1.0"
    / "peak"
    / "SFO"
    / "P1U"
    / "scenarios"
    / "base_peak"
    / "opendss_no_loadshapes"
    / "p1uhs0_1247"
    / "p1uhs0_1247--p1udt104"
    / "Master.dss"
)
EV_DATA = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_forecast_dataset.npz"
COORDS = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_station_coordinates.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            value.update(chunk)
    return value.hexdigest()


def compile_feeder(master: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    dss.Text.Command(f'Redirect "{master.resolve()}"')
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        raise RuntimeError("SMART-DS base case did not converge")
    names = list(dss.Loads.AllNames())
    kw, kvar, phases = [], [], []
    for name in names:
        dss.Loads.Name(name)
        kw.append(dss.Loads.kW())
        kvar.append(dss.Loads.kvar())
        phases.append(dss.Loads.Phases())
    return names, np.asarray(kw), np.asarray(kvar), np.asarray(phases)


def apply_ev_load(
    load_names: list[str],
    base_kw: np.ndarray,
    base_kvar: np.ndarray,
    ev_by_load_kw: np.ndarray,
    alpha: float,
    power_factor: float,
) -> None:
    kvar_per_kw = math.tan(math.acos(power_factor))
    for i, name in enumerate(load_names):
        dss.Loads.Name(name)
        addition = alpha * float(ev_by_load_kw[i])
        dss.Loads.kW(float(base_kw[i] + addition))
        dss.Loads.kvar(float(base_kvar[i] + addition * kvar_per_kw))


def circuit_metrics() -> dict[str, float | int | bool | str]:
    dss.Solution.Solve()
    converged = bool(dss.Solution.Converged())
    voltage = np.asarray(dss.Circuit.AllBusMagPu(), dtype=float)
    active_voltage = voltage[voltage > 0.1]
    line_records = []
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        normal = float(dss.Lines.NormAmps())
        current = np.asarray(dss.CktElement.CurrentsMagAng(), dtype=float)[::2]
        ratio = float(current.max() / normal) if normal > 0 and current.size else float("nan")
        line_records.append((ratio, name))
    finite = [item for item in line_records if np.isfinite(item[0])]
    max_ratio, max_name = max(finite, default=(float("nan"), ""))
    losses = dss.Circuit.Losses()
    power = dss.Circuit.TotalPower()
    return {
        "converged": converged,
        "v_min_pu": float(active_voltage.min()) if active_voltage.size else float("nan"),
        "v_max_pu": float(active_voltage.max()) if active_voltage.size else float("nan"),
        "nodes_below_0_95": int(np.count_nonzero(active_voltage < 0.95)),
        "nodes_above_1_05": int(np.count_nonzero(active_voltage > 1.05)),
        "deenergized_nodes": int(np.count_nonzero(voltage <= 0.1)),
        "max_line_loading_pu": max_ratio,
        "limiting_line": max_name,
        "losses_kw": float(losses[0] / 1000.0),
        "source_kw": float(-power[0]),
    }


def feasible(metrics: dict, voltage_low: float, voltage_high: float, line_limit: float) -> bool:
    return bool(
        metrics["converged"]
        and metrics["v_min_pu"] >= voltage_low
        and metrics["v_max_pu"] <= voltage_high
        and metrics["max_line_loading_pu"] <= line_limit
    )


def solve_alpha(
    load_names: list[str],
    base_kw: np.ndarray,
    base_kvar: np.ndarray,
    ev_by_load_kw: np.ndarray,
    power_factor: float,
    voltage_low: float,
    voltage_high: float,
    line_limit: float,
    iterations: int = 14,
) -> tuple[float, dict, dict]:
    apply_ev_load(load_names, base_kw, base_kvar, ev_by_load_kw, 1.0, power_factor)
    unconstrained = circuit_metrics()
    if feasible(unconstrained, voltage_low, voltage_high, line_limit):
        return 1.0, unconstrained, unconstrained
    lower, upper = 0.0, 1.0
    best = None
    for _ in range(iterations):
        middle = (lower + upper) / 2.0
        apply_ev_load(load_names, base_kw, base_kvar, ev_by_load_kw, middle, power_factor)
        metrics = circuit_metrics()
        if feasible(metrics, voltage_low, voltage_high, line_limit):
            lower, best = middle, metrics
        else:
            upper = middle
    if best is None:
        apply_ev_load(load_names, base_kw, base_kvar, ev_by_load_kw, 0.0, power_factor)
        best = circuit_metrics()
    return lower, unconstrained, best


def mapping_for_seed(coords: pd.DataFrame, eligible_loads: np.ndarray, seed: int) -> np.ndarray:
    """Map each shared address to an electrically plausible three-phase load."""
    groups = coords.groupby(["latitude", "longitude"], sort=True).ngroup().to_numpy()
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    group_to_load = rng.choice(eligible_loads, size=len(unique_groups), replace=True)
    # Guarantee broad feeder coverage when there are enough address groups.
    shuffled = rng.permutation(eligible_loads)
    group_to_load[: min(len(unique_groups), len(eligible_loads))] = shuffled[
        : min(len(unique_groups), len(eligible_loads))
    ]
    return group_to_load[groups]


def selected_hours(timestamps: np.ndarray, total_ev: np.ndarray, count: int, seed: int) -> np.ndarray:
    test = np.flatnonzero((timestamps >= np.datetime64("2023-01-01")) & (total_ev > 0))
    n_peak = min(count // 2, len(test))
    peak = test[np.argsort(total_ev[test])[-n_peak:]]
    remaining = np.setdiff1d(test, peak, assume_unique=False)
    rng = np.random.default_rng(seed)
    random = rng.choice(remaining, size=min(count - n_peak, len(remaining)), replace=False)
    return np.sort(np.concatenate([peak, random]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "operational" / "smartds_ev")
    parser.add_argument("--mappings", type=int, default=20)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--all-test-hours", action="store_true")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--penetration", type=float, nargs="+", default=[1.0, 3.0, 5.0, 10.0])
    parser.add_argument("--power-factor", type=float, default=0.98)
    parser.add_argument("--voltage-low", type=float, default=0.95)
    parser.add_argument("--voltage-high", type=float, default=1.05)
    parser.add_argument("--line-limit", type=float, default=1.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = np.load(EV_DATA)
    timestamps = data["timestamp_local"]
    ev = data["energy"].astype(float)
    coords = pd.read_csv(COORDS).set_index("station_id").loc[data["station_id"].astype(str)].reset_index()
    load_names, base_kw, base_kvar, load_phases = compile_feeder(MASTER)
    eligible_loads = np.flatnonzero(load_phases == 3)
    if not len(eligible_loads):
        raise RuntimeError("No three-phase feeder loads are available for Level-2 station embedding")
    apply_ev_load(load_names, base_kw, base_kvar, np.zeros(len(load_names)), 0.0, args.power_factor)
    baseline = circuit_metrics()
    if not feasible(baseline, args.voltage_low, args.voltage_high, args.line_limit):
        raise RuntimeError(f"Base feeder violates the requested feasibility contract: {baseline}")
    hours = (
        np.flatnonzero((timestamps >= np.datetime64("2023-01-01")) & (ev.sum(axis=1) > 0))
        if args.all_test_hours
        else selected_hours(timestamps, ev.sum(axis=1), args.hours, args.seed)
    )
    selected_total_ev = ev[hours].sum(axis=1)
    peak_threshold = float(np.median(selected_total_ev)) if len(selected_total_ev) else float("inf")

    rows: list[dict] = []
    for mapping_seed in range(args.mappings):
        station_to_load = mapping_for_seed(coords, eligible_loads, args.seed + mapping_seed)
        for hour in hours:
            for penetration in args.penetration:
                ev_by_load = np.zeros(len(load_names), dtype=float)
                np.add.at(ev_by_load, station_to_load, ev[hour] * penetration)
                alpha, unconstrained, projected = solve_alpha(
                    load_names,
                    base_kw,
                    base_kvar,
                    ev_by_load,
                    args.power_factor,
                    args.voltage_low,
                    args.voltage_high,
                    args.line_limit,
                )
                requested = float(ev_by_load.sum())
                rows.append(
                    {
                        "mapping_seed": mapping_seed,
                        "hour_index": int(hour),
                        "timestamp_local": str(timestamps[hour]),
                        "sample_type": "upper_activity_half" if ev[hour].sum() >= peak_threshold else "lower_activity_half",
                        "penetration_multiplier": penetration,
                        "requested_ev_kw": requested,
                        "feasible_fraction": alpha,
                        "feasible_ev_kw": requested * alpha,
                        "unconstrained_feasible": feasible(
                            unconstrained, args.voltage_low, args.voltage_high, args.line_limit
                        ),
                        **{f"unconstrained_{key}": value for key, value in unconstrained.items()},
                        **{f"projected_{key}": value for key, value in projected.items()},
                    }
                )
        print(f"mapping {mapping_seed + 1}/{args.mappings} complete")

    scenario_path = args.out / "smartds_ev_scenarios.csv"
    write_csv(scenario_path, rows)
    frame = pd.DataFrame(rows)
    summary_rows = []
    for penetration, group in frame.groupby("penetration_multiplier"):
        mapping_means = group.groupby("mapping_seed").agg(
            feasible_fraction=("feasible_fraction", "mean"),
            unconstrained_feasible=("unconstrained_feasible", "mean"),
            v_min=("unconstrained_v_min_pu", "mean"),
            line_loading=("unconstrained_max_line_loading_pu", "mean"),
        )
        for metric in mapping_means:
            values = mapping_means[metric].to_numpy(dtype=float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(
                {
                    "penetration_multiplier": penetration,
                    "metric": metric,
                    "mapping_level_mean": float(values.mean()),
                    "mapping_level_sd": sd,
                    "ci95_low": float(values.mean() - 1.96 * sd / np.sqrt(len(values))),
                    "ci95_high": float(values.mean() + 1.96 * sd / np.sqrt(len(values))),
                    "mapping_replicates": int(len(values)),
                    "scenario_rows": int(len(group)),
                }
            )
    summary_path = args.out / "smartds_ev_summary.csv"
    write_csv(summary_path, summary_rows)
    manifest = {
        "feeder": {
            "master": MASTER.relative_to(ROOT).as_posix(),
            "sha256": sha256(MASTER),
            "buses": int(dss.Circuit.NumBuses()),
            "nodes": int(dss.Circuit.NumNodes()),
            "loads": len(load_names),
            "lines": len(dss.Lines.AllNames()),
            "transformers": len(dss.Transformers.AllNames()),
            "baseline": baseline,
        },
        "contract": {
            "voltage_pu": [args.voltage_low, args.voltage_high],
            "line_loading_pu_max": args.line_limit,
            "projection": "largest scalar fraction of candidate EV action satisfying full OpenDSS solve",
            "iterations": 14,
        },
        "embedding": (
            "Randomized address-group-to-three-phase-load embeddings; Boulder and SMART-DS are not geographically "
            "co-located. Shared station addresses stay on one feeder load within each mapping. Single-phase 120 V "
            "loads are excluded as implausible connection points for the aggregate public Level-2 station loads."
        ),
        "eligible_three_phase_loads": int(len(eligible_loads)),
        "sampling": {
            "hours": int(len(hours)),
            "all_nonzero_2023_hours": args.all_test_hours,
            "half_peak_half_random": True,
            "mappings": args.mappings,
            "penetration_multipliers": args.penetration,
            "scenario_rows": len(rows),
        },
        "outputs": {
            scenario_path.name: {"bytes": scenario_path.stat().st_size, "sha256": sha256(scenario_path)},
            summary_path.name: {"bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
        },
    }
    manifest_path = args.out / "smartds_ev_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
