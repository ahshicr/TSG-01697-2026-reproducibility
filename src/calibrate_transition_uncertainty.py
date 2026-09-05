"""Estimate observable transition terms and construct honest uncertainty sets.

Only coefficients tied to Boulder EV/EAGLE-I observations are labelled
``estimated``. Packet-network coefficients are labelled ``constructed`` and
unobserved directions remain ``sensitivity-only``; none are relabelled as field
measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[1]
EV_DATA = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_forecast_dataset.npz"
OUTAGE_PANEL = (
    ROOT / "data" / "external" / "processed" / "eaglei_boulder" / "boulder_adjacent_counties_15min.csv.gz"
)
PACKETS = ROOT / "results" / "operational" / "packet_network" / "packet_network_scenarios.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            value.update(chunk)
    return value.hexdigest()


def bounded_fit(x: np.ndarray, y: np.ndarray, upper: float = 0.99) -> np.ndarray:
    return lsq_linear(x, y, bounds=(0.0, upper), lsmr_tol="auto").x


def block_bootstrap_coefficients(
    frame: pd.DataFrame,
    x_columns: list[str],
    y_column: str,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    days = frame["date"].drop_duplicates().to_numpy()
    x_all = frame[x_columns].to_numpy(float)
    y_all = frame[y_column].to_numpy(float)
    day_indices = {
        day: np.flatnonzero(frame["date"].to_numpy() == day) for day in days
    }
    values = []
    for _ in range(replicates):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        indices = np.concatenate([day_indices[day] for day in sampled_days])
        if len(indices) < len(x_columns) + 2:
            continue
        values.append(bounded_fit(x_all[indices], y_all[indices]))
    return np.asarray(values)


def add_parameter(
    rows: list[dict],
    name: str,
    central: float,
    low: float,
    high: float,
    evidence_class: str,
    source: str,
    interpretation: str,
) -> None:
    rows.append(
        {
            "parameter": name,
            "central": float(central),
            "lower": float(low),
            "upper": float(high),
            "evidence_class": evidence_class,
            "source": source,
            "interpretation": interpretation,
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "operational" / "transition_calibration")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--packets", type=Path, default=PACKETS)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data = np.load(EV_DATA)
    timestamps = pd.DatetimeIndex(data["timestamp_local"].astype("datetime64[ns]"))
    arrivals = data["pickup"].astype(float)
    equivalent_activity = arrivals + data["energy"].astype(float) / 7.2
    adjacency = data["adj"].astype(float)
    train_end = int(data["split_train_end_index"])
    val_end = int(data["split_val_end_index"])
    hour_of_week = timestamps.dayofweek.to_numpy() * 24 + timestamps.hour.to_numpy()

    # Station mobility-deficit state relative to a training-period seasonal median.
    seasonal = np.zeros((168, arrivals.shape[1]), dtype=float)
    for hour in range(168):
        rows_at_hour = np.flatnonzero(hour_of_week[:train_end] == hour)
        seasonal[hour] = np.mean(equivalent_activity[rows_at_hour], axis=0) if len(rows_at_hour) else 0.0
    baseline = seasonal[hour_of_week]
    station_deficit = np.where(
        baseline >= 0.01,
        np.clip((baseline - equivalent_activity) / np.maximum(baseline, 0.01), 0.0, 1.0),
        0.0,
    )
    spatial_lag = station_deficit @ adjacency.T
    station_x = np.column_stack(
        [station_deficit[: train_end - 1].ravel(), spatial_lag[: train_end - 1].ravel()]
    )
    station_y = station_deficit[1:train_end].ravel()
    active = np.maximum(station_x.max(axis=1), station_y) > 0.02
    within_fit = bounded_fit(station_x[active], station_y[active])

    # Align county outage observations (UTC) to the Boulder local civil-hour EV table.
    outage = pd.read_csv(OUTAGE_PANEL, dtype={"fips_code": "string"})
    outage = outage.loc[outage["fips_code"].str.zfill(5).eq("08013")].copy()
    outage["run_start_time_utc"] = pd.to_datetime(outage["run_start_time_utc"], utc=True)
    outage["timestamp_local"] = (
        outage["run_start_time_utc"].dt.tz_convert("America/Denver").dt.tz_localize(None).dt.floor("h")
    )
    outage_hourly = outage.groupby("timestamp_local")["customers_out"].max().sort_index()
    ev_city = pd.Series(arrivals.sum(axis=1), index=timestamps)
    # Leakage-free city baseline: prior eight observations at the same hour of week.
    city_frame = pd.DataFrame({"arrivals": ev_city, "hour_of_week": hour_of_week}, index=timestamps)
    city_frame["baseline"] = city_frame.groupby("hour_of_week")["arrivals"].transform(
        lambda series: series.shift(1).rolling(8, min_periods=4).median()
    )
    city_frame["road_threat"] = np.where(
        city_frame["baseline"] >= 1.0,
        np.clip((city_frame["baseline"] - city_frame["arrivals"]) / city_frame["baseline"], 0.0, 1.0),
        np.nan,
    )
    aligned = city_frame.join(outage_hourly.rename("customers_out"), how="inner").dropna()
    training_outages = aligned.loc[aligned.index < timestamps[train_end], "customers_out"]
    outage_scale = max(float(training_outages.quantile(0.99)), 1.0)
    aligned["power_threat"] = np.clip(
        np.log1p(aligned["customers_out"]) / np.log1p(outage_scale), 0.0, 1.0
    )
    aligned["road_next"] = aligned["road_threat"].shift(-1)
    aligned["power_next"] = aligned["power_threat"].shift(-1)
    aligned["next_time"] = aligned.index.to_series().shift(-1)
    aligned = aligned.loc[(aligned["next_time"] - aligned.index.to_series()).eq(pd.Timedelta(hours=1))].dropna()
    aligned["date"] = aligned.index.date
    aligned["intercept"] = 1.0
    train = aligned.loc[(aligned.index < timestamps[train_end])
                        & (aligned["next_time"] < timestamps[train_end])].copy()
    assert train["next_time"].max() < timestamps[train_end]
    test = aligned.loc[(aligned.index >= timestamps[val_end]) & (aligned.index < timestamps[-1])].copy()
    x_columns = ["road_threat", "power_threat", "intercept"]
    road_fit = bounded_fit(train[x_columns].to_numpy(float), train["road_next"].to_numpy(float))
    power_fit = bounded_fit(train[x_columns].to_numpy(float), train["power_next"].to_numpy(float))
    road_boot = block_bootstrap_coefficients(train, x_columns, "road_next", args.bootstrap, rng)
    power_boot = block_bootstrap_coefficients(train, x_columns, "power_next", args.bootstrap, rng)
    test_x = test[x_columns].to_numpy(float)
    test_diagnostics = {
        "aligned_train_transitions": int(len(train)),
        "aligned_test_transitions": int(len(test)),
        "road_rmse_fitted": float(np.sqrt(np.mean((test_x @ road_fit - test["road_next"].to_numpy()) ** 2))),
        "road_rmse_persistence": float(
            np.sqrt(np.mean((test["road_threat"].to_numpy() - test["road_next"].to_numpy()) ** 2))
        ),
        "power_rmse_fitted": float(np.sqrt(np.mean((test_x @ power_fit - test["power_next"].to_numpy()) ** 2))),
        "power_rmse_persistence": float(
            np.sqrt(np.mean((test["power_threat"].to_numpy() - test["power_next"].to_numpy()) ** 2))
        ),
    }

    # Packet simulator maps power loss and congestion to missed control actions.
    packets = pd.read_csv(args.packets)
    packets["unprotected_power"] = packets["power_threat"] * np.clip(
        1.0 - packets["backup_duration_s"] / 120.0, 0.0, 1.0
    )
    packets["overload"] = np.maximum(packets["utilization"] - 0.70, 0.0)
    packets["communication_loss"] = 1.0 - packets["control_on_time_fraction"]
    packet_x_cols = ["unprotected_power", "overload"]
    packet_fit = bounded_fit(packets[packet_x_cols].to_numpy(float), packets["communication_loss"].to_numpy(float))
    packet_prediction = np.clip(packets[packet_x_cols].to_numpy(float) @ packet_fit, 0.0, 1.0)
    packet_rmse = float(np.sqrt(np.mean((packet_prediction - packets["communication_loss"]) ** 2)))
    packet_boot = []
    packet_hours = packets["hour_index"].drop_duplicates().to_numpy()
    packet_x = packets[packet_x_cols].to_numpy(float)
    packet_y = packets["communication_loss"].to_numpy(float)
    packet_hour_values = packets["hour_index"].to_numpy()
    packet_hour_indices = {hour: np.flatnonzero(packet_hour_values == hour) for hour in packet_hours}
    for _ in range(args.bootstrap):
        sample_hours = rng.choice(packet_hours, size=len(packet_hours), replace=True)
        indices = np.concatenate([packet_hour_indices[hour] for hour in sample_hours])
        packet_boot.append(bounded_fit(packet_x[indices], packet_y[indices]))
    packet_boot = np.asarray(packet_boot)

    parameter_rows: list[dict] = []
    add_parameter(
        parameter_rows,
        "threat_persistence",
        within_fit[0],
        max(0.0, within_fit[0] * 0.75),
        min(0.99, within_fit[0] * 1.25),
        "estimated",
        "Boulder station-hour arrivals, 2018-2021",
        "Nonnegative station deficit autoregression; interval is a conservative +/-25% envelope.",
    )
    add_parameter(
        parameter_rows,
        "spatial_spread",
        within_fit[1],
        max(0.0, within_fit[1] * 0.50),
        min(0.99, within_fit[1] * 1.50),
        "estimated",
        "Boulder spatial graph and station-hour arrivals",
        "Neighbour-deficit coefficient; interval is a conservative +/-50% envelope.",
    )
    add_parameter(
        parameter_rows,
        "power_persistence",
        power_fit[1],
        float(np.quantile(power_boot[:, 1], 0.05)),
        float(np.quantile(power_boot[:, 1], 0.95)),
        "estimated",
        "EAGLE-I Boulder County hourly consecutive observations",
        "Predictive persistence of normalized county outage intensity, not component failure probability.",
    )
    add_parameter(
        parameter_rows,
        "pr_coupling",
        road_fit[1],
        float(np.quantile(road_boot[:, 1], 0.05)),
        float(np.quantile(road_boot[:, 1], 0.95)),
        "estimated-observational",
        "UTC-aligned EAGLE-I Boulder outages and local Boulder EV arrival deficits",
        "Predictive power-to-mobility association; not interpreted as a causal coefficient.",
    )
    add_parameter(
        parameter_rows,
        "pc_coupling",
        packet_fit[0],
        float(np.quantile(packet_boot[:, 0], 0.05)),
        float(np.quantile(packet_boot[:, 0], 0.95)),
        "constructed",
        "Packet-level stress simulator with backup-power sensitivity",
        "Unprotected power-threat contribution to missed-deadline control fraction.",
    )
    add_parameter(
        parameter_rows,
        "comm_persistence",
        0.50,
        0.30,
        0.80,
        "sensitivity-only",
        "finite-buffer communication-state envelope",
        "No longitudinal field packet trace identifies hourly communication persistence.",
    )
    for name, central, upper, meaning in (
        ("rc_coupling", 0.04, 0.08, "road-to-communication direction"),
        ("cr_coupling", 0.08, 0.16, "communication-to-road direction"),
        ("cp_coupling", 0.04, 0.08, "communication-to-power-service direction"),
    ):
        add_parameter(
            parameter_rows,
            name,
            central,
            0.0,
            upper,
            "sensitivity-only",
            "sign-constrained mechanism envelope",
            f"No joint field trace identifies the {meaning}; robust evaluation spans zero through the upper bound.",
        )

    parameter_path = args.out / "transition_parameter_uncertainty.csv"
    aligned_path = args.out / "boulder_aligned_hourly_transition_data.csv"
    diagnostic_path = args.out / "transition_diagnostics.json"
    write_csv(parameter_path, parameter_rows)
    aligned.reset_index(names="timestamp_local").to_csv(aligned_path, index=False)
    diagnostics = {
        "station_fit": {
            "samples": int(active.sum()),
            "self_persistence": float(within_fit[0]),
            "spatial_spread": float(within_fit[1]),
        },
        "county_ev_alignment": {
            "EAGLE_timezone": "UTC converted to America/Denver local civil hour before joining",
            "EV_timezone": "source local civil time; no UTC label invented",
            "outage_normalization_p99_customers": outage_scale,
            "coefficients_road_next_on_road_power_intercept": road_fit.tolist(),
            "coefficients_power_next_on_road_power_intercept": power_fit.tolist(),
            **test_diagnostics,
        },
        "packet_fit": {
            "rows": int(len(packets)),
            "coefficients_loss_on_unprotected_power_overload": packet_fit.tolist(),
            "rmse": packet_rmse,
        },
        "classification_rule": {
            "estimated": "fitted to measured public observations",
            "estimated-observational": "fitted association without a causal claim",
            "constructed": "fitted to the disclosed simulator rather than field packets",
            "sensitivity-only": "not identified; varied over a sign-constrained envelope",
        },
    }
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (parameter_path, aligned_path, diagnostic_path)
        },
        "bootstrap_day_blocks": args.bootstrap,
        "inputs": {
            EV_DATA.relative_to(ROOT).as_posix(): sha256(EV_DATA),
            OUTAGE_PANEL.relative_to(ROOT).as_posix(): sha256(OUTAGE_PANEL),
            args.packets.resolve().relative_to(ROOT).as_posix(): sha256(args.packets),
        },
    }
    manifest_path = args.out / "transition_calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"parameters": parameter_rows, "diagnostics": diagnostics, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
