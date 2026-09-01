"""Preprocess the City of Boulder EV charging-session dataset.

The source table contains transaction-level energy and duration, not a metered
hourly power trace.  This script therefore exports two distinct hourly signals:

1. exact transaction-start counts and transaction energy assigned to start hour;
2. a transparent load proxy that spreads each transaction's measured energy
   uniformly over its reported charging duration.

The second signal must not be described as measured hourly power.  Keeping the
two representations separate prevents a reconstructed profile from being
mistaken for a direct observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data" / "external" / "raw" / "boulder_ev_sessions.csv"
OUT_DEFAULT = ROOT / "data" / "external" / "processed" / "boulder_ev"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def duration_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def add_interval_energy(
    target: np.ndarray,
    hour0: pd.Timestamp,
    station_index: int,
    start: pd.Timestamp,
    duration_hours: float,
    energy_kwh: float,
) -> None:
    """Spread energy at constant mean power across intersected clock hours."""
    if not (duration_hours > 0.0 and energy_kwh > 0.0):
        return
    stop = start + pd.to_timedelta(duration_hours, unit="h")
    first = start.floor("h")
    last = (stop - pd.Timedelta(microseconds=1)).floor("h")
    mean_kw = energy_kwh / duration_hours
    for hour in pd.date_range(first, last, freq="h"):
        overlap_h = (min(stop, hour + pd.Timedelta(hours=1)) - max(start, hour)).total_seconds() / 3600.0
        if overlap_h <= 0:
            continue
        h = int((hour - hour0).total_seconds() // 3600)
        if 0 <= h < target.shape[0]:
            target[h, station_index] += np.float32(mean_kw * overlap_h)


def add_interval_occupancy(
    target: np.ndarray,
    hour0: pd.Timestamp,
    station_index: int,
    start: pd.Timestamp,
    duration_hours: float,
) -> None:
    """Add fractional connected-session hours to each intersected clock hour."""
    if not duration_hours > 0.0:
        return
    stop = start + pd.to_timedelta(duration_hours, unit="h")
    first = start.floor("h")
    last = (stop - pd.Timedelta(microseconds=1)).floor("h")
    for hour in pd.date_range(first, last, freq="h"):
        overlap_h = (min(stop, hour + pd.Timedelta(hours=1)) - max(start, hour)).total_seconds() / 3600.0
        if overlap_h <= 0:
            continue
        h = int((hour - hour0).total_seconds() // 3600)
        if 0 <= h < target.shape[0]:
            target[h, station_index] += np.float32(overlap_h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    raw = args.raw.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw, low_memory=False)
    expected = {
        "Station_Name",
        "Address",
        "Start_Date___Time",
        "End_Date___Time",
        "Total_Duration__hh_mm_ss_",
        "Charging_Time__hh_mm_ss_",
        "Energy__kWh_",
        "Port_Type",
        "ObjectId2",
    }
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    clean = pd.DataFrame(index=df.index)
    clean["source_record_id"] = pd.to_numeric(df["ObjectId2"], errors="coerce").astype("Int64")
    clean["station_name"] = df["Station_Name"].astype("string").str.strip()
    clean["address"] = df["Address"].astype("string").str.strip()
    clean["city"] = df["City"].astype("string").str.strip()
    clean["state_province"] = df["State_Province"].astype("string").str.strip()
    clean["postal_code"] = df["Zip_Postal_Code"].astype("string").str.strip()
    # Source timezone labels are retained but are not used to invent a UTC time:
    # the table mixes MDT/MST labels in ways that do not consistently follow DST.
    clean["start_time_zone_label"] = df["Start_Time_Zone"].astype("string").str.strip()
    clean["end_time_zone_label"] = df["End_Time_Zone"].astype("string").str.strip()
    clean["start_local"] = pd.to_datetime(df["Start_Date___Time"], format="mixed", errors="coerce")
    clean["end_local"] = pd.to_datetime(df["End_Date___Time"], format="mixed", errors="coerce")
    clean["total_duration_h"] = duration_seconds(df["Total_Duration__hh_mm_ss_"]) / 3600.0
    clean["charging_duration_h"] = duration_seconds(df["Charging_Time__hh_mm_ss_"]) / 3600.0
    clean["energy_kwh"] = pd.to_numeric(df["Energy__kWh_"], errors="coerce")
    clean["port_type"] = df["Port_Type"].astype("string").str.strip()
    clean["valid_arrival"] = clean["start_local"].notna() & clean["station_name"].notna()
    clean["valid_energy"] = (
        clean["valid_arrival"]
        & clean["energy_kwh"].gt(0)
        & clean["charging_duration_h"].gt(0)
        & np.isfinite(clean["energy_kwh"])
        & np.isfinite(clean["charging_duration_h"])
    )
    clean["valid_connection"] = clean["valid_arrival"] & clean["total_duration_h"].gt(0)
    clean["avg_charging_kw"] = np.where(
        clean["valid_energy"], clean["energy_kwh"] / clean["charging_duration_h"], np.nan
    )
    clean["end_missing"] = clean["end_local"].isna()
    clean["zero_energy"] = clean["energy_kwh"].eq(0)
    clean["duration_inconsistent"] = clean["charging_duration_h"] > clean["total_duration_h"] + (1.0 / 3600.0)

    station_names = sorted(clean.loc[clean["valid_arrival"], "station_name"].dropna().unique().tolist())
    station_map = {name: idx for idx, name in enumerate(station_names)}
    clean["station_index"] = clean["station_name"].map(station_map).astype("Int64")
    clean["station_id"] = clean["station_index"].map(lambda x: f"B{int(x):03d}" if pd.notna(x) else pd.NA).astype("string")

    start_min = clean.loc[clean["valid_arrival"], "start_local"].min().floor("h")
    charging_stop = clean.loc[clean["valid_energy"], "start_local"] + pd.to_timedelta(
        clean.loc[clean["valid_energy"], "charging_duration_h"], unit="h"
    )
    connection_stop = clean.loc[clean["valid_connection"], "start_local"] + pd.to_timedelta(
        clean.loc[clean["valid_connection"], "total_duration_h"], unit="h"
    )
    stop_max = max(charging_stop.max(), connection_stop.max()).ceil("h")
    timestamps = pd.date_range(start_min, stop_max, freq="h", inclusive="left")
    n_hours, n_stations = len(timestamps), len(station_names)

    arrivals = np.zeros((n_hours, n_stations), dtype=np.int32)
    start_energy = np.zeros((n_hours, n_stations), dtype=np.float32)
    load_proxy = np.zeros((n_hours, n_stations), dtype=np.float32)
    connected_fraction = np.zeros((n_hours, n_stations), dtype=np.float32)

    valid_arrivals = clean.loc[clean["valid_arrival"]]
    hour_index = ((valid_arrivals["start_local"].dt.floor("h") - start_min).dt.total_seconds() // 3600).astype(int)
    station_index = valid_arrivals["station_index"].astype(int).to_numpy()
    np.add.at(arrivals, (hour_index.to_numpy(), station_index), 1)

    valid_energy = clean.loc[clean["valid_energy"]]
    energy_hour_index = ((valid_energy["start_local"].dt.floor("h") - start_min).dt.total_seconds() // 3600).astype(int)
    energy_station_index = valid_energy["station_index"].astype(int).to_numpy()
    np.add.at(
        start_energy,
        (energy_hour_index.to_numpy(), energy_station_index),
        valid_energy["energy_kwh"].astype(np.float32).to_numpy(),
    )

    for row in valid_energy.itertuples(index=False):
        add_interval_energy(
            load_proxy,
            start_min,
            int(row.station_index),
            row.start_local,
            float(row.charging_duration_h),
            float(row.energy_kwh),
        )
    for row in clean.loc[clean["valid_connection"]].itertuples(index=False):
        add_interval_occupancy(
            connected_fraction,
            start_min,
            int(row.station_index),
            row.start_local,
            float(row.total_duration_h),
        )

    station_rows = []
    for name, idx in station_map.items():
        rows = clean.loc[clean["station_name"].eq(name)]
        station_rows.append(
            {
                "station_id": f"B{idx:03d}",
                "station_index": idx,
                "station_name": name,
                "address": rows["address"].mode(dropna=True).iloc[0],
                "city": rows["city"].mode(dropna=True).iloc[0],
                "state_province": rows["state_province"].mode(dropna=True).iloc[0],
                "postal_code": rows["postal_code"].mode(dropna=True).iloc[0],
                "session_count": int(len(rows)),
                "positive_energy_session_count": int(rows["valid_energy"].sum()),
                "measured_energy_kwh": float(rows.loc[rows["valid_energy"], "energy_kwh"].sum()),
            }
        )
    stations = pd.DataFrame(station_rows)

    clean_path = out / "boulder_ev_sessions_clean.csv.gz"
    station_path = out / "boulder_ev_stations.csv"
    hourly_path = out / "boulder_ev_hourly.npz"
    clean.to_csv(clean_path, index=False, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")
    stations.to_csv(station_path, index=False)
    np.savez_compressed(
        hourly_path,
        timestamp_local=timestamps.to_numpy(dtype="datetime64[m]"),
        station_id=stations["station_id"].to_numpy(dtype="U4"),
        arrivals=arrivals,
        transaction_start_energy_kwh=start_energy,
        reconstructed_load_kwh=load_proxy,
        connected_session_hours=connected_fraction,
    )

    split = {
        "strategy": "fixed chronological split; no random leakage across time",
        "train": {"start": "2018-01-01 00:00:00", "end_exclusive": "2022-01-01 00:00:00"},
        "validation": {"start": "2022-01-01 00:00:00", "end_exclusive": "2023-01-01 00:00:00"},
        "test": {"start": "2023-01-01 00:00:00", "end_exclusive": "2023-12-06 00:00:00"},
    }
    quality = {
        "source": {
            "title": "Electric Vehicle Charging Station Data",
            "publisher": "City of Boulder, Colorado",
            "license": "CC0 1.0",
            "raw_path": raw.relative_to(ROOT).as_posix(),
            "raw_bytes": raw.stat().st_size,
            "raw_sha256": sha256_file(raw),
        },
        "shape": {
            "raw_rows": int(len(df)),
            "stations": n_stations,
            "hourly_rows": n_hours,
            "start_local_min": str(clean["start_local"].min()),
            "start_local_max": str(clean["start_local"].max()),
            "hourly_end_exclusive": str(stop_max),
        },
        "quality_flags": {
            "missing_start": int(clean["start_local"].isna().sum()),
            "missing_end": int(clean["end_missing"].sum()),
            "duplicate_source_record_id": int(clean["source_record_id"].duplicated().sum()),
            "duplicate_full_rows": int(df.duplicated().sum()),
            "zero_energy_sessions": int(clean["zero_energy"].sum()),
            "negative_energy_sessions": int(clean["energy_kwh"].lt(0).sum()),
            "invalid_energy_sessions": int((~clean["valid_energy"]).sum()),
            "duration_inconsistent_sessions": int(clean["duration_inconsistent"].sum()),
            "avg_charging_power_above_19_2_kw": int(clean["avg_charging_kw"].gt(19.2).sum()),
        },
        "conservation_checks": {
            "measured_positive_energy_kwh": float(valid_energy["energy_kwh"].sum()),
            "transaction_start_energy_kwh": float(start_energy.sum(dtype=np.float64)),
            "reconstructed_load_energy_kwh": float(load_proxy.sum(dtype=np.float64)),
            "arrival_records": int(valid_arrivals.shape[0]),
            "hourly_arrivals": int(arrivals.sum(dtype=np.int64)),
        },
        "representations": {
            "arrivals": "Exact count of source transactions by local start hour and station.",
            "transaction_start_energy_kwh": "Measured session energy assigned to its local start hour.",
            "reconstructed_load_kwh": "Proxy only: measured session energy uniformly spread over reported charging duration.",
            "connected_session_hours": "Fractional connection-hours reconstructed from reported total duration.",
        },
        "timezone_note": (
            "Times are retained as source local civil times. Source MDT/MST labels are stored in the clean table "
            "but not converted to UTC because the labels are not consistently aligned with calendar DST."
        ),
        "split": split,
        "outputs": {},
    }
    for path in (clean_path, station_path, hourly_path):
        quality["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    quality_path = out / "boulder_ev_quality_report.json"
    quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = """# Processed City of Boulder EV sessions

This directory is generated by `src/preprocess_boulder_ev.py` from the official
City of Boulder transaction table. Raw source provenance and hashes are in
`../../metadata/` and `boulder_ev_quality_report.json`.

## Files

- `boulder_ev_sessions_clean.csv.gz`: cleaned transaction-level observations and QC flags.
- `boulder_ev_stations.csv`: stable station index, address, counts, and measured-energy totals.
- `boulder_ev_hourly.npz`: dense hourly arrays for 50 stations.
- `boulder_ev_quality_report.json`: source hash, exclusions, conservation checks, split contract, and output hashes.

## Critical interpretation rule

`arrivals` and `transaction_start_energy_kwh` are direct aggregations of reported
transactions. `reconstructed_load_kwh` is **not metered hourly power**: it is a
load proxy obtained by spreading measured session energy uniformly over reported
charging duration. Analyses using this proxy must include representation and
station-to-feeder mapping sensitivity tests.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    # Fail the build if basic conservation does not hold to float32 tolerance.
    measured = float(valid_energy["energy_kwh"].sum())
    reconstructed = float(load_proxy.sum(dtype=np.float64))
    if abs(reconstructed - measured) > max(1e-2, measured * 2e-6):
        raise RuntimeError(f"Energy reconstruction failed: measured={measured}, reconstructed={reconstructed}")
    if int(arrivals.sum()) != int(valid_arrivals.shape[0]):
        raise RuntimeError("Arrival aggregation failed conservation check")
    print(json.dumps(quality, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
