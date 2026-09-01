#!/usr/bin/env python3
"""Aggregate NYC TLC trips into hourly zone-level tensors.

The raw parquet files stay outside the HPC project directory.  The generated
NPZ file is compact enough to synchronize to the cluster and contains all
quantities required by the forecaster and the coupled-infrastructure twin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_ZONES = 263
EV_SHARE = 0.18
KWH_PER_MILE = 0.24


def sha256(path: Path, block: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def topk_adjacency(od: np.ndarray, k: int = 8) -> np.ndarray:
    mat = od.astype(np.float64)
    mat = mat + mat.T
    np.fill_diagonal(mat, 0.0)
    keep = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        if mat[i].sum() <= 0:
            continue
        idx = np.argpartition(mat[i], -k)[-k:]
        keep[i, idx] = mat[i, idx]
    keep = np.maximum(keep, keep.T)
    keep += np.eye(mat.shape[0])
    deg = keep.sum(axis=1)
    deg[deg == 0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(deg)
    return (keep * inv_sqrt[:, None] * inv_sqrt[None, :]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    start = pd.Timestamp(f"{args.year}-01-01 00:00:00")
    end = pd.Timestamp(f"{args.year + 1}-01-01 00:00:00")
    n_hours = int((end - start) / pd.Timedelta(hours=1))

    pickup = np.zeros((n_hours, N_ZONES), dtype=np.float32)
    dropoff = np.zeros_like(pickup)
    energy = np.zeros_like(pickup)
    pickup_energy = np.zeros_like(pickup)
    miles = np.zeros_like(pickup)
    od = np.zeros((N_ZONES, N_ZONES), dtype=np.float64)

    meta = {
        "dataset": "NYC TLC yellow taxi trip records",
        "year": args.year,
        "months": args.months,
        "n_zones": N_ZONES,
        "ev_share": EV_SHARE,
        "kwh_per_mile": KWH_PER_MILE,
        "raw_files": [],
        "rows_raw": 0,
        "rows_used": 0,
    }

    cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
    ]

    for month in args.months:
        path = args.raw_dir / f"yellow_tripdata_{args.year}-{month:02d}.parquet"
        print(f"reading {path}")
        df = pd.read_parquet(path, columns=cols)
        raw_rows = int(len(df))
        df = df.rename(
            columns={
                "tpep_pickup_datetime": "pickup_time",
                "tpep_dropoff_datetime": "dropoff_time",
                "PULocationID": "pu",
                "DOLocationID": "do",
                "trip_distance": "distance",
            }
        )
        mask = (
            (df["pickup_time"] >= start)
            & (df["pickup_time"] < end)
            & df["pu"].between(1, N_ZONES)
            & df["do"].between(1, N_ZONES)
            & (df["distance"] > 0.05)
            & (df["distance"] < 80.0)
        )
        df = df.loc[mask, ["pickup_time", "pu", "do", "distance"]].copy()
        used_rows = int(len(df))
        hour = ((df["pickup_time"] - start) / pd.Timedelta(hours=1)).astype(np.int64).to_numpy()
        pu = df["pu"].to_numpy(np.int64) - 1
        do = df["do"].to_numpy(np.int64) - 1
        dist = df["distance"].to_numpy(np.float32)
        e = dist * KWH_PER_MILE * EV_SHARE

        np.add.at(pickup, (hour, pu), 1.0)
        np.add.at(dropoff, (hour, do), 1.0)
        np.add.at(energy, (hour, do), e)
        np.add.at(pickup_energy, (hour, pu), e)
        np.add.at(miles, (hour, pu), dist)
        np.add.at(od, (pu, do), 1.0)

        meta["rows_raw"] += raw_rows
        meta["rows_used"] += used_rows
        meta["raw_files"].append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows_raw": raw_rows,
                "rows_used": used_rows,
            }
        )
        print(f"  rows used {used_rows:,}/{raw_rows:,}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    trips_by_zone = pickup.sum(axis=0)
    alpha = pickup_energy.sum(axis=0) / np.maximum(trips_by_zone, 1.0)
    alpha = np.clip(alpha, 0.05, np.percentile(alpha[trips_by_zone > 0], 95)).astype(np.float32)
    mean_hourly_energy = energy.mean(axis=0)
    base_power = (6.0 + 0.45 * np.sqrt(mean_hourly_energy + 1.0)).astype(np.float32)
    feeder_limit = (base_power + 1.8 * np.maximum(energy.mean(axis=0), 0.3) + 8.0).astype(np.float32)
    comm_capacity = (20.0 + 1.6 * np.sqrt(pickup.mean(axis=0) + dropoff.mean(axis=0) + 1.0)).astype(np.float32)
    adj = topk_adjacency(od)
    hours = pd.date_range(start=start, periods=n_hours, freq="h").astype(str).to_numpy()

    np.savez_compressed(
        out / f"nyc_tlc_{args.year}_hourly.npz",
        pickup=pickup,
        dropoff=dropoff,
        energy=energy,
        pickup_energy=pickup_energy,
        miles=miles,
        od=od.astype(np.float32),
        adj=adj,
        alpha=alpha,
        base_power=base_power,
        feeder_limit=feeder_limit,
        comm_capacity=comm_capacity,
        hours=hours,
        zone_ids=np.arange(1, N_ZONES + 1, dtype=np.int16),
    )
    meta["hours"] = n_hours
    meta["processed_file"] = f"nyc_tlc_{args.year}_hourly.npz"
    meta["processed_sha256"] = sha256(out / meta["processed_file"])
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ["rows_raw", "rows_used", "hours"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
