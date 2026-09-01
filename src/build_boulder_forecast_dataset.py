"""Build the leakage-safe Boulder EV forecasting tensor and spatial graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_hourly.npz"
COORDS = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_station_coordinates.csv"
OUT = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_forecast_dataset.npz"
ROLLOUT_OUT = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_rollout_dataset.npz"
MANIFEST = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_forecast_dataset.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            value.update(chunk)
    return value.hexdigest()


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def spatial_graph(distance: np.ndarray, neighbours: int = 4) -> np.ndarray:
    n = distance.shape[0]
    nonzero = distance[distance > 1e-9]
    bandwidth = float(np.median(nonzero))
    weights = np.exp(-((distance / max(bandwidth, 1e-6)) ** 2))
    np.fill_diagonal(weights, 0.0)
    keep = np.zeros_like(weights, dtype=bool)
    for i in range(n):
        indices = np.argsort(distance[i])
        indices = indices[indices != i][:neighbours]
        keep[i, indices] = True
    weights = np.where(keep | keep.T, weights, 0.0)
    weights += np.eye(n)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def main() -> None:
    source = np.load(SOURCE)
    station_ids = source["station_id"].astype(str)
    coords = pd.read_csv(COORDS).set_index("station_id").loc[station_ids]
    if coords[["latitude", "longitude"]].isna().any().any():
        raise RuntimeError("Every forecast station requires a documented coordinate or fallback")
    distance = haversine_km(coords["latitude"].to_numpy(), coords["longitude"].to_numpy()).astype(np.float32)
    adjacency = spatial_graph(distance)
    timestamps = source["timestamp_local"].astype("datetime64[m]")
    train_end = int(np.searchsorted(timestamps, np.datetime64("2022-01-01T00:00")))
    val_end = int(np.searchsorted(timestamps, np.datetime64("2023-01-01T00:00")))
    np.savez_compressed(
        OUT,
        timestamp_local=timestamps,
        station_id=station_ids.astype("U4"),
        pickup=source["arrivals"].astype(np.float32),
        energy=source["reconstructed_load_kwh"].astype(np.float32),
        transaction_start_energy_kwh=source["transaction_start_energy_kwh"].astype(np.float32),
        connected_session_hours=source["connected_session_hours"].astype(np.float32),
        adj=adjacency,
        distance_km=distance,
        split_train_end_index=np.int64(train_end),
        split_val_end_index=np.int64(val_end),
    )
    train_energy = source["reconstructed_load_kwh"][:train_end].astype(float)
    mean_energy = train_energy.mean(axis=0)
    share = mean_energy + 0.10 * max(float(mean_energy.mean()), 1e-6)
    share /= share.sum()
    base_power_mw = (6.0 * share).astype(np.float32)
    feeder_headroom_mw = np.maximum(np.quantile(train_energy, 0.995, axis=0) * 1.5 / 1000.0, 0.002)
    feeder_limit_mw = (base_power_mw + feeder_headroom_mw).astype(np.float32)
    comm_capacity_proxy = (
        2.0 + 0.5 * source["arrivals"][:train_end].mean(axis=0) + 0.1 * source["connected_session_hours"][:train_end].mean(axis=0)
    ).astype(np.float32)
    np.savez_compressed(
        ROLLOUT_OUT,
        timestamp_local=timestamps,
        station_id=station_ids.astype("U4"),
        pickup=source["arrivals"].astype(np.float32),
        energy=source["reconstructed_load_kwh"].astype(np.float32),
        adj=adjacency,
        base_power=base_power_mw,
        feeder_limit=feeder_limit_mw,
        comm_capacity=comm_capacity_proxy,
        split_train_end_index=np.int64(train_end),
        split_val_end_index=np.int64(val_end),
    )
    report = {
        "purpose": "Independent real-EV forecasting and rollout validation dataset",
        "features": {
            "pickup": "Exact transaction-start count per station-hour.",
            "energy": "Reconstructed load proxy in kWh per hour; not metered hourly power.",
            "transaction_start_energy_kwh": "Measured session energy assigned to start hour.",
        },
        "rollout_service_proxies": {
            "status": "constructed; not measured utility capacities",
            "base_power": "6 MW SMART-DS feeder base allocated across stations by smoothed training energy share",
            "feeder_limit": "base power plus 1.5 x training 99.5th-percentile station EV load headroom",
            "comm_capacity": "dimensionless hourly service proxy; packet validation is reported separately",
            "rule": "No electrical or packet conclusion may rely on these proxies; use OpenDSS and packet outputs.",
        },
        "graph": {
            "type": "symmetric four-nearest-address graph with Gaussian distance weights and row normalization",
            "coordinate_precision": (
                "49 station records use Census address-range coordinates; one uses an explicitly marked "
                "OpenStreetMap road-level fallback. Multiple ports at one address may share coordinates."
            ),
            "row_sum_max_abs_error": float(np.max(np.abs(adjacency.sum(axis=1) - 1.0))),
            "nonzero_directed_entries": int(np.count_nonzero(adjacency)),
            "distance_km_max": float(distance.max()),
        },
        "split": {
            "train": {"start": str(timestamps[0]), "end_exclusive": "2022-01-01T00:00"},
            "validation": {"start": "2022-01-01T00:00", "end_exclusive": "2023-01-01T00:00"},
            "test": {"start": "2023-01-01T00:00", "end_exclusive": str(timestamps[-1] + np.timedelta64(1, "h"))},
            "train_end_index": train_end,
            "validation_end_index": val_end,
        },
        "shape": {"hours": int(len(timestamps)), "stations": int(len(station_ids)), "features": 2},
        "outputs": {
            OUT.name: {"path": OUT.relative_to(ROOT).as_posix(), "bytes": OUT.stat().st_size, "sha256": sha256(OUT)},
            ROLLOUT_OUT.name: {
                "path": ROLLOUT_OUT.relative_to(ROOT).as_posix(),
                "bytes": ROLLOUT_OUT.stat().st_size,
                "sha256": sha256(ROLLOUT_OUT),
            },
        },
    }
    MANIFEST.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
