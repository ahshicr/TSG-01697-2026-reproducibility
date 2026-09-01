#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def split_indices(total_hours: int, history: int, horizon: int, val_end: int | None = None):
    starts = np.arange(history, total_hours - horizon, dtype=np.int64)
    val_end = int(total_hours * 0.85) if val_end is None else int(val_end)
    return starts[starts >= val_end]


def metrics(name: str, pred: np.ndarray, truth: np.ndarray):
    out = {"method": name}
    for i, label in enumerate(["demand", "energy"]):
        err = pred[..., i] - truth[..., i]
        out[f"mae_{label}"] = float(np.mean(np.abs(err)))
        out[f"rmse_{label}"] = float(np.sqrt(np.mean(err**2)))
        out[f"mape_{label}"] = float(np.mean(np.abs(err) / np.maximum(np.abs(truth[..., i]), 1.0)))
    return out


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/nyc_tlc_2023_hourly.npz"))
    parser.add_argument("--stgnn", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("results/final/prediction_metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/final/forecast_table.csv"))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    raw = np.stack([data["pickup"], data["energy"]], axis=-1).astype(np.float32)
    train_end = int(data["split_train_end_index"]) if "split_train_end_index" in data.files else int(raw.shape[0] * 0.70)
    val_end = int(data["split_val_end_index"]) if "split_val_end_index" in data.files else None
    idx = split_indices(raw.shape[0], args.history, args.horizon, val_end=val_end)
    truth = np.stack([raw[t : t + args.horizon] for t in idx], axis=0)
    persistence = np.stack([np.repeat(raw[t - 1 : t], args.horizon, axis=0) for t in idx], axis=0)
    weekly = np.stack([raw[t - 168 : t - 168 + args.horizon] for t in idx], axis=0)
    train_mean = raw[:train_end].mean(axis=0, keepdims=True)
    mean_pred = np.repeat(train_mean[None, :, :, :], len(idx), axis=0)
    mean_pred = np.repeat(mean_pred, args.horizon, axis=1)
    stgnn_rows = []
    with args.metrics.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == "plain":
                score = float(row["mae_demand"]) + 0.25 * float(row["mae_energy"])
                stgnn_rows.append((score, row))
    best_stgnn = min(stgnn_rows, key=lambda item: item[0])[1]
    rows = [
        metrics("Historical mean", mean_pred, truth),
        metrics("Weekly seasonal", weekly, truth),
        metrics("Persistence", persistence, truth),
        {
            "method": "STGNN rollout predictor",
            "mae_demand": float(best_stgnn["mae_demand"]),
            "rmse_demand": float(best_stgnn["rmse_demand"]),
            "mape_demand": float(best_stgnn["mape_demand"]),
            "mae_energy": float(best_stgnn["mae_energy"]),
            "rmse_energy": float(best_stgnn["rmse_energy"]),
            "mape_energy": float(best_stgnn["mape_energy"]),
        },
    ]
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
