#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    parser.add_argument("--stgnn", type=Path, default=None,
                        help="Optional archive, which must match validation-only selection.")
    parser.add_argument("--metrics", type=Path, default=Path("results/final/prediction_metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/final/forecast_table.csv"))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    raw = np.stack([data["pickup"], data["energy"]], axis=-1).astype(np.float32)
    train_end = int(data["split_train_end_index"]) if "split_train_end_index" in data.files else int(raw.shape[0] * 0.70)
    val_end = int(data["split_val_end_index"]) if "split_val_end_index" in data.files else None
    with args.metrics.open("r", newline="", encoding="utf-8") as handle:
        model_rows = [row for row in csv.DictReader(handle) if row["mode"] == "plain"]
    if not model_rows or any("best_val_score" not in row for row in model_rows):
        raise ValueError("Validation scores are required for forecast selection")
    selected = min(model_rows, key=lambda row: float(row["best_val_score"]))
    selected_path = args.metrics.parent / f"forecast_plain_seed{selected['seed']}.npz"
    if args.stgnn is not None and args.stgnn.resolve() != selected_path.resolve():
        raise ValueError("Supplied forecast differs from the validation-selected model")
    archive = np.load(selected_path)
    idx = archive["indices"].astype(np.int64)
    expected = split_indices(raw.shape[0], args.history, args.horizon, val_end=val_end)
    if not np.array_equal(idx, expected):
        raise ValueError("Forecast archive does not cover the declared common test windows")
    truth = np.stack([raw[t : t + args.horizon] for t in idx], axis=0)
    persistence = np.stack([np.repeat(raw[t - 1 : t], args.horizon, axis=0) for t in idx], axis=0)
    weekly = np.stack([raw[t - 168 : t - 168 + args.horizon] for t in idx], axis=0)
    train_mean = raw[:train_end].mean(axis=0, keepdims=True)
    mean_pred = np.repeat(train_mean[None, :, :, :], len(idx), axis=0)
    mean_pred = np.repeat(mean_pred, args.horizon, axis=1)
    if archive["pred"].shape != truth.shape:
        raise ValueError("Forecast and baseline dimensions differ")
    if not np.allclose(archive["truth"], truth, atol=1e-4, rtol=1e-5):
        raise ValueError("Forecast targets do not match the original test observations")
    rows = [
        metrics("Historical mean", mean_pred, truth),
        metrics("Weekly seasonal", weekly, truth),
        metrics("Persistence", persistence, truth),
        metrics("Graph recurrent forecaster", archive["pred"], truth),
    ]
    write_csv(args.out, rows)
    manifest = {
        "selection": "minimum validation score, not test error",
        "selected_model": selected_path.name,
        "selected_validation_score": float(selected["best_val_score"]),
        "common_test_windows": len(idx),
        "history": args.history, "horizon": args.horizon,
        "training_end_exclusive": train_end,
        "first_test_hour": int(idx.min()), "last_test_hour": int(idx.max()),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), args.data, args.metrics, selected_path]},
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
