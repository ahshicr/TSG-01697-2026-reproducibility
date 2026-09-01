"""Five-fold geographic station holdout for the Boulder EV dataset.

Held-out stations contribute neither training targets nor neighbour inputs.
The pooled ridge model uses calendar, coordinates, and activity at non-held-out
stations. This is a deliberately hard zero-shot spatial-transfer test, separate
from the main chronological GraphGRU experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_forecast_dataset.npz"
COORDS = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_station_coordinates.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            value.update(chunk)
    return value.hexdigest()


def calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    doy = timestamps.dayofyear.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.arange(len(timestamps)) / max(len(timestamps) - 1, 1),
        ]
    ).astype(np.float32)


def balanced_longitudinal_blocks(coordinates: np.ndarray, folds: int) -> np.ndarray:
    """Partition unique locations into contiguous, near-balanced west-east blocks."""
    frame = pd.DataFrame({"latitude": coordinates[:, 0], "longitude": coordinates[:, 1]})
    groups = (
        frame.groupby(["latitude", "longitude"], sort=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["longitude", "latitude"])
        .reset_index(drop=True)
    )
    counts = groups["count"].to_numpy(dtype=int)
    prefix = np.concatenate([[0], np.cumsum(counts)])
    target = len(coordinates) / folds
    n_groups = len(groups)
    dp = np.full((folds + 1, n_groups + 1), np.inf)
    previous = np.full((folds + 1, n_groups + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, folds + 1):
        for end in range(k, n_groups + 1):
            for start in range(k - 1, end):
                cost = dp[k - 1, start] + (prefix[end] - prefix[start] - target) ** 2
                if cost < dp[k, end]:
                    dp[k, end], previous[k, end] = cost, start
    boundaries = []
    end = n_groups
    for k in range(folds, 0, -1):
        start = int(previous[k, end])
        boundaries.append((start, end))
        end = start
    group_fold = np.empty(n_groups, dtype=int)
    for fold, (start, end) in enumerate(reversed(boundaries)):
        group_fold[start:end] = fold
    lookup = {
        (float(row.latitude), float(row.longitude)): int(group_fold[i])
        for i, row in groups.iterrows()
    }
    return np.asarray([lookup[(float(lat), float(lon))] for lat, lon in coordinates], dtype=int)


def fold_context(raw: np.ndarray, adjacency: np.ndarray, training_stations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = adjacency.astype(float).copy()
    weights[:, ~training_stations] = 0.0
    np.fill_diagonal(weights, 0.0)
    row_sum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sum, out=np.zeros_like(weights), where=row_sum > 1e-12)
    neighbour = np.einsum("tnf,sn->tsf", raw, weights, optimize=True).astype(np.float32)
    global_context = raw[:, training_stations].mean(axis=1).astype(np.float32)
    return neighbour, global_context


def make_features(
    times: np.ndarray,
    stations: np.ndarray,
    calendar: np.ndarray,
    coordinates: np.ndarray,
    neighbour: np.ndarray,
    global_context: np.ndarray,
    raw: np.ndarray | None = None,
    station_calibration: np.ndarray | None = None,
) -> np.ndarray:
    pieces = [calendar[times], coordinates[stations]]
    for lag in (1, 24, 168):
        pieces.append(neighbour[times - lag, stations])
        pieces.append(global_context[times - lag])
        if raw is not None:
            pieces.append(raw[times - lag, stations])
    if station_calibration is not None:
        pieces.append(station_calibration[stations])
    return np.column_stack(pieces).astype(np.float32)


def targets(raw: np.ndarray, times: np.ndarray, stations: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([raw[times + h, stations] for h in range(horizon)], axis=1).astype(np.float32)


def sample_pairs(
    rng: np.random.Generator,
    time_start: int,
    time_end: int,
    stations: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = max(time_end - time_start, 0) * len(stations)
    if total <= maximum:
        time = np.repeat(np.arange(time_start, time_end), len(stations))
        station = np.tile(stations, time_end - time_start)
        return time.astype(int), station.astype(int)
    return (
        rng.integers(time_start, time_end, size=maximum, dtype=np.int64),
        rng.choice(stations, size=maximum, replace=True).astype(np.int64),
    )


def metric_rows(fold: int, method: str, pred: np.ndarray, truth: np.ndarray) -> list[dict]:
    rows = []
    for horizon in range(pred.shape[1]):
        for feature, label in enumerate(("arrivals", "energy_kwh")):
            error = pred[:, horizon, feature] - truth[:, horizon, feature]
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "horizon_h": horizon + 1,
                    "target": label,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mape_floor1": float(np.mean(np.abs(error) / np.maximum(np.abs(truth[:, horizon, feature]), 1.0))),
                    "n": int(len(error)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "real_ev_spatial_holdout")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-train-pairs", type=int, default=250_000)
    parser.add_argument("--max-validation-pairs", type=int, default=100_000)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--include-target-history",
        action="store_true",
        help="Use a held-out station's observed lags and pre-2022 calibration at inference, while keeping it out of fitting.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA)
    raw = np.stack([data["pickup"], data["energy"]], axis=-1).astype(np.float32)
    timestamps = pd.DatetimeIndex(data["timestamp_local"].astype("datetime64[ns]"))
    station_ids = data["station_id"].astype(str)
    coords_frame = pd.read_csv(COORDS).set_index("station_id").loc[station_ids]
    coordinates = coords_frame[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    coordinate_scaler = StandardScaler().fit(coordinates)
    coordinates_scaled = coordinate_scaler.transform(coordinates).astype(np.float32)
    adjacency = data["adj"].astype(np.float32)
    train_end = int(data["split_train_end_index"])
    val_end = int(data["split_val_end_index"])
    calendar = calendar_features(timestamps)
    station_calibration = np.column_stack(
        [
            np.log1p(raw[:train_end].mean(axis=0)),
            np.log1p(np.quantile(raw[:train_end], 0.95, axis=0)),
        ]
    ).reshape(len(station_ids), -1).astype(np.float32)
    cluster = balanced_longitudinal_blocks(coordinates, args.folds)
    rng = np.random.default_rng(args.seed)

    metrics = []
    prediction_blocks = []
    split_rows = []
    model_files = []
    for fold in range(args.folds):
        heldout = cluster == fold
        training = ~heldout
        heldout_indices = np.flatnonzero(heldout)
        training_indices = np.flatnonzero(training)
        neighbour, global_context = fold_context(raw, adjacency, training)
        train_t, train_s = sample_pairs(
            rng, 168, train_end - args.horizon, training_indices, args.max_train_pairs
        )
        val_t, val_s = sample_pairs(
            rng, train_end, val_end - args.horizon, training_indices, args.max_validation_pairs
        )
        test_t, test_s = sample_pairs(
            rng,
            val_end,
            len(timestamps) - args.horizon,
            heldout_indices,
            maximum=(len(timestamps) - args.horizon - val_end) * len(heldout_indices),
        )
        target_raw = raw if args.include_target_history else None
        calibration = station_calibration if args.include_target_history else None
        x_train = make_features(
            train_t, train_s, calendar, coordinates_scaled, neighbour, global_context, target_raw, calibration
        )
        x_val = make_features(
            val_t, val_s, calendar, coordinates_scaled, neighbour, global_context, target_raw, calibration
        )
        x_test = make_features(
            test_t, test_s, calendar, coordinates_scaled, neighbour, global_context, target_raw, calibration
        )
        y_train = targets(raw, train_t, train_s, args.horizon)
        y_val = targets(raw, val_t, val_s, args.horizon)
        y_test = targets(raw, test_t, test_s, args.horizon)
        target_scale = np.maximum(y_train.reshape(-1, 2).std(axis=0), 1e-6)
        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)
        y_train_scaled = y_train / target_scale[None, None, :]

        candidates = []
        for alpha in args.alphas:
            model = Ridge(alpha=alpha)
            model.fit(x_train_scaled, y_train_scaled.reshape(len(y_train_scaled), -1))
            val_pred = np.maximum(
                model.predict(x_val_scaled).reshape(-1, args.horizon, 2) * target_scale[None, None, :],
                0.0,
            )
            score = float(np.mean(np.abs(val_pred[..., 0] - y_val[..., 0])))
            score += 0.25 * float(np.mean(np.abs(val_pred[..., 1] - y_val[..., 1])))
            candidates.append((score, alpha, model))
        validation_score, selected_alpha, model = min(candidates, key=lambda item: item[0])
        pred = np.maximum(
            model.predict(x_test_scaled).reshape(-1, args.horizon, 2) * target_scale[None, None, :], 0.0
        ).astype(np.float32)
        weekly = targets(raw, test_t - 168, test_s, args.horizon)
        # Calendar-only pooled seasonal baseline from training stations.
        hour_of_week = timestamps.dayofweek.to_numpy() * 24 + timestamps.hour.to_numpy()
        seasonal_table = np.zeros((168, 2), dtype=float)
        for h in range(168):
            selection = np.flatnonzero((hour_of_week[:train_end] == h))
            seasonal_table[h] = raw[selection][:, training].mean(axis=(0, 1)) if len(selection) else 0.0
        seasonal = np.stack(
            [seasonal_table[hour_of_week[test_t + h]][:, None, :].repeat(1, axis=1) for h in range(args.horizon)],
            axis=1,
        )[:, :, 0, :].astype(np.float32)

        metrics.extend(metric_rows(fold, "spatial_pooled_ridge", pred, y_test))
        metrics.extend(metric_rows(fold, "weekly_station_history", weekly, y_test))
        metrics.extend(metric_rows(fold, "pooled_calendar_mean", seasonal, y_test))
        prediction_blocks.append(
            {
                "fold": np.full(len(test_t), fold, dtype=np.int16),
                "time_index": test_t.astype(np.int32),
                "station_index": test_s.astype(np.int16),
                "pred": pred,
                "truth": y_test,
                "weekly": weekly,
                "seasonal": seasonal,
            }
        )
        model_path = args.out / f"spatial_holdout_fold{fold}_model.npz"
        np.savez_compressed(
            model_path,
            coef=model.coef_.astype(np.float32),
            intercept=model.intercept_.astype(np.float32),
            feature_mean=scaler.mean_.astype(np.float32),
            feature_scale=scaler.scale_.astype(np.float32),
            target_scale=target_scale.astype(np.float32),
            selected_alpha=np.float64(selected_alpha),
            heldout_station_index=heldout_indices.astype(np.int16),
        )
        model_files.append(model_path)
        for station in range(len(station_ids)):
            split_rows.append(
                {
                    "fold": fold,
                    "station_index": station,
                    "station_id": station_ids[station],
                    "role": "heldout" if heldout[station] else "training",
                    "latitude": coordinates[station, 0],
                    "longitude": coordinates[station, 1],
                }
            )
        print(
            f"fold {fold + 1}/{args.folds}: heldout={len(heldout_indices)} alpha={selected_alpha} "
            f"val_score={validation_score:.5f} test_pairs={len(test_t)}"
        )

    metrics_path = args.out / "spatial_holdout_metrics.csv"
    splits_path = args.out / "spatial_holdout_splits.csv"
    predictions_path = args.out / "spatial_holdout_predictions.npz"
    write_csv(metrics_path, metrics)
    write_csv(splits_path, split_rows)
    np.savez_compressed(
        predictions_path,
        **{
            key: np.concatenate([block[key] for block in prediction_blocks], axis=0)
            for key in prediction_blocks[0]
        },
    )
    manifest = {
        "design": (
            f"{args.folds} contiguous west-east coordinate blocks balanced by station count; each fold withholds "
            "one block from training targets and "
            "neighbour inputs, then evaluates all its stations during the 2023 temporal test period."
        ),
        "features": (
            "calendar, station coordinates, and lagged neighbour/global activity from non-held-out stations; "
            + (
                "a held-out station's 1 h, 24 h, and 168 h observed lags and pre-2022 calibration are available "
                "only as deployment-time inputs; no held-out target enters model fitting"
                if args.include_target_history
                else "no lag or label from a held-out station enters model fitting (strict zero-shot mode)"
            )
        ),
        "include_target_history": args.include_target_history,
        "folds": args.folds,
        "horizon": args.horizon,
        "train_end_exclusive": str(timestamps[train_end]),
        "validation_end_exclusive": str(timestamps[val_end]),
        "alphas": args.alphas,
        "outputs": {},
    }
    for path in [metrics_path, splits_path, predictions_path, *model_files]:
        manifest["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest_path = args.out / "spatial_holdout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
