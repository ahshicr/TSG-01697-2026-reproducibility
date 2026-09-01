#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models import (
    GraphGRUForecaster,
    count_parameters,
    mae,
    masked_mape,
    normalized_laplacian_smoothness,
    rmse,
    set_seed,
)


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, indices: np.ndarray, history: int, horizon: int):
        self.features = features.astype(np.float32)
        self.indices = indices.astype(np.int64)
        self.history = history
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        t = int(self.indices[item])
        x = self.features[t - self.history : t]
        y = self.features[t : t + self.horizon]
        return x, y, t


def inverse_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x * std + mean).clamp_min(0.0)


def split_indices(
    total_hours: int,
    history: int,
    horizon: int,
    train_end: int | None = None,
    val_end: int | None = None,
):
    starts = np.arange(history, total_hours - horizon, dtype=np.int64)
    train_end = int(total_hours * 0.70) if train_end is None else int(train_end)
    val_end = int(total_hours * 0.85) if val_end is None else int(val_end)
    if not history < train_end < val_end < total_hours - horizon:
        raise ValueError(
            f"Invalid chronological split: history={history}, train_end={train_end}, "
            f"val_end={val_end}, total_hours={total_hours}, horizon={horizon}"
        )
    train = starts[starts < train_end]
    val = starts[(starts >= train_end) & (starts < val_end)]
    test = starts[starts >= val_end]
    return train, val, test


def evaluate(model, loader, device, mean, std):
    model.eval()
    pred_raw_all = []
    truth_raw_all = []
    indices = []
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            pred_raw_all.append(inverse_norm(pred, mean, std).cpu())
            truth_raw_all.append(inverse_norm(y, mean, std).cpu())
            indices.append(idx.cpu().numpy())
    pred_raw = torch.cat(pred_raw_all, dim=0)
    truth_raw = torch.cat(truth_raw_all, dim=0)
    return {
        "mae_demand": float(mae(pred_raw[..., 0], truth_raw[..., 0]).item()),
        "rmse_demand": float(rmse(pred_raw[..., 0], truth_raw[..., 0]).item()),
        "mape_demand": float(masked_mape(pred_raw[..., 0], truth_raw[..., 0]).item()),
        "mae_energy": float(mae(pred_raw[..., 1], truth_raw[..., 1]).item()),
        "rmse_energy": float(rmse(pred_raw[..., 1], truth_raw[..., 1]).item()),
        "mape_energy": float(masked_mape(pred_raw[..., 1], truth_raw[..., 1]).item()),
        "pred": pred_raw.numpy().astype(np.float32),
        "truth": truth_raw.numpy().astype(np.float32),
        "indices": np.concatenate(indices).astype(np.int64),
    }


def train_one(args, data, mode: str, seed: int, device: torch.device):
    set_seed(seed)
    raw = np.stack([data["pickup"], data["energy"]], axis=-1).astype(np.float32)
    raw_log = np.log1p(raw)
    train_end = int(data["split_train_end_index"]) if "split_train_end_index" in data.files else None
    val_end = int(data["split_val_end_index"]) if "split_val_end_index" in data.files else None
    train_idx, val_idx, test_idx = split_indices(
        raw.shape[0], args.history, args.horizon, train_end=train_end, val_end=val_end
    )
    normalization_end = int(raw.shape[0] * 0.70) if train_end is None else train_end
    train_slice = raw_log[:normalization_end]
    mean_np = train_slice.reshape(-1, raw.shape[-1]).mean(axis=0).astype(np.float32)
    std_np = (train_slice.reshape(-1, raw.shape[-1]).std(axis=0) + 1e-6).astype(np.float32)
    features = (raw_log - mean_np) / std_np

    train_loader = DataLoader(
        WindowDataset(features, train_idx, args.history, args.horizon),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        WindowDataset(features, val_idx, args.history, args.horizon),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        WindowDataset(features, test_idx, args.history, args.horizon),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = GraphGRUForecaster(
        n_features=2,
        hidden_dim=args.hidden,
        horizon=args.horizon,
        adj=data["adj"],
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    demand_total = np.maximum(data["pickup"].sum(axis=0), 1.0)
    alpha_np = data["energy"].sum(axis=0) / demand_total
    active = demand_total > np.percentile(demand_total, 10)
    cap = np.percentile(alpha_np[active], 95) if np.any(active) else np.percentile(alpha_np, 95)
    alpha_np = np.clip(alpha_np, 0.01, cap).astype(np.float32)
    alpha = torch.as_tensor(alpha_np, dtype=torch.float32, device=device)
    energy_scale = torch.as_tensor(np.maximum(data["energy"].std(), 1.0), dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    history_rows = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            fit_loss = torch.nn.functional.smooth_l1_loss(pred, y)
            loss = fit_loss
            physics_loss = pred.new_tensor(0.0)
            smooth_loss = pred.new_tensor(0.0)
            if mode == "physics":
                pred_raw = inverse_norm(pred, mean, std)
                demand = pred_raw[..., 0]
                energy = pred_raw[..., 1]
                expected_energy = demand * alpha.view(1, 1, -1)
                physics_loss = torch.mean(((energy - expected_energy) / energy_scale) ** 2)
                smooth_loss = normalized_laplacian_smoothness(pred_raw[..., :1], model.adj)
                loss = loss + args.physics_lambda * physics_loss + args.smooth_lambda * smooth_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate(model, val_loader, device, mean, std)
        val_score = val_metrics["mae_demand"] + 0.25 * val_metrics["mae_energy"]
        if val_score < best_val:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())
        history_rows.append(
            {
                "mode": mode,
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_score": val_score,
                "val_mae_demand": val_metrics["mae_demand"],
                "val_mae_energy": val_metrics["mae_energy"],
                "physics_loss": float(physics_loss.detach().cpu().item()),
                "smooth_loss": float(smooth_loss.detach().cpu().item()),
            }
        )
        if epoch % max(args.log_every, 1) == 0 or epoch == 1:
            elapsed = time.time() - started
            print(
                f"mode={mode} seed={seed} epoch={epoch:03d} "
                f"loss={np.mean(losses):.5f} val={val_score:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, mean, std)
    model_dir = args.out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean_np,
            "std": std_np,
            "args": vars(args),
            "mode": mode,
            "seed": seed,
            "parameters": count_parameters(model),
            "split_train_end_index": normalization_end,
            "split_val_end_index": int(raw.shape[0] * 0.85) if val_end is None else val_end,
        },
        model_dir / f"forecaster_{mode}_seed{seed}.pt",
    )
    np.savez_compressed(
        args.out / f"forecast_{mode}_seed{seed}.npz",
        indices=test_metrics.pop("indices"),
        pred=test_metrics.pop("pred"),
        truth=test_metrics.pop("truth"),
    )
    return history_rows, {
        "mode": mode,
        "seed": seed,
        "parameters": count_parameters(model),
        "best_val_score": best_val,
        "seconds": time.time() - started,
        **test_metrics,
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def jsonable_args(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/nyc_tlc_2023_hourly.npz"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--physics-lambda", type=float, default=0.08)
    parser.add_argument("--smooth-lambda", type=float, default=0.002)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--modes", nargs="+", default=["plain", "physics"])
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data, allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")
    all_history = []
    all_metrics = []
    for mode in args.modes:
        for seed in args.seeds:
            hist, metrics = train_one(args, data, mode, seed, device)
            all_history.extend(hist)
            all_metrics.append(metrics)
            write_csv(args.out / "training_history.csv", all_history)
            write_csv(args.out / "prediction_metrics.csv", all_metrics)
    (args.out / "training_manifest.json").write_text(
        json.dumps({"device": str(device), "torch": torch.__version__, "args": jsonable_args(args)}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
