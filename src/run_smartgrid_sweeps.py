#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys


def run_case(args, case):
    out_dir = args.results / case["case_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.forecast_results.glob("forecast_*.npz"):
        shutil.copy2(path, out_dir / path.name)
    shutil.copy2(args.forecast_results / "prediction_metrics.csv", out_dir / "prediction_metrics.csv")
    cmd = [
        sys.executable,
        "-u",
        "src/simulate_rollout.py",
        "--data",
        str(args.data),
        "--results",
        str(out_dir),
        "--scenarios",
        str(args.scenarios),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--energy-scale",
        str(case["energy_scale"]),
        "--feeder-margin-scale",
        str(case["feeder_margin_scale"]),
        "--charge-capacity-factor",
        str(case["charge_capacity_factor"]),
        "--restore-scale",
        str(case["restore_scale"]),
        "--pc-coupling",
        str(case["pc_coupling"]),
        "--rc-coupling",
        str(case["rc_coupling"]),
        "--cr-coupling",
        str(case["cr_coupling"]),
        "--cp-coupling",
        str(case["cp_coupling"]),
        "--power-derate",
        str(case["power_derate"]),
        "--restore-reference-multiplier",
        str(case["restore_reference_multiplier"]),
    ]
    print("running", case["case_id"], flush=True)
    with (out_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return out_dir


def case_rows():
    base = {
        "energy_scale": 1.0,
        "feeder_margin_scale": 1.0,
        "charge_capacity_factor": 1.22,
        "restore_scale": 1.0,
        "pc_coupling": 0.28,
        "rc_coupling": 0.08,
        "cr_coupling": 0.18,
        "cp_coupling": 0.08,
        "power_derate": 0.35,
        "restore_reference_multiplier": 1.8,
    }
    cases = []
    stress = [
        ("stress_normal", 1.0, 1.00, 1.22),
        ("stress_elevated", 1.35, 0.35, 1.35),
        ("stress_severe", 1.70, 0.20, 1.45),
    ]
    for case_id, energy_scale, feeder_margin_scale, charge_capacity_factor in stress:
        row = dict(base)
        row.update(
            {
                "family": "power_stress",
                "case_id": case_id,
                "label": case_id.replace("stress_", "").title(),
                "energy_scale": energy_scale,
                "feeder_margin_scale": feeder_margin_scale,
                "charge_capacity_factor": charge_capacity_factor,
            }
        )
        cases.append(row)

    coupling = [
        ("coupling_weak", 0.12, 0.04, 0.08, 0.04),
        ("coupling_base", 0.28, 0.08, 0.18, 0.08),
        ("coupling_strong", 0.44, 0.12, 0.28, 0.14),
    ]
    for case_id, pc, rc, cr, cp in coupling:
        row = dict(base)
        row.update(
            {
                "family": "coupling",
                "case_id": case_id,
                "label": case_id.replace("coupling_", "").title(),
                "pc_coupling": pc,
                "rc_coupling": rc,
                "cr_coupling": cr,
                "cp_coupling": cp,
            }
        )
        cases.append(row)

    restore = [
        ("restore_low", 0.65),
        ("restore_base", 1.00),
        ("restore_high", 1.35),
    ]
    for case_id, restore_scale in restore:
        row = dict(base)
        row.update(
            {
                "family": "restoration",
                "case_id": case_id,
                "label": case_id.replace("restore_", "").title(),
                "energy_scale": 1.35,
                "feeder_margin_scale": 0.35,
                "charge_capacity_factor": 1.35,
                "restore_scale": restore_scale,
            }
        )
        cases.append(row)
    return cases


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/nyc_tlc_2023_hourly.npz"))
    parser.add_argument("--results", type=Path, default=Path("results/smartgrid"))
    parser.add_argument("--forecast-results", type=Path, default=Path("results/final"))
    parser.add_argument("--scenarios", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = case_rows()
    write_manifest(args.results / "sweep_manifest.csv", rows)
    for case in rows:
        run_case(args, case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
