"""Check paired behavior after refactoring and chronological normalization."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

import simulate_rollout_revised as current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("previous_simulation", args.previous)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    config = current.build_parser().parse_args([])
    payload = current.payload_from_args(config)
    for task in [(0, "nominal"), (3, "ood")]:
        current.init_worker(payload)
        actual = current.simulate_one(task)
        previous.init_worker(payload)
        expected = previous.simulate_one(task)
        for new, old in zip(actual, expected):
            assert new.keys() == old.keys()
            for key in new:
                if key == "latency_ms":
                    continue
                a, b = new[key], old[key]
                if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
                    continue
                assert a == b, (task, new["policy"], key, a, b)
    data = np.load(config.data)
    end = int(data["split_train_end_index"])
    np.testing.assert_array_equal(payload["mean_energy"], data["energy"][:end].mean(axis=0))
    np.testing.assert_array_equal(payload["mean_demand"], data["pickup"][:end].mean(axis=0))
    print("PASS: 12 policy trajectories match the previous implementation under identical inputs.")
    print(f"PASS: normalization and charging budget use only {end} training hours.")
    print("Old/new aggregate mean energy:", float(data["energy"].mean(axis=0).sum()), float(payload["mean_energy"].sum()))


if __name__ == "__main__":
    main()
