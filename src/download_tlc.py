#!/usr/bin/env python3
"""Download NYC TLC yellow taxi parquet files used by the experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        length = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if length:
                pct = 100 * done / length
                print(f"\r  {done / 1e6:8.1f}/{length / 1e6:8.1f} MB {pct:5.1f}%", end="")
                sys.stdout.flush()
    if length:
        print()
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--out", type=Path, default=Path("../datasets/nyc_tlc_2023"))
    args = parser.parse_args()

    for month in args.months:
        name = f"yellow_tripdata_{args.year}-{month:02d}.parquet"
        download(f"{BASE_URL}/{name}", args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
