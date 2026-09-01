#!/usr/bin/env python3
"""Verify all raw and processed data against the committed metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("../datasets/nyc_tlc_2023"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    failures = []
    for item in metadata["raw_files"]:
        path = args.raw_dir / item["file"]
        if not path.exists():
            failures.append(f"missing: {path}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != int(item["bytes"]):
            failures.append(f"size mismatch: {path} expected={item['bytes']} actual={actual_size}")
        if actual_hash != item["sha256"]:
            failures.append(f"SHA-256 mismatch: {path} expected={item['sha256']} actual={actual_hash}")
        print(f"OK raw {path.name} bytes={actual_size} sha256={actual_hash}")
    processed = args.processed_dir / metadata["processed_file"]
    if not processed.exists():
        failures.append(f"missing: {processed}")
    else:
        actual_hash = sha256(processed)
        if actual_hash != metadata["processed_sha256"]:
            failures.append(
                f"SHA-256 mismatch: {processed} expected={metadata['processed_sha256']} actual={actual_hash}"
            )
        print(f"OK processed {processed.name} bytes={processed.stat().st_size} sha256={actual_hash}")
    if failures:
        print("\n".join(failures))
        return 1
    print("All files match data/processed/metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

