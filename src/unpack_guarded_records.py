"""Restore losslessly compressed study records without replacing existing data."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ('guarded_validation_20260905', 'guarded_test_20260905',
           'guarded_external_20260905')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def restore(root=ROOT, check_only=False):
    root = Path(root).resolve()
    with (root / 'COMPRESSED_RECORDS.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Empty or repeated record paths')
    checked = []
    # Validate every existing target and compressed input before creating a file.
    for row in rows:
        target = (root / row['path']).resolve()
        archive = (root / row['archive']).resolve()
        if not target.is_relative_to(root) or not archive.is_relative_to(root):
            raise ValueError('Record outside package')
        parts = target.relative_to(root).parts
        if len(parts) < 4 or parts[0] != 'results' or parts[1] not in STUDIES:
            raise ValueError('Unexpected study path')
        if target.name not in ('rollout_scenarios.csv', 'candidate_scores.jsonl'):
            raise ValueError('Unexpected record name')
        if archive != target.with_suffix(target.suffix + '.gz'):
            raise ValueError('Archive and record path mismatch')
        if archive.stat().st_size != int(row['compressed_bytes']) or digest(archive) != row['compressed_sha256']:
            raise ValueError('Compressed content changed: ' + row['archive'])
        if target.exists() and (target.stat().st_size != int(row['bytes']) or digest(target) != row['sha256']):
            raise ValueError('Preserve conflicting existing record: ' + row['path'])
        checked.append((row, target, archive))
    restored = 0
    for row, target, archive in checked:
        if check_only or target.exists():
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as output:
                temporary = Path(output.name)
                value, size = hashlib.sha256(), 0
                with gzip.open(archive, 'rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        size += len(block)
                        if size > int(row['bytes']):
                            raise ValueError('Expanded record exceeds declared size')
                        value.update(block)
                        output.write(block)
            if size != int(row['bytes']) or value.hexdigest() != row['sha256']:
                raise ValueError('Restored content mismatch: ' + row['path'])
            # Exclusive creation also protects against a concurrent writer.
            with target.open('xb') as output, temporary.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
            restored += 1
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return dict(status='PASS', records=len(rows), restored=restored,
                uncompressed_bytes=sum(int(row['bytes']) for row in rows), check_only=check_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, default=ROOT)
    parser.add_argument('--check-only', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(restore(arguments.package_root, arguments.check_only), indent=2))
