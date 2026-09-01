#!/usr/bin/env python3
"""Download and verify external datasets used by the substantive revision.

The downloader is intentionally independent of HPC.  It preserves publisher
metadata, uses resumable transfers for large Figshare files, writes atomically,
and produces SHA-256/MD5 manifests inside the project.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib.parse import quote
from xml.etree import ElementTree
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
RAW = EXTERNAL / "raw"
METADATA = EXTERNAL / "metadata"

EAGLE_ARTICLE_ID = 24237376
EAGLE_API = f"https://api.figshare.com/v2/articles/{EAGLE_ARTICLE_ID}"

BOULDER_LAYER = (
    "https://services.arcgis.com/ePKBjXrBZ2vEEgWd/arcgis/rest/services/"
    "Electric_Vehicle_Charging_Station_Data/FeatureServer/0"
)

SMARTDS_BUCKET = "https://oedi-data-lake.s3.amazonaws.com"
SMARTDS_FEEDER_PREFIX = (
    "SMART-DS/v1.0/peak/SFO/P1U/scenarios/base_peak/opendss_no_loadshapes/"
    "p1uhs0_1247/p1uhs0_1247--p1udt104/"
)
SMARTDS_SUPPORT_KEYS = (
    "SMART-DS/v1.0/peak/SFO/P1U/scenarios/base_peak/geojson/p1uhs0_1247--p1udt104.json",
    "SMART-DS/v1.0/peak/SFO/P1U/scenarios/base_peak/metrics.csv",
    "SMART-DS/v1.0/User_Guide/Readme.md",
    "SMART-DS/v1.0/User_Guide/Readme.pdf",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    client = requests.Session()
    client.headers.update({"User-Agent": "TSG-01697-2026-reproducibility/1.0"})
    client.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    return client


def digest(path: Path, algorithm: str = "sha256", block: int = 2**20) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download_resumable(file_record: dict[str, Any]) -> dict[str, Any]:
    destination = RAW / "eaglei" / file_record["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(file_record["size"])
    expected_md5 = file_record.get("supplied_md5") or file_record.get("computed_md5")

    if destination.exists() and destination.stat().st_size == expected_size:
        actual_md5 = digest(destination, "md5")
        if not expected_md5 or actual_md5 == expected_md5:
            return {
                "name": destination.name,
                "path": destination.relative_to(ROOT).as_posix(),
                "bytes": expected_size,
                "md5": actual_md5,
                "sha256": digest(destination),
                "status": "verified-existing",
            }

    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected_size:
        raise RuntimeError(f"Partial file exceeds expected size: {partial}")

    headers = {"Range": f"bytes={start}-"} if start else {}
    with session().get(file_record["download_url"], headers=headers, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        if start and response.status_code != 206:
            start = 0
        mode = "ab" if start and response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=4 * 2**20):
                if chunk:
                    handle.write(chunk)

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"Size mismatch for {destination.name}: {actual_size} != {expected_size}")
    actual_md5 = digest(partial, "md5")
    if expected_md5 and actual_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {destination.name}: {actual_md5} != {expected_md5}")
    os.replace(partial, destination)
    return {
        "name": destination.name,
        "path": destination.relative_to(ROOT).as_posix(),
        "bytes": actual_size,
        "md5": actual_md5,
        "sha256": digest(destination),
        "status": "downloaded",
    }


def download_eaglei(workers: int) -> None:
    client = session()
    article = client.get(EAGLE_API, timeout=60).json()
    article["retrieved_at_utc"] = utc_now()
    write_json(METADATA / "eaglei_figshare_article_24237376.json", article)

    files = article["files"]
    print(f"EAGLE-I: {len(files)} files, {sum(int(item['size']) for item in files):,} bytes")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(download_resumable, item): item["name"] for item in files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results):02d}/{len(files):02d}] {result['status']}: {result['name']} ({result['bytes']:,} bytes)")

    manifest = {
        "dataset": article["title"],
        "article_id": EAGLE_ARTICLE_ID,
        "doi": article.get("doi"),
        "license": article.get("license"),
        "source": f"https://figshare.com/articles/dataset/{EAGLE_ARTICLE_ID}",
        "retrieved_at_utc": utc_now(),
        "files": sorted(results, key=lambda item: item["name"]),
    }
    write_json(METADATA / "eaglei_manifest.json", manifest)


def download_boulder() -> None:
    client = session()
    layer_metadata = client.get(BOULDER_LAYER, params={"f": "json"}, timeout=60).json()
    if "error" in layer_metadata:
        raise RuntimeError(layer_metadata["error"])
    count_payload = client.get(
        f"{BOULDER_LAYER}/query",
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=60,
    ).json()
    expected_count = int(count_payload["count"])
    fields = [field["name"] for field in layer_metadata["fields"]]
    object_id = layer_metadata["objectIdField"]

    metadata_record = {
        "dataset": "City of Boulder Electric Vehicle Charging Station Data",
        "source": BOULDER_LAYER,
        "source_item": "https://www.arcgis.com/home/item.html?id=95992b3938be4622b07f0b05eba95d4c",
        "license": "CC0",
        "retrieved_at_utc": utc_now(),
        "expected_records": expected_count,
        "layer": layer_metadata,
    }
    write_json(METADATA / "boulder_ev_sessions_arcgis.json", metadata_record)

    destination = RAW / "boulder_ev_sessions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".csv.part")
    page_size = min(1000, int(layer_metadata.get("maxRecordCount", 1000)))
    rows_written = 0
    seen: set[Any] = set()
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for offset in range(0, expected_count, page_size):
            payload = client.get(
                f"{BOULDER_LAYER}/query",
                params={
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "false",
                    "orderByFields": f"{object_id} ASC",
                    "resultOffset": offset,
                    "resultRecordCount": page_size,
                    "f": "json",
                },
                timeout=120,
            ).json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            features = payload.get("features", [])
            if not features:
                raise RuntimeError(f"Empty Boulder page at offset {offset}")
            for feature in features:
                attributes = feature["attributes"]
                oid = attributes[object_id]
                if oid in seen:
                    raise RuntimeError(f"Duplicate object ID {oid}")
                seen.add(oid)
                writer.writerow(attributes)
                rows_written += 1
            print(f"Boulder EV: {rows_written:,}/{expected_count:,}")

    if rows_written != expected_count:
        raise RuntimeError(f"Boulder count mismatch: {rows_written} != {expected_count}")
    os.replace(partial, destination)
    manifest = {
        "dataset": metadata_record["dataset"],
        "source": BOULDER_LAYER,
        "license": "CC0",
        "retrieved_at_utc": utc_now(),
        "records": rows_written,
        "file": {
            "path": destination.relative_to(ROOT).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": digest(destination),
        },
    }
    write_json(METADATA / "boulder_ev_sessions_manifest.json", manifest)
    print(f"Boulder EV complete: {rows_written:,} rows, {destination.stat().st_size:,} bytes")


def list_smartds_prefix(prefix: str) -> list[dict[str, Any]]:
    """List all objects below a public SMART-DS S3 prefix."""
    client = session()
    continuation: str | None = None
    records: list[dict[str, Any]] = []
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation:
            params["continuation-token"] = continuation
        response = client.get(f"{SMARTDS_BUCKET}/", params=params, timeout=120)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        for content in root.findall("s3:Contents", namespace):
            key = content.findtext("s3:Key", namespaces=namespace)
            if not key:
                continue
            records.append(
                {
                    "key": key,
                    "size": int(content.findtext("s3:Size", default="0", namespaces=namespace)),
                    "etag": (content.findtext("s3:ETag", default="", namespaces=namespace) or "").strip('"'),
                    "last_modified": content.findtext("s3:LastModified", namespaces=namespace),
                }
            )
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=namespace) or "false").lower()
        if truncated != "true":
            break
        continuation = root.findtext("s3:NextContinuationToken", namespaces=namespace)
        if not continuation:
            raise RuntimeError("SMART-DS listing was truncated without a continuation token")
    return records


def smartds_object_metadata(key: str) -> dict[str, Any]:
    for record in list_smartds_prefix(key):
        if record["key"] == key:
            return record
    raise RuntimeError(f"SMART-DS object not found: {key}")


def download_smartds_object(record: dict[str, Any]) -> dict[str, Any]:
    key = record["key"]
    version_relative = Path(key).relative_to("SMART-DS/v1.0")
    destination = RAW / "smartds_v1.0" / version_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(record["size"])
    expected_md5 = record.get("etag", "")
    if destination.exists() and destination.stat().st_size == expected_size:
        actual_md5 = digest(destination, "md5")
        if not expected_md5 or "-" in expected_md5 or actual_md5 == expected_md5:
            return {
                **record,
                "path": destination.relative_to(ROOT).as_posix(),
                "md5": actual_md5,
                "sha256": digest(destination),
                "status": "verified-existing",
            }

    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected_size:
        raise RuntimeError(f"Partial file exceeds expected size: {partial}")
    headers = {"Range": f"bytes={start}-"} if start else {}
    url = f"{SMARTDS_BUCKET}/{quote(key, safe='/')}"
    with session().get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        if start and response.status_code != 206:
            start = 0
        mode = "ab" if start and response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=4 * 2**20):
                if chunk:
                    handle.write(chunk)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"SMART-DS size mismatch for {key}")
    actual_md5 = digest(partial, "md5")
    if expected_md5 and "-" not in expected_md5 and actual_md5 != expected_md5:
        raise RuntimeError(f"SMART-DS MD5 mismatch for {key}: {actual_md5} != {expected_md5}")
    os.replace(partial, destination)
    return {
        **record,
        "path": destination.relative_to(ROOT).as_posix(),
        "md5": actual_md5,
        "sha256": digest(destination),
        "status": "downloaded",
    }


def download_smartds(workers: int) -> None:
    """Download a complete, independently solvable SMART-DS feeder subset."""
    records = list_smartds_prefix(SMARTDS_FEEDER_PREFIX)
    records.extend(smartds_object_metadata(key) for key in SMARTDS_SUPPORT_KEYS)
    deduplicated = {item["key"]: item for item in records}
    records = [deduplicated[key] for key in sorted(deduplicated)]
    print(f"SMART-DS: {len(records)} files, {sum(int(item['size']) for item in records):,} bytes")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(download_smartds_object, item): item["key"] for item in records}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results):02d}/{len(records):02d}] {result['status']}: {result['key']}")
    manifest = {
        "dataset": "SMART-DS Synthetic Electrical Network Data, version 1.0",
        "selected_network": "SFO P1U base-peak feeder p1uhs0_1247--p1udt104",
        "selection_reason": (
            "Complete OpenDSS feeder with coordinates and reference analysis; selected as an independent "
            "distribution-network validation case, not as a geographically co-located Boulder feeder."
        ),
        "source_bucket": SMARTDS_BUCKET,
        "catalog": "https://data.openei.org/submissions/2981",
        "license": "CC BY 4.0",
        "retrieved_at_utc": utc_now(),
        "files": sorted(results, key=lambda item: item["key"]),
    }
    write_json(METADATA / "smartds_v1.0_sfo_p1u_dt104_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("boulder", "eaglei", "smartds", "all"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.dataset in ("boulder", "all"):
        download_boulder()
    if args.dataset in ("eaglei", "all"):
        download_eaglei(args.workers)
    if args.dataset in ("smartds", "all"):
        download_smartds(args.workers)
    print(f"Completed in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
