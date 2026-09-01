"""Geocode Boulder charging-station addresses with the U.S. Census API.

The station transaction source does not include coordinates.  Coordinates are
therefore kept as a separately sourced enrichment with match status, returned
address, TIGER line identifier, raw responses, and hashes.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
STATIONS = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_stations.csv"
OUT = ROOT / "data" / "external" / "processed" / "boulder_ev" / "boulder_ev_station_coordinates.csv"
RAW = ROOT / "data" / "external" / "raw" / "census_geocoder" / "boulder_station_geocodes.json"
REPORT = ROOT / "data" / "external" / "metadata" / "boulder_station_geocoding_manifest.json"
ENDPOINT = "https://geocoding.geo.census.gov/geocoder/geographies/address"
NOMINATIM = "https://nominatim.openstreetmap.org/search"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    stations = pd.read_csv(STATIONS, dtype={"postal_code": "string"})
    raw_records = []
    output_records = []
    client = requests.Session()
    client.headers.update({"User-Agent": "TSG-01697-2026-reproducibility/1.0"})
    cache: dict[tuple[str, str], dict] = {}
    for number, row in enumerate(stations.itertuples(index=False), start=1):
        key = (str(row.address), str(row.postal_code))
        if key not in cache:
            params = {
                "street": row.address,
                "city": row.city,
                "state": "CO",
                "zip": row.postal_code,
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "format": "json",
            }
            response = client.get(ENDPOINT, params=params, timeout=90)
            response.raise_for_status()
            cache[key] = response.json()
            time.sleep(0.10)
        payload = cache[key]
        raw_records.append({"station_id": row.station_id, "query": key, "response": payload})
        matches = payload.get("result", {}).get("addressMatches", [])
        match = matches[0] if matches else None
        coordinates = match.get("coordinates", {}) if match else {}
        tiger = match.get("tigerLine", {}) if match else {}
        match_status = "matched" if match else "unmatched"
        matched_address = match.get("matchedAddress") if match else None
        geocoder = "U.S. Census Geocoder Public_AR_Current/Current_Current"
        if not match:
            # Fallback is used only when the Census address-range service has no
            # match. Nominatim's road-level result remains visibly lower precision.
            query = f"{row.address}, {row.city}, CO {row.postal_code}, USA"
            osm_response = client.get(
                NOMINATIM,
                params={"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1},
                timeout=90,
            )
            osm_response.raise_for_status()
            osm_matches = osm_response.json()
            raw_records[-1]["nominatim_fallback"] = osm_matches
            if osm_matches:
                osm = osm_matches[0]
                coordinates = {"x": float(osm["lon"]), "y": float(osm["lat"])}
                matched_address = osm.get("display_name")
                match_status = "street_fallback"
                geocoder = "OpenStreetMap Nominatim road-level fallback (ODbL 1.0)"
            time.sleep(1.05)
        output_records.append(
            {
                "station_id": row.station_id,
                "station_name": row.station_name,
                "input_address": row.address,
                "input_postal_code": row.postal_code,
                "match_status": match_status,
                "matched_address": matched_address,
                "longitude": coordinates.get("x"),
                "latitude": coordinates.get("y"),
                "tiger_line_id": tiger.get("tigerLineId"),
                "tiger_side": tiger.get("side"),
                "geocoder": geocoder,
            }
        )
        print(f"[{number:02d}/{len(stations):02d}] {row.station_id}: {match_status}")

    result = pd.DataFrame(output_records)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(raw_records, indent=2, ensure_ascii=False), encoding="utf-8")
    result.to_csv(OUT, index=False)
    manifest = {
        "source": ENDPOINT,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "stations": int(len(result)),
        "unique_address_queries": int(len(cache)),
        "matched": int(result["match_status"].eq("matched").sum()),
        "street_fallback": int(result["match_status"].eq("street_fallback").sum()),
        "unmatched": int(result["match_status"].eq("unmatched").sum()),
        "interpretation": (
            "Address-range coordinates used for distance graphs and routing sensitivity; "
            "they are not charger-meter coordinates."
        ),
        "fallback_source": "OpenStreetMap Nominatim, ODbL 1.0; used only for unmatched addresses",
        "files": {
            RAW.relative_to(ROOT).as_posix(): {"bytes": RAW.stat().st_size, "sha256": sha256(RAW)},
            OUT.relative_to(ROOT).as_posix(): {"bytes": OUT.stat().st_size, "sha256": sha256(OUT)},
        },
    }
    REPORT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
