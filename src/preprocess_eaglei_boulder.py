"""Extract Boulder-area outage evidence from the EAGLE-I county panel.

This script links datasets only at their defensible spatial/temporal scale:
City of Boulder EV transactions are paired with Boulder County outage context.
It does not claim station-level outage labels or a causal utility linkage.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data" / "external" / "raw" / "eaglei"
OUT_DEFAULT = ROOT / "data" / "external" / "processed" / "eaglei_boulder"
METADATA = ROOT / "data" / "external" / "metadata"

# Boulder County plus its six directly adjacent Colorado counties.
COUNTIES = {
    "08013": "Boulder",
    "08014": "Broomfield",
    "08047": "Gilpin",
    "08049": "Grand",
    "08059": "Jefferson",
    "08069": "Larimer",
    "08123": "Weld",
}


def digest(path: Path, algorithm: str = "sha256", block: int = 2**20) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            value.update(chunk)
    return value.hexdigest()


def load_expected_files(raw: Path) -> list[Path]:
    manifest_path = METADATA / "eaglei_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "EAGLE-I manifest is not present; allow the verified download to complete before preprocessing"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [ROOT / item["path"] for item in manifest["files"] if item["name"].startswith("eaglei_outages_")]
    files = sorted(files)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Manifest-listed outage files are missing: {missing}")
    if len(files) != 12:
        raise RuntimeError(f"Expected 12 annual outage files (2014-2025), found {len(files)}")
    return files


def events_from_series(series: pd.Series, threshold: int, max_gap_minutes: int = 45) -> pd.DataFrame:
    active = series.loc[series.ge(threshold)].dropna().sort_index()
    columns = [
        "event_id",
        "threshold_customers",
        "start",
        "end_exclusive",
        "duration_h",
        "peak_customers_out",
        "customer_hours_above_threshold",
        "time_to_peak_h",
        "post_peak_half_recovery_h",
    ]
    if active.empty:
        return pd.DataFrame(columns=columns)
    group_id = active.index.to_series().diff().gt(pd.Timedelta(minutes=max_gap_minutes)).cumsum()
    records = []
    for event_no, (_, values) in enumerate(active.groupby(group_id), start=1):
        start = values.index.min()
        end = values.index.max() + pd.Timedelta(minutes=15)
        peak = float(values.max())
        peak_time = values.idxmax()
        after_peak = values.loc[peak_time:]
        half_candidates = after_peak.loc[after_peak.le(max(threshold, peak / 2.0))]
        half_h = (
            float((half_candidates.index[0] - peak_time).total_seconds() / 3600.0)
            if not half_candidates.empty
            else np.nan
        )
        records.append(
            {
                "event_id": f"T{threshold:04d}_{event_no:05d}",
                "threshold_customers": threshold,
                "start": start,
                "end_exclusive": end,
                "duration_h": (end - start).total_seconds() / 3600.0,
                "peak_customers_out": peak,
                "customer_hours_above_threshold": float(values.sum() * 0.25),
                "time_to_peak_h": (peak_time - start).total_seconds() / 3600.0,
                "post_peak_half_recovery_h": half_h,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    annual_files = load_expected_files(raw)

    selected_chunks: list[pd.DataFrame] = []
    source_schemas: dict[str, list[str]] = {}
    raw_rows = 0
    selected_rows = 0
    for path in annual_files:
        year_selected = 0
        for chunk in pd.read_csv(
            path,
            dtype={"fips_code": "string", "county": "string", "state": "string"},
            chunksize=args.chunksize,
        ):
            source_schemas[path.name] = list(chunk.columns)
            if "customers_out" not in chunk.columns and "sum" in chunk.columns:
                chunk = chunk.rename(columns={"sum": "customers_out"})
            if "customers_out" not in chunk.columns:
                raise ValueError(f"No outage-count field in {path.name}: {list(chunk.columns)}")
            raw_rows += len(chunk)
            chunk["fips_code"] = chunk["fips_code"].str.zfill(5)
            keep = chunk.loc[chunk["fips_code"].isin(COUNTIES)].copy()
            if keep.empty:
                continue
            keep["customers_out"] = pd.to_numeric(keep["customers_out"], errors="coerce")
            keep["run_start_time_utc"] = pd.to_datetime(keep["run_start_time"], errors="coerce", utc=True)
            keep = keep.drop(columns=["run_start_time"])
            keep = keep.dropna(subset=["run_start_time_utc", "customers_out"])
            keep["source_year"] = int(path.stem[-4:])
            selected_chunks.append(keep)
            year_selected += len(keep)
            selected_rows += len(keep)
        print(f"{path.name}: selected {year_selected:,} rows")

    panel = pd.concat(selected_chunks, ignore_index=True)
    duplicate_rows = int(panel.duplicated(["fips_code", "run_start_time_utc"]).sum())
    # A duplicated county-time record cannot represent two independent counts;
    # preserve the maximum reported outage and retain a duplication audit count.
    panel = (
        panel.groupby(["fips_code", "run_start_time_utc"], as_index=False)
        .agg(
            county=("county", "first"),
            state=("state", "first"),
            customers_out=("customers_out", "max"),
            source_year=("source_year", "first"),
        )
        .sort_values(["run_start_time_utc", "fips_code"])
    )
    panel["customers_out"] = panel["customers_out"].astype("int64")
    panel_path = out / "boulder_adjacent_counties_15min.csv.gz"
    with gzip.open(panel_path, "wt", encoding="utf-8", newline="") as handle:
        panel.to_csv(handle, index=False, date_format="%Y-%m-%d %H:%M:%S")

    pivot = panel.pivot(index="run_start_time_utc", columns="fips_code", values="customers_out").sort_index()
    boulder = pivot["08013"].dropna()
    events = pd.concat([events_from_series(boulder, threshold) for threshold in (50, 100, 500)], ignore_index=True)
    events_path = out / "boulder_outage_events_threshold_sensitivity.csv"
    events.to_csv(events_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    changes = pivot.diff()
    level_corr = pivot.corr(method="spearman", min_periods=96)
    change_corr = changes.corr(method="spearman", min_periods=96)
    level_corr.to_csv(out / "county_outage_level_spearman.csv")
    change_corr.to_csv(out / "county_outage_change_spearman.csv")

    coverage = pd.DataFrame(
        {
            "fips_code": list(COUNTIES),
            "county": [COUNTIES[fips] for fips in COUNTIES],
            "observations": [int(pivot[fips].notna().sum()) for fips in COUNTIES],
            "first_observation": [str(pivot[fips].first_valid_index()) for fips in COUNTIES],
            "last_observation": [str(pivot[fips].last_valid_index()) for fips in COUNTIES],
            "median_customers_out": [float(pivot[fips].median()) for fips in COUNTIES],
            "p99_customers_out": [float(pivot[fips].quantile(0.99)) for fips in COUNTIES],
            "max_customers_out": [float(pivot[fips].max()) for fips in COUNTIES],
        }
    )
    coverage_path = out / "county_coverage_summary.csv"
    coverage.to_csv(coverage_path, index=False)

    event_summary = {}
    for threshold, rows in events.groupby("threshold_customers"):
        event_summary[str(int(threshold))] = {
            "events": int(len(rows)),
            "duration_h_median": float(rows["duration_h"].median()),
            "duration_h_p90": float(rows["duration_h"].quantile(0.9)),
            "peak_customers_median": float(rows["peak_customers_out"].median()),
            "peak_customers_p90": float(rows["peak_customers_out"].quantile(0.9)),
            "half_recovery_h_median": float(rows["post_peak_half_recovery_h"].median()),
            "half_recovery_observed_fraction": float(rows["post_peak_half_recovery_h"].notna().mean()),
        }

    outputs = [
        panel_path,
        events_path,
        coverage_path,
        out / "county_outage_level_spearman.csv",
        out / "county_outage_change_spearman.csv",
    ]
    report = {
        "source": {
            "dataset": "EAGLE-I Recorded Electricity Outages 2014-2025",
            "doi": "10.6084/m9.figshare.24237376.v4",
            "license": "CC BY 4.0",
            "annual_files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in annual_files
            ],
        },
        "scope": {
            "target": "Boulder County and six directly adjacent Colorado counties",
            "fips": COUNTIES,
            "linkage_rule": (
                "County-level disturbance context for City of Boulder EV transactions; no station-level "
                "outage label and no causal utility linkage are asserted."
            ),
        },
        "counts": {
            "source_rows_scanned": raw_rows,
            "selected_rows_before_deduplication": selected_rows,
            "duplicate_county_time_rows": duplicate_rows,
            "selected_rows_after_deduplication": int(len(panel)),
            "boulder_observations": int(boulder.shape[0]),
        },
        "source_schema_by_year": source_schemas,
        "schema_harmonization": {"eaglei_outages_2023.csv": {"sum": "customers_out"}},
        "time": {
            "timezone": "UTC",
            "first": str(panel["run_start_time_utc"].min()),
            "last": str(panel["run_start_time_utc"].max()),
        },
        "missing_value_semantics": (
            "The source omits zero-outage rows; a missing county-time entry can mean zero outages or a data-collection "
            "gap and is therefore retained as missing rather than silently imputed to zero."
        ),
        "event_definition": (
            "A threshold event is a sequence of observations at or above the threshold with no gap above 45 min. "
            "Results are reported at 50, 100, and 500 customers to expose threshold sensitivity."
        ),
        "event_summary": event_summary,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in outputs
        },
    }
    report_path = out / "eaglei_boulder_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = """# EAGLE-I Boulder-area processed data

Generated by `src/preprocess_eaglei_boulder.py` from all verified EAGLE-I
annual files (2014–2025). The panel contains Boulder County and the six directly
adjacent Colorado counties at the source 15-minute resolution. Source timestamps
are explicitly stored as UTC. Missing county-time rows are not filled with zero,
because the source documentation states that they may denote either zero outages
or a collection gap.

The linkage to the City of Boulder EV session dataset is deliberately limited:
it supplies county-scale disturbance timing, magnitude, duration, and spatial
co-movement priors. It does not turn county outage counts into station outage
labels. Three outage thresholds are retained so conclusions cannot depend on a
single arbitrary event definition.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
