# Third-party data, provenance, and permitted reuse

This file records the exact external data used by the revised experiments.
The three sources serve different evidentiary roles and are not represented as
a geographically co-located utility system.

## City of Boulder EV charging sessions

- Dataset: Electric Vehicle Charging Station Data
- Publisher: City of Boulder, Colorado
- Source: <https://open-data.bouldercolorado.gov/datasets/electric-vehicle-charging-station-data/about>
- Local raw file: `raw/boulder_ev_sessions.csv`
- License: CC0 1.0 / public-domain dedication
- Role: measured charging transactions, station arrivals, session energy, and
  the fixed temporal/spatial forecast tests
- Boundary: hourly within-session load is reconstructed by uniformly spreading
  measured session energy over the reported duration; it is not metered hourly
  power.

The authoritative retrieval record, byte count, SHA-256, ArcGIS endpoint, and
access time are in `metadata/boulder_ev_sessions_manifest.json`.

## EAGLE-I recorded electricity outages

- Dataset: The Environment for Analysis of Geo-Located Energy Information's
  Recorded Electricity Outages 2014--2025
- Article DOI: <https://doi.org/10.1038/s41597-024-03095-5>
- Dataset DOI: <https://doi.org/10.6084/m9.figshare.24237376.v4>
- Local raw directory: `raw/eaglei/`
- License: CC BY 4.0
- Attribution: Brelsford et al., *Scientific Data* (2024), and the accompanying
  Figshare dataset
- Role: county-level outage persistence and event-duration context
- Boundary: absent county-time rows remain missing because the release does
  not distinguish true zero outages from collection gaps. These data are not
  station-level labels, utility work orders, or causal crew traces.

All 17 repository files, checksums, and access metadata are in
`metadata/eaglei_manifest.json`.

## SMART-DS synthetic distribution network

- Dataset: SMART-DS Synthetic Electrical Network Data, version 1.0
- Publisher: National Renewable Energy Laboratory / OEDI
- Catalog: <https://data.openei.org/submissions/2981>
- Selected case: SFO P1U base-peak feeder
  `p1uhs0_1247--p1udt104`
- Local raw directory: `raw/smartds_v1.0/`
- License: CC BY 4.0
- Role: independent OpenDSS action-in-the-loop electrical feasibility
- Boundary: this is a realistic synthetic San Francisco feeder, not the
  Boulder utility network. The published OpenDSS files are unmodified; EV
  requests are mapped to electrically plausible three-phase loads.

The 11 selected files, object-store keys, byte counts, ETags, MD5 values,
SHA-256 values, and access times are in
`metadata/smartds_v1.0_sfo_p1u_dt104_manifest.json`.

## Integrity verification

From the project root, run:

```powershell
python src\verify_revision_package.py --full-raw-hash
```

This recomputes SHA-256 for all 29 external raw files (11,678,014,648 bytes)
and writes `results/operational/revision_package_verification.json`. The local
archive may be redistributed only under the licenses above; downstream users
remain responsible for retaining required attribution.
