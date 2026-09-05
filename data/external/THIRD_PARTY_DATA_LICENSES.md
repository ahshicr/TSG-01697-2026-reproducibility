# Public data provenance and original licences

The sources have different evidentiary roles and are not a jointly observed,
geographically co-located utility system. This deposit does not relicense them.

## City of Boulder charging transactions

The City of Boulder Electric Vehicle Charging Station Data are released under
CC0 1.0. The original catalog is
<https://open-data.bouldercolorado.gov/datasets/electric-vehicle-charging-station-data/about>.
The 2018–2023 transactions supply measured arrivals and session energy.
Hourly load shapes are reconstructed, not measured within-session profiles.
Processed transactions, station summaries, tensors, quality records and
geographical coordinates are included. The original access time, source URL,
byte count and digest are in `metadata/boulder_ev_sessions_manifest.json`.

## EAGLE-I county outages

The reused release is The Environment for Analysis of Geo-Located Energy
Information's Recorded Electricity Outages 2014–2025, version 4,
<https://doi.org/10.6084/m9.figshare.24237376.v4>, under CC BY 4.0.
Attribution is to Brelsford and colleagues and the accompanying data article,
<https://doi.org/10.1038/s41597-024-03095-5>.
The article describes the earlier coverage through 2022, while the reused
version extends the archive. The release metadata and original checksums are
retained in `metadata/eaglei_manifest.json` and the Figshare source record.

This deposit contains the processed Boulder and adjacent-county observations,
coverage summaries and event tables, not all national raw files. Missing rows
remain missing. County recovery times are not utility work orders or individual
crew traces. The original national archive remains accessible from its publisher.

## SMART-DS electrical network

SMART-DS Synthetic Electrical Network Data, version 1.0, are published by NREL
through OEDI under CC BY 4.0. The catalog is
<https://data.openei.org/submissions/2981>. The complete selected SFO P1U
base-peak feeder and its required OpenDSS dependencies are included unchanged.
It is a synthetic San Francisco-area network, not Boulder's utility feeder.
The original object keys, checksums and access records are in
`metadata/smartds_v1.0_sfo_p1u_dt104_manifest.json`.

## Generated material and integrity

Simulated packet traffic, route scenarios, derived forecast arrays and statistical
comparisons are identified separately from observations. Original attribution
and licence notices are retained. Public access for reviewing and reproducing
the study does not change ownership of third-party material.

Run `python src/verify_reviewer_minimal.py` from the snapshot root to check the
released files and statistics. The original raw-data digests in source metadata
describe the larger acquisition archive. They are not a claim that all raw
national data files are included in this smaller deposit.
