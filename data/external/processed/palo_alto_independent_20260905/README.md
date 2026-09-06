# Independent municipal charging input

This dataset supports retrospective geographic evaluation of the fixed reference-protection decision rule. It is not a jointly measured infrastructure restoration event.

## Source and rights

Creator and data owner: City of Palo Alto, California.

Source title: Electric Vehicle Charging Station Usage (July 2011 to December 2020).

The [official source record](https://data.paloalto.gov/dataviews/257812/electric-vehicle-charging-station-usage-july-2011-dec-2020/) identifies the February 2021 publication. Its ordinary export is limited to 10,000 rows. The complete source used here has 259,415 rows and was obtained through the source dataset download route recorded in the preprocessing metadata.

Use of the municipal data and derived data is subject to the [City's Open Data Terms and Conditions of Use](https://www.paloalto.gov/Departments/Information-Technology/Open-Data-Portal/Terms-of-Use), dated February 16, 2018 and checked September 5, 2026. These are custom source terms, not CC0 or an author-granted open-data licence. The City does not endorse this analysis. No warranty of source accuracy is asserted. Source rights are not transferred by an accompanying software licence.

The original table is retained locally with SHA-256 `e65f5f5d3861cdd6bf3da2de97aabe590d7708e001db1532202eec52081ba82c`. It contains unrelated fields such as user identifiers. Those fields are not loaded for analysis and are not included in this processed release. The public derivative contains station-level aggregates and public station coordinates only.

## Included inputs and units

`station_hour_inputs.npz` contains the following arrays. Time is the source local civil clock, with 35,064 hourly bins from January 2017 through December 2020. Reported elapsed charging durations determine interval lengths, including daylight-saving transitions. There is no claim of a separately metered UTC power series.

| Array | Meaning and unit |
|---|---|
| `timestamp_local` | Local hourly bin start, NumPy datetime |
| `station_id` | Stable anonymous station index, 33 stations |
| `pickup` | Observed session arrivals within an hour, counts |
| `energy` | Reconstructed energy in the hour, kWh |
| `connected_session_hours` | Reconstructed connected-session duration intersecting the hour, hours |
| `adj` | Row-normalized spatial graph from four neighbours, dimensionless |
| `distance_km` | Pairwise great-circle station distance, km |
| `comm_capacity` | Constructed communication-service capacity proxy from 2017–2018 means, model units |
| `split_train_end_index` | First hour of January 2019 |
| `split_val_end_index` | First hour of January 2020, used as the evaluation boundary |

`station_coordinates.csv` maps station indices to source station names and median latitude/longitude from 2017–2018. Coordinates are in degrees. No user identity or individual vehicle trajectory is represented.

`issued_forecasts.npz` contains 8,778 issued windows, their first-hour indices, and two fixed six-hour forecast inputs. Each forecast has dimensions window × horizon × station × two features, arrivals followed by kWh. The graph forecast retains the selected Boulder model's learned weights and original normalization, replacing only its nonlearned adjacency buffer. The weekly forecast uses the same hours of the preceding week. Both are retained, with no external outcome-based selection. Because the original model was trained on a later period, its use is explicitly retrospective, not a historical deployment claim.

## Transformation and quality checks

Only positive finite session energy, valid start times, valid coordinates, positive charging duration at most 48 hours, and charging duration no greater than reported connection duration are used. Exact duplicates across analysis fields are removed. Station eligibility requires at least 100 valid sessions during 2017–2018 and never uses evaluation-year activity.

The rules retain 159,587 sessions across 33 stations, including 20,042 in 2020. Hourly energy is uniform within each reported charging interval, not measured hourly power. The reconstructed total is 1,416,780.194 kWh. Independent direct interval overlap reconstruction checks every station-hour value and every arrival count. The largest cell discrepancy is approximately 4.77 × 10^-7 kWh from stored float32 precision.

The preparation report preserves the complete filtering counts, temporal rules, source and model hashes, and original array hashes. The study protocol was fixed before policy outcomes. The 2019 series is historical input where needed, not a source for selecting the protection margin.

## Execution scope

Demand, station geometry and budgets are transferred to the fixed service-restoration benchmark. Budget statistics use 2017–2018 only. The central transition coefficients, uncertainty set, objective weights, selected protection margin, packet response functions, repair-duration prior and SMART-DS feeder retain the source benchmark's definitions. Route travel uses the new coordinates. Packet response functions are assigned deterministically by station index. They are not Palo Alto packet measurements.

The experiment therefore checks independent geographic charging input and station placement, not local feeder identification, real crew logs or a jointly observed disaster. Both forecast inputs run all eight fixed conditions and both common-route groups. Results must not be selected by favourable sign.

## Reproduction

The existing prepared arrays suffice for policy execution. From the repository root, the independent runner accepts these inputs and the frozen selection report. Source-download and preparation scripts are also supplied for full reconstruction. Rebuilding the original sessions requires obtaining the municipal table from its owner under the source terms. The separate verification script checks full source-to-hour reconstruction, all weekly predictions, selected graph predictions and fresh policy executions when that original source is present.
