# Data dictionary

## Charging and outage inputs

Boulder station identifiers link the transaction summary, station coordinates,
and forecast tensors. `pickup` denotes measured session arrivals per station
and hour, not taxi trips. `energy` is kWh assigned to an hour by distributing a
session's measured total over its duration. It is reconstructed hourly demand,
not a metered within-session power trace. Times use the recorded local calendar,
with preprocessing rules and exact split indices stored in the NPZ and JSON.

Outage counts are customers without service at county/time observations.
Unrecorded county/time pairs remain missing. Event duration and post-peak
half-recovery time are in hours. They are service-time context, not work orders
or observations of individual repair crews. Coordinates are latitude/longitude
in degrees and are used to construct distances, not observed travel routes.

## Integrated scenario tables

Each `rollout_scenarios.csv` contains one row per policy and paired scenario.
The condition is identified by its parent directory and `specification.json`.

| Field or family | Meaning and units |
|---|---|
| `scenario_id`, `group`, `policy` | Pairing identifier, threat class and policy |
| `first_hour` | Start index in the processed hourly time series |
| `smartds_mapping_index` | Index among the twenty declared station/load mappings |
| `true_*` | Realized transition coefficients shared within the pair |
| `cost` | Undiscounted dimensionless total under stated reciprocal-unit weights |
| Mobility loss | Arrival-weighted service disruption in trip equivalents, not measured journey time |
| Unserved charging and curtailed charging | kWh over the service horizon |
| Electrical energy loss | MWh over the hourly solutions |
| Voltage and line-loading fields | Per-unit values from OpenDSS |
| `mean_smartds_projection_fraction` | Mean retained charging fraction between zero and one |
| `smartds_raw_infeasible_hours` | Number of raw requests failing an electrical requirement |
| `smartds_projected_infeasible_hours` | Number failing after the accepted projection |
| `crew_job_count` | Requested jobs, including undelivered work orders |
| `crew_jobs_dispatched` | Jobs admitted by dispatch and assigned to routes |
| `crew_jobs_completed` | Repairs completed within the service horizon |
| `crew_completion_fraction` | Completed/requested fraction for the scenario |
| `crew_mean_completion_h` | Mean scheduled completion time of dispatched jobs in hours |
| `crew_total_travel_h` | Sum of scheduled route travel in hours |
| `mean_control_action_fraction` | Mean timely packet-delivery fraction used by execution |
| `latency_ms` | Decision computation wall time per hour, not packet latency |

Route and completion fields retain serialized per-crew visits and event times.
Printed event strings have finite precision, while cost/state calculations use
their internal numeric values. A requested priority is not a completion event.
Use the simulator's returned-field definitions for every detailed auxiliary
column rather than inferring its meaning from a shortened heading.

## Independent electrical measurements

The `electrical_precision_20260905` result directory contains
`scenario_measurements.csv` and `hourly_accepted_loads.csv`. The scenario table
retains the original policy outputs alongside `precision_losses_mwh_tol8`,
`precision_losses_mwh_tol10`, and corresponding minimum-voltage and infeasible-
hour fields. The suffix specifies the independent OpenDSS convergence tolerance.
Loss is summed over six hourly solutions in MWh. Minimum voltage is the minimum
over those same hours and the fixed base-case active-node set, in p.u.

The hourly table is keyed by scenario, policy, and hour. `accepted_ev_load_kw`
is the vector of accepted incremental EV active power at the feeder loads,
not total base-plus-EV load. Both contexts receive that identical vector.
`loss_mwh_tol8/tol10`, `vmin_pu_tol8/tol10`, `vmax_pu_tol8/tol10`, and
`max_line_pu_tol8/tol10` are independently solved measurements.
`feasible_tol8/tol10` and `iterations_tol8/tol10` record solver checks.

Only the two 1e-10 loss and minimum-voltage fields replace their original
coarse-solver counterparts in the copied analysis frame. No frozen CSV is
overwritten. The inference manifest records measurement and repeat hashes.

## Statistical outputs

`paired_comparisons.csv` defines the condition, group, policy, comparator,
sample size, test family, both means, paired difference, marginal bootstrap
interval, relative reduction, paired t and Wilcoxon probabilities and their
separate Holm adjustments. A negative cost difference favors the policy.
Positive percentage reduction favors the policy. The sign interpretation of
a physical endpoint depends on that endpoint.

`cost_risk_summary.csv` contains the empirical mean, quantiles, fractional
upper-tail means and maximum for each condition/policy. `execution_summary.csv`
contains electrical counts, retained actions, requests, dispatch and completion.
`runtime_descriptive.csv` contains descriptive computation times only.

Figure 2's numerical source rows are retained next to the figure. Figure 1 is
a conceptual schematic and does not encode measurements or estimated effects.
