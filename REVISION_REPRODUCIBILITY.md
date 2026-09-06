# Evidence and execution map

## Chronology and policy definitions

Boulder observations from 2018 through 2021 supply training statistics.
2022 supplies model selection and the route-score validation comparison.
2023 supplies the revised evaluation. Every training or validation forecast
target stays within its declared period. Budget and normalization means use
training observations only. Missing forecasts fall back to past observations,
not realized future demand.

The service selectors receive the same forecasts, current threats and
backlogs, jobs, travel and service durations, dispatch inputs, and six route
candidates. The candidate set includes transition-derived priorities and is
shared even with the no-transition selector. Their distinction is the score:

| Selector | State prediction used to score common routes |
|---|---|
| Forecast matched | Holds unrepaired threats fixed |
| Central PC | Propagates the fitted central transition |
| Robust PC | Uses the largest service cost over 21 declared matrices |
| PC with reference protection | Uses the largest proposed-minus-reference difference at a common matrix and keeps the exact reference route unless the fixed margin is passed |

The protection margin was selected from four declared values using 2022 only,
and retained for all later comparisons. A second route group removes transition
coefficients from route generation entirely. Within each group, the compared
selectors share the same six candidates. This is an ablation, not a comparison
of protected cost on one group with central cost on another group.

The primary study also retains static, greedy, oracle, exposure-matched, and
exposure-central policies. Only the oracle uses future innovations and demand.
Routes are selected at initial dispatch and remain committed. Hourly forecasts
and charging and communication allocations are updated thereafter. Travel or
unfinished repair does not clear a threat. Station reassignment is a separate
experiment, not an extra operation in the primary loop.

## Supporting materials

| Evidence | Released location | Main use |
|---|---|---|
| Processed measured charging and coordinates | `data/external/processed/boulder_ev` | Demand, graph, electrical mapping and crew travel |
| County outage observations and events | `data/external/processed/eaglei_boulder` | Separate outage calibration and service-duration context |
| Original selected feeder | `data/external/raw/smartds_v1.0` | Online electrical feasibility |
| Strict forecast repetitions and trained weights | `results/real_ev_strict_20260905` | Table II and fixed policy inputs |
| Transition calibration and packet response surface | `results/calibration_strict_20260905`, `results/packet_training_20260905` | Table III and execution inputs |
| Validation comparisons and fixed-method decision | `results/service_score_validation_20260905` | Documented design choice before revised testing |
| All declared integrated experiments | `results/submission_service_20260905` | Tables V, VI, VII, X and the manuscript appendix comparisons |
| Standalone electrical checks | `results/standalone_smartds_strict_20260905` | Table VIII and Figure 2 electrical panel |
| Independent primary electrical measurements | `results/electrical_precision_20260905` | Manuscript appendix physical outcome table, loss and minimum voltage |
| Electrical precision repeats | `results/electrical_precision_smoke_20260905`, `results/electrical_precision_reverse_20260905` | Solver tolerance and execution order checks |
| Station assignment | `results/operational/station_choice` | Table IX |
| Separate route-scale comparisons | `results/operational/crew_routing_128` | Table XI and Figure 2 route panel |
| Separate packet model study | `results/operational/packet_network` | Queue, erasure, deadline and proxy comparisons |
| Geographical forecast holdout | `results/real_ev_spatial_holdout_inductive` | Independent station-block comparison |
| Historical method after training-only correction | `results/submission_20260905/primary`, `results/submission_20260905/source_snapshot` | Superseded method evidence, not current primary results |
| Fixed protection protocol and parameter ledger | `config/guarded_improvement_protocol.md`, `config/guarded_parameter_ledger.csv` | Predeclared margin selection, tests and settings |
| All margin-selection records | `results/guarded_validation_20260905` | Four options and the fixed selection decision |
| Protected domestic conditions | `results/guarded_test_20260905` | Eight conditions, both route groups and all numerical outcomes |
| Independent municipal station-hour input | `data/external/processed/palo_alto_independent_20260905` | Prepared charging, coordinates and both fixed forecasts |
| Independent protected conditions | `results/guarded_external_20260905` | Both forecasts, all eight conditions and both route groups |
| Independent new choice and mean-statistic reconstruction | `results/guarded_verification_20260905.json` | Every saved finite choice, fallback and both new test families |
| Independent original municipal reconstruction | `results/independent_input_verification_20260905.json` | Full source-to-hour reconstruction and forecast checks made locally with the original source |

The complete parameter ledger is `config/full_parameter_ledger.csv`.
The complete proofs and parameter/comparison tables are in the manuscript
appendix, within the same PDF. No separate supplementary submission is needed.
The marked manuscript retains old material, so its numbering differs from the
clean manuscript. Its red strikeout denotes deletion and blue denotes addition.
Old images are reduced and crossed out. Structurally changed tables are shown
as complete old and new tables rather than mixed columns.

## Statistical interpretation

Paired rows share scenario inputs. Bootstrap intervals concern paired mean
differences and are marginal, not simultaneous intervals. Families contain
15 primary cost comparisons, 9 physical comparisons, 45 integrated sensitivity
comparisons, 2 backup comparisons, and 1 descriptive reference-subset comparison.
The reference subset is not the full 4,096-case result. Weight experiments rerun
selection and execution, rather than reweighting fixed trajectories.

Physical outcomes are recorded separately from the policy score. Completion
fraction uses requested jobs as denominator, including requests not delivered.
Scheduled completion and total travel describe dispatched routes, including
work finishing beyond the six-hour service horizon. Computation time is not
packet delay and is not used as an inferential physical endpoint.

The original execution solver uses convergence tolerance 1e-4. Its very small
loss differences required a numerical precision check. Separate OpenDSS
contexts measured identical accepted load vectors at tolerances 1e-8 and 1e-10,
with at most 100 iterations, for all 4,096 central and matched pairs. Neither
context changes scores, executed service, or routes. All original records are
preserved. The manuscript appendix physical outcome table uses the 1e-10 loss and voltage measurements.
The other seven endpoints still use the original execution records. The largest
scenario loss difference between the two tighter tolerances was 9.94e-8 MWh.
A 32-pair reversed-order repeat differed by at most 3.10e-8 MWh. The small mean
loss increase remains, and the voltage mean interval includes zero. The
measurement verifier checks exact pairing, unchanged policy outcomes, all
hourly aggregates, load identity on repeats, and explicitly stated numerical
resolution thresholds. Those thresholds are measurement checks, not a means
of choosing favorable statistical conclusions.

The robust extension does not establish primary mean or tail superiority.
Empirical upper-tail means include fractional mass at the probability boundary
and are descriptive, not tail-significance tests. Statistical uncertainty is
conditional on the specified public data and simulation, not evidence of
variation across independently observed utility systems.

The protected study uses separate 80-comparison domestic and 160-comparison
external families. A priori noninferiority allows a cost difference of at most
0.05% of paired reference cost. The reported positive-excess comparison averages
the six fixed coefficient-error conditions within each scenario ID before
resampling IDs. It is not a favorable-condition subset. Tail intervals are
exploratory and not multiplicity-adjusted. In particular, the protected rule's
original-route upper-five-percent mean cost is higher than central PC, with a
positive marginal interval. Noninferiority of the overall mean does not negate
that adverse tail result or establish realized improvement on every scenario.

## Physical and mathematical scope

The SMART-DS network is a synthetic feeder and is not geographically co-located
with Boulder. Twenty declared address-group mappings are used. The EV additions
are balanced three-phase, wye-connected, constant-PQ loads with lagging power
factor 0.98. Voltage, line loading and convergence are checked after a request.
Bisection retains a tested feasible action and reapplies it before crediting
service. This is not a proof of globally maximal AC charging capacity.

The primary measured-load gate is nonbinding. The binding companion increases
both demand and charging budget twentyfold. No feeder switching, fault isolation,
network reconnection, or equipment-parameter repair is simulated. Crew completion
removes an abstract service threat, not an observed utility switching action.

The continuous proof concerns the stated prediction, service and allocation
map. Packet deadlines, AC accept/project decisions, finite route choices and
integer completion are separate operations. The route regret inequality is
conditional on score errors. Exhaustive small-instance tests do not establish
a population-wide score-error bound.

The added theorem compares candidate and reference scores at a common matrix
and explicitly accounts for differential score error. Identical fallback routes
have identical execution under the same exogenous inputs. The proof does not
turn estimated model membership or finite numerical checks into an unconditional
guarantee for an unseen real system. The general baseline-relative robust
improvement principle is attributed to earlier work rather than claimed as new.

## Reproduction scope

The supplied processed inputs and trained forecasts suffice for the frozen
integrated simulation. Trained weights and training code are also supplied.
The full national EAGLE-I archive is not duplicated because county extracts and
source metadata suffice for this deposit. Original acquisition and processing
code remain available. Source records identify the original public releases.

Scripts retained for historical experiments are not alternate entry points for
the current primary result. In particular, `simulate_rollout_revised.py` alone
retains the historical selector default. Use `run_frozen_service_experiments.py`.
Older broad package-check scripts are not evidence that this snapshot passes.
The earlier read-only checker is `verify_reviewer_minimal.py`. The current
protected execution entry is `reproduce_reference_protection.py`, whose optional
full-statistics mode calls the independently written `verify_guarded_records.py`.
The full raw-input reconstruction checker is `verify_independent_input.py`, which
requires the separately acquired original municipal source and is not a
prerequisite for executing the released processed inputs.
