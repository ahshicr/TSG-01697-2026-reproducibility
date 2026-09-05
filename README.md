# TSG-01697-2026 reproducibility materials

**Transition Based Closed Loop Restoration for Mobility Coupled Distribution
Systems with Executable Actions**

This snapshot contains the corrected, frozen service-cost experiments,
processed public data, trained models, executable simulation, statistical
results, and manuscript sources. It does not require HPC access.

## Current result and earlier versions

On 4,096 paired scenarios, central PC reduces mean dimensionless cost by
0.102684% relative to the control with the same service score but without
propagation. The paired difference is -1.143570 with marginal 95% bootstrap
interval [-1.525666, -0.770441]. The primary Holm family contains fifteen
comparisons. The overall and compound-stress mean effects are supported, but
the nominal, single-domain, and cascade mean intervals include zero.

Service scoring has larger gains over exposure scoring. The robust extension
has higher primary mean and conditional tail costs than central PC. Under
some transition misspecifications it helps, and under others it does not.
All 23 declared conditions and all 72 statistical comparisons are retained.

Earlier repository versions reported a 0.6327% primary gain. That result is
superseded. Subsequent checks identified normalization and budget statistics
outside the training period, test-based selection in the forecast table,
and forecast targets crossing chronological boundaries. The affected
calculations were repeated. The training-only rerun of the historical
exposure method does not preserve its earlier claimed advantage. Its primary
records and source snapshot are retained here, separately from current results.
No earlier tag should be interpreted as the current scientific result.

The service-cost design was fixed after 2022 validation comparisons and before
the revised 2023 evaluation. The test year had previously been inspected for
diagnosis. This is a frozen revision evaluation, not a claim that the year had
never been examined. Design decisions and source digests are preserved.

## Verify without rerunning the simulations

Use Python 3.13.7 and the pinned environment in `requirements.txt`. For the
read-only statistical check, NumPy, pandas, and SciPy suffice.

```powershell
python -m pip install -r requirements.txt
python src/verify_reviewer_minimal.py
```

The verifier checks the snapshot's byte counts and SHA-256 digests, the frozen
method and inputs, the paired records in all 23 conditions, and recomputes all
72 reported comparisons, including bootstrap intervals and separate Holm
corrections for paired t and Wilcoxon tests. It also recomputes empirical cost
quantiles and fractional upper-tail means. Passing it does not prove a theorem,
external field validity, or publication readiness.

## Run the executable method

From this repository's root, use the frozen runner, not the older simulator's
default entry point.

```powershell
python -u src/run_frozen_service_experiments.py --output results/reviewer_smoke_local --conditions primary --smoke-scenarios 16 --workers 2
```

This evaluates all eight primary policies with common inputs. OpenDSS checks
each hourly charging request. Packet delivery determines effective service and
individual dispatch, and recorded integer completion events change threats.
The small run is a reproducibility check, not the paper's full statistical study.

The two electrical measurement endpoints in the manuscript's physical outcome table use
independent high-precision OpenDSS solves of the accepted loads. The original
execution tolerance remains unchanged. Repeat this measurement on 32 paired
scenarios, then check the simulation and all saved validation/test forecasts:

```powershell
python -u src/replay_electrical_precision.py --output results/reviewer_precision_local --scenarios 32 --workers 2 --reverse-policy-order
python -u src/verify_reviewer_execution.py --output results/reviewer_execution_check_local.json
```

The second command compares all eight policies in the small simulation and
re-evaluates every validation and test window for all three trained models.
It checks the independent electrical measurements against the full released
precision replay. It does not retrain the models. Original coarse solver loss
and voltage fields remain available as diagnostics, but are not the secondary
measurements used for inference. The verifier does not require those superseded
fields or computation wall times to be identical across runs.

The complete run is:

```powershell
python -u src/run_frozen_service_experiments.py --output results/reviewer_full_local --workers 16
```

The full experiment is computationally larger than the read-only verification.
Do not overwrite released results to perform a new run. Decision wall times
depend on the machine and concurrent execution and are not expected to match.
The full independent electrical measurement can also be repeated using
`replay_electrical_precision.py --output results/reviewer_precision_full_local
--scenarios 4096 --workers 16`. All repeat commands require fresh output paths.

## Regenerate statistics and manuscript tables

```powershell
python src/analyse_frozen_service_experiments.py
python src/make_operational_tables.py
python src/make_manuscript_appendix.py
python src/make_submission_figures.py
```

The manuscript appendix contains the necessary parameter and comparison tables.
The optional repository reader report is not a separate journal submission attachment.

These commands regenerate outputs from the released scenario rows and input
summaries. Run snapshot verification first. Generated image or PDF metadata
may differ, and editing a released file intentionally invalidates its checksum.
Figure 1 is a retained conceptual illustration, not a quantitative result.
Figure 2 is regenerated from numerical source records.

See `REVISION_REPRODUCIBILITY.md` for evidence mappings and study boundaries,
`DATA_DICTIONARY.md` for variables and units, and
`data/external/THIRD_PARTY_DATA_LICENSES.md` for original data attribution.

## Access and versioning

The material is provided for review and reproduction of the reported study.
Third-party data retain their original licences and attribution. This deposit
does not apply a blanket new licence to those data. Historical versions are
retained in repository history and are not silently amended.
