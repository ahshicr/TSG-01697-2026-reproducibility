# TSG-01697-2026 reproducibility materials

This repository contains the code, processed inputs, scenario-level outputs,
parameter ledger, manuscript source, and a self-checking reproduction path for:

> **Transition-Aware Closed-Loop Restoration for Mobility-Coupled Distribution
> Systems with Measured Charging and Executable Actions**

Manuscript number: **TSG-01697-2026**.

## Main result and scientific positioning

Across 4,096 paired scenarios, transition-aware central PC has 0.6327% lower
mean dimensionless cost than the forecast-matched no-transition baseline
(paired difference -6.2730; 20,000-resample bootstrap 95% interval
[-7.0372, -5.5195]).

The negative robustness result is retained. Direct comparison shows that the
21-matrix robust extension is 0.1708% worse in mean cost than central PC,
increases CVaR95 by 4.48 and CVaR99 by 3.19 cost units, and does not reduce the
observed maximum cost. The manuscript therefore treats central PC as the main
method and robust PC as a sensitivity extension.

## What is included

- processed Boulder charging inputs and the fixed demand forecast;
- the selected SMART-DS/OpenDSS feeder used inside the execution loop;
- EAGLE-I-derived event summaries and packet-response inputs;
- scenario-level results for all 4,096 paired scenarios and sensitivity runs;
- analysis, simulation, plotting, and verification code;
- a machine-readable complete parameter ledger;
- clean and line-numbered manuscript sources and PDFs; and
- `SHA256SUMS.csv`, which fixes the byte count and SHA-256 digest of every
  released file.

The original 11.68 GB EAGLE-I archive and other publicly downloadable raw data
are not duplicated here. `data/external/THIRD_PARTY_DATA_LICENSES.md` and the
metadata manifests preserve their original source URLs, licences, versions,
access records, and identifiers. The supplied scenario rows are sufficient to
regenerate the principal statistical results without HPC access.

## Environment

The checked environment uses Python 3.13.7 and the pinned packages in
`requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Verify the released snapshot

From the repository root:

```powershell
python src\verify_reviewer_minimal.py
```

The verifier checks every file in `SHA256SUMS.csv`, the 4,096-by-six paired
scenario structure, the central-PC primary effect, and the direct
robust-versus-central risk result.

## Regenerate the principal statistics

```powershell
python src\paired_statistics.py `
  --scenarios results\operational\boulder_robust_rollout\rollout_scenarios.csv `
  --output results\operational\boulder_robust_rollout\paired_statistics_reproduced.csv `
  --bootstrap-replicates 20000 --seed 41073 --policy pc_rollout

python src\objective_weight_sensitivity.py `
  --scenarios results\operational\boulder_robust_rollout\rollout_scenarios.csv `
  --output results\operational\boulder_robust_rollout\objective_weight_sensitivity_reproduced.csv `
  --policy pc_rollout --comparators forecast_matched `
  --bootstrap-replicates 20000 --seed 41073

python src\robustness_positioning_analysis.py `
  --output results\operational\boulder_robust_rollout\robust_vs_central_risk_reproduced.csv `
  --bootstrap-replicates 20000 --seed 20260901
```

## Run a short executable closed loop

```powershell
python -u src\simulate_rollout_revised.py `
  --results results\reviewer_smoke_local --scenarios 16 --workers 2
```

Every policy-hour invokes OpenDSS. Packet delivery gates remote actions and
work-order dispatch; only integer crew-completion events clear realized threat.
The full primary command and the evidence boundaries are documented in
`REVISION_REPRODUCIBILITY.md`.

## Policy definitions

- **Forecast-matched** uses the common demand forecast, current threat, and
  backlog, but no propagation matrix or multi-step transition rollout.
- **Central PC** scores the common route portfolio with the calibrated central
  transition matrix.
- **Robust PC** scores the same portfolio by worst predicted loss over the
  declared 21-matrix set.
- Only the oracle receives realized future demand and transition innovations.
- All six policies pass through the same packet, OpenDSS, station-flow, and
  integer-crew execution gates within a paired scenario.

## Version and identifier policy

The submission snapshot is tagged `v2026.09.01`. No separate repository DOI is
required or claimed; reused public datasets retain their existing identifiers.
