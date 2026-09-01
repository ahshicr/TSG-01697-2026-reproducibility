# Reproducing the transition-aware-PC revision

Run all commands from the repository root. The fastest audited route uses the
released scenario-level rows; it does not require the 11.68 GB raw EAGLE-I
archive, a GPU, or HPC access.

## 1. Install the checked environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Check file integrity and headline numerical contracts

```powershell
python src\verify_reviewer_minimal.py
```

This checks `SHA256SUMS.csv`, all 4,096 paired scenarios and six policies per
scenario, the 0.6327% central-PC primary effect, and the stored
robust-versus-central risk result.

## 3. Regenerate the principal statistical outputs

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

The primary inferential family is central PC versus forecast-matched for total
cost, overall and within four prespecified threat classes. The analysis uses
paired mean effects, 20,000 paired bootstrap resamples, paired t tests,
two-sided Wilcoxon tests, and Holm correction.

## 4. Run the executable loop

Short smoke test:

```powershell
python -u src\simulate_rollout_revised.py `
  --results results\reviewer_smoke_local --scenarios 16 --workers 2
```

Full primary run:

```powershell
python -u src\simulate_rollout_revised.py `
  --results results\operational\boulder_robust_rollout_reproduced `
  --scenarios 4096 --workers 8
```

The released main run used seed 2026. The scenario seed is independent of the
worker count. Online SMART-DS projection and four routed integer crews are on by
default. Packet delivery gates remote actions and work-order dispatch; a threat
state changes only after a logged crew-completion event.

## 5. Evidence boundaries

- Boulder arrivals and session energy are measured, while the hourly
  within-session load profile is reconstructed and conservation checked.
- EAGLE-I informs outage persistence and recovery context; it is not treated as
  station-failure labels or observed utility crew records.
- SMART-DS is an independent synthetic SFO feeder, not the Boulder utility
  network. It validates charging-action feasibility, not switching, fault
  isolation, topology reconfiguration, or equipment repair.
- Packet traffic, station capacity, backup duration, and unidentified
  transition directions are declared constructed or sensitivity-only factors.
- The relative policy benefit remains an estimate in a model-based transition
  environment, even though physical feasibility is checked independently.

## 6. Public raw-data provenance

The selected SMART-DS feeder and the processed inputs required by the released
loop are included. The larger public raw files are not duplicated. Their source
URLs, licences, versions, access records, and hashes are retained in
`data/external/THIRD_PARTY_DATA_LICENSES.md` and `data/external/metadata/`.
