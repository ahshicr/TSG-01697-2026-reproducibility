#!/usr/bin/env python3
"""Read-only verification of a frozen revision's files and numerical results.

This checks the declared snapshot and recomputes its paired statistical tables.
It does not claim to prove mathematics, field validity, or submission readiness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_frozen_service_experiments import GROUPS, PHYSICAL, comparison, cvar, paired
from paired_statistics import holm_adjust

CONDITIONS = [
    'primary','electrical_stress','backup_0','backup_60','backup_300',
    'matrix_scale_075','matrix_scale_125','matrix_noise_015','crews_12','crews_24',
    'zero_comm_persistence','zero_pc_coupling','zero_rc_coupling',
    'zero_cr_coupling','zero_cp_coupling','zero_all_unsupported',
    'expanded_unsupported','nonlinear_ood','weight_mobility_x2',
    'weight_energy_x2','weight_communication_x2','weight_power_x2','weight_cascade_half',
]
FAMILIES = {'primary_cost_15':15, 'physical_secondary_9':9,
            'reference_subset_descriptive_1':1, 'sensitivity_cost':45, 'backup_effect_2':2}


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual,expected,label):
    require(np.allclose(actual,expected,rtol=1e-10,atol=1e-9,equal_nan=True),label)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-only',action='store_true',
                        help='Working-tree numerical checks only, without asserting archive completeness.')
    args=parser.parse_args()
    root=args.package_root.resolve()
    verified_files=0
    if not args.data_only:
        with (root/'SHA256SUMS.csv').open(encoding='utf-8-sig',newline='') as handle:
            rows=list(csv.DictReader(handle))
        require(bool(rows),'Empty release digest list')
        require(len({r['path'] for r in rows})==len(rows),'Duplicate archive path')
        for row in rows:
            path=(root/row['path']).resolve()
            require(path.is_relative_to(root),'Archive path outside package')
            require(path.is_file(),'Missing '+row['path'])
            require(path.stat().st_size==int(row['bytes']),'Size changed '+row['path'])
            require(sha256(path)==row['sha256'],'Content changed '+row['path'])
        verified_files=len(rows)

    result_root=root/'results/submission_service_20260905'
    inference=result_root/'inference'
    manifest=json.loads((inference/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['full_condition_set_complete'],'Incomplete condition set')
    require(set(manifest['finished_conditions'])==set(CONDITIONS),'Changed declared conditions')
    require(manifest['families']==FAMILIES,'Changed comparison families')
    decision_path=root/'results/service_score_validation_20260905/method_decision.json'
    decision=json.loads(decision_path.read_text(encoding='utf-8'))
    forecast_root=root/'results/real_ev_strict_20260905'
    metrics=pd.read_csv(forecast_root/'prediction_metrics.csv')
    selected=metrics.loc[metrics.best_val_score.idxmin()]
    forecast=forecast_root/f'forecast_{selected["mode"]}_seed{int(selected["seed"])}.npz'
    sources={name:root/'src'/name for name in [
        'run_frozen_service_experiments.py','simulate_rollout_revised.py',
        'service_route_scoring.py','smartds_ev_feasibility.py']}
    sources.update({
        'transition_parameter_uncertainty.csv':root/'results/calibration_strict_20260905/transition_parameter_uncertainty.csv',
        'packet_network_scenarios.csv':root/'results/packet_training_20260905/packet_network_scenarios.csv',
        forecast.name:forecast})
    current={name:sha256(path) for name,path in sources.items()}
    for name,digest in decision['locked_source_sha256'].items():
        require(current.get(name)==digest,'Changed frozen method/input '+name)

    frames={}
    for condition in CONDITIONS:
        folder=result_root/condition
        completed=json.loads((folder/'completion.json').read_text(encoding='utf-8'))
        path=folder/'rollout_scenarios.csv'
        require(completed['source_sha256']==current,'Different generating source '+condition)
        require(completed['design_freeze_sha256']==sha256(decision_path),'Different method decision '+condition)
        require(completed['started_unix']>=decision['frozen_unix'],'Test precedes freeze '+condition)
        require(sha256(path)==completed['scenario_sha256']==manifest['condition_records'][condition],
                'Scenario digest mismatch '+condition)
        frame=pd.read_csv(path)
        expected=4096 if condition=='primary' else 1024
        require(frame.scenario_id.nunique()==expected,'Scenario count '+condition)
        require(len(frame)==completed['rows']==expected*len(completed['policies']),'Policy rows '+condition)
        require(not frame.duplicated(['scenario_id','policy']).any(),'Duplicate pair '+condition)
        require(set(frame.policy)==set(completed['policies']),'Policy coverage '+condition)
        require(frame.groupby('scenario_id').policy.nunique().eq(len(completed['policies'])).all(),
                'Unpaired policy coverage '+condition)
        shared=['group','first_hour','smartds_mapping_index']+[c for c in frame if c.startswith('true_')]
        require(frame.groupby('scenario_id')[shared].nunique(dropna=False).eq(1).all().all(),
                'Different paired inputs '+condition)
        require(frame.smartds_projected_infeasible_hours.eq(0).all(),'Projected failure '+condition)
        require((frame.crew_jobs_completed<=frame.crew_jobs_dispatched).all() and
                (frame.crew_jobs_dispatched<=frame.crew_job_count).all(),'Crew count ordering '+condition)
        same(frame.crew_completion_fraction,
             frame.crew_jobs_completed/frame.crew_job_count.clip(lower=1),
             'Requested-job completion denominator '+condition)
        frames[condition]=frame

    from verify_electrical_precision import physical_frame
    physical, precision_metadata = physical_frame(root, frames['primary'])
    require(manifest.get('physical_measurement') == precision_metadata, 'Electrical measurement provenance differs')
    recorded=pd.read_csv(inference/'paired_comparisons.csv')
    require(recorded.groupby('family').size().to_dict()==FAMILIES,'Statistical row count')
    keys=['condition','policy','comparator','metric','group']
    require(not recorded.duplicated(keys).any(),'Duplicate statistical comparison')
    recomputed=[]
    for row in recorded.itertuples():
        random_state=41073
        if row.family=='primary_cost_15':
            random_state+=1009*GROUPS.index(row.group)
        elif row.family=='physical_secondary_9':
            random_state=91073+9176*PHYSICAL.index(row.metric)
        if row.family=='physical_secondary_9':
            a,b=paired(physical,row.policy,row.comparator,row.metric,row.group)
        elif row.condition=='reference_1024':
            frame=frames['primary'].loc[lambda f:f.scenario_id.lt(1024)]
            a,b=paired(frame,row.policy,row.comparator,row.metric,row.group)
        elif row.family=='backup_effect_2':
            a=frames[row.condition].query('policy == "pc_rollout"').sort_values('scenario_id')
            b=frames['backup_0'].query('policy == "pc_rollout"').sort_values('scenario_id')
            require(np.array_equal(a.scenario_id,b.scenario_id) and np.array_equal(a.first_hour,b.first_hour),
                    'Unpaired backup comparison')
            a,b=a.cost.to_numpy(),b.cost.to_numpy()
        else:
            a,b=paired(frames[row.condition],row.policy,row.comparator,row.metric,row.group)
        values=comparison(a,b,manifest['bootstrap_replicates'],random_state)
        for key,value in values.items():
            same(getattr(row,key),value,'Recomputed '+row.condition+' '+row.metric+' '+key)
        recomputed.append(values)
    for family in FAMILIES:
        indices=recorded.index[recorded.family.eq(family)].tolist()
        for field,raw in [('holm_t','paired_t_p'),('holm_wilcoxon','wilcoxon_p')]:
            adjusted=holm_adjust([recomputed[i][raw] for i in indices])
            same(recorded.loc[indices,field],adjusted,'Multiplicity correction '+family+' '+field)
    risks=pd.read_csv(inference/'cost_risk_summary.csv')
    for row in risks.itertuples():
        values=frames[row.condition].loc[lambda f:f.policy.eq(row.policy),'cost'].to_numpy()
        expected=[values.mean(),np.quantile(values,.95),np.quantile(values,.99),
                  cvar(values,.95),cvar(values,.99),values.max()]
        same([getattr(row,k) for k in ['mean','q95','q99','cvar95','cvar99','maximum']],expected,
             'Empirical cost distribution '+row.condition+' '+row.policy)
    print(json.dumps({'status':'PASS','archive_digests_checked':not args.data_only,
        'hashed_files':verified_files,'conditions':len(frames),'statistical_comparisons':len(recorded),
        'primary_rows':len(frames['primary']),
        'scope':'Frozen source correspondence, paired records, recomputed statistics and empirical tails. '
                'Not a mathematical proof, field validation or submission clearance.'},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
