"""Paired inference with explicit comparison families and exact scenario keys."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paired_statistics import bootstrap_mean_ci, holm_adjust, wilcoxon_p


GROUPS = ['all', 'nominal', 'single_domain', 'cascade', 'ood']
PHYSICAL = ['unserved_energy','losses_mwh','min_voltage_pu','mean_control_action_fraction',
            'mean_smartds_projection_fraction','smartds_curtailed_energy_kwh',
            'crew_completion_fraction','crew_mean_completion_h','crew_total_travel_h']


def paired(frame, policy, comparator, metric, group='all'):
    selected = frame if group=='all' else frame[frame.group.eq(group)]
    selected = selected[selected.policy.isin([policy,comparator])]
    if selected.duplicated(['scenario_id','policy']).any():
        raise ValueError('Duplicate scenario-policy pair')
    pivot = selected.pivot(index='scenario_id',columns='policy',values=metric).sort_index()
    if pivot.empty or pivot[[policy,comparator]].isna().any().any():
        raise ValueError(f'Incomplete pairing for {group} {policy} {comparator} {metric}')
    return pivot[policy].to_numpy(float), pivot[comparator].to_numpy(float)


def comparison(a,b,replicates,random_state=41073):
    delta = np.asarray(a)-np.asarray(b)
    if np.ptp(delta)==0:
        low=high=float(delta[0])
        t_p=1. if delta[0]==0 else 0.
    else:
        low,high=bootstrap_mean_ci(delta,random_state,replicates)
        t_p=float(stats.ttest_rel(a,b).pvalue)
    reference=float(np.mean(b))
    return dict(n_pairs=len(delta),policy_mean=float(np.mean(a)),comparator_mean=reference,
        mean_difference=float(delta.mean()),ci95_low=low,ci95_high=high,
        reduction_percent=-100*float(delta.mean())/reference if abs(reference)>1e-12 else np.nan,
        paired_t_p=t_p,wilcoxon_p=wilcoxon_p(delta))


def adjust(rows,family):
    selected=[row for row in rows if row['family']==family]
    for row,t,w in zip(selected,holm_adjust([r['paired_t_p'] for r in selected]),
                       holm_adjust([r['wilcoxon_p'] for r in selected])):
        row.update(holm_t=t,holm_wilcoxon=w,family_size=len(selected))


def cvar(values,q):
    descending=np.sort(values)[::-1]
    count=len(values)*(1-q)
    whole=int(np.floor(count))
    fraction=count-whole
    total=descending[:whole].sum()
    if fraction>0:
        total+=fraction*descending[whole]
    return float(total/count)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/submission_service_20260905'))
    parser.add_argument('--replicates',type=int,default=20000)
    parser.add_argument('--complete-only',action='store_true',help='Analyse finished conditions without claiming all conditions are complete.')
    args=parser.parse_args()
    from run_frozen_service_experiments import CONDITIONS
    missing=[name for name in CONDITIONS if not (args.root/name/'completion.json').exists()]
    if missing and not args.complete_only:
        raise RuntimeError('Experiments unfinished: '+', '.join(missing))
    frames={}
    manifests={}
    for name in CONDITIONS:
        marker=args.root/name/'completion.json'
        if not marker.exists():
            continue
        manifest=json.loads(marker.read_text(encoding='utf-8'))
        path=marker.with_name('rollout_scenarios.csv')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['scenario_sha256']
        frame=pd.read_csv(path)
        assert len(frame)==manifest['rows']
        assert not frame.duplicated(['scenario_id','policy']).any()
        assert frame.groupby('scenario_id').policy.nunique().eq(len(manifest['policies'])).all()
        assert frame.groupby('scenario_id')[['group','first_hour','smartds_mapping_index']].nunique().eq(1).all().all()
        assert frame.smartds_projected_infeasible_hours.eq(0).all()
        frames[name],manifests[name]=frame,manifest
    if 'primary' not in frames:
        raise RuntimeError('Primary experiment must finish first')
    rows=[]
    primary=frames['primary']
    for comparator in ['forecast_matched','exposure_matched','exposure_central']:
        for group in GROUPS:
            a,b=paired(primary,'pc_rollout',comparator,'cost',group)
            rows.append(dict(condition='primary',policy='pc_rollout',comparator=comparator,
                metric='cost',group=group,family='primary_cost_15',
                **comparison(a,b,args.replicates,41073+1009*GROUPS.index(group))))
    from verify_electrical_precision import physical_frame
    physical, precision_metadata = physical_frame(Path(__file__).resolve().parents[1], primary)
    for metric in PHYSICAL:
        a,b=paired(physical,'pc_rollout','forecast_matched',metric)
        rows.append(dict(condition='primary',policy='pc_rollout',comparator='forecast_matched',
            metric=metric,group='all',family='physical_secondary_9',
            **comparison(a,b,args.replicates,91073+9176*PHYSICAL.index(metric))))
    # The same prefix supplies the reference for the 1024-pair weight runs.
    # It is not a second independent confirmation of the primary result.
    a,b=paired(primary[primary.scenario_id.lt(1024)],'pc_rollout','forecast_matched','cost')
    rows.append(dict(condition='reference_1024',policy='pc_rollout',comparator='forecast_matched',
        metric='cost',group='all',family='reference_subset_descriptive_1',
        **comparison(a,b,args.replicates)))
    for name,frame in frames.items():
        comparisons=[('robust_pc_rollout','pc_rollout')]
        if name!='primary':
            comparisons.insert(0,('pc_rollout','forecast_matched'))
        for policy,comparator in comparisons:
            a,b=paired(frame,policy,comparator,'cost')
            rows.append(dict(condition=name,policy=policy,comparator=comparator,
                metric='cost',group='all',family='sensitivity_cost',
                **comparison(a,b,args.replicates)))
    if all(name in frames for name in ['backup_0','backup_60','backup_300']):
        for name in ['backup_60','backup_300']:
            a=frames[name].query('policy == "pc_rollout"').sort_values('scenario_id')
            b=frames['backup_0'].query('policy == "pc_rollout"').sort_values('scenario_id')
            assert np.array_equal(a.scenario_id,b.scenario_id) and np.array_equal(a.first_hour,b.first_hour)
            rows.append(dict(condition=name,policy='pc_rollout',comparator='same_policy_no_backup',
                metric='cost',group='all',family='backup_effect_2',
                **comparison(a.cost.to_numpy(),b.cost.to_numpy(),args.replicates)))
    for family in sorted({r['family'] for r in rows}):
        adjust(rows,family)
    output=args.root/'inference'
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'paired_comparisons.csv',index=False)
    risks=[]
    for name,frame in frames.items():
        for policy,values in frame.groupby('policy'):
            cost=values.cost.to_numpy(float)
            risks.append(dict(condition=name,policy=policy,n=len(cost),mean=float(cost.mean()),
                q95=float(np.quantile(cost,.95)),q99=float(np.quantile(cost,.99)),
                cvar95=cvar(cost,.95),cvar99=cvar(cost,.99),maximum=float(cost.max())))
    pd.DataFrame(risks).to_csv(output/'cost_risk_summary.csv',index=False)
    execution=[]
    runtime=[]
    for name,frame in frames.items():
        for policy,part in [('all_policies',frame), *list(frame.groupby('policy'))]:
            execution.append(dict(condition=name,policy=policy,
                n_scenarios=part.scenario_id.nunique(),policy_scenario_rows=len(part),
                policy_hours=len(part)*manifests[name]['horizon'],
                raw_infeasible_hours=float(part.smartds_raw_infeasible_hours.sum()),
                projected_infeasible_hours=float(part.smartds_projected_infeasible_hours.sum()),
                mean_projection_fraction=float(part.mean_smartds_projection_fraction.mean()),
                curtailed_energy_kwh=float(part.smartds_curtailed_energy_kwh.sum()),
                requested_jobs=float(part.crew_job_count.sum()),
                dispatched_jobs=float(part.crew_jobs_dispatched.sum()),
                completed_jobs=float(part.crew_jobs_completed.sum()),
                mean_requested_job_completion_fraction=float(part.crew_completion_fraction.mean()),
                mean_completion_h=float(part.crew_mean_completion_h.mean()),
                mean_total_travel_h=float(part.crew_total_travel_h.mean()),
                mean_timely_action_fraction=float(part.mean_control_action_fraction.mean())))
            runtime.append(dict(condition=name,policy=policy,
                mean_decision_ms=float(part.latency_ms.mean()),
                median_decision_ms=float(part.latency_ms.median()),
                q25_decision_ms=float(part.latency_ms.quantile(.25)),
                q75_decision_ms=float(part.latency_ms.quantile(.75))))
    pd.DataFrame(execution).to_csv(output/'execution_summary.csv',index=False)
    pd.DataFrame(runtime).to_csv(output/'runtime_descriptive.csv',index=False)
    report=dict(finished_conditions=list(frames),unfinished_conditions=missing,
        full_condition_set_complete=not missing,bootstrap_replicates=args.replicates,
        families={name:sum(r['family']==name for r in rows) for name in sorted({r['family'] for r in rows})},
        confidence_scope='Monte Carlo scenario uncertainty conditional on the specified public data and simulation model.',
        tail_scope='Empirical marginal quantiles and fractional upper-tail means, not inferential tail superiority tests.',
        runtime_scope='Measured decision wall time per hour under concurrent execution, not packet delay and not a controlled runtime hypothesis test.',
        physical_measurement=precision_metadata,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        condition_records={name:manifest['scenario_sha256'] for name,manifest in manifests.items()})
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
