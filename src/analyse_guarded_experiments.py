"""Predeclared selection and paired inference for baseline protection.

Validation selects only the margin. Test inference retains both route groups,
all conditions and negative outcomes. Cross-condition composites resample
scenario identifiers, preserving the shared random conditions within each ID.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import stats

import guarded_route_scoring as guard
import run_guarded_experiments as runner
from paired_statistics import bootstrap_mean_ci, holm_adjust, wilcoxon_p

BOOTSTRAPS = 20000
TAIL_BOOTSTRAPS = 2000
STAT_RANDOM_STATE = 202609055
GROUPS = ('published', 'independent')
SECONDARY = ('unserved_energy', 'crew_completion_fraction', 'crew_mean_completion_h',
             'crew_total_travel_h', 'smartds_curtailed_energy_kwh')


def load(root, conditions, phase):
    frames, records = {}, {}
    for condition in conditions:
        record = json.loads((root/condition/'completion.json').read_text(encoding='utf-8'))
        if record['phase'] != phase:
            raise RuntimeError('Wrong experiment phase')
        source = root/condition/'rollout_scenarios.csv'
        assert runner.digest(source) == record['scenario_sha256']
        frames[condition] = pd.read_csv(source)
        records[condition] = record
    assert all(record['source_sha256'] == next(iter(records.values()))['source_sha256']
               for record in records.values())
    return frames, records


def array(frame, policy, metric='cost'):
    part = frame.loc[frame.policy.eq(policy)].sort_values('scenario_id')
    if part.scenario_id.duplicated().any() or part.empty:
        raise RuntimeError('Missing or duplicated policy rows')
    return part[metric].to_numpy(float)


def select(root):
    destination = root/'method_decision.json'
    if destination.exists():
        raise RuntimeError('Preserve the locked validation choice')
    frames, records = load(root, runner.VALIDATION_CONDITIONS, 'validation')
    assert all(record['scenarios'] == 512 for record in records.values())
    decisions, gates, all_options = {}, {}, []
    for group in GROUPS:
        baseline = array(frames['primary'], group+'/matched')
        central = array(frames['primary'], group+'/central')
        options = []
        for rate in guard.RATES:
            policy = group+f'/protected_{rate:g}'
            proposed = array(frames['primary'], policy)
            primary_penalty = float((proposed-central).mean()/baseline.mean())
            excess = []
            for condition in runner.VALIDATION_CONDITIONS[1:]:
                a = array(frames[condition], policy)
                b = array(frames[condition], group+'/matched')
                excess.append(float(np.maximum(a-b, 0).mean()))
            options.append(dict(group=group, rate=rate, primary_relative_penalty=primary_penalty,
                primary_screen_pass=bool(primary_penalty <= .0005),
                equal_condition_positive_excess=float(np.mean(excess)), per_condition_excess=excess))
        eligible = [row for row in options if row['primary_screen_pass']]
        best = min(eligible or options, key=lambda row:(row['equal_condition_positive_excess'], row['rate']))
        decisions[group] = best['rate']
        gates[group] = bool(eligible)
        all_options.extend(options)
    sim_sources = next(iter(records.values()))['source_sha256']
    output = dict(selected_rates=decisions, validation_screen_pass=gates,
        source_sha256=sim_sources, protocol_sha256=runner.PROTOCOL_SHA,
        selection_script_sha256=runner.digest(__file__), frozen_unix=time.time(),
        validation_records={name:runner.digest(root/name/'completion.json') for name in records},
        all_margin_options=all_options,
        statistics_specification=dict(mean_bootstraps=BOOTSTRAPS, tail_bootstraps=TAIL_BOOTSTRAPS,
            random_state=STAT_RANDOM_STATE, family='All declared test cost comparisons receive Holm correction.',
            composite='Average six condition-specific positive-excess differences within each paired scenario ID.',
            adoption='Per-group validation screen, primary noninferiority, positive-excess superiority and external check.'),
        test_rule='No post-test margin, objective, model, route-rule or condition tuning.')
    destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


def mean_record(a, b):
    delta = a-b
    if len(a) != len(b) or not np.isfinite(delta).all():
        raise RuntimeError('Nonfinite or unpaired comparison')
    low, high = bootstrap_mean_ci(delta, STAT_RANDOM_STATE, BOOTSTRAPS)
    p = float(stats.ttest_rel(a, b).pvalue) if np.std(delta) > 0 else 1.0
    return dict(n=len(delta), proposed_mean=float(a.mean()), reference_mean=float(b.mean()),
        difference=float(delta.mean()), ci_low=low, ci_high=high,
        reduction_percent=float(-100*delta.mean()/b.mean()) if b.mean() != 0 else 0.0,
        paired_t_p=p, wilcoxon_p=wilcoxon_p(delta))


def empirical_tail(values, fraction):
    """Upper expected shortfall with fractional boundary mass, including ties."""
    values = np.sort(np.asarray(values, dtype=float))
    mass = (1-fraction)*len(values)
    whole = int(np.floor(mass+1e-12))
    remainder = mass-whole
    total = values[len(values)-whole:].sum() if whole else 0.0
    if remainder > 1e-12:
        total += remainder*values[len(values)-whole-1]
    return float(total/mass)


def tail_record(a, b, metric):
    functions = {'q95':lambda x:float(np.quantile(x,.95)),
                 'q99':lambda x:float(np.quantile(x,.99)),
                 'es95':lambda x:empirical_tail(x,.95),
                 'maximum':lambda x:float(np.max(x))}
    function = functions[metric]
    rng = np.random.default_rng(STAT_RANDOM_STATE)
    samples = np.empty(TAIL_BOOTSTRAPS)
    for i in range(TAIL_BOOTSTRAPS):
        indices = rng.integers(len(a), size=len(a))
        samples[i] = function(a[indices])-function(b[indices])
    low, high = np.quantile(samples, [.025,.975])
    return dict(n=len(a), proposed=function(a), reference=function(b),
        difference=function(a)-function(b), ci_low=float(low), ci_high=float(high),
        scope='Paired marginal bootstrap interval, exploratory tail endpoint')


def analyse(root, frozen_path, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'completion.json').exists():
        raise RuntimeError('Preserve the previous inference')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if frozen['selection_script_sha256'] != runner.digest(__file__):
        raise RuntimeError('Inference specification changed after validation')
    frames, records = load(root, runner.TEST_CONDITIONS, 'test')
    if any(record['source_sha256'] != frozen['source_sha256'] for record in records.values()):
        raise RuntimeError('Test source differs from validation freeze')
    means, tails, secondary, diagnostics, gates = [], [], [], [], []
    for condition, frame in frames.items():
        for group in GROUPS:
            protected = f'protected_{frozen["selected_rates"][group]:g}'
            comparisons = [('central','matched'), ('robust','central'),
                           (protected,'matched'), (protected,'central')]
            if group == 'independent':
                comparisons += [('observed','matched'),('observed','central')]
            for policy, comparator in comparisons:
                a, b = array(frame,group+'/'+policy), array(frame,group+'/'+comparator)
                label = dict(condition=condition, group=group, policy=policy, comparator=comparator)
                means.append({**label, **mean_record(a,b)})
                if policy in (protected, 'observed'):
                    for metric in SECONDARY:
                        secondary.append({**label, 'metric':metric,
                            **mean_record(array(frame,group+'/'+policy,metric),array(frame,group+'/'+comparator,metric))})
                if policy == protected and comparator == 'central':
                    for metric in ('q95','q99','es95','maximum'):
                        tails.append({**label,'metric':metric,**tail_record(a,b,metric)})
            baseline = array(frame,group+'/matched')
            for policy in ('matched', 'central', 'robust', protected)+(('observed',) if group == 'independent' else ()):
                part = frame.loc[frame.policy.eq(group+'/'+policy)]
                a = array(frame,group+'/'+policy)
                delta = a-baseline
                diagnostics.append(dict(condition=condition, group=group, policy=policy,
                    mean_cost=float(a.mean()), q95=float(np.quantile(a,.95)),
                    q99=float(np.quantile(a,.99)), es95=empirical_tail(a,.95), maximum_cost=float(a.max()),
                    positive_excess=float(np.maximum(delta,0).mean()),
                    worse_fraction=float((delta>1e-8).mean()),
                    max_excess=float(np.maximum(delta,0).max()),
                    guard_acceptance=float(part.guard_accepted.mean()) if policy == protected else 0.0,
                    infeasible_hours=int(part.smartds_projected_infeasible_hours.sum())))
    for row, p in zip(means, holm_adjust([row['paired_t_p'] for row in means])):
        row['holm_paired_t'] = p
    for row, p in zip(means, holm_adjust([row['wilcoxon_p'] for row in means])):
        row['holm_wilcoxon'] = p
    for group in GROUPS:
        policy = f'{group}/protected_{frozen["selected_rates"][group]:g}'
        primary = frames['primary']
        # A sample-wise relative margin accounts for the reference mean in the uncertainty.
        ni = array(primary,policy)-array(primary,group+'/central')-.0005*array(primary,group+'/matched')
        low, high = bootstrap_mean_ci(ni,STAT_RANDOM_STATE,BOOTSTRAPS)
        # 97.5% upper bootstrap bound is Bonferroni one-sided 95% for two route groups.
        composites = []
        for condition in runner.VALIDATION_CONDITIONS[1:]:
            frame = frames[condition]
            f = array(frame,group+'/matched')
            composites.append(np.maximum(array(frame,policy)-f,0)-np.maximum(array(frame,group+'/central')-f,0))
        composite = np.stack(composites).mean(axis=0)
        record = mean_record(composite,np.zeros_like(composite))
        gates.append(dict(group=group,validation_screen_pass=frozen['validation_screen_pass'][group],
            primary_noninferiority_mean=float(ni.mean()),primary_noninferiority_upper=high,
            primary_noninferiority_pass=bool(high<0),positive_excess_difference=record['difference'],
            positive_excess_ci_low=record['ci_low'],positive_excess_ci_high=record['ci_high'],
            positive_excess_p=record['paired_t_p'],external_gate='PENDING_INDEPENDENT_INPUT_EVALUATION'))
    for row, adjusted in zip(gates,holm_adjust([row['positive_excess_p'] for row in gates])):
        row['positive_excess_holm_p'] = adjusted
        row['positive_excess_pass'] = bool(row['positive_excess_difference']<0 and adjusted<.05
                                          and row['positive_excess_ci_high']<0)
    for name, rows in [('paired_costs',means),('tails',tails),('secondary',secondary),('diagnostics',diagnostics)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    result = dict(status='STATISTICS_COMPLETE_EXTERNAL_ADOPTION_PENDING',gates=gates,
        cost_comparisons=len(means),tail_comparisons=len(tails),secondary_comparisons=len(secondary),
        frozen_sha256=runner.digest(frozen_path),script_sha256=runner.digest(__file__),
        source_records={name:runner.digest(root/name/'completion.json') for name in records},
        outputs={name:runner.digest(output/(name+'.csv')) for name in ('paired_costs','tails','secondary','diagnostics')})
    (output/'completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('select','test'),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,default=Path('results/guarded_validation_20260905/method_decision.json'))
    parser.add_argument('--output',type=Path,default=Path('results/guarded_test_20260905/inference'))
    args=parser.parse_args()
    if args.phase=='select':
        select(args.root)
    else:
        analyse(args.root,args.freeze,args.output)


if __name__=='__main__':
    main()
