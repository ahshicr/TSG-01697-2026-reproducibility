"""Independent reconstruction of saved choices, paired outcomes and statistics.

This checker does not call the guard selector or the statistical analysis
helpers. It checks recorded computations, not universal mathematical claims.
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

ROOT = Path(__file__).resolve().parents[1]
RATES = (0., .00025, .0005, .001)
CONDITIONS = ('primary', 'matrix_scale_075', 'matrix_scale_125',
              'matrix_noise_015', 'expanded_unsupported',
              'zero_all_unsupported', 'nonlinear_ood', 'electrical_stress')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(a, b, atol=1e-9):
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=atol)


def bootstrap(delta):
    # A different chunk size preserves the same random stream and provides an
    # independently implemented exact reconstruction of the declared interval.
    random = np.random.default_rng(202609055)
    values = []
    for offset in range(0, 20000, 127):
        ids = random.integers(len(delta), size=(min(127, 20000-offset), len(delta)))
        values.extend(np.mean(delta[ids], axis=1))
    return np.quantile(values, [.025, .975])


def adjusted(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i:pvalues[i])
    out = np.zeros(len(pvalues))
    maximum = 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, min(1., pvalues[i]*(len(order)-rank)))
        out[i] = maximum
    return out


def records_check(directory, rates, frozen):
    completion = read(directory/'completion.json')
    assert sha(directory/'rollout_scenarios.csv') == completion['scenario_sha256']
    assert sha(directory/'candidate_scores.jsonl') == completion['scores_sha256']
    for path, expected in completion['source_sha256'].items():
        assert sha(ROOT/path) == expected, path
    frame = pd.read_csv(directory/'rollout_scenarios.csv', keep_default_na=False)
    assert len(frame) == completion['rows']
    assert not frame[['scenario_id','policy']].duplicated().any()
    expected_policies = {group+'/'+policy for group in ('published','independent')
        for policy in ['matched','central','robust']+[f'protected_{r:g}' for r in rates]
        +(['observed'] if group=='independent' else [])}
    assert set(frame.policy) == expected_policies
    assert frame.scenario_id.nunique() == completion['scenarios']
    assert frame.groupby('scenario_id').size().eq(len(expected_policies)).all()
    shared = ['group','first_hour','smartds_mapping_index','crew_job_count','crew_jobs_dispatched']
    shared += [c for c in frame if c.startswith('true_')]
    assert frame.groupby('scenario_id')[shared].nunique().eq(1).all().all()
    assert frame.smartds_projected_infeasible_hours.eq(0).all()
    assert (frame.executed_charge_capacity <= frame.requested_charge_capacity+1e-4).all()
    assert frame.crew_jobs_completed.le(frame.crew_jobs_dispatched).all()
    assert frame.crew_jobs_dispatched.le(frame.crew_job_count).all()
    indexed = frame.set_index(['scenario_id','policy'])
    execution = list(frame.columns[frame.columns.get_loc('cost'):frame.columns.get_loc('crew_routes')+1])
    counts = dict(scenarios=completion['scenarios'], rows=len(frame), choices=0, accepted=0, fallback=0)
    identifiers = set()
    with (directory/'candidate_scores.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            identifier = record['scenario_id']
            assert identifier not in identifiers
            identifiers.add(identifier)
            for group, detail in record['candidate_scores'].items():
                names = detail['candidate_names']
                assert names == sorted(names) and len(names) == 6
                matched = np.asarray(detail['matched_scores'])
                center = np.asarray(detail['central_scores'])
                matrices = np.asarray(detail['matrix_scores'])
                assert matrices.shape == (21,6) and np.isfinite(matrices).all()
                reference = int(np.argmin(matched))
                assert reference == detail['reference']
                base = indexed.loc[(identifier, group+'/matched')]
                assert base.selected_candidate == names[reference]
                for policy, index in [('central',np.argmin(center)),('robust',np.argmin(matrices.max(axis=0)))]:
                    row = indexed.loc[(identifier,group+'/'+policy)]
                    assert row.selected_candidate == names[int(index)]
                bounds = np.max(matrices-matrices[:,reference,None], axis=0)
                assert bounds[reference] == 0.
                candidate = int(np.argmin(bounds))
                tolerance = 1e-10*max(1., np.abs(matrices).max())
                for rate in rates:
                    margin = rate*max(1., abs(center[reference]))
                    accepted = bounds[candidate] < -margin-tolerance
                    selected = candidate if accepted else reference
                    row = indexed.loc[(identifier,group+f'/protected_{rate:g}')]
                    assert row.selected_candidate == names[selected]
                    assert row.reference_candidate == names[reference]
                    assert row.guard_accepted == int(accepted)
                    close([row.guard_margin,row.guard_tolerance,row.guard_upper_difference],
                          [margin,tolerance,bounds[selected]])
                    counts['choices'] += 1
                    if accepted:
                        counts['accepted'] += 1
                        assert np.all(matrices[:,selected]-matrices[:,reference] < -margin-tolerance)
                    else:
                        counts['fallback'] += 1
                        assert row[execution].equals(base[execution]), (directory,identifier,group)
    assert len(identifiers) == completion['scenarios']
    if completion['phase'] != 'validation':
        assert completion['started_unix'] > frozen['frozen_unix']
        assert set(rates) == set(frozen['selected_rates'].values())
    return counts, frame


def infer_check(directory, frames, external=False):
    result = read(directory/'completion.json')
    for name, value in result['outputs'].items():
        assert sha(directory/(name+'.csv')) == value
    table = pd.read_csv(directory/'paired_costs.csv')
    pvalues, ranks = [], []
    primary_intervals = 0
    for row in table.itertuples():
        key = (row.forecast+'/' if external else '')+row.condition
        wide = frames[key].pivot(index='scenario_id', columns='policy', values='cost').sort_index()
        a = wide[row.group+'/'+row.policy].to_numpy(float)
        b = wide[row.group+'/'+row.comparator].to_numpy(float)
        d = a-b
        close([a.mean(),b.mean(),d.mean(),-100*d.mean()/b.mean()],
              [row.proposed_mean,row.reference_mean,row.difference,row.reduction_percent])
        p = float(stats.ttest_1samp(d,0.).pvalue) if np.std(d)>0 else 1.
        nz = d[np.abs(d)>1e-12]
        w = float(stats.wilcoxon(nz,method='auto').pvalue) if len(nz) else 1.
        close([p,w], [row.paired_t_p,row.wilcoxon_p], atol=1e-13)
        pvalues.append(p)
        ranks.append(w)
        if row.condition == 'primary':
            close(bootstrap(d),[row.ci_low,row.ci_high])
            primary_intervals += 1
    close(adjusted(pvalues),table.holm_paired_t,atol=1e-12)
    close(adjusted(ranks),table.holm_wilcoxon,atol=1e-12)
    gates = result['external_gates'] if external else result['gates']
    composite_pvalues = []
    for gate in gates:
        prefix = gate['forecast']+'/' if external else ''
        wide = frames[prefix+'primary'].pivot(index='scenario_id',columns='policy',values='cost').sort_index()
        group = gate['group']
        shifted = (wide[group+'/protected_0.001']-wide[group+'/central']-.0005*wide[group+'/matched']).to_numpy(float)
        upper = bootstrap(shifted)[1]
        close(upper,gate['noninferiority_upper'] if external else gate['primary_noninferiority_upper'])
        assert (upper<0) == (gate['external_noninferiority_pass'] if external else gate['primary_noninferiority_pass'])
        if not external:
            differences=[]
            for condition in CONDITIONS[1:7]:
                wide=frames[condition].pivot(index='scenario_id',columns='policy',values='cost').sort_index()
                base=wide[group+'/matched'].to_numpy(float)
                differences.append(np.maximum(wide[group+'/protected_0.001'].to_numpy(float)-base,0)
                    -np.maximum(wide[group+'/central'].to_numpy(float)-base,0))
            difference=np.mean(differences,axis=0)
            close(difference.mean(),gate['positive_excess_difference'])
            close(bootstrap(difference),[gate['positive_excess_ci_low'],gate['positive_excess_ci_high']])
            composite_pvalues.append(float(stats.ttest_1samp(difference,0).pvalue))
    if not external:
        close(adjusted(composite_pvalues),[g['positive_excess_holm_p'] for g in gates],atol=1e-13)
    return dict(mean_comparisons=len(table),exact_primary_intervals=primary_intervals,
                all_mean_tests_and_Holm_recomputed=True,adoption_intervals_recomputed=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/guarded_verification_20260905.json'))
    args=parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve completed verification')
    start=time.time()
    frozen=read(ROOT/'results/guarded_validation_20260905/method_decision.json')
    report,frames={},{}
    studies=[('validation',ROOT/'results/guarded_validation_20260905',RATES,CONDITIONS[:7]),
        ('test',ROOT/'results/guarded_test_20260905',(.001,),CONDITIONS),
        ('graph',ROOT/'results/guarded_external_20260905/graph',(.001,),CONDITIONS),
        ('weekly',ROOT/'results/guarded_external_20260905/weekly',(.001,),CONDITIONS)]
    for phase,root,rates,conditions in studies:
        for condition in conditions:
            counts,frame=records_check(root/condition,rates,frozen)
            report[phase+'/'+condition]=counts
            if phase!='validation':
                frames[phase+'/'+condition]=frame
            print('Verified '+phase+'/'+condition,flush=True)
    old=pd.read_csv(ROOT/'results/submission_service_20260905/primary/rollout_scenarios.csv',keep_default_na=False)
    original=old.set_index(['scenario_id','policy'])
    current=frames['test/primary'].set_index(['scenario_id','policy'])
    old_pairs=0
    for short,legacy in [('matched','forecast_matched'),('central','pc_rollout'),('robust','robust_pc_rollout')]:
        fields=['cost','unserved_energy','crew_completion_events','crew_routes']
        a=original.xs(legacy,level='policy')[fields].sort_index()
        b=current.xs('published/'+short,level='policy')[fields].sort_index()
        assert a.equals(b), legacy
        old_pairs+=len(a)
    domestic=infer_check(ROOT/'results/guarded_test_20260905/inference',
                        {k.removeprefix('test/'):v for k,v in frames.items() if k.startswith('test/')})
    external=infer_check(ROOT/'results/guarded_external_20260905/inference',
                        {k:v for k,v in frames.items() if not k.startswith('test/')},external=True)
    result=dict(status='PASS',records=report,original_compatible_pairs=old_pairs,
        domestic_statistics=domestic,external_statistics=external,
        totals={key:sum(v[key] for v in report.values()) for key in ('scenarios','rows','choices','accepted','fallback')},
        frozen_sha256=sha(ROOT/'results/guarded_validation_20260905/method_decision.json'),
        script_sha256=sha(__file__),seconds=time.time()-start,
        scope='All recorded finite choices and fallback execution, source identity, primary intervals and all mean-test families. Not a global proof.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':
    main()
