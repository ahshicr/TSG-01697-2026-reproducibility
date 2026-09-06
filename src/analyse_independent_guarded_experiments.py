"""Independent-input inference and explicit conjunction of adoption conditions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import analyse_guarded_experiments as analysis
import run_guarded_experiments as runner
from paired_statistics import bootstrap_mean_ci, holm_adjust


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/guarded_external_20260905'))
    parser.add_argument('--freeze',type=Path,default=Path('results/guarded_validation_20260905/method_decision.json'))
    parser.add_argument('--domestic-inference',type=Path,default=Path('results/guarded_test_20260905/inference/completion.json'))
    parser.add_argument('--output',type=Path,default=Path('results/guarded_external_20260905/inference'))
    args=parser.parse_args()
    if (args.output/'completion.json').exists():
        raise RuntimeError('Preserve completed independent inference')
    frozen=json.loads(args.freeze.read_text(encoding='utf-8'))
    assert frozen['selection_script_sha256'] == runner.digest(analysis.__file__)
    domestic=json.loads(args.domestic_inference.read_text(encoding='utf-8'))
    assert domestic['frozen_sha256'] == runner.digest(args.freeze)
    args.output.mkdir(parents=True,exist_ok=True)
    means,tails,diagnostics,external_gates=[],[],[],[]
    records={}
    for forecast in ('graph','weekly'):
        frames,provenance=analysis.load(args.root/forecast,runner.TEST_CONDITIONS,'external_'+forecast)
        assert all(record['method_decision_sha256'] == runner.digest(args.freeze) for record in provenance.values())
        assert all(record['scenarios'] == 1024 for record in provenance.values())
        for condition,frame in frames.items():
            records[forecast+'/'+condition]=runner.digest(args.root/forecast/condition/'completion.json')
            for group in analysis.GROUPS:
                protected=f'protected_{frozen["selected_rates"][group]:g}'
                comparisons=[('central','matched'),('robust','central'),(protected,'matched'),(protected,'central')]
                if group=='independent':
                    comparisons += [('observed','matched'),('observed','central')]
                for policy,comparator in comparisons:
                    a,b=analysis.array(frame,group+'/'+policy),analysis.array(frame,group+'/'+comparator)
                    label=dict(forecast=forecast,condition=condition,group=group,policy=policy,comparator=comparator)
                    means.append({**label,**analysis.mean_record(a,b)})
                    if policy==protected and comparator=='central':
                        for metric in ('q95','q99','es95','maximum'):
                            tails.append({**label,'metric':metric,**analysis.tail_record(a,b,metric)})
                f=analysis.array(frame,group+'/matched')
                for policy in ('matched','central','robust',protected)+(('observed',) if group=='independent' else ()):
                    part=frame.loc[frame.policy.eq(group+'/'+policy)]
                    a=analysis.array(frame,group+'/'+policy)
                    diagnostics.append(dict(forecast=forecast,condition=condition,group=group,policy=policy,
                        mean_cost=float(a.mean()),q95=float(np.quantile(a,.95)),q99=float(np.quantile(a,.99)),
                        es95=analysis.empirical_tail(a,.95),maximum_cost=float(a.max()),
                        positive_excess=float(np.maximum(a-f,0).mean()),worse_fraction=float((a-f>1e-8).mean()),
                        guard_acceptance=float(part.guard_accepted.mean()) if policy==protected else 0.0,
                        crew_completion_fraction=float(part.crew_completion_fraction.mean()),
                        crew_mean_completion_h=float(part.crew_mean_completion_h.mean()),
                        crew_total_travel_h=float(part.crew_total_travel_h.mean()),
                        communication_loss=float(part.comm_loss.mean()),
                        raw_infeasible_hours=int(part.smartds_raw_infeasible_hours.sum()),
                        infeasible_hours=int(part.smartds_projected_infeasible_hours.sum()),
                        curtailed_energy_kwh=float(part.smartds_curtailed_energy_kwh.sum())))
        for group in analysis.GROUPS:
            primary=frames['primary']
            name=f'{group}/protected_{frozen["selected_rates"][group]:g}'
            delta=analysis.array(primary,name)-analysis.array(primary,group+'/central')
            shifted=delta-.0005*analysis.array(primary,group+'/matched')
            low,upper=bootstrap_mean_ci(shifted,analysis.STAT_RANDOM_STATE,analysis.BOOTSTRAPS)
            external_gates.append(dict(forecast=forecast,group=group,guard_minus_central=float(delta.mean()),
                noninferiority_shifted_mean=float(shifted.mean()),noninferiority_upper=upper,
                external_noninferiority_pass=bool(upper<0)))
    for row,p in zip(means,holm_adjust([row['paired_t_p'] for row in means])):
        row['holm_paired_t']=p
    for row,p in zip(means,holm_adjust([row['wilcoxon_p'] for row in means])):
        row['holm_wilcoxon']=p
    decisions=[]
    for gate in domestic['gates']:
        group=gate['group']
        external_pass=all(row['external_noninferiority_pass'] for row in external_gates if row['group']==group)
        passed=bool(gate['validation_screen_pass'] and gate['primary_noninferiority_pass']
                    and gate['positive_excess_pass'] and external_pass)
        decisions.append(dict(group=group,validation_screen_pass=gate['validation_screen_pass'],
            domestic_noninferiority_pass=gate['primary_noninferiority_pass'],
            positive_excess_pass=gate['positive_excess_pass'],external_noninferiority_pass=external_pass,
            numerical_performance_adoption_pass=passed,
            final_adoption_requires='Independent verification of information, execution, statistics and complete mathematical statements.'))
    for name,rows in [('paired_costs',means),('tails',tails),('diagnostics',diagnostics)]:
        pd.DataFrame(rows).to_csv(args.output/(name+'.csv'),index=False)
    result=dict(status='INDEPENDENT_INFERENCE_COMPLETE',external_gates=external_gates,adoption_decisions=decisions,
        cost_comparisons=len(means),tail_comparisons=len(tails),source_records=records,
        frozen_sha256=runner.digest(args.freeze),domestic_inference_sha256=runner.digest(args.domestic_inference),
        script_sha256=runner.digest(__file__),
        outputs={name:runner.digest(args.output/(name+'.csv')) for name in ('paired_costs','tails','diagnostics')},
        scope='Independent geographic charging and constructed service-restoration execution, not joint field validation.')
    (args.output/'completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
