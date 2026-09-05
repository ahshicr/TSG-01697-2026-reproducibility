"""Compare the declared route scores on validation-year episodes only.

All settings are inherited unchanged from the reference simulation. There is no
parameter search. Five variants expose separately the effect of the service
objective and of propagating the threat matrix within a common route portfolio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from scipy import stats

import simulate_rollout_revised as sim
import service_route_scoring as score
from paired_statistics import bootstrap_mean_ci, holm_adjust, wilcoxon_p

EXPOSURE_SELECTOR = sim.route_portfolio_plan
VARIANTS = [
    ('exposure_matched','forecast_matched',False),
    ('exposure_central','pc_rollout',False),
    ('service_matched','forecast_matched',True),
    ('service_central','pc_rollout',True),
    ('service_robust','robust_pc_rollout',True),
]


def initialize(payload):
    sim.init_worker(payload)
    sim.forecast_at = score.causal_forecast_at


def evaluate(task):
    identifier, group = task
    rng = np.random.default_rng(sim.G['seed']+identifier*7919)
    hours = sim.G['valid_first_hours']
    first = int(hours[int(rng.integers(len(hours)))])
    threat, innovations = sim.scenario_inputs(rng,group)
    coefficients = {name:float(rng.uniform(*bounds))
                    for name,bounds in sim.G['transition_uncertainty_bounds'].items()}
    crew = sim.prepare_crew_scenario(rng,threat)
    rows = []
    for variant,policy,use_service in VARIANTS:
        sim.route_portfolio_plan = score.select_service_route if use_service else EXPOSURE_SELECTOR
        outcome = sim.evaluate_policy(policy,first,threat,innovations,crew,
            identifier % len(sim.G['smartds_mappings']),coefficients)
        rows.append(dict(scenario_id=identifier,group=group,first_hour=first,
                         policy=variant,**outcome))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast-results',type=Path,default=Path('results/real_ev_strict_20260905'))
    parser.add_argument('--calibration',type=Path,default=Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv'))
    parser.add_argument('--packets',type=Path,default=Path('results/packet_training_20260905/packet_network_scenarios.csv'))
    parser.add_argument('--output',type=Path,default=Path('results/service_score_validation_20260905'))
    parser.add_argument('--scenarios',type=int,default=512)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    if (args.output/'completion.json').exists():
        raise RuntimeError('Completed validation already exists. Do not overwrite it.')
    if not (args.forecast_results/'training_manifest.json').exists():
        raise RuntimeError('All declared forecasting runs must finish before validation model selection.')
    config = sim.build_parser().parse_args([])
    config.forecast_results = args.forecast_results
    config.robust_uncertainty_file = args.calibration
    config.packet_results = args.packets
    payload = sim.payload_from_args(config)
    selected = sim.read_best_forecast(args.forecast_results/'prediction_metrics.csv')
    validation_path = selected.with_name('validation_'+selected.name)
    validation = np.load(validation_path)
    data = np.load(config.data)
    indices = validation['indices'].astype(int)
    train_end,val_end = int(data['split_train_end_index']),int(data['split_val_end_index'])
    assert indices.min()>=train_end and indices.max()+config.horizon<=val_end
    index_set = set(indices.tolist())
    firsts = [i for i in indices if all(i+h in index_set for h in range(config.horizon))]
    payload.update(forecast_index={int(h):i for i,h in enumerate(indices)},
                   forecast_pred=validation['pred'].astype(np.float32),
                   valid_first_hours=np.asarray(firsts),seed=9202201)
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [Path(__file__),Path(sim.__file__),Path(score.__file__),args.calibration,
                Path(sim.solve_alpha.__code__.co_filename),
                args.packets,selected,validation_path,config.data]}
    specification = dict(period='2022 validation only',scenarios=args.scenarios,
        variants=VARIANTS,source_sha256=sources,random_state=payload['seed'],
        first_hour_min=int(min(firsts)),first_hour_max=int(max(firsts)),
        policy_selection='Three models compared by 2022 validation forecast error, never by policy test cost.',
        score_design='Same service equations and weights as the declared objective, with packet and AC factors fixed to one.',
        control='Same service score with uncompleted threats held fixed, no matrix propagation.',
        packet_environment='Fixed response surface generated only from training-period activity.',
        tuning='No weights, capacities or forecast settings changed for route score comparison.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'specification.json').write_text(json.dumps(specification,indent=2),encoding='utf-8')
    start=time.time()
    rows=[]
    tasks=[(i,sim.GROUPS[i%4]) for i in range(args.scenarios)]
    with mp.Pool(args.workers,initializer=initialize,initargs=(payload,)) as pool:
        for i,chunk in enumerate(pool.imap_unordered(evaluate,tasks,chunksize=4),1):
            rows.extend(chunk)
            if i%64==0:
                print(f'Validation {i}/{args.scenarios}, {time.time()-start:.1f}s',flush=True)
    rows.sort(key=lambda r:(r['scenario_id'],r['policy']))
    sim.write_csv(args.output/'rollout_scenarios.csv',rows)
    comparisons=[]
    for policy,baseline in [('service_central','service_matched'),('service_central','exposure_matched'),
                            ('service_central','exposure_central'),('service_robust','service_central')]:
        a=np.array([r['cost'] for r in rows if r['policy']==policy])
        b=np.array([r['cost'] for r in rows if r['policy']==baseline])
        delta=a-b
        low,high=bootstrap_mean_ci(delta,9122201,20000)
        comparisons.append(dict(policy=policy,comparator=baseline,n=len(a),policy_mean=float(a.mean()),
            comparator_mean=float(b.mean()),difference=float(delta.mean()),ci_low=low,ci_high=high,
            improvement_percent=float(-100*delta.mean()/b.mean()),
            paired_t_p=float(stats.ttest_rel(a,b).pvalue),wilcoxon_p=wilcoxon_p(delta)))
    for r,t,w in zip(comparisons,holm_adjust([r['paired_t_p'] for r in comparisons]),
                     holm_adjust([r['wilcoxon_p'] for r in comparisons])):
        r.update(holm_t=t,holm_wilcoxon=w)
    sim.write_csv(args.output/'comparisons.csv',comparisons)
    (args.output/'completion.json').write_text(json.dumps({**specification,
        'seconds':time.time()-start,'rows':len(rows),'comparisons':comparisons},indent=2),encoding='utf-8')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
