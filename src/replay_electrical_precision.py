"""Independent precision replay of the frozen policies' accepted charging loads.

The original simulator, decisions and execution gate are unchanged. Two
separate OpenDSS contexts measure the exact accepted load with tighter solver
tolerances. This is a numerical measurement check, not algorithm development.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
import opendssdirect as dss

import run_frozen_service_experiments as frozen
import service_route_scoring as score
import simulate_rollout_revised as sim

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_PROJECT=sim.project_smartds_action
CONTEXTS=[]
OBSERVATIONS=[]
TOLERANCES=(1e-8,1e-10)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def init_worker(payload):
    global CONTEXTS
    score.init_worker(payload)
    CONTEXTS=[]
    for tolerance in TOLERANCES:
        context=dss.NewContext()
        context.Text.Command(f'Redirect "{payload["smartds_master"]}"')
        context.Solution.Convergence(tolerance)
        context.Solution.MaxIterations(100)
        context.Solution.Solve()
        if not context.Solution.Converged():
            raise RuntimeError('Independent base case failed to converge')
        names=tuple(context.Circuit.AllNodeNames())
        active=np.asarray(context.Circuit.AllBusMagPu(),float)>.1
        CONTEXTS.append((tolerance,context,names,active))
    sim.project_smartds_action=project_and_measure


def project_and_measure(requested,mapping_index):
    alpha,raw,accepted=ORIGINAL_PROJECT(requested,mapping_index)
    ev_by_load=np.zeros(len(sim.G['smartds_load_names']))
    np.add.at(ev_by_load,sim.G['smartds_mappings'][mapping_index],np.asarray(requested,float)*alpha)
    reactive_factor=math.tan(math.acos(sim.G['smartds_power_factor']))
    record=dict(retained_fraction=alpha,original_loss_mwh=accepted['losses_kw']/1000,
                original_vmin_pu=accepted['v_min_pu'],mapping_index=mapping_index,
                accepted_ev_load_kw=json.dumps(ev_by_load.tolist(),separators=(',',':')))
    for tolerance,context,names,active in CONTEXTS:
        for i,name in enumerate(sim.G['smartds_load_names']):
            context.Loads.Name(name)
            context.Loads.kW(float(sim.G['smartds_base_kw'][i]+ev_by_load[i]))
            context.Loads.kvar(float(sim.G['smartds_base_kvar'][i]+ev_by_load[i]*reactive_factor))
        context.Solution.Solve()
        if names!=tuple(context.Circuit.AllNodeNames()):
            raise RuntimeError('Independent node identities changed')
        voltage=np.asarray(context.Circuit.AllBusMagPu(),float)[active]
        ratios=[]
        for line in context.Lines.AllNames():
            context.Lines.Name(line)
            currents=np.asarray(context.CktElement.CurrentsMagAng(),float)[::2]
            normal=float(context.Lines.NormAmps())
            if normal>0 and currents.size:
                ratios.append(float(currents.max()/normal))
        feasible=(context.Solution.Converged() and voltage.min()>=sim.G['smartds_voltage_low']
                  and voltage.max()<=sim.G['smartds_voltage_high']
                  and max(ratios)<=sim.G['smartds_line_limit'])
        suffix='tol8' if tolerance==1e-8 else 'tol10'
        record.update({f'loss_mwh_{suffix}':float(context.Circuit.Losses()[0]/1e6),
                       f'vmin_pu_{suffix}':float(voltage.min()),
                       f'vmax_pu_{suffix}':float(voltage.max()),
                       f'max_line_pu_{suffix}':max(ratios),
                       f'feasible_{suffix}':bool(feasible),
                       f'iterations_{suffix}':int(context.Solution.Iterations())})
    OBSERVATIONS.append(record)
    # Never feed the tighter-context measurements back to the frozen policy.
    return alpha,raw,accepted


def evaluate(task):
    global OBSERVATIONS
    OBSERVATIONS=[]
    rows=frozen.evaluate(task)
    horizon=sim.G['horizon']
    if len(OBSERVATIONS)!=len(rows)*horizon:
        raise RuntimeError('Unexpected number of electrical actions')
    hours=[]
    for index,row in enumerate(rows):
        observations=OBSERVATIONS[index*horizon:(index+1)*horizon]
        for hour,record in enumerate(observations):
            hours.append(dict(scenario_id=row['scenario_id'],policy=row['policy'],hour=hour,**record))
        for suffix in ('tol8','tol10'):
            row['precision_losses_mwh_'+suffix]=sum(record['loss_mwh_'+suffix] for record in observations)
            row['precision_min_voltage_pu_'+suffix]=min(record['vmin_pu_'+suffix] for record in observations)
            row['precision_infeasible_hours_'+suffix]=sum(not record['feasible_'+suffix] for record in observations)
    return rows,hours


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/electrical_precision_20260905'))
    parser.add_argument('--scenarios',type=int,default=4096)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--reverse-policy-order',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a fresh output path to preserve every precision check')
    decision_path=ROOT/'results/service_score_validation_20260905/method_decision.json'
    decision=json.loads(decision_path.read_text(encoding='utf-8'))
    config=sim.build_parser().parse_args([])
    config.data=ROOT/config.data
    config.forecast_results=ROOT/'results/real_ev_strict_20260905'
    config.robust_uncertainty_file=ROOT/'results/calibration_strict_20260905/transition_parameter_uncertainty.csv'
    config.packet_results=ROOT/'results/packet_training_20260905/packet_network_scenarios.csv'
    selected=sim.read_best_forecast(config.forecast_results/'prediction_metrics.csv')
    paths=[Path(sim.__file__),Path(score.__file__),Path(sim.solve_alpha.__code__.co_filename),
           config.robust_uncertainty_file,config.packet_results,selected]
    hashes={path.name:digest(path) for path in paths}
    if hashes!=decision['locked_source_sha256']:
        raise RuntimeError('Original method/input hash differs from frozen study')
    payload=sim.payload_from_args(config)
    forecast_hours=set(payload['forecast_index'])
    payload['valid_first_hours']=np.asarray([h for h in payload['valid_first_hours']
        if all(h+k in forecast_hours for k in range(payload['horizon']))])
    payload['policies']=['forecast_matched','pc_rollout']
    if args.reverse_policy_order:
        payload['policies'].reverse()
    started=time.time()
    specification=dict(scenarios=args.scenarios,policies=payload['policies'],
        workers=args.workers,tolerances=TOLERANCES,maximum_iterations=100,
        original_gate_tolerance=float(dss.Solution.Convergence()),
        frozen_source_sha256=hashes,precision_source_sha256=digest(Path(__file__)),
        independence='Separate OpenDSS contexts measure identical accepted loads without changing original execution or scores.',
        started_unix=started)
    args.output.mkdir(parents=True)
    (args.output/'specification.json').write_text(json.dumps(specification,indent=2),encoding='utf-8')
    rows,hours=[],[]
    tasks=[(i,sim.GROUPS[i%len(sim.GROUPS)]) for i in range(args.scenarios)]
    with mp.Pool(args.workers,initializer=init_worker,initargs=(payload,)) as pool:
        for index,(batch,trace) in enumerate(pool.imap_unordered(evaluate,tasks,chunksize=4),1):
            rows.extend(batch)
            hours.extend(trace)
            if index%128==0 or index==args.scenarios:
                print(f'Precision replay {index}/{args.scenarios}, {time.time()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows).sort_values(['scenario_id','policy']).reset_index(drop=True)
    trace=pd.DataFrame(hours).sort_values(['scenario_id','policy','hour']).reset_index(drop=True)
    original=pd.read_csv(ROOT/'results/submission_service_20260905/primary/rollout_scenarios.csv')
    original=original[original.scenario_id.lt(args.scenarios)&original.policy.isin(payload['policies'])]
    original=original.sort_values(['scenario_id','policy']).reset_index(drop=True)
    exclude={'latency_ms','losses_mwh','min_voltage_pu'}
    exact=[]
    for column in original.columns:
        if column not in exclude:
            if not original[column].fillna('').equals(frame[column].fillna('')):
                # CSV round trips can differ in their last decimal digit.
                if not (pd.api.types.is_numeric_dtype(original[column]) and
                        np.allclose(original[column],frame[column],rtol=1e-12,atol=1e-12,equal_nan=True)):
                    raise RuntimeError('A policy outcome changed during precision replay: '+column)
            exact.append(column)
    frame.to_csv(args.output/'scenario_measurements.csv',index=False)
    trace.to_csv(args.output/'hourly_accepted_loads.csv',index=False)
    report={**specification,'finished_unix':time.time(),'elapsed_s':time.time()-started,
        'rows':len(frame),'hours':len(trace),'policy_fields_unchanged':exact,
        'scenario_sha256':digest(args.output/'scenario_measurements.csv'),
        'hourly_sha256':digest(args.output/'hourly_accepted_loads.csv'),
        'infeasible_hours_tol8':int((~trace.feasible_tol8).sum()),
        'infeasible_hours_tol10':int((~trace.feasible_tol10).sum()),
        'maximum_scenario_loss_tolerance_difference_mwh':float(np.max(np.abs(frame.precision_losses_mwh_tol8-frame.precision_losses_mwh_tol10))),
        'maximum_scenario_voltage_tolerance_difference_pu':float(np.max(np.abs(frame.precision_min_voltage_pu_tol8-frame.precision_min_voltage_pu_tol10)))}
    (args.output/'completion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
