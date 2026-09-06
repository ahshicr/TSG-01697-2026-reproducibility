"""Locked policy transfer to independent municipal demand and station geometry."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import pandas as pd

import run_guarded_experiments as runner
import simulate_rollout_revised as sim

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '40899a79c74b8d629dd300380c7a7c866133a74719a52a9291337a8ef2dda221'


def external_payload(args, condition, forecast_kind):
    payload = runner.build_payload(args,condition)
    data = np.load(args.inputs/'station_hour_inputs.npz')
    forecasts = np.load(args.inputs/'issued_forecasts.npz')
    coordinates = pd.read_csv(args.inputs/'station_coordinates.csv')
    training_end = int(data['split_train_end_index'])
    station_ids = data['station_id'].astype(str)
    assert np.array_equal(coordinates.station_id.to_numpy(str),station_ids)
    raw = np.stack([data['pickup'],data['energy']],axis=-1).astype(np.float32)
    mean_demand = data['pickup'][:training_end].mean(axis=0).astype(np.float32)
    mean_energy = data['energy'][:training_end].mean(axis=0).astype(np.float32)
    config = sim.build_parser().parse_args([])
    charge_factor = runner.CONDITIONS[condition].get('charge_capacity_factor',config.charge_capacity_factor)
    indices = forecasts['indices'].astype(int)
    assert indices.min() >= int(data['split_val_end_index'])
    issued = set(indices)
    firsts = np.asarray([hour for hour in indices if all(hour+h in issued for h in range(payload['horizon']))])
    distance = sim.haversine_km(coordinates.latitude.to_numpy(),coordinates.longitude.to_numpy())
    travel = np.rint(distance*config.crew_road_circuity/config.crew_speed_kmh*60).astype(np.int64)
    eligible = []
    for index,name in enumerate(payload['smartds_load_names']):
        sim.smartds_dss.Loads.Name(name)
        if sim.smartds_dss.Loads.Phases() == 3:
            eligible.append(index)
    assert eligible
    mappings = np.stack([sim.mapping_for_seed(coordinates,np.asarray(eligible),config.smartds_mapping_seed+i)
        for i in range(config.smartds_mappings)])
    packet = payload['packet_response_surface']
    # Constructed benchmark surfaces, explicitly not measurements from this city.
    packet = {key:value[:len(station_ids)].copy() for key,value in packet.items()}
    assert packet['points'].shape[0] == len(station_ids)
    base_restore = float(max(10.0,mean_demand.sum()*.018))
    payload.update(raw=raw,adj=data['adj'].astype(np.float32),mean_demand=mean_demand,mean_energy=mean_energy,
        station_ids=station_ids,forecast_pred=forecasts[forecast_kind+'_pred'],
        forecast_index={int(hour):i for i,hour in enumerate(indices)},valid_first_hours=firsts,
        seed=202609052,total_charge=float(mean_energy.sum()*charge_factor),
        total_comm=float(data['comm_capacity'].sum()*config.comm_capacity_factor),
        base_total_restore=base_restore,total_restore=base_restore*config.restore_scale,
        crew_travel_minutes=travel,crew_depot_index=int(np.argmin(distance.sum(axis=1))),
        smartds_mappings=mappings,packet_response_surface=packet,normalization_train_end=training_end,
        normalization_rule='Independent station means and capacities use 2017-2018 only.',
        forecast_file='fixed_'+forecast_kind+'_independent_input')
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,default=Path('data/external/processed/palo_alto_independent_20260905'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--forecast-results',type=Path,default=Path('results/real_ev_strict_20260905'))
    parser.add_argument('--calibration',type=Path,default=Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv'))
    parser.add_argument('--packets',type=Path,default=Path('results/packet_training_20260905/packet_network_scenarios.csv'))
    parser.add_argument('--freeze',type=Path,default=Path('results/guarded_validation_20260905/method_decision.json'))
    parser.add_argument('--forecasts',nargs='+',choices=('graph','weekly'),default=['graph','weekly'])
    parser.add_argument('--conditions',nargs='+',choices=runner.TEST_CONDITIONS,default=list(runner.TEST_CONDITIONS))
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--smoke-scenarios',type=int)
    args=parser.parse_args()
    args.phase='test'
    frozen=json.loads(args.freeze.read_text(encoding='utf-8'))
    assert frozen['source_sha256'] == runner.source_hashes(args)
    preparation=json.loads((args.inputs/'preparation.json').read_text(encoding='utf-8'))
    assert preparation['protocol_sha256'] == PROTOCOL_SHA
    assert runner.digest(ROOT/'config/independent_input_protocol_20260905.md') == PROTOCOL_SHA
    assert preparation['script_sha256'] == runner.digest(ROOT/'src/prepare_independent_ev_input.py')
    for name,sha in preparation['outputs'].items():
        assert runner.digest(args.inputs/name) == sha
    sources={**frozen['source_sha256'],
        'src/run_independent_guarded_experiments.py':runner.digest(__file__),
        'src/prepare_independent_ev_input.py':preparation['script_sha256'],
        **{(args.inputs/name).as_posix():sha for name,sha in preparation['outputs'].items()}}
    for forecast in args.forecasts:
        for condition in args.conditions:
            out=args.output/forecast/condition
            completed=out/'completion.json'
            count=args.smoke_scenarios or 1024
            if completed.exists():
                record=json.loads(completed.read_text(encoding='utf-8'))
                assert record['source_sha256'] == sources and record['scenarios'] == count
                assert record['scenario_sha256'] == runner.digest(out/'rollout_scenarios.csv')
                print(f'Already complete external/{forecast}/{condition}',flush=True)
                continue
            payload=external_payload(args,condition,forecast)
            out.mkdir(parents=True,exist_ok=True)
            specification=dict(phase='external_'+forecast,condition=condition,scenarios=count,
                protocol_sha256=PROTOCOL_SHA,source_sha256=sources,
                method_decision_sha256=runner.digest(args.freeze),preparation_sha256=runner.digest(args.inputs/'preparation.json'),
                random_state=payload['seed'],forecast=forecast,guard_rates=list(payload['guard_rates']),
                stations=len(payload['station_ids']),training_mean_cutoff='2019-01-01',evaluation_year=2020,
                benchmark_other_inputs='Source abstract transition, packet, outage and repair models are transferred.',
                independent_observations='Municipal charging sessions and station coordinates only.',
                budgets={key:payload[key] for key in ('total_charge','total_comm','total_restore')},
                started_unix=time.time())
            (out/'specification.json').write_text(json.dumps(specification,indent=2),encoding='utf-8')
            print(f'Starting external/{forecast}/{condition}: {count} scenarios',flush=True)
            rows,details=[],[]
            tasks=[(i,sim.GROUPS[i%len(sim.GROUPS)]) for i in range(count)]
            with mp.Pool(args.workers,initializer=runner.initialize,initargs=(payload,)) as pool:
                for index,(part,info) in enumerate(pool.imap_unordered(runner.evaluate,tasks,chunksize=2),1):
                    rows.extend(part)
                    details.append(info)
                    if index%128==0 or index==count:
                        print(f'{forecast}/{condition}: {index}/{count}, {time.time()-specification["started_unix"]:.1f}s',flush=True)
            rows.sort(key=lambda row:(row['scenario_id'],row['policy']))
            details.sort(key=lambda row:row['scenario_id'])
            assert len({(row['scenario_id'],row['policy']) for row in rows}) == len(rows)
            assert all(row['smartds_projected_infeasible_hours'] == 0 for row in rows)
            sim.write_csv(out/'rollout_scenarios.csv',rows)
            with (out/'candidate_scores.jsonl').open('w',encoding='utf-8') as handle:
                for record in details:
                    handle.write(json.dumps(record)+'\n')
            completed.write_text(json.dumps({**specification,'rows':len(rows),
                'seconds':time.time()-specification['started_unix'],
                'scenario_sha256':runner.digest(out/'rollout_scenarios.csv'),
                'scores_sha256':runner.digest(out/'candidate_scores.jsonl')},indent=2),encoding='utf-8')
            print(f'Completed external/{forecast}/{condition}',flush=True)


if __name__=='__main__':
    main()
