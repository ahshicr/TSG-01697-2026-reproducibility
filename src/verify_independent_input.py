"""Rebuild external hourly energy, replay causal forecasts and actual policies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from models import GraphGRUForecaster
import run_guarded_experiments as runner
import run_independent_guarded_experiments as external


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/independent_input_verification_20260905.json'))
    args=parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve completed external input verification')
    start=time.time()
    directory=Path('data/external/processed/palo_alto_independent_20260905')
    record=json.loads((directory/'preparation.json').read_text(encoding='utf-8'))
    data=np.load(directory/'station_hour_inputs.npz')
    forecast=np.load(directory/'issued_forecasts.npz')
    stations=pd.read_csv(directory/'station_coordinates.csv')
    source=Path('data/external/independent_check_20260905/palo_alto_official_sessions.csv')
    assert runner.digest(source)==record['source_sha256']
    frame=pd.read_csv(source,usecols=record['raw_input_fields_loaded'],low_memory=False).drop_duplicates()
    frame['start']=pd.to_datetime(frame['Start Date'],format='mixed',errors='coerce')
    frame['seconds']=pd.to_timedelta(frame['Charging Time (hh:mm:ss)']).dt.total_seconds()
    frame['connection']=pd.to_timedelta(frame['Total Duration (hh:mm:ss)']).dt.total_seconds()
    frame['energy']=pd.to_numeric(frame['Energy (kWh)'])
    keep=frame.start.between('2017-01-01','2021-01-01',inclusive='left')
    keep &= frame.seconds.gt(0)&frame.seconds.le(48*3600)&frame.seconds.le(frame.connection)
    keep &= np.isfinite(frame.energy)&frame.energy.gt(0)
    keep &= frame.Latitude.between(-90,90)&frame.Longitude.between(-180,180)
    keep &= frame['Station Name'].notna()
    frame=frame.loc[keep]
    training=frame.loc[frame.start.lt('2019-01-01')]
    eligible=sorted(training.groupby('Station Name').size().loc[lambda s:s.ge(100)].index)
    assert eligible==stations.source_station_name.tolist()
    frame=frame.loc[frame['Station Name'].isin(eligible)]
    assert len(frame)==record['valid_retained_sessions']
    assert frame.start.dt.year.value_counts().sort_index().to_dict()=={
        int(k):v for k,v in record['retained_sessions_by_year'].items()}
    coordinates=training.groupby('Station Name')[['Latitude','Longitude']].median().loc[eligible]
    np.testing.assert_allclose(coordinates.to_numpy(),stations[['latitude','longitude']].to_numpy(),atol=1e-12)
    energy=np.zeros_like(data['energy'],dtype=np.float64)
    arrivals=np.zeros_like(data['pickup'])
    origin=pd.Timestamp('2017-01-01')
    offsets=(frame.start-origin).dt.total_seconds().to_numpy()/3600
    mapping={name:i for i,name in enumerate(eligible)}
    indexes=frame['Station Name'].map(mapping).to_numpy(int)
    # Direct overlap accumulation, independent of the production difference-array method.
    for left,duration,total,zone in zip(offsets,frame.seconds.to_numpy()/3600,frame.energy,indexes):
        right=min(float(left+duration),float(len(energy)))
        bins=np.arange(int(np.floor(left)),int(np.ceil(right)))
        overlap=np.maximum(0,np.minimum(right,bins+1)-np.maximum(left,bins))
        energy[bins,zone]+=float(total)/duration*overlap
        arrivals[int(left),zone]+=1
    np.testing.assert_allclose(energy.astype(np.float32),data['energy'],rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(arrivals,data['pickup'])
    energy_error=float(np.max(np.abs(energy-data['energy'])))
    raw=np.stack([data['pickup'],data['energy']],axis=-1)
    indices=forecast['indices']
    for offset,t in enumerate(indices):
        np.testing.assert_array_equal(forecast['weekly_pred'][offset],raw[t-168:t-162])
        assert t-162 < t
    checkpoint_path=Path('results/real_ev_strict_20260905/models/forecaster_plain_seed0.pt')
    assert runner.digest(checkpoint_path)==record['checkpoint_sha256']
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config=checkpoint['args']
    model=GraphGRUForecaster(2,config['hidden'],config['horizon'],data['adj'],config['dropout'])
    state=dict(checkpoint['state_dict'])
    state['adj']=torch.from_numpy(data['adj'])
    model.load_state_dict(state)
    model.eval()
    assert all(torch.equal(value.detach(),checkpoint['state_dict'][name]) for name,value in model.named_parameters())
    selected=np.linspace(0,len(indices)-1,32,dtype=int)
    features=(np.log1p(raw)-checkpoint['mean'])/checkpoint['std']
    history=np.stack([features[t-config['history']:t] for t in indices[selected]])
    with torch.no_grad():
        replay=torch.expm1(model(torch.from_numpy(history))*torch.from_numpy(checkpoint['std'])
                           +torch.from_numpy(checkpoint['mean'])).clamp_min(0).numpy()
    np.testing.assert_allclose(replay,forecast['graph_pred'][selected],rtol=3e-6,atol=3e-6)
    forecast_error=float(np.max(np.abs(replay-forecast['graph_pred'][selected])))
    specification=argparse.Namespace(phase='test',inputs=directory,
        forecast_results=Path('results/real_ev_strict_20260905'),
        calibration=Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv'),
        packets=Path('results/packet_training_20260905/packet_network_scenarios.csv'),
        freeze=Path('results/guarded_validation_20260905/method_decision.json'))
    execution=[]
    exact=['cost','unserved_energy','mobility_delay','comm_loss','crew_completion_events','crew_routes',
           'crew_jobs_completed','crew_job_count','crew_jobs_dispatched',
           'requested_charge_capacity','executed_charge_capacity','smartds_projected_infeasible_hours']
    for kind in ('graph','weekly'):
        for condition in ('primary','electrical_stress'):
            payload=external.external_payload(specification,condition,kind)
            runner.initialize(payload)
            saved=pd.read_csv(Path('results/guarded_external_20260905')/kind/condition/'rollout_scenarios.csv',keep_default_na=False)
            indexed=saved.set_index(['scenario_id','policy'])
            for identifier in (0,1,127,767):
                rows,_=runner.evaluate((identifier,runner.sim.GROUPS[identifier%4]))
                for row in rows:
                    reference=indexed.loc[(identifier,row['policy'])]
                    for field in exact:
                        if isinstance(row[field],str):
                            assert row[field]==reference[field],(kind,condition,identifier,field)
                        else:
                            np.testing.assert_allclose(row[field],reference[field],rtol=1e-13,atol=1e-10)
                execution.append(dict(forecast=kind,condition=condition,scenario=identifier,policy_rows=len(rows)))
    result=dict(status='PASS',sessions_reconstructed=len(frame),energy_max_hourly_error_kwh=energy_error,
        all_station_hour_energy_and_arrivals_reconstructed=True,weekly_windows_exact=len(indices),
        graph_windows_replayed=len(selected),graph_max_absolute_error=forecast_error,
        learned_parameters_unchanged=True,execution_replays=execution,
        preparation_sha256=runner.digest(directory/'preparation.json'),
        source_sha256=runner.digest(source),script_sha256=runner.digest(__file__),seconds=time.time()-start,
        scope='Independent reconstruction of finite data and replayed executions, not a field deployment claim.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
