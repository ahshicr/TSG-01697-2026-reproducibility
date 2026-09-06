"""Prepare independent municipal station-hour inputs under fixed filtering rules.

User identifiers and unrelated transaction fields are never loaded. Reconstructed
hourly energy is not represented as metered hourly power. Forecast parameters and
normalization are transferred without learning from the external evaluation year.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from build_boulder_forecast_dataset import haversine_km, spatial_graph
from models import GraphGRUForecaster

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/external/independent_check_20260905/palo_alto_official_sessions.csv'
OUT = ROOT/'data/external/processed/palo_alto_independent_20260905'
CHECKPOINT = ROOT/'results/real_ev_strict_20260905/models/forecaster_plain_seed0.pt'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval_totals(starts, durations, totals, stations, origin, hours, n):
    """Accumulate piecewise constant intervals exactly at hourly boundaries."""
    seconds = (starts.to_numpy(dtype='datetime64[ns]')-origin.to_datetime64())/np.timedelta64(1,'s')
    duration = np.asarray(durations,float)
    end = seconds+duration
    left = np.maximum(seconds,0)/3600
    right = np.minimum(end,hours*3600)/3600
    power = np.asarray(totals,float)/(duration/3600)
    result = np.zeros((hours,n),dtype=np.float64)
    difference = np.zeros((hours+1,n),dtype=np.float64)
    useful = right>left
    left,right,power,stations = left[useful],right[useful],power[useful],np.asarray(stations)[useful]
    first = np.floor(left).astype(int)
    last = np.ceil(right).astype(int)-1
    single = first==last
    np.add.at(result,(first[single],stations[single]),power[single]*(right[single]-left[single]))
    multiple = ~single
    np.add.at(result,(first[multiple],stations[multiple]),power[multiple]*(first[multiple]+1-left[multiple]))
    np.add.at(result,(last[multiple],stations[multiple]),power[multiple]*(right[multiple]-last[multiple]))
    np.add.at(difference,(first[multiple]+1,stations[multiple]),power[multiple])
    np.add.at(difference,(last[multiple],stations[multiple]),-power[multiple])
    result += np.cumsum(difference,axis=0)[:hours]
    assert result.min() > -1e-8
    result = np.maximum(result,0)
    expected = float(np.sum(power*(right-left)))
    assert np.isclose(result.sum(),expected,rtol=1e-10,atol=1e-6)
    return result.astype(np.float32), expected


def main():
    if (OUT/'preparation.json').exists():
        raise RuntimeError('Preserve the prepared external inputs')
    OUT.mkdir(parents=True,exist_ok=True)
    source_metadata = json.loads((SOURCE.parent/'palo_alto_source.json').read_text(encoding='utf-8'))
    assert digest(SOURCE) == source_metadata['sha256']
    protocol = ROOT/'config/independent_input_protocol_20260905.md'
    fields = ['Station Name','Start Date','End Date','Total Duration (hh:mm:ss)',
              'Charging Time (hh:mm:ss)','Energy (kWh)','Latitude','Longitude']
    original = pd.read_csv(SOURCE,usecols=fields,low_memory=False)
    frame = original.drop_duplicates().copy()
    duplicate_count = len(original)-len(frame)
    frame['start'] = pd.to_datetime(frame['Start Date'],format='mixed',errors='coerce')
    frame['energy'] = pd.to_numeric(frame['Energy (kWh)'],errors='coerce')
    frame['duration_s'] = pd.to_timedelta(frame['Charging Time (hh:mm:ss)'],errors='coerce').dt.total_seconds()
    frame['connection_s'] = pd.to_timedelta(frame['Total Duration (hh:mm:ss)'],errors='coerce').dt.total_seconds()
    frame['latitude'] = pd.to_numeric(frame['Latitude'],errors='coerce')
    frame['longitude'] = pd.to_numeric(frame['Longitude'],errors='coerce')
    rules = dict(start=frame['start'].notna(), station=frame['Station Name'].notna(),
        energy=np.isfinite(frame.energy)&frame.energy.gt(0),
        charging=np.isfinite(frame.duration_s)&frame.duration_s.gt(0)&frame.duration_s.le(48*3600),
        connection=np.isfinite(frame.connection_s)&frame.connection_s.gt(0)&frame.duration_s.le(frame.connection_s),
        coordinates=frame.latitude.between(-90,90)&frame.longitude.between(-180,180))
    valid = np.logical_and.reduce(list(rules.values()))
    frame = frame.loc[valid].copy()
    frame = frame.loc[frame.start.ge('2017-01-01')&frame.start.lt('2021-01-01')].copy()
    training = frame.loc[frame.start.lt('2019-01-01')]
    counts = training.groupby('Station Name').size()
    retained = sorted(counts[counts>=100].index)
    if not 1 < len(retained) <= 50:
        raise RuntimeError('Unexpected station eligibility or packet benchmark size')
    excluded_station_rows = int((~frame['Station Name'].isin(retained)).sum())
    frame = frame.loc[frame['Station Name'].isin(retained)].copy()
    station_numbers = {name:i for i,name in enumerate(retained)}
    station_ids = np.array([f'P{i:03}' for i in range(len(retained))])
    station_index = frame['Station Name'].map(station_numbers).to_numpy(int)
    coordinates = training.groupby('Station Name')[['latitude','longitude']].median().loc[retained]
    coordinates.insert(0,'station_id',station_ids)
    coordinates.reset_index().rename(columns={'Station Name':'source_station_name'}).to_csv(OUT/'station_coordinates.csv',index=False)
    origin, stop = pd.Timestamp('2017-01-01'), pd.Timestamp('2021-01-01')
    timestamps = pd.date_range(origin,stop,freq='h',inclusive='left')
    hours,n = len(timestamps),len(retained)
    arrivals = np.zeros((hours,n),dtype=np.float32)
    indices = ((frame.start.to_numpy(dtype='datetime64[ns]')-origin.to_datetime64())/np.timedelta64(1,'h')).astype(int)
    np.add.at(arrivals,(indices,station_index),1)
    energy,expected_energy = interval_totals(frame.start,frame.duration_s,frame.energy,station_index,origin,hours,n)
    connected,expected_connected = interval_totals(frame.start,frame.connection_s,frame.connection_s/3600,station_index,origin,hours,n)
    train_end = int(timestamps.searchsorted(pd.Timestamp('2019-01-01')))
    test_start = int(timestamps.searchsorted(pd.Timestamp('2020-01-01')))
    distance = haversine_km(coordinates.latitude.to_numpy(),coordinates.longitude.to_numpy())
    adjacency = spatial_graph(distance)
    comm = 2.0+.5*arrivals[:train_end].mean(axis=0)+.1*connected[:train_end].mean(axis=0)
    np.savez_compressed(OUT/'station_hour_inputs.npz',timestamp_local=timestamps.to_numpy(dtype='datetime64[m]'),
        station_id=station_ids,pickup=arrivals,energy=energy,connected_session_hours=connected,
        adj=adjacency,distance_km=distance.astype(np.float32),comm_capacity=comm.astype(np.float32),
        split_train_end_index=np.int64(train_end),split_val_end_index=np.int64(test_start))
    # Author-created checkpoint from the recorded, already verified local source.
    checkpoint = torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    config = checkpoint['args']
    model = GraphGRUForecaster(2,config['hidden'],config['horizon'],adjacency,config['dropout'])
    state = dict(checkpoint['state_dict'])
    state['adj'] = torch.as_tensor(adjacency)
    model.load_state_dict(state)
    model.eval()
    for name,parameter in model.named_parameters():
        assert torch.equal(parameter.detach(),checkpoint['state_dict'][name]),name
    raw = np.stack([arrivals,energy],axis=-1)
    mean,std = checkpoint['mean'],checkpoint['std']
    features = (np.log1p(raw)-mean)/std
    forecast_indices = np.arange(test_start,hours-config['horizon'],dtype=np.int64)
    predictions = []
    torch.set_num_threads(4)
    with torch.no_grad():
        for offset in range(0,len(forecast_indices),128):
            batch = forecast_indices[offset:offset+128]
            inputs = np.stack([features[t-config['history']:t] for t in batch])
            output = model(torch.from_numpy(inputs))
            predictions.append(torch.expm1(output*torch.from_numpy(std)+torch.from_numpy(mean)).clamp_min(0).numpy())
    predictions = np.concatenate(predictions).astype(np.float32)
    weekly = np.stack([raw[t-168:t-168+config['horizon']] for t in forecast_indices]).astype(np.float32)
    assert np.isfinite(predictions).all() and np.isfinite(weekly).all()
    assert np.all(forecast_indices-168+config['horizon'] <= forecast_indices)
    np.savez_compressed(OUT/'issued_forecasts.npz',indices=forecast_indices,graph_pred=predictions,weekly_pred=weekly)
    report = dict(source_sha256=digest(SOURCE),source_records=len(original),exact_duplicate_analysis_records=duplicate_count,
        failed_rules_nonexclusive={key:int((~value).sum()) for key,value in rules.items()},
        valid_retained_sessions=len(frame),excluded_station_rows=excluded_station_rows,stations=n,
        retained_sessions_by_year=frame.start.dt.year.value_counts().sort_index().to_dict(),
        training_station_eligibility='At least 100 valid 2017-2018 sessions. No test-year station selection.',
        time_basis='Source local civil timestamps and reported elapsed charging durations.',
        reconstructed_energy_kwh=float(energy.sum(dtype=np.float64)),expected_in_window_energy_kwh=expected_energy,
        energy_boundary_truncation_kwh=float(frame.energy.sum()-expected_energy),
        connected_hours=float(connected.sum(dtype=np.float64)),expected_connected_hours=expected_connected,
        raw_input_fields_loaded=fields,user_identifiers_loaded=False,
        training_mean_cutoff='2019-01-01',evaluation_year=2020,forecast_windows=len(forecast_indices),
        checkpoint_sha256=digest(CHECKPOINT),learned_weights_unchanged=True,original_normalization_unchanged=True,
        changed_buffer='Nonlearned station adjacency only',
        forecast_rule='Fixed retrospective geographic transfer and separately reported weekly persistence. No external fitting or selection.',
        protocol_sha256=digest(protocol),script_sha256=digest(__file__),prepared_unix=time.time(),
        outputs={p.name:digest(p) for p in [OUT/'station_hour_inputs.npz',OUT/'issued_forecasts.npz',OUT/'station_coordinates.csv']})
    (OUT/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
