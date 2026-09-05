"""Compare a fresh simulation with released pairs and replay trained forecasts.

Computation wall time is machine dependent. The original solver's loss and
voltage fields are retained as diagnostics, while the separately measured
high-precision fields used in inference require a fresh electrical replay.
The forecast check evaluates every released test/validation window for all
three trained repetitions, without fitting a model or changing a result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--smoke',type=Path,default=Path('results/reviewer_smoke_local/primary/rollout_scenarios.csv'))
    parser.add_argument('--precision-repeat',type=Path,default=Path('results/reviewer_precision_local'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=args.package_root.resolve()
    sys.path.insert(0,str(root/'src'))
    import torch
    from torch.utils.data import DataLoader
    from models import GraphGRUForecaster, count_parameters
    from train_forecaster import WindowDataset, evaluate, split_indices
    torch.set_num_threads(4)
    expected=pd.read_csv(root/'results/submission_service_20260905/primary/rollout_scenarios.csv')
    actual=pd.read_csv(root/args.smoke)
    keys=['scenario_id','policy']
    require(not actual.duplicated(keys).any(),'Duplicate fresh scenario/policy')
    require(set(actual.columns)==set(expected.columns),'Scenario fields differ')
    expected=expected.merge(actual[keys],on=keys,how='inner').set_index(keys).sort_index()
    actual=actual.set_index(keys).sort_index()
    require(expected.index.equals(actual.index),'Fresh pairs do not match released identifiers')
    require(len(actual)>=16*8,'Fewer than 16 eight-policy paired scenarios')
    require(actual.groupby(level='scenario_id').size().eq(8).all(),'Fresh scenarios do not cover all eight policies')
    differences={}
    superseded_measurements={}
    mismatches=[]
    for column in expected.columns:
        if column=='latency_ms':
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            x,y=expected[column].to_numpy(float),actual[column].to_numpy(float)
            differences[column]=float(np.nanmax(np.abs(x-y)))
            if column in ['losses_mwh','min_voltage_pu']:
                superseded_measurements[column]=differences[column]
            elif not np.allclose(x,y,atol=1e-8,rtol=1e-9,equal_nan=True):
                mismatches.append(column)
        else:
            require(expected[column].fillna('').equals(actual[column].fillna('')),
                    'Fresh route, event, or category differs: '+column)
    print(json.dumps({'scenario_mismatches':mismatches,'maximum_differences':differences}),flush=True)
    from verify_electrical_precision import CANONICAL, check_run, compare_repeat, physical_frame
    frozen=pd.read_csv(root/'results/submission_service_20260905/primary/rollout_scenarios.csv')
    _,precision_provenance=physical_frame(root,frozen)
    precise,precise_hours,_=check_run(root,CANONICAL,frozen,4096)
    repeated,repeated_hours,_=check_run(root,args.precision_repeat,frozen)
    require(len(repeated)>=32*2,'Fewer than 32 fresh paired electrical precision scenarios')
    precision_repeat=compare_repeat(precise,precise_hours,repeated,repeated_hours)
    print(json.dumps({'fresh_electrical_precision':precision_repeat}),flush=True)
    data=np.load(root/'data/external/processed/boulder_ev/boulder_ev_forecast_dataset.npz',allow_pickle=True)
    raw=np.stack([data['pickup'],data['energy']],axis=-1).astype(np.float32)
    log=np.log1p(raw)
    train_end=int(data['split_train_end_index'])
    val_end=int(data['split_val_end_index'])
    train_mean=log[:train_end].reshape(-1,2).mean(axis=0).astype(np.float32)
    train_std=(log[:train_end].reshape(-1,2).std(axis=0)+1e-6).astype(np.float32)
    metrics=pd.read_csv(root/'results/real_ev_strict_20260905/prediction_metrics.csv')
    forecasts=[]
    for _,row in metrics.iterrows():
        suffix=f'{row["mode"]}_seed{int(row["seed"])}'
        checkpoint_path=root/f'results/real_ev_strict_20260905/models/forecaster_{suffix}.pt'
        # These are author-created checkpoints from this checksum-verified
        # snapshot. Do not use this command with an untrusted replacement file.
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        config=checkpoint['args']
        require(np.array_equal(train_mean,checkpoint['mean']),'Mean not training-only')
        require(np.array_equal(train_std,checkpoint['std']),'Scale not training-only')
        require(checkpoint['split_train_end_index']==train_end,'Training boundary changed')
        require(checkpoint['split_val_end_index']==val_end,'Validation boundary changed')
        model=GraphGRUForecaster(2,config['hidden'],config['horizon'],data['adj'],config['dropout'])
        model.load_state_dict(checkpoint['state_dict'])
        require(count_parameters(model)==int(row.parameters),'Parameter count differs')
        features=(log-train_mean)/train_std
        _,val_indices,test_indices=split_indices(len(raw),config['history'],config['horizon'],train_end,val_end)
        for period,indices,prefix in [('validation',val_indices,'validation_'),('test',test_indices,'')]:
            stored=np.load(root/f'results/real_ev_strict_20260905/{prefix}forecast_{suffix}.npz')
            require(np.array_equal(stored['indices'],indices),'Forecast window indices differ')
            loader=DataLoader(WindowDataset(features,indices,config['history'],config['horizon']),
                              batch_size=config['batch_size'],shuffle=False,num_workers=0)
            result=evaluate(model,loader,torch.device('cpu'),torch.from_numpy(train_mean),torch.from_numpy(train_std))
            require(np.allclose(result['pred'],stored['pred'],rtol=2e-6,atol=2e-5),'Model predictions differ: '+suffix+' '+period)
            require(np.allclose(result['truth'],stored['truth'],rtol=2e-6,atol=2e-5),'Stored forecast truths differ')
            if period=='validation':
                val_score=result['mae_demand']+.25*result['mae_energy']
                require(abs(val_score-row.best_val_score)<2e-6,'Validation selection score differs')
            else:
                for metric in ['mae_demand','rmse_demand','mae_energy','rmse_energy']:
                    require(abs(result[metric]-row[metric])<2e-5,'Reported forecast metric differs: '+metric)
            forecasts.append(dict(model=suffix,period=period,windows=len(indices),
                prediction_values=int(stored['pred'].size),
                maximum_prediction_difference=float(np.max(np.abs(result['pred']-stored['pred'])))))
            print(f'Forecast replay passed {suffix} {period}',flush=True)
    report=dict(status='PASS' if not mismatches else 'NUMERICAL_REVIEW_REQUIRED',package_root=str(root),fresh_rows=len(actual),
        scenario_mismatches=mismatches,
        fresh_scenarios=actual.index.get_level_values('scenario_id').nunique(),
        scenario_maximum_absolute_differences=differences,
        excluded_field='latency_ms is computation wall time and machine dependent',
        superseded_raw_measurement_differences=superseded_measurements,
        electrical_precision_provenance=precision_provenance,
        fresh_electrical_precision=precision_repeat,
        forecasts=forecasts,scope='Fresh paired execution and complete checkpoint forecast replay. Not retraining or field validation.')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    require(not mismatches,'Fresh numerical differences require review: '+', '.join(mismatches))


if __name__=='__main__':
    main()
