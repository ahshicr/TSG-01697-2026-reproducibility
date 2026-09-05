"""Integrated experiments after validation-only route-score development.

The method decision and its source hashes must be frozen before this runner
reads test outcomes. Historical exposure-score results remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

import simulate_rollout_revised as sim
import service_route_scoring as score

EXPOSURE_SELECTOR = sim.route_portfolio_plan


UNSUPPORTED = ('comm_persistence', 'pc_coupling', 'rc_coupling', 'cr_coupling', 'cp_coupling')
CONDITIONS = {
    'primary': {'scenarios': 4096},
    'electrical_stress': {'energy_scale': 20., 'charge_capacity_factor': 24.4},
    'backup_0': {'packet_traffic_multiplier': 2., 'packet_backup_duration_s': 0.},
    'backup_60': {'packet_traffic_multiplier': 2., 'packet_backup_duration_s': 60.},
    'backup_300': {'packet_traffic_multiplier': 2., 'packet_backup_duration_s': 300.},
    'matrix_scale_075': {'policy_matrix_scale': .75},
    'matrix_scale_125': {'policy_matrix_scale': 1.25},
    'matrix_noise_015': {'policy_matrix_noise': .15},
    'crews_12': {'integer_crews': 12},
    'crews_24': {'integer_crews': 24},
    **{f'zero_{name}': {'realized_zero': [name]} for name in UNSUPPORTED},
    'zero_all_unsupported': {'realized_zero': list(UNSUPPORTED)},
    'expanded_unsupported': {'realized_range_factor': 2.},
    'nonlinear_ood': {'realized_transition_mode': 'nonlinear_saturation'},
    'weight_mobility_x2': {'cost_mobility': 3.6},
    'weight_energy_x2': {'cost_unserved': 5.2},
    'weight_communication_x2': {'cost_comm': 3.4},
    'weight_power_x2': {'cost_power_service': 2.4},
    'weight_cascade_half': {'cost_cascade': 6.},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(task):
    identifier, group = task
    rng = np.random.default_rng(sim.G['seed'] + identifier * 7919)
    hours = sim.G['valid_first_hours']
    first = int(hours[int(rng.integers(len(hours)))])
    threat, innovations = sim.scenario_inputs(rng, group)
    coefficients = {name: float(rng.uniform(*bounds))
                    for name, bounds in sim.G['transition_uncertainty_bounds'].items()}
    for name in sim.G.get('realized_zero', []):
        coefficients[name] = 0.
    for name in UNSUPPORTED:
        coefficients[name] *= sim.G.get('realized_range_factor', 1.)
    crew = sim.prepare_crew_scenario(rng, threat)
    mapping = identifier % len(sim.G['smartds_mappings'])
    rows = []
    for policy in sim.G['policies']:
        base_policy = {'exposure_matched': 'forecast_matched',
                       'exposure_central': 'pc_rollout'}.get(policy, policy)
        sim.route_portfolio_plan = (EXPOSURE_SELECTOR if policy.startswith('exposure_')
                                    else score.select_service_route)
        outcome = sim.evaluate_policy(base_policy, first, threat, innovations, crew, mapping, coefficients)
        rows.append(dict(scenario_id=identifier, group=group, first_hour=first,
            policy=policy, smartds_mapping_index=mapping,
            **{f'true_{name}': value for name, value in coefficients.items()}, **outcome))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/submission_service_20260905'))
    parser.add_argument('--freeze', type=Path, default=Path('results/service_score_validation_20260905/method_decision.json'))
    parser.add_argument('--forecast-results', type=Path, default=Path('results/real_ev_strict_20260905'))
    parser.add_argument('--calibration', type=Path, default=Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv'))
    parser.add_argument('--packets', type=Path, default=Path('results/packet_training_20260905/packet_network_scenarios.csv'))
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--smoke-scenarios', type=int, default=None)
    args = parser.parse_args()
    decision = json.loads(args.freeze.read_text(encoding='utf-8'))
    if decision.get('selected_design') != 'continuous_service_score':
        raise RuntimeError('Service-score design has not been selected and frozen')
    selected = sim.read_best_forecast(args.forecast_results/'prediction_metrics.csv')
    paths = [Path(__file__), Path(sim.__file__), Path(score.__file__),
             Path(sim.solve_alpha.__code__.co_filename), args.calibration, args.packets, selected]
    sources = {path.name: digest(path) for path in paths}
    for key, value in decision['locked_source_sha256'].items():
        if sources.get(key) != value:
            raise RuntimeError(f'Source changed after validation freeze: {key}')
    for name in args.conditions:
        changes = CONDITIONS[name]
        count = args.smoke_scenarios or changes.get('scenarios', 1024)
        out = args.output/name
        marker = out/'completion.json'
        if marker.exists():
            existing = json.loads(marker.read_text(encoding='utf-8'))
            if (existing['source_sha256'] != sources or existing['scenario_count'] != count
                    or digest(out/'rollout_scenarios.csv') != existing['scenario_sha256']):
                raise RuntimeError(f'Existing experiment differs: {out}')
            print(f'Already complete {name}', flush=True)
            continue
        config = sim.build_parser().parse_args([])
        config.forecast_results = args.forecast_results
        config.robust_uncertainty_file = args.calibration
        config.packet_results = args.packets
        custom = {}
        for key, value in changes.items():
            if key.startswith('realized_'):
                custom[key] = value
            elif key != 'scenarios':
                setattr(config, key, value)
        payload = sim.payload_from_args(config)
        payload.update(custom)
        # Require an issued forecast at every hourly decision in each episode.
        forecast_hours = set(payload['forecast_index'])
        payload['valid_first_hours'] = np.asarray([hour for hour in payload['valid_first_hours']
            if all(hour + h in forecast_hours for h in range(payload['horizon']))])
        if custom or name.startswith('weight_'):
            payload['policies'] = ['forecast_matched', 'pc_rollout', 'robust_pc_rollout']
        elif name == 'primary':
            payload['policies'] += ['exposure_matched', 'exposure_central']
        started = time.time()
        specification = dict(condition=name, changes=changes, scenario_count=count,
            source_sha256=sources, design_freeze_sha256=digest(args.freeze),
            period='2023 chronological test after 2022 route-score validation',
            policies=payload['policies'], horizon=payload['horizon'], random_state=payload['seed'],
            normalization_rule=payload['normalization_rule'], training_end=payload['normalization_train_end'],
            crew_prior_cutoff=payload['crew_prior_cutoff'], crew_prior_events=payload['crew_prior_events'],
            crew_prior_observed_times=len(payload['crew_recovery_h']),
            budgets={key:payload[key] for key in ('total_charge','total_comm','total_restore')},
            objective_weights={key:payload[key] for key in ('cost_mobility','cost_unserved','cost_comm','cost_power_service','cost_cascade')},
            central_policy_coefficients=payload['central_policy_coefficients'],
            realized_reference_bounds=payload['transition_uncertainty_bounds'],
            route_selection='Same continuous service objective for central and no-transition controls.',
            weight_sensitivity='When weights change, all candidate scores and executed costs use those weights.',
            started_unix=started)
        out.mkdir(parents=True, exist_ok=True)
        (out/'specification.json').write_text(json.dumps(specification,indent=2),encoding='utf-8')
        rows = []
        tasks = [(i, sim.GROUPS[i%len(sim.GROUPS)]) for i in range(count)]
        print(f'Starting {name}: {count} scenarios', flush=True)
        with mp.Pool(args.workers, initializer=score.init_worker, initargs=(payload,)) as pool:
            for index, chunk in enumerate(pool.imap_unordered(evaluate,tasks,chunksize=4),1):
                rows.extend(chunk)
                if index%128==0 or index==count:
                    print(f'{name}: {index}/{count}, {time.time()-started:.1f}s', flush=True)
        rows.sort(key=lambda row:(row['scenario_id'],row['policy']))
        assert len(rows)==count*len(payload['policies'])
        assert len({(r['scenario_id'],r['policy']) for r in rows})==len(rows)
        assert all(r['smartds_projected_infeasible_hours']==0 for r in rows)
        sim.write_csv(out/'rollout_scenarios.csv', rows)
        sim.write_csv(out/'rollout_summary.csv', sim.aggregate(rows))
        marker.write_text(json.dumps({**specification, 'rows':len(rows),
            'elapsed_s':time.time()-started,
            'scenario_sha256':digest(out/'rollout_scenarios.csv')},indent=2),encoding='utf-8')
        print(f'Completed {name}', flush=True)


if __name__ == '__main__':
    main()
