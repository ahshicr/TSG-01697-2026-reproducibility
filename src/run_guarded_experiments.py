"""Paired execution of the predeclared baseline-protection study.

The original source, normalization and results remain unchanged. Candidate
decisions are computed before any policy execution. Equivalent routes share
an execution result, with dispatch scoring time recorded separately.
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
import service_route_scoring as service
import guarded_route_scoring as guard
from run_frozen_service_experiments import CONDITIONS, UNSUPPORTED

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = 'd17f8af1fc74d62deb2ff32de39f502b0bd1d6a76a625bbf41d791f096911aa2'
VALIDATION_CONDITIONS = ('primary', 'matrix_scale_075', 'matrix_scale_125', 'matrix_noise_015',
                         'expanded_unsupported', 'zero_all_unsupported', 'nonlinear_ood')
TEST_CONDITIONS = VALIDATION_CONDITIONS+('electrical_stress',)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(payload):
    sim.init_worker(payload)
    sim.forecast_at = service.causal_forecast_at


def evaluate(task):
    identifier, group = task
    rng = np.random.default_rng(sim.G['seed']+identifier*7919)
    hours = sim.G['valid_first_hours']
    first = int(hours[int(rng.integers(len(hours)))])
    threat, innovations = sim.scenario_inputs(rng, group)
    coefficients = {name: float(rng.uniform(*bounds))
                    for name, bounds in sim.G['transition_uncertainty_bounds'].items()}
    for name in sim.G.get('realized_zero', []):
        coefficients[name] = 0.0
    for name in UNSUPPORTED:
        coefficients[name] *= sim.G.get('realized_range_factor', 1.0)
    crew = sim.prepare_crew_scenario(rng, threat)
    mapping = identifier % len(sim.G['smartds_mappings'])
    before = time.perf_counter()
    actions = sim.packet_action_fraction(threat)
    decisions, diagnostics = guard.dispatch_decisions(first, threat, np.zeros(threat.shape[1], dtype=np.float32),
        actions, crew, rates=sim.G['guard_rates'])
    dispatch_seconds = time.perf_counter()-before
    # The decision function receives none of the sampled future inputs above.
    executed = {}
    rows = []
    for policy, (plan, metadata) in decisions.items():
        key = (plan['route_log'], plan['event_log'], tuple(plan['completion_by_zone']),
               tuple(plan['requested_jobs']), tuple(plan['dispatched_jobs']), plan['total_travel_h'])
        if key not in executed:
            # All compared policies use identical service allocations for a
            # fixed route. The old policy-specific restoration priority is not
            # executed when the integer plan override is present.
            executed[key] = sim.evaluate_policy('forecast_matched', first, threat, innovations,
                crew, mapping, coefficients, crew_plan_override=plan)
        outcome = dict(executed[key])
        outcome.update(crew_route_source_policy=plan['source_policy'], route_score=plan['route_score'])
        rows.append(dict(scenario_id=identifier, group=group, first_hour=first, policy=policy,
            smartds_mapping_index=mapping,
            **{f'true_{name}': value for name, value in coefficients.items()}, **metadata,
            dispatch_all_variants_ms=1000*dispatch_seconds, **outcome))
    details = dict(scenario_id=identifier, group=group, first_hour=first,
        candidate_scores=diagnostics, unique_executions=len(executed), policy_rows=len(rows))
    return rows, details


def source_hashes(args):
    chosen = sim.read_best_forecast(args.forecast_results/'prediction_metrics.csv')
    config = sim.build_parser().parse_args([])
    paths = [Path(__file__), Path(guard.__file__), Path(service.__file__), Path(sim.__file__),
        Path(sim.solve_alpha.__code__.co_filename),
        Path(__file__).with_name('run_frozen_service_experiments.py'),
        args.calibration, args.packets, chosen, chosen.with_name('validation_'+chosen.name),
        config.data]
    return {path.resolve().relative_to(ROOT).as_posix(): digest(path) for path in paths}


def build_payload(args, condition):
    config = sim.build_parser().parse_args([])
    config.forecast_results = args.forecast_results
    config.robust_uncertainty_file = args.calibration
    config.packet_results = args.packets
    custom = {}
    for key, value in CONDITIONS[condition].items():
        if key.startswith('realized_'):
            custom[key] = value
        elif key != 'scenarios':
            setattr(config, key, value)
    payload = sim.payload_from_args(config)
    payload.update(custom)
    if args.phase == 'validation':
        chosen = sim.read_best_forecast(args.forecast_results/'prediction_metrics.csv')
        values = np.load(chosen.with_name('validation_'+chosen.name))
        data = np.load(config.data)
        indices = values['indices'].astype(int)
        assert indices.min() >= int(data['split_train_end_index'])
        assert indices.max()+config.horizon <= int(data['split_val_end_index'])
        payload.update(forecast_index={int(h):i for i,h in enumerate(indices)},
            forecast_pred=values['pred'].astype(np.float32), valid_first_hours=indices,
            seed=9202201)
    hours = set(payload['forecast_index'])
    payload['valid_first_hours'] = np.asarray([h for h in payload['valid_first_hours']
        if all(h+t in hours for t in range(payload['horizon']))])
    if args.phase == 'test':
        frozen = json.loads(args.freeze.read_text(encoding='utf-8'))
        payload['guard_rates'] = tuple(sorted(set(frozen['selected_rates'].values())))
    else:
        payload['guard_rates'] = guard.RATES
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('validation', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forecast-results', type=Path, default=Path('results/real_ev_strict_20260905'))
    parser.add_argument('--calibration', type=Path, default=Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv'))
    parser.add_argument('--packets', type=Path, default=Path('results/packet_training_20260905/packet_network_scenarios.csv'))
    parser.add_argument('--freeze', type=Path, default=Path('results/guarded_validation_20260905/method_decision.json'))
    parser.add_argument('--conditions', nargs='+', choices=TEST_CONDITIONS)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--smoke-scenarios', type=int)
    args = parser.parse_args()
    sources = source_hashes(args)
    protocol = ROOT/'config/guarded_improvement_protocol.md'
    if digest(protocol) != PROTOCOL_SHA:
        raise RuntimeError('Predeclared protocol changed')
    if args.phase == 'test':
        frozen = json.loads(args.freeze.read_text(encoding='utf-8'))
        if frozen['source_sha256'] != sources or frozen['protocol_sha256'] != PROTOCOL_SHA:
            raise RuntimeError('Code or data changed after validation selection')
    selected_conditions = args.conditions or (VALIDATION_CONDITIONS if args.phase == 'validation' else TEST_CONDITIONS)
    if args.phase == 'validation' and set(selected_conditions)-set(VALIDATION_CONDITIONS):
        raise RuntimeError('Unspecified validation condition')
    for condition in selected_conditions:
        count = args.smoke_scenarios or (512 if args.phase == 'validation' else
                                         (4096 if condition == 'primary' else 1024))
        out = args.output/condition
        completed = out/'completion.json'
        if completed.exists():
            record = json.loads(completed.read_text(encoding='utf-8'))
            if record['source_sha256'] != sources or record['scenarios'] != count:
                raise RuntimeError('Preserve completed results from a different specification')
            assert record['scenario_sha256'] == digest(out/'rollout_scenarios.csv')
            assert record['scores_sha256'] == digest(out/'candidate_scores.jsonl')
            print(f'Already complete {args.phase}/{condition}', flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        payload = build_payload(args, condition)
        specification = dict(phase=args.phase, condition=condition, scenarios=count, source_sha256=sources,
            protocol_sha256=PROTOCOL_SHA, guard_rates=list(payload['guard_rates']),
            random_state=payload['seed'], changes=CONDITIONS[condition],
            dispatch_rule='All route decisions precede all realized policy execution.',
            cache_rule='Exact same route shares one deterministic execution, not a predicted outcome.',
            latency_rule='Dispatch all-variant time and repeated service time are separate diagnostics.',
            started_unix=time.time())
        (out/'specification.json').write_text(json.dumps(specification, indent=2), encoding='utf-8')
        print(f'Starting {args.phase}/{condition}: {count} paired scenarios', flush=True)
        tasks = [(i, sim.GROUPS[i%len(sim.GROUPS)]) for i in range(count)]
        rows, details = [], []
        with mp.Pool(args.workers, initializer=initialize, initargs=(payload,)) as pool:
            for index, (part, info) in enumerate(pool.imap_unordered(evaluate, tasks, chunksize=2), 1):
                rows.extend(part)
                details.append(info)
                if index % 64 == 0 or index == count:
                    print(f'{condition}: {index}/{count}, {time.time()-specification["started_unix"]:.1f}s', flush=True)
        rows.sort(key=lambda row:(row['scenario_id'], row['policy']))
        details.sort(key=lambda row:row['scenario_id'])
        assert len({(row['scenario_id'],row['policy']) for row in rows}) == len(rows)
        assert all(row['smartds_projected_infeasible_hours'] == 0 for row in rows)
        sim.write_csv(out/'rollout_scenarios.csv', rows)
        with (out/'candidate_scores.jsonl').open('w', encoding='utf-8') as handle:
            for detail in details:
                handle.write(json.dumps(detail)+'\n')
        completed.write_text(json.dumps({**specification, 'rows':len(rows),
            'seconds':time.time()-specification['started_unix'],
            'scenario_sha256':digest(out/'rollout_scenarios.csv'),
            'scores_sha256':digest(out/'candidate_scores.jsonl')}, indent=2), encoding='utf-8')
        print(f'Completed {args.phase}/{condition}', flush=True)


if __name__ == '__main__':
    main()
