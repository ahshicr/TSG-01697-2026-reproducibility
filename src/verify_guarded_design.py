"""Algebra, information isolation and actual execution checks for protection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

import guarded_route_scoring as guard
import run_guarded_experiments as runner
import service_route_scoring as service
import simulate_rollout_revised as sim


def algebra_checks():
    # Taking separate maxima would incorrectly accept this alternative.
    assert guard.choose_protected([[100, 99], [0, 10]], 0, 100, 0).selected == 0
    assert guard.choose_protected([[100, 98], [0, -2]], 0, 100, 0).selected == 1
    assert guard.choose_protected([[100, 98], [0, -2]], 0, 100, .03).selected == 0
    assert guard.choose_protected([[2, 2], [4, 4]], 1, 2, 0).selected == 1
    invalid = [([], 0, 1, 0), ([[1, np.nan]], 0, 1, 0), ([[1]], 1, 1, 0),
               ([[1]], 0, 1, -.1), ([[1]], 0, np.inf, 0)]
    for arguments in invalid:
        try:
            guard.choose_protected(*arguments)
        except ValueError:
            continue
        raise AssertionError(('Invalid input accepted', arguments))
    rng = np.random.default_rng(202609051)
    accepted = 0
    for _ in range(2000):
        models, routes = int(rng.integers(1, 23)), int(rng.integers(1, 12))
        scores = rng.normal(100, 30, (models, routes))
        reference = int(rng.integers(routes))
        choice = guard.choose_protected(scores, reference, scores[0, reference], .0005)
        true_model = int(rng.integers(models))
        errors = rng.uniform(-.5, .5, routes)
        true_cost = scores[true_model]+errors
        actual_difference = true_cost[choice.selected]-true_cost[reference]
        assert actual_difference <= choice.upper_difference+1.0+1e-12
        if choice.accepted:
            accepted += 1
            assert np.all(scores[:, choice.selected]-scores[:, reference] < -choice.margin)
        else:
            assert choice.selected == reference and actual_difference == 0.0
    return dict(random_cases=2000, accepted=accepted, invalid_cases=len(invalid),
        separate_maxima_counterexample=True, ties_return_reference=True)


def execution_checks(count):
    args = argparse.Namespace(phase='validation',
        forecast_results=Path('results/real_ev_strict_20260905'),
        calibration=Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv'),
        packets=Path('results/packet_training_20260905/packet_network_scenarios.csv'))
    payload = runner.build_payload(args, 'primary')
    runner.initialize(payload)
    sim.route_portfolio_plan = service.select_service_route
    records = []
    for identifier in range(count):
        rng = np.random.default_rng(sim.G['seed']+identifier*7919)
        first = int(sim.G['valid_first_hours'][int(rng.integers(len(sim.G['valid_first_hours'])))])
        threat, innovations = sim.scenario_inputs(rng, sim.GROUPS[identifier%4])
        coefficients = {key:float(rng.uniform(*bounds))
                        for key,bounds in sim.G['transition_uncertainty_bounds'].items()}
        crew = sim.prepare_crew_scenario(rng, threat)
        backlog = np.zeros(threat.shape[1], dtype=np.float32)
        packets = sim.packet_action_fraction(threat)
        choices, _ = guard.dispatch_decisions(first, threat, backlog, packets, crew)
        # Changing unread future demand does not change dispatch selection.
        raw_future = sim.G['raw'][first:first+sim.G['horizon']].copy()
        sim.G['raw'][first:first+sim.G['horizon']] = 10000.0
        future_changed, _ = guard.dispatch_decisions(first, threat, backlog, packets, crew)
        sim.G['raw'][first:first+sim.G['horizon']] = raw_future
        assert all(choices[key][0]['route_log'] == future_changed[key][0]['route_log'] for key in choices)
        assert all(choices[key][1] == future_changed[key][1] for key in choices)
        # Unknown coefficients cannot enter either proposals or the observed score.
        forecast = service.causal_forecast_at(first, sim.G['horizon'])
        plans = guard.independent_candidates(threat, backlog, forecast, packets, crew)
        original_center = dict(sim.G['central_policy_coefficients'])
        original_variants = sim.G['robust_policy_variants']
        changed_center = {key:(999. if key in guard.UNKNOWN else value) for key,value in original_center.items()}
        sim.G['central_policy_coefficients'] = changed_center
        sim.G['robust_policy_variants'] = [changed_center]
        changed_plans = guard.independent_candidates(threat, backlog, forecast, packets, crew)
        assert all(plans[key]['route_log'] == changed_plans[key]['route_log'] for key in plans)
        assert guard.observation_coefficients(changed_center) == guard.observation_coefficients(original_center)
        sim.G['central_policy_coefficients'] = original_center
        sim.G['robust_policy_variants'] = original_variants
        for short, old_policy in [('matched','forecast_matched'),('central','pc_rollout'),('robust','robust_pc_rollout')]:
            plan = choices['published/'+short][0]
            _, _, priority = sim.policy_action(old_policy, first, sim.G['horizon'], threat, backlog, None)
            old_plan = service.select_service_route(old_policy, first, threat, backlog, packets, crew, priority)
            assert plan['route_log'] == old_plan['route_log']
            assert plan['event_log'] == old_plan['event_log']
            assert plan['source_policy'] == old_plan['source_policy']
            original = sim.evaluate_policy(old_policy, first, threat, innovations, crew,
                identifier%len(sim.G['smartds_mappings']), coefficients)
            override = sim.evaluate_policy('forecast_matched', first, threat, innovations, crew,
                identifier%len(sim.G['smartds_mappings']), coefficients, crew_plan_override=plan)
            exact = ['cost', 'crew_completion_events', 'crew_routes', 'unserved_energy',
                'mobility_delay', 'comm_loss', 'requested_charge_capacity', 'executed_charge_capacity',
                'smartds_projected_infeasible_hours', 'crew_completion_fraction']
            assert all(original[key] == override[key] for key in exact), (identifier, short)
        for group in ['published','independent']:
            reference = choices[group+'/matched'][0]
            for rate in guard.RATES:
                plan, info = choices[group+f'/protected_{rate:g}']
                if not info['guard_accepted']:
                    assert plan['route_log'] == reference['route_log']
                    np.testing.assert_array_equal(plan['completion_by_zone'], reference['completion_by_zone'])
        records.append(dict(scenario=identifier, published_policies_reexecuted=3,
            future_isolation=True, observed_parameter_isolation=True, fallback_route_identity=True))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenarios', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the previous check')
    started = time.time()
    algebra = algebra_checks()
    executed = execution_checks(args.scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='PASS', algebra=algebra, execution=executed,
        seconds=time.time()-started, source_sha256={
            Path(p).name:runner.digest(p) for p in [__file__, guard.__file__, runner.__file__, sim.__file__, service.__file__]},
        scope='Finite numerical examples and direct implementation correspondence, not a global proof or field guarantee.')
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
