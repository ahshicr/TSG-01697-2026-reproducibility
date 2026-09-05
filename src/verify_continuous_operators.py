"""Numerical checks alongside, not in place of, the operator proofs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import simulate_rollout_revised as sim
import service_route_scoring as score
from train_forecaster import split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/submission_operator_checks.json'))
    args = parser.parse_args()
    config = sim.build_parser().parse_args([])
    config.forecast_results = Path('results/real_ev_strict_20260905')
    config.robust_uncertainty_file = Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv')
    config.packet_results = Path('results/packet_training_20260905/packet_network_scenarios.csv')
    payload = sim.payload_from_args(config)
    adjacency = payload['adj'].astype(float)
    identity = np.eye(len(adjacency))
    zero_block = np.zeros_like(adjacency)
    c = payload['central_policy_coefficients']
    blocks = [c[name]*identity+c['spatial_spread']*adjacency
              for name in ['road_persistence','power_persistence','comm_persistence']]
    matrix = np.block([[blocks[0],c['pr_coupling']*identity,c['cr_coupling']*identity],
        [zero_block,blocks[1],c['cp_coupling']*adjacency],
        [c['rc_coupling']*adjacency,c['pc_coupling']*identity,blocks[2]]])
    spectral_radius = float(np.abs(np.linalg.eigvals(matrix)).max())
    operator_norm = float(np.linalg.norm(matrix,2))
    assert matrix.min()>=0 and operator_norm>=spectral_radius
    sim.G.clear()
    sim.G.update(payload)
    sim.G['smartds_online'] = False
    sim.G['packet_response_surface'] = None
    score.install()
    rng = np.random.default_rng(71430)
    differences = []
    for trial in range(24):
        n, h = sim.G['adj'].shape[0], sim.G['horizon']
        first = 12
        forecast = rng.uniform(0, 1, (h, n, 2)).astype(np.float32)
        forecast[:, :, 1] *= 15
        raw = np.zeros((first + 2*h, n, 2), np.float32)
        raw[first:first+h] = forecast
        sim.G['raw'] = raw
        sim.G['forecast_index'] = {first+s:s for s in range(h)}
        sim.G['forecast_pred'] = np.stack([
            np.concatenate([forecast[s:], np.repeat(forecast[-1:], s, axis=0)]) for s in range(h)])
        initial = rng.uniform(0, .9, (3,n)).astype(np.float32)
        completion = rng.uniform(0, 8, n)
        plan = dict(requested_jobs=np.arange(n), dispatched_jobs=np.arange(n),
                    completion_by_zone=completion, total_travel_h=1., route_log='',
                    event_log='', source_policy='fixed', route_score=0.)
        expected = score.projected_service_cost(initial, forecast, plan, sim.G['central_policy_coefficients'])
        actual = sim.evaluate_policy('forecast_matched', first, initial,
            np.zeros((h,3,n),np.float32), {}, 0, sim.G['central_policy_coefficients'],
            crew_plan_override=plan)
        differences.append(abs(actual['cost']-expected))
        assert np.isclose(actual['cost'], expected, atol=.01, rtol=1e-6), (trial, actual['cost'],expected)
        # Future observations and sampled realized matrices are not score inputs.
        sim.G['raw'][first:] = rng.uniform(100,200,sim.G['raw'][first:].shape)
        assert score.projected_service_cost(initial, forecast, plan,
                   sim.G['central_policy_coefficients']) == expected
    # Missing forecast uses only the previous observation.
    sim.G['forecast_index'] = {}
    sim.G['raw'][11] = 7
    assert np.all(score.causal_forecast_at(12, 6) == 7)
    tr, va, te = split_indices(51924,12,6,35047,43807)
    assert tr.max()+6 <= 35047 and va.min()>=35047 and va.max()+6<=43807 and te.min()>=43807
    # LayerNorm Jacobians cover constant, nearly constant and dispersed inputs.
    ln = torch.nn.LayerNorm(8, eps=1e-5).double()
    ln.weight.data.copy_(torch.arange(1,9,dtype=torch.float64)/4)
    layer_ratios = []
    for scale in [0,1e-9,1e-4,1,100]:
        vec = torch.tensor(rng.normal(size=8)*scale, dtype=torch.float64,requires_grad=True)
        jac = torch.autograd.functional.jacobian(ln,vec)
        upper = float(ln.weight.detach().abs().max()) / np.sqrt(ln.eps)
        ratio = float(torch.linalg.matrix_norm(jac,ord=2)) / upper
        assert ratio <= 1+1e-10
        layer_ratios.append(ratio)
    grid = torch.linspace(-12,12,10001,dtype=torch.float64,requires_grad=True)
    torch.nn.functional.gelu(grid, approximate='none').sum().backward()
    gelu_max = float(grid.grad.abs().max())
    assert gelu_max <= 1+1/np.sqrt(2*np.pi*np.e)
    report = {
        'neutral_execution_matches_continuous_cost': True,
        'neutral_execution_cases': len(differences), 'maximum_absolute_cost_roundoff': max(differences),
        'future_raw_not_read_by_score': True, 'missing_forecast_is_past_only': True,
        'training_target_windows_disjoint': True, 'validation_target_windows_disjoint': True,
        'central_transition_spectral_radius': spectral_radius,
        'central_transition_euclidean_norm': operator_norm,
        'layernorm_jacobian_bound_checks': layer_ratios, 'gelu_max_sampled_derivative': gelu_max,
        'scope': 'Finite numerical checks of real-arithmetic operator formulas, not a global proof.',
        'source_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path(__file__),Path(score.__file__),Path(sim.__file__)]},
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
