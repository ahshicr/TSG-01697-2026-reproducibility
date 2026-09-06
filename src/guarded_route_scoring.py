"""Finite-route baseline protection, with a parameter-independent control group.

Selection uses dispatch-time inputs only. Realized propagation, innovations,
electrical solutions and policy costs are deliberately absent from the API.
The original simulator and published route scorer remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import simulate_rollout_revised as sim
import service_route_scoring as service

RATES = (0.0, 0.00025, 0.0005, 0.001)
OBSERVED = ('road_persistence', 'spatial_spread', 'power_persistence', 'pr_coupling')
UNKNOWN = ('comm_persistence', 'pc_coupling', 'rc_coupling', 'cr_coupling', 'cp_coupling')


@dataclass(frozen=True)
class GuardChoice:
    selected: int
    reference: int
    accepted: bool
    upper_difference: float
    margin: float
    numerical_tolerance: float


def choose_protected(scores, reference, central_reference, rate):
    """Minimize the maximum *paired* model difference, not a difference of maxima.

    Equality or an advantage no larger than the numerical tolerance returns the
    reference. Candidate columns have an externally fixed deterministic order.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError('A nonempty, finite model-by-route score matrix is required')
    if not 0 <= reference < scores.shape[1] or not np.isfinite(central_reference):
        raise ValueError('Invalid reference route or normalization')
    if not np.isfinite(rate) or rate < 0:
        raise ValueError('The margin rate must be finite and nonnegative')
    differences = scores - scores[:, reference:reference+1]
    upper = differences.max(axis=0)
    assert upper[reference] == 0.0
    proposed = int(np.argmin(upper))
    margin = float(rate * max(1.0, abs(central_reference)))
    tolerance = float(1e-10 * max(1.0, np.abs(scores).max()))
    accepted = bool(upper[proposed] < -margin-tolerance)
    selected = proposed if accepted else int(reference)
    return GuardChoice(selected, int(reference), accepted, float(upper[selected]), margin, tolerance)


def observation_coefficients(central):
    """Zero is an explicit scoring ablation, not an estimate of unknown reality."""
    if set(central) != set(OBSERVED+UNKNOWN):
        raise ValueError('Every declared transition coefficient must be classified')
    return {name: float(central[name]) if name in OBSERVED else 0.0 for name in central}


def independent_candidates(threat, backlog, forecast, packet_fraction, crew):
    """Six route rules using no transition coefficient, including no hidden proposals."""
    road, power, comm = threat
    direct = (sim.G['restore_road_weight']*road + sim.G['restore_power_weight']*power
              + sim.G['restore_comm_weight']*comm)
    exposure = 1.0 + .25*forecast[0, :, 0]/np.maximum(sim.G['mean_demand'], 1e-6)
    exposure += .25*forecast[0, :, 1]/np.maximum(sim.G['mean_energy'], 1e-6)
    priorities = {
        'training_demand': sim.G['mean_demand'],
        'current_threat': direct,
        'current_exposure': direct*np.clip(exposure, 1.0, 6.0),
        'forecast_demand': forecast[..., 0].sum(axis=0),
        'forecast_energy': forecast[..., 1].sum(axis=0)+sim.G['backlog_score']*backlog,
        'uniform': np.ones(threat.shape[1], dtype=np.float32),
    }
    return {name: sim.build_crew_plan(priority, packet_fraction, crew)
            for name, priority in sorted(priorities.items())}


def assess_candidates(plans, threat, backlog, forecast, central, matrices, *, observed=False):
    names = sorted(plans)
    if not names or not matrices:
        raise ValueError('At least one route and one declared matrix are required')
    def cost(name, matrix, propagate=True):
        return service.projected_service_cost(threat, forecast, plans[name], matrix,
            propagate=propagate, initial_backlog=backlog)
    matched = np.array([cost(name, central, False) for name in names], dtype=float)
    center = np.array([cost(name, central) for name in names], dtype=float)
    ensemble = np.array([[cost(name, matrix) for name in names] for matrix in matrices], dtype=float)
    reference = int(np.argmin(matched))
    choices = {
        'matched': (reference, matched[reference], None),
        'central': (int(np.argmin(center)), float(center.min()), None),
        'robust': (int(np.argmin(ensemble.max(axis=0))), float(ensemble.max(axis=0).min()), None),
    }
    if observed:
        coefficients = observation_coefficients(central)
        observed_scores = np.array([cost(name, coefficients) for name in names], dtype=float)
        choices['observed'] = (int(np.argmin(observed_scores)), float(observed_scores.min()), None)
    for rate in RATES:
        choice = choose_protected(ensemble, reference, center[reference], rate)
        choices[f'protected_{rate:g}'] = (choice.selected, choice.upper_difference, choice)
    output = {}
    for policy, (index, value, guard) in choices.items():
        plan = dict(plans[names[index]])
        plan.update(source_policy=names[index], route_score=float(value))
        metadata = dict(selected_candidate=names[index], reference_candidate=names[reference],
            guard_accepted=int(guard.accepted) if guard is not None else -1,
            guard_upper_difference=guard.upper_difference if guard is not None else 0.0,
            guard_margin=guard.margin if guard is not None else 0.0,
            guard_tolerance=guard.numerical_tolerance if guard is not None else 0.0,
            candidate_count=len(names), model_count=len(matrices))
        output[policy] = (plan, metadata)
    diagnostics = dict(candidate_names=names, reference=reference,
        matched_scores=matched.tolist(), central_scores=center.tolist(),
        matrix_scores=ensemble.tolist())
    return output, diagnostics


def dispatch_decisions(first_hour, threat, backlog, packet_fraction, crew, *, rates=RATES):
    forecast = service.causal_forecast_at(first_hour, sim.G['horizon'])
    central, matrices = sim.G['central_policy_coefficients'], sim.G['robust_policy_variants']
    legacy = sim.route_candidate_plans(first_hour, threat, backlog, packet_fraction, crew)
    independent = independent_candidates(threat, backlog, forecast, packet_fraction, crew)
    results, records = {}, {}
    for group, plans in [('published', legacy), ('independent', independent)]:
        choices, diagnostics = assess_candidates(plans, threat, backlog, forecast, central,
            matrices, observed=group == 'independent')
        for policy, choice in choices.items():
            if policy.startswith('protected_') and float(policy.removeprefix('protected_')) not in rates:
                continue
            results[group+'/'+policy] = choice
        records[group] = diagnostics
    return results, records
