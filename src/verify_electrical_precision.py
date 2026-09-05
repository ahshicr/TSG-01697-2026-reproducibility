"""Check independent numerical measurements without modifying frozen decisions.

The original execution records are immutable. Only the two electrical
measurement columns used for secondary inference are replaced in a returned
copy, after full pairing, hourly aggregation and unchanged-policy checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICIES = ['forecast_matched', 'pc_rollout']
KEYS = ['scenario_id', 'policy']
MEASUREMENTS = ['losses_mwh', 'min_voltage_pu']
# Measurement resolution checks, not hypothesis-test acceptance thresholds.
# The loss tolerance is 0.2 Wh over the six-hour scenario. No policy outcome
# is allowed to change to satisfy these numerical comparisons.
RESOLUTION = {'losses_mwh': 2e-7, 'min_voltage_pu': 2e-9}
CANONICAL = Path('results/electrical_precision_20260905')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_run(root, folder, original, expected_count=None):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = root / folder
    completion = json.loads((folder / 'completion.json').read_text(encoding='utf-8'))
    decision = json.loads((root / 'results/service_score_validation_20260905/method_decision.json').read_text(encoding='utf-8'))
    require(completion['frozen_source_sha256'] == decision['locked_source_sha256'], 'Precision method differs from frozen design')
    require(completion['precision_source_sha256'] == digest(root / 'src/replay_electrical_precision.py'), 'Precision generating source differs')
    require(completion['tolerances'] == [1e-8, 1e-10] and completion['maximum_iterations'] == 100, 'Precision solver settings differ')
    for name, key in [('scenario_measurements.csv', 'scenario_sha256'), ('hourly_accepted_loads.csv', 'hourly_sha256')]:
        require(digest(folder / name) == completion[key], 'Precision data digest differs: ' + name)
    frame = pd.read_csv(folder / 'scenario_measurements.csv').set_index(KEYS).sort_index()
    trace = pd.read_csv(folder / 'hourly_accepted_loads.csv').set_index(KEYS + ['hour']).sort_index()
    require(frame.index.is_unique and trace.index.is_unique, 'Duplicate precision pair or hour')
    count = completion['scenarios']
    require(expected_count is None or count == expected_count, 'Wrong precision scenario count')
    expected_keys = pd.MultiIndex.from_product([range(count), POLICIES], names=KEYS)
    require(frame.index.equals(expected_keys), 'Incomplete precision policy pairs')
    expected_hours = pd.MultiIndex.from_product([range(count), POLICIES, range(6)], names=KEYS + ['hour'])
    require(trace.index.equals(expected_hours), 'Incomplete precision hourly records')
    require(len(frame) == completion['rows'] and len(trace) == completion['hours'], 'Precision completion counts differ')
    expected = original.set_index(KEYS).sort_index().loc[expected_keys]
    unchanged = []
    for column in expected.columns:
        if column in MEASUREMENTS + ['latency_ms']:
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            require(np.allclose(expected[column], frame[column], atol=1e-12, rtol=1e-12, equal_nan=True), 'Policy output changed: ' + column)
        else:
            require(expected[column].fillna('').equals(frame[column].fillna('')), 'Policy route/event changed: ' + column)
        unchanged.append(column)
    group = trace.groupby(level=KEYS)
    tolerance_difference = {}
    for suffix in ['tol8', 'tol10']:
        require(trace['feasible_' + suffix].eq(True).all(), 'High precision infeasible action')
        numeric = trace[[f'{name}_{suffix}' for name in ['loss_mwh', 'vmin_pu', 'vmax_pu', 'max_line_pu']]].to_numpy()
        require(np.isfinite(numeric).all(), 'Nonfinite electrical measurement')
        require(trace['vmin_pu_' + suffix].ge(.95).all() and trace['vmax_pu_' + suffix].le(1.05).all(), 'Voltage bounds fail')
        require(trace['max_line_pu_' + suffix].le(1.).all(), 'Line bound fails')
        require(trace['iterations_' + suffix].between(1, 100).all(), 'Iteration count outside settings')
        require(frame['precision_infeasible_hours_' + suffix].eq(0).all(), 'Scenario infeasibility differs')
        for metric, field, operation in [('losses_mwh', 'loss_mwh', 'sum'), ('min_voltage_pu', 'vmin_pu', 'min')]:
            aggregated = getattr(group[field + '_' + suffix], operation)()
            require(np.allclose(frame['precision_' + metric + '_' + suffix], aggregated, rtol=1e-12, atol=1e-12), 'Hourly aggregation differs: ' + metric)
    for metric in MEASUREMENTS:
        delta = np.abs(frame['precision_' + metric + '_tol8'] - frame['precision_' + metric + '_tol10'])
        tolerance_difference[metric] = float(delta.max())
        require(delta.max() <= RESOLUTION[metric], 'Solver tolerance resolution not met: ' + metric)
    # Validate recorded accepted actions as physical nonnegative load vectors.
    loads = np.asarray([json.loads(value) for value in trace.accepted_ev_load_kw], dtype=float)
    require(loads.shape[0] == len(trace) and loads.ndim == 2 and np.isfinite(loads).all() and (loads >= 0).all(), 'Invalid accepted loads')
    return frame, trace, dict(scenarios=count, rows=len(frame), hours=len(trace),
        scenario_sha256=completion['scenario_sha256'], hourly_sha256=completion['hourly_sha256'],
        unchanged_policy_columns=unchanged, maximum_tolerance_difference=tolerance_difference)


def compare_repeat(reference, reference_trace, repeat, repeat_trace):
    original = reference.loc[repeat.index]
    hours = reference_trace.loc[repeat_trace.index]
    require(hours.mapping_index.equals(repeat_trace.mapping_index), 'Repeat mapping changed')
    for field in ['accepted_ev_load_kw', 'retained_fraction']:
        if field == 'accepted_ev_load_kw':
            a = np.asarray([json.loads(v) for v in hours[field]])
            b = np.asarray([json.loads(v) for v in repeat_trace[field]])
        else:
            a, b = hours[field], repeat_trace[field]
        require(np.allclose(a, b, rtol=1e-12, atol=1e-12), 'Repeat accepted action changed')
    maximum = {}
    for metric in MEASUREMENTS:
        column = 'precision_' + metric + '_tol10'
        maximum[metric] = float(np.abs(original[column] - repeat[column]).max())
        require(maximum[metric] <= RESOLUTION[metric], 'Repeat measurement resolution not met: ' + metric)
    return dict(scenarios=len(repeat)//2, maximum_repeat_difference=maximum, accepted_actions_unchanged=True)


def physical_frame(root, original):
    """Return a copy with verified loss/voltage measurements for the two controls."""
    root = Path(root).resolve()
    precise, trace, report = check_run(root, CANONICAL, original, 4096)
    repeats = {}
    for relative in ['results/electrical_precision_smoke_20260905', 'results/electrical_precision_reverse_20260905']:
        repeat, repeat_trace, checked = check_run(root, relative, original, 32)
        repeats[relative] = dict(**compare_repeat(precise, trace, repeat, repeat_trace),
                                 scenario_sha256=checked['scenario_sha256'], hourly_sha256=checked['hourly_sha256'])
    output = original.set_index(KEYS).copy()
    for metric in MEASUREMENTS:
        output.loc[precise.index, metric] = precise['precision_' + metric + '_tol10']
    metadata = dict(**report, folder=CANONICAL.as_posix(), measurement_tolerance=1e-10,
        comparison_tolerance=1e-8, maximum_iterations=100, resolution=RESOLUTION, repeats=repeats,
        replaced_secondary_measurements=MEASUREMENTS,
        scope='Independent measurements of identical accepted loads. Frozen decisions and original records unchanged.')
    return output.reset_index(), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fresh-repeat', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    original = pd.read_csv(root / 'results/submission_service_20260905/primary/rollout_scenarios.csv')
    _, report = physical_frame(root, original)
    if args.fresh_repeat:
        reference, hours, _ = check_run(root, CANONICAL, original, 4096)
        repeat, repeat_hours, _ = check_run(root, args.fresh_repeat, original)
        report['fresh_repeat'] = compare_repeat(reference, hours, repeat, repeat_hours)
    report['status'] = 'PASS'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
