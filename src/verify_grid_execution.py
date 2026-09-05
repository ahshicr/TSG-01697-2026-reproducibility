"""Verify that the returned charging action equals the actual solved circuit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import smartds_ev_feasibility as grid


def main():
    root = Path(__file__).resolve().parents[1]
    old_path = root / 'results/submission_20260905/source_snapshot/smartds_ev_feasibility.py'
    spec = importlib.util.spec_from_file_location('old_grid', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rng = np.random.default_rng(73491)
    records = []
    for scale in (0., 1., 20., 50., 100., 500.):
        grid.dss.Text.Command('Clear')
        names, kw, kvar, phases = grid.compile_feeder(grid.MASTER)
        added = np.zeros(len(names))
        added[phases == 3] = rng.uniform(.5, 1.5, np.count_nonzero(phases == 3)) * scale
        alpha, raw, accepted = grid.solve_alpha(names, kw, kvar, added, .98, .95, 1.05, 1.)
        actual_kw, actual_kvar = [], []
        for name in names:
            grid.dss.Loads.Name(name)
            actual_kw.append(grid.dss.Loads.kW())
            actual_kvar.append(grid.dss.Loads.kvar())
        assert np.allclose(actual_kw, kw + alpha * added, rtol=0, atol=1e-9)
        assert np.allclose(actual_kvar, kvar + alpha * added * np.tan(np.arccos(.98)),
                           rtol=0, atol=1e-9)
        repeated = grid.circuit_metrics()
        assert grid.feasible(accepted, .95, 1.05, 1.)
        assert grid.feasible(repeated, .95, 1.05, 1.)
        # A base-energized node becoming deenergized must fail the voltage check.
        volts = np.asarray(grid.dss.Circuit.AllBusMagPu(), dtype=float)
        volts[np.flatnonzero(grid.BASE_ACTIVE_NODES)[0]] = 0.
        with patch.object(type(grid.dss.Circuit), 'AllBusMagPu', return_value=volts.tolist()):
            collapsed = grid.circuit_metrics()
        assert collapsed['v_min_pu'] == 0.
        assert not grid.feasible(collapsed, .95, 1.05, 1.)
        grid.dss.Text.Command('Clear')
        old_names, old_kw, old_kvar, _ = old.compile_feeder(grid.MASTER)
        previous_alpha, _, previous_metrics = old.solve_alpha(
            old_names, old_kw, old_kvar, added, .98, .95, 1.05, 1.)
        assert alpha == previous_alpha
        records.append(dict(request_scale=scale, accepted_fraction=alpha,
            raw_feasible=grid.feasible(raw, .95, 1.05, 1.),
            returned_and_solved_actions_match=True,
            repeat_solve_feasible=True, collapse_detected=True,
            old_fraction_preserved=True,
            voltage_replay_difference=accepted['v_min_pu']-previous_metrics['v_min_pu'],
            loading_replay_difference=accepted['max_line_loading_pu']-previous_metrics['max_line_loading_pu']))
    assert any(not r['raw_feasible'] for r in records)
    report = dict(cases=records, all_checks_pass=True,
        scope='Deterministic operator tests, including infeasible requests and a mocked voltage collapse.',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), Path(grid.__file__)]})
    output = root / 'results/grid_execution_checks_20260905.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
