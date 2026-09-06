"""Verify a public snapshot and freshly execute the protected route method.

No municipal user-level source table or HPC account is needed. This is a small
execution correspondence check, not the complete statistical experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import run_guarded_experiments as runner
import run_independent_guarded_experiments as external
from unpack_guarded_records import ROOT, digest, restore


def verify_snapshot():
    with (ROOT / 'SHA256SUMS.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Empty or repeated manifest paths')
    for row in rows:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Manifest path outside package')
        if path.stat().st_size != int(row['bytes']) or digest(path) != row['sha256']:
            raise ValueError('Snapshot content changed: ' + row['path'])
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-statistics', action='store_true',
                        help='Also independently recompute every new mean-test family and adoption interval.')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Use a fresh report path')
    os.chdir(ROOT)
    started = time.time()
    verified = verify_snapshot()
    restored = restore()
    specification = argparse.Namespace(phase='test',
        inputs=Path('data/external/processed/palo_alto_independent_20260905'),
        forecast_results=Path('results/real_ev_strict_20260905'),
        calibration=Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv'),
        packets=Path('results/packet_training_20260905/packet_network_scenarios.csv'),
        freeze=Path('results/guarded_validation_20260905/method_decision.json'))
    frozen = json.loads(specification.freeze.read_text(encoding='utf-8'))
    if frozen['source_sha256'] != runner.source_hashes(specification):
        raise ValueError('Frozen source or model differs')
    if digest(ROOT/'config/guarded_improvement_protocol.md') != runner.PROTOCOL_SHA:
        raise ValueError('Changed method protocol')
    preparation = json.loads((specification.inputs/'preparation.json').read_text(encoding='utf-8'))
    for name, expected in preparation['outputs'].items():
        if digest(specification.inputs/name) != expected:
            raise ValueError('Changed external input: '+name)
    fields = ['cost', 'unserved_energy', 'mobility_delay', 'comm_loss',
              'crew_completion_events', 'crew_routes', 'crew_jobs_completed',
              'crew_job_count', 'crew_jobs_dispatched', 'requested_charge_capacity',
              'executed_charge_capacity', 'smartds_projected_infeasible_hours',
              'selected_candidate', 'reference_candidate', 'guard_accepted']
    executions = []
    for phase in ('domestic', 'graph', 'weekly'):
        for condition in ('primary', 'electrical_stress'):
            payload = (runner.build_payload(specification, condition) if phase == 'domestic'
                       else external.external_payload(specification, condition, phase))
            runner.initialize(payload)
            directory = (ROOT/'results/guarded_test_20260905' if phase == 'domestic'
                         else ROOT/'results/guarded_external_20260905'/phase)
            saved = pd.read_csv(directory/condition/'rollout_scenarios.csv', keep_default_na=False)
            indexed = saved.set_index(['scenario_id', 'policy'])
            # Include early and later scenario IDs, with exactly the published inputs.
            for identifier in (0, 1, 127, 767):
                rows, detail = runner.evaluate((identifier, runner.sim.GROUPS[identifier % 4]))
                for row in rows:
                    reference = indexed.loc[(identifier, row['policy'])]
                    for field in fields:
                        if isinstance(row[field], str):
                            if row[field] != reference[field]:
                                raise AssertionError((phase, condition, identifier, field))
                        else:
                            np.testing.assert_allclose(row[field], reference[field], rtol=1e-12, atol=1e-9)
                executions.append(dict(phase=phase, condition=condition, scenario_id=identifier,
                                       policy_rows=len(rows), unique_executions=detail['unique_executions']))
            print('Fresh execution agrees: '+phase+'/'+condition, flush=True)
    statistics = None
    if args.full_statistics:
        statistics_path = args.output.with_name(args.output.stem+'_statistics.json')
        subprocess.run([sys.executable, 'src/verify_guarded_records.py', '--output',
                        str(statistics_path)], check=True)
        statistics = json.loads(statistics_path.read_text(encoding='utf-8'))
        statistics = {key: value for key, value in statistics.items() if key != 'records'}
    result = dict(status='PASS', snapshot_files_verified=verified, restored_records=restored,
        scenario_replays=len(executions), policy_rows_reexecuted=sum(row['policy_rows'] for row in executions),
        executions=executions, statistics=statistics, script_sha256=digest(__file__),
        elapsed_seconds=time.time()-started,
        scope='Public-input protected, central, robust, matched and observed-score execution on both route groups. '
              'No retraining, raw session reconstruction, universal theorem validation or field-validity claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key != 'executions'}, indent=2))


if __name__ == '__main__':
    main()
