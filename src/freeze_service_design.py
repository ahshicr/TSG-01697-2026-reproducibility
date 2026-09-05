"""Record the validation decision before running the revised test experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import simulate_rollout_revised as sim
import service_route_scoring as score


def main():
    root = Path('results/service_score_validation_20260905')
    target = root/'method_decision.json'
    if target.exists():
        raise RuntimeError('A method decision already exists and must not be overwritten')
    validation = json.loads((root/'completion.json').read_text(encoding='utf-8'))
    assert validation['rows'] == 512*5
    selected = sim.read_best_forecast(Path('results/real_ev_strict_20260905/prediction_metrics.csv'))
    locked = [Path(sim.__file__), Path(score.__file__), Path(sim.solve_alpha.__code__.co_filename),
              Path('results/calibration_strict_20260905/transition_parameter_uncertainty.csv'),
              Path('results/packet_training_20260905/packet_network_scenarios.csv'), selected]
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in locked}
    assert all(validation['source_sha256'][key] == value for key,value in hashes.items())
    decision = dict(
        selected_design='continuous_service_score', frozen_unix=time.time(),
        locked_source_sha256=hashes,
        validation_completion_sha256=hashlib.sha256((root/'completion.json').read_bytes()).hexdigest(),
        rationale='The service-cost score follows the stated service equations and lowers validation cost relative to both exposure-score versions.',
        required_qualification='Transition rollout adds no statistically resolved benefit over the same service score without transition on validation. No transition superiority or robust superiority is established.',
        main_controls=['forecast_matched','pc_rollout','robust_pc_rollout',
                       'exposure_matched','exposure_central','static','greedy','oracle'],
        test_rule='All variants and fixed conditions will be reported. No weights, capacities, matrix parameters or forecast settings will be selected from revised test results.',
        historical_context='The test year was previously inspected for diagnosis of the historical method. This is a frozen revision evaluation, not a claim that the year was never inspected.',
        validation_comparisons=validation['comparisons'])
    target.write_text(json.dumps(decision,indent=2),encoding='utf-8')
    print(json.dumps(decision,indent=2))


if __name__ == '__main__':
    main()
