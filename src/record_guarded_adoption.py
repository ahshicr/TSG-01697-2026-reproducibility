"""Record the fixed acceptance decision without overwriting validation choices."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'results/guarded_adoption_20260905.json'
    if output.exists():
        raise RuntimeError('Preserve the recorded adoption decision')
    paths={
        'selection':ROOT/'results/guarded_validation_20260905/method_decision.json',
        'external_inference':ROOT/'results/guarded_external_20260905/inference/completion.json',
        'records_verification':ROOT/'results/guarded_verification_20260905.json',
        'independent_input_verification':ROOT/'results/independent_input_verification_20260905.json',
        'design_checks':ROOT/'results/guarded_design_checks_final_20260905.json',
        'mathematical_statement':ROOT/'paper/theory_reference_protection.tex',
        'method_statement':ROOT/'paper/method_reference_protection.tex'}
    record={name:json.loads(path.read_text(encoding='utf-8')) for name,path in paths.items() if path.suffix=='.json'}
    assert all(record[name]['status']=='PASS' for name in
        ('records_verification','independent_input_verification','design_checks'))
    for name in ('external_inference','records_verification'):
        assert record[name]['frozen_sha256']==digest(paths['selection'])
    proof=paths['mathematical_statement'].read_text(encoding='utf-8')
    assert all(fragment in proof for fragment in
        [r'Theorem 3',r'\textit{Proof.}',r'\label{eq:differential-error-condition}',
         r'\label{eq:protected-execution-bound}',r'\varepsilon_{\rm num}',
         'outside the continuous map of Corollary 1'])
    decisions=record['external_inference']['adoption_decisions']
    assert all(row['numerical_performance_adoption_pass'] for row in decisions)
    result=dict(status='ADOPT_REFERENCE_PROTECTION',
        margin_ratios=record['selection']['selected_rates'],
        original_routes='Primary protected route comparison, retaining central and worst absolute score controls.',
        independent_routes='Separate construction and scoring ablation, not an isolated between-portfolio score comparison.',
        passed=decisions,
        mathematical_scope='Complete finite-score derivation and conditional execution bound in Theorem 3. The numerical checker is not a theorem prover.',
        interpretation='Reduced positive excess relative to matched with fixed average-cost noninferiority criteria, not uniform total-cost tail superiority.',
        deliverable_status='Manuscript, correspondence, references, layout, marked changes and public release require final synchronization.',
        evidence={name:{'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)} for name,path in paths.items()},
        script_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
