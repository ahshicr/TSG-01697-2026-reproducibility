"""Render manuscript tables from the frozen, verified paired records."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
NAMES={'primary':'Original','matrix_scale_075':r'$0.75M$',
    'matrix_scale_125':r'$1.25M$','matrix_noise_015':'Noise 0.15',
    'expanded_unsupported':'Couplings doubled','zero_all_unsupported':'Couplings zero',
    'nonlinear_ood':'Nonlinear','electrical_stress':'Electrical stress'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(name,caption,label,columns,header,rows):
    value='\n'.join([r'\begin{table}[t]',r'\centering',r'\caption{'+caption+'}',
        r'\label{'+label+'}',r'\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}',
        r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule',
        *[r' & '.join(row)+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (PAPER/name).write_text(value,encoding='utf-8')


def interval(row):
    return f"{row.difference:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]"


def select(frame,**fields):
    for key,value in fields.items():
        frame=frame.loc[frame[key].eq(value)]
    assert len(frame)==1,fields
    return frame.iloc[0]


def main():
    verification=json.loads((ROOT/'results/guarded_verification_20260905.json').read_text())
    assert verification['status']=='PASS'
    home=ROOT/'results/guarded_test_20260905/inference'
    transfer=ROOT/'results/guarded_external_20260905/inference'
    costs=pd.read_csv(home/'paired_costs.csv')
    diagnostics=pd.read_csv(home/'diagnostics.csv')
    external=pd.read_csv(transfer/'paired_costs.csv')
    completion=json.loads((transfer/'completion.json').read_text())
    frozen=json.loads((ROOT/'results/guarded_validation_20260905/method_decision.json').read_text())
    rows=[]
    for condition,title in NAMES.items():
        delta=select(costs,condition=condition,group='published',policy='protected_0.001',comparator='central')
        c=select(diagnostics,condition=condition,group='published',policy='central')
        g=select(diagnostics,condition=condition,group='published',policy='protected_0.001')
        rows.append([title,interval(delta),f'{c.positive_excess:.4f}',f'{g.positive_excess:.4f}'])
    table('table_guarded_primary.tex',
        r'Reference protection on the original common routes. $G$ denotes PC with reference protection and $C$ central PC. $E^+$ is the mean positive excess cost above forecast matched. Intervals are paired marginal 95\% intervals. The original condition has 4,096 pairs and each other condition has 1,024 pairs.',
        'tab:guarded-primary',r'@{}p{.22\columnwidth}p{.40\columnwidth}rr@{}',
        r'Condition & $G-C$ [95\% CI] & $E_C^+$ & $E_G^+$',rows)
    rows=[]
    for kind in ('graph','weekly'):
        for group in ('published','independent'):
            delta=select(external,forecast=kind,condition='primary',group=group,policy='protected_0.001',comparator='central')
            matched=select(external,forecast=kind,condition='primary',group=group,policy='protected_0.001',comparator='matched')
            gate=next(g for g in completion['external_gates'] if g['forecast']==kind and g['group']==group)
            rows.append([('Graph' if kind=='graph' else 'Weekly')+', '+('original' if group=='published' else 'independent'),
                interval(delta),f'{matched.reduction_percent:.4f}',f'{gate["noninferiority_upper"]:.4f}'])
    table('table_guarded_transfer.tex',
        r'Independent municipal charging input with 1,024 pairs per row. Original and independent identify the two common route portfolios. Reduction is relative to forecast matched. The final column is the one-sided 97.5\% upper bound after subtracting the fixed 0.05\% noninferiority margin.',
        'tab:guarded-transfer',r'@{}p{.24\columnwidth}p{.38\columnwidth}rr@{}',
        r'Forecast and routes & $G-C$ [95\% CI] & Red. (\%) & NI upper',rows)
    for group,title in [('published','original'),('independent','parameter-independent')]:
        rows=[]
        for condition,label in NAMES.items():
            values=[select(external,forecast=kind,condition=condition,group=group,
                           policy='protected_0.001',comparator='central')
                    for kind in ('graph','weekly')]
            rows.append([label,*[interval(value) for value in values]])
        table('table_guarded_external_'+group+'.tex',
            'All independent-input conditions on the '+title+r' common routes. '
            r'Differences are protected minus central cost on 1,024 paired scenarios. '
            r'Positive values mean higher protected cost. The 95\% intervals are marginal, '
            r'and mean-test correction covers all 160 external comparisons.',
            'tab:guarded-external-'+group,r'@{}p{.20\columnwidth}p{.37\columnwidth}p{.37\columnwidth}@{}',
            r'Condition & Graph $G-C$ [95\% CI] & Weekly $G-C$ [95\% CI]',rows)
    rows=[]
    for condition,title in NAMES.items():
        observed=select(costs,condition=condition,group='independent',policy='observed',comparator='central')
        guarded=select(costs,condition=condition,group='independent',policy='protected_0.001',comparator='central')
        rows.append([title,interval(observed),interval(guarded)])
    table('table_guarded_ablation.tex',
        r'Separate route construction without transition parameters. $O$ retains only observation supported terms in the score. $G$ uses reference protection. Both differences use central PC on this same route portfolio, not central PC on the original routes. All eight conditions are retained.',
        'tab:guarded-ablation',r'@{}p{.20\columnwidth}p{.37\columnwidth}p{.37\columnwidth}@{}',
        r'Condition & $O-C$ [95\% CI] & $G-C$ [95\% CI]',rows)
    rows=[]
    for option in frozen['all_margin_options']:
        rows.append(['Original' if option['group']=='published' else 'Independent',
            f'{option["rate"]:g}',f'{100*option["primary_relative_penalty"]:.5f}',
            f'{option["equal_condition_positive_excess"]:.5f}'])
    table('table_guarded_validation.tex',
        r'All validation margins on 512 paired scenarios per condition. The third column is the guard minus central cost as a percentage of matched cost in the original condition. The final column averages positive excess above matched over the six perturbed conditions. All rows passed the 0.05\% screen. The fixed rule selected $\nu=0.001$ for each portfolio.',
        'tab:guarded-validation',r'@{}lrrr@{}',
        r'Routes & $\nu$ & Penalty (\%) & Mean $E^+$',rows)
    tails=pd.read_csv(home/'tails.csv')
    rows=[]
    for group in ('published','independent'):
        for metric,name in [('q95',r'95\% quantile'),('q99',r'99\% quantile'),('es95',r'Upper 5\% mean'),('maximum','Observed maximum')]:
            value=select(tails,condition='primary',group=group,metric=metric)
            rows.append(['Original' if group=='published' else 'Independent',name,
                         f'{value.difference:.4f}' if metric=='maximum' else interval(value)])
    table('table_guarded_tails.tex',
        r'Original-condition cost tails for reference protection minus central PC. Intervals use 2,000 paired resamples and are exploratory, without multiplicity correction. The upper mean includes fractional quantile-boundary mass. The sample maximum is descriptive rather than an inferential bound on future extremes.',
        'tab:guarded-tails',r'@{}lp{.28\columnwidth}p{.42\columnwidth}@{}',
        r'Routes & Metric & $G-C$ [95\% CI]',rows)
    # Keep Table V's protected mean bound to the same scenario-derived record.
    primary_guard=select(costs,condition='primary',group='published',
                         policy='protected_0.001',comparator='matched')
    summary_path=PAPER/'table_robust_boulder.tex'
    summary=summary_path.read_text(encoding='utf-8')
    row=(f'PC with reference protection & {primary_guard.proposed_mean:.2f} & '
         f'{primary_guard.reduction_percent:.4f}'+r' \\')
    if 'PC with reference protection &' in summary:
        summary=re.sub(r'(?m)^PC with reference protection &.*$',lambda _:row,summary)
    else:
        summary=re.sub(r'(?m)^(Robust PC &.*)$',lambda m:m[0]+'\n'+row,summary)
    summary_path.write_text(summary,encoding='utf-8')
    outputs=[p for p in sorted(PAPER.glob('table_guarded_*.tex')) if p.name!='table_guarded_parameters.tex']+[summary_path]
    report=dict(status='RESULT_TABLES_GENERATED',tables={p.name:digest(p) for p in outputs},
        documented_parameter_table={'table_guarded_parameters.tex':digest(PAPER/'table_guarded_parameters.tex')},
        script_sha256=digest(__file__),sources={str(p.relative_to(ROOT)):digest(p) for p in
            [home/'paired_costs.csv',home/'diagnostics.csv',home/'tails.csv',transfer/'paired_costs.csv',transfer/'completion.json',
             ROOT/'results/guarded_validation_20260905/method_decision.json',
             ROOT/'config/guarded_parameter_ledger.csv']})
    (ROOT/'results/guarded_table_build_20260905.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
