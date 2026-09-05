"""Render tables from the frozen experiment's single statistical result set.

The default requires all declared conditions. A partial build is for the
working manuscript only and records every omitted table in its manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
RESULTS = ROOT / 'results/submission_service_20260905'
LABELS = {'forecast_matched':'Forecast matched', 'pc_rollout':'Central PC',
          'robust_pc_rollout':'Robust PC', 'exposure_matched':'Exposure matched',
          'exposure_central':'Exposure central', 'static':'Static',
          'greedy':'Greedy', 'oracle':'Oracle'}
GROUPS = {'all':'All', 'nominal':'Nominal', 'single_domain':'Single domain',
          'cascade':'Cascade', 'ood':'OOD compound'}
WEIGHTS = {'reference_1024':'Reference', 'weight_mobility_x2':r'$2\times$ mobility',
           'weight_energy_x2':r'$2\times$ unserved energy',
           'weight_communication_x2':r'$2\times$ communication',
           'weight_power_x2':r'$2\times$ power service',
           'weight_cascade_half':r'$0.5\times$ cascade'}
CONDITIONS = {'primary':'Primary', 'electrical_stress':r'$20\times$ EV load',
              'backup_0':'No backup', 'backup_60':'60 s backup', 'backup_300':'300 s backup',
              'matrix_scale_075':r'$0.75M$', 'matrix_scale_125':r'$1.25M$',
              'matrix_noise_015':'Coefficient noise 0.15', 'crews_12':'12 crews',
              'crews_24':'24 crews', 'zero_comm_persistence':r'$\rho_c=0$',
              'zero_pc_coupling':r'$\tau_{pc}=0$', 'zero_rc_coupling':r'$\tau_{rc}=0$',
              'zero_cr_coupling':r'$\tau_{cr}=0$', 'zero_cp_coupling':r'$\tau_{cp}=0$',
              'zero_all_unsupported':'All constructed couplings zero',
              'expanded_unsupported':'Constructed couplings doubled',
              'nonlinear_ood':'Nonlinear realized transition', **WEIGHTS}


def number(value, digits=4):
    if abs(value) < .5*10**(-digits):
        value = 0.
    return f'{value:.{digits}f}'


def interval(row, digits=4):
    return f'[{number(row.ci95_low,digits)}, {number(row.ci95_high,digits)}]'


def probability(value):
    if value == 0:
        return r'$<10^{-300}$'
    if value < .0001:
        exponent = int(f'{value:.2e}'.split('e')[1])
        mantissa = float(f'{value:.2e}'.split('e')[0])
        return rf'${mantissa:g}\times10^{{{exponent}}}$'
    return number(value,4)


def precise(value):
    if value == 0:
        return '0'
    if abs(value) < .001:
        mantissa, exponent = f'{value:.3e}'.split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    return number(value,6)


def tabular(columns, header, rows):
    return ('\\begin{tabular}{@{}'+columns+'@{}}\n\\toprule\n'+header+r' \\'
            +'\n\\midrule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)
            +'\n\\bottomrule\n\\end{tabular}\n')


def table(caption, label, body, resize=True):
    if resize:
        body = '\\resizebox{\\columnwidth}{!}{%\n'+body+'}\n'
    return ('\\begin{table}[t]\n\\centering\n\\caption{'+caption+'}\n'
            +'\\label{'+label+'}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n'
            +body+'\\end{table}\n')


def build(partial=False):
    inference = RESULTS/'inference'
    manifest = json.loads((inference/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['full_condition_set_complete'] and not partial:
        raise RuntimeError('All declared conditions must finish before a submission table build')
    for condition, expected in manifest['condition_records'].items():
        actual = hashlib.sha256((RESULTS/condition/'rollout_scenarios.csv').read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Statistical input changed: {condition}')
    stats = pd.read_csv(inference/'paired_comparisons.csv')
    risk = pd.read_csv(inference/'cost_risk_summary.csv')
    execution = pd.read_csv(inference/'execution_summary.csv')
    outputs = {}
    skipped = []
    def select(condition, comparator='forecast_matched', policy='pc_rollout', group='all'):
        view = stats[(stats.condition==condition)&(stats.comparator==comparator)
                     &(stats.policy==policy)&(stats.group==group)&(stats.metric=='cost')]
        if len(view) != 1:
            raise ValueError(f'Expected one comparison for {condition} {policy} {comparator} {group}')
        return view.iloc[0]

    rows=[]
    for group,label in GROUPS.items():
        row=select('primary',group=group)
        rows.append([label,str(int(row.n_pairs)),number(row.reduction_percent),
                     number(row.mean_difference),interval(row)])
    body=tabular('lrrrr',r'Group & Pairs & Reduction (\%) & $\Delta$ & $\Delta$ 95\% CI',rows)
    body='\\resizebox{\\columnwidth}{!}{%\n'+body+'}\n\\vspace{3pt}\n'
    means=risk[risk.condition.eq('primary')].set_index('policy')['mean']
    rows=[[label,number(means[policy],2),number(100*(means.forecast_matched-means[policy])/means.forecast_matched)]
          for policy,label in LABELS.items()]
    body+=tabular('lrr',r'Policy & Mean cost & Reduction (\%)',rows)
    outputs['table_robust_boulder.tex']=table(
        'Central PC compared with the same service score without threat propagation, '
        'followed by all policy means. Positive reduction denotes lower cost than forecast matched. '
        r'$\Delta$ is central minus matched cost. Intervals are paired marginal intervals. '
        'All policies share 4,096 scenarios and the oracle alone observes future demand and innovations.',
        'tab:robust-boulder',body,resize=False)

    if all(condition in stats.condition.values for condition in WEIGHTS):
        rows=[]
        for condition,label in WEIGHTS.items():
            row=select(condition)
            rows.append([label,number(row.reduction_percent),number(row.mean_difference)+' '+interval(row)])
        outputs['table_objective_sensitivity.tex']=table(
            'Objective sensitivity of central PC relative to forecast matched. All rows use the '
            'same 1,024 scenario inputs. Candidate routes are selected again under each stated '
            'objective. Reference is the corresponding subset of the primary experiment, not its full 4,096 pairs.',
            'tab:objective-sensitivity',tabular('lrr',r'Weights & Reduction (\%) & Difference (95\% CI)',rows))
    else:
        skipped.append('table_objective_sensitivity.tex')

    rows=[]
    for condition,label in [('primary','Measured load'),('electrical_stress',r'$20\times$ EV load')]:
        row=execution[(execution.condition==condition)&execution.policy.eq('all_policies')].iloc[0]
        rows.append([label,f'{int(row.n_scenarios):,}',f'{int(row.policy_hours):,}',
                     str(int(row.raw_infeasible_hours)),str(int(row.projected_infeasible_hours))])
    body=tabular('lrrrr','Condition & Pairs & AC hours & Raw infeas. & After proj.',rows)
    body='\\resizebox{\\columnwidth}{!}{%\n'+body+'}\n\\vspace{3pt}\n'
    rows=[]
    for condition,label in [('primary','Measured load'),('electrical_stress',r'$20\times$ EV load')]:
        row=execution[(execution.condition==condition)&execution.policy.eq('all_policies')].iloc[0]
        rows.append([label,number(100*row.mean_projection_fraction,2),
                     number(100*row.mean_requested_job_completion_fraction,2)])
    body+=tabular('lrr',r'Condition & Action retained (\%) & Jobs completed (\%)',rows)
    outputs['table_integrated_execution.tex']=table(
        'Electrical projection and repair completion in the integrated experiments. '
        'Primary includes eight policies and stress includes six. Action retention is the mean '
        'hourly fraction. Job completion is the mean scenario fraction of requested jobs completed within the horizon.',
        'tab:integrated-execution',body,resize=False)

    if 'backup_effect_2' in stats.family.values:
        rows=[]
        for condition,label in [('backup_0','0 s'),('backup_60','60 s'),('backup_300','300 s')]:
            row=execution[(execution.condition==condition)&execution.policy.eq('pc_rollout')].iloc[0]
            delta=select(condition,'same_policy_no_backup') if condition!='backup_0' else None
            rows.append([label,number(row.mean_timely_action_fraction,3),
                         number(row.mean_requested_job_completion_fraction,3),
                         number(delta.reduction_percent,2) if delta is not None else '0.00',
                         interval(delta,2) if delta is not None else 'Reference'])
        outputs['table_packet_feedback.tex']=table(
            r'Backup duration for central PC at $2\times$ packet traffic on 1,024 paired scenarios. '
            'Completion is the fraction of requested jobs completed. Cost differences compare '
            'the same policy with no backup.',
            'tab:packet-feedback',tabular('lrrrr',r'Backup & Timely action & Completion & Reduction (\%) & Difference 95\% CI',rows))
    else:
        skipped.append('table_packet_feedback.tex')

    # Supplementary statistical tables retain every comparison rather than
    # selecting only favorable endpoints or threat groups.
    extra=[]
    primary=stats[stats.family.eq('primary_cost_15')]
    rows=[[LABELS[row.comparator],GROUPS[row.group],str(int(row.n_pairs)),
           number(row.mean_difference),interval(row),probability(row.holm_t),probability(row.holm_wilcoxon)]
          for row in primary.itertuples()]
    extra.append(table('Central PC against both service and exposure score controls. '
        'Holm correction includes all fifteen comparisons separately for each test. '
        'Bootstrap intervals are marginal intervals rather than simultaneous intervals.',
        'tab:supp-primary',tabular('llrrrrr',r'Comparator & Group & Pairs & $\Delta$ & 95\% CI & Holm $t$ & Holm Wilcoxon',rows)))
    if manifest['full_condition_set_complete']:
        rows=[]
        for row in stats[stats.family.eq('sensitivity_cost')].itertuples():
            comparison='R minus C' if row.policy=='robust_pc_rollout' else 'C minus F'
            rows.append([CONDITIONS[row.condition],comparison,str(int(row.n_pairs)),number(row.mean_difference),
                         interval(row),probability(row.holm_t),probability(row.holm_wilcoxon)])
        # Longtable is allowed to continue across pages in the supplement.
        body=tabular('llrrrrr',r'Condition & Comparison & Pairs & $\Delta$ & 95\% CI & Holm $t$ & Holm Wilcoxon',rows)
        body=body.replace('\\begin{tabular}{','\\begin{longtable}{').replace('\\end{tabular}','\\end{longtable}')
        body=body.replace('\\toprule\n',r'\caption{All integrated sensitivity comparisons. C denotes central PC, F forecast matched, and R robust PC. The mean difference is first policy minus second policy. Holm correction includes the full sensitivity family. Intervals are marginal.}\label{tab:supp-sensitivity}\\'+'\n\\toprule\n',1)
        repeated=(r'\endfirsthead'+'\n'+r'\multicolumn{7}{l}{\textit{Table S3 continued}}\\'+'\n'
                  +r'\toprule'+'\n'+r'Condition & Comparison & Pairs & $\Delta$ & 95\% CI & Holm $t$ & Holm Wilcoxon \\'+'\n'
                  +r'\midrule\endhead'+'\n'+r'\midrule\multicolumn{7}{r}{\textit{Continued on next page}}\\\endfoot'+'\n'
                  +r'\bottomrule\endlastfoot'+'\n')
        body=body.replace('\\midrule\n','\\midrule\n'+repeated,1)
        body=body.replace('\\bottomrule\n\\end{longtable}','\\end{longtable}')
        extra.append('\\clearpage\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n'+body)
    else:
        skipped.append('supplementary sensitivity table')
    if manifest['full_condition_set_complete']:
        endpoint_labels={
            'unserved_energy':'Unserved energy (kWh)',
            'losses_mwh':'Electrical loss (MWh)',
            'min_voltage_pu':'Minimum voltage (p.u.)',
            'mean_control_action_fraction':'Timely action fraction',
            'mean_smartds_projection_fraction':'Retained action fraction',
            'smartds_curtailed_energy_kwh':'Curtailment (kWh)',
            'crew_completion_fraction':'Completed/requested jobs',
            'crew_mean_completion_h':'Scheduled completion (h)',
            'crew_total_travel_h':'Scheduled travel (h)'}
        rows=[]
        for row in stats[stats.family.eq('physical_secondary_9')].itertuples():
            bounds='['+precise(row.ci95_low)+', '+precise(row.ci95_high)+']'
            means=([f'{row.policy_mean:.9f}',f'{row.comparator_mean:.9f}']
                   if row.metric in ['losses_mwh','min_voltage_pu']
                   else [precise(row.policy_mean),precise(row.comparator_mean)])
            rows.append([endpoint_labels[row.metric],*means,bounds,probability(row.holm_t),
                         probability(row.holm_wilcoxon)])
        extra.append('\\clearpage\n'+table(
            'Physical outcomes of central PC (C) and forecast matched (F) on 4,096 paired scenarios. '
            'Intervals concern C minus F. Holm correction covers all nine endpoints separately for each test. '
            'Completion fraction uses requested jobs, including undelivered requests. Scheduled completion '
            'and travel include dispatched jobs finishing beyond the horizon. Decision computation time is '
            'excluded because concurrent execution is not a controlled timing comparison. '
            'Electrical loss and minimum voltage use independent OpenDSS measurements of identical accepted loads '
            'at convergence tolerance $10^{-10}$, with $10^{-8}$ comparison solves and at most 100 iterations. '
            'These measurements do not alter requests, accepted actions, routes, or costs.',
            'tab:supp-physical',tabular('lrrrrr',r'Endpoint & C mean & F mean & Difference 95\% CI & Holm $t$ & Holm Wilcoxon',rows)))
        rows=[]
        for policy,label in LABELS.items():
            row=risk[risk.condition.eq('primary') & risk.policy.eq(policy)].iloc[0]
            rows.append([label]+[number(row[key],2) for key in ['mean','q95','q99','cvar95','cvar99','maximum']])
        extra.append(table(
            'Primary empirical cost distribution. Quantiles and upper tail means describe the same 4,096 '
            'scenarios per policy. Tail means include a fractional observation at their probability boundary. '
            'These descriptive values are not inferential evidence of tail superiority.',
            'tab:supp-cost-tail',tabular('lrrrrrr',r'Policy & Mean & 95\% quantile & 99\% quantile & Upper 5\% mean & Upper 1\% mean & Maximum',rows)))
    outputs['supplementary_experiment_tables.tex']='\n'.join(extra)
    for name,content in outputs.items():
        (PAPER/name).write_text(content,encoding='utf-8')
    report={'complete_submission_build':manifest['full_condition_set_complete'] and not skipped,
            'generated':list(outputs),'omitted_until_complete':skipped,
            'inference_sha256':hashlib.sha256((inference/'paired_comparisons.csv').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (inference/'table_build_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--complete-only',action='store_true')
    build(partial=parser.parse_args().complete_only)
