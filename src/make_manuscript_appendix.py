"""Print the full declared settings and comparisons inside the manuscript.

Grouped parameter rows reduce repeated prose, not coverage. Values are read
from the same parameter ledger as the optional repository reader report.
Every ledger entry is either printed here or in the main transition table.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re

from make_parameter_supplement import display, tex

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'


def main():
    source = ROOT / 'config/full_parameter_ledger.csv'
    entries = list(csv.DictReader(source.open(encoding='utf-8', newline='')))
    ledger = {(r['group'], r['symbol']): r for r in entries}
    if len(ledger) != len(entries):
        raise ValueError('Repeated parameter identity')
    covered = set()
    rows = [[], [], []]
    def value_cell(group, symbol):
        value = display(ledger[(group, symbol)]['value'])
        if (group, symbol) == ('packet', 'erasure'):
            return r'$\min(0.25,\allowbreak 0.002+\allowbreak 0.08z^c)$'
        if (group, symbol) == ('station_flow', 'capacity'):
            return r'$\max(1.2q_{0.995},\allowbreak 7.2)$'
        if re.fullmatch(r'1e-\d+', value):
            return r'$10^{-'+str(int(value.split('-')[1]))+'}$'
        if re.fullmatch(r'\d+(?:\.\d+)?(?: \d+(?:\.\d+)?)+', value):
            return '(' + r',\allowbreak '.join(value.split()) + ')'
        return tex(value)
    def add(part, group, symbols, name, meaning):
        keys = [(group, symbol) for symbol in symbols.split()]
        if covered.intersection(keys):
            raise ValueError('Parameter covered twice: ' + name)
        values = [value_cell(*k) for k in keys]
        covered.update(keys)
        rows[part].append([name, ', '.join(values), meaning])

    add(0,'forecast','L H',r'History $L$, horizon $H$', 'Observed and predicted hours.')
    add(0,'forecast','hidden epochs batch', 'Hidden units, epochs, batch', 'Hidden width, maximum epochs and batch samples. Checkpoint selected by validation error.')
    add(0,'forecast','learning_rate weight_decay dropout', 'Learning rate, decay, dropout', 'AdamW rate, weight decay and dropout fraction, shared by all chronological fits.')
    add(0,'forecast','epsilon_LN Huber_delta gradient_clip',r'$\epsilon_{\rm LN}$, Huber threshold, gradient clip', 'Variance stabilizer, normalized loss threshold and Euclidean gradient bound.')
    add(0,'forecast','training_runs graph_neighbours', 'Training repetitions, neighbours', 'Independent fits and geographical neighbours. Training uses 2018 to 2021, selection uses 2022. Graph uses station coordinates.')
    add(0,'forecast','exposure_weights exposure_bounds training_mean_floor', 'Exposure weights, bounds, mean floor', 'Arrival and energy weights, clipping interval and positive training mean floor in the corresponding arrival or energy units.')
    add(0,'objective','gamma',r'Discount $\gamma$', 'Undiscounted six hour score.')
    add(0,'objective','lambda_m lambda_e lambda_c lambda_p lambda_z',r'$\lambda_m,\lambda_e,\lambda_c,\lambda_p,\lambda_z$', 'Reciprocal service units defined in the objective. Sensitivities use 3.6, 5.2, 3.4, 2.4 and 6, one weight at a time.')
    add(0,'service','beta_m beta_e beta_r beta_p',r'$\beta_m,\beta_e,\beta_r,\beta_p$', 'Mobility, energy, road threat and power threat conversions to communication load.')
    add(0,'service','alpha_c beta_mc rho_q',r'$\alpha_c,\beta_{mc},\rho_q$', 'Communication derating, communication shortfall to mobility loss and backlog carryover. Carryover is below one.')
    add(0,'service','communication_loss_clip', 'Communication shortfall cap', 'Maximum fractional reduction of charging support.')
    add(0,'allocator','kappa_q kappa_ch kappa_E kappa_com',r'$\kappa_q,\kappa_{\rm ch},\kappa_E,\kappa_{\rm com}$', 'Backlog and threat weights in charging, then energy and threat weights in communication.')
    add(0,'allocator','omega_p omega_c omega_r omega_Gamma',r'$\omega_p,\omega_c,\omega_r,\omega_\Gamma$', 'Power, communication, road and predicted marginal benefit weights in repair priority.')
    add(0,'allocator','eta_service eta_repair epsilon',r'Service reserve, repair reserve, $\epsilon$', 'Uniform reserve fractions and positive score stabilizer.')
    add(0,'budget','B_ch',r'Charging budget $B^{\rm ch}$', 'kWh per hour. Training station mean energy sum multiplied by 1.22. Electrical stress uses factor 24.4 with twenty times demand.')
    add(0,'budget','B_com',r'Communication budget $B^{\rm com}$', 'Service units. Training capacity proxy sum multiplied by 0.72. The proxy uses training arrivals and connected session hours.')
    add(0,'budget','B_res',r'Repair priority budget $B^{\rm res}$', 'Priority units. Maximum of 10 and 0.018 times summed training mean arrivals. All three budgets use 2018 to 2021 only.')
    add(0,'repair','road_effect power_effect communication_effect', 'Road, power, communication repair effect', 'Threat reduction per quantum in continuous priority prediction, not realized repair.')
    add(0,'repair','normalized_priority_cap', 'Predictive repair quantum cap', 'Priority divided by the per zone reference, clipped from zero to this cap.')

    add(1,'crew','vehicles jobs', 'Crews, maximum jobs', 'Crews and work orders. Sensitivity uses 12 and 24 crews. Job count follows initial threat.')
    add(1,'crew','speed circuity', 'Travel speed, circuity', 'km per hour and road to great circle distance ratio at Boulder coordinates.')
    add(1,'crew','service_time', 'Repair duration', 'Hours. EAGLE-I 50 customer half recovery prior, 255 observations among 1,467 events ending before 2022, clipped to 0.25 to 8 hours.')
    add(1,'crew','job_spread_weights job_threshold', 'Spatial job weights, inclusion threshold', 'One and two neighbour contributions to initial threat, then the inclusion threshold. Jobs are fixed before policy selection.')
    add(1,'crew','scaled_duration_bounds', 'Scaled service duration interval', 'Hours after scaling the base prior by 0.5, 1 or 2, then clipping.')
    add(1,'crew','solver', 'Route construction', 'Priority per incremental completion time, fixed ties and no time cutoff. Common candidates for all selectors.')
    add(1,'packet','duration arrival_rate service_rate', 'Queue window, arrival rate, service rate', 'Seconds and packets per second. Activity is connected sessions plus arrivals from twenty training hours. Traffic factors are 0.5, 1, 2 and 4.')
    add(1,'packet','buffer control_fraction deadline', 'Buffer, control share, deadline', 'Packets, fraction and milliseconds.')
    add(1,'packet','max_retries backoff_mean', 'Retries, mean backoff', 'Maximum retransmissions and exponential retry wait in seconds.')
    add(1,'packet','erasure backup', 'Erasure probability, backup', 'Threat dependent erasure and backup in seconds. Sensitivity uses 0 and 300 seconds.')
    packet_comm = value_cell('packet', 'communication_rate_derating')
    packet_power = value_cell('packet', 'power_rate_derating')
    packet_context = value_cell('packet', 'context_ratio')
    add(1,'packet','communication_rate_derating power_rate_derating minimum_service_rate context_ratio', 'Queue derating, rate floor, context ratio',
        r'Base rate factor $1-' + packet_comm + r'z^c$, multiplied by $1-' + packet_power +
        r'z^p$ after backup. Rate floor in packets per second. Constructed $z^c/z^p=' + packet_context + '$.')
    add(1,'packet','power_lookup_scale communication_lookup_scale', 'Power and communication lookup scales', 'Distance scales for nearest training response selection.')
    add(1,'grid','feeder power_factor', 'Feeder, power factor', 'Complete public feeder and lagging EV power factor.')
    add(1,'grid','load_model voltage_band line_limit', 'EV load model, voltage band, line limit', 'Balanced three phase wye increments at 17 loads. Voltage and loading in p.u. Base loads unchanged.')
    add(1,'grid','bisection_iterations execution_convergence_tolerance', 'Bisection steps, execution tolerance', 'Fraction resolution about 0.000061. Original OpenDSS convergence criterion unchanged.')
    add(1,'grid','measurement_convergence_tolerance measurement_max_iterations', 'Measurement tolerance, iteration limit', 'Independent measurements for all central and matched pairs. Second tolerance is 0.00000001. Accepted actions unchanged.')
    add(1,'grid','mappings mapping_seed', 'Mappings, initialization index', 'Station to load assignments. Scenario index modulo twenty selects the shared assignment.')
    add(1,'station_flow','capacity', 'Station capacity', 'kWh per hour. Training energy quantile at probability 0.995. Stress proxy, not nameplate power.')
    add(1,'station_flow','closure_fraction demand_multiplier distance_beta loyalty', 'Closure, demand, distance, loyalty', 'Fraction, multiplier, reciprocal km and cost units. Full factorial at 24 selected 2023 hours.')
    add(2,'scenario','initial_zone_counts initial_severities innovation_scales', 'Affected zones, severity, innovation', 'Values follow nominal, single domain, cascade and compound order within each sequence. Nominal affects road only, single domain draws one of the three domains uniformly, and the remaining classes affect all three.')
    add(2,'scenario','initial_multiplier innovation_multiplier', 'Initial and innovation multipliers', 'Independent uniform draws over the displayed intervals. Nominal innovations affect only the first sampled location. Other classes perturb every sampled location in each affected domain.')
    add(2,'scenario','location_uniform_mix', 'Location sampling offset', 'Sampling probability is proportional to the training mean arrival count plus this fraction of its spatial mean. Locations are sampled without replacement.')
    add(2,'scenario','initial_threat_cap transition_threat_cap', 'Initial and propagated state caps', 'Normalized upper limits. Both initial and propagated states are clipped below by zero. The second value is also the nonlinear saturation scale.')
    add(2,'statistics','n_primary bootstrap holm_family physical_family', 'Primary pairs, resamples, primary tests, physical tests', 'Four equal threat groups, percentile 95 percent intervals and separate Holm adjustments for paired mean and rank tests. Integrated sensitivity has 45 comparisons.')
    add(2,'statistics','scenario_initialization scenario_stride coefficient_noise_index', 'Initial index, scenario stride, noise index', 'A scenario uses the initial index plus its identifier times the stride. The last index fixes controller coefficient noise. The paired policies share the resulting inputs. No index was retuned.')
    transition = {k for k in ledger if k[0] == 'transition'}
    if covered | transition != set(ledger):
        raise ValueError('Unprinted settings: ' + str(set(ledger) - covered - transition))

    output = [r'\appendix[Parameters and additional comparisons]',r'\label{app:parameters}',
        r'Tables~\ref{tab:scenario-parameters}, \ref{tab:complete-parameters}, and \ref{tab:execution-parameters} state the numerical settings, their roles, sources, selection periods, and tested ranges.\AITextRef\ Grouped values follow the listed parameter order. Unless a data source or sensitivity range is stated, values are declared benchmark settings held fixed across policies. They were not retuned on the revised test. Table~\ref{tab:calibrated-uncertainty} separately gives transition estimates and intervals. Long estimates are rounded for display.',
        r'Table~\ref{tab:supp-primary} retains the complete primary comparison family. Table~\ref{tab:supp-sensitivity} gives every integrated sensitivity comparison, Table~\ref{tab:supp-physical} gives all secondary service and physical outcomes, and Table~\ref{tab:supp-cost-tail} gives the empirical cost tails. The numerical record retains all paired mean and rank tests, including unfavorable comparisons.']
    labels = ['tab:complete-parameters','tab:execution-parameters','tab:scenario-parameters']
    for number in [2, 0, 1]:
        label = labels[number]
        caption = ['Prediction, objective, service and allocation settings.', 'Crew, communication, electrical and station settings.', 'Scenario construction and statistical settings.'][number]
        body = '\n'.join(' & '.join(row) + r' \\' for row in rows[number])
        environment = 'table' if number == 2 else 'table*'
        width = r'\columnwidth' if number == 2 else r'\textwidth'
        columns = (r'@{}p{0.22'+width+r'}p{0.22'+width+r'}p{0.49'+width+r'}@{}') if number == 2 else (
            r'@{}p{0.235'+width+r'}p{0.18'+width+r'}p{0.55'+width+r'}@{}')
        stretch = '1.12' if number == 2 else '1.0'
        placement = '[!t]' if number == 2 else '[p]'
        output.append('\n'.join([r'\begin{'+environment+'}'+placement,r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
            r'\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{'+stretch+'}',
            r'\begin{tabular}{'+columns+'}',
            r'\toprule Parameter & Value & Unit, role, source, selection and sensitivity \\ \midrule',body,
            r'\bottomrule\end{tabular}',r'\end{'+environment+'}']))
    # Copy every numerical row from the already verified statistical renderer.
    statistics = (PAPER / 'supplementary_experiment_tables.tex').read_text(encoding='utf-8')
    numerical_rows = lambda value: sorted(
        line.strip() for line in value.splitlines()
        if '&' in line and re.search(r'&\s*-?\d', line) and line.rstrip().endswith(r'\\')
        and not line.lstrip().startswith(('Comparator &', 'Condition &', 'Endpoint &', 'Policy &')))
    source_rows = numerical_rows(statistics)
    statistics = statistics.replace('\\clearpage\n', '')
    statistics = statistics.replace('\\begin{table}[t]', '\\begin{table*}[t]').replace('\\end{table}', '\\end{table*}')
    statistics = statistics.replace('\\resizebox{\\columnwidth}', '\\resizebox{\\textwidth}')
    match = re.search(r'\\begin\{longtable\}\{(@\{\}[^\n]+)\}\n(.*?)\\end\{longtable\}', statistics, re.S)
    if not match:
        raise ValueError('Expected one full sensitivity table')
    columns, body = match.groups()
    cap = re.match(r'\\caption\{(.*?)\}\\label\{([^}]+)\}\\\\\n', body, re.S)
    if not cap:
        raise ValueError('Sensitivity caption structure changed')
    caption, label = cap.groups()
    body = body[cap.end():]
    body = re.sub(r'\\endfirsthead.*?\\endlastfoot\n', '', body, flags=re.S)
    replacement = '\n'.join([r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\footnotesize\setlength{\tabcolsep}{4pt}',r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{'+columns+'}',body,r'\bottomrule\end{tabular}}',r'\end{table*}'])
    statistics = statistics[:match.start()] + replacement + statistics[match.end():]
    # Pack the three shorter comparison tables together before the long
    # sensitivity table. Only float order changes, never numerical rows.
    tables = re.findall(r'\\begin\{table\*\}.*?\\end\{table\*\}', statistics, re.S)
    by_label = {re.search(r'\\label\{([^}]+)\}', table).group(1): table for table in tables}
    order = ['tab:supp-primary', 'tab:supp-physical', 'tab:supp-cost-tail', 'tab:supp-sensitivity']
    if set(by_label) != set(order):
        raise ValueError('Unexpected statistical table coverage')
    statistics = '\n\n'.join(by_label[label] for label in order)
    if numerical_rows(statistics) != source_rows or len(source_rows) != 77:
        raise ValueError('Statistical numerical rows changed during appendix layout')
    output.append(statistics)
    target = PAPER / 'manuscript_appendix.tex'
    target.write_text('\n\n'.join(output) + '\n', encoding='utf-8')
    report = dict(ledger_entries=len(ledger),appendix_entries=len(covered),
        unchanged_statistical_rows=len(source_rows),
        transition_table_entries=len(transition),grouped_rows=[len(r) for r in rows],
        ledger_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        statistical_source_sha256=hashlib.sha256((PAPER/'supplementary_experiment_tables.tex').read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (ROOT/'results/submission_service_20260905/inference/appendix_build_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
