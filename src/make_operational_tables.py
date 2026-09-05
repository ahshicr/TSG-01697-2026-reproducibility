#!/usr/bin/env python3
"""Generate manuscript tables from the reported numerical results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def write(name: str, value: str) -> None:
    (PAPER / name).write_text(value.strip() + "\n", encoding="utf-8")


def forecast_table(source: Path | None = None) -> None:
    frame = pd.read_csv(source or ROOT / "results/real_ev_strict_20260905/forecast_table.csv")
    rows = []
    for _, row in frame.iterrows():
        label = str(row.method).replace("STGNN rollout predictor", "Graph recurrent forecaster")
        rows.append(
            f"{label} & {row.mae_demand:.3f} & {row.rmse_demand:.3f} & "
            f"{row.mae_energy:.3f} & {row.rmse_energy:.3f} \\\\"
        )
    write(
        "table_real_ev_forecast.tex",
        r"""
\begin{table}[t]
\centering
\caption{Calendar split test performance on measured Boulder charging sessions in 2023}
\label{tab:real-ev-forecast}
\begin{tabular}{lrrrr}
\toprule
Method & \multicolumn{2}{c}{Arrivals} & \multicolumn{2}{c}{Energy (kWh)} \\
 & MAE & RMSE & MAE & RMSE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )


def data_table() -> None:
    write(
        "table_evidence_sources.tex",
        r"""
\begin{table*}[t]
\centering
\caption{Independent evidence sources and their distinct roles}
\label{tab:evidence-sources}
\begin{tabular}{p{0.17\textwidth}p{0.18\textwidth}p{0.20\textwidth}p{0.35\textwidth}}
\toprule
Source & Scale & Observed fields & Role in this study \\
\midrule
City of Boulder EV sessions & 148,136 transactions from 50 stations between 2018 and 2023 & Arrival time, charging duration, energy (kWh), station and address & Measured charging demand with calendar periods for training, selection, and testing \\
EAGLE-I & 249,316,543 national rows with 819,528 retained in Boulder and six adjacent counties between 2014 and 2025 & County, UTC timestamp, customers out & Power state persistence, event and recovery distributions, and uncertainty bounds with missing rows preserved \\
SMART-DS SFO P1U feeder & 80 buses, 219 nodes, 27 loads, 67 lines, 13 transformers & OpenDSS circuit, equipment and peak load operating point & Online projection for all primary policy hours and the integrated electrical stress test \\
Packet and route models & 12,000 packet stress rows and 4,096 paired main scenarios with up to 36 jobs and four crews & Queue events, retries, deadlines, backup, integer crews, travel, service and completion times & Requested controls and threat updates after a recorded repair completion event \\
\bottomrule
\end{tabular}
\end{table*}
""",
    )


def calibration_table(source: Path | None = None) -> None:
    frame = pd.read_csv(source or ROOT / "results/calibration_strict_20260905/transition_parameter_uncertainty.csv")
    names = {
        "threat_persistence": r"$\rho_r$",
        "spatial_spread": r"$\sigma$",
        "power_persistence": r"$\rho_p$",
        "pr_coupling": r"$\tau_{pr}$",
        "pc_coupling": r"$\tau_{pc}$",
        "comm_persistence": r"$\rho_c$",
        "rc_coupling": r"$\tau_{rc}$",
        "cr_coupling": r"$\tau_{cr}$",
        "cp_coupling": r"$\tau_{cp}$",
    }
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"{names[row.parameter]} & {row.central:.3f} & [{row.lower:.3f}, {row.upper:.3f}] & "
            f"{str(row.evidence_class).replace('-', ' ')} \\\\"
        )
    write(
        "table_calibrated_uncertainty.tex",
        r"""
\begin{table}[t]
\centering
\caption{Propagation coefficients and prespecified uncertainty set}
\label{tab:calibrated-uncertainty}
\begin{tabular}{lrrl}
\toprule
Parameter & Central & Interval & Evidence class \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )


def primary_policy_table() -> None:
    stats = pd.read_csv(ROOT / "results/operational/boulder_robust_rollout/paired_statistics_central.csv")
    scenarios = pd.read_csv(ROOT / "results/operational/boulder_robust_rollout/rollout_scenarios.csv")
    view = stats[(stats.comparator == "forecast_matched") & (stats.metric == "cost")].copy()
    order = ["all", "nominal", "single_domain", "cascade", "ood"]
    labels = {
        "all": "All",
        "nominal": "Nominal",
        "single_domain": "Single domain",
        "cascade": "Cascade",
        "ood": "OOD compound",
    }
    rows = []
    for group in order:
        row = view[view.group == group].iloc[0]
        rows.append(
            f"{labels[group]} & {int(row.n_pairs)} & {row.policy_mean:.2f}/{row.comparator_mean:.2f} & "
            f"{row.relative_reduction_percent:.4f} & "
            f"[{row.mean_difference_bootstrap_ci95_low:.4f}, {row.mean_difference_bootstrap_ci95_high:.4f}] \\\\"
        )
    policy_labels = {
        "static": "Static",
        "greedy": "Greedy",
        "forecast_matched": "Forecast matched",
        "pc_rollout": "Central PC",
        "robust_pc_rollout": "Robust PC",
        "oracle": "Oracle",
    }
    mean_cost = scenarios.groupby("policy")["cost"].mean()
    matched = float(mean_cost["forecast_matched"])
    policy_rows = []
    for policy in ("static", "greedy", "forecast_matched", "pc_rollout", "robust_pc_rollout", "oracle"):
        value = float(mean_cost[policy])
        reduction = 100.0 * (matched - value) / matched
        policy_rows.append(f"{policy_labels[policy]} & {value:.2f} & {reduction:.3f} \\\\ ")
    write(
        "table_robust_boulder.tex",
        r"""
\begin{table}[t]
\centering
\caption{Primary transition comparison and policy cost summary under sampled uncertainty. Positive reduction denotes lower cost than forecast matched. C/M gives the central PC and matched mean costs. The oracle alone receives future demand and innovations.}
\label{tab:robust-boulder}
\scriptsize
\setlength{\tabcolsep}{1.1pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lrrrr@{}}
\toprule
Threat group & Pairs & C/M cost & Reduction (\%) & $\Delta$ 95\% CI \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}

\vspace{3pt}
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}lrr@{}}
\toprule
\multicolumn{3}{l}{\textit{Policy mean over the same 4,096 scenarios}} \\
Policy & Mean cost & Reduction (\%) \\
\midrule
""" + "\n".join(policy_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )


def objective_sensitivity_table() -> None:
    frame = pd.read_csv(
        ROOT / "results/operational/boulder_robust_rollout/objective_weight_sensitivity_central.csv"
    )
    frame = frame[frame.comparator == "forecast_matched"].copy()
    labels = {
        "reference": "Reference weights",
        "mobility_x2": r"$2\times$ mobility weight",
        "energy_x2": r"$2\times$ unserved energy weight",
        "communication_x2": r"$2\times$ communication weight",
        "power_service_x2": r"$2\times$ power service weight",
        "cascade_half": r"$0.5\times$ cascade weight",
    }
    rows = []
    for scheme in labels:
        row = frame[frame.scheme == scheme].iloc[0]
        rows.append(
            f"{labels[scheme]} & {row.relative_reduction_percent:.4f} & "
            f"{row.mean_difference:.4f} [{row.ci95_low:.4f}, {row.ci95_high:.4f}] \\\\"
        )
    write(
        "table_objective_sensitivity.tex",
        r"""
\begin{table}[t]
\centering
\caption{Objective sensitivity of central PC relative to forecast matched on the same 4,096 paired trajectories. Policies retain their original actions under alternative weights.}
\label{tab:objective-sensitivity}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrr}
\toprule
Evaluation weights & Reduction (\%) & Paired difference (95\% CI) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}
""",
    )


def smartds_table() -> None:
    frame = pd.read_csv(ROOT / "results/standalone_smartds_strict_20260905/stress/smartds_ev_summary.csv")
    pivot = frame.pivot(index="penetration_multiplier", columns="metric", values="mapping_level_mean")
    rows = []
    for penetration, row in pivot.iterrows():
        rows.append(
            f"{penetration:.0f}$\\times$ & {100*row.unconstrained_feasible:.1f} & "
            f"{100*row.feasible_fraction:.1f} & {row.v_min:.3f} & {row.line_loading:.3f} \\\\"
        )
    write(
        "table_smartds_action_loop.tex",
        r"""
\begin{table}[t]
\centering
\caption{SMART-DS AC action projection (20 mappings, 24 hours each)}
\label{tab:smartds-action-loop}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
EV load & Raw feasible (\%) & Action retained (\%) & Raw $V_{\min}$ & Raw line p.u. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}
""",
    )


def integrated_execution_table() -> None:
    frame = pd.read_csv(ROOT / "results/operational/integrated_execution_summary.csv")
    labels = {
        "measured_load_primary": r"Measured load",
        "20x_ev_execution_stress": r"$20\times$ EV stress",
    }
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"{labels[row.condition]} & {int(row.paired_scenarios):,} & "
            f"{int(row.online_smartds_policy_hours):,} & {int(row.raw_infeasible_policy_hours):,} & "
            f"{int(row.projected_infeasible_policy_hours):,} & {100*row.mean_projection_fraction:.2f} & "
            f"{100*row.mean_crew_completion_fraction:.2f} \\\\"
        )
    write(
        "table_integrated_execution.tex",
        r"""
\begin{table}[t]
\centering
\caption{SMART-DS projection and routed crew events inside the closed loop evaluation}
\label{tab:integrated-execution}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Condition & Pairs & AC hours & Raw infeas. & After proj. \\
\midrule
""" + "\n".join(row.rsplit(" & ", 2)[0] + r" \\" for row in rows) + r"""
\bottomrule
\end{tabular}
\vspace{2pt}

\begin{tabular}{lrr}
\toprule
Condition & Action retained (\%) & Crew completed (\%) \\
\midrule
""" + "\n".join(
            f"{labels[row.condition]} & "
            f"{100*row.mean_projection_fraction:.2f} & {100*row.mean_crew_completion_fraction:.2f} \\\\"
            for _, row in frame.iterrows()
        ) + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )


def packet_table() -> None:
    frame = pd.read_csv(ROOT / "results/operational/packet_feedback_statistics_central.csv")
    rows = []
    labels = {"backup_60s": "60 s", "backup_300s": "300 s"}
    for condition in ("backup_60s", "backup_300s"):
        row = frame[
            (frame.row_type == "paired_comparison")
            & (frame.condition == condition)
            & (frame.group == "all")
        ].iloc[0]
        rows.append(
            f"{labels[condition]} & {row.mean_control_action_fraction:.3f} & "
            f"{row.executed_to_requested_restoration:.3f} & "
            f"{row.relative_cost_reduction_percent:.2f} & "
            f"[{row.cost_bootstrap_ci95_low:.1f}, {row.cost_bootstrap_ci95_high:.1f}] \\\\"
        )
    write(
        "table_packet_feedback.tex",
        r"""
\begin{table}[t]
\centering
\caption{Backup power effect with packet feedback at $2\times$ traffic with $n=1024$ paired scenarios}
\label{tab:packet-feedback}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Backup & Timely action & Restoration enacted & Cost reduction (\%) & Difference 95\% CI \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}
""",
    )


def station_choice_table() -> None:
    frame = pd.read_csv(ROOT / "results/operational/station_choice/station_choice_summary.csv")
    frame = frame[(frame.demand_multiplier == 1.0) & (frame.distance_beta_per_km == 1.0)]
    rows = []
    for closure in (0.1, 0.2, 0.3):
        part = frame[frame.closure_fraction == closure]
        served = {
            method: float(
                part[(part.method == method) & (part.metric == "served_fraction")].iloc[0]["mean"]
            )
            for method in ("no_choice", "nearest", "logit", "coordinated")
        }
        distance = float(
            part[
                (part.method == "coordinated")
                & (part.metric == "energy_weighted_extra_distance_km")
            ].iloc[0]["mean"]
        )
        rows.append(
            f"{100*closure:.0f} & {100*served['no_choice']:.1f} & {100*served['nearest']:.1f} & "
            f"{100*served['logit']:.1f} & {100*served['coordinated']:.1f} & {distance:.2f} \\\\"
        )
    write(
        "table_station_choice.tex",
        r"""
\begin{table}[t]
\centering
\caption{Station choice under measured demand for 24 peak hours}
\label{tab:station-choice}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{rrrrrr}
\toprule
Closure (\%) & No choice & Nearest & Logit & Coordinated & Extra km \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}
""",
    )


def crew_table() -> None:
    stats_path = ROOT / "results/operational/crew_routing_128/crew_routing_paired_statistics.csv"
    if not stats_path.exists():
        stats_path = ROOT / "results/operational/crew_routing/crew_routing_paired_statistics.csv"
    frame = pd.read_csv(stats_path)
    frame = frame[frame.comparison == "route_aware_pc_minus_forecast_matched"]
    rows = []
    for _, row in frame.sort_values(["crews", "service_scale"]).iterrows():
        rows.append(
            f"{int(row.crews)} & {row.service_scale:.1f} & {row.relative_reduction_percent:.2f} & "
            f"[{row.bootstrap_ci95_low:.2f}, {row.bootstrap_ci95_high:.2f}] & "
            f"{float(row.holm_wilcoxon_p):.3g} \\\\"
        )
    write(
        "table_route_crew.tex",
        r"""
\begin{table}[t]
\centering
\caption{Route evaluated PC compared with matched routing}
\label{tab:route-crew}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{rrrrr}
\toprule
Crews & Service scale & Risk reduction (\%) & Difference 95\% CI & Holm $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}
""",
    )


def main() -> int:
    from make_frozen_service_tables import build
    # Refuse a final build while any declared frozen experiment is unfinished.
    # The old exposure-study renderers above remain historical helpers only.
    build()
    data_table()
    forecast_table()
    calibration_table()
    smartds_table()
    station_choice_table()
    crew_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
