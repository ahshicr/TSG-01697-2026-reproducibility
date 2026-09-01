#!/usr/bin/env python3
"""Generate revision tables and submission-grade figures from source CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False

BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
PURPLE = "#7A4EA3"
GREY = "#767676"
LIGHT_BLUE = "#DDEAF6"
LIGHT_TEAL = "#DFF2F0"
LIGHT_RED = "#F6E1DF"
LIGHT_GREY = "#EEEEEE"

POLICY_ORDER = ["static", "greedy", "forecast_matched", "pc_rollout", "oracle"]
POLICY_LABEL = {
    "static": "Static",
    "greedy": "Greedy",
    "forecast_matched": "Forecast-matched",
    "pc_rollout": "PC rollout",
    "oracle": "Oracle",
}


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, width, height, text, face, edge=GREY, fontsize=5.9):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color=GREY, connectionstyle="arc3", text=None, text_xy=None):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    if text and text_xy:
        ax.text(*text_xy, text, color=color, fontsize=6.5, ha="center", va="center")


def make_system_figure(paper: Path):
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.05, 5.75, "a", fontweight="bold", fontsize=10)
    ax.text(0.35, 5.35, "Observed state at hour $t$", fontweight="bold", fontsize=7.2)
    box(ax, (0.25, 3.55), 2.0, 1.25, "Mobility demand\nCharging backlog\nRoad / power / communication\nthreat intensity", LIGHT_GREY, fontsize=5.6)
    box(ax, (2.75, 4.05), 1.75, 0.9, "Graph-recurrent\ndemand forecast", LIGHT_BLUE, edge=BLUE)
    box(ax, (2.75, 2.85), 1.75, 0.9, "Current threat\nobservation", LIGHT_TEAL, edge=TEAL)
    arrow(ax, (2.25, 4.25), (2.75, 4.45))
    arrow(ax, (2.25, 4.05), (2.75, 3.3))

    ax.text(4.85, 5.35, "Information-matched decisions", fontweight="bold", fontsize=7.2)
    box(ax, (4.75, 3.95), 2.2, 1.0, "Forecast-matched baseline\nCurrent weighted threat\n(no transition rollout)", "#F2F2F2", fontsize=5.5)
    box(ax, (4.75, 2.4), 2.2, 1.15, "PC rollout\nAction-dependent $M$ rollout\nMarginal multi-step benefit", LIGHT_RED, edge=RED, fontsize=5.5)
    arrow(ax, (4.5, 4.45), (4.75, 4.45), color=BLUE)
    arrow(ax, (4.5, 3.3), (4.75, 4.2), color=TEAL)
    arrow(ax, (4.5, 4.35), (4.75, 3.15), color=BLUE)
    arrow(ax, (4.5, 3.3), (4.75, 2.95), color=TEAL)

    box(ax, (7.45, 3.15), 1.8, 1.1, "Concave allocator\nCharging / communication /\nrestoration resources", "#F6EBD8", edge="#B07A28", fontsize=5.3)
    arrow(ax, (6.95, 4.35), (7.45, 3.85))
    arrow(ax, (6.95, 2.95), (7.45, 3.5), color=RED)
    box(ax, (9.75, 3.25), 1.85, 0.9, "Apply first action\nObserve hour $t+1$", LIGHT_TEAL, edge=TEAL, fontsize=5.7)
    arrow(ax, (9.25, 3.7), (9.75, 3.7), color=TEAL)
    arrow(ax, (10.65, 3.25), (1.25, 3.5), color=TEAL, connectionstyle="arc3,rad=-0.28", text="closed-loop update", text_xy=(6.2, 1.55))

    ax.text(0.05, 1.15, "b", fontweight="bold", fontsize=10)
    box(ax, (0.35, 0.35), 2.5, 0.75, "Served hourly charging (kWh)", LIGHT_GREY, fontsize=5.7)
    box(ax, (3.45, 0.35), 2.55, 0.75, "Deterministic zone-to-bus\nbenchmark mapping", LIGHT_BLUE, edge=BLUE, fontsize=5.7)
    box(ax, (6.6, 0.35), 2.15, 0.75, "IEEE 33-bus\nLinDistFlow", LIGHT_TEAL, edge=TEAL, fontsize=5.7)
    box(ax, (9.35, 0.22), 2.3, 1.0, "External outcomes\nVoltage / thermal proxy / losses\nExcluded from policy scores", LIGHT_RED, edge=RED, fontsize=5.4)
    arrow(ax, (2.85, 0.72), (3.45, 0.72))
    arrow(ax, (6.0, 0.72), (6.6, 0.72))
    arrow(ax, (8.75, 0.72), (9.35, 0.72))
    ax.text(11.95, 0.02, "Benchmark validation; not an NYC feeder model", ha="right", va="bottom", fontsize=5.3, color=GREY)
    save_figure(fig, paper / "fig_system_revision")


def relative_ci(row):
    denominator = float(row["comparator_mean"])
    low_delta = float(row["mean_difference_bootstrap_ci95_low"])
    high_delta = float(row["mean_difference_bootstrap_ci95_high"])
    return -100 * high_delta / denominator, -100 * low_delta / denominator


def make_evidence_figure(scenarios: pd.DataFrame, stats_df: pd.DataFrame, sensitivity: pd.DataFrame, paper: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [1.0, 1.28, 1.08]})

    primary = stats_df[(stats_df["primary_family"] == True) & (stats_df["group"] != "all")].copy()  # noqa: E712
    group_order = ["nominal", "single_domain", "cascade", "ood"]
    labels = ["Nominal", "Single-domain", "Cascade", "OOD compound"]
    primary["group"] = pd.Categorical(primary["group"], group_order, ordered=True)
    primary = primary.sort_values("group")
    est = primary["relative_reduction_percent"].to_numpy()
    bounds = np.asarray([relative_ci(row) for _, row in primary.iterrows()])
    y = np.arange(len(labels))[::-1]
    colors = [GREY, TEAL, RED, RED]
    axes[0].axvline(0, color="#AAAAAA", lw=0.8, ls="--")
    for yi, value, (low, high), color in zip(y, est, bounds, colors):
        axes[0].plot([low, high], [yi, yi], color=color, lw=1.4)
        axes[0].plot(value, yi, "o", color=color, ms=4.5)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Cost reduction vs. matched baseline (%)")
    axes[0].set_title("Paired effect (95% bootstrap CI)", fontsize=8)
    axes[0].text(-0.18, 1.05, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=9)

    condition_order = ["central", "matrix_scale_075", "matrix_scale_125", "matrix_noise_015", "integer_crews_12", "integer_crews_24"]
    condition_label = ["Central", "$0.75M$", "$1.25M$", "15% noise", "12 crews", "24 crews"]
    central = stats_df[(stats_df["primary_family"] == True) & stats_df["group"].isin(["all", "cascade", "ood"])].copy()  # noqa: E712
    central["condition"] = "central"
    robust = pd.concat([central, sensitivity], ignore_index=True)
    for group, label, color, marker in [("all", "All", BLUE, "o"), ("cascade", "Cascade", TEAL, "s"), ("ood", "OOD", RED, "^")]:
        subset = robust[robust["group"] == group].set_index("condition")
        values = [float(subset.loc[item, "relative_reduction_percent"]) for item in condition_order]
        axes[1].plot(range(len(values)), values, color=color, marker=marker, ms=4, lw=1.3, label=label)
    axes[1].axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    axes[1].set_xticks(range(len(condition_label)), condition_label, rotation=35, ha="right")
    axes[1].set_ylabel("Cost reduction (%)")
    axes[1].set_title("Misspecification and crew projection", fontsize=8)
    axes[1].legend(ncol=3, fontsize=6.5, loc="upper left")
    axes[1].text(-0.15, 1.05, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=9)

    physical = scenarios.groupby("policy").agg(
        worst_min_voltage=("min_voltage_pu", "min"),
        thermal_branch_hours=("thermal_overloaded_branch_hours", "sum"),
    ).loc[POLICY_ORDER]
    x = np.arange(len(POLICY_ORDER))
    margin_mpu = 1000.0 * (physical["worst_min_voltage"].to_numpy() - 0.9)
    bars = axes[2].bar(x, margin_mpu, color=[GREY, "#AAAAAA", TEAL, RED, PURPLE], width=0.68)
    for bar, hours in zip(bars, physical["thermal_branch_hours"].to_numpy()):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{int(hours)} br-h", ha="center", va="bottom", fontsize=5.2)
    axes[2].set_xticks(x, ["Static", "Greedy", "Matched", "PC", "Oracle"], rotation=32, ha="right")
    axes[2].set_ylabel("Worst voltage margin above 0.9 p.u.\n($10^{-3}$ p.u.)")
    axes[2].set_ylim(0, max(margin_mpu) * 1.32)
    axes[2].set_title("Independent feeder check", fontsize=8)
    axes[2].text(-0.15, 1.05, "c", transform=axes[2].transAxes, fontweight="bold", fontsize=9)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.28, wspace=0.48)
    save_figure(fig, paper / "fig_revision_evidence")


def p_text(value: float) -> str:
    if value < 1e-99:
        return r"$<10^{-99}$"
    return f"{value:.2g}"


def write_tables(scenarios: pd.DataFrame, stats_df: pd.DataFrame, sensitivity: pd.DataFrame, paper: Path):
    grouped = scenarios.groupby("policy").mean(numeric_only=True).loc[POLICY_ORDER]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Closed-loop performance over 4096 paired scenarios. Cost is a dimensionless weighted benchmark objective; other units are shown explicitly.}",
        r"\label{tab:rollout}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Policy & Cost $\\downarrow$ & Delay (trip-equiv.) $\\downarrow$ & Unserved (kWh) $\\downarrow$ & Comm. loss (service units) $\\downarrow$ & Continuity $\\uparrow$ & Latency (ms) $\\downarrow$ \\\\",
        r"\midrule",
    ]
    for policy in POLICY_ORDER:
        row = grouped.loc[policy]
        name = POLICY_LABEL[policy]
        if policy == "pc_rollout":
            name = r"\textbf{PC rollout}"
        lines.append(
            f"{name} & {row['cost']:.1f} & {row['mobility_delay']:.1f} & {row['unserved_energy']:.1f} & {row['comm_loss']:.2f} & {row['service_continuity']:.4f} & {row['latency_ms']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (paper / "table_rollout.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    primary = stats_df[stats_df["primary_family"] == True].copy()  # noqa: E712
    order = ["all", "nominal", "single_domain", "cascade", "ood"]
    primary["group"] = pd.Categorical(primary["group"], order, ordered=True)
    primary = primary.sort_values("group")
    label = {"all": "All", "nominal": "Nominal", "single_domain": "Single domain", "cascade": "Cascade", "ood": "OOD compound"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Prespecified paired comparison of PC rollout with the forecast-matched baseline. Confidence intervals are percentile bootstrap intervals for the paired cost difference (PC minus baseline); $P_{\mathrm{H}}$ is the Holm-adjusted two-sided Wilcoxon value for the five-test primary family.}",
        r"\label{tab:threat-class}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Threat class & $n$ & Matched cost & PC cost & Reduction & Paired difference (95\\% CI) & $P_{\\mathrm{H}}$ \\\\",
        r"\midrule",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"{label[str(row['group'])]} & {int(row['n_pairs'])} & {row['comparator_mean']:.1f} & {row['policy_mean']:.1f} & {row['relative_reduction_percent']:.3f}\\% & {row['mean_difference_pc_minus_comparator']:.1f} [{row['mean_difference_bootstrap_ci95_low']:.1f}, {row['mean_difference_bootstrap_ci95_high']:.1f}] & {p_text(float(row['holm_p_primary_wilcoxon']))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (paper / "table_threat_class.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    physical = scenarios.groupby("policy").agg(
        worst_min_voltage=("min_voltage_pu", "min"),
        voltage_bus_hours=("voltage_violating_bus_hours", "sum"),
        thermal_branch_hours=("thermal_overloaded_branch_hours", "sum"),
        thermal_overload_puh=("thermal_overload_pu_hours", "sum"),
        mean_losses_mwh=("losses_mwh", "mean"),
    ).loc[POLICY_ORDER]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{External IEEE 33-bus LinDistFlow check over 4096 scenarios (six hours each). The branch threshold is a documented 1.25-times-base-flow proxy because case33bw does not provide operational ratings. These outcomes do not enter any policy score.}",
        r"\label{tab:grid-validation}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        "Policy & Worst $V_{\\min}$ & Voltage violations & Branch overloads & Overload (p.u.-h) & Losses (MWh) \\\\",
        r"\midrule",
    ]
    for policy in POLICY_ORDER:
        row = physical.loc[policy]
        lines.append(
            f"{POLICY_LABEL[policy]} & {row['worst_min_voltage']:.4f} & {int(row['voltage_bus_hours'])} & {int(row['thermal_branch_hours'])} & {row['thermal_overload_puh']:.4f} & {row['mean_losses_mwh']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (paper / "table_grid_validation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    condition_label = {
        "matrix_scale_075": r"Policy uses $0.75M$",
        "matrix_scale_125": r"Policy uses $1.25M$",
        "matrix_noise_015": r"15\% coefficient noise",
        "integer_crews_12": "12 integer crews",
        "integer_crews_24": "24 integer crews",
    }
    pivot = sensitivity.pivot(index="condition", columns="group", values="relative_reduction_percent")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Sensitivity of PC-rollout cost reduction relative to the forecast-matched baseline (1024 paired scenarios per condition). The realized environment remains at the central matrix when the policy matrix is perturbed.}",
        r"\label{tab:restore-sensitivity}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        "Condition & All & Cascade & OOD \\\\",
        r"\midrule",
    ]
    for condition in condition_label:
        row = pivot.loc[condition]
        lines.append(f"{condition_label[condition]} & {row['all']:.2f}\\% & {row['cascade']:.2f}\\% & {row['ood']:.2f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paper / "table_restore_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    parameter_table = r"""\begin{table*}[t]
\centering
\caption{Complete grouping of revision-evaluation parameters. Values not identified from the TLC records are declared benchmark assumptions and are challenged in the sensitivity analysis; the exact machine-readable configuration accompanies every run.}
\label{tab:parameters}
\scriptsize
\begin{tabular}{p{0.19\linewidth}p{0.24\linewidth}p{0.48\linewidth}}
\toprule
Group & Values & Meaning / provenance \\
\midrule
Forecast & history $12$ h; horizon $H=6$ h; hidden 64; epochs 80; batch 64; learning rate 0.002; weight decay $10^{-4}$; dropout 0.1 & Prespecified training configuration; chronological train/validation/test split; seed selected only by validation score. \\
Data conversion & EV share 0.18; energy 0.24 kWh/mile & Declared TLC-to-charging proxy assumptions, common to all policies. \\
Service resources & charging $1.22\bar e$; communication $0.72\bar c$; restoration $0.018\sum_i\bar m_i$ & Capacity scales relative to historical mean demand; benchmark assumptions. \\
Propagation $M$ & persistence 0.48; spatial 0.20; $(\tau_{pc},\tau_{rc},\tau_{cr},\tau_{cp})=(0.14,0.04,0.08,0.04)$ & Nonnegative central scenario coefficients; estimated spectral radius 0.774; policy-only scale/noise perturbations reported separately. \\
Restoration effects & road 0.08; power 0.12; communication 0.10 & Normalized one-step threat-intensity relief per per-zone reference resource. \\
State recursion & backlog carryover 0.22; backlog score 1.0 & $\rho_q<1$ ensures bounded carryover under bounded arrivals; the score coefficient is a benchmark assumption. \\
Service scores & energy weight 0.70; threat weights 1.50/1.50; reserve fractions 0.04/0.02 & Communication energy weight; charging/communication threat weights; service/restoration uniform reserves. \\
Restoration score & power 1.60; communication 1.30; road 1.10; rollout benefit 0.80 & Domain and marginal multi-step benefit weights. \\
Communication proxy & mobility 0.13; energy 0.19; road 6.0; power 3.5; direct derating 0.40 & Hourly aggregate service model; not packet-level queue dynamics. \\
Mobility proxy & communication effect 0.45 & Converts aggregate communication shortfall to delayed-trip equivalents. \\
Stage objective & mobility 1.8; unserved energy 2.6; communication 1.7; power-service loss 1.2; cross-domain exposure 12.0 & Dimensionless relative weights, fixed across policies; unmet charging appears once. \\
Feeder check & EV power factor 0.97; thermal proxy 1.25 times base flow; $V\in[0.9,1.1]$ p.u. & External IEEE 33-bus LinDistFlow settings, excluded from policy scores. \\
Monte Carlo & 4096 paired scenarios; seed 2026; four equal threat classes & Prespecified primary evaluation; identical realization for every policy within a pair. \\
\bottomrule
\end{tabular}
\end{table*}
"""
    (paper / "table_parameters.tex").write_text(parameter_table, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/revision"))
    parser.add_argument("--sensitivity", type=Path, default=Path("results/revision_sensitivity/sensitivity_summary.csv"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    scenarios = pd.read_csv(args.results / "rollout_scenarios.csv")
    stats_df = pd.read_csv(args.results / "paired_statistics.csv")
    sensitivity = pd.read_csv(args.sensitivity)
    make_system_figure(args.paper)
    make_evidence_figure(scenarios, stats_df, sensitivity, args.paper)
    write_tables(scenarios, stats_df, sensitivity, args.paper)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
