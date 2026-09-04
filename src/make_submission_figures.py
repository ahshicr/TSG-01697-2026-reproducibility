#!/usr/bin/env python3
"""Submission figures for the executable PC revision."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

BLUE = "#0F4D92"
BLUE2 = "#3775BA"
BLUE_SOFT = "#DCEAF7"
TEAL = "#42949E"
TEAL_SOFT = "#DDEFF0"
VIOLET = "#7C5AA6"
VIOLET_SOFT = "#EEE7F5"
GREEN = "#2E8B57"
GREEN_SOFT = "#DDF3DE"
ORANGE = "#D97720"
ORANGE_SOFT = "#FBE7D2"
RED = "#B64342"
GREY = "#5E6470"
GREY_SOFT = "#F1F2F4"


def export(fig: plt.Figure, stem: str) -> dict[str, object]:
    outputs = {}
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        path = PAPER / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = {"path": path.name, "bytes": path.stat().st_size}
    png = PAPER / f"{stem}.png"
    with Image.open(png) as image:
        outputs["png"]["pixels"] = list(image.size)
    svg_text = (PAPER / f"{stem}.svg").read_text(encoding="utf-8")
    outputs["svg"]["editable_text_nodes"] = svg_text.count("<text")
    return outputs


def box(ax, xy, width, height, text, face, edge, fontsize=7, weight="normal", subtext=None):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * (0.59 if subtext else 0.5),
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#20242A",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.1,
    )
    if subtext:
        ax.text(
            xy[0] + width / 2,
            xy[1] + height * 0.24,
            subtext,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=GREY,
            fontsize=max(fontsize - 1.2, 4.8),
        )
    return patch


def arrow(ax, start, end, color=GREY, connection="arc3", lw=1.1, mutation=8):
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        connectionstyle=connection,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=0,
    )
    ax.add_patch(patch)


def poly_arrow(ax, points, color=GREY, lw=1.1, mutation=8):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], transform=ax.transAxes, color=color, lw=lw, zorder=0)
    arrow(ax, points[-2], points[-1], color=color, lw=lw, mutation=mutation)


def make_system_figure() -> tuple[plt.Figure, dict[str, object]]:
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.set_axis_off()
    ax.text(0.09, 0.92, "Independent evidence", transform=ax.transAxes, ha="center", fontweight="bold", color=GREY)
    box(ax, (0.015, 0.66), 0.16, 0.16, "Boulder EV sessions", BLUE_SOFT, BLUE, weight="bold", subtext="148,136 transactions")
    box(ax, (0.015, 0.43), 0.16, 0.16, "EAGLE-I outages", VIOLET_SOFT, VIOLET, weight="bold", subtext="county event traces")
    box(ax, (0.015, 0.20), 0.16, 0.16, "SMART-DS feeder", GREEN_SOFT, GREEN, weight="bold", subtext="independent OpenDSS")

    box(ax, (0.235, 0.63), 0.17, 0.19, "Demand forecast\n+ observed state", BLUE_SOFT, BLUE, weight="bold")
    box(ax, (0.235, 0.37), 0.17, 0.19, "Calibrated\ntransition model", VIOLET_SOFT, VIOLET, weight="bold")

    box(ax, (0.455, 0.47), 0.17, 0.25, "Central PC\nrollout", ORANGE_SOFT, ORANGE, fontsize=8, weight="bold", subtext="finite horizon score")
    ax.text(0.54, 0.40, "requested actions", transform=ax.transAxes, ha="center", color=GREY, fontsize=6)

    execution = [
        ((0.68, 0.72), "AC action\nprojection", GREEN_SOFT, GREEN),
        ((0.68, 0.53), "Packet deadline\nenactment", TEAL_SOFT, TEAL),
        ((0.68, 0.34), "Station\nflow", BLUE_SOFT, BLUE2),
        ((0.68, 0.15), "Integer crew\nroutes", VIOLET_SOFT, VIOLET),
    ]
    for (x, y), label, face, edge in execution:
        box(ax, (x, y), 0.15, 0.12, label, face, edge, fontsize=6.4, weight="bold")
        arrow(ax, (0.625, 0.59), (x, y + 0.06), color=edge)
    ax.text(0.755, 0.92, "Execution conditions", transform=ax.transAxes, ha="center", fontweight="bold", color=GREY)

    box(ax, (0.875, 0.43), 0.11, 0.27, "Applied action\n+ next observed\nstate", GREY_SOFT, GREY, fontsize=7, weight="bold")
    for (_, y), _, _, edge in execution:
        arrow(ax, (0.83, y + 0.06), (0.875, 0.565), color=edge, lw=0.9, mutation=7)

    arrow(ax, (0.175, 0.74), (0.235, 0.72), color=BLUE)
    arrow(ax, (0.175, 0.51), (0.235, 0.47), color=VIOLET)
    poly_arrow(ax, [(0.175, 0.28), (0.19, 0.12), (0.64, 0.12), (0.70, 0.76)], color=GREEN)
    arrow(ax, (0.405, 0.72), (0.455, 0.63), color=BLUE)
    arrow(ax, (0.405, 0.47), (0.455, 0.56), color=VIOLET)
    arrow(ax, (0.93, 0.43), (0.32, 0.63), color=RED, connection="arc3,rad=-0.34", lw=1.2)
    ax.text(0.65, 0.045, "hourly receding horizon feedback", transform=ax.transAxes, ha="center", color=RED, fontsize=6.3)
    ax.text(0.015, 0.04, "Evidence sources enter through distinct model variables.", transform=ax.transAxes, color=GREY, fontsize=5.7)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    return fig, {
        "core_conclusion": "Requested PC actions affect the next state only after four execution gates.",
        "archetype": "compact process schematic",
    }


def make_evidence_figure() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    central = pd.read_csv(ROOT / "results/operational/boulder_robust_rollout/paired_statistics_central.csv")
    central = central[(central.comparator == "forecast_matched") & (central.metric == "cost")].copy()
    central["reduction"] = central.relative_reduction_percent
    central["reduction_low"] = -100 * central.mean_difference_bootstrap_ci95_high / central.comparator_mean
    central["reduction_high"] = -100 * central.mean_difference_bootstrap_ci95_low / central.comparator_mean
    central["panel"] = "a"

    smart = pd.read_csv(ROOT / "results/operational/smartds_ev/smartds_ev_summary.csv")
    smart = smart.pivot(index="penetration_multiplier", columns="metric", values="mapping_level_mean").reset_index()
    smart["panel"] = "b"

    packet_raw = {}
    for label, directory in (
        ("0 s", "boulder_packet_feedback_no_backup"),
        ("60 s", "boulder_packet_feedback_backup_60s"),
        ("300 s", "boulder_packet_feedback_backup_300s"),
    ):
        frame = pd.read_csv(ROOT / f"results/operational/{directory}/rollout_scenarios.csv")
        frame = frame[frame.policy == "pc_rollout"]
        packet_raw[label] = float(frame.mean_control_action_fraction.mean())
    packet_stats = pd.read_csv(ROOT / "results/operational/packet_feedback_statistics_central.csv")
    packet_cmp = packet_stats[(packet_stats.row_type == "paired_comparison") & (packet_stats.group == "all")]
    packet_values = pd.DataFrame(
        {
            "backup": ["0 s", "60 s", "300 s"],
            "cost_reduction": [
                0.0,
                float(packet_cmp[packet_cmp.condition == "backup_60s"].relative_cost_reduction_percent.iloc[0]),
                float(packet_cmp[packet_cmp.condition == "backup_300s"].relative_cost_reduction_percent.iloc[0]),
            ],
            "action_fraction": [packet_raw["0 s"], packet_raw["60 s"], packet_raw["300 s"]],
        }
    )
    packet_values["panel"] = "c"

    crew = pd.read_csv(ROOT / "results/operational/crew_routing_128/crew_routing_paired_statistics.csv")
    crew = crew[crew.comparison == "route_aware_pc_minus_forecast_matched"].copy()
    crew["panel"] = "d"

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))
    ax = axes[0, 0]
    order = ["all", "nominal", "single_domain", "cascade", "ood"]
    labels = ["All", "Nominal", "Single domain", "Cascade", "OOD compound"]
    view = central.set_index("group").loc[order]
    y = np.arange(len(order))[::-1]
    x = view.reduction.to_numpy()
    xerr = np.vstack([x - view.reduction_low.to_numpy(), view.reduction_high.to_numpy() - x])
    ax.barh(y, x, xerr=xerr, color=BLUE_SOFT, edgecolor=BLUE, ecolor=BLUE, capsize=2.5, linewidth=0.9)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Central PC cost reduction (%)")
    ax.set_xlim(-0.04, 1.25)
    ax.text(0.98, 0.97, "paired bootstrap 95% CI", transform=ax.transAxes, ha="right", va="top", color=GREY, fontsize=5.8)
    ax.set_title("Effect of transition evaluation", loc="left", fontweight="bold", fontsize=8)

    ax = axes[0, 1]
    multipliers = smart.penetration_multiplier.to_numpy()
    x = np.arange(len(multipliers))
    width = 0.36
    ax.bar(x - width / 2, 100 * smart.unconstrained_feasible, width, color=GREY_SOFT, edgecolor=GREY, linewidth=0.8, label="Raw feasible")
    ax.bar(x + width / 2, 100 * smart.feasible_fraction, width, color=GREEN_SOFT, edgecolor=GREEN, linewidth=0.8, label="Action retained")
    ax.set_xticks(x, [f"{int(v)}×" for v in multipliers])
    ax.set_ylim(15, 112)
    ax.set_ylabel("Scenario/action fraction (%)")
    ax.set_xlabel("Measured EV load multiplier")
    ax.legend(loc="upper center", ncol=2, fontsize=6)
    ax.set_title("SMART-DS AC action loop", loc="left", fontweight="bold", fontsize=8)

    ax = axes[1, 0]
    bars = ax.bar(packet_values.backup, packet_values.cost_reduction, color=[GREY_SOFT, TEAL, BLUE], edgecolor=[GREY, TEAL, BLUE], linewidth=0.8)
    ax.set_ylabel("Cost reduction vs no backup (%)")
    ax.set_xlabel("Backup duration")
    ax.set_ylim(0, 31)
    for bar, reduction, fraction in zip(bars, packet_values.cost_reduction, packet_values.action_fraction):
        ax.text(bar.get_x() + bar.get_width() / 2, reduction + 0.9, f"timely {fraction:.3f}", ha="center", va="bottom", fontsize=5.8, color=GREY)
    ax.set_title("Packet feedback at 2× traffic", loc="left", fontweight="bold", fontsize=8)

    ax = axes[1, 1]
    route = crew.pivot(index="crews", columns="service_scale", values="relative_reduction_percent").loc[[4, 12, 24], [0.5, 1.0, 2.0]]
    group_x = np.arange(route.shape[0])
    route_width = 0.24
    route_colors = [BLUE_SOFT, TEAL_SOFT, VIOLET_SOFT]
    route_edges = [BLUE, TEAL, VIOLET]
    for j, scale in enumerate(route.columns):
        ax.bar(
            group_x + (j - 1) * route_width,
            route[scale].to_numpy(),
            route_width,
            color=route_colors[j],
            edgecolor=route_edges[j],
            linewidth=0.8,
            label=f"Scale {scale:g}",
        )
    ax.set_xticks(group_x, ["4", "12", "24"])
    ax.set_xlabel("Integer crews")
    ax.set_ylabel("Integrated risk reduction (%)")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=5.6, columnspacing=0.8, handlelength=1.2)
    ax.set_title("Executable route evaluation", loc="left", fontweight="bold", fontsize=8, pad=12)

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.94, bottom=0.10, hspace=0.48, wspace=0.38)

    source = pd.concat(
        [
            central[["panel", "group", "n_pairs", "reduction", "reduction_low", "reduction_high"]],
            smart[["panel", "penetration_multiplier", "unconstrained_feasible", "feasible_fraction"]],
            packet_values,
            crew[["panel", "crews", "service_scale", "n_pairs", "relative_reduction_percent", "bootstrap_ci95_low", "bootstrap_ci95_high", "holm_wilcoxon_p"]],
        ],
        ignore_index=True,
        sort=False,
    )
    return fig, source, {
        "core_conclusion": "Central transition evaluation improves the matched baseline, while execution constraints determine larger operational effects.",
        "archetype": "quantitative grid",
    }


def main() -> int:
    system_path = PAPER / "fig_executable_system.png"
    if not system_path.exists():
        raise FileNotFoundError(
            "The GPT Image system schematic is missing from paper/fig_executable_system.png"
        )
    with Image.open(system_path) as system_image:
        system_outputs = {
            "png": {
                "path": system_path.name,
                "bytes": system_path.stat().st_size,
                "pixels": list(system_image.size),
            }
        }
    system_contract = {
        "core_conclusion": "Requested PC actions affect the next state only after the execution conditions are satisfied.",
        "archetype": "continuous infrastructure schematic",
        "generation": "GPT Image 2 with manuscript-specific labels and subsequent visual inspection",
    }

    evidence, source, evidence_contract = make_evidence_figure()
    evidence_outputs = export(evidence, "fig_operational_evidence")
    plt.close(evidence)
    source_path = PAPER / "fig_operational_evidence_source_data.csv"
    source.to_csv(source_path, index=False)

    qa = {
        "backend": "GPT Image 2 for the system schematic and Python/matplotlib for quantitative panels",
        "final_width_mm": 182.88,
        "editable_text_required": True,
        "fig_executable_system": {"contract": system_contract, "outputs": system_outputs},
        "fig_operational_evidence": {
            "contract": evidence_contract,
            "statistics": "paired bootstrap 95% intervals; Holm-corrected tests in manuscript tables; no significance stars",
            "source_data": source_path.name,
            "outputs": evidence_outputs,
        },
    }
    (PAPER / "figure_qa_manifest.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
