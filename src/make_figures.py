#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def box(ax, xy, text, fc, w=2.4, h=0.72):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.1,
        edgecolor="#28323c",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, a, b):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="#28323c",
            connectionstyle="arc3,rad=0.0",
        )
    )


def system_figure(out: Path):
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    box(ax, (0.45, 3.8), "Observed mobility\nand charging states", "#d8eef4")
    box(ax, (0.45, 2.25), "Mobility, power and\ncommunication threats", "#f7e2cf")
    box(ax, (3.65, 3.8), "Spatiotemporal\ngraph predictor", "#e7ebfb")
    box(ax, (3.65, 2.25), "Monotone threat\npropagation", "#f4d8dd")
    box(ax, (6.85, 3.05), "Smart grid\ndigital twin rollout", "#e3f0d6")
    box(ax, (6.85, 1.35), "Risk calibrated\nrestoration action", "#efe0f4")
    arrow(ax, (2.85, 4.16), (3.62, 4.16))
    arrow(ax, (2.85, 2.61), (3.62, 2.61))
    arrow(ax, (6.05, 4.16), (6.82, 3.55))
    arrow(ax, (6.05, 2.61), (6.82, 3.23))
    arrow(ax, (8.05, 3.02), (8.05, 2.1))
    arrow(ax, (6.85, 1.72), (1.65, 1.72))
    ax.text(4.25, 1.46, "closed loop update", fontsize=8, color="#28323c")
    ax.text(0.35, 0.45, "Service loss combines charging backlog, feeder margin violation, communication loss and mobility delay.", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def bubble_figure(rows, out: Path):
    policies = ["static", "greedy", "plain_rollout", "pc_rollout", "oracle"]
    labels = {
        "static": "Static",
        "greedy": "Greedy",
        "plain_rollout": "Neural rollout",
        "pc_rollout": "PC rollout",
        "oracle": "Oracle",
    }
    colors = {
        "static": "#7b8da0",
        "greedy": "#d17a22",
        "plain_rollout": "#4c78a8",
        "pc_rollout": "#2f9d55",
        "oracle": "#7a4ea3",
    }
    grouped = {p: [] for p in policies}
    for row in rows:
        grouped[row["policy"]].append(row)
    summary = []
    for policy in policies:
        vals = grouped[policy]
        summary.append(
            {
                "policy": policy,
                "cost": sum(float(v["cost_mean"]) for v in vals) / len(vals),
                "continuity": sum(float(v["service_continuity_mean"]) for v in vals) / len(vals),
            }
        )
    order = sorted(summary, key=lambda item: item["cost"], reverse=True)
    names = [labels[item["policy"]] for item in order]
    y_pos = list(range(len(order)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), gridspec_kw={"width_ratios": [1.35, 1.0]})
    cost_ax, continuity_ax = axes
    cost_values = [item["cost"] for item in order]
    cost_colors = [colors[item["policy"]] for item in order]
    cost_ax.barh(y_pos, cost_values, color=cost_colors, alpha=0.86)
    cost_ax.set_yticks(y_pos)
    cost_ax.set_yticklabels(names, fontsize=8)
    cost_ax.invert_yaxis()
    cost_ax.set_xlabel("Mean rollout cost")
    cost_ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.8)
    cost_ax.spines[["top", "right"]].set_visible(False)
    for y, value in zip(y_pos, cost_values):
        cost_ax.text(value + 55, y, f"{value:.0f}", va="center", fontsize=7.5)

    interruption = [(1.0 - item["continuity"]) * 1000.0 for item in order]
    continuity_ax.barh(y_pos, interruption, color=cost_colors, alpha=0.86)
    continuity_ax.set_yticks(y_pos)
    continuity_ax.set_yticklabels([])
    continuity_ax.invert_yaxis()
    continuity_ax.set_xlabel("Service interruption per 1000")
    continuity_ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.8)
    continuity_ax.spines[["top", "right"]].set_visible(False)
    for y, value in zip(y_pos, interruption):
        continuity_ax.text(value + 0.8, y, f"{value:.1f}", va="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/final"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    args.paper.mkdir(parents=True, exist_ok=True)
    system_figure(args.paper / "fig_system.pdf")
    bubble_figure(read_rows(args.results / "rollout_summary.csv"), args.paper / "fig_policy_bubbles.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
