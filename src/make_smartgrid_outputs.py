#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import math

import matplotlib.pyplot as plt


POLICY_NAMES = {
    "plain_rollout": "Neural rollout",
    "pc_rollout": "PC rollout",
    "greedy": "Greedy",
    "static": "Static",
    "oracle": "Oracle",
}


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits=1):
    return f"{float(value):.{digits}f}"


def avg_policy(rows, policy):
    vals = [row for row in rows if row["policy"] == policy]
    metrics = [
        "cost_mean",
        "mobility_delay_mean",
        "unserved_energy_mean",
        "comm_loss_mean",
        "voltage_violation_mean",
        "service_continuity_mean",
    ]
    return {metric: statistics.fmean(float(row[metric]) for row in vals) for metric in metrics}


def collect(results: Path):
    manifest = read_csv(results / "sweep_manifest.csv")
    rows = []
    for case in manifest:
        summary = read_csv(results / case["case_id"] / "rollout_summary.csv")
        pc = avg_policy(summary, "pc_rollout")
        neural = avg_policy(summary, "plain_rollout")
        greedy = avg_policy(summary, "greedy")
        oracle = avg_policy(summary, "oracle")
        out = dict(case)
        out.update(
            {
                "pc_cost": pc["cost_mean"],
                "neural_cost": neural["cost_mean"],
                "greedy_cost": greedy["cost_mean"],
                "oracle_cost": oracle["cost_mean"],
                "pc_gain_neural": 100.0 * (neural["cost_mean"] - pc["cost_mean"]) / neural["cost_mean"],
                "pc_gain_greedy": 100.0 * (greedy["cost_mean"] - pc["cost_mean"]) / greedy["cost_mean"],
                "oracle_gap": 100.0 * (pc["cost_mean"] - oracle["cost_mean"]) / oracle["cost_mean"],
                "pc_voltage": pc["voltage_violation_mean"],
                "neural_voltage": neural["voltage_violation_mean"],
                "pc_unserved": pc["unserved_energy_mean"],
                "neural_unserved": neural["unserved_energy_mean"],
                "pc_comm_loss": pc["comm_loss_mean"],
                "neural_comm_loss": neural["comm_loss_mean"],
                "pc_continuity": pc["service_continuity_mean"],
                "neural_continuity": neural["service_continuity_mean"],
            }
        )
        rows.append(out)
    return rows


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def table_power_and_coupling(rows):
    selected = [row for row in rows if row["family"] in {"power_stress", "coupling"}]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Power oriented stress and coupling sensitivity of the rollout policies.}",
        r"\label{tab:smartgrid-stress}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Family & Case & PC cost & Gain vs. neural & Feeder violation & Unserved charging \\",
        r"\midrule",
    ]
    names = {"power_stress": "Grid stress", "coupling": "Cyber power coupling"}
    for row in selected:
        lines.append(
            f"{names[row['family']]} & {row['label']} & {fmt(row['pc_cost'], 1)} & "
            f"{fmt(row['pc_gain_neural'], 1)}\\% & {fmt(row['pc_voltage'], 2)} & "
            f"{fmt(row['pc_unserved'], 1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def table_restoration(rows):
    selected = [row for row in rows if row["family"] == "restoration"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Restoration budget sensitivity under stressed charging operation.}",
        r"\label{tab:restore-sensitivity}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Budget & PC cost & Gain & Feeder viol. & Continuity \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['label']} & {fmt(row['pc_cost'], 1)} & {fmt(row['pc_gain_neural'], 1)}\\% & "
            f"{fmt(row['pc_voltage'], 2)} & {fmt(row['pc_continuity'], 4)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def stress_figure(rows, out: Path):
    selected = [row for row in rows if row["family"] == "power_stress"]
    selected.sort(key=lambda row: {"Normal": 0, "Elevated": 1, "Severe": 2}[row["label"]])
    labels = [row["label"] for row in selected]
    x_pos = list(range(len(selected)))
    width = 0.34
    neural_color = "#4c78a8"
    pc_color = "#2f9d55"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))

    feeder_ax, unserved_ax = axes
    neural_feeder = [math.log10(1.0 + float(row["neural_voltage"])) for row in selected]
    pc_feeder = [math.log10(1.0 + float(row["pc_voltage"])) for row in selected]
    feeder_ax.bar([x - width / 2 for x in x_pos], neural_feeder, width, label="Neural rollout", color=neural_color, alpha=0.86)
    feeder_ax.bar([x + width / 2 for x in x_pos], pc_feeder, width, label="PC rollout", color=pc_color, alpha=0.86)
    feeder_ax.set_xticks(x_pos)
    feeder_ax.set_xticklabels(labels, fontsize=8)
    feeder_ax.set_ylabel("log10(1 + feeder violation)")
    feeder_ax.grid(True, axis="y", color="#e8e8e8", linewidth=0.8)
    feeder_ax.spines[["top", "right"]].set_visible(False)

    neural_unserved = [float(row["neural_unserved"]) for row in selected]
    pc_unserved = [float(row["pc_unserved"]) for row in selected]
    unserved_ax.bar([x - width / 2 for x in x_pos], neural_unserved, width, label="Neural rollout", color=neural_color, alpha=0.86)
    unserved_ax.bar([x + width / 2 for x in x_pos], pc_unserved, width, label="PC rollout", color=pc_color, alpha=0.86)
    unserved_ax.set_xticks(x_pos)
    unserved_ax.set_xticklabels(labels, fontsize=8)
    unserved_ax.set_ylabel("Unserved charging demand")
    unserved_ax.grid(True, axis="y", color="#e8e8e8", linewidth=0.8)
    unserved_ax.spines[["top", "right"]].set_visible(False)
    unserved_ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/smartgrid"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()

    rows = collect(args.results)
    write_csv(args.results / "smartgrid_summary.csv", rows)
    args.paper.mkdir(parents=True, exist_ok=True)
    (args.paper / "table_smartgrid_stress.tex").write_text(table_power_and_coupling(rows), encoding="utf-8")
    (args.paper / "table_restore_sensitivity.tex").write_text(table_restoration(rows), encoding="utf-8")
    stress_figure(rows, args.paper / "fig_smartgrid_stress.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
