#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(x, digits=2):
    return f"{float(x):.{digits}f}"


def best_by_mode(rows):
    out = {}
    for row in rows:
        mode = row["mode"]
        score = float(row["mae_demand"]) + 0.25 * float(row["mae_energy"])
        if mode not in out or score < out[mode][0]:
            out[mode] = (score, row)
    return {k: v[1] for k, v in out.items()}


def table_prediction(rows) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Forecasting accuracy on the held out NYC TLC 2023 test period.}",
        r"\label{tab:forecast}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Forecaster & Demand MAE & Demand RMSE & Energy MAE & Energy RMSE \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {fmt(row['mae_demand'])} & {fmt(row['rmse_demand'])} "
            f"& {fmt(row['mae_energy'])} & {fmt(row['rmse_energy'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def table_rollout(rows) -> str:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    order = ["static", "greedy", "plain_rollout", "pc_rollout", "oracle"]
    names = {
        "static": "Static",
        "greedy": "Greedy",
        "plain_rollout": "Neural rollout",
        "pc_rollout": "PC rollout",
        "oracle": "Oracle",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Threat responsive rollout performance aggregated over all scenario classes.}",
        r"\label{tab:rollout}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Policy & Cost $\downarrow$ & Mobility delay $\downarrow$ & Unserved energy $\downarrow$ & Voltage viol. $\downarrow$ & Continuity $\uparrow$ \\",
        r"\midrule",
    ]
    for policy in order:
        vals = grouped[policy]
        mean_cost = statistics.fmean(float(v["cost_mean"]) for v in vals)
        delay = statistics.fmean(float(v["mobility_delay_mean"]) for v in vals)
        energy = statistics.fmean(float(v["unserved_energy_mean"]) for v in vals)
        volt = statistics.fmean(float(v["voltage_violation_mean"]) for v in vals)
        cont = statistics.fmean(float(v["service_continuity_mean"]) for v in vals)
        bold_l = r"\textbf{" if policy == "pc_rollout" else ""
        bold_r = "}" if policy == "pc_rollout" else ""
        lines.append(
            f"{names[policy]} & {bold_l}{fmt(mean_cost, 1)}{bold_r} & "
            f"{bold_l}{fmt(delay, 1)}{bold_r} & {bold_l}{fmt(energy, 1)}{bold_r} & "
            f"{bold_l}{fmt(volt, 2)}{bold_r} & {bold_l}{fmt(cont, 4)}{bold_r} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def table_group(rows) -> str:
    by = {(row["policy"], row["group"]): row for row in rows}
    group_names = {
        "nominal": "Nominal",
        "single_domain": "Single domain",
        "cascade": "Cascading",
        "ood": "Unseen compound",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{PC rollout gains by threat class relative to neural rollout.}",
        r"\label{tab:threat-class}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Threat class & Neural cost & PC cost & Gain \\",
        r"\midrule",
    ]
    for group in ["nominal", "single_domain", "cascade", "ood"]:
        plain = float(by[("plain_rollout", group)]["cost_mean"])
        pc = float(by[("pc_rollout", group)]["cost_mean"])
        gain = 100.0 * (plain - pc) / plain
        lines.append(f"{group_names[group]} & {fmt(plain, 1)} & {fmt(pc, 1)} & {fmt(gain, 1)}\\% \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def macros(rollout_rows) -> str:
    by_policy = defaultdict(list)
    for row in rollout_rows:
        by_policy[row["policy"]].append(row)
    cost_plain = statistics.fmean(float(v["cost_mean"]) for v in by_policy["plain_rollout"])
    cost_greedy = statistics.fmean(float(v["cost_mean"]) for v in by_policy["greedy"])
    cost_pc = statistics.fmean(float(v["cost_mean"]) for v in by_policy["pc_rollout"])
    cost_oracle = statistics.fmean(float(v["cost_mean"]) for v in by_policy["oracle"])
    pc_vs_plain = 100.0 * (cost_plain - cost_pc) / cost_plain
    pc_vs_greedy = 100.0 * (cost_greedy - cost_pc) / cost_greedy
    oracle_gap = 100.0 * (cost_pc - cost_oracle) / cost_oracle
    return "\n".join(
        [
            f"\\newcommand{{\\CostGainPlain}}{{{pc_vs_plain:.1f}\\%}}",
            f"\\newcommand{{\\CostGainGreedy}}{{{pc_vs_greedy:.1f}\\%}}",
            f"\\newcommand{{\\OracleGap}}{{{oracle_gap:.1f}\\%}}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    forecast_table = args.results / "forecast_table.csv"
    pred = read_csv(forecast_table if forecast_table.exists() else args.results / "prediction_metrics.csv")
    rollout = read_csv(args.results / "rollout_summary.csv")
    args.paper.mkdir(parents=True, exist_ok=True)
    (args.paper / "table_forecast.tex").write_text(table_prediction(pred), encoding="utf-8")
    (args.paper / "table_rollout.tex").write_text(table_rollout(rollout), encoding="utf-8")
    (args.paper / "table_threat_class.tex").write_text(table_group(rollout), encoding="utf-8")
    (args.paper / "result_macros.tex").write_text(macros(rollout), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
