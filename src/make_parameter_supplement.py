#!/usr/bin/env python3
"""Render the parameter table as a standalone supplement."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config" / "full_parameter_ledger.csv"
OUTPUT = ROOT / "paper" / "supplementary_parameters.tex"


def tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_\allowbreak{}",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def display(value: str) -> str:
    """Use manuscript wording without changing the calculation settings."""
    direct = {
        "mapping_seed": "mapping_index",
        "seeded station-to-load embeddings": "predefined station to load assignments",
        "geographic rank and seeded assignment": "geographic rank and fixed initialization",
        "station mapping random seed": "station mapping initialization index",
        "prespecified seed": "fixed before evaluation",
        "all mappings": "all mapping assignments",
        "seed 2026": "fixed replicate index 2026",
    }
    rendered = direct.get(str(value), str(value))
    rendered = rendered.replace("min(0.25;", "min(0.25,").replace("max(q0.995*1.2;", "max(q0.995*1.2,")
    rendered = rendered.replace("undiscounted implementation", "undiscounted calculation")
    rendered = rendered.replace("numerical implementation", "numerical calculation")
    rendered = rendered.replace("implementation rule", "fixed route rule")
    rendered = rendered.replace("implementation setting", "calculation setting")
    rendered = rendered.replace("SMART-DS", "SMARTDS_TOKEN").replace("EAGLE-I", "EAGLEI_TOKEN")
    rendered = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", rendered)
    rendered = re.sub(r"(?<=\d)-(?=\d)", " to ", rendered)
    return rendered.replace("SMARTDS_TOKEN", "SMART-DS").replace("EAGLEI_TOKEN", "EAGLE-I")


def main() -> int:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        r"\documentclass[9pt]{article}",
        r"\usepackage[a4paper,margin=12mm]{geometry}",
        r"\usepackage{booktabs,longtable,array,pdflscape,ragged2e}",
        r"\usepackage[T1]{fontenc}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\LTpre}{2pt}",
        r"\setlength{\LTpost}{2pt}",
        r"\setlength{\emergencystretch}{1em}",
        r"\begin{document}",
        r"\renewcommand{\thetable}{S\arabic{table}}",
        r"\begin{landscape}",
        r"\begin{center}",
        r"{\Large Supplementary Parameter Table}\par",
        r"\vspace{3pt}",
        r"{\small Transition Based Closed Loop Restoration for Mobility Coupled Distribution Systems with Executable Actions}",
        r"\end{center}",
        r"\textbf{Scope.} This table reports the implemented value, unit, role, calibration rule, selection data, and tested range for every manuscript coefficient and major execution setting.  Dimensionless objective weights define a benchmark score rather than monetary cost.",
        r"\tiny",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{longtable}{>{\RaggedRight\arraybackslash}p{0.06\linewidth}>{\RaggedRight\arraybackslash}p{0.07\linewidth}>{\RaggedRight\arraybackslash}p{0.09\linewidth}>{\RaggedRight\arraybackslash}p{0.06\linewidth}>{\RaggedRight\arraybackslash}p{0.13\linewidth}>{\RaggedRight\arraybackslash}p{0.14\linewidth}>{\RaggedRight\arraybackslash}p{0.14\linewidth}>{\RaggedRight\arraybackslash}p{0.12\linewidth}}",
        r"\caption{Complete parameters and numerical settings.}\label{tab:complete-parameters}\\",
        r"\toprule",
        r"Group & Symbol & Value & Unit & Role & Source / calibration & Selection data & Sensitivity / range \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{l}{\textit{Table S1 continued}}\\",
        r"\toprule",
        r"Group & Symbol & Value & Unit & Role & Source / calibration & Selection data & Sensitivity / range \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{8}{r}{\textit{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in rows:
        fields = [
            row["group"],
            row["symbol"],
            row["value"],
            row["unit"],
            row["role"],
            row["source_or_calibration"],
            row["selection_data"],
            row["sensitivity_or_range"],
        ]
        lines.append(" & ".join(tex(display(field)) for field in fields) + r" \\")
    lines.extend(
        [
            r"\end{longtable}",
            r"\textbf{Execution note.} Continuous restoration values are requested priorities used for candidate scoring.  They do not reduce the realized threat state.  Packet delivery determines dispatched work orders.  The deterministic integer route records completion events.  Only those events clear the corresponding realized threat state.",
            r"\end{landscape}",
            r"\end{document}",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
