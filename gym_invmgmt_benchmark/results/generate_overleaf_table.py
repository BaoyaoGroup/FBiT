#!/usr/bin/env python3
"""Generate the compact Overleaf table from the archived mean/std CSV."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "fbit_final_vs_official_all_mean_sd.csv"
OUTPUT = HERE / "fbit_final_vs_official_all_mean_sd_overleaf.tex"

COLUMNS = [
    ("排名", "Rank"),
    ("Method", "Method"),
    ("Profit_All", r"Profit$_{\mathrm{All}}$"),
    ("Profit_Base", r"Profit$_{\mathrm{Base}}$"),
    ("Profit_Serial", r"Profit$_{\mathrm{Serial}}$"),
    ("FillRate_All(%)", r"Fill rate$_{\mathrm{All}}$ (\%)"),
    ("ServiceLevel_All(%)", r"Service level$_{\mathrm{All}}$ (\%)"),
    ("Unfulfilled_All", r"Unfulfilled$_{\mathrm{All}}$"),
    ("AvgInv_All", r"Avg. inv.$_{\mathrm{All}}$"),
]

# Explicit labels make the information condition visible and avoid treating
# DAgger-B/DAgger-G's model-variant suffixes as Blind/Informed indicators.
METHOD_LABELS = {
    "Oracle": "Oracle",
    "MSSP-I": "MSSP (I)",
    "MSSP": "MSSP (B)",
    "FBiT-final": "FBiT-final (B)",
    "PPO-Transformer": "PPO-Transformer (I)",
    "Residual": "Residual (I)",
    "Residual-B": "Residual (B)",
    "EchelonApprox-I": "EchelonApprox (I)",
    "EchelonApprox": "EchelonApprox (B)",
    "s,S-I": "(s,S) (I)",
    "s,S": "(s,S) (B)",
    "Newsvendor-I": "Newsvendor (I)",
    "Newsvendor": "Newsvendor (B)",
    "ExpSmoothing-I": "ExpSmoothing (I)",
    "ExpSmoothing": "ExpSmoothing (B)",
    "DLP-I": "DLP (I)",
    "DLP": "DLP (B)",
    "PPO-GNN": "PPO-GNN (I)",
    "PPO-GNN-B": "PPO-GNN (B)",
    "ST-PPO": "ST-PPO (I)",
    "ST-PPO-B": "ST-PPO (B)",
    "SAC": "SAC (I)",
    "SAC-B": "SAC (B)",
    "PPO-MLP": "PPO-MLP (I)",
    "PPO-MLP-B": "PPO-MLP (B)",
    "PPO-MLP-v1": "PPO-MLP-v1 (I)",
}


def latex_text(value: str) -> str:
    """Escape ordinary text for a LaTeX table cell."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def metric(value: str) -> str:
    """Typeset mean +/- standard deviation in math mode."""
    mean, sd = value.split("±", maxsplit=1)
    return rf"${mean} \pm {sd}$"


def render() -> str:
    with INPUT.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    lines = [
        r"% Required packages: \usepackage{booktabs,graphicx}",
        r"%                    \usepackage[table]{xcolor}",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Performance comparison (mean $\pm$ standard deviation). "
        r"The suffixes (I) and (B) denote informed and blind information access, "
        r"respectively. Methods are ranked by mean overall profit.}",
        r"  \label{tab:fbit-final-comparison}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.8pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{clccccccc}",
        r"    \toprule",
        "    " + " & ".join(label for _, label in COLUMNS) + r" \\",
        r"    \midrule",
    ]

    for row in rows:
        method = METHOD_LABELS.get(row["Method"], row["Method"])
        values = [row["排名"], latex_text(method)]
        values.extend(metric(row[key]) for key, _ in COLUMNS[2:])
        if row["Method"] == "FBiT-final":
            body = " & ".join(r"\bfseries\boldmath " + value for value in values)
            lines.append(r"    \rowcolor{gray!15}" + body + r" \\")
        else:
            body = " & ".join(values) + r" \\"
            lines.append("    " + body)

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT.write_text(render(), encoding="utf-8")
    print(OUTPUT)
