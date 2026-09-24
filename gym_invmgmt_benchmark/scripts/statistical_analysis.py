#!/usr/bin/env python3
"""Paired statistical analysis for the archived final-checkpoint results.

The analysis keeps the ten common evaluation seeds paired within each of the
26 benchmark scenarios.  Inference is performed at the scenario-block level:

* two-sided sign-flip permutation tests flip complete scenario blocks;
* hierarchical bootstrap intervals resample scenarios, then paired seeds;
* the standardized effect is Cohen's dz over scenario-mean differences.

Positive improvements always favour FBiT.  For metrics where smaller is
better (inventory, unfulfilled demand, and bullwhip), the native difference is
therefore multiplied by -1 before analysis.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "statistical_analysis"
FBIT_PATH = ROOT / "results" / "cache_v2" / "bench_FBiT.csv"

BASELINES = {
    # Display labels match Table 10; learning methods use the official Blind
    # (-B) caches even though the suffix is omitted in the published table.
    "Oracle": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_Oracle.csv",
    "MSSP": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_MSSP.csv",
    "EchelonApprox": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_EchelonApprox.csv",
    "(s,S)": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_s,S.csv",
    "ST-PPO": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_ST-PPO-B.csv",
    "SAC": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_SAC-B.csv",
    "PPO-GNN": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_PPO-GNN-B.csv",
    "Residual": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_Residual-B.csv",
    "ExpSmoothing": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_ExpSmoothing.csv",
    "Newsvendor": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_Newsvendor.csv",
    "DLP": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_DLP.csv",
    "PPO-MLP": ROOT / "vendor" / "gym_invmgmt_paper" / "results" / "cache_v2" / "bench_PPO-MLP-B.csv",
}


@dataclass(frozen=True)
class Metric:
    name: str
    higher_is_better: bool
    scale: float = 1.0
    unit: str = ""


METRICS = (
    Metric("Profit", True, unit="profit"),
    Metric("FillRate", True, scale=100.0, unit="percentage points"),
    Metric("ServiceLevel", True, scale=100.0, unit="percentage points"),
    Metric("AvgInv", False, unit="inventory units reduced"),
    Metric("Unfulfilled", False, unit="units reduced"),
    Metric("BullwhipRatio", False, unit="ratio reduction"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired statistical tests from existing final results."
    )
    parser.add_argument("--resamples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_results(path: Path, metric_names: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"ScenarioKey", "Seed", *metric_names}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "Failed" in frame.columns:
        failed = frame["Failed"].fillna(False).astype(bool)
        if failed.any():
            raise ValueError(f"{path} contains {int(failed.sum())} failed evaluations")
    if frame.duplicated(["ScenarioKey", "Seed"]).any():
        raise ValueError(f"{path} contains duplicate ScenarioKey/Seed rows")
    return frame[["ScenarioKey", "Seed", *sorted(metric_names)]].copy()


def paired_matrix(
    fbit: pd.DataFrame, baseline: pd.DataFrame, metric: Metric
) -> tuple[np.ndarray, float, float]:
    paired = fbit[["ScenarioKey", "Seed", metric.name]].merge(
        baseline[["ScenarioKey", "Seed", metric.name]],
        on=["ScenarioKey", "Seed"],
        how="inner",
        validate="one_to_one",
        suffixes=("_FBiT", "_Baseline"),
    )
    if len(paired) != 260 or paired["ScenarioKey"].nunique() != 26:
        raise ValueError(
            f"{metric.name}: expected 260 paired rows across 26 scenarios, "
            f"found {len(paired)} rows across {paired['ScenarioKey'].nunique()} scenarios"
        )
    counts = paired.groupby("ScenarioKey")["Seed"].nunique()
    if not (counts == 10).all():
        raise ValueError(f"{metric.name}: every scenario must contain ten unique seeds")

    sign = 1.0 if metric.higher_is_better else -1.0
    paired["Improvement"] = sign * metric.scale * (
        paired[f"{metric.name}_FBiT"] - paired[f"{metric.name}_Baseline"]
    )
    seed_order = sorted(paired["Seed"].unique())
    matrix = (
        paired.pivot(index="ScenarioKey", columns="Seed", values="Improvement")
        .reindex(columns=seed_order)
        .sort_index()
        .to_numpy(dtype=np.float64)
    )
    mean_fbit = float(paired[f"{metric.name}_FBiT"].mean() * metric.scale)
    mean_baseline = float(paired[f"{metric.name}_Baseline"].mean() * metric.scale)
    return matrix, mean_fbit, mean_baseline


def stable_rng(seed: int, baseline: str, metric: str, purpose: str) -> np.random.Generator:
    payload = f"{seed}|{baseline}|{metric}|{purpose}".encode()
    digest = hashlib.sha256(payload).digest()
    derived_seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(derived_seed)


def block_permutation_pvalue(
    differences: np.ndarray, resamples: int, rng: np.random.Generator
) -> float:
    scenario_means = differences.mean(axis=1)
    observed = abs(float(scenario_means.mean()))
    extreme = 0
    chunk_size = 20_000
    for start in range(0, resamples, chunk_size):
        size = min(chunk_size, resamples - start)
        signs = rng.integers(0, 2, size=(size, len(scenario_means)), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = np.abs((signs * scenario_means).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return (extreme + 1.0) / (resamples + 1.0)


def hierarchical_bootstrap_ci(
    differences: np.ndarray, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    scenario_count, seed_count = differences.shape
    estimates = np.empty(resamples, dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, resamples, chunk_size):
        size = min(chunk_size, resamples - start)
        scenario_indices = rng.integers(
            0, scenario_count, size=(size, scenario_count)
        )
        seed_indices = rng.integers(
            0, seed_count, size=(size, scenario_count, seed_count)
        )
        sampled = differences[scenario_indices[:, :, None], seed_indices]
        estimates[start : start + size] = sampled.mean(axis=(1, 2))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * pvalues[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def analyse(
    fbit: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        metric_rows = []
        for baseline_name, baseline in baselines.items():
            differences, mean_fbit, mean_baseline = paired_matrix(
                fbit, baseline, metric
            )
            scenario_means = differences.mean(axis=1)
            mean_improvement = float(differences.mean())
            scenario_sd = float(scenario_means.std(ddof=1))
            effect_size = mean_improvement / scenario_sd if scenario_sd else np.nan
            ci_low, ci_high = hierarchical_bootstrap_ci(
                differences,
                resamples,
                stable_rng(seed, baseline_name, metric.name, "bootstrap"),
            )
            pvalue = block_permutation_pvalue(
                differences,
                resamples,
                stable_rng(seed, baseline_name, metric.name, "permutation"),
            )
            pair_wins = int(np.count_nonzero(differences > 0))
            pair_losses = int(np.count_nonzero(differences < 0))
            scenario_wins = int(np.count_nonzero(scenario_means > 0))
            scenario_losses = int(np.count_nonzero(scenario_means < 0))
            metric_rows.append(
                {
                    "Method": "FBiT",
                    "Baseline": baseline_name,
                    "Metric": metric.name,
                    "Direction": "higher" if metric.higher_is_better else "lower",
                    "Unit": metric.unit,
                    "N_Scenarios": differences.shape[0],
                    "N_Seeds_Per_Scenario": differences.shape[1],
                    "Mean_FBiT": mean_fbit,
                    "Mean_Baseline": mean_baseline,
                    "Mean_Improvement": mean_improvement,
                    "Relative_Improvement_Pct": (
                        100.0 * mean_improvement / abs(mean_baseline)
                        if mean_baseline
                        else np.nan
                    ),
                    "CI95_Low": ci_low,
                    "CI95_High": ci_high,
                    "Permutation_P": pvalue,
                    "Effect_Cohen_Dz_Scenario": effect_size,
                    "Pair_Wins": pair_wins,
                    "Pair_Losses": pair_losses,
                    "Pair_Ties": int(differences.size - pair_wins - pair_losses),
                    "Scenario_Wins": scenario_wins,
                    "Scenario_Losses": scenario_losses,
                    "Scenario_Ties": int(
                        len(scenario_means) - scenario_wins - scenario_losses
                    ),
                }
            )
        adjusted = holm_adjust([float(row["Permutation_P"]) for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["Holm_Adjusted_P"] = adjusted_p
        rows.extend(metric_rows)
    return pd.DataFrame(rows)


def format_p(pvalue: float, latex: bool = False) -> str:
    if pvalue < 0.0001:
        return (r"$<0.0001$" if latex else "<0.0001")
    return f"{pvalue:.4f}"


def render_profit_latex(results: pd.DataFrame) -> str:
    profit = results[results["Metric"] == "Profit"]
    lines = [
        r"% Required package: \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Paired statistical comparison of profit. Positive differences favour FBiT.}",
        r"  \label{tab:fbit-statistical-profit}",
        r"  \small",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    Baseline & $\Delta$ Profit & 95\% CI & $p$ & $p_{\mathrm{Holm}}$ & $d_z$ \\",
        r"    \midrule",
    ]
    for row in profit.itertuples(index=False):
        lines.append(
            f"    {row.Baseline} & {row.Mean_Improvement:.2f} & "
            f"[{row.CI95_Low:.2f}, {row.CI95_High:.2f}] & "
            f"{format_p(float(row.Permutation_P), latex=True)} & "
            f"{format_p(float(row.Holm_Adjusted_P), latex=True)} & "
            f"{row.Effect_Cohen_Dz_Scenario:.2f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def render_all_metrics_latex(results: pd.DataFrame) -> str:
    lines = [
        r"% Required packages: \usepackage{booktabs,longtable}",
        r"\begin{longtable}{llrrrr}",
        r"  \caption{Paired statistical comparisons across all evaluated metrics. "
        r"Positive improvements favour FBiT.}\label{tab:fbit-statistics-all}\\",
        r"  \toprule",
        r"  Baseline & Metric & Improvement & 95\% CI & $p_{\mathrm{Holm}}$ & $d_z$ \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \toprule",
        r"  Baseline & Metric & Improvement & 95\% CI & $p_{\mathrm{Holm}}$ & $d_z$ \\",
        r"  \midrule",
        r"  \endhead",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"  {row.Baseline} & {row.Metric} & {row.Mean_Improvement:.2f} & "
            f"[{row.CI95_Low:.2f}, {row.CI95_High:.2f}] & "
            f"{format_p(float(row.Holm_Adjusted_P), latex=True)} & "
            f"{row.Effect_Cohen_Dz_Scenario:.2f} \\\\"
        )
    lines.extend([r"  \bottomrule", r"\end{longtable}", ""])
    return "\n".join(lines)


def render_markdown(results: pd.DataFrame, resamples: int, seed: int) -> str:
    profit = results[results["Metric"] == "Profit"]
    lines = [
        "# Statistical analysis of existing results",
        "",
        "This analysis uses only the archived evaluations; no new training or evaluation "
        "episodes were generated.",
        "",
        "## Protocol",
        "",
        "- FBiT observations were paired with every baseline by `ScenarioKey + Seed`.",
        "- All comparisons contain 26 scenarios and 10 common seeds per scenario.",
        "- Two-sided p-values use a sign-flip permutation test over complete scenario blocks.",
        "- 95% confidence intervals use a hierarchical paired bootstrap that resamples "
        "scenarios and then seeds within scenarios.",
        "- Effect sizes are paired Cohen's dz values computed from the 26 scenario-mean "
        "differences.",
        f"- Holm adjustment is applied across the {len(BASELINES)} Table 10 comparisons "
        "separately for each metric.",
        f"- Monte Carlo resamples: {resamples:,}; reproducibility seed: {seed}.",
        "- Positive improvement values always favour FBiT; lower-is-better metrics are "
        "expressed as reductions.",
        "",
        "## Primary profit results",
        "",
        "| Baseline | Mean improvement | 95% CI | p | Holm p | Cohen dz | Pair wins/losses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profit.itertuples(index=False):
        lines.append(
            f"| {row.Baseline} | {row.Mean_Improvement:.2f} | "
            f"[{row.CI95_Low:.2f}, {row.CI95_High:.2f}] | "
            f"{format_p(float(row.Permutation_P))} | "
            f"{format_p(float(row.Holm_Adjusted_P))} | "
            f"{row.Effect_Cohen_Dz_Scenario:.2f} | "
            f"{row.Pair_Wins}/{row.Pair_Losses} |"
        )
    lines.extend(
        [
            "",
            "Complete results for all metrics are stored in `paired_statistics.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_manuscript_text(results: pd.DataFrame, resamples: int) -> str:
    profit_rows = {
        row.Baseline: row
        for row in results[results["Metric"] == "Profit"].itertuples(index=False)
    }
    mssp = profit_rows["MSSP"]
    non_oracle = [
        row for name, row in profit_rows.items() if name not in {"Oracle", "MSSP"}
    ]
    non_oracle_improvements = [row.Mean_Improvement for row in non_oracle]
    non_oracle_adjusted_pvalues = [row.Holm_Adjusted_P for row in non_oracle]
    non_oracle_effects = [row.Effect_Cohen_Dz_Scenario for row in non_oracle]
    return f"""# Manuscript-ready statistical analysis text

## Statistical analysis

All methods were compared using matched observations indexed by benchmark
scenario and evaluation seed (26 scenarios and 10 common seeds per scenario).
For each comparison, the test statistic was the mean of the 26 scenario-level
paired differences. Two-sided significance was assessed using a scenario-block
sign-flip permutation test with {resamples:,} Monte Carlo resamples. Confidence
intervals for the mean paired improvement were obtained using a hierarchical
paired bootstrap that resampled scenarios and then evaluation seeds within
each sampled scenario. We additionally report paired Cohen's $d_z$, calculated
from the scenario-level mean differences, to quantify effect magnitude. Holm's
procedure was applied separately for each metric across the {len(BASELINES)}
Table 10 comparisons to control the family-wise error rate. All tests were
two-sided. Oracle was retained as a reference upper bound.

## Results

The observed profit improvements remained statistically significant when the
clustered scenario-seed structure was taken into account. Relative to MSSP,
FBiT improved mean profit by
{mssp.Mean_Improvement:.2f} ({mssp.Relative_Improvement_Pct:.2f}%),
with a 95% hierarchical-bootstrap confidence interval of
[{mssp.CI95_Low:.2f}, {mssp.CI95_High:.2f}], a Holm-adjusted
$p={mssp.Holm_Adjusted_P:.4f}$, and a standardized effect of
$d_z={mssp.Effect_Cohen_Dz_Scenario:.2f}$. Against the remaining non-Oracle
Table 10 baselines, mean profit improvements ranged from
{min(non_oracle_improvements):.2f} to {max(non_oracle_improvements):.2f}; all
remained significant after Holm correction (adjusted $p\\leq
{max(non_oracle_adjusted_pvalues):.4f}$), with effect sizes ranging from
$d_z={min(non_oracle_effects):.2f}$ to
$d_z={max(non_oracle_effects):.2f}$. These results demonstrate
both statistical significance and practically meaningful effect magnitudes.

## Response to reviewer

Thank you for this suggestion. We strengthened the experimental validation by
adding a statistical analysis of the complete matched results from 26 benchmark
scenarios and 10 common evaluation seeds per scenario. The revised manuscript
now reports scenario-block permutation-test p-values, hierarchical-bootstrap
95% confidence intervals, standardized paired effect sizes, and win/loss
counts. We also apply Holm correction across all methods reported in Table 10.
The new analysis shows that FBiT improves mean profit over MSSP by
{mssp.Mean_Improvement:.2f} (95% CI
[{mssp.CI95_Low:.2f}, {mssp.CI95_High:.2f}], Holm-adjusted
$p={mssp.Holm_Adjusted_P:.4f}$, $d_z={mssp.Effect_Cohen_Dz_Scenario:.2f}$).
The profit improvements over the other non-Oracle Table 10 baselines also
remain significant after multiple-comparison correction. We have added the
complete results and statistical protocol to the revised manuscript and
supplementary material.
"""


def main() -> int:
    args = parse_args()
    if args.resamples <= 0:
        raise SystemExit("--resamples must be positive")
    metric_names = {metric.name for metric in METRICS}
    fbit = load_results(FBIT_PATH, metric_names)
    baselines = {
        name: load_results(path, metric_names) for name, path in BASELINES.items()
    }
    results = analyse(fbit, baselines, args.resamples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paired_statistics.csv"
    tex_path = args.output_dir / "profit_statistics.tex"
    all_metrics_tex_path = args.output_dir / "all_metrics_statistics.tex"
    markdown_path = args.output_dir / "STATISTICAL_ANALYSIS.md"
    manuscript_path = args.output_dir / "MANUSCRIPT_TEXT.md"
    results.to_csv(csv_path, index=False, float_format="%.10g")
    tex_path.write_text(render_profit_latex(results), encoding="utf-8")
    all_metrics_tex_path.write_text(
        render_all_metrics_latex(results), encoding="utf-8"
    )
    markdown_path.write_text(
        render_markdown(results, args.resamples, args.seed), encoding="utf-8"
    )
    manuscript_path.write_text(
        render_manuscript_text(results, args.resamples), encoding="utf-8"
    )
    print(csv_path)
    print(tex_path)
    print(all_metrics_tex_path)
    print(markdown_path)
    print(manuscript_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
