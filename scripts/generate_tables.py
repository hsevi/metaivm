"""
Generate LaTeX tables from experiment results.

Reads per-seed CSVs from results/seed_{i}/, computes mean +/- std,
and outputs .tex files ready for the paper.  Also reads the aggregated
CSVs as a fallback when per-seed files are not yet available.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
AGG_DIR = RESULTS_DIR / "aggregated"

SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(mean, std, bold=False):
    """Format mean +/- std for LaTeX."""
    s = f"{mean:.3f} \\pm {std:.3f}"
    if bold:
        return f"\\mathbf{{{s}}}"
    return f"${s}$"


def _read_seed_csvs(filename, key_col="method"):
    """Read per-seed CSVs and return a list of DataFrames (one per seed).

    Falls back to the aggregated CSV if per-seed files are missing.
    """
    frames = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"seed_{seed}" / filename
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        agg_path = AGG_DIR / filename
        if agg_path.exists():
            logger.info(f"No per-seed files for {filename}; using aggregated")
            return [pd.read_csv(agg_path)]
        return []

    return frames


def _aggregate_seed_frames(frames, key_col, metric_cols):
    """Compute mean +/- std across seed DataFrames for each metric.

    Returns a DataFrame with columns: key_col, metric_mean, metric_std
    for each metric in *metric_cols*.
    """
    if not frames:
        return pd.DataFrame()

    # Stack all seed frames and group by key
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for key_val, grp in combined.groupby(key_col):
        row = {key_col: key_val}
        for mc in metric_cols:
            # The per-seed CSVs may already store *_mean columns (from
            # per-seed aggregation).  Use those if available, otherwise
            # fall back to the raw column name.
            col = f"{mc}_mean" if f"{mc}_mean" in grp.columns else mc
            if col in grp.columns:
                vals = grp[col].dropna()
                row[f"{mc}_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{mc}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_table1():
    """Table 1: Main comparison results (mean +/- std over seeds)."""
    # Try per-seed first
    frames = _read_seed_csvs("exp1_main_table.csv", key_col="method")
    if frames:
        df = _aggregate_seed_frames(
            frames, "method",
            ["regret", "eps_success", "ndcg_at_k", "spearman"],
        )
    else:
        path = AGG_DIR / "exp1_main_table.csv"
        if not path.exists():
            logger.warning("exp1_main_table.csv not found")
            return ""
        df = pd.read_csv(path)

    if df.empty:
        return ""

    # Find best (lowest regret)
    best_regret_idx = df["regret_mean"].idxmin()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main comparison of clustering evaluation methods. "
        r"Regret = best\_true\_AMI $-$ true\_AMI of top-1 predicted. "
        r"Mean $\pm$ std over 5 seeds.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Regret ($\downarrow$) & $\epsilon$-success@0.01 ($\uparrow$) "
        r"& NDCG@5 ($\uparrow$) & Spearman ($\uparrow$) \\",
        r"\midrule",
    ]

    for idx, row in df.iterrows():
        name = row["method"]
        bold_r = idx == best_regret_idx
        cols = [
            name,
            _fmt(row["regret_mean"], row["regret_std"], bold=bold_r),
            _fmt(row["eps_success_mean"], row["eps_success_std"]),
            _fmt(row["ndcg_at_k_mean"], row["ndcg_at_k_std"]),
            _fmt(row["spearman_mean"], row["spearman_std"]),
        ]
        lines.append(" & ".join(cols) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out_path = AGG_DIR / "table1_main.tex"
    out_path.write_text(tex)
    logger.info(f"Saved Table 1 to {out_path}")
    return tex


def generate_table2():
    """Table 2: Feature ablation results (mean +/- std over seeds)."""
    frames = _read_seed_csvs("exp2_feature_ablation.csv", key_col="feature_set")
    if frames:
        df = _aggregate_seed_frames(
            frames, "feature_set",
            ["regret", "ndcg_at_k", "spearman"],
        )
    else:
        path = AGG_DIR / "exp2_feature_ablation.csv"
        if not path.exists():
            logger.warning("exp2_feature_ablation.csv not found")
            return ""
        df = pd.read_csv(path)

    if df.empty:
        return ""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Feature set ablation (XGBoost model). Mean $\pm$ std over 5 seeds.}",
        r"\label{tab:features}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Feature Set & Regret ($\downarrow$) & NDCG@5 ($\uparrow$) & Spearman ($\uparrow$) \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        fs = row["feature_set"].replace("_", r"\_")
        cols = [
            fs,
            _fmt(row["regret_mean"], row["regret_std"]),
            _fmt(row["ndcg_at_k_mean"], row["ndcg_at_k_std"]),
            _fmt(row["spearman_mean"], row["spearman_std"]),
        ]
        lines.append(" & ".join(cols) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out_path = AGG_DIR / "table2_features.tex"
    out_path.write_text(tex)
    logger.info(f"Saved Table 2 to {out_path}")
    return tex


def generate_table3():
    """Table 3: Leave-one-algo-out results (mean +/- std over seeds)."""
    frames = _read_seed_csvs("exp3_leave_one_algo.csv", key_col="held_out_algo")
    if frames:
        df = _aggregate_seed_frames(
            frames, "held_out_algo",
            ["regret_with", "regret_without"],
        )
        if not df.empty:
            df["delta_regret"] = df["regret_without_mean"] - df["regret_with_mean"]
    else:
        path = AGG_DIR / "exp3_leave_one_algo.csv"
        if not path.exists():
            logger.warning("exp3_leave_one_algo.csv not found")
            return ""
        df = pd.read_csv(path)

    if df.empty:
        return ""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Leave-one-algorithm-out evaluation. Mean $\pm$ std over 5 seeds.}",
        r"\label{tab:leave_algo}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Held-out Algo & Regret (with) & Regret (without) & $\Delta$ \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        algo = row["held_out_algo"]
        cols = [
            algo,
            _fmt(row["regret_with_mean"], row["regret_with_std"]),
            _fmt(row["regret_without_mean"], row["regret_without_std"]),
            f"${row['delta_regret']:+.3f}$",
        ]
        lines.append(" & ".join(cols) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out_path = AGG_DIR / "table3_leave_algo.tex"
    out_path.write_text(tex)
    logger.info(f"Saved Table 3 to {out_path}")
    return tex


def generate_table4():
    """Table 4: Runtime comparison."""
    path = AGG_DIR / "exp4_runtime.csv"
    if not path.exists():
        logger.warning("exp4_runtime.csv not found")
        return ""

    df = pd.read_csv(path)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Runtime comparison (seconds per dataset).}",
        r"\label{tab:runtime}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Method & Time/dataset (sec) \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(f"{row['method']} & ${row['time_per_dataset_sec']:.2f}$ \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out_path = AGG_DIR / "table4_runtime.tex"
    out_path.write_text(tex)
    logger.info(f"Saved Table 4 to {out_path}")
    return tex


def generate_table5():
    """Table 5: IVM correlation comparison (from Experiment 5 diagnostics)."""
    path = AGG_DIR / "exp5_diagnostics" / "correlation_comparison.csv"
    if not path.exists():
        logger.warning("correlation_comparison.csv not found")
        return ""

    df = pd.read_csv(path)
    # The last row is the MEAN (+/- std) summary row
    summary = df[df["dataset_id"] == "MEAN (+/- std)"]
    detail = df[df["dataset_id"] != "MEAN (+/- std)"]

    if summary.empty:
        logger.warning("No summary row in correlation_comparison.csv")
        return ""

    rho_cols = [c for c in df.columns if c.startswith("rho_")]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Spearman correlation between predicted scores and true AMI, "
        r"averaged across test datasets. Compares raw IVM rankers with the "
        r"learned model's predictions.}",
        r"\label{tab:correlation}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Score Source & Mean Spearman $\rho$ \\",
        r"\midrule",
    ]

    for col in rho_cols:
        label = col.replace("rho_", "").replace("_", " ")
        val = summary[col].values[0]
        lines.append(f"{label} & {val} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out_path = AGG_DIR / "table5_correlation.tex"
    out_path.write_text(tex)
    logger.info(f"Saved Table 5 to {out_path}")
    return tex


def generate_all():
    """Generate all LaTeX tables."""
    generate_table1()
    generate_table2()
    generate_table3()
    generate_table4()
    generate_table5()
    logger.info("All tables generated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all()
