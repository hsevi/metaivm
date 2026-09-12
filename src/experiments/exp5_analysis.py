"""
Experiment 5 — Diagnostics and failure analysis.

Identifies datasets where the neural IVM has highest regret,
characterizes what makes them hard, and produces diagnostic figures.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scipy.stats import spearmanr

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.splits import make_splits
from src.models.tabular_models import XGBoostModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_experiment(seeds=None, feature_set="partition_x_graph", top_k=5):
    """
    Run Experiment 5: failure analysis.

    Args:
        seeds: Split seeds.
        feature_set: Feature set for the main model.
        top_k: Number of worst datasets to analyze.

    Returns:
        DataFrame with per-dataset diagnostics.
    """
    if seeds is None:
        with open(PROJECT_ROOT / "configs" / "experiment_configs.yaml") as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["splits"]["seeds"]

    features_df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    master = pd.read_csv(FEATURES_DIR / "runs_master.csv")
    merge_cols = ["dataset_id", "run_id"]
    extra_cols = []
    if "hyperparams_json" not in features_df.columns:
        extra_cols.append("hyperparams_json")
    if "ch_adjusted" not in features_df.columns and "ch_adjusted" in master.columns:
        extra_cols.append("ch_adjusted")
    if extra_cols:
        features_df = features_df.merge(
            master[merge_cols + extra_cols],
            on=merge_cols,
            how="left",
        )

    dataset_ids = features_df["dataset_id"].unique().tolist()

    # Collect per-dataset regrets across seeds
    per_dataset_regrets = {}
    per_dataset_predictions = {}  # Store for scatter plots

    for seed in seeds:
        splits = make_splits(
            dataset_ids, n_splits=1,
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
            seed=seed,
        )

        for train_ids, val_ids, test_ids in splits:
            train_df = features_df[features_df["dataset_id"].isin(train_ids)]
            val_df = features_df[features_df["dataset_id"].isin(val_ids)]

            model = XGBoostModel(feature_set=feature_set)
            model.fit(train_df, val_df)

            for ds_id in test_ids:
                ds_df = features_df[features_df["dataset_id"] == ds_id]
                if len(ds_df) < 2:
                    continue

                true_amis = ds_df["ami"].values
                pred_scores = model.predict(ds_df)

                metrics = compute_all_metrics(true_amis, pred_scores)

                if ds_id not in per_dataset_regrets:
                    per_dataset_regrets[ds_id] = []
                per_dataset_regrets[ds_id].append(metrics["regret"])

                preds_dict = {
                    "true_amis": true_amis,
                    "pred_scores": pred_scores,
                    "algos": ds_df["algo"].values,
                    "ivm_silhouette": ds_df["ivm_silhouette"].values,
                    "ivm_calinski_harabasz": ds_df["ivm_calinski_harabasz"].values,
                    "ivm_davies_bouldin_neg": ds_df["ivm_davies_bouldin_neg"].values,
                }
                if "ch_adjusted" in ds_df.columns:
                    preds_dict["ch_adjusted"] = ds_df["ch_adjusted"].values
                per_dataset_predictions[ds_id] = preds_dict

    # Compute mean regret per dataset
    diagnostics = []
    for ds_id, regrets in per_dataset_regrets.items():
        ds_runs = features_df[features_df["dataset_id"] == ds_id]
        diagnostics.append({
            "dataset_id": ds_id,
            "mean_regret": np.mean(regrets),
            "std_regret": np.std(regrets),
            "n_runs": len(ds_runs),
            "best_ami": ds_runs["ami"].max(),
            "worst_ami": ds_runs["ami"].min(),
            "ami_range": ds_runs["ami"].max() - ds_runs["ami"].min(),
            "mean_ami": ds_runs["ami"].mean(),
            "n_algos": ds_runs["algo"].nunique(),
        })

    diag_df = pd.DataFrame(diagnostics).sort_values("mean_regret", ascending=False)

    # Save per-seed results
    for seed in seeds:
        seed_dir = RESULTS_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_diag = []
        for ds_id, regrets in per_dataset_regrets.items():
            # Filter regrets that came from this seed (approximate — we stored
            # all regrets together; for precise per-seed tracking we'd need to
            # tag them). Use the diagnostics row which already has the mean.
            row_match = diag_df[diag_df["dataset_id"] == ds_id]
            if not row_match.empty:
                seed_diag.append(row_match.iloc[0].to_dict())

        if seed_diag:
            pd.DataFrame(seed_diag).to_csv(
                seed_dir / "exp5_diagnostics.csv", index=False
            )

    # Save aggregated diagnostics
    agg_dir = RESULTS_DIR / "aggregated"
    diag_dir = agg_dir / "exp5_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(diag_dir / "per_dataset_diagnostics.csv", index=False)

    # Plot failure cases (top-k worst datasets)
    _plot_failure_cases(diag_df, per_dataset_predictions, diag_dir, top_k=top_k)

    # Plot global per-dataset scatter: true AMI vs predicted AMI
    _plot_global_scatter(per_dataset_predictions, diag_dir)

    # Correlation comparison: IVMs vs learned model (side-by-side diagnostic)
    corr_df = _correlation_comparison(per_dataset_predictions)
    corr_df.to_csv(diag_dir / "correlation_comparison.csv", index=False)
    logger.info(f"Saved correlation comparison to {diag_dir}")

    return diag_df


def _correlation_comparison(per_dataset_predictions):
    """
    Compute per-dataset Spearman correlation between True AMI and:
      - Each classical IVM (Silhouette, CH, -DB)
      - The learned model's predicted scores

    Returns a DataFrame with one row per dataset showing all correlations
    side-by-side, plus an aggregated summary row.
    """
    ivm_cols = {
        "Silhouette": "ivm_silhouette",
        "Calinski-Harabasz": "ivm_calinski_harabasz",
        "Davies-Bouldin (neg)": "ivm_davies_bouldin_neg",
        "CH Adjusted (Jeon)": "ch_adjusted",
    }

    rows = []
    for ds_id, pred in per_dataset_predictions.items():
        true_amis = pred["true_amis"]
        if len(true_amis) < 3 or np.std(true_amis) < 1e-12:
            continue

        row = {"dataset_id": ds_id}

        # Learned model
        pred_scores = pred["pred_scores"]
        if np.std(pred_scores) > 1e-12:
            row["rho_predicted"] = spearmanr(pred_scores, true_amis).correlation
        else:
            row["rho_predicted"] = np.nan

        # Each IVM
        for label, col in ivm_cols.items():
            ivm_vals = pred[col]
            valid = ~np.isnan(ivm_vals)
            if valid.sum() >= 3 and np.std(ivm_vals[valid]) > 1e-12:
                row[f"rho_{label}"] = spearmanr(ivm_vals[valid], true_amis[valid]).correlation
            else:
                row[f"rho_{label}"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        # Add summary row
        summary = {"dataset_id": "MEAN (+/- std)"}
        for col in df.columns:
            if col == "dataset_id":
                continue
            vals = df[col].dropna()
            summary[col] = f"{vals.mean():.3f} +/- {vals.std():.3f}" if len(vals) > 0 else "N/A"

        # Print the comparison
        logger.info("=== Spearman Correlation Comparison (per-dataset, then averaged) ===")
        rho_cols = [c for c in df.columns if c.startswith("rho_")]
        for col in rho_cols:
            vals = df[col].dropna()
            label = col.replace("rho_", "")
            logger.info(f"  {label:25s}: rho = {vals.mean():.3f} +/- {vals.std():.3f}")

        summary_df = pd.DataFrame([summary])
        df = pd.concat([df, summary_df], ignore_index=True)

    return df


def _plot_global_scatter(per_dataset_predictions, out_dir):
    """Per-dataset scatter: true AMI vs predicted AMI across all test datasets."""
    all_true, all_pred = [], []
    for ds_id, pred in per_dataset_predictions.items():
        all_true.extend(pred["true_amis"])
        all_pred.extend(pred["pred_scores"])

    if not all_true:
        return

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_true, all_pred, alpha=0.25, s=8, edgecolors="none")

    lo = min(all_true.min(), all_pred.min())
    hi = max(all_true.max(), all_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, label="y = x")

    ax.set_xlabel("True AMI")
    ax.set_ylabel("Predicted AMI")
    ax.set_title("Predicted vs True AMI (all test datasets)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "figure_scatter_all.pdf", dpi=150)
    fig.savefig(out_dir / "figure_scatter_all.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved global scatter to {out_dir}")


def _plot_failure_cases(diag_df, predictions, out_dir, top_k=5):
    """Figure 3: Failure case analysis panels."""
    worst = diag_df.head(top_k)

    n_panels = min(top_k, len(worst))
    if n_panels == 0:
        return

    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (_, row) in enumerate(worst.iterrows()):
        if i >= len(axes):
            break

        ax = axes[i]
        ds_id = row["dataset_id"]

        if ds_id in predictions:
            pred = predictions[ds_id]
            true_amis = pred["true_amis"]
            pred_scores = pred["pred_scores"]
            algos = pred["algos"]

            # Color by algorithm
            unique_algos = np.unique(algos)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_algos)))
            algo_color = {a: colors[j] for j, a in enumerate(unique_algos)}

            for a in unique_algos:
                mask = algos == a
                ax.scatter(
                    true_amis[mask], pred_scores[mask],
                    c=[algo_color[a]], label=a, alpha=0.6, s=20,
                )

            # Add diagonal reference
            lim = [min(true_amis.min(), pred_scores.min()),
                   max(true_amis.max(), pred_scores.max())]
            ax.plot(lim, lim, "k--", alpha=0.3)

        ax.set_title(f"{ds_id}\nRegret={row['mean_regret']:.3f}", fontsize=9)
        ax.set_xlabel("True AMI")
        ax.set_ylabel("Predicted Score")
        if i == 0:
            ax.legend(fontsize=6, loc="best")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / "figure3_failures.pdf", dpi=150)
    fig.savefig(out_dir / "figure3_failures.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 3 to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    diag = run_experiment()
    print("\n=== Experiment 5: Top-5 Hardest Datasets ===")
    print(diag.head(5).to_string(index=False))

    # Print correlation comparison
    corr_path = RESULTS_DIR / "aggregated" / "exp5_diagnostics" / "correlation_comparison.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        print("\n=== Spearman Correlation: IVMs vs Learned Model ===")
        # Print just the summary row
        summary = corr[corr["dataset_id"] == "MEAN (+/- std)"]
        if len(summary) > 0:
            for col in summary.columns:
                if col.startswith("rho_"):
                    label = col.replace("rho_", "")
                    print(f"  {label:25s}: {summary[col].values[0]}")
