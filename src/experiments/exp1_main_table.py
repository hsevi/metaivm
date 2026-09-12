"""
Experiment 1 — Main comparison table.

Run all methods on all seeds. Produce Table 1 (main results)
and Figures 1-2 (regret boxplot, scatter).
"""

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.evaluation.run_evaluation import evaluate_method
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.splits import make_splits
from src.models.baselines import RandomBaseline, GlobalBestConfig, IVMRanker
from src.models.tabular_models import RidgeModel, MLPModel, XGBoostModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"


def build_methods(include_gnn=False):
    """Build all methods to evaluate."""
    methods = {
        "Random": RandomBaseline(seed=42),
        "Global best config": GlobalBestConfig(),
        "Silhouette": IVMRanker("silhouette"),
        "Calinski-Harabasz": IVMRanker("calinski_harabasz"),
        "Davies-Bouldin": IVMRanker("davies_bouldin"),
        "CH Adjusted (Jeon)": IVMRanker("ch_adjusted"),
        "Ridge": RidgeModel(feature_set="partition_x_graph"),
        "MLP": MLPModel(feature_set="partition_x_graph"),
        "MetaIVM (XGBoost)": XGBoostModel(feature_set="partition_x_graph"),
    }

    if include_gnn:
        try:
            from src.models.gnn_model import GNNSurrogate
            methods["GNN"] = GNNSurrogate()
        except ImportError:
            logger.warning("GNN model unavailable (torch_geometric not installed)")

    return methods


def run_experiment(seeds=None, include_gnn=False, include_stability=False):
    """
    Run Experiment 1: main comparison.

    Args:
        seeds: List of split seeds. Defaults to [0,1,2,3,4].
        include_gnn: Whether to include GNN model.
        include_stability: Whether to include stability baseline.

    Returns:
        DataFrame with results.
    """
    if seeds is None:
        with open(PROJECT_ROOT / "configs" / "experiment_configs.yaml") as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["splits"]["seeds"]

    features_df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    # Need hyperparams_json for some methods — merge from master
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

    methods = build_methods(include_gnn=include_gnn)

    if include_stability:
        from src.models.baselines import StabilityBaseline
        methods["Stability"] = StabilityBaseline()

    # Evaluate all methods
    all_results = []
    all_per_dataset = {}

    for name, method in methods.items():
        logger.info(f"=== Evaluating: {name} ===")
        result = evaluate_method(
            method, features_df, seeds,
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
        )

        row = {"method": name}
        for metric, stats in result["overall"].items():
            if metric in ("dataset_id", "seed"):
                continue
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        all_results.append(row)
        all_per_dataset[name] = result["per_dataset"]

    results_df = pd.DataFrame(all_results)

    # Save results per seed
    for seed in seeds:
        seed_dir = RESULTS_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_rows = []
        for name, per_ds in all_per_dataset.items():
            seed_metrics = [m for m in per_ds if m["seed"] == seed]
            if seed_metrics:
                avg = {}
                for k in seed_metrics[0]:
                    if k in ("dataset_id", "seed"):
                        continue
                    vals = [m[k] for m in seed_metrics if not np.isnan(m.get(k, np.nan))]
                    avg[k] = np.mean(vals) if vals else np.nan
                avg["method"] = name
                seed_rows.append(avg)

        pd.DataFrame(seed_rows).to_csv(seed_dir / "exp1_main_table.csv", index=False)

    # Save aggregated
    agg_dir = RESULTS_DIR / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(agg_dir / "exp1_main_table.csv", index=False)

    # Generate figures
    _plot_regret_boxplot(all_per_dataset, agg_dir)
    _plot_scatter(all_per_dataset, features_df, agg_dir)

    return results_df


def _plot_regret_boxplot(all_per_dataset, out_dir):
    """Figure 1: Regret distribution per method."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = []
    for name, metrics in all_per_dataset.items():
        for m in metrics:
            data.append({"Method": name, "Regret": m["regret"]})

    if not data:
        return

    df = pd.DataFrame(data)
    order = df.groupby("Method")["Regret"].median().sort_values().index.tolist()

    sns.boxplot(data=df, x="Method", y="Regret", order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Selection Regret Distribution by Method")
    ax.set_ylabel("Regret (lower is better)")
    plt.tight_layout()
    fig.savefig(out_dir / "figure1_regret_boxplot.pdf", dpi=150)
    fig.savefig(out_dir / "figure1_regret_boxplot.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 1 to {out_dir}")


def _plot_scatter(all_per_dataset, features_df, out_dir):
    """Figure 2: Per-dataset scatter — predicted AMI vs true AMI for best method.

    Re-runs the best method to collect per-run predicted scores alongside
    true AMI, then plots one point per (dataset, run) pair.
    """
    from src.evaluation.splits import make_splits

    # Find best method by mean regret
    method_regrets = {}
    for name, metrics in all_per_dataset.items():
        regrets = [m["regret"] for m in metrics]
        method_regrets[name] = np.mean(regrets) if regrets else np.inf

    best_method = min(method_regrets, key=method_regrets.get)
    logger.info(f"Best method for scatter: {best_method}")

    # Re-instantiate and re-run the best method so we can capture per-run
    # predicted scores (the harness only stores aggregate metrics).
    method_map = build_methods()
    if best_method not in method_map:
        logger.warning(f"Cannot re-instantiate {best_method} for scatter")
        return

    method = method_map[best_method]
    dataset_ids = features_df["dataset_id"].unique().tolist()

    with open(PROJECT_ROOT / "configs" / "experiment_configs.yaml") as f:
        cfg = yaml.safe_load(f)
    seeds = cfg["splits"]["seeds"]

    all_true, all_pred, all_ds = [], [], []

    for seed in seeds:
        splits = make_splits(
            dataset_ids, n_splits=1,
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
            seed=seed,
        )
        for train_ids, val_ids, test_ids in splits:
            train_df = features_df[features_df["dataset_id"].isin(train_ids)]
            val_df = features_df[features_df["dataset_id"].isin(val_ids)]
            method.fit(train_df, val_df)

            for ds_id in test_ids:
                ds_df = features_df[features_df["dataset_id"] == ds_id]
                if len(ds_df) < 2:
                    continue
                true_amis = ds_df["ami"].values
                pred_scores = method.predict(ds_df)
                all_true.extend(true_amis)
                all_pred.extend(pred_scores)
                all_ds.extend([ds_id] * len(true_amis))

    if not all_true:
        return

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_true, all_pred, alpha=0.3, s=10, edgecolors="none")

    # Diagonal reference line
    lo = min(all_true.min(), all_pred.min())
    hi = max(all_true.max(), all_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, label="y = x")

    ax.set_xlabel("True AMI")
    ax.set_ylabel("Predicted AMI")
    ax.set_title(f"Predicted vs True AMI — {best_method}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "figure2_scatter.pdf", dpi=150)
    fig.savefig(out_dir / "figure2_scatter.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved Figure 2 (scatter) to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_experiment(include_gnn=False, include_stability=False)
    print("\n=== Experiment 1: Main Comparison Table ===")
    print(results.to_string(index=False))
