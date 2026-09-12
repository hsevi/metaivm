"""
Experiment 2 — Feature set ablation (exploratory dump).

NOTE: The canonical generator for Table 6 (tab:ablation) in the paper is
`scripts/ablation_table6.py`, which evaluates exactly the Table 6 rows against
the committed FEATURE_SETS. This module iterates all FEATURE_SETS keys as a
broader exploratory sweep and is kept for convenience; use ablation_table6.py
to reproduce the paper table.

Fix model = XGBoost, vary feature set.

KEY comparison: "IVM as raw ranker" (baselines) versus "IVM only" (learned
XGBoost model using only IVM scores as features), demonstrating that *learning
to combine* IVMs still falls far short of the partition-structural feature set.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.run_evaluation import evaluate_method
from src.evaluation.metrics import aggregate_metrics
from src.models.tabular_models import XGBoostModel, FEATURE_SETS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_experiment(seeds=None, model_class=None):
    """
    Run Experiment 2: feature ablation.

    Args:
        seeds: Split seeds.
        model_class: Model class to use (default: XGBoostModel).

    Returns:
        DataFrame with ablation results.
    """
    if seeds is None:
        with open(PROJECT_ROOT / "configs" / "experiment_configs.yaml") as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["splits"]["seeds"]

    if model_class is None:
        model_class = XGBoostModel

    features_df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    master = pd.read_csv(FEATURES_DIR / "runs_master.csv")
    if "hyperparams_json" not in features_df.columns:
        features_df = features_df.merge(
            master[["dataset_id", "run_id", "hyperparams_json"]],
            on=["dataset_id", "run_id"],
            how="left",
        )

    feature_set_names = list(FEATURE_SETS.keys())
    all_results = []
    all_per_dataset = {}  # fs_name -> list of per-dataset metric dicts

    for fs_name in feature_set_names:
        logger.info(f"=== Feature set: {fs_name} ===")
        method = model_class(feature_set=fs_name)

        result = evaluate_method(
            method, features_df, seeds,
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
        )

        row = {"feature_set": fs_name}
        for metric, stats in result["overall"].items():
            if metric in ("dataset_id", "seed"):
                continue
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        all_results.append(row)
        all_per_dataset[fs_name] = result["per_dataset"]

    results_df = pd.DataFrame(all_results)

    # Save per-seed results
    for seed in seeds:
        seed_dir = RESULTS_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_rows = []
        for fs_name, per_ds in all_per_dataset.items():
            seed_metrics = [m for m in per_ds if m["seed"] == seed]
            if seed_metrics:
                agg = aggregate_metrics(seed_metrics)
                flat = {"feature_set": fs_name}
                for metric, stats in agg.items():
                    if metric in ("dataset_id", "seed"):
                        continue
                    flat[f"{metric}_mean"] = stats["mean"]
                    flat[f"{metric}_std"] = stats["std"]
                seed_rows.append(flat)

        if seed_rows:
            pd.DataFrame(seed_rows).to_csv(
                seed_dir / "exp2_feature_ablation.csv", index=False
            )

    # Save aggregated
    agg_dir = RESULTS_DIR / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(agg_dir / "exp2_feature_ablation.csv", index=False)

    # Log the KEY comparison: IVM-only learned model vs raw IVM rankers
    _log_ivm_comparison(results_df)

    return results_df


def _log_ivm_comparison(results_df):
    """Log the key comparison between IVM-as-raw-ranker and IVM-only learned model."""
    ivm_only = results_df[results_df["feature_set"] == "ivm_only"]
    if ivm_only.empty:
        return

    ivm_regret = ivm_only["regret_mean"].values[0]
    ivm_spearman = ivm_only["spearman_mean"].values[0]

    logger.info("=" * 60)
    logger.info("KEY COMPARISON: Learning to combine IVMs")
    logger.info(f"  XGBoost(ivm_only) regret  = {ivm_regret:.4f}")
    logger.info(f"  XGBoost(ivm_only) spearman = {ivm_spearman:.4f}")
    logger.info(
        "  Compare with Table 1 IVM ranker baselines (Silhouette, CH, DB)"
    )
    logger.info(
        "  If ivm_only regret < best single IVM regret, learning helps."
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_experiment()
    print("\n=== Experiment 2: Feature Ablation ===")
    print(results.to_string(index=False))
