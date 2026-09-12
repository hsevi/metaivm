"""
Experiment 6 — Cross-domain transfer.

Tests whether meta-learned evaluator generalizes across dataset domains:
  - Train on synthetic → test on real (OpenML + image + text)
  - Train on real → test on synthetic
  - Train on OpenML → test on non-OpenML real (image + text)
  - Leave-one-domain-out

Compares XGBoost (ours) vs classical IVMs in each setting.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import compute_all_metrics, aggregate_metrics
from src.models.tabular_models import XGBoostModel
from src.models.baselines import IVMRanker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"


def _evaluate_split(model, features_df, train_ids, test_ids):
    """Evaluate a model on a fixed train/test split (no val, use train as val)."""
    train_df = features_df[features_df["dataset_id"].isin(train_ids)]
    test_df = features_df[features_df["dataset_id"].isin(test_ids)]

    if len(train_df) == 0 or len(test_df) == 0:
        return []

    # Use 80% train, 20% val from training set
    train_ds = list(set(train_ids))
    rng = np.random.default_rng(42)
    rng.shuffle(train_ds)
    split_pt = max(1, int(len(train_ds) * 0.8))
    actual_train = set(train_ds[:split_pt])
    actual_val = set(train_ds[split_pt:])

    tr_df = features_df[features_df["dataset_id"].isin(actual_train)]
    val_df = features_df[features_df["dataset_id"].isin(actual_val)]

    model.fit(tr_df, val_df)

    per_dataset = []
    for ds_id in test_ids:
        ds_df = test_df[test_df["dataset_id"] == ds_id]
        if len(ds_df) < 2:
            continue
        true_amis = ds_df["ami"].values
        pred_scores = model.predict(ds_df)
        metrics = compute_all_metrics(true_amis, pred_scores)
        metrics["dataset_id"] = ds_id
        per_dataset.append(metrics)

    return per_dataset


def _evaluate_ivm(ivm_ranker, features_df, test_ids):
    """Evaluate an IVM ranker on test datasets (no training needed)."""
    test_df = features_df[features_df["dataset_id"].isin(test_ids)]
    per_dataset = []
    for ds_id in test_ids:
        ds_df = test_df[test_df["dataset_id"] == ds_id]
        if len(ds_df) < 2:
            continue
        true_amis = ds_df["ami"].values
        pred_scores = ivm_ranker.predict(ds_df)
        metrics = compute_all_metrics(true_amis, pred_scores)
        metrics["dataset_id"] = ds_id
        per_dataset.append(metrics)
    return per_dataset


def run_experiment():
    """Run cross-domain transfer experiment."""
    features_df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    master = pd.read_csv(FEATURES_DIR / "runs_master.csv")
    reg = pd.read_csv(PROJECT_ROOT / "data" / "dataset_registry.csv")

    # Merge needed columns
    merge_cols = ["dataset_id", "run_id"]
    extra_cols = []
    if "hyperparams_json" not in features_df.columns:
        extra_cols.append("hyperparams_json")
    if "ch_adjusted" not in features_df.columns and "ch_adjusted" in master.columns:
        extra_cols.append("ch_adjusted")
    if extra_cols:
        features_df = features_df.merge(
            master[merge_cols + extra_cols], on=merge_cols, how="left"
        )

    # Build source mapping
    ds_source = dict(zip(reg["dataset_id"], reg["source"]))
    available_ds = set(features_df["dataset_id"].unique())

    synthetic_ids = [d for d in available_ds if ds_source.get(d) == "synthetic"]
    openml_ids = [d for d in available_ds if ds_source.get(d) == "openml"]
    image_ids = [d for d in available_ds if ds_source.get(d) == "image"]
    text_ids = [d for d in available_ds if ds_source.get(d) == "text"]
    real_ids = openml_ids + image_ids + text_ids

    logger.info(f"Synthetic: {len(synthetic_ids)}, OpenML: {len(openml_ids)}, "
                f"Image: {len(image_ids)}, Text: {len(text_ids)}")

    # Define transfer scenarios
    scenarios = {
        "synth → real": (synthetic_ids, real_ids),
        "real → synth": (real_ids, synthetic_ids),
        "synth → OpenML": (synthetic_ids, openml_ids),
        "OpenML → synth": (openml_ids, synthetic_ids),
        "synth → image+text": (synthetic_ids, image_ids + text_ids),
        "OpenML → image+text": (openml_ids, image_ids + text_ids),
    }

    # Methods to evaluate
    methods = {
        "XGBoost (ours)": lambda: XGBoostModel(feature_set="partition_x_graph"),
        "Silhouette": lambda: IVMRanker("silhouette"),
        "Calinski-Harabasz": lambda: IVMRanker("calinski_harabasz"),
        "CH Adjusted (Jeon)": lambda: IVMRanker("ch_adjusted"),
    }

    # Also add the "standard" in-distribution baseline using random 60/20/20 splits
    # for reference
    all_rows = []

    for scenario_name, (train_ids, test_ids) in scenarios.items():
        if len(test_ids) < 2:
            logger.warning(f"Skipping {scenario_name}: not enough test datasets")
            continue

        logger.info(f"\n=== {scenario_name} (train={len(train_ids)}, test={len(test_ids)}) ===")

        for method_name, method_factory in methods.items():
            method = method_factory()

            if isinstance(method, IVMRanker):
                per_ds = _evaluate_ivm(method, features_df, test_ids)
            else:
                per_ds = _evaluate_split(method, features_df, train_ids, test_ids)

            if not per_ds:
                continue

            agg = aggregate_metrics(per_ds)
            row = {
                "scenario": scenario_name,
                "method": method_name,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
            }
            for metric, stats in agg.items():
                if metric in ("dataset_id", "seed"):
                    continue
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]

            all_rows.append(row)
            logger.info(f"  {method_name}: regret={row.get('regret_mean', 'N/A'):.4f}, "
                        f"spearman={row.get('spearman_mean', 'N/A'):.4f}")

    results_df = pd.DataFrame(all_rows)

    # Save
    agg_dir = RESULTS_DIR / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(agg_dir / "exp6_cross_domain.csv", index=False)
    logger.info(f"\nSaved to {agg_dir / 'exp6_cross_domain.csv'}")

    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_experiment()

    print("\n=== Experiment 6: Cross-Domain Transfer ===")
    # Pretty print as pivot table
    if len(results) > 0:
        for scenario in results["scenario"].unique():
            print(f"\n--- {scenario} ---")
            sub = results[results["scenario"] == scenario]
            for _, row in sub.iterrows():
                print(f"  {row['method']:25s}: regret={row['regret_mean']:.4f} ± {row['regret_std']:.4f}, "
                      f"ρ={row['spearman_mean']:.4f} ± {row['spearman_std']:.4f}")
