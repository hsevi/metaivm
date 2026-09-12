"""
Evaluation harness for Neural IVM.

Runs a method through train/val/test splits and computes all metrics.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics, aggregate_metrics
from src.evaluation.splits import make_splits

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_method(method, features_df, split_seeds, n_splits=5,
                    train_frac=0.6, val_frac=0.2, test_frac=0.2,
                    eps=0.01, k=5):
    """
    Evaluate a method across multiple dataset-level splits.

    Args:
        method: Object with fit(train_df, val_df) and predict(test_df) methods.
                predict should return an array of scores (higher = better).
        features_df: DataFrame with all features (from all_features.csv).
                     Must have columns: dataset_id, run_id, ami, and feature columns.
        split_seeds: List of random seeds for splits.
        n_splits: Number of splits per seed.
        train_frac, val_frac, test_frac: Split fractions.
        eps: Epsilon for eps_success metric.
        k: k for NDCG@k.

    Returns:
        Dict with:
            "per_seed": list of per-seed aggregated metrics
            "overall": aggregated metrics across all seeds
            "per_dataset": list of all per-dataset metric dicts
    """
    dataset_ids = features_df["dataset_id"].unique().tolist()
    all_per_dataset = []

    for seed in split_seeds:
        splits = make_splits(
            dataset_ids, n_splits=1,
            train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
            seed=seed,
        )

        for train_ids, val_ids, test_ids in splits:
            train_df = features_df[features_df["dataset_id"].isin(train_ids)]
            val_df = features_df[features_df["dataset_id"].isin(val_ids)]
            test_df = features_df[features_df["dataset_id"].isin(test_ids)]

            # Train
            method.fit(train_df, val_df)

            # Evaluate on each test dataset
            for ds_id in test_ids:
                ds_df = test_df[test_df["dataset_id"] == ds_id]
                if len(ds_df) < 2:
                    continue

                true_amis = ds_df["ami"].values
                predicted_scores = method.predict(ds_df)

                metrics = compute_all_metrics(true_amis, predicted_scores, eps=eps, k=k)
                metrics["dataset_id"] = ds_id
                metrics["seed"] = seed
                all_per_dataset.append(metrics)

    # Aggregate per seed
    per_seed_results = []
    for seed in split_seeds:
        seed_metrics = [m for m in all_per_dataset if m["seed"] == seed]
        if seed_metrics:
            per_seed_results.append(aggregate_metrics(seed_metrics))

    # Overall aggregation
    overall = aggregate_metrics(all_per_dataset) if all_per_dataset else {}

    return {
        "per_seed": per_seed_results,
        "overall": overall,
        "per_dataset": all_per_dataset,
    }


def evaluate_all_methods(methods_dict, features_df, split_seeds,
                         n_splits=5, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    """
    Evaluate multiple methods and return a comparison table.

    Args:
        methods_dict: Dict of method_name -> method object.
        features_df: DataFrame with all features.
        split_seeds: List of seeds.

    Returns:
        DataFrame with one row per method, columns for each metric (mean +/- std).
    """
    rows = []

    for name, method in methods_dict.items():
        logger.info(f"Evaluating: {name}")
        result = evaluate_method(
            method, features_df, split_seeds,
            n_splits=n_splits,
            train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        )

        row = {"method": name}
        for metric, stats in result["overall"].items():
            if metric in ("dataset_id", "seed"):
                continue
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)

    return pd.DataFrame(rows)
