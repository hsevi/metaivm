"""
Experiment 3 — Leave-one-algorithm-family-out.

For each algorithm family, remove ALL runs of that family from training,
then evaluate ONLY on runs from the removed family on held-out datasets.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import compute_all_metrics, aggregate_metrics
from src.evaluation.splits import make_splits
from src.models.tabular_models import XGBoostModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"

ALGO_FAMILIES = [
    "KMeans", "GaussianMixture", "SpectralClustering",
    "DBSCAN", "HDBSCAN", "AgglomerativeClustering",
]


def run_experiment(seeds=None, feature_set="partition_x_graph"):
    """
    Run Experiment 3: leave-one-algorithm-out.

    Returns:
        DataFrame with results per held-out algorithm.
    """
    if seeds is None:
        with open(PROJECT_ROOT / "configs" / "experiment_configs.yaml") as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["splits"]["seeds"]

    features_df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    master = pd.read_csv(FEATURES_DIR / "runs_master.csv")
    if "hyperparams_json" not in features_df.columns:
        features_df = features_df.merge(
            master[["dataset_id", "run_id", "hyperparams_json"]],
            on=["dataset_id", "run_id"],
            how="left",
        )

    dataset_ids = features_df["dataset_id"].unique().tolist()
    all_results = []
    # Track per-seed metrics for per-seed CSV saving
    per_seed_algo_metrics = {seed: [] for seed in seeds}

    for held_out_algo in ALGO_FAMILIES:
        logger.info(f"=== Held-out algorithm: {held_out_algo} ===")

        # Check if this algo exists in the data
        algo_mask = features_df["algo"] == held_out_algo
        if algo_mask.sum() == 0:
            logger.warning(f"No runs for {held_out_algo}, skipping")
            continue

        per_dataset_metrics_with = []
        per_dataset_metrics_without = []

        for seed in seeds:
            seed_metrics_with = []
            seed_metrics_without = []

            splits = make_splits(
                dataset_ids, n_splits=1,
                train_frac=0.6, val_frac=0.2, test_frac=0.2,
                seed=seed,
            )

            for train_ids, val_ids, test_ids in splits:
                # "With algo" — normal training
                train_with = features_df[features_df["dataset_id"].isin(train_ids)]
                val_with = features_df[features_df["dataset_id"].isin(val_ids)]

                model_with = XGBoostModel(feature_set=feature_set)
                model_with.fit(train_with, val_with)

                # "Without algo" — remove held-out algo from training
                train_without = train_with[train_with["algo"] != held_out_algo]
                val_without = val_with[val_with["algo"] != held_out_algo]

                model_without = XGBoostModel(feature_set=feature_set)
                if len(train_without) > 0 and len(val_without) > 0:
                    model_without.fit(train_without, val_without)
                else:
                    continue

                # Evaluate on test datasets, but ONLY on held-out algo runs
                for ds_id in test_ids:
                    ds_all = features_df[features_df["dataset_id"] == ds_id]
                    ds_algo = ds_all[ds_all["algo"] == held_out_algo]

                    if len(ds_algo) < 2:
                        continue

                    true_amis = ds_algo["ami"].values

                    # With algo in training
                    pred_with = model_with.predict(ds_algo)
                    m_with = compute_all_metrics(true_amis, pred_with)
                    m_with["dataset_id"] = ds_id
                    per_dataset_metrics_with.append(m_with)
                    seed_metrics_with.append(m_with)

                    # Without algo in training
                    pred_without = model_without.predict(ds_algo)
                    m_without = compute_all_metrics(true_amis, pred_without)
                    m_without["dataset_id"] = ds_id
                    per_dataset_metrics_without.append(m_without)
                    seed_metrics_without.append(m_without)

            # Store per-seed aggregation
            agg_w = aggregate_metrics(seed_metrics_with) if seed_metrics_with else {}
            agg_wo = aggregate_metrics(seed_metrics_without) if seed_metrics_without else {}
            per_seed_algo_metrics[seed].append({
                "held_out_algo": held_out_algo,
                "regret_with_mean": agg_w.get("regret", {}).get("mean", np.nan),
                "regret_with_std": agg_w.get("regret", {}).get("std", np.nan),
                "regret_without_mean": agg_wo.get("regret", {}).get("mean", np.nan),
                "regret_without_std": agg_wo.get("regret", {}).get("std", np.nan),
                "delta_regret": (
                    agg_wo.get("regret", {}).get("mean", np.nan)
                    - agg_w.get("regret", {}).get("mean", np.nan)
                ),
            })

        # Overall aggregate
        agg_with = aggregate_metrics(per_dataset_metrics_with) if per_dataset_metrics_with else {}
        agg_without = aggregate_metrics(per_dataset_metrics_without) if per_dataset_metrics_without else {}

        row = {"held_out_algo": held_out_algo}
        row["regret_with_mean"] = agg_with.get("regret", {}).get("mean", np.nan)
        row["regret_with_std"] = agg_with.get("regret", {}).get("std", np.nan)
        row["regret_without_mean"] = agg_without.get("regret", {}).get("mean", np.nan)
        row["regret_without_std"] = agg_without.get("regret", {}).get("std", np.nan)
        row["delta_regret"] = row["regret_without_mean"] - row["regret_with_mean"]
        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # Save per-seed results
    for seed in seeds:
        seed_dir = RESULTS_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_rows = per_seed_algo_metrics.get(seed, [])
        if seed_rows:
            pd.DataFrame(seed_rows).to_csv(
                seed_dir / "exp3_leave_one_algo.csv", index=False
            )

    # Save aggregated
    agg_dir = RESULTS_DIR / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(agg_dir / "exp3_leave_one_algo.csv", index=False)

    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_experiment()
    print("\n=== Experiment 3: Leave-One-Algo-Out ===")
    print(results.to_string(index=False))
