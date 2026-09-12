"""
Experiment: GNN meta-evaluator evaluation.
Runs GNN through the same 5-seed dataset-level splits as exp1.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import make_splits
from src.evaluation.metrics import compute_all_metrics, aggregate_metrics
from src.models.gnn_model import GNNSurrogate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"


def run_gnn_experiment(n_seeds=5, max_nodes=5000):
    """Run GNN through 5-seed evaluation, skipping datasets with >max_nodes points."""
    master = pd.read_csv(PROJECT_ROOT / "data" / "features" / "runs_master.csv")
    registry = pd.read_csv(PROJECT_ROOT / "data" / "dataset_registry.csv")

    # Filter out large datasets to avoid memory issues
    large_ds = registry[registry["n_samples"] > max_nodes]["dataset_id"].tolist()
    if large_ds:
        logger.info(f"Skipping {len(large_ds)} datasets with >{max_nodes} samples: {large_ds}")
        master = master[~master["dataset_id"].isin(large_ds)]

    dataset_ids = sorted(master["dataset_id"].unique())
    logger.info(f"Running GNN on {len(dataset_ids)} datasets, {len(master)} runs")

    all_seed_metrics = []

    for seed in range(n_seeds):
        logger.info(f"=== Seed {seed} ===")
        splits = make_splits(dataset_ids, n_splits=1, seed=seed)
        train_ids, val_ids, test_ids = splits[0]

        train_df = master[master["dataset_id"].isin(train_ids)]
        val_df = master[master["dataset_id"].isin(val_ids)]
        test_df = master[master["dataset_id"].isin(test_ids)]

        logger.info(f"Train: {len(train_ids)} datasets ({len(train_df)} runs), "
                     f"Val: {len(val_ids)} datasets ({len(val_df)} runs), "
                     f"Test: {len(test_ids)} datasets ({len(test_df)} runs)")

        # Train GNN
        t0 = time.time()
        gnn = GNNSurrogate(
            hidden_dim=128,
            n_layers=3,
            lr=0.001,
            epochs=100,
            patience=15,
            knn_k=10,
            batch_size=32,
            seed=seed,
        )
        gnn.fit(train_df, val_df)
        train_time = time.time() - t0
        logger.info(f"Training took {train_time:.1f}s")

        # Evaluate per test dataset
        per_dataset_metrics = []
        for ds_id in test_ids:
            ds_df = test_df[test_df["dataset_id"] == ds_id]
            if len(ds_df) < 2:
                continue

            t0 = time.time()
            predictions = gnn.predict(ds_df)
            inf_time = time.time() - t0

            true_amis = ds_df["ami"].values
            metrics = compute_all_metrics(true_amis, predictions)
            metrics["dataset_id"] = ds_id
            metrics["inference_time"] = inf_time
            per_dataset_metrics.append(metrics)

        if per_dataset_metrics:
            seed_agg = aggregate_metrics(per_dataset_metrics)
            # Flatten: {"regret": {"mean":..,"std":..}} -> {"regret_mean":..,"regret_std":..}
            flat = {}
            for k, v in seed_agg.items():
                if isinstance(v, dict):
                    flat[f"{k}_mean"] = v["mean"]
                    flat[f"{k}_std"] = v["std"]
                else:
                    flat[k] = v
            flat["seed"] = seed
            flat["train_time"] = train_time
            all_seed_metrics.append(flat)
            logger.info(f"Seed {seed} — Regret: {flat['regret_mean']:.4f} ± {flat['regret_std']:.4f}, "
                         f"Spearman: {flat['spearman_mean']:.4f}")

            # Save per-seed results
            seed_dir = RESULTS_DIR / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(per_dataset_metrics).to_csv(seed_dir / "exp_gnn_per_dataset.csv", index=False)

    # Aggregate across seeds
    if all_seed_metrics:
        results = {}
        for metric in ["regret_mean", "eps_success_mean", "ndcg_at_k_mean", "spearman_mean", "kendall_mean"]:
            vals = [s[metric] for s in all_seed_metrics if metric in s]
            results[metric] = np.mean(vals)
            results[metric.replace("_mean", "_seed_std")] = np.std(vals)

        results_df = pd.DataFrame([{"method": "GNN", **results}])
        agg_dir = RESULTS_DIR / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(agg_dir / "exp_gnn.csv", index=False)

        logger.info("\n=== GNN RESULTS (mean ± std across seeds) ===")
        logger.info(f"Regret:       {results['regret_mean']:.4f} ± {results.get('regret_mean_seed_std', 0):.4f}")
        logger.info(f"eps_success:  {results['eps_success_mean']:.4f} ± {results.get('eps_success_mean_seed_std', 0):.4f}")
        logger.info(f"NDCG@5:       {results['ndcg_at_k_mean']:.4f} ± {results.get('ndcg_at_k_mean_seed_std', 0):.4f}")
        logger.info(f"Spearman:     {results['spearman_mean']:.4f} ± {results.get('spearman_mean_seed_std', 0):.4f}")

        # Compare with XGBoost
        exp1_path = agg_dir / "exp1_main_table.csv"
        if exp1_path.exists():
            exp1 = pd.read_csv(exp1_path)
            xgb_row = exp1[exp1["method"].str.contains("XGBoost")]
            if len(xgb_row):
                xgb_regret = xgb_row.iloc[0]["regret_mean"]
                logger.info(f"\nXGBoost regret: {xgb_regret:.4f}")
                logger.info(f"GNN regret:     {results['regret_mean']:.4f}")
                if results['regret_mean'] < xgb_regret:
                    logger.info(">>> GNN WINS <<<")
                else:
                    logger.info(f">>> XGBoost still better (delta={results['regret_mean'] - xgb_regret:.4f}) <<<")

    return all_seed_metrics


if __name__ == "__main__":
    run_gnn_experiment()
