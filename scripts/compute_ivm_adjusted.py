"""
Compute adjusted IVM scores (Jeon et al. TPAMI 2025) for all clustering runs.
Adds CH_A column to runs_master.csv.

Uses btwim library: pip install btwim
"""
import sys
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from btwim import calinski_harabasz as ch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = DATA_DIR / "clustering_runs"
FEATURES_DIR = DATA_DIR / "features"

ITER_NUM = 20  # Monte Carlo iterations for adjustment


def main():
    master_path = FEATURES_DIR / "runs_master.csv"
    master = pd.read_csv(master_path)
    logger.info(f"Loaded {len(master)} runs")

    # Check if CH_A already computed
    if "ch_adjusted" in master.columns:
        n_missing = master["ch_adjusted"].isna().sum()
        if n_missing == 0:
            logger.info("CH_A already computed for all runs")
            return
        logger.info(f"CH_A missing for {n_missing} runs, computing...")
    else:
        master["ch_adjusted"] = np.nan

    # Cache loaded data
    X_cache = {}
    dataset_ids = master["dataset_id"].unique()

    n_total = len(master)
    n_computed = 0
    n_failed = 0

    for idx, row in master.iterrows():
        if not np.isnan(master.at[idx, "ch_adjusted"]):
            continue

        ds_id = row["dataset_id"]
        algo = row["algo"]
        run_id = row["run_id"]

        # Load X (cached)
        if ds_id not in X_cache:
            proc_dir = PROCESSED_DIR / ds_id
            try:
                X_cache[ds_id] = np.load(proc_dir / "X.npy")
            except Exception as e:
                logger.warning(f"Cannot load X for {ds_id}: {e}")
                X_cache[ds_id] = None

        X = X_cache[ds_id]
        if X is None:
            n_failed += 1
            continue

        # Load clustering labels
        runs_dir = RUNS_DIR / ds_id
        label_path = runs_dir / f"{algo}_{run_id}.npy"
        try:
            labels = np.load(label_path)
        except Exception as e:
            logger.warning(f"Cannot load labels for {ds_id}/{algo}_{run_id}: {e}")
            n_failed += 1
            continue

        # Need at least 2 clusters
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) < 2:
            n_failed += 1
            continue

        # Remove noise points (label=-1) for the computation
        mask = labels >= 0
        X_clean = X[mask]
        labels_clean = labels[mask]

        # Relabel to 0..k-1
        _, labels_clean = np.unique(labels_clean, return_inverse=True)

        try:
            ch_a = ch.btw(X_clean, labels_clean, iter_num=ITER_NUM)
            master.at[idx, "ch_adjusted"] = ch_a
        except Exception as e:
            n_failed += 1

        n_computed += 1
        if n_computed % 500 == 0:
            logger.info(f"  Computed {n_computed} CH_A scores...")
            # Save checkpoint
            master.to_csv(master_path, index=False)

        # Free memory for large datasets
        if n_computed % 1000 == 0:
            gc.collect()

    master.to_csv(master_path, index=False)
    logger.info(f"Done: {n_computed} computed, {n_failed} failed")
    logger.info(f"CH_A stats: mean={master['ch_adjusted'].mean():.4f}, "
                f"std={master['ch_adjusted'].std():.4f}, "
                f"NaN={master['ch_adjusted'].isna().sum()}")


if __name__ == "__main__":
    main()
