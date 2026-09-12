"""
Memory-safe clustering for large datasets (>10k samples).
Only runs lightweight algorithms: KMeans, GMM (diag only), DBSCAN.
Skips: AgglomerativeClustering, HDBSCAN, SpectralClustering, GMM(full).

Usage: python scripts/cluster_large_safe.py <dataset_id>
"""
import sys
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

from src.data.clustering_runner import run_single


def main():
    dataset_id = sys.argv[1]
    proc_dir = PROCESSED_DIR / dataset_id

    X = np.load(proc_dir / "X.npy")
    y_true = np.load(proc_dir / "y_true.npy")
    n, d = X.shape
    logger.info(f"{dataset_id}: n={n}, d={d}")

    # Lightweight configs only
    configs = []

    # KMeans
    for k in [2, 3, 5, 7, 10, 15, 20]:
        for n_init in [10]:
            configs.append(("KMeans", {"n_clusters": k, "n_init": n_init}))

    # GMM diag only (full covariance OOMs on high-d)
    for k in [2, 3, 5, 7, 10, 15, 20]:
        configs.append(("GaussianMixture", {"n_components": k, "covariance_type": "diag"}))

    # DBSCAN — lightweight, no distance matrix
    for eps in [0.3, 0.5, 1.0, 2.0]:
        for ms in [3, 5, 10]:
            configs.append(("DBSCAN", {"eps": eps, "min_samples": ms}))

    results = []
    for algo, params in configs:
        k_param = params.get("n_clusters") or params.get("n_components")
        if k_param and k_param >= n:
            continue
        try:
            result = run_single(dataset_id, algo, params, X, y_true)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed {algo}/{params}: {e}")
        gc.collect()

    if not results:
        logger.warning(f"No results for {dataset_id}")
        sys.exit(0)

    # Append to master
    master_path = FEATURES_DIR / "runs_master.csv"
    new_df = pd.DataFrame(results)

    if master_path.exists():
        existing = pd.read_csv(master_path)
        existing = existing[existing["dataset_id"] != dataset_id]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(master_path, index=False)
    logger.info(f"Saved {len(results)} runs for {dataset_id}. Total: {len(combined)}")


if __name__ == "__main__":
    main()
