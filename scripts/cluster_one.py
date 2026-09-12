"""
Cluster a single dataset in its own process.
Memory-safe: skips heavy algorithms for large datasets.

Usage:
    python scripts/cluster_one.py <dataset_id>
"""
import sys
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clustering_runner import run_single, _expand_configs

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def main():
    dataset_id = sys.argv[1]
    proc_dir = PROCESSED_DIR / dataset_id

    if not (proc_dir / "X.npy").exists():
        logger.error(f"{dataset_id} not preprocessed, skipping")
        sys.exit(0)

    X = np.load(proc_dir / "X.npy")
    y_true = np.load(proc_dir / "y_true.npy")
    n, d = X.shape
    logger.info(f"Loaded {dataset_id}: n={n}, d={d}")

    with open(CONFIGS_DIR / "clustering_configs.yaml") as f:
        config_dict = yaml.safe_load(f)
    configs = _expand_configs(config_dict)

    results = []
    for algo, params in configs:
        # Skip SpectralClustering — O(n³)
        if algo == "SpectralClustering":
            continue
        # Skip AgglomerativeClustering for n > 10k — O(n²) memory
        if algo == "AgglomerativeClustering" and n > 10000:
            continue
        # Skip HDBSCAN for n > 25k
        if algo == "HDBSCAN" and n > 25000:
            continue
        # Skip k > n
        k_param = params.get("n_clusters") or params.get("n_components")
        if k_param and k_param >= n:
            continue

        result = run_single(dataset_id, algo, params, X, y_true)
        if result is not None:
            results.append(result)
        gc.collect()

    if not results:
        logger.warning(f"No results for {dataset_id}")
        sys.exit(0)

    # Append to master
    master_path = FEATURES_DIR / "runs_master.csv"
    new_df = pd.DataFrame(results)

    if master_path.exists():
        existing = pd.read_csv(master_path)
        # Remove any existing rows for this dataset (in case of partial run)
        existing = existing[existing["dataset_id"] != dataset_id]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(master_path, index=False)
    logger.info(f"Saved {len(results)} runs for {dataset_id}. Total: {len(combined)}")


if __name__ == "__main__":
    main()
