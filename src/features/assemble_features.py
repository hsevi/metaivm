"""
Feature assembly for Neural IVM.

Combines all feature modules into a single feature table:
data/features/all_features.csv — one row per (dataset_id, run_id).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.features.partition_features import compute_partition_features
from src.features.dataset_descriptors import compute_dataset_descriptors
from src.features.partition_x_features import compute_partition_x_features
from src.features.partition_graph_features import compute_partition_graph_features
from src.features.ivm_features import ivm_features_from_master_row

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = DATA_DIR / "clustering_runs"
FEATURES_DIR = DATA_DIR / "features"

# Algorithm families for one-hot encoding
ALGO_FAMILIES = [
    "KMeans",
    "GaussianMixture",
    "SpectralClustering",
    "DBSCAN",
    "HDBSCAN",
    "AgglomerativeClustering",
]

# Key hyperparameters to encode numerically
NUMERIC_HYPERPARAMS = ["n_clusters", "n_components", "eps", "min_samples", "min_cluster_size"]


def _encode_algo_hyperparams(algo, hyperparams_json):
    """One-hot encode algorithm + numeric hyperparams."""
    features = {}

    # One-hot for algorithm family
    for fam in ALGO_FAMILIES:
        features[f"algo_{fam}"] = 1.0 if algo == fam else 0.0

    # Numeric hyperparams (0 if not applicable)
    if isinstance(hyperparams_json, str):
        try:
            params = json.loads(hyperparams_json)
        except (json.JSONDecodeError, TypeError):
            params = {}
    elif isinstance(hyperparams_json, dict):
        params = hyperparams_json
    else:
        params = {}
    for hp in NUMERIC_HYPERPARAMS:
        features[f"hp_{hp}"] = float(params.get(hp, 0))

    return features


def assemble_features(knn_k=10):
    """
    Assemble all features into a single table.

    Args:
        knn_k: Which kNN graph to use for graph features.

    Returns:
        DataFrame with all features + target AMI.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load master table
    master_path = FEATURES_DIR / "runs_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(
            f"runs_master.csv not found at {master_path}. "
            "Run clustering_runner.py first."
        )
    master = pd.read_csv(master_path)
    logger.info(f"Loaded {len(master)} runs from master table")

    # Cache dataset descriptors (same for all runs on a dataset)
    dataset_desc_cache = {}
    dataset_ids = master["dataset_id"].unique()

    for ds_id in dataset_ids:
        proc_dir = PROCESSED_DIR / ds_id
        try:
            X = np.load(proc_dir / "X.npy")
            dataset_desc_cache[ds_id] = compute_dataset_descriptors(X)
        except Exception as e:
            logger.warning(f"Cannot compute descriptors for {ds_id}: {e}")
            dataset_desc_cache[ds_id] = {}

    logger.info(f"Computed dataset descriptors for {len(dataset_desc_cache)} datasets")

    # Process each run
    all_rows = []

    for idx, row in master.iterrows():
        ds_id = row["dataset_id"]
        algo = row["algo"]
        run_id = row["run_id"]

        # Load data
        proc_dir = PROCESSED_DIR / ds_id
        runs_dir = RUNS_DIR / ds_id

        try:
            X = np.load(proc_dir / "X.npy")
            labels = np.load(runs_dir / f"{algo}_{run_id}.npy")
        except Exception as e:
            logger.warning(f"Cannot load data for {ds_id}/{algo}_{run_id}: {e}")
            continue

        # 1. Partition features
        part_feats = compute_partition_features(labels)

        # 2. Dataset descriptors (cached)
        ds_feats = dataset_desc_cache.get(ds_id, {})

        # 3. Partition + X features
        px_feats = compute_partition_x_features(X, labels)

        # 4. Partition + graph features
        graph_feats = {}
        graph_path = proc_dir / f"A_knn_k{knn_k}.npz"
        if graph_path.exists():
            try:
                A = sparse.load_npz(graph_path)
                graph_feats = compute_partition_graph_features(A, labels)
            except Exception as e:
                logger.warning(f"Graph features failed for {ds_id}/{run_id}: {e}")

        # 5. IVM features (from master row)
        ivm_feats = ivm_features_from_master_row(row)

        # 6. Algorithm/hyperparam encoding
        algo_feats = _encode_algo_hyperparams(algo, row["hyperparams_json"])

        # Combine all
        combined = {
            "dataset_id": ds_id,
            "run_id": run_id,
            "algo": algo,
            "ami": row["ami"],
        }
        combined.update(part_feats)
        combined.update({f"ds_{k}": v for k, v in ds_feats.items()})
        combined.update(px_feats)
        combined.update({f"graph_{k}": v for k, v in graph_feats.items()})
        combined.update(ivm_feats)
        combined.update(algo_feats)

        all_rows.append(combined)

        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1}/{len(master)} runs")

    df = pd.DataFrame(all_rows)

    # Save
    out_path = FEATURES_DIR / "all_features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows x {len(df.columns)} columns to {out_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = assemble_features()
    print(f"\nFeature table: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nNaN fraction per column:")
    print(df.isnull().mean().sort_values(ascending=False).head(10))
