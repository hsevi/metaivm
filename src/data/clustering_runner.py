"""
Clustering runner for Neural IVM.

Runs all algorithm configurations on all datasets, computes AMI and IVM scores,
and saves results to a master table.
"""

import hashlib
import json
import logging
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.cluster import (
    KMeans,
    SpectralClustering,
    DBSCAN,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = DATA_DIR / "clustering_runs"
FEATURES_DIR = DATA_DIR / "features"
CONFIGS_DIR = PROJECT_ROOT / "configs"

SILHOUETTE_MAX_SAMPLES = 10000


def _params_hash(algo, params):
    """Deterministic hash of algorithm + hyperparams for run ID."""
    key = json.dumps({"algo": algo, "params": params}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _expand_configs(config_dict):
    """
    Expand a config dict (algo -> {param: [values]}) into a list of
    (algo, params_dict) tuples representing the full grid.
    """
    configs = []
    for algo, param_grid in config_dict.items():
        keys = sorted(param_grid.keys())
        values = [param_grid[k] for k in keys]
        for combo in product(*values):
            params = dict(zip(keys, combo))
            configs.append((algo, params))
    return configs


def _make_clusterer(algo, params):
    """Instantiate a clustering algorithm from name and params."""
    if algo == "KMeans":
        return KMeans(
            n_clusters=params["n_clusters"],
            n_init=params["n_init"],
            random_state=0,
        )
    elif algo == "GaussianMixture":
        return GaussianMixture(
            n_components=params["n_components"],
            covariance_type=params["covariance_type"],
            random_state=0,
            max_iter=300,
        )
    elif algo == "SpectralClustering":
        return SpectralClustering(
            n_clusters=params["n_clusters"],
            affinity=params["affinity"],
            random_state=0,
            n_init=10,
        )
    elif algo == "DBSCAN":
        return DBSCAN(
            eps=params["eps"],
            min_samples=params["min_samples"],
        )
    elif algo == "HDBSCAN":
        import hdbscan

        return hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
        )
    elif algo == "AgglomerativeClustering":
        return AgglomerativeClustering(
            n_clusters=params["n_clusters"],
            linkage=params["linkage"],
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def _compute_ivm_scores(X, labels):
    """
    Compute classical IVM scores. Returns dict with silhouette, CH, DB.
    Returns NaN for scores that can't be computed.
    """
    scores = {
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }

    # Need at least 2 clusters and not all points in one cluster
    unique_labels = np.unique(labels)
    # Exclude noise label (-1) for counting
    cluster_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(cluster_labels)

    if n_clusters < 2:
        return scores

    # For IVM computation, only use non-noise points
    mask = labels >= 0
    if mask.sum() < n_clusters + 1:
        return scores

    X_clean = X[mask]
    labels_clean = labels[mask]

    try:
        # Subsample for silhouette if large
        if X_clean.shape[0] > SILHOUETTE_MAX_SAMPLES:
            rng = np.random.default_rng(0)
            idx = rng.choice(X_clean.shape[0], SILHOUETTE_MAX_SAMPLES, replace=False)
            scores["silhouette"] = silhouette_score(X_clean[idx], labels_clean[idx])
        else:
            scores["silhouette"] = silhouette_score(X_clean, labels_clean)
    except Exception:
        pass

    try:
        scores["calinski_harabasz"] = calinski_harabasz_score(X_clean, labels_clean)
    except Exception:
        pass

    try:
        scores["davies_bouldin"] = davies_bouldin_score(X_clean, labels_clean)
    except Exception:
        pass

    return scores


def run_single(dataset_id, algo, params, X, y_true):
    """
    Run a single clustering configuration on a dataset.

    Returns a dict with results, or None on failure.
    """
    run_id = _params_hash(algo, params)
    params_json = json.dumps(params, sort_keys=True)

    try:
        clusterer = _make_clusterer(algo, params)

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if algo == "GaussianMixture":
                clusterer.fit(X)
                labels = clusterer.predict(X)
            else:
                labels = clusterer.fit_predict(X)
        runtime = time.time() - t0

        # Compute metrics
        ami = adjusted_mutual_info_score(y_true, labels)
        ivm_scores = _compute_ivm_scores(X, labels)

        # Count clusters
        unique = np.unique(labels)
        n_clusters_found = len(unique[unique >= 0])
        noise_fraction = (labels == -1).mean() if -1 in labels else 0.0

        # Save labels
        ds_runs_dir = RUNS_DIR / dataset_id
        ds_runs_dir.mkdir(parents=True, exist_ok=True)
        np.save(ds_runs_dir / f"{algo}_{run_id}.npy", labels)

        return {
            "dataset_id": dataset_id,
            "algo": algo,
            "hyperparams_json": params_json,
            "run_id": run_id,
            "ami": ami,
            "silhouette": ivm_scores["silhouette"],
            "calinski_harabasz": ivm_scores["calinski_harabasz"],
            "davies_bouldin": ivm_scores["davies_bouldin"],
            "n_clusters_found": n_clusters_found,
            "noise_fraction": noise_fraction,
            "runtime_sec": runtime,
        }

    except Exception as e:
        logger.warning(f"Failed: {dataset_id} / {algo} / {params_json}: {e}")
        return None


def run_dataset(dataset_id, configs):
    """Run all clustering configs on a single dataset."""
    proc_dir = PROCESSED_DIR / dataset_id

    try:
        X = np.load(proc_dir / "X.npy")
        y_true = np.load(proc_dir / "y_true.npy")
    except Exception as e:
        logger.error(f"Cannot load processed data for {dataset_id}: {e}")
        return []

    n = X.shape[0]
    results = []

    for algo, params in configs:
        # Skip SpectralClustering entirely — O(n³), not scalable
        if algo == "SpectralClustering":
            logger.debug(f"Skipping SpectralClustering on {dataset_id} (n={n})")
            continue

        # Skip AgglomerativeClustering for large datasets — O(n²) memory
        if algo == "AgglomerativeClustering" and n > 12000:
            logger.debug(f"Skipping AgglomerativeClustering on {dataset_id} (n={n})")
            continue

        # Skip HDBSCAN for very large + high-dim datasets — memory heavy
        if algo == "HDBSCAN" and n > 30000:
            logger.debug(f"Skipping HDBSCAN on {dataset_id} (n={n})")
            continue

        # Skip configs requesting more clusters than samples
        k_param = params.get("n_clusters") or params.get("n_components")
        if k_param and k_param >= n:
            continue

        result = run_single(dataset_id, algo, params, X, y_true)
        if result is not None:
            results.append(result)

    logger.info(f"Completed {len(results)} runs for {dataset_id}")
    return results


def run_all(dataset_ids=None, registry_path=None):
    """
    Run all clustering configurations on all datasets sequentially.

    Processes one dataset at a time with explicit memory cleanup to avoid OOM.

    Args:
        dataset_ids: List of dataset IDs. If None, reads from registry.
        registry_path: Path to registry CSV. If None, uses default.

    Returns:
        DataFrame with master results table.
    """
    import gc

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load configs
    with open(CONFIGS_DIR / "clustering_configs.yaml") as f:
        config_dict = yaml.safe_load(f)
    configs = _expand_configs(config_dict)
    logger.info(f"Expanded to {len(configs)} algorithm configurations")

    # Get dataset IDs
    if dataset_ids is None:
        reg_path = registry_path or DATA_DIR / "dataset_registry_selected.csv"
        if not reg_path.exists():
            reg_path = DATA_DIR / "dataset_registry.csv"
        registry = pd.read_csv(reg_path)
        dataset_ids = registry["dataset_id"].tolist()

    # Filter to only preprocessed datasets
    dataset_ids = [
        ds_id for ds_id in dataset_ids if (PROCESSED_DIR / ds_id / "X.npy").exists()
    ]
    logger.info(f"Running on {len(dataset_ids)} preprocessed datasets")

    # Check for existing master table to resume from
    master_path = FEATURES_DIR / "runs_master.csv"
    if master_path.exists():
        existing = pd.read_csv(master_path)
        done_ids = set(existing["dataset_id"].unique())
        remaining = [d for d in dataset_ids if d not in done_ids]
        logger.info(f"Resuming: {len(done_ids)} datasets already done, {len(remaining)} remaining")
        all_rows = existing.to_dict("records")
        dataset_ids = remaining
    else:
        all_rows = []

    # Run sequentially with GC between datasets
    for i, ds_id in enumerate(dataset_ids):
        logger.info(f"[{i+1}/{len(dataset_ids)}] Processing {ds_id}...")
        results = run_dataset(ds_id, configs)
        all_rows.extend(results)

        # Save incrementally every 5 datasets
        if (i + 1) % 5 == 0 or i == len(dataset_ids) - 1:
            master = pd.DataFrame(all_rows)
            master.to_csv(master_path, index=False)
            logger.info(f"  Checkpoint: saved {len(master)} total runs")

        # Explicit memory cleanup
        gc.collect()

    master = pd.DataFrame(all_rows)
    master.to_csv(master_path, index=False)
    logger.info(f"Saved {len(master)} runs to {master_path}")

    return master


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    master = run_all()
    print(f"\nTotal runs: {len(master)}")
    print(f"Datasets: {master['dataset_id'].nunique()}")
    print(f"Mean AMI: {master['ami'].mean():.3f}")
    print(f"\nRuns per algorithm:")
    print(master.groupby("algo").size())
