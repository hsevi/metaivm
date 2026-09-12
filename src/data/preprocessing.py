"""
Preprocessing pipeline for Neural IVM.

For each dataset: impute missing values, standardize features,
compute kNN graphs, and save processed arrays.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

KNN_K_VALUES = [10, 15, 20]


def preprocess_dataset(dataset_id, max_missing_frac=0.5):
    """
    Preprocess a single dataset: impute, standardize, compute kNN graphs.

    Args:
        dataset_id: String identifier for the dataset.
        max_missing_frac: Drop columns with more than this fraction missing.

    Returns:
        True if successful, False otherwise.
    """
    raw_dir = RAW_DIR / dataset_id
    out_dir = PROCESSED_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        X = np.load(raw_dir / "X.npy", allow_pickle=True).astype(np.float64)
        y = np.load(raw_dir / "y_true.npy", allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load {dataset_id}: {e}")
        return False

    n_original, d_original = X.shape

    # Drop columns with too many missing values
    missing_frac = np.isnan(X).mean(axis=0)
    keep_cols = missing_frac <= max_missing_frac
    if keep_cols.sum() < X.shape[1]:
        logger.info(
            f"{dataset_id}: dropping {X.shape[1] - keep_cols.sum()} columns "
            f"with >{max_missing_frac*100:.0f}% missing"
        )
    X = X[:, keep_cols]

    if X.shape[1] == 0:
        logger.error(f"{dataset_id}: all columns dropped, skipping")
        return False

    # Impute remaining missing values with median
    if np.isnan(X).any():
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

    # Drop rows that are still problematic (all same value etc.)
    # Remove constant columns
    col_std = np.std(X, axis=0)
    non_const = col_std > 1e-10
    if non_const.sum() < X.shape[1]:
        logger.info(
            f"{dataset_id}: dropping {X.shape[1] - non_const.sum()} constant columns"
        )
    X = X[:, non_const]

    if X.shape[1] == 0:
        logger.error(f"{dataset_id}: all columns constant, skipping")
        return False

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save processed data
    np.save(out_dir / "X.npy", X.astype(np.float32))
    np.save(out_dir / "y_true.npy", y)

    # Compute kNN graphs
    for k in KNN_K_VALUES:
        if k >= X.shape[0]:
            logger.warning(
                f"{dataset_id}: k={k} >= n={X.shape[0]}, skipping kNN graph"
            )
            continue

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Build symmetric adjacency matrix
        n = X.shape[0]
        rows = np.repeat(np.arange(n), k)
        cols = indices.ravel()
        data = np.ones(len(rows), dtype=np.float32)
        A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        # Symmetrize
        A = ((A + A.T) > 0).astype(np.float32)

        sparse.save_npz(out_dir / f"A_knn_k{k}.npz", A)

    logger.info(
        f"Preprocessed {dataset_id}: "
        f"{n_original}x{d_original} -> {X.shape[0]}x{X.shape[1]}"
    )
    return True


def preprocess_all(dataset_ids=None):
    """
    Preprocess all datasets (or a specified subset).

    Args:
        dataset_ids: List of dataset IDs. If None, reads from registry.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_ids is None:
        registry = pd.read_csv(DATA_DIR / "dataset_registry.csv")
        dataset_ids = registry["dataset_id"].tolist()

    results = {}
    for ds_id in dataset_ids:
        success = preprocess_dataset(ds_id)
        results[ds_id] = success

    n_success = sum(results.values())
    n_total = len(results)
    logger.info(f"Preprocessed {n_success}/{n_total} datasets successfully")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = preprocess_all()
    n_ok = sum(results.values())
    print(f"\nPreprocessed {n_ok}/{len(results)} datasets")
