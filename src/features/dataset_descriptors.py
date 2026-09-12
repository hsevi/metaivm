"""
Dataset descriptor features for Neural IVM.

Computed from X only — same for all clustering runs on a given dataset.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist


def compute_dataset_descriptors(X, max_pairwise_samples=2000):
    """
    Compute dataset-level descriptors from the data matrix X.

    Args:
        X: (n_samples, n_features) array.
        max_pairwise_samples: Max samples for pairwise distance stats.

    Returns:
        Dict of feature name -> value.
    """
    n, d = X.shape
    features = {}

    # Basic shape
    features["n_samples"] = n
    features["n_features"] = d
    features["log_n"] = float(np.log(n + 1))
    features["log_d"] = float(np.log(d + 1))
    features["d_over_n"] = float(d / n)

    # Per-feature statistics (aggregate across features)
    col_means = np.nanmean(X, axis=0)
    col_stds = np.nanstd(X, axis=0)

    features["mean_of_means"] = float(np.mean(col_means))
    features["mean_of_stds"] = float(np.mean(col_stds))

    # Skewness and kurtosis
    with np.errstate(invalid="ignore"):
        col_skew = stats.skew(X, axis=0, nan_policy="omit")
        col_kurt = stats.kurtosis(X, axis=0, nan_policy="omit")
    features["mean_of_skewness"] = float(np.nanmean(col_skew))
    features["mean_of_kurtosis"] = float(np.nanmean(col_kurt))

    # Sparsity
    features["sparsity"] = float(np.mean(np.abs(X) < 1e-10))

    # Intrinsic dimensionality estimate (PCA-based: how many PCs for 95% variance)
    try:
        if min(n, d) > 1:
            centered = X - X.mean(axis=0)
            # Use SVD on centered data for proper eigenvalue decomposition
            # For efficiency, use covariance matrix when d < n, else use Gram matrix
            if d <= n:
                cov = centered.T @ centered / (n - 1)
                eigvals = np.linalg.eigvalsh(cov)
            else:
                gram = centered @ centered.T / (n - 1)
                eigvals = np.linalg.eigvalsh(gram)
            eigvals = np.sort(eigvals)[::-1]
            eigvals = eigvals[eigvals > 0]  # drop numerical negatives
            total_var = eigvals.sum()
            if total_var > 1e-12:
                cumvar = np.cumsum(eigvals) / total_var
                intrinsic_dim = int(np.searchsorted(cumvar, 0.95) + 1)
                features["intrinsic_dim_pca95"] = min(intrinsic_dim, min(n, d))
            else:
                features["intrinsic_dim_pca95"] = 1
        else:
            features["intrinsic_dim_pca95"] = d
    except Exception:
        features["intrinsic_dim_pca95"] = d

    # Pairwise distance statistics (subsample if needed)
    try:
        if n > max_pairwise_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, max_pairwise_samples, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        dists = pdist(X_sub, metric="euclidean")
        features["pairwise_dist_mean"] = float(np.mean(dists))
        features["pairwise_dist_std"] = float(np.std(dists))
        features["pairwise_dist_median"] = float(np.median(dists))
    except Exception:
        features["pairwise_dist_mean"] = np.nan
        features["pairwise_dist_std"] = np.nan
        features["pairwise_dist_median"] = np.nan

    return features
