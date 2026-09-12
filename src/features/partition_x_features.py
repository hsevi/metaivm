"""
Partition + X features for Neural IVM.

Features that use both the partition labels and the data matrix X.
"""

import numpy as np
from scipy.spatial.distance import pdist


def compute_partition_x_features(X, labels, max_within_cluster_samples=5000):
    """
    Compute features using both partition labels and data X.

    Args:
        X: (n_samples, n_features) array.
        labels: Array of cluster labels (may include -1 for noise).
        max_within_cluster_samples: Max points per cluster for density computation.

    Returns:
        Dict of feature name -> value.
    """
    features = {}

    # Filter out noise
    mask = labels >= 0
    X_clean = X[mask]
    labels_clean = labels[mask]

    unique_clusters = np.unique(labels_clean)
    n_clusters = len(unique_clusters)

    if n_clusters < 1 or len(X_clean) < 2:
        return _default_features()

    # Compute cluster centroids
    centroids = np.array([X_clean[labels_clean == c].mean(axis=0) for c in unique_clusters])

    # Within-cluster dispersion (variance)
    within_vars = []
    for c in unique_clusters:
        X_c = X_clean[labels_clean == c]
        if len(X_c) > 1:
            within_vars.append(np.mean(np.var(X_c, axis=0)))
        else:
            within_vars.append(0.0)
    within_vars = np.array(within_vars)

    features["within_cluster_var_mean"] = float(np.mean(within_vars))
    features["within_cluster_var_std"] = float(np.std(within_vars))

    # Between-cluster dispersion
    features["between_cluster_var"] = float(np.mean(np.var(centroids, axis=0)))

    # Dispersion ratio
    mean_within = np.mean(within_vars)
    features["dispersion_ratio"] = float(
        features["between_cluster_var"] / (mean_within + 1e-12)
    )

    # Per-cluster density: mean pairwise distance within each cluster
    cluster_densities = []
    for c in unique_clusters:
        X_c = X_clean[labels_clean == c]
        if len(X_c) > 1:
            if len(X_c) > max_within_cluster_samples:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(X_c), max_within_cluster_samples, replace=False)
                X_c = X_c[idx]
            dists = pdist(X_c, metric="euclidean")
            cluster_densities.append(float(np.mean(dists)))
        else:
            cluster_densities.append(0.0)
    cluster_densities = np.array(cluster_densities)

    features["cluster_density_mean"] = float(np.mean(cluster_densities))
    features["cluster_density_std"] = float(np.std(cluster_densities))
    features["cluster_density_max"] = float(np.max(cluster_densities))

    # Centroid separation
    if n_clusters >= 2:
        centroid_dists = pdist(centroids, metric="euclidean")
        features["centroid_sep_min"] = float(np.min(centroid_dists))
        features["centroid_sep_mean"] = float(np.mean(centroid_dists))
        features["centroid_sep_std"] = float(np.std(centroid_dists))
    else:
        features["centroid_sep_min"] = 0.0
        features["centroid_sep_mean"] = 0.0
        features["centroid_sep_std"] = 0.0

    return features


def _default_features():
    """Return default values when features can't be computed."""
    return {
        "within_cluster_var_mean": np.nan,
        "within_cluster_var_std": np.nan,
        "between_cluster_var": np.nan,
        "dispersion_ratio": np.nan,
        "cluster_density_mean": np.nan,
        "cluster_density_std": np.nan,
        "cluster_density_max": np.nan,
        "centroid_sep_min": np.nan,
        "centroid_sep_mean": np.nan,
        "centroid_sep_std": np.nan,
    }
