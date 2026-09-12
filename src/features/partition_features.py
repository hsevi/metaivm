"""
Partition-only features for Neural IVM.

Computed from predicted cluster labels only (no X, no A).
"""

import numpy as np


def compute_partition_features(labels):
    """
    Compute features from partition labels.

    Args:
        labels: Array of cluster labels (may include -1 for noise).

    Returns:
        Dict of feature name -> value.
    """
    features = {}

    # Separate noise and cluster labels
    noise_mask = labels == -1
    n_total = len(labels)
    n_noise = noise_mask.sum()

    cluster_labels = labels[~noise_mask]
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    features["n_clusters"] = n_clusters
    features["noise_fraction"] = n_noise / n_total if n_total > 0 else 0.0

    if n_clusters == 0:
        # All noise — fill with defaults
        features["cluster_size_min"] = 0
        features["cluster_size_max"] = 0
        features["cluster_size_mean"] = 0.0
        features["cluster_size_std"] = 0.0
        features["cluster_size_median"] = 0.0
        features["size_ratio"] = 0.0
        features["entropy"] = 0.0
        features["gini"] = 0.0
        features["imbalance"] = 1.0
        features["singleton_fraction"] = 0.0
        return features

    # Cluster sizes
    sizes = np.array([np.sum(cluster_labels == c) for c in unique_clusters])

    features["cluster_size_min"] = int(sizes.min())
    features["cluster_size_max"] = int(sizes.max())
    features["cluster_size_mean"] = float(sizes.mean())
    features["cluster_size_std"] = float(sizes.std())
    features["cluster_size_median"] = float(np.median(sizes))

    # Size ratio (cap at large value to avoid inf downstream)
    features["size_ratio"] = (
        float(sizes.max() / sizes.min()) if sizes.min() > 0 else float("nan")
    )

    # Shannon entropy of cluster size distribution
    probs = sizes / sizes.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    features["entropy"] = float(entropy)

    # Gini coefficient
    sorted_sizes = np.sort(sizes)
    n = len(sorted_sizes)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_sizes) - (n + 1) * np.sum(sorted_sizes)) / (
        n * np.sum(sorted_sizes) + 1e-12
    )
    features["gini"] = float(gini)

    # Imbalance
    features["imbalance"] = float(1.0 - sizes.min() / (sizes.max() + 1e-12))

    # Singleton fraction (clusters with exactly 1 point)
    features["singleton_fraction"] = float(np.mean(sizes == 1))

    return features
