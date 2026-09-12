"""
Partition + graph features for Neural IVM.

Features using partition labels and the kNN adjacency matrix A.
"""

import numpy as np
from scipy import sparse


def compute_partition_graph_features(A, labels):
    """
    Compute features from partition labels and kNN adjacency graph.

    Args:
        A: Sparse adjacency matrix (n x n), symmetric.
        labels: Array of cluster labels (may include -1 for noise).

    Returns:
        Dict of feature name -> value.
    """
    features = {}

    n = A.shape[0]
    mask = labels >= 0
    unique_clusters = np.unique(labels[mask])
    n_clusters = len(unique_clusters)

    if n_clusters < 1:
        return _default_features()

    # Ensure CSR format for efficient row slicing
    A_csr = sparse.csr_matrix(A)

    # Total number of edges (undirected, so count each once)
    total_edges = A_csr.nnz // 2  # symmetric

    if total_edges == 0:
        return _default_features()

    # Cut fraction: fraction of edges crossing cluster boundaries
    # For each non-zero entry (i, j), check if labels[i] != labels[j]
    rows, cols = A_csr.nonzero()
    # Only upper triangle to avoid double counting
    upper_mask = rows < cols
    rows_u = rows[upper_mask]
    cols_u = cols[upper_mask]

    cross_edges = np.sum(labels[rows_u] != labels[cols_u])
    features["cut_fraction"] = float(cross_edges / (len(rows_u) + 1e-12))

    # Per-cluster metrics
    ncuts = []
    edge_densities = []
    conductances = []

    degrees = np.array(A_csr.sum(axis=1)).ravel()

    for c in unique_clusters:
        c_mask = labels == c
        c_indices = np.where(c_mask)[0]
        n_c = len(c_indices)

        if n_c < 2:
            edge_densities.append(0.0)
            ncuts.append(0.0)
            conductances.append(1.0)
            continue

        # Edges within this cluster
        A_cluster = A_csr[np.ix_(c_indices, c_indices)]
        internal_edges = A_cluster.nnz // 2

        # Possible edges
        possible_edges = n_c * (n_c - 1) // 2
        edge_densities.append(
            float(internal_edges / (possible_edges + 1e-12))
        )

        # Degree sum for this cluster
        cluster_degree_sum = degrees[c_indices].sum()

        # Cut edges for this cluster. degrees[c_indices].sum() counts each internal
        # edge twice and each boundary (cut) edge once, so cut = degree_sum - 2*internal.
        cut_c = int(cluster_degree_sum - 2 * internal_edges)

        # Normalized cut contribution
        ncut_c = cut_c / (cluster_degree_sum + 1e-12)
        ncuts.append(float(ncut_c))

        # Conductance
        vol_c = cluster_degree_sum
        vol_comp = degrees.sum() - vol_c
        min_vol = min(vol_c, vol_comp)
        conductances.append(float(cut_c / (min_vol + 1e-12)))

    features["normalized_cut_proxy"] = float(np.sum(ncuts))
    features["within_cluster_edge_density"] = float(np.mean(edge_densities))
    features["within_edge_density_std"] = float(np.std(edge_densities))
    features["conductance_min"] = float(np.min(conductances))
    features["conductance_mean"] = float(np.mean(conductances))

    return features


def _default_features():
    """Return default values when features can't be computed."""
    return {
        "cut_fraction": np.nan,
        "normalized_cut_proxy": np.nan,
        "within_cluster_edge_density": np.nan,
        "within_edge_density_std": np.nan,
        "conductance_min": np.nan,
        "conductance_mean": np.nan,
    }
