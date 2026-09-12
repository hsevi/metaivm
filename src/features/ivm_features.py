"""
IVM-as-features for Neural IVM.

Wraps classical Internal Validity Measures as features.
Primarily reformats scores already computed during clustering runs,
but can also recompute from scratch.
"""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

SILHOUETTE_MAX_SAMPLES = 10000


def compute_ivm_features(X, labels):
    """
    Compute classical IVM scores as features.

    Args:
        X: (n_samples, n_features) array.
        labels: Array of cluster labels (may include -1 for noise).

    Returns:
        Dict of feature name -> value.
    """
    features = {
        "ivm_silhouette": np.nan,
        "ivm_calinski_harabasz": np.nan,
        "ivm_davies_bouldin_neg": np.nan,  # negated so higher = better
    }

    mask = labels >= 0
    X_clean = X[mask]
    labels_clean = labels[mask]

    unique = np.unique(labels_clean)
    if len(unique) < 2 or len(X_clean) < len(unique) + 1:
        return features

    try:
        if X_clean.shape[0] > SILHOUETTE_MAX_SAMPLES:
            rng = np.random.default_rng(0)
            idx = rng.choice(X_clean.shape[0], SILHOUETTE_MAX_SAMPLES, replace=False)
            features["ivm_silhouette"] = float(silhouette_score(X_clean[idx], labels_clean[idx]))
        else:
            features["ivm_silhouette"] = float(silhouette_score(X_clean, labels_clean))
    except Exception:
        pass

    try:
        features["ivm_calinski_harabasz"] = float(calinski_harabasz_score(X_clean, labels_clean))
    except Exception:
        pass

    try:
        features["ivm_davies_bouldin_neg"] = float(-davies_bouldin_score(X_clean, labels_clean))
    except Exception:
        pass

    return features


def ivm_features_from_master_row(row):
    """
    Extract IVM features from a runs_master.csv row (already computed).

    Args:
        row: Dict-like with 'silhouette', 'calinski_harabasz', 'davies_bouldin'.

    Returns:
        Dict of feature name -> value.
    """
    return {
        "ivm_silhouette": row.get("silhouette", np.nan),
        "ivm_calinski_harabasz": row.get("calinski_harabasz", np.nan),
        "ivm_davies_bouldin_neg": -row["davies_bouldin"] if not np.isnan(row.get("davies_bouldin", np.nan)) else np.nan,
    }
