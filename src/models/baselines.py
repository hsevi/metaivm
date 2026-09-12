"""
Baseline methods for Neural IVM.

Each baseline implements fit(train_df, val_df) and predict(test_df).
predict returns an array of scores (higher = better predicted quality).
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RandomBaseline:
    """Score each run uniformly at random."""

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, train_df, val_df):
        pass

    def predict(self, test_df):
        rng = np.random.default_rng(self.seed)
        return rng.random(len(test_df))


class GlobalBestConfig:
    """
    On training data, find the single (algo, hyperparams) config with
    highest average AMI. At test time, give that config score=1, all others 0.
    """

    def __init__(self):
        self.best_algo = None
        self.best_params = None

    def fit(self, train_df, val_df):
        # Average AMI per (algo, hyperparams_json) across training datasets
        grouped = train_df.groupby(["algo", "hyperparams_json"])["ami"].mean()
        best_idx = grouped.idxmax()
        self.best_algo, self.best_params = best_idx
        logger.debug(f"GlobalBest: {self.best_algo} {self.best_params} "
                     f"(avg AMI={grouped[best_idx]:.3f})")

    def predict(self, test_df):
        scores = np.zeros(len(test_df))
        mask = (test_df["algo"] == self.best_algo) & \
               (test_df["hyperparams_json"] == self.best_params)
        scores[mask.values] = 1.0
        return scores


class IVMRanker:
    """
    Rank runs by a single IVM score. Higher score = predicted better.
    """

    def __init__(self, ivm_name="silhouette"):
        """
        Args:
            ivm_name: One of "silhouette", "calinski_harabasz", "davies_bouldin".
                      For DB, scores are negated (lower DB = better).
        """
        self.ivm_name = ivm_name
        self.negate = ivm_name == "davies_bouldin"

    def fit(self, train_df, val_df):
        pass

    def predict(self, test_df):
        # Map IVM name to feature column
        col_map = {
            "silhouette": "ivm_silhouette",
            "calinski_harabasz": "ivm_calinski_harabasz",
            "davies_bouldin": "ivm_davies_bouldin_neg",  # already negated
            "ch_adjusted": "ch_adjusted",
        }
        col = col_map.get(self.ivm_name, self.ivm_name)

        if col in test_df.columns:
            scores = test_df[col].values.copy()
        else:
            # Fallback: try raw column name from master table
            scores = test_df.get(self.ivm_name, pd.Series(np.zeros(len(test_df)))).values.copy()
            if self.negate:
                scores = -scores

        # Replace NaN with worst possible score
        scores = np.nan_to_num(scores, nan=-1e10)
        return scores


class StabilityBaseline:
    """
    Rank runs by clustering stability.
    For each run, apply T perturbations (subsampling), rerun the same
    algorithm+hyperparams, compute mean pairwise ARI.
    Higher stability = predicted better.

    Note: This is computationally expensive. Only used for Experiment 4.
    """

    def __init__(self, n_perturbations=20, subsample_frac=0.8, seed=42):
        self.n_perturbations = n_perturbations
        self.subsample_frac = subsample_frac
        self.seed = seed

    def fit(self, train_df, val_df):
        pass

    def predict(self, test_df):
        """
        Compute stability scores. Requires access to raw data X and
        the ability to re-run clustering algorithms.

        For efficiency, this reads X from disk and re-runs clustering.
        """
        from pathlib import Path
        from sklearn.metrics import adjusted_rand_score

        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

        scores = np.zeros(len(test_df))
        rng = np.random.default_rng(self.seed)

        # Import clustering runner to reuse _make_clusterer
        from src.data.clustering_runner import _make_clusterer

        for i, (idx, row) in enumerate(test_df.iterrows()):
            ds_id = row["dataset_id"]
            algo = row["algo"]
            params = json.loads(row["hyperparams_json"])

            try:
                X = np.load(PROCESSED_DIR / ds_id / "X.npy")
            except Exception:
                scores[i] = 0.0
                continue

            n = X.shape[0]
            n_sub = int(n * self.subsample_frac)

            # Generate perturbed partitions
            perturbed_labels = []
            for t in range(self.n_perturbations):
                sub_idx = rng.choice(n, n_sub, replace=False)
                X_sub = X[sub_idx]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clusterer = _make_clusterer(algo, params)
                        if algo == "GaussianMixture":
                            clusterer.fit(X_sub)
                            labels_sub = clusterer.predict(X_sub)
                        else:
                            labels_sub = clusterer.fit_predict(X_sub)
                    perturbed_labels.append((sub_idx, labels_sub))
                except Exception:
                    continue

            if len(perturbed_labels) < 2:
                scores[i] = 0.0
                continue

            # Mean pairwise ARI on common points
            aris = []
            for a in range(len(perturbed_labels)):
                for b in range(a + 1, len(perturbed_labels)):
                    idx_a, lab_a = perturbed_labels[a]
                    idx_b, lab_b = perturbed_labels[b]

                    # Find common points
                    common = np.intersect1d(idx_a, idx_b)
                    if len(common) < 10:
                        continue

                    # Map to local indices
                    map_a = {v: j for j, v in enumerate(idx_a)}
                    map_b = {v: j for j, v in enumerate(idx_b)}
                    local_a = np.array([map_a[c] for c in common])
                    local_b = np.array([map_b[c] for c in common])

                    ari = adjusted_rand_score(lab_a[local_a], lab_b[local_b])
                    aris.append(ari)

            scores[i] = float(np.mean(aris)) if aris else 0.0

        return scores
