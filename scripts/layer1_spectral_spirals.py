"""
Focused Spectral sweep on the spirals dataset — addressing the methodological
gap in the main Section A probe, where Spectral was run with a single
n_neighbors value (10) and default RBF gamma. For spirals_3arms_n0.05 those
defaults likely produce a kNN graph with too many cross-arm shortcuts, and an
RBF kernel that doesn't separate arms.

This script tries a proper Spectral grid:
  - nearest_neighbors affinity × n_neighbors ∈ {3, 5, 7, 10, 15, 20}
  - rbf affinity × gamma ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0}
  - k ∈ [2..10]

Reports: best AMI per (affinity, hyperparameter), to see if there's a regime
where Spectral genuinely solves spirals.
"""
from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    ds = "synth_spirals_3arms_n0.05"
    X = np.load(PROJECT_ROOT / "data" / "processed" / ds / "X.npy").astype(np.float64)
    y = np.load(PROJECT_ROOT / "data" / "processed" / ds / "y_true.npy").astype(np.int64)
    Z = StandardScaler().fit_transform(X)
    print(f"Dataset: {ds}  N={Z.shape[0]}  d={Z.shape[1]}  classes={len(np.unique(y))}")
    print()

    best_overall = (-1.0, None)

    # kNN affinity sweep
    print("=== nearest_neighbors affinity ===")
    print(f"{'n_neighbors':<14} {'k':<4} {'AMI':>8}")
    best_knn = -1.0
    for n_neighbors in [3, 5, 7, 10, 15, 20]:
        for k in range(2, 11):
            t = time.time()
            try:
                labels = SpectralClustering(
                    n_clusters=k, affinity="nearest_neighbors",
                    n_neighbors=n_neighbors, random_state=0,
                    assign_labels="kmeans",
                ).fit_predict(Z)
                ami = adjusted_mutual_info_score(y, labels)
                if ami > best_knn:
                    best_knn = ami
                    print(f"{n_neighbors:<14} {k:<4} {ami:>8.4f}  ★ ({time.time()-t:.1f}s)")
                    if ami > best_overall[0]:
                        best_overall = (ami, f"knn n_neighbors={n_neighbors}, k={k}")
            except Exception as e:
                pass
    print(f"Best kNN-affinity AMI: {best_knn:.4f}")
    print()

    # rbf affinity sweep
    print("=== rbf affinity ===")
    print(f"{'gamma':<10} {'k':<4} {'AMI':>8}")
    best_rbf = -1.0
    for gamma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        for k in range(2, 11):
            try:
                labels = SpectralClustering(
                    n_clusters=k, affinity="rbf",
                    gamma=gamma, random_state=0,
                    assign_labels="kmeans",
                ).fit_predict(Z)
                ami = adjusted_mutual_info_score(y, labels)
                if ami > best_rbf:
                    best_rbf = ami
                    print(f"{gamma:<10} {k:<4} {ami:>8.4f}  ★")
                    if ami > best_overall[0]:
                        best_overall = (ami, f"rbf gamma={gamma}, k={k}")
            except Exception as e:
                pass
    print(f"Best rbf-affinity AMI: {best_rbf:.4f}")
    print()

    print("=" * 60)
    print(f"BASELINE (from main probe): 0.1129")
    print(f"BEST Spectral (this sweep): {best_overall[0]:.4f}  via {best_overall[1]}")
    print(f"GAIN: {best_overall[0] - 0.1129:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
