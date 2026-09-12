"""
MetaIVM-guided local search: use MetaIVM as an objective to refine clusterings.

For each meta-test dataset:
1. Start from initial partitions (KMeans, DBSCAN, etc.)
2. Iteratively refine by swapping points, splitting/merging clusters
3. Accept moves that improve MetaIVM score
4. Compare refined AMI vs initial AMI

Also runs oracle-guided search (true AMI) as upper bound.
"""

import sys
import gc
import logging
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "aggregated"


# ──────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (lightweight, same as in training)
# ──────────────────────────────────────────────────────────────────────

def extract_features(X, labels, knn_graph=None):
    """Extract partition + dataset + graph features from (X, labels).
    Returns a dict of feature values."""
    n, d = X.shape
    unique_labels = np.unique(labels[labels >= 0])  # exclude noise
    k = len(unique_labels)

    if k < 1:
        return None

    # Cluster sizes
    sizes = np.array([np.sum(labels == c) for c in unique_labels])
    size_probs = sizes / sizes.sum()

    feat = {}
    feat["n_clusters"] = k
    feat["noise_fraction"] = np.mean(labels < 0)
    feat["cluster_size_min"] = sizes.min()
    feat["cluster_size_max"] = sizes.max()
    feat["cluster_size_mean"] = sizes.mean()
    feat["cluster_size_std"] = sizes.std()
    feat["cluster_size_median"] = np.median(sizes)
    feat["size_ratio"] = sizes.min() / (sizes.max() + 1e-12)
    feat["entropy"] = -np.sum(size_probs * np.log(size_probs + 1e-12))
    feat["gini"] = 1 - np.sum(size_probs ** 2)
    feat["imbalance"] = sizes.std() / (sizes.mean() + 1e-12)
    feat["singleton_fraction"] = np.mean(sizes == 1)

    # Dataset descriptors
    feat["ds_n_samples"] = n
    feat["ds_n_features"] = d
    feat["ds_log_n"] = np.log(n + 1)
    feat["ds_log_d"] = np.log(d + 1)
    feat["ds_d_over_n"] = d / n
    col_means = np.nanmean(X, axis=0)
    col_stds = np.nanstd(X, axis=0)
    feat["ds_mean_of_means"] = np.nanmean(col_means)
    feat["ds_mean_of_stds"] = np.nanmean(col_stds)
    from scipy.stats import skew, kurtosis as kurt_func
    feat["ds_mean_of_skewness"] = np.nanmean(skew(X, axis=0, nan_policy="omit"))
    feat["ds_mean_of_kurtosis"] = np.nanmean(kurt_func(X, axis=0, nan_policy="omit"))
    feat["ds_sparsity"] = np.mean(np.abs(X) < 1e-8)
    from sklearn.decomposition import PCA
    try:
        pca = PCA(random_state=0).fit(X[:min(n, 2000)])
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        feat["ds_intrinsic_dim_pca95"] = np.searchsorted(cumvar, 0.95) + 1
    except:
        feat["ds_intrinsic_dim_pca95"] = d
    from sklearn.metrics import pairwise_distances
    sample_idx = np.random.RandomState(0).choice(n, min(500, n), replace=False)
    dists = pairwise_distances(X[sample_idx])
    triu = dists[np.triu_indices_from(dists, k=1)]
    feat["ds_pairwise_dist_mean"] = triu.mean()
    feat["ds_pairwise_dist_std"] = triu.std()
    feat["ds_pairwise_dist_median"] = np.median(triu)

    # Partition-data interaction features
    mask = labels >= 0
    X_clean = X[mask]
    labels_clean = labels[mask]
    if k >= 2 and len(X_clean) > k:
        centroids = np.array([X_clean[labels_clean == c].mean(axis=0) for c in unique_labels])
        within_vars = []
        for c in unique_labels:
            Xc = X_clean[labels_clean == c]
            if len(Xc) > 1:
                within_vars.append(np.mean(np.var(Xc, axis=0)))
            else:
                within_vars.append(0.0)
        within_vars = np.array(within_vars)
        feat["within_cluster_var_mean"] = within_vars.mean()
        feat["within_cluster_var_std"] = within_vars.std()

        global_centroid = X_clean.mean(axis=0)
        between_var = np.sum([sizes[i] * np.sum((centroids[i] - global_centroid) ** 2)
                             for i in range(k)]) / n
        feat["between_cluster_var"] = between_var
        feat["dispersion_ratio"] = within_vars.mean() / (between_var + 1e-12)

        # Cluster densities and centroid separations
        from sklearn.metrics import pairwise_distances as pdist
        centroid_dists = pdist(centroids)
        np.fill_diagonal(centroid_dists, np.inf)
        feat["centroid_sep_min"] = centroid_dists.min()
        feat["centroid_sep_mean"] = centroid_dists[centroid_dists < np.inf].mean()
        feat["centroid_sep_std"] = centroid_dists[centroid_dists < np.inf].std()

        cluster_densities = []
        for i, c in enumerate(unique_labels):
            Xc = X_clean[labels_clean == c]
            if len(Xc) > 1:
                intra = pdist(Xc).mean()
                cluster_densities.append(1.0 / (intra + 1e-12))
            else:
                cluster_densities.append(0.0)
        feat["cluster_density_mean"] = np.mean(cluster_densities)
        feat["cluster_density_std"] = np.std(cluster_densities)
        feat["cluster_density_max"] = np.max(cluster_densities)
    else:
        for key in ["within_cluster_var_mean", "within_cluster_var_std",
                     "between_cluster_var", "dispersion_ratio",
                     "centroid_sep_min", "centroid_sep_mean", "centroid_sep_std",
                     "cluster_density_mean", "cluster_density_std", "cluster_density_max"]:
            feat[key] = 0.0

    # Graph features
    if knn_graph is not None:
        from scipy.sparse import issparse
        if issparse(knn_graph):
            rows, cols = knn_graph.nonzero()
        else:
            rows, cols = np.where(knn_graph > 0)
        total_edges = len(rows)
        if total_edges > 0 and k >= 2:
            within = np.sum(labels[rows] == labels[cols])
            feat["graph_cut_fraction"] = 1 - within / total_edges
            feat["graph_normalized_cut_proxy"] = feat["graph_cut_fraction"]

            # Within-cluster edge density per cluster
            wc_densities = []
            for c in unique_labels:
                members = set(np.where(labels == c)[0])
                nc = len(members)
                if nc < 2:
                    wc_densities.append(0.0)
                    continue
                wc_edges = sum(1 for r, c_ in zip(rows, cols) if r in members and c_ in members)
                max_edges = nc * (nc - 1)
                wc_densities.append(wc_edges / (max_edges + 1e-12))
            feat["graph_within_cluster_edge_density"] = np.mean(wc_densities)
            feat["graph_within_edge_density_std"] = np.std(wc_densities)

            # Conductance
            conductances = []
            for c in unique_labels:
                members = set(np.where(labels == c)[0])
                cut = sum(1 for r, c_ in zip(rows, cols) if r in members and c_ not in members)
                vol = sum(1 for r in rows if r in members)
                vol_comp = sum(1 for r in rows if r not in members)
                denom = min(vol, vol_comp)
                conductances.append(cut / (denom + 1e-12))
            feat["graph_conductance_min"] = np.min(conductances)
            feat["graph_conductance_mean"] = np.mean(conductances)
        else:
            for key in ["graph_cut_fraction", "graph_normalized_cut_proxy",
                        "graph_within_cluster_edge_density", "graph_within_edge_density_std",
                        "graph_conductance_min", "graph_conductance_mean"]:
                feat[key] = 0.0
    else:
        for key in ["graph_cut_fraction", "graph_normalized_cut_proxy",
                    "graph_within_cluster_edge_density", "graph_within_edge_density_std",
                    "graph_conductance_min", "graph_conductance_mean"]:
            feat[key] = 0.0

    return feat


def features_to_array(feat_dict, feat_cols):
    """Convert feature dict to numpy array in the right column order."""
    return np.array([feat_dict.get(c, 0.0) for c in feat_cols], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────
# LOCAL SEARCH
# ──────────────────────────────────────────────────────────────────────

def score_partition(X, labels, knn_graph, feat_cols, scaler, model):
    """Score a partition using MetaIVM."""
    feat = extract_features(X, labels, knn_graph)
    if feat is None:
        return -np.inf
    arr = features_to_array(feat, feat_cols).reshape(1, -1)
    arr = np.nan_to_num(arr)
    arr_s = scaler.transform(arr)
    return model.predict(arr_s)[0]


def local_search_refine(X, labels_init, knn_graph, feat_cols, scaler, model,
                        max_iters=30, patience=5, use_oracle=False, y_true=None,
                        n_swap_candidates=50):
    """
    Refine a partition via local search guided by MetaIVM (or oracle AMI).

    Moves:
    1. Point swap: move a random point to its nearest cluster
    2. Cluster split: split the largest cluster via 2-means
    3. Cluster merge: merge the two closest clusters
    """
    labels = labels_init.copy()
    n = len(labels)

    if use_oracle:
        current_score = adjusted_mutual_info_score(y_true, labels)
    else:
        current_score = score_partition(X, labels, knn_graph, feat_cols, scaler, model)

    best_labels = labels.copy()
    best_score = current_score
    no_improve = 0

    for iteration in range(max_iters):
        improved = False

        # --- Move 1: Point swaps (try swapping boundary points) ---
        unique_labels = np.unique(labels[labels >= 0])
        k = len(unique_labels)
        if k < 2:
            break

        # Find boundary points (points whose nearest neighbor is in a different cluster)
        centroids = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
        boundary_points = []
        for i in range(n):
            if labels[i] < 0:
                continue
            dists_to_centroids = np.linalg.norm(centroids - X[i], axis=1)
            own_cluster_idx = np.where(unique_labels == labels[i])[0][0]
            dists_to_centroids[own_cluster_idx] = np.inf
            nearest_other = unique_labels[np.argmin(dists_to_centroids)]
            if dists_to_centroids.min() < np.linalg.norm(X[i] - centroids[own_cluster_idx]) * 1.5:
                boundary_points.append((i, nearest_other))

        # Try swapping up to n_swap_candidates boundary points
        rng = np.random.RandomState(iteration)
        if len(boundary_points) > n_swap_candidates:
            idxs = rng.choice(len(boundary_points), n_swap_candidates, replace=False)
            boundary_points = [boundary_points[i] for i in idxs]

        for point_idx, target_cluster in boundary_points:
            old_cluster = labels[point_idx]
            # Don't empty a cluster
            if np.sum(labels == old_cluster) <= 2:
                continue
            labels[point_idx] = target_cluster
            if use_oracle:
                new_score = adjusted_mutual_info_score(y_true, labels)
            else:
                new_score = score_partition(X, labels, knn_graph, feat_cols, scaler, model)
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_labels = labels.copy()
            else:
                labels[point_idx] = old_cluster  # revert

        # --- Move 2: Split largest cluster ---
        unique_labels = np.unique(labels[labels >= 0])
        k = len(unique_labels)
        sizes = {c: np.sum(labels == c) for c in unique_labels}
        largest = max(sizes, key=sizes.get)
        if sizes[largest] >= 6:
            mask = labels == largest
            X_sub = X[mask]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sub_labels = KMeans(n_clusters=2, n_init=5, random_state=iteration).fit_predict(X_sub)
                new_label = max(unique_labels) + 1
                candidate = labels.copy()
                candidate[mask] = np.where(sub_labels == 0, largest, new_label)
                if use_oracle:
                    split_score = adjusted_mutual_info_score(y_true, candidate)
                else:
                    split_score = score_partition(X, candidate, knn_graph, feat_cols, scaler, model)
                if split_score > current_score:
                    labels = candidate
                    current_score = split_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_labels = labels.copy()
            except:
                pass

        # --- Move 3: Merge two closest clusters ---
        unique_labels = np.unique(labels[labels >= 0])
        k = len(unique_labels)
        if k >= 3:
            centroids = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
            from sklearn.metrics import pairwise_distances
            cdists = pairwise_distances(centroids)
            np.fill_diagonal(cdists, np.inf)
            i, j = np.unravel_index(cdists.argmin(), cdists.shape)
            candidate = labels.copy()
            candidate[candidate == unique_labels[j]] = unique_labels[i]
            if use_oracle:
                merge_score = adjusted_mutual_info_score(y_true, candidate)
            else:
                merge_score = score_partition(X, candidate, knn_graph, feat_cols, scaler, model)
            if merge_score > current_score:
                labels = candidate
                current_score = merge_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_labels = labels.copy()

        if not improved:
            no_improve += 1
            if no_improve >= patience:
                break
        else:
            no_improve = 0

    return best_labels, best_score


# ──────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ──────────────────────────────────────────────────────────────────────

def main():
    from src.evaluation.splits import make_splits
    from scipy.sparse import load_npz

    # Load features for training MetaIVM
    features_df = pd.read_csv(DATA_DIR / "features" / "all_features.csv")
    feat_cols = [c for c in features_df.columns
                 if c not in ["dataset_id", "run_id", "algo", "ami"]]

    dataset_ids = features_df["dataset_id"].unique().tolist()

    all_results = []

    for seed in range(5):
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*60}")

        splits = make_splits(dataset_ids, n_splits=1, train_frac=0.6,
                            val_frac=0.2, test_frac=0.2, seed=seed)
        train_ids, val_ids, test_ids = splits[0]

        # Train MetaIVM (MLP) on train+val
        train_df = features_df[features_df["dataset_id"].isin(train_ids)]
        val_df = features_df[features_df["dataset_id"].isin(val_ids)]

        X_train = np.nan_to_num(train_df[feat_cols].values.astype(np.float64))
        y_train = train_df["ami"].values
        X_val = np.nan_to_num(val_df[feat_cols].values.astype(np.float64))
        y_val = val_df["ami"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_all = np.vstack([X_train_s, X_val_s])
        y_all = np.concatenate([y_train, y_val])
        val_frac = max(len(X_val_s) / len(X_all), 0.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MLPRegressor(
                hidden_layer_sizes=(512, 256, 128),
                learning_rate_init=1e-3,
                max_iter=500, early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=val_frac,
                random_state=seed,
            )
            model.fit(X_all, y_all)
        logger.info(f"MetaIVM MLP trained (loss={model.loss_:.4f})")

        # Evaluate on meta-test datasets
        test_count = 0
        for ds_id in test_ids:
            proc_dir = PROCESSED_DIR / ds_id
            if not (proc_dir / "X.npy").exists():
                continue

            X = np.load(proc_dir / "X.npy")
            y_true = np.load(proc_dir / "y_true.npy")
            n = X.shape[0]

            # Skip very large datasets for speed
            if n > 5000:
                continue

            # Load kNN graph if available
            knn_path = proc_dir / "knn_k15.npz"
            knn_graph = None
            if knn_path.exists():
                knn_graph = load_npz(knn_path)

            k_true = len(np.unique(y_true))

            # Generate initial partitions from different algorithms
            initial_partitions = []

            # KMeans with various k
            for k in [2, 3, 5, max(2, k_true - 1), k_true, k_true + 1, k_true + 3, 10]:
                if k >= n or k < 2:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
                        initial_partitions.append((f"KMeans_k{k}", labels))
                except:
                    pass

            # GMM
            for k in [k_true, max(2, k_true - 1), k_true + 2]:
                if k >= n or k < 2:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        labels = GaussianMixture(n_components=k, random_state=seed).fit_predict(X)
                        initial_partitions.append((f"GMM_k{k}", labels))
                except:
                    pass

            # Agglomerative
            for k in [k_true, max(2, k_true - 1)]:
                if k >= n or k < 2:
                    continue
                try:
                    labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
                    initial_partitions.append((f"Agglo_k{k}", labels))
                except:
                    pass

            if not initial_partitions:
                continue

            test_count += 1

            for algo_name, labels_init in initial_partitions:
                ami_init = adjusted_mutual_info_score(y_true, labels_init)

                # MetaIVM-guided refinement
                labels_metaivm, _ = local_search_refine(
                    X, labels_init, knn_graph, feat_cols, scaler, model,
                    max_iters=20, patience=3, n_swap_candidates=30,
                )
                ami_metaivm = adjusted_mutual_info_score(y_true, labels_metaivm)

                # Oracle-guided refinement (upper bound)
                labels_oracle, _ = local_search_refine(
                    X, labels_init, knn_graph, feat_cols, scaler, model,
                    max_iters=20, patience=3, n_swap_candidates=30,
                    use_oracle=True, y_true=y_true,
                )
                ami_oracle = adjusted_mutual_info_score(y_true, labels_oracle)

                # Silhouette-guided selection (for comparison)
                try:
                    sil_init = silhouette_score(X[:min(n, 5000)], labels_init[:min(n, 5000)])
                except:
                    sil_init = -1

                all_results.append({
                    "seed": seed,
                    "dataset_id": ds_id,
                    "algo": algo_name,
                    "n_samples": n,
                    "k_true": k_true,
                    "ami_initial": ami_init,
                    "ami_metaivm_refined": ami_metaivm,
                    "ami_oracle_refined": ami_oracle,
                    "delta_metaivm": ami_metaivm - ami_init,
                    "delta_oracle": ami_oracle - ami_init,
                })

            if test_count % 10 == 0:
                logger.info(f"  Processed {test_count} test datasets")

        logger.info(f"Seed {seed}: processed {test_count} test datasets")

    # ── Results ──
    res = pd.DataFrame(all_results)
    res.to_csv(RESULTS_DIR / "exp_local_search.csv", index=False)

    print("\n" + "=" * 70)
    print("METAIVM-GUIDED LOCAL SEARCH RESULTS")
    print("=" * 70)

    print(f"\nTotal: {len(res)} (dataset, algorithm) pairs across {res['dataset_id'].nunique()} datasets, {res['seed'].nunique()} seeds")

    print(f"\n{'Metric':<30s} {'Mean':>8s} {'Std':>8s} {'Median':>8s}")
    print("-" * 58)
    print(f"{'AMI initial':<30s} {res['ami_initial'].mean():>8.4f} {res['ami_initial'].std():>8.4f} {res['ami_initial'].median():>8.4f}")
    print(f"{'AMI MetaIVM-refined':<30s} {res['ami_metaivm_refined'].mean():>8.4f} {res['ami_metaivm_refined'].std():>8.4f} {res['ami_metaivm_refined'].median():>8.4f}")
    print(f"{'AMI Oracle-refined':<30s} {res['ami_oracle_refined'].mean():>8.4f} {res['ami_oracle_refined'].std():>8.4f} {res['ami_oracle_refined'].median():>8.4f}")

    print(f"\n{'Δ MetaIVM (improvement)':<30s} {res['delta_metaivm'].mean():>8.4f} {res['delta_metaivm'].std():>8.4f} {res['delta_metaivm'].median():>8.4f}")
    print(f"{'Δ Oracle (upper bound)':<30s} {res['delta_oracle'].mean():>8.4f} {res['delta_oracle'].std():>8.4f} {res['delta_oracle'].median():>8.4f}")

    # How often does MetaIVM improve?
    improved = (res["delta_metaivm"] > 0.01).mean()
    hurt = (res["delta_metaivm"] < -0.01).mean()
    neutral = 1 - improved - hurt
    print(f"\nMetaIVM refinement: improved {improved:.1%}, neutral {neutral:.1%}, hurt {hurt:.1%}")

    oracle_improved = (res["delta_oracle"] > 0.01).mean()
    print(f"Oracle refinement: improved {oracle_improved:.1%}")

    # Efficiency: how close does MetaIVM get to oracle?
    room = res["delta_oracle"] - res["delta_metaivm"]
    mask = res["delta_oracle"] > 0.01
    if mask.sum() > 0:
        efficiency = res.loc[mask, "delta_metaivm"].mean() / res.loc[mask, "delta_oracle"].mean()
        print(f"\nMetaIVM captures {efficiency:.1%} of oracle improvement (on improvable cases)")

    # Per-algorithm breakdown
    print(f"\n{'Algorithm':<20s} {'AMI init':>10s} {'AMI meta':>10s} {'Δ meta':>10s} {'AMI oracle':>10s}")
    print("-" * 65)
    for algo in sorted(res["algo"].unique()):
        sub = res[res["algo"] == algo]
        print(f"{algo:<20s} {sub['ami_initial'].mean():>10.4f} {sub['ami_metaivm_refined'].mean():>10.4f} "
              f"{sub['delta_metaivm'].mean():>+10.4f} {sub['ami_oracle_refined'].mean():>10.4f}")


if __name__ == "__main__":
    main()
