"""
Experiment: MetaIVM vs CH for GSC hyperparameter selection.

Tests whether MetaIVM (trained on synthetic data only) selects better
(t, α) hyperparameters for GSC than Calinski-Harabasz index.

Uses the Laplacian class from the GSC paper directly.
"""
import sys
import warnings
import logging
import gc
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.metrics import (
    adjusted_mutual_info_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Add GSC's Laplacian class
sys.path.insert(0, "/tmp/gsc_repo/scikit-learn/sklearn/manifold")
from _laplacian import Laplacian

# Add our project
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.splits import make_splits
from src.models.tabular_models import get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "aggregated"


# ── Load datasets ──
def load_datasets():
    """Load the 6 target datasets."""
    datasets = {}

    # Iris (sanity check - CH works well)
    iris = load_iris()
    datasets["Iris"] = (iris.data, iris.target, 3)

    # Wine (sanity check - CH works well)
    wine = load_wine()
    datasets["Wine"] = (wine.data, wine.target, 3)

    # Wdbc (disagreement - CH ≠ AMI)
    wdbc = load_breast_cancer()
    datasets["Wdbc"] = (wdbc.data, wdbc.target, 2)

    # Seeds
    try:
        seeds = fetch_openml("seeds", version=1, as_frame=False, parser="auto")
        datasets["Seeds"] = (seeds.data, seeds.target.astype(int), 3)
    except:
        logger.warning("Seeds not available from OpenML, skipping")

    # Segmentation
    try:
        seg = fetch_openml("segment", version=1, as_frame=False, parser="auto")
        X_seg = seg.data[:, :19]  # First 19 numeric features
        y_seg = seg.target.astype(int) if seg.target.dtype != object else pd.factorize(seg.target)[0]
        datasets["Segmentation"] = (X_seg, y_seg, 7)
    except:
        logger.warning("Segmentation not available, skipping")

    # MNIST64 (digits dataset from sklearn)
    from sklearn.datasets import load_digits
    digits = load_digits()
    datasets["MNIST64"] = (digits.data, digits.target, 10)

    for name, (X, y, k) in datasets.items():
        logger.info(f"  {name}: n={X.shape[0]}, d={X.shape[1]}, k={k}")

    return datasets


# ── GSC implementation ──
def run_gsc(X, k, t, alpha, laplacian_method="norm"):
    """
    Run Generalized Spectral Clustering with parameters (t, alpha).
    Returns cluster labels.
    """
    n = X.shape[0]

    # Build directed kNN graph (same as GSC paper)
    K = max(int(np.ceil(np.log(n))), 2)
    knn = kneighbors_graph(X, n_neighbors=K, mode="distance", include_self=False)
    # Convert to binary adjacency
    adjacency = (knn > 0).astype(float)

    # Compute generalized Laplacian
    measure = (t, alpha, 1)  # gamma=1 (no teleportation)
    try:
        lap = Laplacian(adjacency, standard=False, measure=measure)
    except Exception as e:
        return None

    # Get the Laplacian matrix
    if laplacian_method == "norm":
        L, sqrt_d = lap.normalized()
    elif laplacian_method == "unnorm":
        L, _ = lap.unnormalized()
    elif laplacian_method == "random_walk":
        L, _ = lap.random_walk()
    else:
        raise ValueError(f"Unknown method: {laplacian_method}")

    # Compute eigenvectors
    try:
        if hasattr(L, 'toarray'):
            L_dense = L.toarray()
        else:
            L_dense = np.array(L)

        # Handle NaN/Inf
        L_dense = np.nan_to_num(L_dense, nan=0.0, posinf=1e10, neginf=-1e10)

        eigenvalues, eigenvectors = eigh(L_dense)
        # Take k smallest eigenvectors (skip the first constant one)
        embedding = eigenvectors[:, :k]

        if laplacian_method == "norm":
            # Renormalize rows
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
    except Exception as e:
        return None

    # Cluster the embedding
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init=100, random_state=42)
            labels = km.fit_predict(embedding)
        return labels
    except:
        return None


# ── Feature extraction for MetaIVM ──
def extract_metaivm_features(X, labels):
    """Extract the 28 partition+dataset+graph features MetaIVM uses."""
    n, d = X.shape
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        return None

    # Partition features
    sizes = np.array([np.sum(labels == c) for c in unique_labels])
    probs = sizes / sizes.sum()

    feat = {}
    feat["n_clusters"] = k
    feat["noise_fraction"] = 0.0
    feat["cluster_size_min"] = sizes.min()
    feat["cluster_size_max"] = sizes.max()
    feat["cluster_size_mean"] = sizes.mean()
    feat["cluster_size_std"] = sizes.std()
    feat["cluster_size_median"] = np.median(sizes)
    feat["size_ratio"] = sizes.min() / (sizes.max() + 1e-12)
    feat["entropy"] = -np.sum(probs * np.log(probs + 1e-12))
    feat["gini"] = 1 - np.sum(probs ** 2)
    feat["imbalance"] = sizes.std() / (sizes.mean() + 1e-12)
    feat["singleton_fraction"] = np.sum(sizes == 1) / k

    # Dataset descriptors
    feat["ds_n_samples"] = n
    feat["ds_n_features"] = d
    feat["ds_log_n"] = np.log(n)
    feat["ds_log_d"] = np.log(d + 1)
    feat["ds_d_over_n"] = d / n

    col_means = np.nanmean(X, axis=0)
    col_stds = np.nanstd(X, axis=0)
    from scipy.stats import skew, kurtosis
    col_skew = skew(X, axis=0, nan_policy="omit")
    col_kurt = kurtosis(X, axis=0, nan_policy="omit")

    feat["ds_mean_of_means"] = np.nanmean(col_means)
    feat["ds_mean_of_stds"] = np.nanmean(col_stds)
    feat["ds_mean_of_skewness"] = np.nanmean(col_skew)
    feat["ds_mean_of_kurtosis"] = np.nanmean(col_kurt)
    feat["ds_sparsity"] = np.mean(X == 0)

    from sklearn.decomposition import PCA
    try:
        pca = PCA(random_state=42).fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        feat["ds_intrinsic_dim_pca95"] = np.searchsorted(cumvar, 0.95) + 1
    except:
        feat["ds_intrinsic_dim_pca95"] = d

    from sklearn.metrics import pairwise_distances
    if n > 2000:
        idx = np.random.default_rng(0).choice(n, 2000, replace=False)
        dists = pairwise_distances(X[idx])
    else:
        dists = pairwise_distances(X)
    triu = dists[np.triu_indices_from(dists, k=1)]
    feat["ds_pairwise_dist_mean"] = triu.mean()
    feat["ds_pairwise_dist_std"] = triu.std()
    feat["ds_pairwise_dist_median"] = np.median(triu)

    # Within/between cluster variance
    centroids = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
    within_vars = []
    for c in unique_labels:
        Xc = X[labels == c]
        if len(Xc) > 1:
            within_vars.append(np.mean(np.var(Xc, axis=0)))
        else:
            within_vars.append(0)
    feat["within_cluster_var_mean"] = np.mean(within_vars)
    feat["within_cluster_var_std"] = np.std(within_vars)
    feat["between_cluster_var"] = np.mean(np.var(centroids, axis=0))
    feat["dispersion_ratio"] = feat["within_cluster_var_mean"] / (feat["between_cluster_var"] + 1e-12)

    # Cluster density
    cluster_densities = []
    for c in unique_labels:
        Xc = X[labels == c]
        if len(Xc) > 1:
            cd = pairwise_distances(Xc).mean()
            cluster_densities.append(cd)
        else:
            cluster_densities.append(0)
    feat["cluster_density_mean"] = np.mean(cluster_densities)
    feat["cluster_density_std"] = np.std(cluster_densities)
    feat["cluster_density_max"] = np.max(cluster_densities)

    # Centroid separation
    if k > 1:
        cdists = pairwise_distances(centroids)
        triu_c = cdists[np.triu_indices_from(cdists, k=1)]
        feat["centroid_sep_min"] = triu_c.min()
        feat["centroid_sep_mean"] = triu_c.mean()
        feat["centroid_sep_std"] = triu_c.std()
    else:
        feat["centroid_sep_min"] = feat["centroid_sep_mean"] = feat["centroid_sep_std"] = 0

    # Graph features (kNN graph)
    from sklearn.neighbors import NearestNeighbors
    knn_k = min(15, n - 1)
    nn = NearestNeighbors(n_neighbors=knn_k).fit(X)
    A = nn.kneighbors_graph(mode="connectivity")
    A = ((A + A.T) > 0).astype(float)

    src, dst = A.nonzero()
    m = len(src)
    within = sum(1 for i in range(m) if labels[src[i]] == labels[dst[i]])
    feat["graph_cut_fraction"] = 1 - within / (m + 1e-12)
    feat["graph_normalized_cut_proxy"] = feat["graph_cut_fraction"]
    feat["graph_within_cluster_edge_density"] = within / (m + 1e-12)

    # Per-cluster edge density std
    cluster_edge_densities = []
    for c in unique_labels:
        members = set(np.where(labels == c)[0])
        nc = len(members)
        if nc < 2:
            cluster_edge_densities.append(0)
            continue
        internal = sum(1 for i in range(m) if src[i] in members and dst[i] in members) / 2
        max_e = nc * (nc - 1) / 2
        cluster_edge_densities.append(internal / (max_e + 1e-12))
    feat["graph_within_edge_density_std"] = np.std(cluster_edge_densities)

    # Conductance
    conductances = []
    for c in unique_labels:
        members = set(np.where(labels == c)[0])
        cut = sum(1 for i in range(m) if (src[i] in members) != (dst[i] in members))
        vol = sum(A[j, :].sum() for j in members)
        vol_comp = sum(A[j, :].sum() for j in range(n) if j not in members)
        denom = min(vol, vol_comp)
        conductances.append(cut / (denom + 1e-12))
    feat["graph_conductance_min"] = np.min(conductances)
    feat["graph_conductance_mean"] = np.mean(conductances)

    return feat


def main():
    logger.info("=" * 60)
    logger.info("GSC HYPERPARAMETER SELECTION: MetaIVM vs CH")
    logger.info("=" * 60)

    # Step 1: Load datasets
    logger.info("\n--- Loading datasets ---")
    datasets = load_datasets()

    # Step 2: Train MetaIVM on synthetic data ONLY
    logger.info("\n--- Training MetaIVM on synthetic data only ---")
    features_df = pd.read_csv(PROJECT_ROOT / "data" / "features" / "all_features.csv")
    reg = pd.read_csv(PROJECT_ROOT / "data" / "dataset_registry.csv")

    # Get synthetic dataset IDs only
    synth_ids = reg[reg["source"] == "synthetic"]["dataset_id"].tolist()
    synth_df = features_df[features_df["dataset_id"].isin(synth_ids)]
    logger.info(f"Synthetic training data: {len(synth_df)} runs from {synth_df['dataset_id'].nunique()} datasets")

    feat_cols = get_feature_columns(features_df, "partition_x_graph")

    # Split synthetic into train/val
    synth_unique = synth_df["dataset_id"].unique().tolist()
    splits = make_splits(synth_unique, n_splits=1, train_frac=0.8, val_frac=0.2, test_frac=0.0, seed=42)
    train_ids, val_ids, _ = splits[0]

    train_df = synth_df[synth_df["dataset_id"].isin(train_ids)]
    val_df = synth_df[synth_df["dataset_id"].isin(val_ids)]

    X_tr = np.nan_to_num(train_df[feat_cols].values.astype(np.float64))
    y_tr = train_df["ami"].values
    X_va = np.nan_to_num(val_df[feat_cols].values.astype(np.float64))
    y_va = val_df["ami"].values

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30,
        random_state=42, verbosity=0, n_jobs=1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    logger.info(f"MetaIVM trained on {len(train_ids)} synthetic datasets (val: {len(val_ids)})")

    # Step 3: Run GSC grid search + score with MetaIVM
    logger.info("\n--- Running GSC grid search ---")

    t_values = list(range(0, 26))
    alpha_values = [round(a, 1) for a in np.arange(0.0, 1.6, 0.1)]
    logger.info(f"Grid: t ∈ [0, 25], α ∈ [0.0, 1.5] → {len(t_values) * len(alpha_values)} configs")

    all_results = []

    for ds_name, (X_raw, y_true, k_true) in datasets.items():
        logger.info(f"\n  === {ds_name} (n={X_raw.shape[0]}, k={k_true}) ===")

        # Standardize
        X = StandardScaler().fit_transform(X_raw)

        best_ch, best_ch_ami, best_ch_params = -np.inf, 0, (0, 0)
        best_meta, best_meta_ami, best_meta_params = -np.inf, 0, (0, 0)
        best_oracle_ami, best_oracle_params = 0, (0, 0)
        n_valid = 0

        for t, alpha in product(t_values, alpha_values):
            labels = run_gsc(X, k_true, t, alpha, laplacian_method="norm")
            if labels is None:
                continue

            n_unique = len(np.unique(labels))
            if n_unique < 2:
                continue

            n_valid += 1
            ami = adjusted_mutual_info_score(y_true, labels)

            # CH score
            try:
                ch = calinski_harabasz_score(X, labels)
            except:
                ch = -np.inf

            # MetaIVM score
            feats = extract_metaivm_features(X, labels)
            if feats is not None:
                feat_vec = np.array([[feats.get(c, 0) for c in feat_cols]])
                feat_vec = np.nan_to_num(feat_vec.astype(np.float64))
                meta_score = model.predict(feat_vec)[0]
            else:
                meta_score = -np.inf

            # Track best
            if ch > best_ch:
                best_ch = ch
                best_ch_ami = ami
                best_ch_params = (t, alpha)

            if meta_score > best_meta:
                best_meta = meta_score
                best_meta_ami = ami
                best_meta_params = (t, alpha)

            if ami > best_oracle_ami:
                best_oracle_ami = ami
                best_oracle_params = (t, alpha)

        logger.info(f"    Valid configs: {n_valid}")
        logger.info(f"    CH    → (t={best_ch_params[0]}, α={best_ch_params[1]:.1f}) → AMI = {best_ch_ami:.4f}")
        logger.info(f"    Meta  → (t={best_meta_params[0]}, α={best_meta_params[1]:.1f}) → AMI = {best_meta_ami:.4f}")
        logger.info(f"    Oracle→ (t={best_oracle_params[0]}, α={best_oracle_params[1]:.1f}) → AMI = {best_oracle_ami:.4f}")

        all_results.append({
            "dataset": ds_name,
            "n": X.shape[0],
            "k": k_true,
            "n_valid_configs": n_valid,
            "ch_ami": best_ch_ami,
            "ch_params": f"t={best_ch_params[0]},α={best_ch_params[1]:.1f}",
            "metaivm_ami": best_meta_ami,
            "metaivm_params": f"t={best_meta_params[0]},α={best_meta_params[1]:.1f}",
            "oracle_ami": best_oracle_ami,
            "oracle_params": f"t={best_oracle_params[0]},α={best_oracle_params[1]:.1f}",
            "ch_regret": best_oracle_ami - best_ch_ami,
            "metaivm_regret": best_oracle_ami - best_meta_ami,
        })

    # Summary
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(RESULTS_DIR / "exp_gsc_metaivm.csv", index=False)

    print("\n" + "=" * 80)
    print("GSC HYPERPARAMETER SELECTION: MetaIVM (synth-only) vs CH")
    print("=" * 80)
    print(f"\n{'Dataset':<15s} {'CH AMI':>8s} {'Meta AMI':>10s} {'Oracle':>8s} {'CH Reg':>8s} {'Meta Reg':>10s} {'Winner':>8s}")
    print("-" * 72)
    for _, row in res_df.iterrows():
        winner = "MetaIVM" if row["metaivm_ami"] > row["ch_ami"] + 0.005 else (
            "CH" if row["ch_ami"] > row["metaivm_ami"] + 0.005 else "Tie")
        print(f"{row['dataset']:<15s} {row['ch_ami']:>8.4f} {row['metaivm_ami']:>10.4f} {row['oracle_ami']:>8.4f} "
              f"{row['ch_regret']:>8.4f} {row['metaivm_regret']:>10.4f} {winner:>8s}")

    print(f"\nMean CH regret:      {res_df['ch_regret'].mean():.4f}")
    print(f"Mean MetaIVM regret: {res_df['metaivm_regret'].mean():.4f}")
    print(f"MetaIVM wins: {(res_df['metaivm_ami'] > res_df['ch_ami'] + 0.005).sum()}")
    print(f"CH wins:      {(res_df['ch_ami'] > res_df['metaivm_ami'] + 0.005).sum()}")
    print(f"Ties:         {((res_df['metaivm_ami'] - res_df['ch_ami']).abs() <= 0.005).sum()}")


if __name__ == "__main__":
    main()
