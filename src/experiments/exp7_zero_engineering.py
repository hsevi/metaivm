"""
Experiment 7 — Zero Feature Engineering.

Compares approaches that require no hand-crafted features:
  1. Graph embeddings (spectral) + XGBoost
  2. GNN end-to-end
  3. Brute-force auto-features + XGBoost

Against the hand-crafted feature baseline.
"""

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.evaluation.metrics import compute_all_metrics, aggregate_metrics
from src.evaluation.splits import make_splits

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = DATA_DIR / "clustering_runs"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"


# ─────────────────────────────────────────────────────────────
# 1. Graph Spectral Embeddings → XGBoost
# ─────────────────────────────────────────────────────────────

def compute_graph_embeddings(dataset_id, labels, knn_k=10, n_components=32):
    """
    Compute graph spectral embeddings of the kNN graph, then pool
    by cluster membership to get fixed-size representation.

    Returns a flat feature vector (no hand-crafted features needed).
    """
    proc_dir = PROCESSED_DIR / dataset_id
    graph_path = proc_dir / f"A_knn_k{knn_k}.npz"
    if not graph_path.exists():
        return None

    A = sparse.load_npz(graph_path)
    n = A.shape[0]

    # Spectral embedding of the graph
    n_comp = min(n_components, n - 2, A.shape[0] - 2)
    if n_comp < 2:
        return None

    try:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        node_embs = svd.fit_transform(A.astype(np.float64))
    except Exception:
        return None

    # Pool by cluster
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    k = len(cluster_labels)

    if k < 1:
        return None

    # Per-cluster pooling: mean and std of node embeddings
    cluster_means = []
    cluster_stds = []
    cluster_sizes = []
    for c in cluster_labels:
        mask = labels == c
        c_embs = node_embs[mask]
        cluster_means.append(c_embs.mean(axis=0))
        cluster_stds.append(c_embs.std(axis=0) if c_embs.shape[0] > 1 else np.zeros(n_comp))
        cluster_sizes.append(mask.sum())

    cluster_means = np.array(cluster_means)
    cluster_stds = np.array(cluster_stds)
    cluster_sizes = np.array(cluster_sizes, dtype=float)

    # Global pooling of cluster-level stats
    features = []

    # Mean of cluster means (n_comp values)
    features.extend(cluster_means.mean(axis=0))
    # Std of cluster means (captures separation)
    features.extend(cluster_means.std(axis=0))
    # Mean of cluster stds (captures compactness)
    features.extend(cluster_stds.mean(axis=0))
    # Cluster size statistics
    features.extend([
        k,
        np.mean(cluster_sizes),
        np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-10),
        np.min(cluster_sizes),
        np.max(cluster_sizes),
    ])

    # Noise fraction
    noise_frac = (labels == -1).sum() / len(labels)
    features.append(noise_frac)

    # Global mean of all node embeddings
    features.extend(node_embs.mean(axis=0))

    return np.array(features, dtype=np.float64)


def build_graph_embedding_features(master_df, n_components=32):
    """Build graph embedding features for all runs."""
    logger.info("Computing graph embeddings for all runs...")
    all_rows = []

    # Cache: compute embeddings per (dataset, run) pair
    dataset_ids = master_df["dataset_id"].unique()

    for i, (idx, row) in enumerate(master_df.iterrows()):
        ds_id = row["dataset_id"]
        algo = row["algo"]
        run_id = row["run_id"]

        # Load labels
        label_path = RUNS_DIR / ds_id / f"{algo}_{run_id}.npy"
        try:
            labels = np.load(label_path)
        except Exception:
            all_rows.append(None)
            continue

        emb = compute_graph_embeddings(ds_id, labels, n_components=n_components)
        all_rows.append(emb)

        if (i + 1) % 2000 == 0:
            logger.info(f"  {i+1}/{len(master_df)} embeddings computed")

    # Convert to DataFrame
    valid_mask = [e is not None for e in all_rows]
    valid_embs = [e for e in all_rows if e is not None]

    if not valid_embs:
        return None, valid_mask

    # Pad to max length
    max_len = max(len(e) for e in valid_embs)
    padded = np.zeros((len(valid_embs), max_len))
    for i, e in enumerate(valid_embs):
        padded[i, :len(e)] = e

    col_names = [f"gemb_{j}" for j in range(max_len)]
    emb_df = pd.DataFrame(padded, columns=col_names)

    # Add metadata
    valid_rows = master_df[valid_mask].reset_index(drop=True)
    emb_df["dataset_id"] = valid_rows["dataset_id"].values
    emb_df["run_id"] = valid_rows["run_id"].values
    emb_df["algo"] = valid_rows["algo"].values
    emb_df["ami"] = valid_rows["ami"].values

    logger.info(f"Graph embeddings: {len(emb_df)} runs, {max_len} features")
    return emb_df, valid_mask


# ─────────────────────────────────────────────────────────────
# 2. Brute-force auto-features
# ─────────────────────────────────────────────────────────────

def compute_auto_features(dataset_id, labels):
    """
    Compute brute-force features from (X, labels) without any design.
    Just dump per-cluster stats and pairwise centroid distances.
    """
    proc_dir = PROCESSED_DIR / dataset_id
    X = np.load(proc_dir / "X.npy")
    n, d = X.shape

    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    k = len(cluster_labels)

    features = {}
    features["n_samples"] = n
    features["n_features"] = d
    features["n_clusters"] = k
    features["noise_frac"] = (labels == -1).sum() / n

    if k < 1:
        return features

    # Per-cluster: mean, var of each feature → then aggregate across clusters
    cluster_means = []
    cluster_vars = []
    cluster_sizes = []
    for c in cluster_labels:
        mask = labels == c
        Xc = X[mask]
        cluster_means.append(Xc.mean(axis=0))
        cluster_vars.append(Xc.var(axis=0) if Xc.shape[0] > 1 else np.zeros(d))
        cluster_sizes.append(mask.sum())

    cluster_means = np.array(cluster_means)  # (k, d)
    cluster_vars = np.array(cluster_vars)    # (k, d)
    cluster_sizes = np.array(cluster_sizes, dtype=float)

    # Cluster size stats
    features["size_mean"] = np.mean(cluster_sizes)
    features["size_std"] = np.std(cluster_sizes)
    features["size_min"] = np.min(cluster_sizes)
    features["size_max"] = np.max(cluster_sizes)
    features["size_ratio"] = np.max(cluster_sizes) / (np.min(cluster_sizes) + 1)

    # Within-cluster variance (averaged across features, then across clusters)
    features["within_var_mean"] = np.mean(cluster_vars)
    features["within_var_std"] = np.std(np.mean(cluster_vars, axis=1))

    # Between-cluster variance (variance of centroids)
    features["between_var"] = np.mean(np.var(cluster_means, axis=0))

    # Centroid pairwise distances
    if k >= 2:
        from scipy.spatial.distance import pdist
        dists = pdist(cluster_means)
        features["centroid_dist_mean"] = np.mean(dists)
        features["centroid_dist_std"] = np.std(dists)
        features["centroid_dist_min"] = np.min(dists)
    else:
        features["centroid_dist_mean"] = 0
        features["centroid_dist_std"] = 0
        features["centroid_dist_min"] = 0

    # Dispersion ratio
    within = np.mean(cluster_vars)
    between = features["between_var"]
    features["dispersion_ratio"] = between / (within + 1e-10)

    # Entropy of cluster sizes
    p = cluster_sizes / cluster_sizes.sum()
    features["entropy"] = -np.sum(p * np.log(p + 1e-10))

    # Global data stats
    features["global_var_mean"] = np.mean(np.var(X, axis=0))

    return features


def build_auto_features(master_df):
    """Build brute-force auto-features for all runs."""
    logger.info("Computing auto-features for all runs...")
    all_rows = []

    for i, (idx, row) in enumerate(master_df.iterrows()):
        ds_id = row["dataset_id"]
        algo = row["algo"]
        run_id = row["run_id"]

        label_path = RUNS_DIR / ds_id / f"{algo}_{run_id}.npy"
        try:
            labels = np.load(label_path)
            feats = compute_auto_features(ds_id, labels)
        except Exception:
            feats = {}

        feats["dataset_id"] = ds_id
        feats["run_id"] = run_id
        feats["algo"] = algo
        feats["ami"] = row["ami"]
        all_rows.append(feats)

        if (i + 1) % 2000 == 0:
            logger.info(f"  {i+1}/{len(master_df)} auto-features computed")

    df = pd.DataFrame(all_rows)
    feat_cols = [c for c in df.columns if c not in {"dataset_id", "run_id", "algo", "ami"}]
    logger.info(f"Auto-features: {len(df)} runs, {len(feat_cols)} features")
    return df


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_xgboost_on_features(features_df, seeds, label=""):
    """Train XGBoost on given features and evaluate."""
    from xgboost import XGBRegressor

    meta_cols = {"dataset_id", "run_id", "algo", "ami", "hyperparams_json"}
    feat_cols = [c for c in features_df.columns if c not in meta_cols]

    dataset_ids = features_df["dataset_id"].unique().tolist()
    all_per_dataset = []

    for seed in seeds:
        splits = make_splits(
            dataset_ids, n_splits=1,
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
            seed=seed,
        )

        for train_ids, val_ids, test_ids in splits:
            train_df = features_df[features_df["dataset_id"].isin(train_ids)]
            test_df = features_df[features_df["dataset_id"].isin(test_ids)]

            X_train = train_df[feat_cols].fillna(0).values
            y_train = train_df["ami"].values

            model = XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=seed, n_jobs=-1,
            )
            model.fit(X_train, y_train)

            for ds_id in test_ids:
                ds_df = test_df[test_df["dataset_id"] == ds_id]
                if len(ds_df) < 2:
                    continue
                X_test = ds_df[feat_cols].fillna(0).values
                pred = model.predict(X_test)
                metrics = compute_all_metrics(ds_df["ami"].values, pred)
                metrics["dataset_id"] = ds_id
                metrics["seed"] = seed
                all_per_dataset.append(metrics)

        gc.collect()

    if not all_per_dataset:
        return {}

    agg = aggregate_metrics(all_per_dataset)
    return agg


def evaluate_gnn(features_df, seeds):
    """Evaluate GNN end-to-end."""
    from src.models.gnn_model import GNNSurrogate

    dataset_ids = features_df["dataset_id"].unique().tolist()
    all_per_dataset = []

    for seed in seeds:
        splits = make_splits(
            dataset_ids, n_splits=1,
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
            seed=seed,
        )

        for train_ids, val_ids, test_ids in splits:
            train_df = features_df[features_df["dataset_id"].isin(train_ids)]
            val_df = features_df[features_df["dataset_id"].isin(val_ids)]
            test_df = features_df[features_df["dataset_id"].isin(test_ids)]

            gnn = GNNSurrogate(
                hidden_dim=128, n_layers=3, lr=0.001,
                epochs=80, patience=15, batch_size=32, seed=seed,
            )
            gnn.fit(train_df, val_df)

            for ds_id in test_ids:
                ds_df = test_df[test_df["dataset_id"] == ds_id]
                if len(ds_df) < 2:
                    continue
                pred = gnn.predict(ds_df)
                metrics = compute_all_metrics(ds_df["ami"].values, pred)
                metrics["dataset_id"] = ds_id
                metrics["seed"] = seed
                all_per_dataset.append(metrics)

        gc.collect()

    if not all_per_dataset:
        return {}

    return aggregate_metrics(all_per_dataset)


def run_experiment():
    """Run all zero-engineering experiments."""
    with open(PROJECT_ROOT / "configs" / "experiment_configs.yaml") as f:
        cfg = yaml.safe_load(f)
    seeds = cfg["splits"]["seeds"]

    master = pd.read_csv(FEATURES_DIR / "runs_master.csv")
    features_df = pd.read_csv(FEATURES_DIR / "all_features.csv")

    # Merge hyperparams_json for GNN
    if "hyperparams_json" not in features_df.columns:
        features_df = features_df.merge(
            master[["dataset_id", "run_id", "hyperparams_json"]],
            on=["dataset_id", "run_id"], how="left",
        )

    results = []

    # ── 1. Hand-crafted features (reference) ──
    logger.info("\n=== Hand-crafted features (reference) ===")
    agg = evaluate_xgboost_on_features(features_df, seeds, "hand-crafted")
    results.append({
        "method": "XGBoost (hand-crafted, 56 feat)",
        "feature_engineering": "heavy",
        "regret_mean": agg["regret"]["mean"],
        "regret_std": agg["regret"]["std"],
        "spearman_mean": agg["spearman"]["mean"],
        "spearman_std": agg["spearman"]["std"],
    })
    logger.info(f"  regret={agg['regret']['mean']:.4f}, rho={agg['spearman']['mean']:.4f}")

    # ── 2. Graph embeddings → XGBoost ──
    logger.info("\n=== Graph spectral embeddings → XGBoost ===")
    emb_df, _ = build_graph_embedding_features(master, n_components=32)
    if emb_df is not None:
        agg = evaluate_xgboost_on_features(emb_df, seeds, "graph-emb")
        results.append({
            "method": "XGBoost (graph embeddings)",
            "feature_engineering": "none",
            "regret_mean": agg["regret"]["mean"],
            "regret_std": agg["regret"]["std"],
            "spearman_mean": agg["spearman"]["mean"],
            "spearman_std": agg["spearman"]["std"],
        })
        logger.info(f"  regret={agg['regret']['mean']:.4f}, rho={agg['spearman']['mean']:.4f}")

    # ── 3. Auto-features → XGBoost ──
    logger.info("\n=== Auto-features (brute-force) → XGBoost ===")
    auto_df = build_auto_features(master)
    agg = evaluate_xgboost_on_features(auto_df, seeds, "auto")
    results.append({
        "method": "XGBoost (auto-features, 18 feat)",
        "feature_engineering": "minimal",
        "regret_mean": agg["regret"]["mean"],
        "regret_std": agg["regret"]["std"],
        "spearman_mean": agg["spearman"]["mean"],
        "spearman_std": agg["spearman"]["std"],
    })
    logger.info(f"  regret={agg['regret']['mean']:.4f}, rho={agg['spearman']['mean']:.4f}")

    # ── 4. GNN end-to-end ──
    logger.info("\n=== GNN end-to-end ===")
    try:
        agg = evaluate_gnn(features_df, seeds)
        results.append({
            "method": "GNN (end-to-end)",
            "feature_engineering": "none",
            "regret_mean": agg["regret"]["mean"],
            "regret_std": agg["regret"]["std"],
            "spearman_mean": agg["spearman"]["mean"],
            "spearman_std": agg["spearman"]["std"],
        })
        logger.info(f"  regret={agg['regret']['mean']:.4f}, rho={agg['spearman']['mean']:.4f}")
    except Exception as e:
        logger.warning(f"GNN failed: {e}")

    # Save
    results_df = pd.DataFrame(results)
    agg_dir = RESULTS_DIR / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(agg_dir / "exp7_zero_engineering.csv", index=False)
    logger.info(f"\nSaved to {agg_dir / 'exp7_zero_engineering.csv'}")

    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_experiment()
    print("\n=== Experiment 7: Zero Feature Engineering ===")
    for _, row in results.iterrows():
        print(f"  {row['method']:40s} [{row['feature_engineering']:8s}]: "
              f"regret={row['regret_mean']:.4f} ± {row['regret_std']:.4f}, "
              f"ρ={row['spearman_mean']:.4f} ± {row['spearman_std']:.4f}")
