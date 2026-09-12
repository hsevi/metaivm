"""
Experiment: MetaIVM for Attributed Graph Community Detection.

Attributed graphs have both graph structure AND node features.
Classical IVMs can only use node features (ignoring graph), while
MetaIVM can combine both — giving it a unique advantage.

Datasets: Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers,
          Coauthor-CS, Coauthor-Physics
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import community as community_louvain
import leidenalg
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "aggregated"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 1. LOAD ATTRIBUTED GRAPH DATASETS
# ──────────────────────────────────────────────────────────────────────

def load_attributed_graphs():
    """Load standard attributed graph benchmarks via PyG."""
    import torch
    from torch_geometric.datasets import Planetoid, Amazon, Coauthor

    cache_dir = PROJECT_ROOT / "data" / "pyg_cache"
    graphs = []

    # Planetoid: Cora, CiteSeer, PubMed
    for name in ["Cora", "CiteSeer", "PubMed"]:
        try:
            dataset = Planetoid(root=str(cache_dir), name=name)
            data = dataset[0]
            X = data.x.numpy()
            y = data.y.numpy()
            edge_index = data.edge_index.numpy()

            G = nx.Graph()
            G.add_nodes_from(range(X.shape[0]))
            G.add_edges_from(zip(edge_index[0], edge_index[1]))
            # Remove self-loops
            G.remove_edges_from(nx.selfloop_edges(G))

            graphs.append({
                "graph_id": f"pyg_{name.lower()}",
                "name": name,
                "G": G, "X": X, "y_true": y,
                "n": X.shape[0], "d": X.shape[1],
                "m": G.number_of_edges(),
                "k_true": len(np.unique(y)),
                "source": "planetoid",
            })
            logger.info(f"  Loaded {name}: n={X.shape[0]}, d={X.shape[1]}, "
                       f"m={G.number_of_edges()}, k={len(np.unique(y))}")
        except Exception as e:
            logger.warning(f"  Failed to load {name}: {e}")

    # Amazon: Photo, Computers
    for name in ["Photo", "Computers"]:
        try:
            dataset = Amazon(root=str(cache_dir), name=name)
            data = dataset[0]
            X = data.x.numpy()
            y = data.y.numpy()
            edge_index = data.edge_index.numpy()

            # Subsample if too large (keep connected component)
            max_n = 3000
            if X.shape[0] > max_n:
                rng = np.random.default_rng(42)
                idx = rng.choice(X.shape[0], max_n, replace=False)
                idx_set = set(idx)
                idx_map = {old: new for new, old in enumerate(idx)}
                X = X[idx]
                y = y[idx]
                G = nx.Graph()
                G.add_nodes_from(range(len(idx)))
                for i in range(edge_index.shape[1]):
                    u, v = int(edge_index[0, i]), int(edge_index[1, i])
                    if u in idx_set and v in idx_set and u != v:
                        G.add_edge(idx_map[u], idx_map[v])
            else:
                G = nx.Graph()
                G.add_nodes_from(range(X.shape[0]))
                G.add_edges_from(zip(edge_index[0], edge_index[1]))
                G.remove_edges_from(nx.selfloop_edges(G))

            graphs.append({
                "graph_id": f"pyg_amazon_{name.lower()}",
                "name": f"Amazon-{name}",
                "G": G, "X": X, "y_true": y,
                "n": X.shape[0], "d": X.shape[1],
                "m": G.number_of_edges(),
                "k_true": len(np.unique(y)),
                "source": "amazon",
            })
            logger.info(f"  Loaded Amazon-{name}: n={X.shape[0]}, d={X.shape[1]}, "
                       f"m={G.number_of_edges()}, k={len(np.unique(y))}")
        except Exception as e:
            logger.warning(f"  Failed to load Amazon-{name}: {e}")

    # Coauthor: CS, Physics
    for name in ["CS", "Physics"]:
        try:
            dataset = Coauthor(root=str(cache_dir), name=name)
            data = dataset[0]
            X = data.x.numpy()
            y = data.y.numpy()
            edge_index = data.edge_index.numpy()

            max_n = 3000
            if X.shape[0] > max_n:
                rng = np.random.default_rng(42)
                idx = rng.choice(X.shape[0], max_n, replace=False)
                idx_set = set(idx)
                idx_map = {old: new for new, old in enumerate(idx)}
                X = X[idx]
                y = y[idx]
                G = nx.Graph()
                G.add_nodes_from(range(len(idx)))
                for i in range(edge_index.shape[1]):
                    u, v = int(edge_index[0, i]), int(edge_index[1, i])
                    if u in idx_set and v in idx_set and u != v:
                        G.add_edge(idx_map[u], idx_map[v])
            else:
                G = nx.Graph()
                G.add_nodes_from(range(X.shape[0]))
                G.add_edges_from(zip(edge_index[0], edge_index[1]))
                G.remove_edges_from(nx.selfloop_edges(G))

            graphs.append({
                "graph_id": f"pyg_coauthor_{name.lower()}",
                "name": f"Coauthor-{name}",
                "G": G, "X": X, "y_true": y,
                "n": X.shape[0], "d": X.shape[1],
                "m": G.number_of_edges(),
                "k_true": len(np.unique(y)),
                "source": "coauthor",
            })
            logger.info(f"  Loaded Coauthor-{name}: n={X.shape[0]}, d={X.shape[1]}, "
                       f"m={G.number_of_edges()}, k={len(np.unique(y))}")
        except Exception as e:
            logger.warning(f"  Failed to load Coauthor-{name}: {e}")

    logger.info(f"Loaded {len(graphs)} attributed graph datasets")
    return graphs


# ──────────────────────────────────────────────────────────────────────
# 2. COMMUNITY DETECTION ALGORITHMS
# ──────────────────────────────────────────────────────────────────────

def run_community_detection(G, X, y_true, n_nodes):
    """Run multiple community detection algorithms on attributed graph."""
    results = []
    node_list = sorted(G.nodes())
    n = len(node_list)

    # Convert for igraph
    edges = [(u, v) for u, v in G.edges() if u < n and v < n]
    G_ig = ig.Graph(n=n, edges=edges, directed=False)

    # --- Graph-based: Louvain ---
    for res in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        try:
            partition = community_louvain.best_partition(G, resolution=res, random_state=42)
            labels = np.array([partition.get(node, 0) for node in node_list])
            nmi = normalized_mutual_info_score(y_true[:n], labels)
            results.append(("Louvain", {"resolution": res}, labels, nmi))
        except:
            pass

    # --- Graph-based: Leiden ---
    for res in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        try:
            part = leidenalg.find_partition(
                G_ig, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=res, seed=42
            )
            labels = np.array(part.membership)
            nmi = normalized_mutual_info_score(y_true[:n], labels)
            results.append(("Leiden", {"resolution": res}, labels, nmi))
        except:
            pass

    # --- Label propagation ---
    for seed in range(3):
        try:
            comms = nx.community.label_propagation_communities(G)
            labels = np.zeros(n, dtype=int)
            for i, comm in enumerate(comms):
                for node in comm:
                    if node < n:
                        labels[node] = i
            nmi = normalized_mutual_info_score(y_true[:n], labels)
            results.append(("LabelProp", {"seed": seed}, labels, nmi))
        except:
            pass

    # --- KMeans on node attributes (PCA-reduced) ---
    X_use = X[:n]
    if X_use.shape[1] > 50:
        X_pca = PCA(n_components=50, random_state=42).fit_transform(X_use)
    else:
        X_pca = X_use.copy()
    X_scaled = StandardScaler().fit_transform(X_pca)

    for k in [2, 3, 5, 7, 10, 15, 20]:
        if k >= n:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(X_scaled)
                nmi = normalized_mutual_info_score(y_true[:n], labels)
                results.append(("KMeans_attr", {"k": k}, labels, nmi))
        except:
            pass

    # --- Spectral on graph adjacency (small graphs only) ---
    if n <= 3000:
        try:
            A = nx.adjacency_matrix(G).toarray()[:n, :n]
            for k in [2, 3, 5, 7, 10]:
                if k >= n:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                               random_state=42, n_init=10)
                        labels = sc.fit_predict(A)
                        nmi = normalized_mutual_info_score(y_true[:n], labels)
                        results.append(("Spectral_graph", {"k": k}, labels, nmi))
                except:
                    pass
            del A
            import gc; gc.collect()
        except:
            pass

    # --- Greedy modularity ---
    try:
        comms = nx.community.greedy_modularity_communities(G)
        labels = np.zeros(n, dtype=int)
        for i, comm in enumerate(comms):
            for node in comm:
                if node < n:
                    labels[node] = i
        nmi = normalized_mutual_info_score(y_true[:n], labels)
        results.append(("GreedyMod", {}, labels, nmi))
    except:
        pass

    return results


# ──────────────────────────────────────────────────────────────────────
# 3. FEATURES: GRAPH-ONLY, ATTRIBUTE-ONLY, AND COMBINED
# ──────────────────────────────────────────────────────────────────────

def extract_graph_features(G, labels, n):
    """Graph topology features only."""
    node_list = sorted(G.nodes())[:n]
    m = G.number_of_edges()
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    cluster_sizes = np.array([np.sum(labels == c) for c in unique_labels])
    size_probs = cluster_sizes / (cluster_sizes.sum() + 1e-12)

    feat = {
        "g_n_clusters": k,
        "g_entropy": -np.sum(size_probs * np.log(size_probs + 1e-12)),
        "g_gini": 1 - np.sum(size_probs ** 2),
        "g_size_ratio": cluster_sizes.min() / (cluster_sizes.max() + 1e-12),
        "g_imbalance": cluster_sizes.std() / (cluster_sizes.mean() + 1e-12),
        "g_n_nodes": n,
        "g_n_edges": m,
        "g_density": 2 * m / (n * (n - 1) + 1e-12),
    }

    degrees = np.array([G.degree(node) for node in node_list])
    feat["g_avg_degree"] = degrees.mean()
    feat["g_degree_std"] = degrees.std()

    within_edges = 0
    between_edges = 0
    for u, v in G.edges():
        if u < n and v < n:
            if labels[u] == labels[v]:
                within_edges += 1
            else:
                between_edges += 1
    feat["g_within_edge_frac"] = within_edges / (m + 1e-12)

    # Modularity
    try:
        partition_dict = {node_list[i]: int(labels[i]) for i in range(n)}
        feat["g_modularity"] = community_louvain.modularity(partition_dict, G)
    except:
        feat["g_modularity"] = np.nan

    # Per-cluster conductance
    conductances = []
    for c in unique_labels:
        members = set(np.where(labels == c)[0])
        cut_edges = sum(1 for u in members if u < n
                       for v in G.neighbors(node_list[u])
                       if node_list.index(v) not in members)
        vol = sum(G.degree(node_list[u]) for u in members if u < n)
        vol_comp = sum(G.degree(node_list[u]) for u in range(n) if u not in members)
        denom = min(vol, vol_comp) + 1e-12
        conductances.append(cut_edges / denom)
    feat["g_conductance_mean"] = np.mean(conductances)
    feat["g_conductance_min"] = np.min(conductances)

    return feat


def extract_attribute_features(X, labels):
    """Node attribute features — what classical IVMs would use."""
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    feat = {}

    # Classical IVMs on attributes
    if k >= 2 and k < len(X):
        try:
            # Subsample for silhouette if large
            if len(X) > 5000:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(X), 5000, replace=False)
                feat["a_silhouette"] = silhouette_score(X[idx], labels[idx])
            else:
                feat["a_silhouette"] = silhouette_score(X, labels)
        except:
            feat["a_silhouette"] = np.nan
        try:
            feat["a_calinski_harabasz"] = calinski_harabasz_score(X, labels)
        except:
            feat["a_calinski_harabasz"] = np.nan
        try:
            feat["a_davies_bouldin"] = davies_bouldin_score(X, labels)
        except:
            feat["a_davies_bouldin"] = np.nan
    else:
        feat["a_silhouette"] = np.nan
        feat["a_calinski_harabasz"] = np.nan
        feat["a_davies_bouldin"] = np.nan

    # Attribute-based cluster stats
    within_var = []
    for c in unique_labels:
        mask = labels == c
        if mask.sum() > 1:
            within_var.append(np.var(X[mask], axis=0).mean())
    feat["a_within_var_mean"] = np.mean(within_var) if within_var else np.nan
    feat["a_between_var"] = np.var([X[labels == c].mean(axis=0) for c in unique_labels], axis=0).mean()

    # Dataset-level attribute stats
    feat["a_n_features"] = X.shape[1]
    feat["a_attr_mean_std"] = np.std(X, axis=0).mean()
    feat["a_attr_sparsity"] = (X == 0).mean()

    return feat


# ──────────────────────────────────────────────────────────────────────
# 4. MAIN EXPERIMENT
# ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("ATTRIBUTED GRAPH COMMUNITY DETECTION EXPERIMENT")
    logger.info("=" * 60)

    # Load datasets
    logger.info("\n--- Loading attributed graph datasets ---")
    graphs = load_attributed_graphs()

    # Run community detection
    logger.info("\n--- Running community detection ---")
    all_rows = []
    for gdata in graphs:
        gid = gdata["graph_id"]
        G, X, y = gdata["G"], gdata["X"], gdata["y_true"]
        n = min(G.number_of_nodes(), len(X), len(y))

        results = run_community_detection(G, X, y, n)
        logger.info(f"  {gdata['name']}: {len(results)} partitions generated")

        for algo, params, labels, nmi in results:
            labels = labels[:n]
            g_feat = extract_graph_features(G, labels, n)
            a_feat = extract_attribute_features(X[:n], labels)

            row = {
                "graph_id": gid, "name": gdata["name"],
                "source": gdata["source"],
                "n": n, "k_true": gdata["k_true"],
                "algo": algo, "params": str(params), "nmi": nmi,
            }
            row.update(g_feat)
            row.update(a_feat)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    logger.info(f"\nTotal: {len(df)} runs across {df['graph_id'].nunique()} datasets")
    logger.info(f"NMI range: {df['nmi'].min():.3f} - {df['nmi'].max():.3f}")

    # Define feature sets
    g_cols = [c for c in df.columns if c.startswith("g_")]
    a_cols = [c for c in df.columns if c.startswith("a_")]
    all_feat_cols = g_cols + a_cols

    logger.info(f"Graph features: {len(g_cols)}, Attribute features: {len(a_cols)}")

    # Evaluate with leave-one-dataset-out (since we only have 7 datasets)
    graph_ids = df["graph_id"].unique().tolist()

    print("\n" + "=" * 80)
    print("ATTRIBUTED GRAPH COMMUNITY DETECTION RESULTS (Leave-one-out)")
    print("=" * 80)

    methods_results = {}

    for feat_name, feat_cols_used in [
        ("Graph-only", g_cols),
        ("Attribute-only", a_cols),
        ("Graph + Attribute", all_feat_cols),
    ]:
        all_regrets = []
        all_rhos = []

        for test_gid in graph_ids:
            train_df = df[df["graph_id"] != test_gid]
            test_df_g = df[df["graph_id"] == test_gid]

            if len(test_df_g) < 2:
                continue

            X_train = np.nan_to_num(train_df[feat_cols_used].values.astype(np.float64))
            y_train = train_df["nmi"].values
            X_test = np.nan_to_num(test_df_g[feat_cols_used].values.astype(np.float64))
            y_test = test_df_g["nmi"].values

            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            best_nmi = y_test.max()
            regret = best_nmi - y_test[np.argmax(pred)]
            rho, _ = spearmanr(y_test, pred)
            if np.isnan(rho):
                rho = 0.0

            all_regrets.append(regret)
            all_rhos.append(rho)

        methods_results[feat_name] = (np.mean(all_regrets), np.std(all_regrets),
                                       np.mean(all_rhos), np.std(all_rhos))

    # Classical IVM baselines (attribute-based)
    for ivm_name, ivm_col, higher_better in [
        ("Silhouette (attr)", "a_silhouette", True),
        ("CH (attr)", "a_calinski_harabasz", True),
        ("DB (attr)", "a_davies_bouldin", False),
    ]:
        all_regrets = []
        all_rhos = []
        for gid in graph_ids:
            grp = df[df["graph_id"] == gid]
            if len(grp) < 2:
                continue
            y_test = grp["nmi"].values
            ivm_vals = grp[ivm_col].values.copy()
            if not higher_better:
                ivm_vals = -ivm_vals
            ivm_vals = np.nan_to_num(ivm_vals)
            best_nmi = y_test.max()
            regret = best_nmi - y_test[np.argmax(ivm_vals)]
            rho, _ = spearmanr(y_test, ivm_vals)
            if np.isnan(rho):
                rho = 0.0
            all_regrets.append(regret)
            all_rhos.append(rho)
        methods_results[ivm_name] = (np.mean(all_regrets), np.std(all_regrets),
                                      np.mean(all_rhos), np.std(all_rhos))

    # Graph heuristic baselines
    for base_name, base_col, higher_better in [
        ("Modularity", "g_modularity", True),
        ("Conductance", "g_conductance_mean", False),
    ]:
        all_regrets = []
        all_rhos = []
        for gid in graph_ids:
            grp = df[df["graph_id"] == gid]
            if len(grp) < 2:
                continue
            y_test = grp["nmi"].values
            vals = grp[base_col].values.copy()
            if not higher_better:
                vals = -vals
            vals = np.nan_to_num(vals)
            best_nmi = y_test.max()
            regret = best_nmi - y_test[np.argmax(vals)]
            rho, _ = spearmanr(y_test, vals)
            if np.isnan(rho):
                rho = 0.0
            all_regrets.append(regret)
            all_rhos.append(rho)
        methods_results[base_name] = (np.mean(all_regrets), np.std(all_regrets),
                                       np.mean(all_rhos), np.std(all_rhos))

    # Random baseline
    rng = np.random.default_rng(42)
    rand_regrets = []
    for gid in graph_ids:
        grp = df[df["graph_id"] == gid]
        if len(grp) < 2:
            continue
        y_test = grp["nmi"].values
        rand_regrets.append(y_test.max() - y_test[rng.integers(len(y_test))])
    methods_results["Random"] = (np.mean(rand_regrets), np.std(rand_regrets), 0, 0)

    # Print results
    print(f"\n{'Method':<30s} {'Regret':>12s} {'Spearman ρ':>12s}")
    print("-" * 56)
    order = ["Random", "Silhouette (attr)", "CH (attr)", "DB (attr)",
             "Modularity", "Conductance",
             "Graph-only", "Attribute-only", "Graph + Attribute"]
    for method in order:
        if method in methods_results:
            r_m, r_s, s_m, s_s = methods_results[method]
            if method == "Random":
                print(f"{method:<30s} {r_m:>6.4f}±{r_s:.3f} {'--':>12s}")
            else:
                print(f"{method:<30s} {r_m:>6.4f}±{r_s:.3f} {s_m:>6.4f}±{s_s:.3f}")

    # Per-dataset breakdown
    print(f"\n{'Dataset':<25s} {'n':>6s} {'k':>4s} {'Runs':>5s} {'Best NMI':>9s} {'Worst NMI':>10s}")
    print("-" * 62)
    for gid in graph_ids:
        grp = df[df["graph_id"] == gid]
        gdata = [g for g in graphs if g["graph_id"] == gid][0]
        print(f"{gdata['name']:<25s} {gdata['n']:>6d} {gdata['k_true']:>4d} "
              f"{len(grp):>5d} {grp['nmi'].max():>9.3f} {grp['nmi'].min():>10.3f}")

    # Feature importance for combined model
    print("\n--- Top Features (Graph + Attribute model) ---")
    X_all = np.nan_to_num(df[all_feat_cols].values.astype(np.float64))
    y_all = df["nmi"].values
    model_full = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   random_state=42, verbosity=0)
    model_full.fit(X_all, y_all)
    imp = model_full.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    for i in top_idx:
        source = "GRAPH" if all_feat_cols[i].startswith("g_") else "ATTR"
        print(f"  [{source}] {all_feat_cols[i]:<30s}: {imp[i]:.4f}")

    # Save
    df.to_csv(RESULTS_DIR / "attributed_graph_all_runs.csv", index=False)

    print("\n" + "=" * 80)
    print("KEY: Classical IVMs use ONLY attributes (ignore graph structure).")
    print("MetaIVM (Graph+Attr) uses BOTH, giving it a unique advantage.")
    print("MetaIVM (Graph-only) works even WITHOUT attributes.")
    print("=" * 80)


if __name__ == "__main__":
    main()
