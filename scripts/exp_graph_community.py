"""
Experiment: MetaIVM for Graph Community Detection.

Key insight: Classical IVMs (Silhouette, CH, DB) require point coordinates
and CANNOT be applied to graph clustering. MetaIVM with graph-only features
fills this gap.

Pipeline:
1. Generate SBM and LFR graphs with planted communities
2. Run community detection algorithms (Louvain, Leiden, spectral, label prop)
3. Extract graph-only features from (graph, partition) pairs
4. Train MetaIVM to predict NMI from graph features
5. Compare against modularity and normalized cut as baselines
"""

import sys
import logging
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import community as community_louvain
import leidenalg
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from scipy.stats import spearmanr
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "aggregated"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 1. GRAPH GENERATION
# ──────────────────────────────────────────────────────────────────────

def generate_sbm_graph(sizes, p_in, p_out, seed=None):
    """Generate a Stochastic Block Model graph."""
    rng = np.random.default_rng(seed)
    k = len(sizes)
    probs = np.full((k, k), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(0, 2**31)))
    y_true = []
    for i, s in enumerate(sizes):
        y_true.extend([i] * s)
    return G, np.array(y_true)


def generate_lfr_graph(n, tau1, tau2, mu, average_degree, min_community, seed=None):
    """Generate an LFR benchmark graph."""
    try:
        G = nx.LFR_benchmark_graph(
            n, tau1, tau2, mu,
            average_degree=average_degree,
            min_community=min_community,
            max_community=max(min_community * 3, n // 3),
            seed=seed,
        )
        # Extract ground truth communities
        communities = {frozenset(G.nodes[v]["community"]) for v in G}
        y_true = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                y_true[node] = i
        return G, y_true
    except Exception as e:
        logger.warning(f"LFR generation failed: {e}")
        return None, None


def generate_all_graphs():
    """Generate diverse benchmark graphs."""
    graphs = []

    # --- SBM: vary community sizes, mixing, and number of communities ---
    sbm_configs = [
        # (name, sizes, p_in, p_out)
        # Well-separated communities
        ("sbm_3c_100_clear", [100, 100, 100], 0.3, 0.01),
        ("sbm_3c_100_moderate", [100, 100, 100], 0.2, 0.03),
        ("sbm_3c_100_mixed", [100, 100, 100], 0.15, 0.05),
        ("sbm_3c_100_hard", [100, 100, 100], 0.10, 0.06),
        # Different numbers of communities
        ("sbm_2c_200_clear", [200, 200], 0.2, 0.01),
        ("sbm_2c_200_mixed", [200, 200], 0.15, 0.05),
        ("sbm_5c_80_clear", [80, 80, 80, 80, 80], 0.25, 0.01),
        ("sbm_5c_80_moderate", [80, 80, 80, 80, 80], 0.15, 0.03),
        ("sbm_5c_80_hard", [80, 80, 80, 80, 80], 0.10, 0.05),
        ("sbm_8c_50_clear", [50]*8, 0.3, 0.01),
        ("sbm_8c_50_moderate", [50]*8, 0.2, 0.03),
        ("sbm_8c_50_hard", [50]*8, 0.12, 0.05),
        # Imbalanced communities
        ("sbm_imb_clear", [50, 100, 200], 0.25, 0.01),
        ("sbm_imb_moderate", [50, 100, 200], 0.15, 0.03),
        ("sbm_imb_hard", [50, 100, 200], 0.10, 0.05),
        ("sbm_imb_extreme", [20, 50, 100, 300], 0.2, 0.02),
        # Larger graphs
        ("sbm_3c_300_clear", [300, 300, 300], 0.15, 0.005),
        ("sbm_3c_300_moderate", [300, 300, 300], 0.10, 0.015),
        ("sbm_3c_300_hard", [300, 300, 300], 0.07, 0.025),
        ("sbm_5c_200_clear", [200]*5, 0.10, 0.005),
        ("sbm_5c_200_moderate", [200]*5, 0.07, 0.015),
        ("sbm_5c_200_hard", [200]*5, 0.05, 0.02),
        # Very small
        ("sbm_3c_30_clear", [30, 30, 30], 0.5, 0.02),
        ("sbm_3c_30_hard", [30, 30, 30], 0.3, 0.10),
        # Many small communities
        ("sbm_10c_30_clear", [30]*10, 0.4, 0.01),
        ("sbm_10c_30_moderate", [30]*10, 0.25, 0.03),
    ]

    for seed_offset in range(3):  # 3 seeds per config
        for name, sizes, p_in, p_out in sbm_configs:
            seed = 42 + seed_offset * 1000
            G, y = generate_sbm_graph(sizes, p_in, p_out, seed=seed)
            gid = f"{name}_s{seed_offset}"
            graphs.append({"graph_id": gid, "G": G, "y_true": y,
                          "source": "sbm", "n": G.number_of_nodes(),
                          "m": G.number_of_edges(),
                          "k_true": len(np.unique(y))})

    # --- LFR benchmarks ---
    lfr_configs = [
        # (name, n, tau1, tau2, mu, avg_degree, min_community)
        ("lfr_300_low_mix", 300, 2.5, 1.5, 0.1, 10, 20),
        ("lfr_300_med_mix", 300, 2.5, 1.5, 0.3, 10, 20),
        ("lfr_300_high_mix", 300, 2.5, 1.5, 0.5, 10, 20),
        ("lfr_500_low_mix", 500, 2.5, 1.5, 0.1, 15, 30),
        ("lfr_500_med_mix", 500, 2.5, 1.5, 0.3, 15, 30),
        ("lfr_500_high_mix", 500, 2.5, 1.5, 0.5, 15, 30),
        ("lfr_1000_low_mix", 1000, 2.5, 1.5, 0.1, 20, 40),
        ("lfr_1000_med_mix", 1000, 2.5, 1.5, 0.3, 20, 40),
    ]

    for seed_offset in range(3):
        for name, n, tau1, tau2, mu, avg_deg, min_comm in lfr_configs:
            seed = 100 + seed_offset * 1000
            G, y = generate_lfr_graph(n, tau1, tau2, mu, avg_deg, min_comm, seed=seed)
            if G is not None:
                gid = f"{name}_s{seed_offset}"
                graphs.append({"graph_id": gid, "G": G, "y_true": y,
                              "source": "lfr", "n": G.number_of_nodes(),
                              "m": G.number_of_edges(),
                              "k_true": len(np.unique(y))})

    logger.info(f"Generated {len(graphs)} graphs ({sum(1 for g in graphs if g['source']=='sbm')} SBM, "
                f"{sum(1 for g in graphs if g['source']=='lfr')} LFR)")
    return graphs


# ──────────────────────────────────────────────────────────────────────
# 2. COMMUNITY DETECTION ALGORITHMS
# ──────────────────────────────────────────────────────────────────────

def run_community_detection(G, y_true):
    """Run multiple community detection algorithms. Return list of (algo, params, labels, nmi)."""
    n = G.number_of_nodes()
    node_list = sorted(G.nodes())
    results = []

    # Convert to igraph for Leiden
    edges = list(G.edges())
    G_ig = ig.Graph(n=n, edges=edges, directed=False)

    # --- Louvain with different resolutions ---
    for resolution in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        try:
            partition = community_louvain.best_partition(
                G, resolution=resolution, random_state=42
            )
            labels = np.array([partition[node] for node in node_list])
            nmi = normalized_mutual_info_score(y_true, labels)
            results.append(("Louvain", {"resolution": resolution}, labels, nmi))
        except:
            pass

    # --- Leiden with different resolutions ---
    for resolution in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        try:
            part = leidenalg.find_partition(
                G_ig, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution, seed=42
            )
            labels = np.array(part.membership)
            nmi = normalized_mutual_info_score(y_true, labels)
            results.append(("Leiden", {"resolution": resolution}, labels, nmi))
        except:
            pass

    # --- Label propagation (multiple runs since non-deterministic) ---
    for seed in range(5):
        try:
            comms = nx.community.label_propagation_communities(G)
            labels = np.zeros(n, dtype=int)
            for i, comm in enumerate(comms):
                for node in comm:
                    labels[node] = i
            nmi = normalized_mutual_info_score(y_true, labels)
            results.append(("LabelProp", {"seed": seed}, labels, nmi))
        except:
            pass

    # --- Spectral clustering with different k ---
    from sklearn.cluster import SpectralClustering
    A = nx.adjacency_matrix(G).toarray()
    for k in [2, 3, 5, 8, 10]:
        if k >= n:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sc = SpectralClustering(
                    n_clusters=k, affinity="precomputed",
                    random_state=42, n_init=10
                )
                labels = sc.fit_predict(A)
                nmi = normalized_mutual_info_score(y_true, labels)
                results.append(("Spectral", {"k": k}, labels, nmi))
        except:
            pass

    # --- Greedy modularity ---
    try:
        comms = nx.community.greedy_modularity_communities(G)
        labels = np.zeros(n, dtype=int)
        for i, comm in enumerate(comms):
            for node in comm:
                labels[node] = i
        nmi = normalized_mutual_info_score(y_true, labels)
        results.append(("GreedyMod", {}, labels, nmi))
    except:
        pass

    return results


# ──────────────────────────────────────────────────────────────────────
# 3. GRAPH-ONLY FEATURES (no point coordinates needed!)
# ──────────────────────────────────────────────────────────────────────

def extract_graph_features(G, labels):
    """
    Extract features from a (graph, partition) pair using ONLY graph structure.
    No point coordinates needed — this is what makes MetaIVM applicable to
    real-world graphs where classical IVMs cannot be computed.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    node_list = sorted(G.nodes())

    unique_labels = np.unique(labels)
    k = len(unique_labels)

    # --- Partition statistics ---
    cluster_sizes = np.array([np.sum(labels == c) for c in unique_labels])
    size_probs = cluster_sizes / cluster_sizes.sum()
    entropy = -np.sum(size_probs * np.log(size_probs + 1e-12))
    gini = 1 - np.sum(size_probs ** 2)

    feat = {
        "n_clusters": k,
        "entropy": entropy,
        "gini": gini,
        "cluster_size_min": cluster_sizes.min(),
        "cluster_size_max": cluster_sizes.max(),
        "cluster_size_mean": cluster_sizes.mean(),
        "cluster_size_std": cluster_sizes.std(),
        "size_ratio": cluster_sizes.min() / (cluster_sizes.max() + 1e-12),
        "imbalance": cluster_sizes.std() / (cluster_sizes.mean() + 1e-12),
    }

    # --- Graph-level descriptors ---
    degrees = np.array([G.degree(node) for node in node_list])
    feat["graph_n_nodes"] = n
    feat["graph_n_edges"] = m
    feat["graph_density"] = 2 * m / (n * (n - 1) + 1e-12)
    feat["graph_avg_degree"] = degrees.mean()
    feat["graph_degree_std"] = degrees.std()
    feat["graph_degree_max"] = degrees.max()

    # --- Community-structure features (graph + partition) ---
    # Within-cluster edge density
    within_edges = 0
    between_edges = 0
    for u, v in G.edges():
        if labels[u] == labels[v]:
            within_edges += 1
        else:
            between_edges += 1

    feat["within_edge_fraction"] = within_edges / (m + 1e-12)
    feat["between_edge_fraction"] = between_edges / (m + 1e-12)

    # Per-cluster edge density
    cluster_densities = []
    cluster_internal_degrees = []
    for c in unique_labels:
        members = np.where(labels == c)[0]
        nc = len(members)
        member_set = set(members)
        internal = sum(1 for u in members for v in G.neighbors(node_list[u])
                      if node_list.index(v) in member_set) / 2
        max_edges = nc * (nc - 1) / 2 + 1e-12
        cluster_densities.append(internal / max_edges)
        if nc > 0:
            cluster_internal_degrees.append(2 * internal / nc)

    feat["cluster_density_mean"] = np.mean(cluster_densities)
    feat["cluster_density_std"] = np.std(cluster_densities)
    feat["cluster_density_min"] = np.min(cluster_densities)
    feat["cluster_internal_deg_mean"] = np.mean(cluster_internal_degrees)

    # Conductance per cluster
    conductances = []
    for c in unique_labels:
        members = set(np.where(labels == c)[0])
        cut_edges = sum(1 for u in members for v in G.neighbors(node_list[u])
                       if node_list.index(v) not in members)
        vol = sum(G.degree(node_list[u]) for u in members)
        vol_comp = sum(G.degree(node_list[u]) for u in range(n) if u not in members)
        denom = min(vol, vol_comp)
        conductances.append(cut_edges / (denom + 1e-12))

    feat["conductance_mean"] = np.mean(conductances)
    feat["conductance_std"] = np.std(conductances)
    feat["conductance_min"] = np.min(conductances)
    feat["conductance_max"] = np.max(conductances)

    # Modularity (the standard graph clustering "IVM")
    try:
        partition_dict = {node_list[i]: int(labels[i]) for i in range(n)}
        feat["modularity"] = community_louvain.modularity(partition_dict, G)
    except:
        feat["modularity"] = np.nan

    # Normalized cut proxy
    feat["normalized_cut"] = between_edges / (m + 1e-12)

    # Coverage: fraction of edges within communities
    feat["coverage"] = within_edges / (m + 1e-12)

    return feat


# ──────────────────────────────────────────────────────────────────────
# 4. MAIN EXPERIMENT
# ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("GRAPH COMMUNITY DETECTION EXPERIMENT")
    logger.info("=" * 60)

    # Step 1: Generate graphs
    logger.info("\n--- Generating graphs ---")
    graphs = generate_all_graphs()

    # Step 2: Run community detection on all graphs
    logger.info("\n--- Running community detection ---")
    all_rows = []
    for i, gdata in enumerate(graphs):
        gid = gdata["graph_id"]
        G = gdata["G"]
        y_true = gdata["y_true"]

        results = run_community_detection(G, y_true)
        for algo, params, labels, nmi in results:
            features = extract_graph_features(G, labels)
            row = {
                "graph_id": gid,
                "source": gdata["source"],
                "n": gdata["n"],
                "k_true": gdata["k_true"],
                "algo": algo,
                "params": str(params),
                "nmi": nmi,
            }
            row.update(features)
            all_rows.append(row)

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i+1}/{len(graphs)} graphs, {len(all_rows)} total runs")

    df = pd.DataFrame(all_rows)
    logger.info(f"\nTotal: {len(df)} runs across {df['graph_id'].nunique()} graphs")
    logger.info(f"Algorithms: {df['algo'].value_counts().to_dict()}")

    # Step 3: Evaluate — dataset-level splits
    logger.info("\n--- Evaluating MetaIVM vs baselines ---")

    feat_cols = [c for c in df.columns if c not in
                 ["graph_id", "source", "n", "k_true", "algo", "params", "nmi"]]

    graph_ids = df["graph_id"].unique().tolist()

    results_summary = []

    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(graph_ids))
        n_train = int(0.6 * len(graph_ids))
        n_val = int(0.2 * len(graph_ids))

        train_ids = set([graph_ids[i] for i in perm[:n_train]])
        val_ids = set([graph_ids[i] for i in perm[n_train:n_train + n_val]])
        test_ids = set([graph_ids[i] for i in perm[n_train + n_val:]])

        train_df = df[df["graph_id"].isin(train_ids)]
        val_df = df[df["graph_id"].isin(val_ids)]
        test_df = df[df["graph_id"].isin(test_ids)]

        X_train = np.nan_to_num(train_df[feat_cols].values.astype(np.float64))
        y_train = train_df["nmi"].values
        X_val = np.nan_to_num(val_df[feat_cols].values.astype(np.float64))
        y_val = val_df["nmi"].values

        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=20,
            random_state=seed, verbosity=0, n_jobs=1
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate per test graph
        for gid in test_ids:
            grp = test_df[test_df["graph_id"] == gid]
            if len(grp) < 2:
                continue

            true_nmi = grp["nmi"].values
            best_nmi = true_nmi.max()

            # MetaIVM prediction
            X_t = np.nan_to_num(grp[feat_cols].values.astype(np.float64))
            pred = model.predict(X_t)
            metaivm_regret = best_nmi - true_nmi[np.argmax(pred)]
            rho_meta, _ = spearmanr(true_nmi, pred)

            # Modularity baseline (use modularity to select partition)
            mod_vals = grp["modularity"].values
            mod_regret = best_nmi - true_nmi[np.argmax(np.nan_to_num(mod_vals))]
            rho_mod, _ = spearmanr(true_nmi, np.nan_to_num(mod_vals))

            # Coverage baseline
            cov_vals = grp["coverage"].values
            cov_regret = best_nmi - true_nmi[np.argmax(np.nan_to_num(cov_vals))]
            rho_cov, _ = spearmanr(true_nmi, np.nan_to_num(cov_vals))

            # Normalized cut baseline (lower is better → negate)
            ncut_vals = -grp["normalized_cut"].values
            ncut_regret = best_nmi - true_nmi[np.argmax(np.nan_to_num(ncut_vals))]
            rho_ncut, _ = spearmanr(true_nmi, np.nan_to_num(ncut_vals))

            # Conductance baseline (lower is better → negate)
            cond_vals = -grp["conductance_mean"].values
            cond_regret = best_nmi - true_nmi[np.argmax(np.nan_to_num(cond_vals))]
            rho_cond, _ = spearmanr(true_nmi, np.nan_to_num(cond_vals))

            # Random baseline
            rand_regret = best_nmi - true_nmi[np.random.RandomState(seed).randint(len(true_nmi))]

            results_summary.append({
                "seed": seed, "graph_id": gid,
                "metaivm_regret": metaivm_regret,
                "modularity_regret": mod_regret,
                "coverage_regret": cov_regret,
                "ncut_regret": ncut_regret,
                "conductance_regret": cond_regret,
                "random_regret": rand_regret,
                "metaivm_rho": 0 if np.isnan(rho_meta) else rho_meta,
                "modularity_rho": 0 if np.isnan(rho_mod) else rho_mod,
                "coverage_rho": 0 if np.isnan(rho_cov) else rho_cov,
                "ncut_rho": 0 if np.isnan(rho_ncut) else rho_ncut,
                "conductance_rho": 0 if np.isnan(rho_cond) else rho_cond,
            })

    res_df = pd.DataFrame(results_summary)

    # Print results
    print("\n" + "=" * 70)
    print("GRAPH COMMUNITY DETECTION: MODEL SELECTION RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<20s} {'Regret':>10s} {'Spearman ρ':>12s}")
    print("-" * 45)

    for method, regret_col, rho_col in [
        ("Random", "random_regret", None),
        ("Modularity", "modularity_regret", "modularity_rho"),
        ("Coverage", "coverage_regret", "coverage_rho"),
        ("Norm. Cut", "ncut_regret", "ncut_rho"),
        ("Conductance", "conductance_regret", "conductance_rho"),
        ("MetaIVM (ours)", "metaivm_regret", "metaivm_rho"),
    ]:
        r = res_df[regret_col].mean()
        r_std = res_df[regret_col].std()
        if rho_col:
            s = res_df[rho_col].mean()
            s_std = res_df[rho_col].std()
            print(f"{method:<20s} {r:>6.4f}±{r_std:.3f} {s:>8.4f}±{s_std:.3f}")
        else:
            print(f"{method:<20s} {r:>6.4f}±{r_std:.3f} {'--':>12s}")

    # Feature importance
    print("\n--- Top 10 Graph Features ---")
    model_final = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=0, verbosity=0
    )
    X_all = np.nan_to_num(df[feat_cols].values.astype(np.float64))
    y_all = df["nmi"].values
    model_final.fit(X_all, y_all)
    imp = model_final.feature_importances_
    top_idx = np.argsort(imp)[::-1][:10]
    for i in top_idx:
        print(f"  {feat_cols[i]:<35s}: {imp[i]:.4f}")

    # Save
    res_df.to_csv(RESULTS_DIR / "exp_graph_community.csv", index=False)
    df.to_csv(RESULTS_DIR / "graph_community_all_runs.csv", index=False)

    # Note about IVMs
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Classical IVMs (Silhouette, CH, DB) CANNOT be computed")
    print("on these graphs because there are no point coordinates.")
    print("MetaIVM fills this gap using graph-only features.")
    print("=" * 70)


if __name__ == "__main__":
    main()
