"""
Proof-of-concept: ClusterPFN — Meta-learned end-to-end clustering.

Train a GNN on synthetic datasets to predict pairwise "same cluster?" labels.
At inference: kNN graph → GNN → edge probabilities → connected components → clusters.

This POC trains on 2000 easy synthetics and tests on real datasets.
Target: AMI > 0.5 on easy datasets (iris, wine, blobs).

Usage:
    python scripts/poc_clusterpfn.py
"""

import sys
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from torch_geometric.nn import GINConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ──────────────────────────────────────────────────────────────────────
# 1. MODEL: ClusterPFN
# ──────────────────────────────────────────────────────────────────────

class ClusterPFN(nn.Module):
    """
    GNN on kNN graph → pairwise same-cluster prediction.

    Input: node features X (n × d_max), edge_index (2 × E)
    Output: logits per edge (E,) — positive means "same cluster"
    """

    def __init__(self, input_dim=50, hidden_dim=128, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GIN layers with residual connections
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if HAS_PYG:
                self.convs.append(GINConv(mlp))
            else:
                self.convs.append(mlp)  # fallback: no message passing
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Edge prediction head
        # [h_i || h_j || |h_i - h_j| || h_i * h_j] → logit
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _normalize_features(self, X):
        """Pad or truncate features to input_dim."""
        n, d = X.shape
        if d >= self.input_dim:
            return X[:, :self.input_dim]
        else:
            pad = torch.zeros(n, self.input_dim - d, device=X.device)
            return torch.cat([X, pad], dim=1)

    def forward(self, x, edge_index, pred_edges=None):
        """
        Args:
            x: (n, d) node features
            edge_index: (2, E) kNN edges for message passing
            pred_edges: (2, E_pred) edges to predict on. If None, uses edge_index.
        Returns:
            logits: (E_pred,) raw logits per edge
        """
        x = self._normalize_features(x)
        h = self.input_proj(x)

        for conv, bn in zip(self.convs, self.bns):
            if HAS_PYG:
                h_new = conv(h, edge_index)
            else:
                # Fallback: just apply MLP without message passing
                h_new = conv(h)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_p, training=self.training)
            h = h + h_new  # residual

        # Edge predictions
        if pred_edges is None:
            pred_edges = edge_index

        h_i = h[pred_edges[0]]
        h_j = h[pred_edges[1]]
        edge_feat = torch.cat([h_i, h_j, (h_i - h_j).abs(), h_i * h_j], dim=-1)
        logits = self.edge_head(edge_feat).squeeze(-1)
        return logits


# ──────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC DATA GENERATOR
# ──────────────────────────────────────────────────────────────────────

def generate_random_dataset(rng, max_n=500, max_d=20):
    """Generate a random synthetic dataset with ground-truth labels."""
    n = rng.integers(80, max_n)
    generator = rng.choice([
        "blobs", "blobs_overlap", "moons", "circles",
        "ellipsoids", "imbalanced", "density_varying",
    ])

    if generator == "blobs":
        k = rng.integers(2, 10)
        d = rng.integers(2, max_d)
        std = rng.uniform(0.3, 1.5)
        X, y = make_blobs(n_samples=n, n_features=d, centers=k,
                          cluster_std=std, random_state=int(rng.integers(0, 2**31)))

    elif generator == "blobs_overlap":
        k = rng.integers(2, 8)
        d = rng.integers(2, max_d)
        std = rng.uniform(1.5, 4.0)
        X, y = make_blobs(n_samples=n, n_features=d, centers=k,
                          cluster_std=std, random_state=int(rng.integers(0, 2**31)))

    elif generator == "moons":
        X, y = make_moons(n_samples=n, noise=rng.uniform(0.02, 0.15),
                          random_state=int(rng.integers(0, 2**31)))

    elif generator == "circles":
        X, y = make_circles(n_samples=n, noise=rng.uniform(0.02, 0.1),
                            factor=rng.uniform(0.3, 0.7),
                            random_state=int(rng.integers(0, 2**31)))

    elif generator == "ellipsoids":
        k = rng.integers(2, 7)
        d = rng.integers(2, min(10, max_d))
        centers = rng.standard_normal((k, d)) * 4
        X_parts, y_parts = [], []
        for i in range(k):
            A = rng.standard_normal((d, d)) * 0.5
            cov = A @ A.T + np.eye(d) * 0.1
            ni = max(n // k, 10)
            Xi = rng.multivariate_normal(centers[i], cov, size=ni)
            X_parts.append(Xi)
            y_parts.append(np.full(ni, i))
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        n = len(y)

    elif generator == "imbalanced":
        k = rng.integers(2, 6)
        d = rng.integers(2, max_d)
        # Random imbalanced sizes
        alphas = rng.uniform(0.1, 2.0, size=k)
        fracs = rng.dirichlet(alphas)
        sizes = np.maximum((fracs * n).astype(int), 5)
        sizes[-1] = n - sizes[:-1].sum()
        if sizes[-1] < 5:
            sizes[-1] = 5
            n = sizes.sum()
        centers_arr = rng.standard_normal((k, d)) * 4
        X, y = make_blobs(n_samples=list(sizes), n_features=d, centers=centers_arr,
                          cluster_std=rng.uniform(0.5, 2.0),
                          random_state=int(rng.integers(0, 2**31)))

    elif generator == "density_varying":
        k = rng.integers(2, 6)
        d = rng.integers(2, min(8, max_d))
        stds = rng.uniform(0.3, 3.0, size=k)
        centers = rng.standard_normal((k, d)) * 5
        X_parts, y_parts = [], []
        for i in range(k):
            ni = max(n // k, 10)
            Xi, _ = make_blobs(n_samples=ni, n_features=d, centers=[centers[i]],
                               cluster_std=stds[i],
                               random_state=int(rng.integers(0, 2**31)))
            X_parts.append(Xi)
            y_parts.append(np.full(ni, i))
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        n = len(y)

    else:
        k = rng.integers(2, 8)
        d = rng.integers(2, max_d)
        X, y = make_blobs(n_samples=n, n_features=d, centers=k,
                          random_state=int(rng.integers(0, 2**31)))

    # Standardize
    X = StandardScaler().fit_transform(X)
    return X, y


def build_knn_graph(X, k=10):
    """Build kNN graph edge_index."""
    k_actual = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="auto")
    nn.fit(X)
    A = nn.kneighbors_graph(mode="connectivity")
    # Make symmetric
    A = A + A.T
    A[A > 1] = 1
    A_coo = A.tocoo()
    edge_index = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
    return edge_index, A


def make_training_sample(rng, max_n=500, max_d=20, knn_k=10):
    """Generate one training sample: (X_tensor, edge_index, edge_labels, neg_edges, neg_labels)."""
    X, y = generate_random_dataset(rng, max_n=max_n, max_d=max_d)
    edge_index, A = build_knn_graph(X, k=knn_k)

    # Edge labels: 1 if same cluster, 0 if different
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    edge_labels = torch.tensor((y[src] == y[dst]).astype(np.float32))

    # Sample negative edges (random non-kNN pairs)
    n = len(X)
    n_neg = min(len(src), n * knn_k)  # roughly same as positive edges
    neg_src = rng.integers(0, n, size=n_neg)
    neg_dst = rng.integers(0, n, size=n_neg)
    # Ensure they're not self-loops
    mask = neg_src != neg_dst
    neg_src, neg_dst = neg_src[mask], neg_dst[mask]
    neg_edge_index = torch.tensor(np.vstack([neg_src, neg_dst]), dtype=torch.long)
    neg_labels = torch.tensor((y[neg_src] == y[neg_dst]).astype(np.float32))

    X_tensor = torch.tensor(X, dtype=torch.float32)

    return X_tensor, edge_index, edge_labels, neg_edge_index, neg_labels


# ──────────────────────────────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────────────────────────────

def train_clusterpfn(n_episodes=2000, max_n=500, max_d=20, knn_k=10,
                     hidden_dim=128, n_layers=4, lr=1e-3):
    """Train ClusterPFN on synthetic datasets."""
    model = ClusterPFN(input_dim=50, hidden_dim=hidden_dim, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)

    rng = np.random.default_rng(42)
    model.train()

    losses = []
    accs = []
    t0 = time.time()

    for ep in range(n_episodes):
        try:
            X, edge_index, edge_labels, neg_edges, neg_labels = make_training_sample(
                rng, max_n=max_n, max_d=max_d, knn_k=knn_k
            )
        except Exception as e:
            continue

        # Combine positive (kNN) and negative (random) edges
        all_edges = torch.cat([edge_index, neg_edges], dim=1)
        all_labels = torch.cat([edge_labels, neg_labels])

        # Forward
        optimizer.zero_grad()
        logits = model(X, edge_index, pred_edges=all_edges)

        # Weighted BCE — balance positive and negative
        n_pos = all_labels.sum().item()
        n_neg_count = len(all_labels) - n_pos
        if n_pos > 0 and n_neg_count > 0:
            pos_weight = torch.tensor([n_neg_count / n_pos])
        else:
            pos_weight = torch.tensor([1.0])

        loss = F.binary_cross_entropy_with_logits(logits, all_labels, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == all_labels).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if (ep + 1) % 200 == 0:
            avg_loss = np.mean(losses[-200:])
            avg_acc = np.mean(accs[-200:])
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            logger.info(f"  [{ep+1}/{n_episodes}] loss={avg_loss:.4f} acc={avg_acc:.4f} "
                        f"({eps_per_sec:.1f} ep/s, {elapsed:.0f}s elapsed)")

    total_time = time.time() - t0
    logger.info(f"Training done: {n_episodes} episodes in {total_time:.0f}s")
    return model


# ──────────────────────────────────────────────────────────────────────
# 4. INFERENCE
# ──────────────────────────────────────────────────────────────────────

def predict_clusters(model, X, knn_k=10, threshold=0.5):
    """
    Predict clusters for a new dataset using ClusterPFN.

    Args:
        model: trained ClusterPFN
        X: (n, d) numpy array
        knn_k: k for kNN graph
        threshold: probability threshold for "same cluster"

    Returns:
        labels: (n,) cluster assignments
        n_clusters: number of clusters found
    """
    model.eval()
    n = len(X)

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    edge_index, A = build_knn_graph(X_scaled, k=knn_k)

    with torch.no_grad():
        logits = model(X_tensor, edge_index)
        probs = torch.sigmoid(logits).numpy()

    # Build affinity graph: keep edges with prob > threshold
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    mask = probs > threshold
    affinity = csr_matrix(
        (np.ones(mask.sum()), (src[mask], dst[mask])),
        shape=(n, n)
    )
    # Make symmetric
    affinity = affinity + affinity.T
    affinity[affinity > 1] = 1

    # Connected components → cluster labels
    n_components, labels = connected_components(affinity, directed=False)

    return labels, n_components


# ──────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────────────────────────────

def evaluate_on_real_datasets(model):
    """Evaluate on real datasets from the benchmark."""
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    results = []

    # Easy test datasets
    test_datasets = [
        "openml_61",      # iris: 150x4, 3 classes
        "synth_grid_d2_k3_lowoverlap",
        "synth_grid_d5_k3_lowoverlap",
        "synth_ellipsoidal_d3_k3",
        "synth_grid_d2_k6_lowoverlap",
        "synth_s_curve_k3",
        "synth_swiss_roll_k3",
        "synth_mixed_shapes_2d",
        "img_mnist_digits_2k",
        "text_20ng_science",
    ]

    # Also try medium difficulty
    medium_datasets = [
        "openml_1466",    # cardiotocography: 2126x35, 10 classes
        "openml_36",      # segment: 2310x18, 7 classes
        "openml_60",      # waveform: 5000x40, 3 classes
        "synth_grid_d5_k6_medoverlap",
        "synth_imb_moderate_d5",
    ]

    for ds_id in test_datasets + medium_datasets:
        proc_dir = PROCESSED_DIR / ds_id
        if not (proc_dir / "X.npy").exists():
            continue

        X = np.load(proc_dir / "X.npy")
        y_true = np.load(proc_dir / "y_true.npy")
        n, d = X.shape
        k_true = len(np.unique(y_true))

        if n > 5000:
            # Subsample for speed
            idx = np.random.default_rng(42).choice(n, 3000, replace=False)
            X, y_true = X[idx], y_true[idx]
            n = 3000

        # ClusterPFN prediction
        labels_pred, k_pred = predict_clusters(model, X)
        ami = adjusted_mutual_info_score(y_true, labels_pred)
        nmi = normalized_mutual_info_score(y_true, labels_pred)

        # KMeans baseline (oracle k)
        from sklearn.cluster import KMeans
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km_labels = KMeans(n_clusters=k_true, n_init=10, random_state=42).fit_predict(
                StandardScaler().fit_transform(X)
            )
            km_ami = adjusted_mutual_info_score(y_true, km_labels)

        difficulty = "easy" if ds_id in test_datasets else "medium"
        results.append({
            "dataset": ds_id, "n": n, "d": d, "k_true": k_true,
            "k_pred": k_pred, "ami": ami, "nmi": nmi,
            "kmeans_ami": km_ami, "difficulty": difficulty,
        })
        logger.info(f"  {ds_id}: AMI={ami:.3f} (k_pred={k_pred}, k_true={k_true}) | KMeans={km_ami:.3f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("PROOF OF CONCEPT: ClusterPFN")
    logger.info("=" * 60)

    # Train
    logger.info("\n--- Training on 2000 synthetic datasets ---")
    model = train_clusterpfn(
        n_episodes=2000,
        max_n=500,
        max_d=20,
        knn_k=10,
        hidden_dim=128,
        n_layers=4,
        lr=1e-3,
    )

    # Save model
    save_path = PROJECT_ROOT / "models" / "clusterpfn_poc.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    # Evaluate on synthetic validation
    logger.info("\n--- Evaluating on synthetic validation ---")
    rng = np.random.default_rng(9999)
    synth_amis = []
    for i in range(20):
        X, y = generate_random_dataset(rng, max_n=300, max_d=10)
        labels, k = predict_clusters(model, X)
        ami = adjusted_mutual_info_score(y, labels)
        synth_amis.append(ami)
    logger.info(f"Synthetic val AMI: {np.mean(synth_amis):.3f} ± {np.std(synth_amis):.3f}")

    # Evaluate on real datasets
    logger.info("\n--- Evaluating on real datasets ---")
    results = evaluate_on_real_datasets(model)

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print("CLUSTERPFN PROOF-OF-CONCEPT RESULTS")
        print("=" * 70)
        print(f"\n{'Dataset':<35s} {'AMI':>6s} {'KMeans':>7s} {'k_pred':>6s} {'k_true':>6s}")
        print("-" * 65)
        for _, row in df.iterrows():
            marker = "✓" if row["ami"] > 0.5 else "✗"
            print(f"{row['dataset']:<35s} {row['ami']:>6.3f} {row['kmeans_ami']:>7.3f} "
                  f"{row['k_pred']:>6d} {row['k_true']:>6d}  {marker}")

        print(f"\n--- Summary ---")
        easy = df[df["difficulty"] == "easy"]
        med = df[df["difficulty"] == "medium"]
        if len(easy) > 0:
            print(f"Easy datasets:   mean AMI = {easy['ami'].mean():.3f} (KMeans: {easy['kmeans_ami'].mean():.3f})")
        if len(med) > 0:
            print(f"Medium datasets: mean AMI = {med['ami'].mean():.3f} (KMeans: {med['kmeans_ami'].mean():.3f})")
        print(f"All datasets:    mean AMI = {df['ami'].mean():.3f} (KMeans: {df['kmeans_ami'].mean():.3f})")

        # GO/NO-GO
        easy_ami = easy["ami"].mean() if len(easy) > 0 else 0
        print(f"\n{'='*70}")
        if easy_ami > 0.5:
            print(f"GO: Easy AMI = {easy_ami:.3f} > 0.5 threshold. ClusterPFN works!")
            print(f"Proceed to full training (Step 4).")
        elif easy_ami > 0.3:
            print(f"MARGINAL: Easy AMI = {easy_ami:.3f}. Needs more training data or tuning.")
        else:
            print(f"NO-GO: Easy AMI = {easy_ami:.3f} < 0.3. Architecture needs rethinking.")
        print(f"{'='*70}")

        # Save results
        df.to_csv(PROJECT_ROOT / "results" / "aggregated" / "poc_clusterpfn.csv", index=False)


if __name__ == "__main__":
    main()
