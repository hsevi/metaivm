"""
Stage 0 — Graph-Φ (GCN) curiosity check.

The pointwise MLP version (`phi_stage0_oracle.py`) recovered text and image
data completely (AMI -> 0.93+) but couldn't move the needle on spirals
(AMI stuck at ~0.13, same-class NN rate stuck at ~38%). The hypothesis: a
pointwise function cannot resolve manifold structure because the class
identity of a point depends on its NEIGHBORHOOD on the manifold, not on its
raw coordinates alone.

This script tests that hypothesis with a 2-layer graph convolutional network.
Φ now sees not just X[i] but also a weighted sum of its kNN neighbors. If
this fixes spirals, the failure was architectural (pointwise vs neighborhood-
aware), not a fundamental data limitation.

Architecture (very small):
  Â = D^{-1/2} (A + I) D^{-1/2}        # symmetric-normalized kNN adjacency
  H1 = ReLU(Â · X · W1)                 # 32 hidden
  H2 = Â · H1 · W2                      # 16 output
  Z  = normalize(H2)                    # L2-normalize

Training: identical triplet loss / sampling to the pointwise version, but the
forward pass operates on the whole dataset's graph at once each step.

Test datasets — five manifold cases of varying difficulty:
  - synth_moons_noise0.05            (easy 2D manifold)
  - synth_circles_n0.05_f0.5         (concentric rings)
  - synth_swiss_roll_k3              (3D manifold)
  - synth_spirals_3arms_n0.05        (the case the MLP couldn't fix)
  - synth_spirals_3arms_n0.20        (harder version)

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -u -m scripts.phi_stage0_gcn
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- knobs ----
TARGET_DATASETS = [
    # easy manifold cases
    ("synth_moons_noise0.05", "moons"),
    ("synth_circles_n0.05_f0.5", "circles"),
    ("synth_swiss_roll_k3", "swiss_roll"),
    # The hard cases that the MLP failed on
    ("synth_spirals_3arms_n0.05", "spirals-hard"),
    ("synth_spirals_3arms_n0.20", "spirals-very-hard"),
]

D_HIDDEN = 32
D_OUT = 16
KNN_K_GRAPH = 10        # neighbors used to build the GCN adjacency
KNN_K_DIAG = 10         # for same-class-NN diagnostic
TRIPLET_MARGIN = 0.3
EPOCHS = 300
LR = 5e-3
N_TRIPLETS_PER_EPOCH = 3000
N_CAP = 3000
SEED = 0


# ====================================================================
# Build symmetric-normalized kNN adjacency with self-loops
# ====================================================================
def build_norm_adjacency(X: np.ndarray, k: int = KNN_K_GRAPH) -> torch.Tensor:
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)  # drop self
    # Symmetrize (undirected union).
    A = np.zeros((n, n), dtype=np.float32)
    A[rows, cols] = 1.0
    A = np.maximum(A, A.T)
    # Self loops.
    np.fill_diagonal(A, 1.0)
    # Symmetric normalization: D^{-1/2} A D^{-1/2}.
    deg = A.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    A_hat = (deg_inv_sqrt[:, None] * A) * deg_inv_sqrt[None, :]
    return torch.from_numpy(A_hat)


# ====================================================================
# Model
# ====================================================================
class GraphPhi(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = D_HIDDEN, d_out: int = D_OUT):
        super().__init__()
        self.W1 = nn.Linear(d_in, d_hidden, bias=False)
        self.W2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H = F.relu(A_hat @ self.W1(X))
        H = A_hat @ self.W2(H)
        return F.normalize(H, dim=-1)


# ====================================================================
# Triplet sampling (same scheme as the MLP version)
# ====================================================================
def sample_triplets(y: np.ndarray, n_triplets: int, rng: np.random.Generator) -> np.ndarray:
    classes = np.unique(y)
    by_class = {c: np.where(y == c)[0] for c in classes}
    triplets = np.empty((n_triplets, 3), dtype=np.int64)
    n = len(y)
    for i in range(n_triplets):
        anchor = int(rng.integers(0, n))
        c_anchor = int(y[anchor])
        pos_pool = by_class[c_anchor]
        if len(pos_pool) < 2:
            pos = anchor
        else:
            pos = int(rng.choice(pos_pool))
            while pos == anchor:
                pos = int(rng.choice(pos_pool))
        other_classes = [c for c in classes if c != c_anchor]
        if not other_classes:
            neg = int(rng.integers(0, n))
        else:
            c_neg = rng.choice(other_classes)
            neg = int(rng.choice(by_class[c_neg]))
        triplets[i] = (anchor, pos, neg)
    return triplets


def triplet_loss(z_a: torch.Tensor, z_p: torch.Tensor, z_n: torch.Tensor,
                 margin: float = TRIPLET_MARGIN) -> torch.Tensor:
    sim_pos = (z_a * z_p).sum(-1)
    sim_neg = (z_a * z_n).sum(-1)
    return F.relu(margin + sim_neg - sim_pos).mean()


# ====================================================================
# Diagnostic
# ====================================================================
def same_class_nn_rate(Z: np.ndarray, y: np.ndarray, k: int = KNN_K_DIAG) -> float:
    n = Z.shape[0]
    if n <= k:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    return float((y[idx[:, 1:]] == y[:, None]).mean())


def kmeans_ami(Z: np.ndarray, y: np.ndarray, k: int) -> float:
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    return float(adjusted_mutual_info_score(y, labels))


# ====================================================================
# Per-dataset
# ====================================================================
def load_xy(ds_id: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    ds = PROCESSED_DIR / ds_id
    X = np.load(ds / "X.npy").astype(np.float64)
    y = np.load(ds / "y_true.npy").astype(np.int64)
    if X.shape[0] > N_CAP:
        idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            take = max(1, int(N_CAP * len(ci) / len(y)))
            picks = rng.choice(ci, size=min(take, len(ci)), replace=False)
            idx.extend(picks.tolist())
        idx = np.array(sorted(idx))
        X, y = X[idx], y[idx]
    return X, y


def run_one(ds_id: str, dtype: str, rng: np.random.Generator) -> Dict:
    X_raw, y = load_xy(ds_id, rng)
    n, d = X_raw.shape
    n_classes = int(len(np.unique(y)))
    logger.info(f"[{ds_id}] N={n} d={d} classes={n_classes} ({dtype})")

    Xs = StandardScaler().fit_transform(X_raw).astype(np.float32)

    raw_ami = kmeans_ami(Xs, y, n_classes)
    raw_nn = same_class_nn_rate(Xs, y)
    logger.info(f"  baseline KMeans on X     AMI={raw_ami:.4f}  same-class-NN={raw_nn:.3f}")

    # Build kNN graph adjacency from raw features.
    A_hat = build_norm_adjacency(Xs, k=KNN_K_GRAPH)
    X_t = torch.from_numpy(Xs)

    torch.manual_seed(SEED)
    model = GraphPhi(d_in=d, d_hidden=D_HIDDEN, d_out=D_OUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for ep in range(EPOCHS):
        triplets = sample_triplets(y, N_TRIPLETS_PER_EPOCH, rng)
        a_idx = torch.from_numpy(triplets[:, 0])
        p_idx = torch.from_numpy(triplets[:, 1])
        n_idx = torch.from_numpy(triplets[:, 2])
        # Single forward pass over the full graph each step (small N).
        Z = model(X_t, A_hat)
        loss = triplet_loss(Z[a_idx], Z[p_idx], Z[n_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 60 == 0:
            logger.info(f"  epoch {ep+1:3d}  loss={float(loss.item()):.4f}")

    model.eval()
    with torch.no_grad():
        Z = model(X_t, A_hat).cpu().numpy()
    train_s = time.time() - t0

    phi_ami = kmeans_ami(Z, y, n_classes)
    phi_nn = same_class_nn_rate(Z, y)
    logger.info(f"  Graph-Φ KMeans           AMI={phi_ami:.4f}  same-class-NN={phi_nn:.3f}  ({train_s:.1f}s training)")

    return dict(
        dataset_id=ds_id, dtype=dtype, n=n, d=d, n_classes=n_classes,
        raw_ami=raw_ami, raw_nn_rate=raw_nn,
        phi_ami=phi_ami, phi_nn_rate=phi_nn,
        gain=phi_ami - raw_ami, train_seconds=train_s,
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: List[Dict] = []
    t_total = time.time()
    for ds_id, dtype in TARGET_DATASETS:
        try:
            rows.append(run_one(ds_id, dtype, rng))
        except FileNotFoundError as e:
            logger.warning(f"[{ds_id}] not found: {e}")
        except Exception as e:
            logger.exception(f"[{ds_id}] failed: {e}")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "phi_stage0_gcn_results.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  (total {(time.time()-t_total)/60:.1f}m)")

    # Cross-reference against the pointwise-MLP results when available.
    mlp_path = OUT_DIR / "phi_stage0_results.csv"
    mlp_df = pd.read_csv(mlp_path) if mlp_path.exists() else None

    print()
    print("=" * 100)
    print("Φ(X) STAGE 0 — GRAPH-Φ (GCN) ON MANIFOLD DATA")
    print("=" * 100)
    print()
    fmt = "{:<30} {:<18} {:>10} {:>11} {:>11} {:>10} {:>10}"
    print(fmt.format("dataset", "dtype", "raw_AMI", "MLP_Φ_AMI", "GCN_Φ_AMI", "raw_NN", "GCN_NN"))
    print("-" * 100)
    for r in rows:
        mlp_ami = float("nan")
        if mlp_df is not None:
            sub = mlp_df[mlp_df["dataset_id"] == r["dataset_id"]]
            if len(sub) > 0:
                mlp_ami = float(sub["phi_ami"].iloc[0])
        marker = " ★" if r["gain"] >= 0.30 else ("  " if r["gain"] >= 0 else " ↓")
        print(fmt.format(
            r["dataset_id"][:28] + (".." if len(r["dataset_id"]) > 28 else ""),
            r["dtype"][:17],
            f"{r['raw_ami']:.4f}",
            f"{mlp_ami:.4f}" if not np.isnan(mlp_ami) else "    —",
            f"{r['phi_ami']:.4f}{marker}",
            f"{r['raw_nn_rate']:.3f}",
            f"{r['phi_nn_rate']:.3f}",
        ))
    print()

    # Focus on the spirals: did GCN fix what the MLP couldn't?
    spirals_rows = [r for r in rows if "spirals" in r["dataset_id"]]
    if spirals_rows:
        avg_gcn_ami_spirals = float(np.mean([r["phi_ami"] for r in spirals_rows]))
        avg_gcn_nn_spirals = float(np.mean([r["phi_nn_rate"] for r in spirals_rows]))
        print(f"Spirals — mean GCN-Φ AMI:              {avg_gcn_ami_spirals:.4f}")
        print(f"Spirals — mean same-class-NN in GCN-Z: {avg_gcn_nn_spirals:.3f}")
        print()
        if avg_gcn_ami_spirals >= 0.5:
            verdict = "GCN fixes the manifold problem"
            rationale = ("Spirals AMI went from MLP-Φ ~0.13 (effectively zero) to GCN-Φ ≥0.5. "
                         "Confirms the MLP failure was architectural — neighborhood-aware Φ "
                         "recovers what pointwise Φ cannot. Paper 3 architecture story is "
                         "now: pointwise Φ for tabular/text/image, graph-Φ for manifold.")
        elif avg_gcn_ami_spirals >= 0.25:
            verdict = "GCN partially fixes, but not fully"
            rationale = ("Substantial improvement over MLP but well short of saturation. "
                         "Could be: too few GCN layers, too few epochs, kNN graph hyperparameters "
                         "(k=10 may be too high — see Spectral probe diagnostic).")
        else:
            verdict = "GCN does not fix spirals either"
            rationale = ("Neighborhood aggregation alone is not enough. The kNN graph used to "
                         "build the adjacency is itself broken (36% same-class-NN). Φ inherits "
                         "that broken graph. Possible fix: learn the graph jointly with Φ, "
                         "or use spectral preprocessing before GCN.")
        print(f"VERDICT: {verdict}")
        print(f"  {rationale}")
    print("=" * 100)


if __name__ == "__main__":
    main()
