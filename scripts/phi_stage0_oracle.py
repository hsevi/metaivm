"""
Stage 0 — Φ(X) oracle upper bound.

This is the *gating* experiment for the Φ(X) algorithm direction.

Question: For each hard dataset where classical clustering fails, can ANY
learned representation recover the class structure when supervised by the
dataset's OWN ground-truth labels?

This is the *upper bound* of what any meta-learned Φ approach can hope to
achieve. We are NOT testing transfer here — we are testing whether a learned
representation exists at all that fixes the broken local geometry.

Decision:
  - If even oracle-supervised Φ cannot fix these datasets → kill Φ(X) (data
    is genuinely irrecoverable from raw features).
  - If oracle-supervised Φ recovers ≥4/5 datasets → proceed to Stage 1
    (within-family transfer test).

Procedure per dataset:
  1. Load X (raw features) and y (ground-truth labels).
  2. Train a small MLP Φ: R^d → R^16 using triplet loss with cosine margin,
     anchors and positives sharing the same y, negatives drawn from other classes.
  3. Project Z = Φ(X). Run KMeans(Z, k=#classes).
  4. Compute AMI(KMeans(Z), y) and the same-class 10-NN rate in both X and Z.

Outputs:
  results/aggregated/phi_stage0_results.csv
  stdout                                      table + verdict

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -u -m scripts.phi_stage0_oracle
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
from torch.utils.data import DataLoader, TensorDataset

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

# ---------- knobs ----------
TARGET_DATASETS = [
    # Hard manifold case (Spectral and all classical fail; baseline AMI 0.113)
    ("synth_spirals_3arms_n0.05", "manifold-hard"),
    # Noisier manifold
    ("synth_spirals_3arms_n0.20", "manifold-very-hard"),
    # Text (TF-IDF reduced to 50d)
    ("text_20ng_science", "text"),
    # Image (PCA-reduced MNIST)
    ("img_mnist_digits_2k", "image"),
    # Image (PCA-reduced Fashion MNIST — visually harder than digits)
    ("img_fashion_mnist_2k", "image-hard"),
]

D_OUT = 16              # bottleneck dim
HIDDEN = (64, 32)       # MLP architecture
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 256
TRIPLET_MARGIN = 0.3    # cosine-margin for L2-normalised outputs
N_CAP = 3000            # subsample very large datasets
KNN_K = 10              # for same-class-NN diagnostic
SEED = 0


# ====================================================================
# Model
# ====================================================================
class Phi(nn.Module):
    def __init__(self, d_in: int, d_out: int = D_OUT, hidden: Tuple[int, ...] = HIDDEN):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = self.net(X)
        return F.normalize(Z, dim=-1)


# ====================================================================
# Triplet sampling
# ====================================================================
def sample_triplets(y: np.ndarray, n_triplets: int, rng: np.random.Generator) -> np.ndarray:
    """Sample triplet indices (anchor, positive, negative). Positives share
    the anchor's label; negatives are drawn from any other class."""
    classes = np.unique(y)
    by_class = {c: np.where(y == c)[0] for c in classes}
    triplets = np.empty((n_triplets, 3), dtype=np.int64)
    n = len(y)
    for i in range(n_triplets):
        anchor = int(rng.integers(0, n))
        c_anchor = int(y[anchor])
        pos_pool = by_class[c_anchor]
        if len(pos_pool) < 2:
            # Degenerate class — sample anchor itself, training will skip.
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
    """Cosine-margin triplet loss. Inputs are L2-normalised, so the dot
    product equals the cosine similarity in [-1, 1]."""
    sim_pos = (z_a * z_p).sum(-1)
    sim_neg = (z_a * z_n).sum(-1)
    return F.relu(margin + sim_neg - sim_pos).mean()


# ====================================================================
# Same-class NN diagnostic
# ====================================================================
def same_class_nn_rate(Z: np.ndarray, y: np.ndarray, k: int = KNN_K) -> float:
    n = Z.shape[0]
    if n <= k:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    # idx[:, 0] is the point itself; exclude it.
    same = (y[idx[:, 1:]] == y[:, None])
    return float(same.mean())


# ====================================================================
# Per-dataset experiment
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


def kmeans_ami(Z: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> float:
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    return float(adjusted_mutual_info_score(y, labels))


def run_one(ds_id: str, dtype: str, rng: np.random.Generator) -> Dict:
    X_raw, y = load_xy(ds_id, rng)
    n, d = X_raw.shape
    n_classes = int(len(np.unique(y)))
    logger.info(f"[{ds_id}] N={n} d={d} classes={n_classes} ({dtype})")

    # Standardize input (does not give Φ any class info; same prep as classical baselines).
    Xs = StandardScaler().fit_transform(X_raw).astype(np.float32)

    # Baseline: KMeans on raw (standardized) X with the correct k.
    raw_ami = kmeans_ami(Xs, y, n_classes)
    raw_nn = same_class_nn_rate(Xs, y)
    logger.info(f"  baseline KMeans on X    AMI={raw_ami:.4f}  same-class-NN={raw_nn:.3f}")

    # Train Φ on (X, y) — ORACLE supervision.
    torch.manual_seed(SEED)
    model = Phi(d_in=d, d_out=D_OUT, hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    X_t = torch.from_numpy(Xs)

    n_triplets_per_epoch = max(BATCH_SIZE * 4, n * 3)
    t0 = time.time()
    best_loss = float("inf")
    for ep in range(EPOCHS):
        triplets = sample_triplets(y, n_triplets_per_epoch, rng)
        a_idx = triplets[:, 0]
        p_idx = triplets[:, 1]
        n_idx = triplets[:, 2]
        order = rng.permutation(n_triplets_per_epoch)
        ep_loss = 0.0
        n_batches = 0
        for b_start in range(0, n_triplets_per_epoch, BATCH_SIZE):
            b = order[b_start:b_start + BATCH_SIZE]
            a = X_t[a_idx[b]]
            p = X_t[p_idx[b]]
            ng = X_t[n_idx[b]]
            za = model(a)
            zp = model(p)
            zn = model(ng)
            loss = triplet_loss(za, zp, zn)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        ep_loss /= max(1, n_batches)
        if ep_loss < best_loss:
            best_loss = ep_loss
        if (ep + 1) % 60 == 0:
            logger.info(f"  epoch {ep+1:3d}  loss={ep_loss:.4f}")

    # Evaluate on the trained Φ.
    model.eval()
    with torch.no_grad():
        Z = model(X_t).cpu().numpy()
    train_s = time.time() - t0

    phi_ami = kmeans_ami(Z, y, n_classes)
    phi_nn = same_class_nn_rate(Z, y)
    logger.info(f"  Φ KMeans                AMI={phi_ami:.4f}  same-class-NN={phi_nn:.3f}  ({train_s:.1f}s training)")

    gain = phi_ami - raw_ami
    return dict(
        dataset_id=ds_id, dtype=dtype, n=n, d=d, n_classes=n_classes,
        raw_ami=raw_ami, raw_nn_rate=raw_nn,
        phi_ami=phi_ami, phi_nn_rate=phi_nn,
        gain=gain, train_seconds=train_s,
    )


# ====================================================================
# Main
# ====================================================================
def main() -> None:
    rng = np.random.default_rng(SEED)

    rows: List[Dict] = []
    t_total = time.time()
    for ds_id, dtype in TARGET_DATASETS:
        try:
            r = run_one(ds_id, dtype, rng)
            rows.append(r)
        except FileNotFoundError as e:
            logger.warning(f"[{ds_id}] not found: {e}")
        except Exception as e:
            logger.exception(f"[{ds_id}] failed: {e}")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "phi_stage0_results.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  (total {(time.time()-t_total)/60:.1f}m)")

    print()
    print("=" * 92)
    print("Φ(X) STAGE 0 — ORACLE UPPER BOUND")
    print("=" * 92)
    print()
    fmt = "{:<32} {:>11} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("dataset", "dtype", "raw_AMI", "Φ_AMI", "gain", "raw_NN", "Φ_NN"))
    print("-" * 92)
    n_success = 0
    for r in rows:
        ds_label = r["dataset_id"][:30] + (".." if len(r["dataset_id"]) > 30 else "")
        verdict_marker = " ★" if r["gain"] >= 0.30 else ("  " if r["gain"] >= 0 else " ↓")
        if r["gain"] >= 0.30:
            n_success += 1
        print(fmt.format(
            ds_label,
            r["dtype"][:11],
            f"{r['raw_ami']:.4f}",
            f"{r['phi_ami']:.4f}{verdict_marker}",
            f"{r['gain']:+.4f}",
            f"{r['raw_nn_rate']:.3f}",
            f"{r['phi_nn_rate']:.3f}",
        ))
    print()
    print(f"Datasets where Φ improved AMI by ≥0.30:  {n_success}/{len(rows)}")
    print()

    # Verdict
    if n_success >= 4:
        verdict = "PROCEED to Stage 1"
        rationale = ("Oracle-supervised Φ recovers structure on ≥4/5 hard cases. The data is "
                     "recoverable from raw features by SOME learned representation. The within-family "
                     "transfer test (Stage 1) is the next step.")
    elif n_success >= 2:
        verdict = "PARTIAL — interpret carefully"
        rationale = ("Φ helps on some hard datasets but not all. Inspect failures: if failures concentrate "
                     "on specific data types, transfer is harder than expected. Worth Stage 1 with caution.")
    else:
        verdict = "KILL Φ(X) direction"
        rationale = ("Oracle-supervised Φ cannot recover structure on hard datasets even with ground-truth "
                     "labels. The data is irrecoverable from raw features. Φ(X) representation learning is "
                     "structurally limited — no amount of meta-learning will help.")

    print(f"VERDICT: {verdict}")
    print(f"  {rationale}")
    print("=" * 92)


if __name__ == "__main__":
    main()
