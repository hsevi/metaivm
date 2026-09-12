"""
Stage 1 — Does Φ trained jointly across many datasets TRANSFER?

This is the test that ClusterPFN previously failed: train Φ on training-fold
datasets, then deploy it without any per-dataset retraining on a held-out
dataset. If transfer Φ matches the oracle Φ (Stage 0), the meta-learning
direction is alive. If transfer Φ collapses to the raw baseline, Φ overfits
per-dataset and the Φ(X) paper has the same failure mode as ClusterPFN.

Three transfer experiments, all using the same architecture and loss as
Stage 0 (pointwise MLP + triplet loss), differing only in train/test split:

  (1) Text LOO   — 5 folds. Train Φ on 4 text datasets, test on 1.
                    All 5 text datasets are 50d (TF-IDF + SVD).
  (2) Image cross-style — Train Φ on the digits datasets, test on the fashion
                    datasets. Within-image-modality, across visual semantics
                    (handwritten digits vs clothing).
  (3) Text -> image cross-modality (stretch) — Train Φ on text, test on image.
                    Both modalities are 50d but the feature spaces are totally
                    different in structure. Probably fails; informative either way.

For each fold we report:
  baseline_KM   AMI(KMeans(X_test, k=true_k))         on raw test features
  transfer_KM   AMI(KMeans(Φ_trained(X_test), k))     using transfer Φ
  random_KM     AMI(KMeans(Φ_random(X_test), k))      sanity baseline (random Φ)
  oracle_KM     (looked up from Stage 0 CSV when available)
  raw_NN_rate / Φ_NN_rate                              local-geometry diagnostic

Outputs:
  results/aggregated/phi_stage1_transfer.csv
  stdout                                              tables + verdict

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -u -m scripts.phi_stage1_transfer
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
STAGE0_RESULTS = OUT_DIR / "phi_stage0_results.csv"

# ---- architecture: identical to Stage 0 ----
D_OUT = 16
HIDDEN = (64, 32)
TRIPLET_MARGIN = 0.3
EPOCHS = 300
LR = 1e-3
N_TRIPLETS_PER_DATASET_PER_EPOCH = 1024
BATCH_SIZE = 256
N_CAP = 3000
KNN_K = 10
SEED = 0

TEXT_DATASETS = [
    "text_20ng_all",
    "text_20ng_science",
    "text_20ng_politics",
    "text_20ng_comp",
    "text_20ng_rec",
]
IMAGE_DIGITS = ["img_mnist_digits_5k", "img_mnist_digits_2k"]
IMAGE_FASHION = ["img_fashion_mnist_5k", "img_fashion_mnist_2k"]


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
# Triplet sampling (within a single dataset)
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
# Diagnostics
# ====================================================================
def same_class_nn_rate(Z: np.ndarray, y: np.ndarray, k: int = KNN_K) -> float:
    n = Z.shape[0]
    if n <= k:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    return float((y[idx[:, 1:]] == y[:, None]).mean())


def kmeans_ami(Z: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> float:
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    return float(adjusted_mutual_info_score(y, labels))


# ====================================================================
# Data
# ====================================================================
def load_xy(ds_id: str, rng: np.random.Generator):
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


def prepare_train_pool(train_ids: List[str], rng: np.random.Generator):
    """Load every training dataset, standardize per-dataset, return list of (X_t, y_np)."""
    pool = []
    for ds_id in train_ids:
        X, y = load_xy(ds_id, rng)
        Xs = StandardScaler().fit_transform(X).astype(np.float32)
        pool.append((ds_id, torch.from_numpy(Xs), y))
    return pool


# ====================================================================
# Train Φ jointly across multiple training datasets
# ====================================================================
def train_phi(train_pool, d_in: int, rng: np.random.Generator) -> Phi:
    torch.manual_seed(SEED)
    model = Phi(d_in=d_in)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        # Iterate over training datasets, sample triplets within each.
        ds_order = rng.permutation(len(train_pool))
        ep_loss = 0.0
        ep_batches = 0
        for di in ds_order:
            _, X_t, y_np = train_pool[di]
            triplets = sample_triplets(y_np, N_TRIPLETS_PER_DATASET_PER_EPOCH, rng)
            order = rng.permutation(len(triplets))
            for b_start in range(0, len(triplets), BATCH_SIZE):
                b = order[b_start:b_start + BATCH_SIZE]
                a_idx = triplets[b, 0]
                p_idx = triplets[b, 1]
                n_idx = triplets[b, 2]
                za = model(X_t[a_idx])
                zp = model(X_t[p_idx])
                zn = model(X_t[n_idx])
                loss = triplet_loss(za, zp, zn)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                ep_batches += 1
        if (ep + 1) % 60 == 0:
            logger.info(f"  epoch {ep+1:3d}  mean loss={ep_loss/max(1, ep_batches):.4f}")
    return model


# ====================================================================
# Evaluate Φ (trained or random) on a held-out test dataset
# ====================================================================
def eval_phi(model: Phi, X_test_t: torch.Tensor, y_test: np.ndarray) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        Z = model(X_test_t).cpu().numpy()
    n_classes = int(len(np.unique(y_test)))
    ami = kmeans_ami(Z, y_test, n_classes)
    nn_rate = same_class_nn_rate(Z, y_test)
    return ami, nn_rate


# ====================================================================
# Look up Stage 0 oracle if available
# ====================================================================
def stage0_oracle_lookup(ds_id: str) -> float:
    if not STAGE0_RESULTS.exists():
        return float("nan")
    df = pd.read_csv(STAGE0_RESULTS)
    sub = df[df["dataset_id"] == ds_id]
    if len(sub) == 0:
        return float("nan")
    return float(sub["phi_ami"].iloc[0])


# ====================================================================
# One transfer fold
# ====================================================================
def run_fold(train_ids: List[str], test_id: str, label: str, rng: np.random.Generator) -> Dict:
    logger.info(f"[{label}] train={len(train_ids)} datasets  test={test_id}")
    train_pool = prepare_train_pool(train_ids, rng)

    X_test, y_test = load_xy(test_id, rng)
    Xs_test = StandardScaler().fit_transform(X_test).astype(np.float32)
    X_test_t = torch.from_numpy(Xs_test)
    d_in = X_test_t.shape[1]
    n_classes = int(len(np.unique(y_test)))

    raw_ami = kmeans_ami(Xs_test, y_test, n_classes)
    raw_nn = same_class_nn_rate(Xs_test, y_test)
    logger.info(f"  raw      AMI={raw_ami:.4f}  NN={raw_nn:.3f}")

    # Random-init baseline (untrained Φ).
    torch.manual_seed(SEED)
    random_phi = Phi(d_in=d_in)
    random_ami, random_nn = eval_phi(random_phi, X_test_t, y_test)
    logger.info(f"  random Φ AMI={random_ami:.4f}  NN={random_nn:.3f}")

    # Train transfer Φ.
    t0 = time.time()
    model = train_phi(train_pool, d_in=d_in, rng=rng)
    transfer_ami, transfer_nn = eval_phi(model, X_test_t, y_test)
    logger.info(f"  transfer Φ AMI={transfer_ami:.4f}  NN={transfer_nn:.3f}  ({time.time()-t0:.1f}s training)")

    oracle_ami = stage0_oracle_lookup(test_id)

    return dict(
        experiment=label, test_id=test_id, n_train_datasets=len(train_ids),
        n_test=int(X_test.shape[0]), d=d_in, n_classes=n_classes,
        raw_ami=raw_ami, random_ami=random_ami,
        transfer_ami=transfer_ami, oracle_ami=oracle_ami,
        raw_nn=raw_nn, transfer_nn=transfer_nn,
        transfer_minus_raw=transfer_ami - raw_ami,
        transfer_minus_random=transfer_ami - random_ami,
        oracle_minus_transfer=(oracle_ami - transfer_ami) if not np.isnan(oracle_ami) else float("nan"),
    )


# ====================================================================
# Main
# ====================================================================
def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: List[Dict] = []
    t_total = time.time()

    # ----- Experiment 1: Text LOO -----
    logger.info("=== Experiment 1: Text LOO (5 folds) ===")
    for holdout in TEXT_DATASETS:
        train_ids = [d for d in TEXT_DATASETS if d != holdout]
        rows.append(run_fold(train_ids, holdout, "text_loo", rng))

    # ----- Experiment 2: Image cross-style (digits -> fashion) -----
    logger.info("=== Experiment 2: Image cross-style (digits → fashion) ===")
    for holdout in IMAGE_FASHION:
        rows.append(run_fold(IMAGE_DIGITS, holdout, "image_digits→fashion", rng))

    # And the reverse, for completeness.
    logger.info("=== Experiment 2b: Image cross-style (fashion → digits) ===")
    for holdout in IMAGE_DIGITS:
        rows.append(run_fold(IMAGE_FASHION, holdout, "image_fashion→digits", rng))

    # ----- Experiment 3: cross-modality (text → image, stretch) -----
    logger.info("=== Experiment 3: Cross-modality text → image (stretch) ===")
    for holdout in IMAGE_DIGITS + IMAGE_FASHION:
        rows.append(run_fold(TEXT_DATASETS, holdout, "text→image", rng))

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "phi_stage1_transfer.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  (total {(time.time()-t_total)/60:.1f}m)")

    # ============================================================
    # Print summary
    # ============================================================
    print()
    print("=" * 112)
    print("Φ(X) STAGE 1 — DOES Φ TRANSFER ACROSS DATASETS?")
    print("=" * 112)

    def print_section(title: str, sub: pd.DataFrame):
        print()
        print(f"-- {title} --")
        print()
        fmt = "{:<30} {:>10} {:>10} {:>12} {:>10} {:>14} {:>14}"
        print(fmt.format("test_dataset", "raw_AMI", "random_Φ", "transfer_Φ", "oracle_Φ",
                         "Δ vs raw", "Δ vs random"))
        print("-" * 112)
        for _, r in sub.iterrows():
            mark = " ★" if r["transfer_minus_raw"] >= 0.20 else ("  " if r["transfer_minus_raw"] >= 0 else " ↓")
            oracle = f"{r['oracle_ami']:.4f}" if not np.isnan(r["oracle_ami"]) else "    —"
            print(fmt.format(
                r["test_id"][:28] + (".." if len(r["test_id"]) > 28 else ""),
                f"{r['raw_ami']:.4f}",
                f"{r['random_ami']:.4f}",
                f"{r['transfer_ami']:.4f}{mark}",
                oracle,
                f"{r['transfer_minus_raw']:+.4f}",
                f"{r['transfer_minus_random']:+.4f}",
            ))

    print_section("Experiment 1: TEXT LOO (5 folds)", df[df["experiment"] == "text_loo"])
    print_section("Experiment 2: IMAGE digits → fashion", df[df["experiment"] == "image_digits→fashion"])
    print_section("Experiment 2b: IMAGE fashion → digits", df[df["experiment"] == "image_fashion→digits"])
    print_section("Experiment 3: TEXT → IMAGE (cross-modality)", df[df["experiment"] == "text→image"])

    # ============================================================
    # Per-experiment verdict
    # ============================================================
    print()
    print("=" * 112)
    print("VERDICTS BY EXPERIMENT")
    print("=" * 112)
    for label in df["experiment"].unique():
        sub = df[df["experiment"] == label]
        mean_transfer_gain = float(sub["transfer_minus_raw"].mean())
        mean_oracle = float(sub["oracle_ami"].mean(skipna=True))
        mean_transfer = float(sub["transfer_ami"].mean())
        if not np.isnan(mean_oracle) and mean_oracle > 0:
            oracle_recovery = mean_transfer / mean_oracle
        else:
            oracle_recovery = float("nan")
        print()
        print(f"  {label}")
        print(f"    mean(transfer Φ AMI - raw):     {mean_transfer_gain:+.4f}")
        print(f"    mean(transfer Φ AMI):           {mean_transfer:.4f}")
        if not np.isnan(mean_oracle):
            print(f"    mean(Stage 0 oracle AMI):       {mean_oracle:.4f}")
            print(f"    transfer / oracle ratio:        {oracle_recovery:.2%}")
    print("=" * 112)


if __name__ == "__main__":
    main()
