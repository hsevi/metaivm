"""
Stage 1.5 follow-up — does the set-based transformer Φ also fix image transfer?

Stage 1 (pointwise) on image transfer:
  digits → fashion: transfer AMI 0.252 vs raw 0.501  (HURT by 0.25)
  fashion → digits: transfer AMI 0.214 vs raw 0.408  (HURT by 0.19)

Stage 1.5 (transformer) on text: lifted AMI by +0.10 over both raw and pointwise.
This script checks whether the same architecture lesson holds for image data.

Same architecture, hyperparameters, and training protocol as
`phi_stage1_transformer.py`, with the only difference being the dataset
selection (digits/fashion image pairs instead of 20ng text folds).

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -u -m scripts.phi_stage1_transformer_image
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

# --- experiment knobs (matched to phi_stage1_transformer.py for fair comparison) ---
IMAGE_DIGITS = ["img_mnist_digits_5k", "img_mnist_digits_2k"]
IMAGE_FASHION = ["img_fashion_mnist_5k", "img_fashion_mnist_2k"]

D_HIDDEN = 64
D_OUT = 16
N_HEADS = 4
TRIPLET_MARGIN = 0.3
EPOCHS = 150
LR = 5e-4
N_TRIPLETS_PER_DATASET_PER_EPOCH = 1024
TRIPLET_BATCH = 256
N_CAP = 2000
KNN_K = 10
SEED = 0


# ====================================================================
# Architecture (identical to phi_stage1_transformer.py)
# ====================================================================
class SetPhi(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = D_HIDDEN,
                 d_out: int = D_OUT, n_heads: int = N_HEADS):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_hidden)
        self.attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.ff = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2), nn.GELU(),
            nn.Linear(d_hidden * 2, d_hidden),
        )
        self.norm2 = nn.LayerNorm(d_hidden)
        self.proj_out = nn.Linear(d_hidden, d_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.proj_in(X).unsqueeze(0)
        H_attn, _ = self.attn(H, H, H, need_weights=False)
        H = self.norm1(H + H_attn)
        H = self.norm2(H + self.ff(H))
        Z = self.proj_out(H.squeeze(0))
        return F.normalize(Z, dim=-1)


# ====================================================================
# Triplets / losses / diagnostics (matched to text script)
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


def same_class_nn_rate(Z: np.ndarray, y: np.ndarray, k: int = KNN_K) -> float:
    n = Z.shape[0]
    if n <= k:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    return float((y[idx[:, 1:]] == y[:, None]).mean())


def kmeans_ami(Z: np.ndarray, y: np.ndarray, k: int) -> float:
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    return float(adjusted_mutual_info_score(y, labels))


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
    pool = []
    for ds_id in train_ids:
        X, y = load_xy(ds_id, rng)
        Xs = StandardScaler().fit_transform(X).astype(np.float32)
        pool.append((ds_id, torch.from_numpy(Xs), y))
    return pool


def train_phi(train_pool, d_in: int, rng: np.random.Generator) -> SetPhi:
    torch.manual_seed(SEED)
    model = SetPhi(d_in=d_in)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        ds_order = rng.permutation(len(train_pool))
        ep_loss = 0.0
        for di in ds_order:
            _, X_t, y_np = train_pool[di]
            Z_full = model(X_t)
            triplets = sample_triplets(y_np, N_TRIPLETS_PER_DATASET_PER_EPOCH, rng)
            order = rng.permutation(len(triplets))
            ds_loss = 0.0
            for b_start in range(0, len(triplets), TRIPLET_BATCH):
                b = order[b_start:b_start + TRIPLET_BATCH]
                a_idx = torch.from_numpy(triplets[b, 0])
                p_idx = torch.from_numpy(triplets[b, 1])
                n_idx = torch.from_numpy(triplets[b, 2])
                za = Z_full[a_idx]
                zp = Z_full[p_idx]
                zn = Z_full[n_idx]
                ds_loss = ds_loss + triplet_loss(za, zp, zn)
            opt.zero_grad()
            ds_loss.backward()
            opt.step()
            ep_loss += float(ds_loss.item())
        if (ep + 1) % 30 == 0:
            logger.info(f"  epoch {ep+1:3d}  total loss across datasets={ep_loss:.4f}")
    return model


def eval_phi(model: SetPhi, X_test_t: torch.Tensor, y_test: np.ndarray) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        Z = model(X_test_t).cpu().numpy()
    n_classes = int(len(np.unique(y_test)))
    return kmeans_ami(Z, y_test, n_classes), same_class_nn_rate(Z, y_test)


def stage0_oracle_lookup(ds_id: str) -> float:
    p = OUT_DIR / "phi_stage0_results.csv"
    if not p.exists():
        return float("nan")
    df = pd.read_csv(p)
    sub = df[df["dataset_id"] == ds_id]
    return float(sub["phi_ami"].iloc[0]) if len(sub) else float("nan")


def stage1_pointwise_lookup(ds_id: str, experiment: str) -> float:
    p = OUT_DIR / "phi_stage1_transfer.csv"
    if not p.exists():
        return float("nan")
    df = pd.read_csv(p)
    sub = df[(df["test_id"] == ds_id) & (df["experiment"] == experiment)]
    return float(sub["transfer_ami"].iloc[0]) if len(sub) else float("nan")


# ====================================================================
# Fold runner
# ====================================================================
def run_fold(train_ids: List[str], test_id: str, label: str,
             pointwise_experiment_for_lookup: str,
             rng: np.random.Generator) -> Dict:
    logger.info(f"[{label}] train={len(train_ids)} datasets  test={test_id}")
    train_pool = prepare_train_pool(train_ids, rng)

    X_test, y_test = load_xy(test_id, rng)
    Xs_test = StandardScaler().fit_transform(X_test).astype(np.float32)
    X_test_t = torch.from_numpy(Xs_test)
    d_in = X_test_t.shape[1]
    n_classes = int(len(np.unique(y_test)))

    raw_ami = kmeans_ami(Xs_test, y_test, n_classes)
    raw_nn = same_class_nn_rate(Xs_test, y_test)
    logger.info(f"  raw KMeans            AMI={raw_ami:.4f}  NN={raw_nn:.3f}")

    torch.manual_seed(SEED)
    random_phi = SetPhi(d_in=d_in)
    rand_ami, rand_nn = eval_phi(random_phi, X_test_t, y_test)
    logger.info(f"  random SetΦ           AMI={rand_ami:.4f}  NN={rand_nn:.3f}")

    t0 = time.time()
    model = train_phi(train_pool, d_in=d_in, rng=rng)
    transfer_ami, transfer_nn = eval_phi(model, X_test_t, y_test)
    train_s = time.time() - t0
    logger.info(f"  trained SetΦ          AMI={transfer_ami:.4f}  NN={transfer_nn:.3f}  ({train_s:.1f}s)")

    oracle_ami = stage0_oracle_lookup(test_id)
    pointwise_ami = stage1_pointwise_lookup(test_id, pointwise_experiment_for_lookup)

    return dict(
        experiment=label, test_id=test_id, d=d_in, n=int(X_test.shape[0]), n_classes=n_classes,
        raw_ami=raw_ami, random_ami=rand_ami,
        pointwise_transfer_ami=pointwise_ami,
        transformer_transfer_ami=transfer_ami,
        oracle_ami=oracle_ami,
        raw_nn=raw_nn, transformer_nn=transfer_nn,
        gain_vs_raw=transfer_ami - raw_ami,
        gain_vs_pointwise=transfer_ami - pointwise_ami if not np.isnan(pointwise_ami) else float("nan"),
        train_seconds=train_s,
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: List[Dict] = []
    t_total = time.time()

    # Experiment 2 — digits → fashion
    logger.info("=== Experiment 2: image digits → fashion ===")
    for holdout in IMAGE_FASHION:
        rows.append(run_fold(IMAGE_DIGITS, holdout, "image_digits→fashion",
                             pointwise_experiment_for_lookup="image_digits→fashion", rng=rng))

    # Experiment 2b — fashion → digits
    logger.info("=== Experiment 2b: image fashion → digits ===")
    for holdout in IMAGE_DIGITS:
        rows.append(run_fold(IMAGE_FASHION, holdout, "image_fashion→digits",
                             pointwise_experiment_for_lookup="image_fashion→digits", rng=rng))

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "phi_stage1_transformer_image.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  (total {(time.time()-t_total)/60:.1f}m)")

    print()
    print("=" * 120)
    print("Φ STAGE 1.5 — SET-BASED TRANSFORMER Φ on IMAGE cross-style transfer")
    print("=" * 120)

    def print_section(title: str, sub: pd.DataFrame):
        print()
        print(f"-- {title} --")
        print()
        fmt = "{:<30} {:>9} {:>10} {:>13} {:>14} {:>10} {:>10} {:>14}"
        print(fmt.format("test_dataset", "raw_AMI", "random_Φ", "pointwise_Φ", "transformer_Φ",
                         "oracle_Φ", "Δ vs raw", "Δ vs ptwise"))
        print("-" * 120)
        for _, r in sub.iterrows():
            mark = " ★" if r["gain_vs_raw"] >= 0.20 else ("  " if r["gain_vs_raw"] >= 0 else " ↓")
            oracle = f"{r['oracle_ami']:.4f}" if not np.isnan(r["oracle_ami"]) else "    —"
            ptwise = f"{r['pointwise_transfer_ami']:.4f}" if not np.isnan(r["pointwise_transfer_ami"]) else "    —"
            delta_pt = f"{r['gain_vs_pointwise']:+.4f}" if not np.isnan(r["gain_vs_pointwise"]) else "  —"
            print(fmt.format(
                r["test_id"][:28] + (".." if len(r["test_id"]) > 28 else ""),
                f"{r['raw_ami']:.4f}",
                f"{r['random_ami']:.4f}",
                ptwise,
                f"{r['transformer_transfer_ami']:.4f}{mark}",
                oracle,
                f"{r['gain_vs_raw']:+.4f}",
                delta_pt,
            ))

    print_section("digits → fashion", df[df["experiment"] == "image_digits→fashion"])
    print_section("fashion → digits", df[df["experiment"] == "image_fashion→digits"])

    print()
    print("=" * 120)
    print("VERDICTS BY EXPERIMENT")
    print("=" * 120)
    for label in df["experiment"].unique():
        sub = df[df["experiment"] == label]
        mean_transformer = float(sub["transformer_transfer_ami"].mean())
        mean_raw = float(sub["raw_ami"].mean())
        mean_pointwise = float(sub["pointwise_transfer_ami"].mean(skipna=True))
        mean_oracle = float(sub["oracle_ami"].mean(skipna=True))
        gain_arch = mean_transformer - mean_pointwise if not np.isnan(mean_pointwise) else float("nan")
        print()
        print(f"  {label}")
        print(f"    mean(raw_AMI):                {mean_raw:.4f}")
        print(f"    mean(pointwise Φ AMI):        {mean_pointwise:.4f}   (Stage 1)")
        print(f"    mean(transformer Φ AMI):      {mean_transformer:.4f}")
        print(f"    mean(Stage 0 oracle):         {mean_oracle:.4f}")
        print(f"    gain from architecture:       {gain_arch:+.4f}")
        print(f"    gain vs raw:                  {mean_transformer - mean_raw:+.4f}")
    print("=" * 120)


if __name__ == "__main__":
    main()
