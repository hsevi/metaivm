"""
Layer 1 preprocessing sub-probe.

Question we are answering: does applying preprocessing transforms (StandardScaler,
PCA, Kernel PCA, UMAP) to a dataset's raw X *before* running the existing
clustering algorithms expand the reachable partition pool enough to materially
raise the within-pool oracle AMI?

Method (no MetaIVM scoring; oracle-AMI only):
  - Pick a stratified sample of ~20 test datasets (balanced across sources).
  - For each dataset:
      * Load raw X and ground-truth labels y_true.
      * Read cached oracle AMI on the existing raw-X pool (max AMI in runs_master).
      * For each transform T in {StandardScaler, PCA(0.9), PCA(0.95),
        KernelPCA-RBF, UMAP(8D), UMAP(16D)}:
          - Compute Z = T(X).
          - Re-run a compact classical clustering grid on Z (KMeans k∈[2..k_max],
            GMM-diag k∈[2..15], Agglomerative-ward k∈[2..15], DBSCAN over a
            log-eps grid, HDBSCAN).
          - Compute AMI(predicted, y_true) for every partition that is
            non-trivial (k>=2, no all-noise).
          - oracle_T = max of these AMIs.
      * Per-transform gain = oracle_T - oracle_raw.

Output:
  results/aggregated/layer1_probe_per_dataset.csv  - rows (dataset, transform, oracle, gain)
  results/aggregated/layer1_probe_summary.csv      - aggregate by (transform, source)
  stdout                                           - decision verdict

Decision rule (mean gain across all sampled datasets):
  - >= 0.05 -> Full Layer 1 sweep justified.
  -  0.01..0.05 -> Marginal; consider whether the writeup justifies the compute.
  - < 0.01 -> Kill Layer 1. Pool is saturated, go directly to Layer 2.

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -m scripts.layer1_preprocessing_probe
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_MASTER = DATA_DIR / "features" / "runs_master.csv"
REGISTRY = DATA_DIR / "dataset_registry.csv"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- experiment knobs (Option B: compact but informative, ~60 min target) -----
SEED = 0
N_PER_SOURCE = {"synthetic": 5, "openml": 5, "text": 2, "image": 3}  # 15 total
N_CAP = 2500          # cap dataset size for the probe to keep wall-clock bounded
K_MAX = 15            # max k for KMeans / GMM / Agglo sweep (was 20)
DBSCAN_EPS_GRID = 20  # log-grid points (was 30)
DBSCAN_MIN_SAMPLES = [5, 10]
HDBSCAN_MIN_CLUSTER = [5, 10, 20]
UMAP_NEIGHBORS = 15
KPCA_N_COMPONENTS = 16

# transforms to evaluate (Option B menu: 5 transforms incl. raw baseline)
# Dropped from the full menu: kernel_pca_rbf and umap_8d (both deferred to the
# full Layer 1 sweep if this probe returns positive).
TRANSFORMS = [
    "raw",                # baseline (no preprocessing)
    "standard_scaler",
    "pca_90",
    "pca_95",
    "umap_16d",
]


# ====================================================================
# Stratified dataset sampling
# ====================================================================
def sample_datasets(rng: np.random.Generator) -> pd.DataFrame:
    reg = pd.read_csv(REGISTRY)
    runs = pd.read_csv(RUNS_MASTER, usecols=["dataset_id", "ami"])
    valid_ids = set(runs["dataset_id"].unique())
    reg = reg[reg["dataset_id"].isin(valid_ids)]
    sampled = []
    for src, n in N_PER_SOURCE.items():
        pool = reg[reg["source"] == src]
        # Prefer smaller datasets so the probe stays fast.
        pool = pool.sort_values("n_samples")
        # Keep dataset diversity by stratifying within the source.
        if len(pool) <= n:
            picks = pool
        else:
            # Take the smallest 75% by n_samples then random sample within that.
            small = pool.head(int(0.75 * len(pool)) or len(pool))
            picks = small.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        sampled.append(picks)
    out = pd.concat(sampled, ignore_index=True)
    logger.info(
        f"Sampled {len(out)} datasets: "
        + ", ".join(f"{k}={(out['source']==k).sum()}" for k in N_PER_SOURCE)
    )
    return out


# ====================================================================
# Oracle on the existing raw-X pool (from cached AMIs)
# ====================================================================
def cached_oracle(runs_master: pd.DataFrame, dataset_id: str) -> float:
    sub = runs_master[runs_master["dataset_id"] == dataset_id]
    if len(sub) == 0:
        return float("nan")
    return float(sub["ami"].max())


# ====================================================================
# Load X, y_true with optional subsample
# ====================================================================
def load_xy(dataset_id: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    ds_dir = PROCESSED_DIR / dataset_id
    X = np.load(ds_dir / "X.npy").astype(np.float64)
    y = np.load(ds_dir / "y_true.npy").astype(np.int64)
    if X.shape[0] > N_CAP:
        # Stratified by class so we don't kill rare classes.
        idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            take = max(1, int(N_CAP * len(ci) / len(y)))
            picks = rng.choice(ci, size=min(take, len(ci)), replace=False)
            idx.extend(picks.tolist())
        idx = np.array(sorted(idx))
        X, y = X[idx], y[idx]
    return X, y


# ====================================================================
# Transforms
# ====================================================================
def apply_transform(name: str, X: np.ndarray, rng: np.random.Generator):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, KernelPCA

    if name == "raw":
        return X
    if name == "standard_scaler":
        return StandardScaler().fit_transform(X)
    if name == "pca_90":
        n = min(X.shape[1], max(2, int(0.9 * X.shape[1])))
        return PCA(n_components=0.9, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    if name == "pca_95":
        return PCA(n_components=0.95, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    if name == "kernel_pca_rbf":
        n_components = min(KPCA_N_COMPONENTS, X.shape[1])
        # Memory cost is O(N^2). N_CAP keeps this manageable.
        return KernelPCA(
            n_components=n_components, kernel="rbf",
            gamma=1.0 / max(1, X.shape[1]), random_state=0,
        ).fit_transform(StandardScaler().fit_transform(X))
    if name.startswith("umap_"):
        import umap
        d_out = int(name.split("_")[1].rstrip("d"))
        n_neighbors = min(UMAP_NEIGHBORS, max(2, X.shape[0] - 1))
        return umap.UMAP(
            n_components=d_out,
            n_neighbors=n_neighbors,
            random_state=0,
            init="random",  # avoid spectral init failures on small / degenerate inputs
        ).fit_transform(StandardScaler().fit_transform(X))
    raise ValueError(f"unknown transform: {name}")


# ====================================================================
# Compact clustering grid + AMI scoring
# ====================================================================
def cluster_and_score(Z: np.ndarray, y: np.ndarray) -> List[float]:
    """Run a compact classical grid on Z; return AMI of every non-trivial partition."""
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_mutual_info_score
    try:
        import hdbscan as hdbscan_pkg
        have_hdbscan = True
    except Exception:
        have_hdbscan = False

    n = Z.shape[0]
    amis: List[float] = []

    def score(labels: np.ndarray):
        labels = np.asarray(labels)
        # Treat DBSCAN noise (-1) as its own cluster for AMI computation.
        if len(np.unique(labels)) < 2:
            return
        try:
            amis.append(adjusted_mutual_info_score(y, labels))
        except Exception:
            pass

    # KMeans
    for k in range(2, min(K_MAX, n) + 1):
        try:
            score(KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(Z))
        except Exception:
            pass

    # GMM diag
    for k in range(2, min(15, n) + 1):
        try:
            score(GaussianMixture(
                n_components=k, covariance_type="diag", n_init=1,
                random_state=0, max_iter=200,
            ).fit_predict(Z))
        except Exception:
            pass

    # Agglomerative ward (skip if n is huge — already capped to N_CAP)
    if n <= 3000:
        for k in range(2, min(15, n) + 1):
            try:
                score(AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z))
            except Exception:
                pass

    # DBSCAN over log-eps grid
    # Use nearest-neighbor distances to set a sensible eps range.
    try:
        from sklearn.neighbors import NearestNeighbors
        kNN = NearestNeighbors(n_neighbors=min(10, n - 1)).fit(Z)
        d, _ = kNN.kneighbors(Z)
        med_kd = float(np.median(d[:, -1]))
        eps_lo = max(1e-3, 0.1 * med_kd)
        eps_hi = max(eps_lo * 2, 3.0 * med_kd)
        for eps in np.geomspace(eps_lo, eps_hi, num=DBSCAN_EPS_GRID):
            for ms in DBSCAN_MIN_SAMPLES:
                try:
                    score(DBSCAN(eps=eps, min_samples=ms).fit_predict(Z))
                except Exception:
                    pass
    except Exception:
        pass

    # HDBSCAN
    if have_hdbscan:
        for mc in HDBSCAN_MIN_CLUSTER:
            try:
                score(hdbscan_pkg.HDBSCAN(min_cluster_size=mc).fit_predict(Z))
            except Exception:
                pass

    return amis


# ====================================================================
# Main loop
# ====================================================================
def run() -> None:
    rng = np.random.default_rng(SEED)
    sample = sample_datasets(rng)

    runs_master = pd.read_csv(RUNS_MASTER, usecols=["dataset_id", "ami"])

    rows: List[Dict] = []
    t0 = time.time()

    for i, row in sample.iterrows():
        ds = row["dataset_id"]
        src = row["source"]
        ds_t0 = time.time()
        try:
            X, y = load_xy(ds, rng)
        except FileNotFoundError as e:
            logger.warning(f"[{ds}] missing processed data: {e}")
            continue

        oracle_raw_cached = cached_oracle(runs_master, ds)

        per_transform: Dict[str, float] = {}
        for tname in TRANSFORMS:
            tt0 = time.time()
            try:
                Z = apply_transform(tname, X, rng)
                amis = cluster_and_score(Z, y)
                if not amis:
                    per_transform[tname] = float("nan")
                else:
                    per_transform[tname] = max(amis)
            except Exception as e:
                logger.warning(f"[{ds}] transform {tname} failed: {e}")
                per_transform[tname] = float("nan")
            logger.info(
                f"  [{ds}] {tname:18s} oracle={per_transform[tname]:.4f} "
                f"({time.time()-tt0:.1f}s)"
            )

        oracle_raw_new = per_transform.get("raw", float("nan"))
        for tname, oracle_t in per_transform.items():
            gain_vs_cached = oracle_t - oracle_raw_cached if not np.isnan(oracle_t) else float("nan")
            gain_vs_new_raw = oracle_t - oracle_raw_new if not (np.isnan(oracle_t) or np.isnan(oracle_raw_new)) else float("nan")
            rows.append(dict(
                dataset_id=ds,
                source=src,
                transform=tname,
                oracle_t=oracle_t,
                oracle_raw_cached=oracle_raw_cached,
                oracle_raw_new=oracle_raw_new,
                gain_vs_cached=gain_vs_cached,
                gain_vs_new_raw=gain_vs_new_raw,
                n_samples=int(X.shape[0]),
                n_features=int(X.shape[1]),
            ))
        logger.info(f"[{ds}] done in {time.time()-ds_t0:.1f}s  (total so far: {(time.time()-t0)/60:.1f}m)")

    df = pd.DataFrame(rows)
    per_path = OUT_DIR / "layer1_probe_per_dataset.csv"
    df.to_csv(per_path, index=False)
    logger.info(f"Wrote per-dataset rows: {per_path}")

    # ---------- Summary by (transform, source) ----------
    summary_rows = []
    for tname in TRANSFORMS:
        sub_all = df[df["transform"] == tname]
        for slice_name, sub in [("all", sub_all),
                                ("synthetic", sub_all[sub_all["source"] == "synthetic"]),
                                ("openml", sub_all[sub_all["source"] == "openml"]),
                                ("text+image", sub_all[sub_all["source"].isin(["text", "image"])])]:
            if len(sub) == 0:
                continue
            summary_rows.append(dict(
                transform=tname,
                slice=slice_name,
                n=len(sub),
                mean_oracle=float(sub["oracle_t"].mean(skipna=True)),
                mean_gain_vs_cached=float(sub["gain_vs_cached"].mean(skipna=True)),
                mean_gain_vs_new_raw=float(sub["gain_vs_new_raw"].mean(skipna=True)),
                pct_datasets_with_positive_gain=float((sub["gain_vs_cached"] > 0).mean()),
            ))
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "layer1_probe_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote summary: {summary_path}")

    # ---------- Pretty print + verdict ----------
    print()
    print("=" * 86)
    print("LAYER 1 PREPROCESSING SUB-PROBE")
    print("=" * 86)
    print()
    print(f"Sampled {df['dataset_id'].nunique()} datasets; "
          f"{df.groupby('source')['dataset_id'].nunique().to_dict()}")
    print()
    print("Per-transform gain (vs cached raw-X pool oracle):")
    print()
    fmt = "{:<20} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("transform", "all_gain", "synth", "openml", "text+img"))
    print("-" * 86)
    for tname in TRANSFORMS:
        get = lambda s: summary[
            (summary["transform"] == tname) & (summary["slice"] == s)
        ]["mean_gain_vs_cached"]
        a = get("all")
        s = get("synthetic")
        o = get("openml")
        ti = get("text+image")
        print(fmt.format(
            tname,
            f"{a.iloc[0]:.4f}" if len(a) else "  -  ",
            f"{s.iloc[0]:.4f}" if len(s) else "  -  ",
            f"{o.iloc[0]:.4f}" if len(o) else "  -  ",
            f"{ti.iloc[0]:.4f}" if len(ti) else "  -  ",
        ))
    print()

    # Mean over all non-raw transforms — that's the "best preprocessing helps" signal.
    best_per_dataset = df[df["transform"] != "raw"].groupby("dataset_id")["gain_vs_cached"].max()
    mean_best = float(best_per_dataset.mean(skipna=True))
    pct_help = float((best_per_dataset > 0.01).mean())

    print(f"Best-transform-per-dataset mean gain (vs cached raw oracle):  {mean_best:.4f}")
    print(f"Fraction of datasets where any transform gains >0.01 AMI:     {pct_help:.1%}")
    print()

    if mean_best >= 0.05:
        verdict = "SCALE UP"
        rationale = ("Preprocessing transforms close a meaningful fraction of within-pool gap. "
                     "Full Layer 1 sweep is justified.")
    elif mean_best >= 0.01:
        verdict = "MARGINAL"
        rationale = ("Gains exist but are modest. Layer 1 would be a footnote, not a paper section. "
                     "Decide based on writeup ambition; arguably skip to Layer 2.")
    else:
        verdict = "KILL Layer 1"
        rationale = ("Pool is already saturated against preprocessing. Skip to Layer 2 (representation "
                     "learning) for the algorithm direction.")

    print(f"VERDICT: {verdict}")
    print(f"  {rationale}")
    print("=" * 86)


if __name__ == "__main__":
    run()
