"""
Layer 1 Section A mini-probe — does denser hyperparameter coverage in the
*algorithm* dimension (linkages, covariance, affinities) raise the within-pool
oracle materially?

The preprocessing sub-probe (Section B) returned MARGINAL. This probe tests
the three untested Section A expansions, each chosen to target a structural
weakness in the cached pool:

  - Agglomerative with linkages {complete, average, single}: previously only
    ward was sampled. Single-linkage finds chain/manifold clusters, average and
    complete give intermediate shapes.
  - GMM with covariance {full, tied}: previously only diag was sampled. Full
    covariance models elongated/anisotropic clusters that diag cannot.
  - Spectral with affinities {rbf, knn}: previously absent. Standard pipeline
    for non-convex data.

We pick 5 datasets where each expansion has the strongest theoretical reason
to help:

  - 2 manifold datasets (single-linkage candidate)         synth_moons_*, synth_spirals_*
  - 2 ellipsoid / elongated (GMM-full candidate)           synth_ellipsoidal_*, synth_elongated_*
  - 1 text dataset (Spectral candidate)                    text_20ng_*

For each dataset we measure:
  oracle_baseline:        max AMI from a compact baseline grid that excludes
                           all untested expansions (ward-only Agglomerative,
                           diag-only GMM, no Spectral).
  oracle_baseline_plus_X: max AMI when we add expansion X.
  gain_X = oracle_baseline_plus_X - oracle_baseline.

Output:
  results/aggregated/layer1_sectionA_per_dataset.csv
  results/aggregated/layer1_sectionA_summary.csv
  stdout                                              verdict + per-expansion gains

Decision rule for the Section A direction:
  - mean(gain_best_expansion) >= 0.05  → SCALE UP. Run full Section A sweep.
  - 0.01 ≤ ... < 0.05                  → MARGINAL. Conditional / footnote only.
  - < 0.01                             → KILL. Section A is also dead.

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -m scripts.layer1_section_a_probe
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------- knobs ---------
N_CAP = 2500          # cap dataset size
K_MAX = 15            # k sweep for KMeans/GMM/Agglo
SPECTRAL_K_MAX = 12   # Spectral is O(N^3); keep k modest
SPECTRAL_N_LIMIT = 2000  # skip Spectral above this

# Datasets to test, chosen to make each expansion's case as cleanly as possible.
TARGET_DATASETS = [
    # manifold (single-linkage candidate)
    ("synth_spirals_3arms_n0.05", "manifold"),
    ("synth_moons_noise0.05", "manifold"),
    # ellipsoid / elongated (GMM-full candidate)
    ("synth_ellipsoidal_d10_k5", "ellipsoid"),
    ("synth_elongated_aspect10_3d", "ellipsoid"),
    # text (Spectral candidate)
    ("text_20ng_science", "text"),
]


# ====================================================================
# Helpers
# ====================================================================
def load_xy(dataset_id: str, rng: np.random.Generator):
    ds_dir = PROCESSED_DIR / dataset_id
    X = np.load(ds_dir / "X.npy").astype(np.float64)
    y = np.load(ds_dir / "y_true.npy").astype(np.int64)
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


def standardize(X: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


def score(amis: List[float], labels: np.ndarray, y: np.ndarray):
    from sklearn.metrics import adjusted_mutual_info_score
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return
    try:
        amis.append(float(adjusted_mutual_info_score(y, labels)))
    except Exception:
        pass


# ====================================================================
# Clustering grids — each returns a list of AMIs
# ====================================================================
def baseline_grid(Z: np.ndarray, y: np.ndarray) -> List[float]:
    """Compact baseline that excludes all three expansions."""
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    try:
        import hdbscan as hdbscan_pkg
        have_hdbscan = True
    except Exception:
        have_hdbscan = False

    n = Z.shape[0]
    amis: List[float] = []

    # KMeans
    for k in range(2, min(K_MAX, n) + 1):
        try:
            score(amis, KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(Z), y)
        except Exception:
            pass

    # GMM diag only (baseline excludes full/tied)
    for k in range(2, min(10, n) + 1):
        try:
            score(amis, GaussianMixture(
                n_components=k, covariance_type="diag", n_init=1,
                random_state=0, max_iter=200,
            ).fit_predict(Z), y)
        except Exception:
            pass

    # Agglomerative ward only (baseline excludes complete/average/single)
    if n <= 3000:
        for k in range(2, min(10, n) + 1):
            try:
                score(amis, AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z), y)
            except Exception:
                pass

    # DBSCAN modest grid
    try:
        kNN = NearestNeighbors(n_neighbors=min(10, n - 1)).fit(Z)
        d, _ = kNN.kneighbors(Z)
        med_kd = float(np.median(d[:, -1]))
        eps_lo = max(1e-3, 0.1 * med_kd)
        eps_hi = max(eps_lo * 2, 3.0 * med_kd)
        for eps in np.geomspace(eps_lo, eps_hi, num=20):
            for ms in [5, 10]:
                try:
                    score(amis, DBSCAN(eps=eps, min_samples=ms).fit_predict(Z), y)
                except Exception:
                    pass
    except Exception:
        pass

    if have_hdbscan:
        for mc in [5, 10, 20]:
            try:
                score(amis, hdbscan_pkg.HDBSCAN(min_cluster_size=mc).fit_predict(Z), y)
            except Exception:
                pass

    return amis


def agglo_extra_linkages(Z: np.ndarray, y: np.ndarray) -> List[float]:
    """Agglomerative with linkages {complete, average, single}."""
    from sklearn.cluster import AgglomerativeClustering
    n = Z.shape[0]
    amis: List[float] = []
    if n > 3000:
        return amis
    for linkage in ("complete", "average", "single"):
        for k in range(2, min(15, n) + 1):
            try:
                score(amis, AgglomerativeClustering(
                    n_clusters=k, linkage=linkage,
                ).fit_predict(Z), y)
            except Exception:
                pass
    return amis


def gmm_extra_covariance(Z: np.ndarray, y: np.ndarray) -> List[float]:
    """GMM with covariance {full, tied}."""
    from sklearn.mixture import GaussianMixture
    n = Z.shape[0]
    amis: List[float] = []
    for cov in ("full", "tied"):
        for k in range(2, min(10, n) + 1):
            try:
                score(amis, GaussianMixture(
                    n_components=k, covariance_type=cov, n_init=1,
                    random_state=0, max_iter=200, reg_covar=1e-4,
                ).fit_predict(Z), y)
            except Exception:
                pass
    return amis


def spectral_extra(Z: np.ndarray, y: np.ndarray) -> List[float]:
    """Spectral clustering with rbf and nearest-neighbors affinities."""
    from sklearn.cluster import SpectralClustering
    n = Z.shape[0]
    amis: List[float] = []
    if n > SPECTRAL_N_LIMIT:
        return amis
    for affinity in ("rbf", "nearest_neighbors"):
        for k in range(2, min(SPECTRAL_K_MAX, n) + 1):
            try:
                kw = dict(n_clusters=k, affinity=affinity, random_state=0,
                          assign_labels="kmeans")
                if affinity == "nearest_neighbors":
                    kw["n_neighbors"] = min(10, n - 1)
                score(amis, SpectralClustering(**kw).fit_predict(Z), y)
            except Exception:
                pass
    return amis


# ====================================================================
# Main loop
# ====================================================================
def run() -> None:
    rng = np.random.default_rng(0)

    rows: List[Dict] = []
    t0 = time.time()

    for ds_id, dtype in TARGET_DATASETS:
        ds_t0 = time.time()
        try:
            X, y = load_xy(ds_id, rng)
        except FileNotFoundError as e:
            logger.warning(f"[{ds_id}] missing processed data: {e}")
            continue
        Z = standardize(X)
        logger.info(f"[{ds_id}] N={X.shape[0]} d={X.shape[1]}  ({dtype})")

        # Baseline
        t = time.time()
        base = baseline_grid(Z, y)
        oracle_base = max(base) if base else float("nan")
        logger.info(f"  baseline               oracle={oracle_base:.4f}  ({time.time()-t:.1f}s, {len(base)} partitions)")

        # Agglomerative linkage expansion
        t = time.time()
        a_amis = agglo_extra_linkages(Z, y)
        oracle_with_agglo = max(base + a_amis) if (base + a_amis) else float("nan")
        gain_agglo = max(0.0, oracle_with_agglo - oracle_base)
        logger.info(f"  + Agglo linkages       oracle={oracle_with_agglo:.4f}  Δ={gain_agglo:+.4f}  ({time.time()-t:.1f}s, +{len(a_amis)} partitions)")

        # GMM covariance expansion
        t = time.time()
        g_amis = gmm_extra_covariance(Z, y)
        oracle_with_gmm = max(base + g_amis) if (base + g_amis) else float("nan")
        gain_gmm = max(0.0, oracle_with_gmm - oracle_base)
        logger.info(f"  + GMM full/tied        oracle={oracle_with_gmm:.4f}  Δ={gain_gmm:+.4f}  ({time.time()-t:.1f}s, +{len(g_amis)} partitions)")

        # Spectral expansion
        t = time.time()
        s_amis = spectral_extra(Z, y)
        oracle_with_spectral = max(base + s_amis) if (base + s_amis) else float("nan")
        gain_spectral = max(0.0, oracle_with_spectral - oracle_base) if not np.isnan(oracle_with_spectral) else float("nan")
        logger.info(f"  + Spectral rbf/knn     oracle={oracle_with_spectral:.4f}  Δ={gain_spectral:+.4f}  ({time.time()-t:.1f}s, +{len(s_amis)} partitions)")

        # Combined all three
        all_extra = a_amis + g_amis + s_amis
        oracle_full = max(base + all_extra) if (base + all_extra) else float("nan")
        gain_full = max(0.0, oracle_full - oracle_base)
        logger.info(f"  + ALL three expansions oracle={oracle_full:.4f}  Δ={gain_full:+.4f}")

        rows.append(dict(
            dataset_id=ds_id, dtype=dtype, n=int(X.shape[0]), d=int(X.shape[1]),
            oracle_baseline=oracle_base,
            oracle_plus_agglo=oracle_with_agglo, gain_agglo=gain_agglo,
            oracle_plus_gmm=oracle_with_gmm, gain_gmm=gain_gmm,
            oracle_plus_spectral=oracle_with_spectral, gain_spectral=gain_spectral,
            oracle_plus_all=oracle_full, gain_all=gain_full,
        ))
        logger.info(f"[{ds_id}] done in {time.time()-ds_t0:.1f}s (total {(time.time()-t0)/60:.1f}m)")

    df = pd.DataFrame(rows)
    per_path = OUT_DIR / "layer1_sectionA_per_dataset.csv"
    df.to_csv(per_path, index=False)
    logger.info(f"Wrote per-dataset rows: {per_path}")

    # ---------- Summary ----------
    print()
    print("=" * 86)
    print("LAYER 1 SECTION A MINI-PROBE — untested algorithm/hyperparameter expansions")
    print("=" * 86)
    print()
    fmt = "{:<32} {:>9} {:>9} {:>9} {:>9} {:>9}"
    print(fmt.format("dataset", "baseline", "+agglo", "+gmm", "+spectral", "+all"))
    print("-" * 86)
    for r in rows:
        ds_label = r["dataset_id"][:30] + (".." if len(r["dataset_id"]) > 30 else "")
        spec = f"{r['gain_spectral']:.4f}" if not np.isnan(r['gain_spectral']) else "  (skip)"
        print(fmt.format(
            ds_label,
            f"{r['oracle_baseline']:.4f}",
            f"{r['gain_agglo']:+.4f}",
            f"{r['gain_gmm']:+.4f}",
            spec,
            f"{r['gain_all']:+.4f}",
        ))
    print()

    # mean gain per expansion
    mean_agglo = float(df["gain_agglo"].mean())
    mean_gmm = float(df["gain_gmm"].mean())
    mean_spec = float(df["gain_spectral"].mean(skipna=True))
    mean_all = float(df["gain_all"].mean())
    print(f"Mean gain — Agglo linkages:        {mean_agglo:.4f}")
    print(f"Mean gain — GMM full/tied:         {mean_gmm:.4f}")
    print(f"Mean gain — Spectral rbf/knn:      {mean_spec:.4f}")
    print(f"Mean gain — ALL three combined:    {mean_all:.4f}")
    print()

    # By dtype
    print("Mean gain ALL three combined, by data type:")
    for dtype in df["dtype"].unique():
        sub = df[df["dtype"] == dtype]
        print(f"  {dtype:12s}  n={len(sub)}  mean Δoracle = {sub['gain_all'].mean():+.4f}")
    print()

    # ---------- Verdict ----------
    if mean_all >= 0.05:
        verdict = "SCALE UP"
        rationale = ("Untested expansions materially raise the within-pool oracle. "
                     "Full Section A sweep on 223 datasets is justified.")
    elif mean_all >= 0.01:
        verdict = "MARGINAL"
        rationale = ("Untested expansions help but the gain is modest. Combined with the marginal "
                     "Section B verdict, Layer 1 as a whole would be a footnote, not a paper. "
                     "Conditional application (linkages on manifold data, GMM-full on ellipsoid) "
                     "is the realistic value.")
    else:
        verdict = "KILL Layer 1"
        rationale = ("Even the previously untested expansions don't move the needle. "
                     "Section A is dead. Pool is fully saturated against classical algorithms. "
                     "Go directly to Layer 2 (Φ representation learning).")

    summary = pd.DataFrame([
        dict(expansion="agglo_linkages", mean_gain=mean_agglo),
        dict(expansion="gmm_full_tied", mean_gain=mean_gmm),
        dict(expansion="spectral_rbf_knn", mean_gain=mean_spec),
        dict(expansion="all_combined", mean_gain=mean_all),
    ])
    summary.to_csv(OUT_DIR / "layer1_sectionA_summary.csv", index=False)
    logger.info(f"Wrote summary: {OUT_DIR / 'layer1_sectionA_summary.csv'}")

    print(f"VERDICT: {verdict}")
    print(f"  {rationale}")
    print("=" * 86)


if __name__ == "__main__":
    run()
