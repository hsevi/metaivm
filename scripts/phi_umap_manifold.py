"""
Does the standard single-cell pipeline (UMAP -> {KMeans, Spectral}) recover
the manifold datasets that pointwise-Φ and graph-Φ couldn't fix?

The hypothesis from the GCN probe: GCN failed on spirals because its input
kNN graph was already broken. UMAP refines the kNN structure using a global
manifold-learning objective; if it cleans up the local geometry enough,
downstream clustering on UMAP coordinates should succeed.

If UMAP + KMeans works on spirals_3arms_n0.05, the "spirals problem" is NOT
beyond existing tools — it just needs the right preprocessing (the same one
single-cell genomics uses).

If UMAP fails too, then spirals at this noise level is genuinely beyond
standard methods.

Test on 5 manifold datasets:
  - synth_moons_noise0.05            (easy)
  - synth_circles_n0.05_f0.5         (concentric, centroid-pathological for KMeans)
  - synth_swiss_roll_k3              (3D manifold)
  - synth_spirals_3arms_n0.05        (the canonical hard case)
  - synth_spirals_3arms_n0.20        (harder)

For each: try UMAP at {2D, 8D, 16D} x {n_neighbors=10, 30} then KMeans(true_k)
and Spectral(true_k). Report the best AMI across this UMAP-side grid.

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -u -m scripts.phi_umap_manifold
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
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DATASETS = [
    ("synth_moons_noise0.05", "moons"),
    ("synth_circles_n0.05_f0.5", "circles"),
    ("synth_swiss_roll_k3", "swiss_roll"),
    ("synth_spirals_3arms_n0.05", "spirals-hard"),
    ("synth_spirals_3arms_n0.20", "spirals-very-hard"),
]

UMAP_DIMS = [2, 8, 16]
UMAP_N_NEIGHBORS = [10, 30]
SEED = 0


def kmeans_ami(Z: np.ndarray, y: np.ndarray, k: int) -> float:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    return float(adjusted_mutual_info_score(y, labels))


def spectral_ami(Z: np.ndarray, y: np.ndarray, k: int) -> float:
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_mutual_info_score
    n = Z.shape[0]
    n_neighbors = min(10, n - 1)
    try:
        labels = SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors",
            n_neighbors=n_neighbors, random_state=0,
            assign_labels="kmeans",
        ).fit_predict(Z)
        from sklearn.metrics import adjusted_mutual_info_score
        return float(adjusted_mutual_info_score(y, labels))
    except Exception:
        return float("nan")


def run_one(ds_id: str, dtype: str) -> Dict:
    from sklearn.preprocessing import StandardScaler
    import umap
    ds = PROCESSED_DIR / ds_id
    X = np.load(ds / "X.npy").astype(np.float64)
    y = np.load(ds / "y_true.npy").astype(np.int64)
    n_classes = int(len(np.unique(y)))
    Xs = StandardScaler().fit_transform(X)

    raw_km_ami = kmeans_ami(Xs, y, n_classes)
    raw_sp_ami = spectral_ami(Xs, y, n_classes)
    logger.info(f"[{ds_id}] N={X.shape[0]} d={X.shape[1]} classes={n_classes}  ({dtype})")
    logger.info(f"  raw KMeans AMI={raw_km_ami:.4f}   raw Spectral AMI={raw_sp_ami:.4f}")

    best_km = -1.0
    best_sp = -1.0
    best_km_cfg = None
    best_sp_cfg = None
    for d_out in UMAP_DIMS:
        for nn in UMAP_N_NEIGHBORS:
            t = time.time()
            try:
                Z = umap.UMAP(
                    n_components=d_out, n_neighbors=nn,
                    random_state=SEED, init="random",
                ).fit_transform(Xs)
            except Exception as e:
                logger.warning(f"  UMAP failed at d={d_out} n_neighbors={nn}: {e}")
                continue
            km_ami = kmeans_ami(Z, y, n_classes)
            sp_ami = spectral_ami(Z, y, n_classes)
            if km_ami > best_km:
                best_km = km_ami
                best_km_cfg = f"UMAP({d_out}D, n_nbrs={nn})"
            if sp_ami > best_sp:
                best_sp = sp_ami
                best_sp_cfg = f"UMAP({d_out}D, n_nbrs={nn})"
            logger.info(f"  UMAP d={d_out} nn={nn}  KM={km_ami:.4f}  SP={sp_ami:.4f}  ({time.time()-t:.1f}s)")

    return dict(
        dataset_id=ds_id, dtype=dtype,
        n=int(X.shape[0]), d=int(X.shape[1]), n_classes=n_classes,
        raw_km=raw_km_ami, raw_sp=raw_sp_ami,
        best_umap_km=best_km, best_umap_km_cfg=best_km_cfg,
        best_umap_sp=best_sp, best_umap_sp_cfg=best_sp_cfg,
        gain_km=best_km - raw_km_ami,
        gain_sp=best_sp - raw_sp_ami,
    )


def main() -> None:
    rows: List[Dict] = []
    t_total = time.time()
    for ds_id, dtype in TARGET_DATASETS:
        try:
            rows.append(run_one(ds_id, dtype))
        except FileNotFoundError as e:
            logger.warning(f"[{ds_id}] not found: {e}")
        except Exception as e:
            logger.exception(f"[{ds_id}] failed: {e}")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "phi_umap_manifold_results.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  (total {(time.time()-t_total)/60:.1f}m)")

    print()
    print("=" * 110)
    print("UMAP + {KMeans, Spectral} ON MANIFOLD DATA — does the standard pipeline fix the spirals?")
    print("=" * 110)
    print()
    fmt = "{:<32} {:<18} {:>9} {:>9} {:>11} {:>11} {:>9} {:>9}"
    print(fmt.format("dataset", "dtype", "raw_KM", "raw_SP", "UMAP+KM", "UMAP+SP", "ΔKM", "ΔSP"))
    print("-" * 110)
    for r in rows:
        km_mark = " ★" if r["gain_km"] >= 0.30 else "  "
        sp_mark = " ★" if r["gain_sp"] >= 0.30 else "  "
        print(fmt.format(
            r["dataset_id"][:30] + (".." if len(r["dataset_id"]) > 30 else ""),
            r["dtype"][:17],
            f"{r['raw_km']:.4f}",
            f"{r['raw_sp']:.4f}",
            f"{r['best_umap_km']:.4f}{km_mark}",
            f"{r['best_umap_sp']:.4f}{sp_mark}",
            f"{r['gain_km']:+.4f}",
            f"{r['gain_sp']:+.4f}",
        ))
    print()

    # Focus on spirals
    spirals_rows = [r for r in rows if "spirals" in r["dataset_id"]]
    if spirals_rows:
        best_spirals = max(max(r["best_umap_km"], r["best_umap_sp"]) for r in spirals_rows)
        avg_spirals = float(np.mean([max(r["best_umap_km"], r["best_umap_sp"]) for r in spirals_rows]))
        print(f"Spirals — best AMI across all UMAP configs:     {best_spirals:.4f}")
        print(f"Spirals — mean best AMI:                        {avg_spirals:.4f}")
        print()
        if avg_spirals >= 0.5:
            verdict = "UMAP fixes the spirals problem"
            rationale = ("The single-cell-style UMAP pipeline recovers the spirals dataset. "
                         "The 'spirals problem' is NOT beyond existing tools — it just requires "
                         "the right preprocessing. The Φ(X) paper should compete against this "
                         "baseline, and the spirals are no longer an unbounded open problem.")
        elif avg_spirals >= 0.25:
            verdict = "UMAP partially fixes spirals"
            rationale = ("Some improvement but not a clean solve. Worth noting but spirals at "
                         "this noise level remain a hard case for all current methods.")
        else:
            verdict = "UMAP also fails on spirals"
            rationale = ("Even the single-cell-style UMAP pipeline doesn't recover spirals. "
                         "This dataset is genuinely beyond what existing clustering paradigms "
                         "can do without labels. The 'spirals problem' is real and bounded.")
        print(f"VERDICT: {verdict}")
        print(f"  {rationale}")
    print("=" * 110)


if __name__ == "__main__":
    main()
