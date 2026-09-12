"""
Compute density-based / noise-aware CVIs (DBCV, S_Dbw) for every cached
partition in data/clustering_runs/.

Reviewer g9yy asked for stronger baselines specifically for density-based
clusters and partitions with noise. Classical CH/Silhouette/DB are inadequate
for pools that include DBSCAN/HDBSCAN outputs. This script produces the CVIs
needed to add matching rows to Table 1 and to demonstrate our handling of
degenerate partitions.

Outputs one row per (dataset_id, run_id) in results/aggregated/density_cvis.csv
with columns: dataset_id, run_id, algo, ami, n_clusters, noise_fraction,
              dbcv, s_dbw, dbcv_status, s_dbw_status

Handling of degenerate partitions (as documented in Appendix I of the paper):
  - Trivial partitions (k=1 after excluding noise) → NaN, status='trivial'
  - >95% noise → NaN, status='all_noise'
  - Singleton-heavy (>50% singletons) → NaN, status='singleton_heavy'
  - Feasible partitions → real number, status='ok'

Wall-clock: DBCV is O(N^2), so datasets with N=5000 dominate the runtime.
Expected total: ~2-4 hours single-thread; parallelizes trivially per-dataset.

Run:
  cd path/to/Neural_IVM
  python -u -m scripts.compute_density_cvis
"""

from __future__ import annotations

import logging
import re
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
CLUSTERING_RUNS_DIR = DATA_DIR / "clustering_runs"
RUNS_MASTER = DATA_DIR / "features" / "runs_master.csv"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Handling knobs (documented in the paper).
MAX_NOISE_FRAC = 0.95
MAX_SINGLETON_FRAC = 0.50


def classify_partition(labels: np.ndarray) -> Tuple[str, int, float]:
    """Return (status, n_clusters_nonnoise, noise_fraction)."""
    noise = (labels == -1).sum()
    noise_frac = float(noise) / len(labels)
    non_noise = labels[labels != -1]
    if len(non_noise) == 0:
        return "all_noise", 0, noise_frac
    unique, counts = np.unique(non_noise, return_counts=True)
    k = len(unique)
    if k < 2:
        return "trivial", k, noise_frac
    if noise_frac > MAX_NOISE_FRAC:
        return "all_noise", k, noise_frac
    singleton_frac = float((counts == 1).sum()) / len(non_noise)
    if singleton_frac > MAX_SINGLETON_FRAC:
        return "singleton_heavy", k, noise_frac
    return "ok", k, noise_frac


DBCV_TIMEOUT_SEC = 60  # per-partition cap — hdbscan.validity_index has pathological
                        # runtime on some partitions (some take 6+ hours). Skip those
                        # and mark as NaN with a "timeout" tag in the CSV.


class _DBCVTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _DBCVTimeout()


def compute_dbcv(X: np.ndarray, labels: np.ndarray):
    """DBCV via hdbscan.validity — handles noise natively (label -1 excluded).
    Returns (value_or_NaN, status) where status is 'ok', 'nan', 'timeout', or 'error'.
    Uses SIGALRM to cap each call at DBCV_TIMEOUT_SEC seconds."""
    import signal
    try:
        from hdbscan.validity import validity_index
    except Exception:
        return float("nan"), "error"

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(DBCV_TIMEOUT_SEC)
    try:
        val = validity_index(X.astype(np.float64), labels.astype(np.int64))
    except _DBCVTimeout:
        return float("nan"), "timeout"
    except Exception:
        return float("nan"), "error"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not np.isfinite(val):
        return float("nan"), "nan"
    return float(val), "ok"


def compute_s_dbw(X: np.ndarray, labels: np.ndarray) -> float:
    """S_Dbw (Halkidi 2001). Lower is better; convert to higher-is-better
    by returning -S_Dbw so it matches the direction of Silhouette/CH."""
    try:
        from s_dbw import S_Dbw
        # s_dbw doesn't accept noise labels — exclude them.
        mask = labels != -1
        if mask.sum() < 4 or len(np.unique(labels[mask])) < 2:
            return float("nan")
        val = S_Dbw(X[mask].astype(np.float64), labels[mask].astype(np.int64))
        if not np.isfinite(val):
            return float("nan")
        return -float(val)  # negate: higher is better
    except Exception:
        return float("nan")


N_CAP = 3000  # DBCV is O(N^2); subsample above this to avoid OOM


def load_dataset(ds_id: str, rng: np.random.Generator | None = None):
    X = np.load(PROCESSED_DIR / ds_id / "X.npy").astype(np.float64)
    if X.shape[0] > N_CAP:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=N_CAP, replace=False)
        idx.sort()
        return X[idx], idx
    return X, None


def process_dataset(ds_id: str, runs_df: pd.DataFrame, rng: np.random.Generator) -> List[Dict]:
    """Compute DBCV, S_Dbw for every cached partition of this dataset."""
    try:
        X, subsample_idx = load_dataset(ds_id, rng=rng)
    except FileNotFoundError:
        logger.warning(f"[{ds_id}] X.npy not found; skipping")
        return []

    ds_run_dir = CLUSTERING_RUNS_DIR / ds_id
    if not ds_run_dir.exists():
        logger.warning(f"[{ds_id}] no cached partitions; skipping")
        return []

    rows: List[Dict] = []
    ds_runs = runs_df[runs_df["dataset_id"] == ds_id]
    t0 = time.time()

    for _, run in ds_runs.iterrows():
        algo = run["algo"]
        run_id = run["run_id"]
        # File name pattern: {algo}_{run_id}.npy
        candidate = ds_run_dir / f"{algo}_{run_id}.npy"
        if not candidate.exists():
            continue
        try:
            labels_full = np.load(candidate).astype(np.int64)
        except Exception as e:
            logger.warning(f"[{ds_id}/{run_id}] load failed: {e}")
            continue

        # Align labels with (possibly subsampled) X.
        if subsample_idx is not None:
            labels = labels_full[subsample_idx]
        else:
            labels = labels_full

        if len(labels) != len(X):
            logger.warning(f"[{ds_id}/{run_id}] length mismatch ({len(labels)} vs {len(X)}); skipping")
            continue

        status, k, noise_frac = classify_partition(labels)
        if status != "ok":
            dbcv = float("nan")
            dbcv_status = status  # 'trivial' / 'all_noise' / 'singleton_heavy'
            s_dbw = float("nan")
        else:
            dbcv, dbcv_status = compute_dbcv(X, labels)
            s_dbw = compute_s_dbw(X, labels)

        rows.append(dict(
            dataset_id=ds_id, run_id=run_id, algo=algo,
            ami=run["ami"],
            n_clusters=int(k),
            dbcv_status=dbcv_status,
            noise_fraction=float(noise_frac),
            status=status,
            dbcv=dbcv,
            s_dbw=s_dbw,
        ))

    logger.info(f"[{ds_id}] processed {len(rows)} partitions in {time.time()-t0:.1f}s")
    return rows


def main() -> None:
    runs_df = pd.read_csv(RUNS_MASTER)
    dataset_ids = sorted(runs_df["dataset_id"].unique())
    logger.info(f"Processing {len(dataset_ids)} datasets, {len(runs_df):,} partitions total")

    rng = np.random.default_rng(0)
    all_rows: List[Dict] = []
    t_total = time.time()
    for i, ds_id in enumerate(dataset_ids):
        rows = process_dataset(ds_id, runs_df, rng)
        all_rows.extend(rows)
        if (i + 1) % 20 == 0:
            elapsed = (time.time() - t_total) / 60
            logger.info(f"  [{i+1}/{len(dataset_ids)}]  elapsed {elapsed:.1f}m  rows so far {len(all_rows):,}")

    out = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "density_cvis.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  ({len(out):,} rows, total {(time.time()-t_total)/60:.1f}m)")

    # Summary
    print()
    print("Partition status distribution:")
    print(out["status"].value_counts().to_string())
    print()
    print("DBCV computation status (on non-degenerate partitions):")
    ok = out[out["status"] == "ok"]
    print(ok["dbcv_status"].value_counts().to_string())
    print()
    print("Availability:")
    print(f"  DBCV non-NaN:  {ok['dbcv'].notna().sum():,} / {len(ok):,}  ({100*ok['dbcv'].notna().mean():.1f}%)")
    print(f"  S_Dbw non-NaN: {ok['s_dbw'].notna().sum():,} / {len(ok):,}  ({100*ok['s_dbw'].notna().mean():.1f}%)")


if __name__ == "__main__":
    main()
