"""
AutoClust-style and PoAC-style baselines evaluated under our 5-seed 60/20/20
dataset-level CV protocol.

AutoClust (Poulakis et al. 2020, ICDM): MLP that maps a fixed vector of internal
CVIs to external agreement (ARI). We reproduce the same regression design but on
our benchmark, using an MLP with the same 10-60-30-10-1 architecture. We use
the subset of AutoClust's 10 CVIs that we can compute cleanly on our benchmark:
Silhouette, Calinski-Harabasz, Davies-Bouldin (from runs_master.csv), plus
Dunn Index (implemented inline), plus DBCV and S_Dbw (from density_cvis.csv
if available). This yields 4-6 CVIs vs AutoClust's original 10, but preserves
the design principle (regression from CVIs to external quality via MLP).

PoAC (da Silva et al. 2024): Random Forest surrogate on dataset meta-features
+ 2 CVIs. We use PyMFE (Lorena et al. 2019) to extract dataset meta-features
and Silhouette + Davies-Bouldin as CVIs. Note we train PoAC on our REAL
algorithm partitions (16,889 runs), not PoAC's noise-augmented labels — this
is the fair comparison because our benchmark exists and gives a stronger
learning signal.

Both baselines use the same 5-seed dataset-level 60/20/20 splits as MetaIVM.

Outputs:
  results/aggregated/baseline_autoclust_style.csv
  results/aggregated/baseline_poac_style.csv
  stdout                                          summary table

Run:
  cd path/to/Neural_IVM
  python -u -m scripts.autoclust_poac_baselines
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import make_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CSV = DATA_DIR / "features" / "all_features.csv"
RUNS_MASTER = DATA_DIR / "features" / "runs_master.csv"
DENSITY_CVIS = PROJECT_ROOT / "results" / "aggregated" / "density_cvis.csv"
CLUSTERING_RUNS = DATA_DIR / "clustering_runs"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


# ====================================================================
# Dunn index (implemented inline — simplest missing AutoClust CVI)
# ====================================================================
def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Dunn = min inter-cluster distance / max intra-cluster distance.

    We use single-linkage inter-cluster distance (nearest points across
    clusters) and diameter (max pairwise distance within a cluster) for
    intra-cluster. Higher is better."""
    from sklearn.metrics.pairwise import euclidean_distances
    mask = labels != -1
    if mask.sum() < 4:
        return float("nan")
    y = labels[mask]
    Z = X[mask]
    clusters = np.unique(y)
    if len(clusters) < 2:
        return float("nan")
    D = euclidean_distances(Z, Z)
    # Intra-cluster diameters
    max_intra = 0.0
    for c in clusters:
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        d_cluster = D[np.ix_(idx, idx)].max()
        max_intra = max(max_intra, d_cluster)
    if max_intra == 0.0:
        return float("nan")
    # Inter-cluster distances (single linkage)
    min_inter = float("inf")
    for i, c1 in enumerate(clusters):
        idx1 = np.where(y == c1)[0]
        for c2 in clusters[i + 1:]:
            idx2 = np.where(y == c2)[0]
            d = D[np.ix_(idx1, idx2)].min()
            min_inter = min(min_inter, d)
    if not np.isfinite(min_inter):
        return float("nan")
    return float(min_inter / max_intra)


# ====================================================================
# Compute Dunn for every partition (fills the AutoClust CVI vector).
# Cached to disk because it takes a while.
# ====================================================================
def compute_or_load_dunn() -> pd.DataFrame:
    out_path = OUT_DIR / "dunn_index.csv"
    if out_path.exists():
        logger.info(f"Loading cached Dunn: {out_path}")
        return pd.read_csv(out_path)
    logger.info("Computing Dunn index for every cached partition (~30-90 min)...")
    runs = pd.read_csv(RUNS_MASTER)
    rows = []
    t_total = time.time()
    for i, ds_id in enumerate(sorted(runs["dataset_id"].unique())):
        try:
            X = np.load(PROCESSED_DIR / ds_id / "X.npy").astype(np.float64)
        except FileNotFoundError:
            continue
        # Subsample if very large — Dunn is O(N^2) memory
        if X.shape[0] > 3000:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], size=3000, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
            idx = None
        ds_runs = runs[runs["dataset_id"] == ds_id]
        t = time.time()
        for _, run in ds_runs.iterrows():
            algo = run["algo"]
            rid = run["run_id"]
            candidate = CLUSTERING_RUNS / ds_id / f"{algo}_{rid}.npy"
            if not candidate.exists():
                continue
            labels = np.load(candidate).astype(np.int64)
            if idx is not None:
                labels_sub = labels[idx]
            else:
                labels_sub = labels
            d = dunn_index(X_sub, labels_sub)
            rows.append(dict(dataset_id=ds_id, run_id=rid, dunn=d))
        if (i + 1) % 20 == 0:
            logger.info(f"  [{i+1}]  elapsed {(time.time()-t_total)/60:.1f}m")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  ({len(df):,} rows)")
    return df


# ====================================================================
# Assemble AutoClust CVI vector for each (dataset, run).
# ====================================================================
def build_autoclust_features(runs: pd.DataFrame) -> pd.DataFrame:
    """CVI vector: Silhouette, CH, DB (from runs_master), Dunn, DBCV, S_Dbw."""
    df = runs[["dataset_id", "run_id", "algo", "ami", "silhouette",
               "calinski_harabasz", "davies_bouldin"]].copy()
    dunn = compute_or_load_dunn()
    df = df.merge(dunn, on=["dataset_id", "run_id"], how="left")
    if DENSITY_CVIS.exists():
        dens = pd.read_csv(DENSITY_CVIS)[["dataset_id", "run_id", "dbcv", "s_dbw"]]
        df = df.merge(dens, on=["dataset_id", "run_id"], how="left")
        logger.info(f"  Merged density CVIs (DBCV, S_Dbw) from {DENSITY_CVIS.name}")
    else:
        logger.info("  density_cvis.csv not available yet — using 4-CVI subset (Sil, CH, DB, Dunn)")
        df["dbcv"] = np.nan
        df["s_dbw"] = np.nan
    return df


# ====================================================================
# PoAC dataset meta-features via PyMFE.
# ====================================================================
def compute_or_load_pymfe() -> pd.DataFrame:
    out_path = OUT_DIR / "pymfe_features.csv"
    if out_path.exists():
        logger.info(f"Loading cached PyMFE features: {out_path}")
        return pd.read_csv(out_path)
    logger.info("Computing PyMFE dataset meta-features (~5-15 min)...")
    from pymfe.mfe import MFE
    rows = []
    runs = pd.read_csv(RUNS_MASTER)
    for i, ds_id in enumerate(sorted(runs["dataset_id"].unique())):
        try:
            X = np.load(PROCESSED_DIR / ds_id / "X.npy").astype(np.float64)
            y = np.load(PROCESSED_DIR / ds_id / "y_true.npy").astype(np.int64)
        except FileNotFoundError:
            continue
        if X.shape[0] > 5000:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], size=5000, replace=False)
            X = X[idx]
            y = y[idx]
        try:
            mfe = MFE(
                groups=["general", "statistical", "info-theory", "concept", "complexity"],
                summary=["mean", "sd"],
            )
            mfe.fit(X, y)
            names, values = mfe.extract(suppress_warnings=True)
            row = {"dataset_id": ds_id}
            for name, val in zip(names, values):
                row[f"pymfe_{name}"] = val
            rows.append(row)
        except Exception as e:
            logger.warning(f"[{ds_id}] pymfe failed: {e}")
        if (i + 1) % 40 == 0:
            logger.info(f"  [{i+1}/223]")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  ({len(df):,} datasets × {len(df.columns)} features)")
    return df


# ====================================================================
# Regret + evaluation utilities.
# ====================================================================
def _prepare_xy(df: pd.DataFrame, cols: List[str]):
    X = df[cols].to_numpy(dtype=np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["ami"].to_numpy(dtype=np.float64, copy=False)
    return X, y


def eval_regret(preds: np.ndarray, ds_ids: np.ndarray, true_ami: np.ndarray) -> float:
    """Selection regret = max_partition(ami) - ami[argmax(pred)] per dataset."""
    df = pd.DataFrame(dict(ds_id=ds_ids, pred=preds, ami=true_ami))
    regrets = []
    for ds_id, sub in df.groupby("ds_id"):
        if len(sub) < 2:
            continue
        picked = sub.iloc[int(sub["pred"].values.argmax())]["ami"]
        best = sub["ami"].max()
        regrets.append(best - picked)
    return float(np.mean(regrets)) if regrets else float("nan")


# ====================================================================
# AutoClust baseline — MLP (10-60-30-10-1 architecture as in the paper).
# We instantiate with input dim = number of available CVIs (4 or 6).
# ====================================================================
def train_autoclust_style(cvi_df: pd.DataFrame, seed: int) -> Dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor

    # Appearance order (NOT sorted) so the folds are identical to exp1_main_table.py
    # (which uses features_df["dataset_id"].unique().tolist()); make_splits permutes
    # indices, so a different ordering under the same seed yields different folds.
    dataset_ids = cvi_df["dataset_id"].unique().tolist()
    splits = make_splits(dataset_ids, n_splits=1, seed=seed)
    train_ids, val_ids, test_ids = splits[0]

    # AutoClust-style partial reproduction: exactly the 4 CVIs the paper commits to
    # (Silhouette, CH, DB, Dunn). DBCV/S_Dbw are deliberately NOT included even when
    # density_cvis.csv is present, so this stays a 4-CVI baseline on any re-run.
    cvi_cols = ["silhouette", "calinski_harabasz", "davies_bouldin", "dunn"]
    cvi_cols = [c for c in cvi_cols if c in cvi_df.columns and cvi_df[c].notna().any()]

    train_df = cvi_df[cvi_df["dataset_id"].isin(train_ids + val_ids)]
    test_df = cvi_df[cvi_df["dataset_id"].isin(test_ids)]

    X_train, y_train = _prepare_xy(train_df, cvi_cols)
    X_test, y_test = _prepare_xy(test_df, cvi_cols)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Architecture matched to AutoClust: 60 -> 30 -> 10 -> 1
    # (their exact architecture, sklearn represents hidden layers explicitly)
    model = MLPRegressor(
        hidden_layer_sizes=(60, 30, 10),
        activation="relu",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=False,
        random_state=0,
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    regret = eval_regret(preds, test_df["dataset_id"].values, y_test)
    return dict(seed=seed, regret=regret, n_cvis=len(cvi_cols), cvis_used=cvi_cols)


# ====================================================================
# PoAC baseline — Random Forest on PyMFE meta-features + 2 CVIs.
# ====================================================================
def train_poac_style(cvi_df: pd.DataFrame, pymfe_df: pd.DataFrame, seed: int) -> Dict:
    from sklearn.ensemble import RandomForestRegressor

    # Merge PyMFE dataset-level features with CVIs.
    df = cvi_df[["dataset_id", "run_id", "algo", "ami", "silhouette", "davies_bouldin"]].merge(
        pymfe_df, on="dataset_id", how="left"
    )
    pymfe_cols = [c for c in df.columns if c.startswith("pymfe_")]
    feat_cols = pymfe_cols + ["silhouette", "davies_bouldin"]

    # Appearance order (NOT sorted), identical to exp1_main_table.py folds.
    dataset_ids = df["dataset_id"].unique().tolist()
    splits = make_splits(dataset_ids, n_splits=1, seed=seed)
    train_ids, val_ids, test_ids = splits[0]

    train_df = df[df["dataset_id"].isin(train_ids + val_ids)]
    test_df = df[df["dataset_id"].isin(test_ids)]

    X_train, y_train = _prepare_xy(train_df, feat_cols)
    X_test, y_test = _prepare_xy(test_df, feat_cols)

    model = RandomForestRegressor(
        n_estimators=100, random_state=0, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    regret = eval_regret(preds, test_df["dataset_id"].values, y_test)
    return dict(seed=seed, regret=regret, n_features=len(feat_cols))


# ====================================================================
# Main
# ====================================================================
def main() -> None:
    logger.info("=== AutoClust-style and PoAC-style baselines ===")

    runs = pd.read_csv(RUNS_MASTER)
    logger.info("Building AutoClust CVI vector...")
    cvi_df = build_autoclust_features(runs)

    logger.info("Building PoAC meta-feature vector...")
    pymfe_df = compute_or_load_pymfe()

    # ---- AutoClust ----
    logger.info("Running AutoClust-style baseline (MLP on CVIs) across 5 seeds...")
    ac_results = []
    for s in SEEDS:
        r = train_autoclust_style(cvi_df, s)
        logger.info(f"  seed {s}: regret={r['regret']:.4f}  n_cvis={r['n_cvis']}")
        ac_results.append(r)

    ac_regrets = np.array([r["regret"] for r in ac_results])
    ac_mean = float(ac_regrets.mean())
    ac_std = float(ac_regrets.std())

    pd.DataFrame(ac_results).to_csv(OUT_DIR / "baseline_autoclust_style.csv", index=False)

    # ---- PoAC ----
    logger.info("Running PoAC-style baseline (RF on PyMFE + Sil/DB) across 5 seeds...")
    po_results = []
    for s in SEEDS:
        r = train_poac_style(cvi_df, pymfe_df, s)
        logger.info(f"  seed {s}: regret={r['regret']:.4f}  n_features={r['n_features']}")
        po_results.append(r)

    po_regrets = np.array([r["regret"] for r in po_results])
    po_mean = float(po_regrets.mean())
    po_std = float(po_regrets.std())

    pd.DataFrame(po_results).to_csv(OUT_DIR / "baseline_poac_style.csv", index=False)

    # ---- Print summary ----
    print()
    print("=" * 72)
    print("AUTOCLUST-STYLE AND POAC-STYLE BASELINES — 5-seed dataset-level CV")
    print("=" * 72)
    print()
    print(f"{'Baseline':<28} {'Regret (mean ± std)':<25} {'Δ vs MetaIVM (0.071)':>22}")
    print("-" * 72)
    print(f"{'AutoClust-style (MLP)':<28} {ac_mean:.4f} ± {ac_std:.4f}         {ac_mean - 0.071:+.4f}")
    print(f"{'PoAC-style (RF)':<28} {po_mean:.4f} ± {po_std:.4f}         {po_mean - 0.071:+.4f}")
    print(f"{'MetaIVM (XGBoost)':<28} {'0.071 (headline)':<25}")
    print(f"{'CH (best classical IVM)':<28} {'0.217':<25}")
    print("=" * 72)


if __name__ == "__main__":
    main()
