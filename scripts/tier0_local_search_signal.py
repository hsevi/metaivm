"""
Tier 0 — Does local search on top of MetaIVM have signal?

Analysis-only probe (no new clustering runs, no local search loop).
For each of 5 seed splits we train MetaIVM (XGBoost, partition_x_graph features,
fixed config — no grid search to keep this fast) on the train+val fold, predict
on the test fold, then compute three diagnostics per test dataset:

  (A) Selection gap on the pool
        = oracle_AMI_in_pool − MetaIVM_picked_AMI
      Headline regret (0.071) decomposes into selection gap (within pool) and
      generation gap (beyond pool). Large (A) → local search can win just by
      re-ranking nearby partitions. Small (A) → MetaIVM already saturates the
      pool; you'd need to *generate* partitions, not just re-pick.

  (B) MetaIVM monotonicity
        = Spearman ρ(MetaIVM predicted scores, true AMI) on the pool, per dataset
      Hi ρ → re-ranking gains little.  Lo ρ → MetaIVM is noisy on the pool and
      local search would routinely flip near-neighbors into improvements.

  (C) Near-neighbor improvement fraction
        For each test dataset, take MetaIVM's pick π*. A "near-neighbor" of π*
        in the pool is a partition with the same algorithm and |n_clusters − k*| ≤ 1.
        Compute the fraction of near-neighbors that have higher AMI than π*.
      This is exactly what a single-cluster-merge/split or k±1 sweep would
      discover. Hi fraction → cheap local moves win. Lo fraction → π* is locally
      optimal in the pool's neighborhood.

Decision rule (printed at the end):

  SCALE UP   — gap ≥ 0.030 AND (1−ρ) ≥ 0.10 AND nbr_frac ≥ 0.20
  PARK       — gap ≤ 0.010 AND ρ ≥ 0.92 AND nbr_frac ≤ 0.05
  AMBIGUOUS  — anything else → run Tier 1 (minimal local search on 5 datasets)

Outputs:
  results/aggregated/tier0_per_dataset.csv  — one row per (seed, dataset)
  results/aggregated/tier0_summary.csv      — aggregate by source slice
  stdout                                    — printed verdict + diagnostics

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -m scripts.tier0_local_search_signal
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import make_splits  # noqa: E402
from src.models.tabular_models import get_feature_columns  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_CSV = DATA_DIR / "features" / "all_features.csv"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET = "partition_x_graph"  # matches the headline (28 features)
SEEDS = [0, 1, 2, 3, 4]

# Decision thresholds (tuned conservatively; treat AMBIGUOUS as the default).
GAP_HI = 0.030       # selection-gap threshold for SCALE UP
GAP_LO = 0.010
RHO_LO_FOR_SCALE = 0.90   # i.e. 1-ρ ≥ 0.10
RHO_HI_FOR_PARK = 0.92
NBR_HI = 0.20        # near-neighbor improvement-fraction
NBR_LO = 0.05

# Filter datasets too small for stable Spearman.
MIN_PARTITIONS = 8
MIN_AMI_STD = 0.01


# ──────────────────────────────────────────────────────────────────────
# Source classification — used purely for the summary breakdown.
# Mirrors the per-source slicing in Section 6.1 of the paper.
# ──────────────────────────────────────────────────────────────────────

def classify_source(dataset_id: str) -> str:
    if dataset_id.startswith("openml_"):
        return "openml"
    if dataset_id.startswith("text_"):
        return "text"
    if dataset_id.startswith("img_"):
        return "image"
    if dataset_id.startswith("synth_"):
        return "synthetic"
    return "other"


# ──────────────────────────────────────────────────────────────────────
# Lightweight XGBoost wrapper — single fixed config, no grid search.
# We only need predictions to be reasonable; absolute regret here is not
# the headline number (the trained models in results/ use grid search).
# ──────────────────────────────────────────────────────────────────────

def _prepare_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["ami"].to_numpy(dtype=np.float64, copy=False)
    return X, y


def train_metaivm(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]):
    import xgboost as xgb

    X_train, y_train = _prepare_xy(train_df, feature_cols)
    X_val, y_val = _prepare_xy(val_df, feature_cols)

    model = xgb.XGBRegressor(
        max_depth=5,
        n_estimators=500,
        learning_rate=0.05,
        early_stopping_rounds=20,
        eval_metric="rmse",
        random_state=0,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def predict_metaivm(model, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X, _ = _prepare_xy(df, feature_cols)
    return model.predict(X)


# ──────────────────────────────────────────────────────────────────────
# Per-dataset diagnostics.
# ──────────────────────────────────────────────────────────────────────

def near_neighbors_mask(df: pd.DataFrame, picked_idx: int) -> np.ndarray:
    """Boolean mask over rows of `df` that are near-neighbors of the picked row.

    Definition: same algorithm AND |n_clusters - picked_n_clusters| ≤ 1,
    excluding the picked row itself. This is exactly the move set a single
    cluster-split / cluster-merge / k±1 perturbation could reach.
    """
    picked = df.iloc[picked_idx]
    same_algo = (df["algo"].values == picked["algo"])
    nc = df["n_clusters"].values
    nc_picked = picked["n_clusters"]
    if pd.isna(nc_picked):
        # Pool entries with NaN cluster count: fall back to algo-only.
        mask = same_algo
    else:
        mask = same_algo & (np.abs(nc - nc_picked) <= 1)
    mask[picked_idx] = False
    return mask


def diagnose_dataset(ds_df: pd.DataFrame, pred_scores: np.ndarray) -> Dict:
    """Compute (A) selection gap, (B) ρ, (C) near-neighbor fraction for one dataset."""
    true_ami = ds_df["ami"].to_numpy()
    n = len(true_ami)

    if n < MIN_PARTITIONS or float(true_ami.std()) < MIN_AMI_STD:
        return {"valid": False, "n_partitions": n}

    oracle = float(true_ami.max())
    picked_idx = int(np.argmax(pred_scores))
    picked_ami = float(true_ami[picked_idx])
    selection_gap = oracle - picked_ami

    rho, _ = spearmanr(pred_scores, true_ami)
    if rho is None or np.isnan(rho):
        rho = 0.0

    nbr_mask = near_neighbors_mask(ds_df.reset_index(drop=True), picked_idx)
    n_nbr = int(nbr_mask.sum())
    if n_nbr > 0:
        nbr_amis = true_ami[nbr_mask]
        nbr_improve_frac = float((nbr_amis > picked_ami).mean())
        best_nbr_gain = float(max(0.0, nbr_amis.max() - picked_ami))
    else:
        nbr_improve_frac = np.nan
        best_nbr_gain = np.nan

    return {
        "valid": True,
        "n_partitions": n,
        "n_neighbors": n_nbr,
        "oracle_ami": oracle,
        "picked_ami": picked_ami,
        "selection_gap": selection_gap,
        "spearman_rho": float(rho),
        "nbr_improve_frac": nbr_improve_frac,
        "best_nbr_gain": best_nbr_gain,
    }


# ──────────────────────────────────────────────────────────────────────
# Main loop — 5 seeds × test datasets.
# ──────────────────────────────────────────────────────────────────────

def run() -> None:
    logger.info(f"Loading features: {FEATURES_CSV}")
    features_df = pd.read_csv(FEATURES_CSV)
    logger.info(f"Loaded {len(features_df):,} runs across {features_df['dataset_id'].nunique()} datasets")

    feature_cols = get_feature_columns(features_df, FEATURE_SET)
    logger.info(f"Using feature set '{FEATURE_SET}' ({len(feature_cols)} features)")

    dataset_ids = features_df["dataset_id"].unique().tolist()

    per_dataset_rows: List[Dict] = []

    for seed in SEEDS:
        splits = make_splits(
            dataset_ids,
            n_splits=1,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
            seed=seed,
        )
        train_ids, val_ids, test_ids = splits[0]

        train_df = features_df[features_df["dataset_id"].isin(train_ids)]
        val_df = features_df[features_df["dataset_id"].isin(val_ids)]

        logger.info(
            f"[seed {seed}] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
            f"datasets; training MetaIVM..."
        )
        model = train_metaivm(train_df, val_df, feature_cols)

        for ds_id in test_ids:
            ds_df = features_df[features_df["dataset_id"] == ds_id].reset_index(drop=True)
            if len(ds_df) < MIN_PARTITIONS:
                continue
            pred = predict_metaivm(model, ds_df, feature_cols)
            diag = diagnose_dataset(ds_df, pred)
            if not diag["valid"]:
                continue
            row = {"seed": seed, "dataset_id": ds_id, "source": classify_source(ds_id)}
            row.update({k: v for k, v in diag.items() if k != "valid"})
            per_dataset_rows.append(row)

        logger.info(f"[seed {seed}] diagnosed {sum(1 for r in per_dataset_rows if r['seed']==seed)} test datasets")

    per_dataset = pd.DataFrame(per_dataset_rows)
    per_dataset_path = OUT_DIR / "tier0_per_dataset.csv"
    per_dataset.to_csv(per_dataset_path, index=False)
    logger.info(f"Wrote per-dataset rows: {per_dataset_path}")

    # ── Aggregate ────────────────────────────────────────────────────
    def agg(df: pd.DataFrame) -> Dict:
        return {
            "n_obs": int(len(df)),
            "selection_gap_mean": float(df["selection_gap"].mean()),
            "selection_gap_median": float(df["selection_gap"].median()),
            "spearman_rho_mean": float(df["spearman_rho"].mean()),
            "spearman_rho_median": float(df["spearman_rho"].median()),
            "nbr_improve_frac_mean": float(df["nbr_improve_frac"].mean(skipna=True)),
            "best_nbr_gain_mean": float(df["best_nbr_gain"].mean(skipna=True)),
            "frac_with_any_nbr_improvement": float(
                (df["nbr_improve_frac"].fillna(0) > 0).mean()
            ),
        }

    slices = {
        "all": per_dataset,
        "synthetic": per_dataset[per_dataset["source"] == "synthetic"],
        "real": per_dataset[per_dataset["source"].isin(["openml", "text", "image"])],
        "openml": per_dataset[per_dataset["source"] == "openml"],
    }
    summary_rows = []
    for name, sub in slices.items():
        if len(sub) == 0:
            continue
        row = {"slice": name}
        row.update(agg(sub))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "tier0_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote summary: {summary_path}")

    # ── Verdict ──────────────────────────────────────────────────────
    overall = agg(per_dataset)
    gap = overall["selection_gap_mean"]
    rho = overall["spearman_rho_mean"]
    nbr = overall["nbr_improve_frac_mean"]
    gain = overall["best_nbr_gain_mean"]

    if gap >= GAP_HI and rho <= RHO_LO_FOR_SCALE and nbr >= NBR_HI:
        verdict = "SCALE UP"
        rationale = (
            "Large selection gap, low monotonicity, and dense better neighbors. "
            "Local search has clear signal; commit to Tier 1/2 or jump to algorithm work."
        )
    elif gap <= GAP_LO and rho >= RHO_HI_FOR_PARK and nbr <= NBR_LO:
        verdict = "PARK"
        rationale = (
            "MetaIVM already saturates the candidate pool. Re-ranking gains nothing; "
            "any algorithm progress must come from generation, not selection. "
            "Commit to the theory paper."
        )
    else:
        verdict = "AMBIGUOUS"
        rationale = (
            "Diagnostics don't agree. Run Tier 1: minimal swap-only local search "
            "on 5 representative datasets to disambiguate."
        )

    print()
    print("=" * 72)
    print(f"TIER 0 VERDICT: {verdict}")
    print("=" * 72)
    print(f"  Mean selection gap (oracle_in_pool − MetaIVM_pick):   {gap:.4f}")
    print(f"  Mean Spearman ρ(MetaIVM score, true AMI):             {rho:.4f}")
    print(f"  Mean near-neighbor improvement fraction:              {nbr:.4f}")
    print(f"  Mean best-neighbor AMI gain (if any):                 {gain:.4f}")
    print(f"  Datasets with ANY better neighbor:                    "
          f"{overall['frac_with_any_nbr_improvement']:.1%}")
    print()
    print(f"  Decision rule:")
    print(f"    SCALE UP  if  gap ≥ {GAP_HI:.3f} AND ρ ≤ {RHO_LO_FOR_SCALE:.2f} AND nbr ≥ {NBR_HI:.2f}")
    print(f"    PARK      if  gap ≤ {GAP_LO:.3f} AND ρ ≥ {RHO_HI_FOR_PARK:.2f} AND nbr ≤ {NBR_LO:.2f}")
    print(f"    AMBIGUOUS otherwise")
    print()
    print(f"  Rationale: {rationale}")
    print("=" * 72)
    print()
    print("Per-source breakdown:")
    print(summary.to_string(index=False))
    print()


if __name__ == "__main__":
    run()
