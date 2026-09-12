"""
Tier 0b — Gap-closure curve as a function of neighborhood radius.

Question: how much of MetaIVM's within-pool selection gap (~0.084) can be
closed by local search if we widen the move set beyond same-algo, k±1?

For each test dataset we hold MetaIVM's picked partition π* fixed and ask, for
each neighborhood definition N below: what is the best AMI reachable in N, and
how much does it improve over π*? Aggregating mean(gain) across all test
datasets gives a curve from "tight same-algo neighborhood" to "anything in the
pool" (the within-pool oracle).

Neighborhood definitions (in increasing breadth):

  N1  hyperparam_tight       same algo,  |Δk| ≤ 1     (Tier 0 baseline)
  N2  hyperparam_medium      same algo,  |Δk| ≤ 2
  N3  hyperparam_wide        same algo,  any k
  N4  cross_algo_tight       any algo,   |Δk| ≤ 1
  N5  cross_algo_medium      any algo,   |Δk| ≤ 3
  N6  feature_top10          top-10 nearest in partition_x_graph feature space
  N7  feature_top25          top-25 nearest in partition_x_graph feature space
  N8  feature_top50          top-50 nearest in partition_x_graph feature space
  N9  pool_oracle            anything in the pool (within-pool ceiling)

The interesting comparison is N1 vs N6/N7 vs N9:

  - If feature-space top-K already closes most of the gap → a feature-space
    local-search algorithm has real headroom.
  - If only N9 closes the gap and N6/N7 plateau low → MetaIVM's misranking
    is *spread across the pool*, not localized, and you need wider moves
    (or actual generation) to close the bulk of regret.

Outputs:
  results/aggregated/tier0b_per_dataset.csv    — (seed, dataset, neighborhood) rows
  results/aggregated/tier0b_radius_curve.csv   — gap-closure curve
  stdout                                       — printed curve + interpretation

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -m scripts.tier0b_neighborhood_radius
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

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

FEATURE_SET = "partition_x_graph"
SEEDS = [0, 1, 2, 3, 4]
MIN_PARTITIONS = 8
MIN_AMI_STD = 0.01

# Neighborhood definitions, in order of increasing breadth.
NEIGHBORHOODS = [
    "N1_hyperparam_tight",
    "N2_hyperparam_medium",
    "N3_hyperparam_wide",
    "N4_cross_algo_tight",
    "N5_cross_algo_medium",
    "N6_feature_top10",
    "N7_feature_top25",
    "N8_feature_top50",
    "N9_pool_oracle",
]


def _prepare_xy(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["ami"].to_numpy(dtype=np.float64, copy=False)
    return X, y


def train_metaivm(train_df, val_df, feature_cols):
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


def predict_metaivm(model, df, feature_cols):
    X, _ = _prepare_xy(df, feature_cols)
    return model.predict(X)


# ──────────────────────────────────────────────────────────────────────
# Neighborhood mask builders.
# Each returns a boolean mask over rows of `ds_df` excluding picked_idx.
# ──────────────────────────────────────────────────────────────────────

def _same_algo(ds_df: pd.DataFrame, picked: pd.Series) -> np.ndarray:
    return ds_df["algo"].values == picked["algo"]


def _delta_k_at_most(ds_df: pd.DataFrame, picked: pd.Series, dmax: int) -> np.ndarray:
    nc = ds_df["n_clusters"].values
    nc_p = picked["n_clusters"]
    if pd.isna(nc_p):
        # If picked has no cluster count, fall back to "all" for the k constraint.
        return np.ones(len(ds_df), dtype=bool)
    diff = np.abs(nc - nc_p)
    return diff <= dmax


def _feature_topk_mask(
    ds_features: np.ndarray, picked_idx: int, k: int
) -> np.ndarray:
    """Top-k nearest rows in L2 feature distance (standardized)."""
    # Standardize per-feature so no single feature dominates the metric.
    mu = ds_features.mean(axis=0, keepdims=True)
    sigma = ds_features.std(axis=0, keepdims=True) + 1e-8
    Z = (ds_features - mu) / sigma
    diff = Z - Z[picked_idx : picked_idx + 1]
    d2 = (diff * diff).sum(axis=1)
    d2[picked_idx] = np.inf  # exclude self
    order = np.argsort(d2)
    chosen = order[:k]
    mask = np.zeros(len(ds_features), dtype=bool)
    mask[chosen] = True
    return mask


def build_masks(
    ds_df: pd.DataFrame, picked_idx: int, ds_features: np.ndarray
) -> Dict[str, np.ndarray]:
    picked = ds_df.iloc[picked_idx]
    same_algo = _same_algo(ds_df, picked)

    masks = {}
    for name, mask in [
        ("N1_hyperparam_tight", same_algo & _delta_k_at_most(ds_df, picked, 1)),
        ("N2_hyperparam_medium", same_algo & _delta_k_at_most(ds_df, picked, 2)),
        ("N3_hyperparam_wide", same_algo.copy()),
        ("N4_cross_algo_tight", _delta_k_at_most(ds_df, picked, 1)),
        ("N5_cross_algo_medium", _delta_k_at_most(ds_df, picked, 3)),
        ("N6_feature_top10", _feature_topk_mask(ds_features, picked_idx, 10)),
        ("N7_feature_top25", _feature_topk_mask(ds_features, picked_idx, 25)),
        ("N8_feature_top50", _feature_topk_mask(ds_features, picked_idx, 50)),
        ("N9_pool_oracle", np.ones(len(ds_df), dtype=bool)),
    ]:
        m = mask.copy()
        m[picked_idx] = False
        masks[name] = m
    return masks


# ──────────────────────────────────────────────────────────────────────
# Diagnostics per dataset.
# ──────────────────────────────────────────────────────────────────────

def diagnose_dataset(
    ds_df: pd.DataFrame, pred_scores: np.ndarray, feature_cols: List[str]
) -> List[Dict]:
    true_ami = ds_df["ami"].to_numpy()
    n = len(true_ami)
    if n < MIN_PARTITIONS or float(true_ami.std()) < MIN_AMI_STD:
        return []

    picked_idx = int(np.argmax(pred_scores))
    picked_ami = float(true_ami[picked_idx])

    ds_features = ds_df[feature_cols].to_numpy(dtype=np.float64)
    ds_features = np.nan_to_num(ds_features, nan=0.0, posinf=0.0, neginf=0.0)

    masks = build_masks(ds_df.reset_index(drop=True), picked_idx, ds_features)

    rows = []
    for name, mask in masks.items():
        n_nbr = int(mask.sum())
        if n_nbr == 0:
            rows.append(
                dict(
                    neighborhood=name,
                    n_neighbors=0,
                    best_in_nbr_ami=np.nan,
                    gain=np.nan,
                    picked_ami=picked_ami,
                )
            )
            continue
        best = float(true_ami[mask].max())
        gain = max(0.0, best - picked_ami)
        rows.append(
            dict(
                neighborhood=name,
                n_neighbors=n_nbr,
                best_in_nbr_ami=best,
                gain=gain,
                picked_ami=picked_ami,
            )
        )
    return rows


# ──────────────────────────────────────────────────────────────────────
# Main.
# ──────────────────────────────────────────────────────────────────────

def run() -> None:
    logger.info(f"Loading features: {FEATURES_CSV}")
    features_df = pd.read_csv(FEATURES_CSV)
    logger.info(f"{len(features_df):,} runs, {features_df['dataset_id'].nunique()} datasets")
    feature_cols = get_feature_columns(features_df, FEATURE_SET)
    logger.info(f"Feature set '{FEATURE_SET}' ({len(feature_cols)} features)")
    dataset_ids = features_df["dataset_id"].unique().tolist()

    rows: List[Dict] = []
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
        logger.info(f"[seed {seed}] training MetaIVM...")
        model = train_metaivm(train_df, val_df, feature_cols)

        for ds_id in test_ids:
            ds_df = features_df[features_df["dataset_id"] == ds_id].reset_index(drop=True)
            if len(ds_df) < MIN_PARTITIONS:
                continue
            pred = predict_metaivm(model, ds_df, feature_cols)
            ds_rows = diagnose_dataset(ds_df, pred, feature_cols)
            for r in ds_rows:
                r.update({"seed": seed, "dataset_id": ds_id})
                rows.append(r)

        n_ds = len({r["dataset_id"] for r in rows if r["seed"] == seed})
        logger.info(f"[seed {seed}] diagnosed {n_ds} datasets")

    per_dataset = pd.DataFrame(rows)
    per_path = OUT_DIR / "tier0b_per_dataset.csv"
    per_dataset.to_csv(per_path, index=False)
    logger.info(f"Wrote per-dataset rows: {per_path}")

    # ── Gap closure curve ───────────────────────────────────────────
    # Mean gain (and median, and "fraction of datasets with ≥0.01 gain")
    # for each neighborhood, ordered by increasing breadth.
    curve_rows = []
    oracle_gap = float(per_dataset[per_dataset["neighborhood"] == "N9_pool_oracle"]["gain"].mean())
    for name in NEIGHBORHOODS:
        sub = per_dataset[per_dataset["neighborhood"] == name]
        mean_gain = float(sub["gain"].mean(skipna=True))
        median_gain = float(sub["gain"].median(skipna=True))
        mean_n_nbr = float(sub["n_neighbors"].mean())
        frac_meaningful = float((sub["gain"] >= 0.01).mean(skipna=True))
        closure_pct = 100.0 * mean_gain / oracle_gap if oracle_gap > 0 else float("nan")
        curve_rows.append(
            dict(
                neighborhood=name,
                mean_n_neighbors=round(mean_n_nbr, 1),
                mean_gain=mean_gain,
                median_gain=median_gain,
                frac_datasets_geq_0p01=frac_meaningful,
                pct_of_pool_oracle_gap_closed=closure_pct,
            )
        )
    curve = pd.DataFrame(curve_rows)
    curve_path = OUT_DIR / "tier0b_radius_curve.csv"
    curve.to_csv(curve_path, index=False)
    logger.info(f"Wrote radius curve: {curve_path}")

    # ── Print ──────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("TIER 0b — Gap-closure curve as a function of neighborhood breadth")
    print("=" * 78)
    print()
    fmt = "{:<24} {:>8} {:>11} {:>11} {:>10} {:>10}"
    print(fmt.format("neighborhood", "mean_n", "mean_gain", "median_gain", "≥0.01 (%)", "closure %"))
    print("-" * 78)
    for r in curve_rows:
        print(
            fmt.format(
                r["neighborhood"],
                f"{r['mean_n_neighbors']:.1f}",
                f"{r['mean_gain']:.4f}",
                f"{r['median_gain']:.4f}",
                f"{100 * r['frac_datasets_geq_0p01']:.1f}",
                f"{r['pct_of_pool_oracle_gap_closed']:.1f}",
            )
        )
    print()
    print(f"  Within-pool oracle gap (N9):  {oracle_gap:.4f}")
    print()

    # ── Interpretation ──────────────────────────────────────────────
    n1 = next(r for r in curve_rows if r["neighborhood"] == "N1_hyperparam_tight")
    n3 = next(r for r in curve_rows if r["neighborhood"] == "N3_hyperparam_wide")
    n7 = next(r for r in curve_rows if r["neighborhood"] == "N7_feature_top25")
    n9 = next(r for r in curve_rows if r["neighborhood"] == "N9_pool_oracle")

    def pct(r):
        return r["pct_of_pool_oracle_gap_closed"]

    print("Reading the curve:")
    print(f"  - Tight same-algo (N1):           closes {pct(n1):.0f}% of within-pool gap")
    print(f"  - Wide same-algo (N3):            closes {pct(n3):.0f}% of within-pool gap")
    print(f"  - Feature-top25 (N6/N7 family):   closes {pct(n7):.0f}% of within-pool gap")
    print(f"  - Whole pool (N9, ceiling):       closes 100% by definition ({n9['mean_gain']:.4f} AMI)")
    print()

    if pct(n7) >= 70:
        verdict = (
            "Most of the gap is reachable by a feature-space local search. "
            "A wider local-search algorithm (cross-algo, feature-distance moves) "
            "has clear room to be a real algorithm contribution."
        )
    elif pct(n7) >= 40:
        verdict = (
            "Feature-space local search closes a meaningful but partial fraction. "
            "Likely a section, not a full paper. Generation (Φ(X) etc.) still motivated "
            "for the remaining gap."
        )
    else:
        verdict = (
            "Feature-space local search closes only a small fraction. "
            "MetaIVM's mis-ranking is spread across the pool — generation, not search, "
            "is the bottleneck. Theory paper + Φ(X) remain the right next steps."
        )
    print(f"VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    run()
