"""
Controlled shifts experiment — how does MetaIVM regret vary across dataset
characteristics (k, d, noise, imbalance, shape)?

Reviewer g9yy asked whether performance is uniform across dataset properties.
We bin the 223 test datasets by five axes and report mean regret per bin:

  - Number of classes k:  buckets {2, 3-5, 6-10, 11-20, 21+}
  - Dimensionality d:     buckets {≤5, 6-15, 16-50, >50}
  - Class imbalance:      ratio max_class_size / min_class_size, buckets {1-1.5, 1.5-3, 3-10, >10}
  - Dataset size n:       buckets {<500, 500-1500, 1500-3000, >3000}
  - Dataset source:       synthetic / openml / text / image

For each bin we report:
  n_datasets, mean_regret, std_regret (across datasets in the bin).

Uses the existing MetaIVM XGBoost model with 5-seed CV. Outputs are the
per-bin table and a plot-ready CSV.

Run:
  cd path/to/Neural_IVM
  python -u -m scripts.controlled_shifts
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

from src.evaluation.splits import make_splits  # noqa: E402
from src.models.tabular_models import get_feature_columns  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_CSV = DATA_DIR / "features" / "all_features.csv"
REGISTRY = DATA_DIR / "dataset_registry.csv"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET = "partition_x_graph"
SEEDS = [0, 1, 2, 3, 4]


def _prepare_xy(df: pd.DataFrame, cols: List[str]):
    X = df[cols].to_numpy(dtype=np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["ami"].to_numpy(dtype=np.float64, copy=False)
    return X, y


def per_dataset_regret_across_seeds(features_df: pd.DataFrame,
                                     feature_cols: List[str]) -> pd.DataFrame:
    """Return one row per (seed, dataset) with regret computed by MetaIVM's
    pick on that dataset's partitions in the test fold."""
    import xgboost as xgb

    dataset_ids = sorted(features_df["dataset_id"].unique())
    rows = []
    for seed in SEEDS:
        splits = make_splits(dataset_ids, n_splits=1, seed=seed)
        train_ids, val_ids, test_ids = splits[0]
        train_df = features_df[features_df["dataset_id"].isin(train_ids + val_ids)]
        test_df = features_df[features_df["dataset_id"].isin(test_ids)]

        X_train, y_train = _prepare_xy(train_df, feature_cols)
        model = xgb.XGBRegressor(
            max_depth=5, n_estimators=500, learning_rate=0.05,
            random_state=0, verbosity=0, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        for ds_id, sub in test_df.groupby("dataset_id"):
            if len(sub) < 2:
                continue
            X_sub, _ = _prepare_xy(sub, feature_cols)
            preds = model.predict(X_sub)
            picked = sub["ami"].iloc[int(preds.argmax())]
            best = sub["ami"].max()
            rows.append(dict(seed=seed, dataset_id=ds_id, regret=best - picked))
        logger.info(f"  seed {seed}: computed regret for {len(rows) - sum(1 for r in rows if r['seed'] < seed)} datasets")
    return pd.DataFrame(rows)


def load_dataset_props() -> pd.DataFrame:
    """Attach dataset-level metadata: n, d, n_classes, source, imbalance."""
    reg = pd.read_csv(REGISTRY)
    all_feats = pd.read_csv(FEATURES_CSV, usecols=[
        "dataset_id", "ds_n_samples", "ds_n_features",
    ]).drop_duplicates(subset=["dataset_id"])

    # Compute imbalance from y_true, cache
    from pathlib import Path
    props = reg.merge(all_feats, on="dataset_id", how="left")

    # imbalance: max_class_size / min_class_size on ground-truth labels
    imbalances = []
    for ds_id in props["dataset_id"]:
        y_path = DATA_DIR / "processed" / ds_id / "y_true.npy"
        if y_path.exists():
            y = np.load(y_path)
            _, counts = np.unique(y, return_counts=True)
            if len(counts) < 2:
                imbalances.append(1.0)
                continue
            imbalances.append(float(counts.max() / max(counts.min(), 1)))
        else:
            imbalances.append(np.nan)
    props["imbalance"] = imbalances
    return props


def bucketize(props: pd.DataFrame) -> pd.DataFrame:
    """Add categorical bucket columns for k, d, n, imbalance."""
    p = props.copy()

    # k buckets
    def k_bucket(k):
        if pd.isna(k):
            return "unknown"
        k = int(k)
        if k == 2: return "2"
        if k <= 5: return "3-5"
        if k <= 10: return "6-10"
        if k <= 20: return "11-20"
        return "21+"

    def d_bucket(d):
        if pd.isna(d):
            return "unknown"
        d = int(d)
        if d <= 5: return "≤5"
        if d <= 15: return "6-15"
        if d <= 50: return "16-50"
        return ">50"

    def n_bucket(n):
        if pd.isna(n):
            return "unknown"
        n = int(n)
        if n < 500: return "<500"
        if n < 1500: return "500-1500"
        if n < 3000: return "1500-3000"
        return ">3000"

    def imb_bucket(im):
        if pd.isna(im):
            return "unknown"
        if im < 1.5: return "balanced (<1.5)"
        if im < 3: return "mild (1.5-3)"
        if im < 10: return "moderate (3-10)"
        return "extreme (>10)"

    p["k_bucket"] = p["n_classes"].apply(k_bucket)
    p["d_bucket"] = p["ds_n_features"].apply(d_bucket)
    p["n_bucket"] = p["ds_n_samples"].apply(n_bucket)
    p["imbalance_bucket"] = p["imbalance"].apply(imb_bucket)
    return p


def summarize_by_axis(regrets: pd.DataFrame, props: pd.DataFrame,
                      axis_col: str, axis_label: str) -> pd.DataFrame:
    """Per-bin mean regret, aggregated across (seed, dataset) rows."""
    joined = regrets.merge(props[["dataset_id", axis_col]], on="dataset_id", how="left")
    agg = joined.groupby(axis_col)["regret"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={
        axis_col: "bin", "mean": "mean_regret", "std": "std_regret",
        "count": "n_dataset_seed_pairs",
    })
    agg["axis"] = axis_label
    return agg[["axis", "bin", "n_dataset_seed_pairs", "mean_regret", "std_regret"]]


def main() -> None:
    t_total = time.time()

    features_df = pd.read_csv(FEATURES_CSV)
    logger.info(f"Loaded {len(features_df):,} rows across {features_df['dataset_id'].nunique()} datasets")
    feature_cols = get_feature_columns(features_df, FEATURE_SET)
    logger.info(f"Feature set '{FEATURE_SET}' ({len(feature_cols)} features)")

    logger.info("Computing per-(seed, dataset) regret via MetaIVM (XGBoost)...")
    regrets = per_dataset_regret_across_seeds(features_df, feature_cols)
    regrets.to_csv(OUT_DIR / "per_dataset_seed_regret.csv", index=False)
    logger.info(f"  {len(regrets):,} (seed, dataset) rows")

    logger.info("Loading dataset properties...")
    props = bucketize(load_dataset_props())

    axes = [
        ("k_bucket", "n_classes"),
        ("d_bucket", "dimensionality"),
        ("n_bucket", "dataset_size"),
        ("imbalance_bucket", "imbalance"),
        ("source", "source"),
    ]
    summary_rows = []
    for col, label in axes:
        s = summarize_by_axis(regrets, props, col, label)
        summary_rows.append(s)
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(OUT_DIR / "controlled_shifts_summary.csv", index=False)
    logger.info(f"Wrote controlled_shifts_summary.csv  (total {(time.time()-t_total)/60:.1f}m)")

    print()
    print("=" * 76)
    print("CONTROLLED SHIFTS — MetaIVM (XGBoost) regret by dataset axis")
    print("=" * 76)
    for label in ["n_classes", "dimensionality", "dataset_size", "imbalance", "source"]:
        sub = summary[summary["axis"] == label]
        print()
        print(f"-- {label} --")
        fmt = "{:<20} {:>18} {:>18} {:>10}"
        print(fmt.format("bin", "N (seed×dataset)", "mean_regret ± std", "hint"))
        print("-" * 76)
        for _, r in sub.iterrows():
            hint = "" if r["mean_regret"] < 0.15 else " ↑"
            print(fmt.format(
                str(r["bin"]),
                int(r["n_dataset_seed_pairs"]),
                f"{r['mean_regret']:.4f} ± {r['std_regret']:.4f}",
                hint,
            ))
    print()
    print("=" * 76)


if __name__ == "__main__":
    main()
