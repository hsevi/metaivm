"""
Leave-One-Generator-Family-Out (LOGFO) experiment.

Reviewer g9yy noted that random 60/20/20 splits may overestimate generalization
because datasets from closely-related synthetic generators end up in both
train and test folds. LOGFO tests transfer at the generator-family level.

Procedure:
  1. Group the 123 synthetic datasets by generator family (first token after 'synth_'):
     grid, blobs, imb, noisy, ellipsoidal, subspace, elongated, moons, circles, spirals,
     varying, aniso, swiss, s, nested, uniform, density, mixed
  2. Focus on the FAMILIES WITH ≥3 datasets — smaller families would produce
     unstable per-family regret estimates.
  3. For each such family F: train MetaIVM (XGBoost, default 28-feature set)
     on (all datasets NOT in F) and evaluate on datasets IN F.
  4. Report per-family regret and mean regret across families.

Output:
  results/aggregated/logfo_results.csv
  stdout                                     summary table + verdict

Run:
  cd path/to/Neural_IVM
  python -u -m scripts.logfo_experiment
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

from src.models.tabular_models import get_feature_columns  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_CSV = DATA_DIR / "features" / "all_features.csv"
REGISTRY = DATA_DIR / "dataset_registry.csv"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET = "partition_x_graph"
MIN_FAMILY_SIZE = 3   # families with fewer datasets are grouped as "other"


def family_from_dataset_id(ds_id: str) -> str:
    """Map dataset_id to a family label.

    Synthetic: extract prefix after 'synth_'. Real: return source (openml/text/image).
    """
    if ds_id.startswith("synth_"):
        return ds_id[len("synth_"):].split("_")[0]
    if ds_id.startswith("openml_"):
        return "openml"
    if ds_id.startswith("text_"):
        return "text"
    if ds_id.startswith("img_"):
        return "image"
    return "other"


def _prepare_xy(df: pd.DataFrame, cols: List[str]):
    X = df[cols].to_numpy(dtype=np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["ami"].to_numpy(dtype=np.float64, copy=False)
    return X, y


def train_xgb(train_df: pd.DataFrame, feature_cols: List[str]):
    import xgboost as xgb
    X, y = _prepare_xy(train_df, feature_cols)
    # Fixed config matched to the paper's default XGBoost.
    model = xgb.XGBRegressor(
        max_depth=5, n_estimators=500, learning_rate=0.05,
        random_state=0, verbosity=0, n_jobs=-1,
    )
    model.fit(X, y)
    return model


def eval_regret(model, test_df: pd.DataFrame, feature_cols: List[str]) -> float:
    X, y = _prepare_xy(test_df, feature_cols)
    preds = model.predict(X)
    regrets = []
    for ds_id, sub in test_df.assign(pred=preds).groupby("dataset_id"):
        if len(sub) < 2:
            continue
        picked = sub.iloc[int(sub["pred"].values.argmax())]["ami"]
        best = sub["ami"].max()
        regrets.append(best - picked)
    return float(np.mean(regrets)) if regrets else float("nan")


def run() -> None:
    features_df = pd.read_csv(FEATURES_CSV)
    logger.info(f"Loaded {len(features_df):,} rows across {features_df['dataset_id'].nunique()} datasets")

    feature_cols = get_feature_columns(features_df, FEATURE_SET)
    logger.info(f"Feature set '{FEATURE_SET}' ({len(feature_cols)} features)")

    # Assign family to each dataset
    features_df["family"] = features_df["dataset_id"].apply(family_from_dataset_id)
    family_counts = features_df.groupby("family")["dataset_id"].nunique().sort_values(ascending=False)
    logger.info(f"Families and sizes:\n{family_counts.to_string()}")

    # Focus on synthetic families with at least MIN_FAMILY_SIZE datasets.
    synth_families = family_counts[
        family_counts >= MIN_FAMILY_SIZE
    ].index.tolist()
    # Drop real-data pseudo-families for the LOGFO analysis.
    synth_families = [f for f in synth_families if f not in ("openml", "text", "image", "other")]
    logger.info(f"Held-out families ({len(synth_families)}): {synth_families}")

    rows: List[Dict] = []
    t_total = time.time()
    for fam in synth_families:
        test_df = features_df[features_df["family"] == fam]
        train_df = features_df[features_df["family"] != fam]
        n_test_datasets = test_df["dataset_id"].nunique()
        n_train_datasets = train_df["dataset_id"].nunique()

        t0 = time.time()
        model = train_xgb(train_df, feature_cols)
        regret = eval_regret(model, test_df, feature_cols)
        elapsed = time.time() - t0

        logger.info(f"  {fam:<14}  regret={regret:.4f}  "
                    f"train_datasets={n_train_datasets}  test_datasets={n_test_datasets}  ({elapsed:.1f}s)")
        rows.append(dict(
            held_out_family=fam,
            n_train_datasets=n_train_datasets,
            n_test_datasets=n_test_datasets,
            n_test_partitions=len(test_df),
            regret=regret,
        ))

    df = pd.DataFrame(rows).sort_values("regret")
    out_path = OUT_DIR / "logfo_results.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  (total {(time.time()-t_total)/60:.1f}m)")

    print()
    print("=" * 72)
    print("LEAVE-ONE-GENERATOR-FAMILY-OUT — synthetic families with N ≥ 3 datasets")
    print("=" * 72)
    print()
    fmt = "{:<16} {:>7} {:>10} {:>15} {:>10}"
    print(fmt.format("held_out_family", "N_test", "N_partitions", "N_train_datasets", "regret"))
    print("-" * 72)
    for _, r in df.iterrows():
        print(fmt.format(
            r["held_out_family"], int(r["n_test_datasets"]),
            int(r["n_test_partitions"]), int(r["n_train_datasets"]),
            f"{r['regret']:.4f}",
        ))
    print()
    mean_reg = float(df["regret"].mean())
    max_reg = float(df["regret"].max())
    print(f"Mean LOGFO regret:      {mean_reg:.4f}")
    print(f"Max LOGFO regret:       {max_reg:.4f}")
    print(f"Standard CV regret:     0.071 (MetaIVM headline)")
    print()
    if mean_reg <= 0.15:
        verdict = ("Cross-family transfer is robust: mean regret ≤ 0.15 "
                   "when entire generator families are held out.")
    elif mean_reg <= 0.25:
        verdict = ("Cross-family transfer degrades modestly. The random-CV number "
                   "overstates by a real but bounded amount; per-family regret varies "
                   "with structural distance from the training pool.")
    else:
        verdict = ("Cross-family transfer degrades substantially. Random CV likely "
                   "overstates generalization; report LOGFO alongside random-CV in the "
                   "revised main table.")
    print(f"VERDICT: {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    run()
