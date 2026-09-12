"""Protocol-specific permutation importance with bootstrap CIs (Table 12).

Reproduces `results/aggregated/permutation_importance.csv`, referenced in the
paper as Table 12 (`tab:app_perm_importance`).

Protocol (matches paper Section 6 / Appendix B verbatim):
  (i) Fix the seed-0 split of the 5-seed 60/20/20 dataset-level protocol
      (train 133 datasets, val 44 datasets, test 46 datasets).
  (ii) Train the default 28-feature XGBoost once on train+val at that split.
  (iii) For each feature and each of 30 independent random permutations,
       shuffle that feature's column across the test-fold rows and recompute
       the selection regret over the 46 test datasets.
  (iv) Construct a 95% bootstrap interval on the mean Delta-regret across the
       30 permutation runs by resampling those 30 realizations 1000 times with
       replacement (numpy.random, fixed seed=0).

The intervals reflect variability across permutation realizations on a single
split; they are NOT aggregated across folds and NO multiple-testing correction
is applied. Significance is protocol-specific: a feature is starred if its
lower CI bound is strictly positive at four-decimal precision.

Outputs:
  results/aggregated/permutation_importance.csv
    columns: feature, mean_delta, ci_low, ci_high, sig

Run:
  cd path/to/Neural_IVM
  python scripts/permutation_importance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import selection_regret
from src.evaluation.splits import make_splits
from src.models.tabular_models import XGBoostModel, get_feature_columns

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUT_CSV = PROJECT_ROOT / "results" / "aggregated" / "permutation_importance.csv"

N_PERMUTATIONS = 30
N_BOOTSTRAP = 1000
SEED = 0
FEATURE_SET = "partition_x_graph"


def _load_features():
    features_df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    master = pd.read_csv(FEATURES_DIR / "runs_master.csv")
    if "hyperparams_json" not in features_df.columns and "hyperparams_json" in master.columns:
        features_df = features_df.merge(
            master[["dataset_id", "run_id", "hyperparams_json"]],
            on=["dataset_id", "run_id"],
            how="left",
        )
    return features_df


def _selection_regret_on_test(model, test_df, feature_cols):
    """Aggregate selection regret across all test datasets."""
    regrets = []
    for ds_id, ds_df in test_df.groupby("dataset_id"):
        if len(ds_df) < 2:
            continue
        scores = model.predict(ds_df)
        true_amis = ds_df["ami"].values
        regrets.append(selection_regret(true_amis, scores))
    return float(np.mean(regrets))


def main():
    features_df = _load_features()
    dataset_ids = features_df["dataset_id"].unique().tolist()
    splits = make_splits(
        dataset_ids, n_splits=1,
        train_frac=0.6, val_frac=0.2, test_frac=0.2,
        seed=SEED,
    )
    train_ids, val_ids, test_ids = splits[0]
    train_df = features_df[features_df["dataset_id"].isin(train_ids)]
    val_df = features_df[features_df["dataset_id"].isin(val_ids)]
    test_df = features_df[features_df["dataset_id"].isin(test_ids)].copy()

    model = XGBoostModel(feature_set=FEATURE_SET)
    model.fit(train_df, val_df)

    feature_cols = get_feature_columns(train_df, FEATURE_SET)
    base_regret = _selection_regret_on_test(model, test_df, feature_cols)
    print(f"Seed-0 test regret (base): {base_regret:.4f}, N_features = {len(feature_cols)}")

    rng = np.random.RandomState(SEED)
    rows = []
    for j, feat in enumerate(feature_cols):
        original = test_df[feat].values.copy()
        deltas = np.empty(N_PERMUTATIONS)
        for k in range(N_PERMUTATIONS):
            perm = rng.permutation(len(test_df))
            test_df[feat] = original[perm]
            perm_regret = _selection_regret_on_test(model, test_df, feature_cols)
            deltas[k] = perm_regret - base_regret
        test_df[feat] = original

        # 1000-bootstrap 95% CI on the mean delta across 30 permutation runs
        boot = rng.choice(deltas, size=(N_BOOTSTRAP, N_PERMUTATIONS), replace=True).mean(axis=1)
        ci_low = float(np.percentile(boot, 2.5))
        ci_high = float(np.percentile(boot, 97.5))
        mean_delta = float(deltas.mean())
        sig = "*" if round(ci_low, 4) > 0 else ""

        rows.append({
            "feature": feat,
            "mean_delta": mean_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "sig": sig,
        })
        print(f"  [{j + 1:2d}/{len(feature_cols)}] {feat:40s} "
              f"delta={mean_delta:+.4f} CI=[{ci_low:+.4f}, {ci_high:+.4f}] {sig}")

    out = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)
    out.to_csv(OUT_CSV, index=False)
    n_sig = (out["sig"] == "*").sum()
    print(f"\nSaved: {OUT_CSV}")
    print(f"{n_sig} of {len(out)} features have strictly positive lower CI at 4-decimal precision.")


if __name__ == "__main__":
    main()
