"""DBCV and S_Dbw selection regret under the paper's 5-seed 60/20/20 protocol.

DBCV and S_Dbw are fixed (training-free) internal validity measures, but to be
apples-to-apples with every other method in Table 2 they are scored on the SAME
46-test-datasets-per-seed folds (230 seed-dataset test evaluations) as MetaIVM,
CH, Silhouette, and DB. This script merges the per-partition CVI values from
compute_density_cvis.py into the feature frame and evaluates a fixed-IVM ranker
through the shared evaluate_method / make_splits code.

Selection direction:
  - DBCV: higher is better (density-based validity index)  -> argmax(dbcv)
  - S_Dbw: lower is better (scatter+density dispersion)     -> argmax(-s_dbw)
Degenerate partitions (NaN CVI: trivial / all-noise / numerical error) receive a
worst-case sentinel (-1e10) so they are never selected, matching Table 17.

Outputs:
  results/aggregated/density_cvi_regret.csv        (per (seed,dataset) regret)
  results/aggregated/density_cvi_regret.summary.json

Run:
  cd path/to/Neural_IVM
  python scripts/density_cvi_regret.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.run_evaluation import evaluate_method

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
AGG = PROJECT_ROOT / "results" / "aggregated"
DENSITY = AGG / "density_cvis.csv"


class DensityCVIRanker:
    """Fixed-IVM ranker over a per-partition density CVI column."""

    def __init__(self, col, negate=False):
        self.col = col
        self.negate = negate

    def fit(self, train_df, val_df):
        pass

    def predict(self, test_df):
        s = test_df[self.col].values.astype(float).copy()
        if self.negate:
            s = -s
        return np.nan_to_num(s, nan=-1e10)


def main(seeds=(0, 1, 2, 3, 4)):
    df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    dens = pd.read_csv(DENSITY)[["dataset_id", "run_id", "dbcv", "s_dbw"]]
    df = df.merge(dens, on=["dataset_id", "run_id"], how="left")

    out = {}
    for name, col, negate in [("DBCV", "dbcv", False), ("S_Dbw", "s_dbw", True)]:
        res = evaluate_method(
            DensityCVIRanker(col, negate=negate), df, list(seeds),
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
        )
        reg = res["overall"]["regret"]
        out[name] = {"regret_mean": reg["mean"], "regret_std": reg["std"],
                     "n_pairs": len(res["per_dataset"])}
        print(f"{name:6s} regret = {reg['mean']:.4f} +/- {reg['std']:.4f}  "
              f"(over {len(res['per_dataset'])} seed-dataset test evaluations)")

    with open(AGG / "density_cvi_regret.summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved summary to {AGG / 'density_cvi_regret.summary.json'}")


if __name__ == "__main__":
    main()
