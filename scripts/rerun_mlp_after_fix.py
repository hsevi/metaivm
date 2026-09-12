"""
Re-run MetaIVM (MLP) after fixing the validation-leakage bug in
`src.models.tabular_models.MLPModel`. Compare against the old Table-1 number
(0.070) to verify the fix does not materially change the headline result.

Uses the same 5-seed 60/20/20 protocol and default 28-feature set.

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -u -m scripts.rerun_mlp_after_fix
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
from src.models.tabular_models import MLPModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_CSV = DATA_DIR / "features" / "all_features.csv"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"

SEEDS = [0, 1, 2, 3, 4]


def eval_regret(preds: np.ndarray, ds_ids: np.ndarray, true_ami: np.ndarray) -> float:
    df = pd.DataFrame(dict(ds_id=ds_ids, pred=preds, ami=true_ami))
    regrets = []
    for _, sub in df.groupby("ds_id"):
        if len(sub) < 2:
            continue
        picked = sub.iloc[int(sub["pred"].values.argmax())]["ami"]
        best = sub["ami"].max()
        regrets.append(best - picked)
    return float(np.mean(regrets)) if regrets else float("nan")


def main() -> None:
    features_df = pd.read_csv(FEATURES_CSV)
    dataset_ids = sorted(features_df["dataset_id"].unique())

    regrets = []
    for seed in SEEDS:
        splits = make_splits(dataset_ids, n_splits=1, seed=seed)
        train_ids, val_ids, test_ids = splits[0]
        train_df = features_df[features_df["dataset_id"].isin(train_ids)]
        val_df = features_df[features_df["dataset_id"].isin(val_ids)]
        test_df = features_df[features_df["dataset_id"].isin(test_ids)]

        t = time.time()
        # max_iter=200: sufficient for convergence on 28-feature tabular data,
        # and needed because with early_stopping=False (the leakage fix) the
        # default max_iter=500 makes the 4x2 grid × 5 seeds take hours.
        model = MLPModel(feature_set="partition_x_graph", max_iter=200)
        model.fit(train_df, val_df)
        preds = model.predict(test_df)
        r = eval_regret(preds, test_df["dataset_id"].values, test_df["ami"].values)
        logger.info(f"  seed {seed}: MLP regret = {r:.4f}  ({time.time()-t:.1f}s)")
        regrets.append(r)

    reg_arr = np.array(regrets)
    mean = float(reg_arr.mean())
    std = float(reg_arr.std())

    df = pd.DataFrame(dict(seed=SEEDS, mlp_regret_after_fix=regrets))
    df.to_csv(OUT_DIR / "mlp_rerun_after_fix.csv", index=False)

    print()
    print("=" * 60)
    print("MLP RE-RUN AFTER VALIDATION-LEAKAGE FIX")
    print("=" * 60)
    print(f"Per-seed regret: {[f'{r:.4f}' for r in regrets]}")
    print(f"Mean regret:     {mean:.4f} ± {std:.4f}")
    print(f"Table 1 old:     0.072 ± 0.098")
    print(f"Delta:           {mean - 0.072:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
