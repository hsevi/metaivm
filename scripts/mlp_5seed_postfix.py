"""MLP 5-seed rerun under the leakage-fixed training procedure.

Reproduces the MLP row of Table 1 (`tab:main`) and the per-seed values in
Table 15 (`tab:app_seeds`, Appendix D):

  MLP regret = 0.082 +/- 0.115 (mean +/- std over 230 seed-dataset pairs)
  Per-seed regret means: 0.093 / 0.096 / 0.096 / 0.056 / 0.070
  Per-seed Spearman means: 0.614 / 0.640 / 0.721 / 0.734 / 0.733

The leakage fix is documented in `src/models/tabular_models.py::MLPModel`:
  - sklearn's internal early_stopping is disabled (`early_stopping=False`),
    so no partition-level random split happens inside sklearn's MLPRegressor;
  - the MLP fits on `X_train` only (not the pooled train+val matrix);
  - outer arch/lr model selection uses the dataset-level held-out val fold.

Outputs:
  results/aggregated/mlp_5seed_postfix.csv        (per (seed, dataset) row: all 5 metrics)
  results/aggregated/mlp_5seed_postfix.summary.json (aggregate + per-seed summary)

Run:
  cd path/to/Neural_IVM
  python scripts/mlp_5seed_postfix.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.run_evaluation import evaluate_method
from src.models.tabular_models import MLPModel

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "mlp_5seed_postfix.csv"
OUT_SUMMARY = OUT_DIR / "mlp_5seed_postfix.summary.json"


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


def main(seeds=(0, 1, 2, 3, 4), max_iter=200):
    features_df = _load_features()

    t0 = time.time()
    mlp = MLPModel(feature_set="partition_x_graph", max_iter=max_iter)
    result = evaluate_method(
        mlp, features_df, list(seeds),
        train_frac=0.6, val_frac=0.2, test_frac=0.2,
    )
    elapsed = time.time() - t0

    per = pd.DataFrame(result["per_dataset"])
    overall = result["overall"]

    print(f"=== MLP 5-seed rerun (post leakage fix), wall-time {elapsed:.1f}s ===")
    print(f"N seed-dataset pairs: {len(per)}")
    for metric in ("regret", "eps_success", "ndcg_at_k", "spearman", "kendall"):
        if metric in overall:
            m, s = overall[metric]["mean"], overall[metric]["std"]
            print(f"  {metric:14s}: {m:.4f} +/- {s:.4f}")

    per_seed = per.groupby("seed").agg(
        {"regret": "mean", "spearman": "mean"}
    ).reset_index()
    print("\nPer-seed means:")
    print(per_seed.to_string(index=False))
    print(f"Std across 5 seed means (regret): {per_seed['regret'].std():.4f}")

    per.to_csv(OUT_CSV, index=False)
    with open(OUT_SUMMARY, "w") as f:
        json.dump(
            {
                metric: {
                    "mean": float(overall[metric]["mean"]),
                    "std": float(overall[metric]["std"]),
                }
                for metric in ("regret", "eps_success", "ndcg_at_k", "spearman", "kendall")
                if metric in overall
            }
            | {
                "n_pairs": len(per),
                "per_seed": per_seed.to_dict(orient="records"),
                "std_across_seed_means_regret": float(per_seed["regret"].std()),
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {OUT_CSV}\n       {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
