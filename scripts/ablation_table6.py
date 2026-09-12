"""Feature-ablation experiment (Table 6 / tab:ablation).

Evaluates XGBoost selection regret and Spearman rho for each feature set in
Table 6, under the shared 5-seed 60/20/20 dataset-level protocol. Every row maps
to a named set in src/models/tabular_models.py::FEATURE_SETS, so the table is
fully reproducible from committed code.

Compact-15 / compact-8 use the feature lists selected by gain importance stable
across training seeds (protocol + lists in Appendix L); they are committed as the
`compact_15` / `compact_8` feature sets.

Run:
  cd path/to/Neural_IVM
  python scripts/ablation_table6.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.run_evaluation import evaluate_method
from src.models.tabular_models import XGBoostModel

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUT = PROJECT_ROOT / "results" / "aggregated" / "exp2_feature_ablation.csv"

# (paper label, FEATURE_SETS key) in Table 6 row order
ROWS = [
    ("Dataset descriptors only", "dataset_desc_only"),
    ("IVM only", "ivm_only"),
    ("Graph only", "graph_only"),
    ("Partition only", "partition_x"),
    ("Partition + dataset", "partition_x_dataset"),
    ("No IVM, no algorithm", "no_ivm_no_algo"),
    ("No IVM features", "no_ivm"),
    ("No algorithm features", "no_algo"),
    ("All features", "all_features"),
    ("Compact: top-15 features", "compact_15"),
    ("Compact: top-8 features", "compact_8"),
]


def main(seeds=(0, 1, 2, 3, 4)):
    df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    master = pd.read_csv(FEATURES_DIR / "runs_master.csv")
    if "hyperparams_json" not in df.columns and "hyperparams_json" in master.columns:
        df = df.merge(master[["dataset_id", "run_id", "hyperparams_json"]],
                      on=["dataset_id", "run_id"], how="left")

    from src.models.tabular_models import get_feature_columns
    rows = []
    for label, key in ROWS:
        d = len(get_feature_columns(df, key))
        res = evaluate_method(XGBoostModel(feature_set=key), df, list(seeds))
        reg = res["overall"]["regret"]
        rho = res["overall"].get("spearman", {"mean": float("nan")})
        rows.append({"feature_set": label, "key": key, "d": d,
                     "regret_mean": reg["mean"], "regret_std": reg["std"],
                     "spearman_mean": rho["mean"]})
        print(f"{label:28s} d={d:2d}  regret={reg['mean']:.4f}  rho={rho['mean']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
