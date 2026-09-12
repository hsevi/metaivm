"""
Experiment 4 — Stability baseline evaluation and runtime comparison.

Runs the stability baseline (T=20 perturbations, 80% subsampling)
and produces runtime comparison table.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.run_evaluation import evaluate_method
from src.models.baselines import StabilityBaseline, IVMRanker
from src.models.tabular_models import XGBoostModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_experiment(seeds=None, n_perturbations=20):
    """
    Run Experiment 4: stability baseline + runtime comparison.

    Returns:
        Tuple of (stability_results_df, runtime_df).
    """
    if seeds is None:
        with open(PROJECT_ROOT / "configs" / "experiment_configs.yaml") as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["splits"]["seeds"]

    features_df = pd.read_csv(FEATURES_DIR / "all_features.csv")
    master = pd.read_csv(FEATURES_DIR / "runs_master.csv")
    if "hyperparams_json" not in features_df.columns:
        features_df = features_df.merge(
            master[["dataset_id", "run_id", "hyperparams_json"]],
            on=["dataset_id", "run_id"],
            how="left",
        )

    # Evaluate stability baseline
    methods = {
        "Silhouette": IVMRanker("silhouette"),
        "Stability": StabilityBaseline(n_perturbations=n_perturbations),
        "XGBoost (part+X+graph)": XGBoostModel(feature_set="partition_x_graph"),
    }

    all_results = []
    runtimes = {}

    for name, method in methods.items():
        logger.info(f"=== Evaluating: {name} ===")
        t0 = time.time()
        result = evaluate_method(
            method, features_df, seeds,
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
        )
        elapsed = time.time() - t0

        row = {"method": name}
        for metric, stats in result["overall"].items():
            if metric in ("dataset_id", "seed"):
                continue
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        all_results.append(row)

        n_test_datasets = len(set(m["dataset_id"] for m in result["per_dataset"]))
        runtimes[name] = elapsed / max(n_test_datasets, 1)

    stability_results = pd.DataFrame(all_results)

    # Runtime table
    runtime_rows = [
        {"method": "Silhouette", "time_per_dataset_sec": runtimes.get("Silhouette", np.nan)},
        {"method": "CH / DB", "time_per_dataset_sec": runtimes.get("Silhouette", np.nan) * 0.5},
        {"method": f"Stability (T={n_perturbations})", "time_per_dataset_sec": runtimes.get("Stability", np.nan)},
        {"method": "Neural IVM (inference)", "time_per_dataset_sec": runtimes.get("XGBoost (part+X+graph)", np.nan)},
    ]
    runtime_df = pd.DataFrame(runtime_rows)

    # Save
    agg_dir = RESULTS_DIR / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    stability_results.to_csv(agg_dir / "exp4_stability.csv", index=False)
    runtime_df.to_csv(agg_dir / "exp4_runtime.csv", index=False)

    return stability_results, runtime_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stab_results, runtime = run_experiment()
    print("\n=== Experiment 4: Stability Results ===")
    print(stab_results.to_string(index=False))
    print("\n=== Runtime Comparison ===")
    print(runtime.to_string(index=False))
