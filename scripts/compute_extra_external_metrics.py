"""
Compute ARI, NMI, V-measure for all existing clustering runs
from saved partition labels. Adds columns to runs_master.csv.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = DATA_DIR / "clustering_runs"
FEATURES_DIR = DATA_DIR / "features"


def main():
    master_path = FEATURES_DIR / "runs_master.csv"
    master = pd.read_csv(master_path)
    logger.info(f"Loaded {len(master)} runs from {master_path}")

    # Check if already computed
    if "ari" in master.columns and master["ari"].notna().mean() > 0.9:
        logger.info("ARI/NMI/V-measure already computed for most runs, skipping")
        return

    ari_list = []
    nmi_list = []
    vmeasure_list = []

    n_missing = 0
    for i, row in master.iterrows():
        ds_id = row["dataset_id"]
        algo = row["algo"]
        run_id = row["run_id"]

        # Load saved labels
        labels_path = RUNS_DIR / ds_id / f"{algo}_{run_id}.npy"
        if not labels_path.exists():
            ari_list.append(np.nan)
            nmi_list.append(np.nan)
            vmeasure_list.append(np.nan)
            n_missing += 1
            continue

        labels = np.load(labels_path)

        # Load ground truth
        y_path = PROCESSED_DIR / ds_id / "y_true.npy"
        if not y_path.exists():
            ari_list.append(np.nan)
            nmi_list.append(np.nan)
            vmeasure_list.append(np.nan)
            n_missing += 1
            continue

        y_true = np.load(y_path)

        # Compute metrics
        try:
            ari_list.append(adjusted_rand_score(y_true, labels))
            nmi_list.append(normalized_mutual_info_score(y_true, labels))
            vmeasure_list.append(v_measure_score(y_true, labels))
        except Exception as e:
            ari_list.append(np.nan)
            nmi_list.append(np.nan)
            vmeasure_list.append(np.nan)
            n_missing += 1

        if (i + 1) % 2000 == 0:
            logger.info(f"  Processed {i+1}/{len(master)} runs")

    master["ari"] = ari_list
    master["nmi"] = nmi_list
    master["v_measure"] = vmeasure_list

    master.to_csv(master_path, index=False)
    logger.info(f"Saved updated master with ARI/NMI/V-measure. Missing: {n_missing}/{len(master)}")

    # Quick summary
    for metric in ["ami", "ari", "nmi", "v_measure"]:
        vals = master[metric].dropna()
        logger.info(f"  {metric}: mean={vals.mean():.3f}, median={vals.median():.3f}, n={len(vals)}")


if __name__ == "__main__":
    main()
