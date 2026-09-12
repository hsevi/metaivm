"""
Scale up: add more synthetic datasets and run clustering on all missing datasets.
Goal: go from 40 -> 80+ usable datasets.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def generate_extra_synthetic(seed=123):
    """Generate additional synthetic datasets for more diversity."""
    from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification

    rng = np.random.default_rng(seed)
    datasets = []

    # --- Swiss roll inspired: 2D spirals ---
    for n_arms, noise in [(2, 0.1), (3, 0.05), (3, 0.2)]:
        n = 800
        t = np.linspace(0, 4 * np.pi, n // n_arms)
        X_list, y_list = [], []
        for arm in range(n_arms):
            offset = arm * 2 * np.pi / n_arms
            x1 = (t + offset) * np.cos(t + offset) + rng.normal(0, noise, len(t))
            x2 = (t + offset) * np.sin(t + offset) + rng.normal(0, noise, len(t))
            X_list.append(np.column_stack([x1, x2]))
            y_list.append(np.full(len(t), arm))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        desc = f"spirals_{n_arms}arms_n{noise:.2f}"
        datasets.append({"dataset_id": f"synth_{desc}", "source": "synthetic",
                         "n_samples": len(y), "n_features": 2, "n_classes": n_arms,
                         "description": desc, "X": X, "y": y})

    # --- Varying density blobs ---
    for d in [2, 5, 15]:
        n_clusters = 5
        centers = rng.standard_normal((n_clusters, d)) * 8
        stds = rng.uniform(0.3, 3.0, n_clusters)
        sizes = rng.integers(80, 400, n_clusters)
        X_list, y_list = [], []
        for c in range(n_clusters):
            X_c = rng.normal(centers[c], stds[c], (sizes[c], d))
            X_list.append(X_c)
            y_list.append(np.full(sizes[c], c))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        desc = f"varying_density_{d}d"
        datasets.append({"dataset_id": f"synth_{desc}", "source": "synthetic",
                         "n_samples": len(y), "n_features": d, "n_classes": n_clusters,
                         "description": desc, "X": X, "y": y})

    # --- Nested rings in 2D ---
    for n_rings in [3, 5]:
        n_per_ring = 300
        X_list, y_list = [], []
        for r_idx in range(n_rings):
            radius = 1.0 + r_idx * 1.5
            theta = rng.uniform(0, 2 * np.pi, n_per_ring)
            x1 = radius * np.cos(theta) + rng.normal(0, 0.15, n_per_ring)
            x2 = radius * np.sin(theta) + rng.normal(0, 0.15, n_per_ring)
            X_list.append(np.column_stack([x1, x2]))
            y_list.append(np.full(n_per_ring, r_idx))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        desc = f"nested_rings_{n_rings}"
        datasets.append({"dataset_id": f"synth_{desc}", "source": "synthetic",
                         "n_samples": len(y), "n_features": 2, "n_classes": n_rings,
                         "description": desc, "X": X, "y": y})

    # --- High-dimensional blobs with more clusters ---
    for d, k in [(20, 15), (30, 10), (50, 20)]:
        n = 2000
        X, y = make_blobs(n_samples=n, n_features=d, centers=k, cluster_std=1.5,
                          random_state=rng.integers(0, 2**31))
        desc = f"blobs_{d}d_{k}k"
        datasets.append({"dataset_id": f"synth_{desc}", "source": "synthetic",
                         "n_samples": n, "n_features": d, "n_classes": k,
                         "description": desc, "X": X, "y": y})

    # --- Anisotropic clusters (random covariance) ---
    for d, k in [(2, 4), (5, 3), (10, 6)]:
        n_per = 200
        X_list, y_list = [], []
        for c in range(k):
            center = rng.standard_normal(d) * 6
            # Random covariance
            A = rng.standard_normal((d, d)) * 0.5
            cov = A @ A.T + 0.1 * np.eye(d)
            X_c = rng.multivariate_normal(center, cov, n_per)
            X_list.append(X_c)
            y_list.append(np.full(n_per, c))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        desc = f"aniso_{d}d_{k}k"
        datasets.append({"dataset_id": f"synth_{desc}", "source": "synthetic",
                         "n_samples": len(y), "n_features": d, "n_classes": k,
                         "description": desc, "X": X, "y": y})

    # --- Moons + blobs combined ---
    n = 600
    X_moons, y_moons = make_moons(n_samples=400, noise=0.1,
                                   random_state=rng.integers(0, 2**31))
    X_blob, y_blob = make_blobs(n_samples=200, n_features=2, centers=[[3, 0]],
                                 cluster_std=0.5, random_state=rng.integers(0, 2**31))
    X = np.vstack([X_moons, X_blob])
    y = np.concatenate([y_moons, y_blob + 2])
    datasets.append({"dataset_id": "synth_moons_plus_blob", "source": "synthetic",
                     "n_samples": len(y), "n_features": 2, "n_classes": 3,
                     "description": "moons_plus_blob", "X": X, "y": y})

    # --- Uniform noise only (adversarial: no clusters) ---
    for d in [2, 10]:
        n = 500
        X = rng.uniform(-5, 5, (n, d))
        # Fake labels via kmeans for ground truth (or just assign random)
        from sklearn.cluster import KMeans
        y = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X)
        desc = f"uniform_noise_{d}d"
        datasets.append({"dataset_id": f"synth_{desc}", "source": "synthetic",
                         "n_samples": n, "n_features": d, "n_classes": 3,
                         "description": desc, "X": X, "y": y})

    # --- Varying n_clusters blobs (same geometry, different k) ---
    for k in [2, 4, 8, 12, 20]:
        n = max(k * 100, 500)
        X, y = make_blobs(n_samples=n, n_features=5, centers=k, cluster_std=1.0,
                          random_state=rng.integers(0, 2**31))
        desc = f"blobs_5d_k{k}"
        datasets.append({"dataset_id": f"synth_{desc}", "source": "synthetic",
                         "n_samples": n, "n_features": 5, "n_classes": k,
                         "description": desc, "X": X, "y": y})

    # --- Elongated with varying aspect ratios ---
    for aspect in [5, 10, 20]:
        n, d, k = 800, 3, 4
        X, y = make_blobs(n_samples=n, n_features=d, centers=k, cluster_std=1.0,
                          random_state=rng.integers(0, 2**31))
        transform = np.diag([aspect, 1.0, 1.0])
        X = X @ transform
        desc = f"elongated_aspect{aspect}_3d"
        datasets.append({"dataset_id": f"synth_{desc}", "source": "synthetic",
                         "n_samples": n, "n_features": d, "n_classes": k,
                         "description": desc, "X": X, "y": y})

    logger.info(f"Generated {len(datasets)} extra synthetic datasets")
    return datasets


def main():
    # Step 1: Generate extra synthetic datasets
    logger.info("=== Step 1: Generating extra synthetic datasets ===")
    new_datasets = generate_extra_synthetic(seed=123)

    # Save raw data
    registry_rows = []
    for ds in new_datasets:
        ds_dir = RAW_DIR / ds["dataset_id"]
        ds_dir.mkdir(parents=True, exist_ok=True)
        np.save(ds_dir / "X.npy", ds["X"])
        np.save(ds_dir / "y_true.npy", ds["y"])
        registry_rows.append({
            "dataset_id": ds["dataset_id"],
            "source": ds["source"],
            "n_samples": ds["n_samples"],
            "n_features": ds["n_features"],
            "n_classes": ds["n_classes"],
            "description": ds["description"],
        })

    # Update registry (append new datasets)
    reg_path = DATA_DIR / "dataset_registry.csv"
    existing_reg = pd.read_csv(reg_path)
    existing_ids = set(existing_reg["dataset_id"])
    new_rows = [r for r in registry_rows if r["dataset_id"] not in existing_ids]
    if new_rows:
        updated_reg = pd.concat([existing_reg, pd.DataFrame(new_rows)], ignore_index=True)
        updated_reg.to_csv(reg_path, index=False)
        logger.info(f"Added {len(new_rows)} new datasets to registry (total: {len(updated_reg)})")
    else:
        logger.info("All synthetic datasets already in registry")
        updated_reg = existing_reg

    # Step 2: Preprocess new datasets
    logger.info("\n=== Step 2: Preprocessing new datasets ===")
    from src.data.preprocessing import preprocess_dataset

    new_preprocessed = 0
    for _, row in updated_reg.iterrows():
        ds_id = row["dataset_id"]
        proc_dir = PROCESSED_DIR / ds_id
        if (proc_dir / "X.npy").exists():
            continue  # already preprocessed
        try:
            preprocess_dataset(ds_id)
            new_preprocessed += 1
        except Exception as e:
            logger.warning(f"Failed to preprocess {ds_id}: {e}")
    logger.info(f"Preprocessed {new_preprocessed} new datasets")

    # Step 3: Run clustering on all datasets that don't have runs yet
    logger.info("\n=== Step 3: Running clustering on missing datasets ===")
    from src.data.clustering_runner import run_all
    master = run_all()

    logger.info(f"\n=== DONE ===")
    logger.info(f"Total datasets in registry: {len(updated_reg)}")
    logger.info(f"Total runs in master: {len(master)}")
    logger.info(f"Datasets with runs: {master['dataset_id'].nunique()}")

    # Step 4: Reassemble features
    logger.info("\n=== Step 4: Reassembling features ===")
    from src.features.assemble_features import assemble_all
    features_df = assemble_all()
    logger.info(f"Feature table: {features_df.shape}")


if __name__ == "__main__":
    main()
