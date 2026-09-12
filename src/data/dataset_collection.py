"""
Dataset collection for Neural IVM.

Downloads OpenML classification datasets and generates synthetic datasets
for use as meta-learning training data for clustering evaluation.
"""

import os
import logging
import hashlib
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yaml
import openml
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _passes_filters(n_samples, n_features, n_classes, filters):
    """Check if a dataset passes the size/dimension/class filters."""
    return (
        filters["min_samples"] <= n_samples <= filters["max_samples"]
        and filters["min_features"] <= n_features <= filters["max_features"]
        and filters["min_classes"] <= n_classes <= filters["max_classes"]
    )


def collect_openml_datasets(config):
    """
    Fetch classification datasets from OpenML.
    Returns list of dicts with dataset metadata and (X, y) arrays.
    """
    openml_cfg = config["openml"]
    filters = openml_cfg["filters"]
    collected = []

    # Fetch CC18 suite datasets
    suite = openml.study.get_suite(openml_cfg["suite_id"])
    candidate_ids = list(suite.data) + openml_cfg.get("extra_ids", [])
    candidate_ids = sorted(set(candidate_ids))

    logger.info(f"Checking {len(candidate_ids)} OpenML dataset candidates...")

    for did in candidate_ids:
        try:
            dataset = openml.datasets.get_dataset(
                did, download_data=True, download_qualities=False
            )
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute
            )

            if y is None:
                logger.debug(f"Dataset {did}: no target, skipping")
                continue

            # Convert to numpy
            X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").values
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))

            if not _passes_filters(n_samples, n_features, n_classes, filters):
                logger.debug(
                    f"Dataset {did}: filtered out "
                    f"(n={n_samples}, d={n_features}, k={n_classes})"
                )
                continue

            dataset_id = f"openml_{did}"
            collected.append(
                {
                    "dataset_id": dataset_id,
                    "source": "openml",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_classes": n_classes,
                    "description": dataset.name,
                    "X": X,
                    "y": y,
                }
            )
            logger.info(
                f"Collected {dataset_id} ({dataset.name}): "
                f"n={n_samples}, d={n_features}, k={n_classes}"
            )

        except Exception as e:
            logger.warning(f"Failed to fetch OpenML dataset {did}: {e}")
            continue

    logger.info(f"Collected {len(collected)} OpenML datasets")
    return collected


def _generate_blobs_variants(rng):
    """Generate blob-based datasets with varying properties."""
    datasets = []
    configs = [
        # (n_samples, n_features, n_centers, cluster_std, description)
        (500, 2, 3, 1.0, "blobs_easy_2d"),
        (500, 2, 5, 1.5, "blobs_overlap_2d"),
        (500, 2, 10, 0.5, "blobs_many_2d"),
        (1000, 10, 3, 1.0, "blobs_easy_10d"),
        (1000, 10, 7, 2.0, "blobs_overlap_10d"),
        (1000, 50, 5, 1.0, "blobs_highd_50"),
        (2000, 100, 10, 1.5, "blobs_highd_100"),
        (5000, 2, 15, 0.8, "blobs_large_2d"),
        (5000, 20, 20, 1.0, "blobs_large_20d"),
        (300, 2, 3, 3.0, "blobs_high_overlap"),
        (500, 5, 4, 0.3, "blobs_well_separated"),
        (1000, 2, 6, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], "blobs_varying_std"),
    ]

    for n, d, k, std, desc in configs:
        X, y = make_blobs(
            n_samples=n,
            n_features=d,
            centers=k,
            cluster_std=std,
            random_state=rng.integers(0, 2**31),
        )
        datasets.append(
            {
                "dataset_id": f"synth_{desc}",
                "source": "synthetic",
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_classes": k,
                "description": desc,
                "X": X,
                "y": y,
            }
        )

    return datasets


def _generate_nonconvex(rng):
    """Generate non-convex cluster datasets (moons, circles)."""
    datasets = []

    # Moons with varying noise
    for noise in [0.05, 0.15, 0.3]:
        n = 1000
        X, y = make_moons(n_samples=n, noise=noise, random_state=rng.integers(0, 2**31))
        desc = f"moons_noise{noise:.2f}"
        datasets.append(
            {
                "dataset_id": f"synth_{desc}",
                "source": "synthetic",
                "n_samples": n,
                "n_features": 2,
                "n_classes": 2,
                "description": desc,
                "X": X,
                "y": y,
            }
        )

    # Circles with varying noise and factor
    for noise, factor in [(0.05, 0.5), (0.1, 0.3), (0.15, 0.7)]:
        n = 1000
        X, y = make_circles(
            n_samples=n,
            noise=noise,
            factor=factor,
            random_state=rng.integers(0, 2**31),
        )
        desc = f"circles_n{noise:.2f}_f{factor:.1f}"
        datasets.append(
            {
                "dataset_id": f"synth_{desc}",
                "source": "synthetic",
                "n_samples": n,
                "n_features": 2,
                "n_classes": 2,
                "description": desc,
                "X": X,
                "y": y,
            }
        )

    return datasets


def _generate_imbalanced(rng):
    """Generate datasets with class imbalance."""
    datasets = []

    imbalance_configs = [
        ([100, 100, 500], 2, "imb_1_1_5_2d"),
        ([50, 200, 500, 1000], 5, "imb_extreme_5d"),
        ([100, 100, 100, 100, 1000], 10, "imb_one_large_10d"),
        ([30, 30, 30, 500], 3, "imb_many_small_3d"),
    ]

    for sizes, d, desc in imbalance_configs:
        k = len(sizes)
        # Generate centers explicitly since n_samples is a list
        rs = rng.integers(0, 2**31)
        centers = rng.standard_normal((k, d)) * 5
        X, y = make_blobs(
            n_samples=sizes,
            n_features=d,
            centers=centers,
            cluster_std=1.0,
            random_state=rs,
        )
        datasets.append(
            {
                "dataset_id": f"synth_{desc}",
                "source": "synthetic",
                "n_samples": X.shape[0],
                "n_features": d,
                "n_classes": len(sizes),
                "description": desc,
                "X": X,
                "y": y,
            }
        )

    return datasets


def _generate_noisy(rng):
    """Generate datasets with uniform noise points added."""
    datasets = []

    noise_configs = [
        (500, 2, 3, 0.1, "noisy_10pct"),
        (500, 2, 3, 0.3, "noisy_30pct"),
        (1000, 5, 5, 0.2, "noisy_20pct_5d"),
    ]

    for n, d, k, noise_frac, desc in noise_configs:
        n_clean = int(n * (1 - noise_frac))
        n_noise = n - n_clean
        X_clean, y_clean = make_blobs(
            n_samples=n_clean,
            n_features=d,
            centers=k,
            cluster_std=1.0,
            random_state=rng.integers(0, 2**31),
        )
        # Uniform noise in the bounding box of the clean data
        lo = X_clean.min(axis=0) - 2
        hi = X_clean.max(axis=0) + 2
        X_noise = rng.uniform(lo, hi, size=(n_noise, d))
        y_noise = np.full(n_noise, k)  # noise gets its own label

        X = np.vstack([X_clean, X_noise])
        y = np.concatenate([y_clean, y_noise])

        datasets.append(
            {
                "dataset_id": f"synth_{desc}",
                "source": "synthetic",
                "n_samples": X.shape[0],
                "n_features": d,
                "n_classes": k + 1,
                "description": desc,
                "X": X,
                "y": y,
            }
        )

    return datasets


def _generate_subspace(rng):
    """Generate datasets where clusters exist in subspaces (irrelevant features)."""
    datasets = []

    configs = [
        (500, 2, 18, 3, "subspace_2of20"),
        (1000, 5, 45, 5, "subspace_5of50"),
    ]

    for n, d_relevant, d_noise, k, desc in configs:
        X_rel, y = make_blobs(
            n_samples=n,
            n_features=d_relevant,
            centers=k,
            cluster_std=1.0,
            random_state=rng.integers(0, 2**31),
        )
        X_noise = rng.standard_normal((n, d_noise))
        X = np.hstack([X_rel, X_noise])

        datasets.append(
            {
                "dataset_id": f"synth_{desc}",
                "source": "synthetic",
                "n_samples": n,
                "n_features": d_relevant + d_noise,
                "n_classes": k,
                "description": desc,
                "X": X,
                "y": y,
            }
        )

    return datasets


def _generate_elongated(rng):
    """Generate datasets with elongated (anisotropic) clusters."""
    datasets = []

    n, d, k = 1000, 2, 3
    X, y = make_blobs(
        n_samples=n,
        n_features=d,
        centers=k,
        cluster_std=1.0,
        random_state=rng.integers(0, 2**31),
    )
    # Apply a random linear transformation to elongate clusters
    transform = np.array([[2.0, 0.5], [0.5, 1.0]])
    X = X @ transform
    datasets.append(
        {
            "dataset_id": "synth_elongated_2d",
            "source": "synthetic",
            "n_samples": n,
            "n_features": d,
            "n_classes": k,
            "description": "elongated_2d",
            "X": X,
            "y": y,
        }
    )

    # Higher-dimensional elongated
    n, d, k = 1000, 10, 5
    X, y = make_blobs(
        n_samples=n,
        n_features=d,
        centers=k,
        cluster_std=1.0,
        random_state=rng.integers(0, 2**31),
    )
    transform = rng.standard_normal((d, d))
    U, _, Vt = np.linalg.svd(transform, full_matrices=False)
    S = np.diag(np.linspace(0.2, 3.0, d))
    transform = U @ S @ Vt
    X = X @ transform
    datasets.append(
        {
            "dataset_id": "synth_elongated_10d",
            "source": "synthetic",
            "n_samples": n,
            "n_features": d,
            "n_classes": k,
            "description": "elongated_10d",
            "X": X,
            "y": y,
        }
    )

    return datasets


def collect_synthetic_datasets(seed=42):
    """Generate all synthetic datasets."""
    rng = np.random.default_rng(seed)
    datasets = []

    datasets.extend(_generate_blobs_variants(rng))
    datasets.extend(_generate_nonconvex(rng))
    datasets.extend(_generate_imbalanced(rng))
    datasets.extend(_generate_noisy(rng))
    datasets.extend(_generate_subspace(rng))
    datasets.extend(_generate_elongated(rng))

    logger.info(f"Generated {len(datasets)} synthetic datasets")
    return datasets


def save_raw_datasets(datasets):
    """Save raw (X, y) to disk and return registry rows."""
    registry_rows = []

    for ds in datasets:
        ds_dir = RAW_DIR / ds["dataset_id"]
        ds_dir.mkdir(parents=True, exist_ok=True)

        np.save(ds_dir / "X.npy", ds["X"])
        np.save(ds_dir / "y_true.npy", ds["y"])

        registry_rows.append(
            {
                "dataset_id": ds["dataset_id"],
                "source": ds["source"],
                "n_samples": ds["n_samples"],
                "n_features": ds["n_features"],
                "n_classes": ds["n_classes"],
                "description": ds["description"],
            }
        )

    return registry_rows


def collect_all(skip_openml=False):
    """
    Main entry point: collect all datasets, save to disk, write registry.

    Args:
        skip_openml: If True, skip OpenML download (useful for testing).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_datasets = []

    # OpenML datasets
    if not skip_openml:
        with open(CONFIGS_DIR / "dataset_list.yaml") as f:
            config = yaml.safe_load(f)
        openml_datasets = collect_openml_datasets(config)
        all_datasets.extend(openml_datasets)
    else:
        logger.info("Skipping OpenML datasets")

    # Synthetic datasets
    synthetic_datasets = collect_synthetic_datasets(seed=42)
    all_datasets.extend(synthetic_datasets)

    # Save and create registry
    registry_rows = save_raw_datasets(all_datasets)
    registry = pd.DataFrame(registry_rows)
    registry_path = DATA_DIR / "dataset_registry.csv"
    registry.to_csv(registry_path, index=False)
    logger.info(f"Saved registry with {len(registry)} datasets to {registry_path}")

    return registry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    registry = collect_all()
    print(f"\nCollected {len(registry)} datasets total")
    print(registry.groupby("source").size())
