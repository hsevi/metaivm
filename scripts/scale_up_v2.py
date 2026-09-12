"""
Scale-up dataset collection from 116 → 200+ datasets.

Adds:
  1. Real-world datasets from diverse domains:
     - Additional OpenML datasets beyond CC18 (selected for diversity)
     - Image embeddings: MNIST, Fashion-MNIST (PCA-reduced)
     - Text: 20Newsgroups subsets (TF-IDF)
  2. More diverse synthetic datasets (MDCGEN-inspired):
     - Controlled overlap, noise, dimensionality, imbalance
     - Different cluster shapes (ellipsoidal, manifold)

Usage:
    python scripts/scale_up_v2.py --step generate   # generate & save new datasets
    python scripts/scale_up_v2.py --step preprocess  # preprocess new datasets
    python scripts/scale_up_v2.py --step cluster      # run clustering on new datasets
    python scripts/scale_up_v2.py --step all          # do everything
"""

import os
import sys
import logging
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REGISTRY_PATH = DATA_DIR / "dataset_registry.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. REAL-WORLD DATASETS
# ──────────────────────────────────────────────────────────────────────

def _save_dataset(dataset_id, X, y, source, description):
    """Save a dataset to raw/ and return a registry row."""
    ds_dir = RAW_DIR / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "X.npy", X.astype(np.float64))
    np.save(ds_dir / "y_true.npy", y.astype(np.int64))
    return {
        "dataset_id": dataset_id,
        "source": source,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "description": description,
    }


def collect_extra_openml():
    """Additional OpenML datasets NOT in CC18, chosen for diversity."""
    import openml
    from sklearn.preprocessing import LabelEncoder

    # Hand-picked OpenML dataset IDs for diversity
    # These cover different domains: biology, finance, physics, social, etc.
    extra_ids = [
        # Small-medium datasets with diverse structure
        61,     # iris (classic, 150x4, 3 classes)
        1471,   # eeg-eye-state (14980x14, 2 classes)
        1476,   # gas-drift (13910x128, 6 classes)
        1478,   # har (10299x561→PCA, 6 classes - human activity recognition)
        300,    # isolet (7797x617→PCA, 26 classes - spoken letters)
        1466,   # cardiotocography (2126x21, 10 classes)
        1596,   # covertype (subset) - large, many classes
        40927,  # CIFAR-10 (if small version available)
        40668,  # connect-4 (67557x42, 3 classes)
        1169,   # airlines (subset)
        42,     # soybean (307x35, 19 classes)
        36,     # segment (2310x16, 7 classes) - might already have as 40984
        60,     # waveform-5000 (5000x40, 3 classes)
        1510,   # wdbc (569x30, 2 classes)
        1504,   # steel-plates-fault (1941x27, 7 classes)
        40496,  # LED-display (500x7, 10 classes)
        40900,  # Satellite (6435x36, 6 classes)
        40536,  # SpeedDating (8378x120, 2 classes)
        23517,  # numerai28.6 (96320x21, 2 classes - finance)
        4135,   # Amazon_employee_access (32769x9, 2 classes)
        1038,   # gina_agnostic (3468x970→PCA, 2 classes)
        1040,   # sylva_prior (14395x108, 2 classes)
        1120,   # MagicTelescope (19020x10, 2 classes)
        41143,  # porto-seguro (subset, insurance)
        41147,  # albert (425240→subset)
        1457,   # amazon (1500x10000→PCA, 50 classes)
        375,    # JapaneseVowels (9961x14, 9 classes)
        1468,   # cnae-9 (1080x856→PCA, 9 classes)
        679,    # rmftsa_sleepdata (1024x2, 4 classes)
        # More diverse ones
        40685,  # shuttle (58000→subset, 7 classes)
        1515,   # micro-mass (571x1300→PCA, 20 classes)
        41026,  # gina (3153x970→PCA, 2 classes)
        40478,  # climate-model-simulation-crashes
        1558,   # bank-marketing (subset)
    ]

    existing = set()
    if REGISTRY_PATH.exists():
        reg = pd.read_csv(REGISTRY_PATH)
        existing = set(reg["dataset_id"].tolist())

    new_rows = []
    max_samples = 20000  # cap to avoid very long clustering runs
    max_features_raw = 2000  # will PCA-reduce if more

    for did in extra_ids:
        ds_id = f"openml_{did}"
        if ds_id in existing:
            logger.info(f"  {ds_id} already exists, skipping")
            continue

        try:
            dataset = openml.datasets.get_dataset(
                did, download_data=True, download_qualities=False
            )
            X_df, y_ser, _, _ = dataset.get_data(
                target=dataset.default_target_attribute
            )
            if y_ser is None:
                continue

            X = pd.DataFrame(X_df).apply(pd.to_numeric, errors="coerce").values
            le = LabelEncoder()
            y = le.fit_transform(y_ser.astype(str))

            # Drop rows with all NaN
            valid_mask = ~np.isnan(X).all(axis=1)
            X, y = X[valid_mask], y[valid_mask]

            # Subsample if too large
            if X.shape[0] > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(X.shape[0], max_samples, replace=False)
                X, y = X[idx], y[idx]

            # PCA reduce if too many features
            if X.shape[1] > 200:
                from sklearn.decomposition import PCA
                from sklearn.impute import SimpleImputer
                X = SimpleImputer(strategy="median").fit_transform(X)
                n_comp = min(50, X.shape[0] - 1, X.shape[1])
                X = PCA(n_components=n_comp, random_state=42).fit_transform(X)

            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))

            # Filter: at least 100 samples, 2 features, 2 classes
            if n_samples < 100 or n_features < 2 or n_classes < 2:
                continue
            if n_classes > 50:
                continue

            row = _save_dataset(ds_id, X, y, "openml", dataset.name)
            new_rows.append(row)
            logger.info(
                f"  Collected {ds_id} ({dataset.name}): "
                f"n={n_samples}, d={n_features}, k={n_classes}"
            )
        except Exception as e:
            logger.warning(f"  Failed OpenML {did}: {e}")
            continue

    logger.info(f"Collected {len(new_rows)} new OpenML datasets")
    return new_rows


def collect_image_datasets():
    """Add image datasets as PCA embeddings."""
    from sklearn.decomposition import PCA
    from sklearn.datasets import fetch_openml

    existing = set()
    if REGISTRY_PATH.exists():
        reg = pd.read_csv(REGISTRY_PATH)
        existing = set(reg["dataset_id"].tolist())

    new_rows = []

    # --- MNIST subsets ---
    mnist_configs = [
        ("img_mnist_digits_5k", "mnist_784", 5000, None, "MNIST digits 5k PCA-50"),
        ("img_mnist_digits_2k", "mnist_784", 2000, None, "MNIST digits 2k PCA-50"),
    ]

    for ds_id, openml_name, n_sub, classes_subset, desc in mnist_configs:
        if ds_id in existing:
            logger.info(f"  {ds_id} already exists, skipping")
            continue
        try:
            logger.info(f"  Fetching {openml_name} for {ds_id}...")
            data = fetch_openml(openml_name, version=1, as_frame=False, parser="auto")
            X, y = data.data, data.target.astype(int)

            if classes_subset is not None:
                mask = np.isin(y, classes_subset)
                X, y = X[mask], y[mask]

            # Subsample
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), min(n_sub, len(X)), replace=False)
            X, y = X[idx], y[idx]

            # PCA to 50 dims
            X = PCA(n_components=50, random_state=42).fit_transform(X.astype(float))

            row = _save_dataset(ds_id, X, y, "image", desc)
            new_rows.append(row)
            logger.info(f"  Saved {ds_id}: n={X.shape[0]}, d={X.shape[1]}, k={len(np.unique(y))}")
        except Exception as e:
            logger.warning(f"  Failed {ds_id}: {e}")

    # --- Fashion-MNIST ---
    fmnist_configs = [
        ("img_fashion_mnist_5k", "Fashion-MNIST", 5000, None, "Fashion-MNIST 5k PCA-50"),
        ("img_fashion_mnist_2k", "Fashion-MNIST", 2000, None, "Fashion-MNIST 2k PCA-50"),
    ]

    for ds_id, openml_name, n_sub, classes_subset, desc in fmnist_configs:
        if ds_id in existing:
            logger.info(f"  {ds_id} already exists, skipping")
            continue
        try:
            logger.info(f"  Fetching {openml_name} for {ds_id}...")
            data = fetch_openml(openml_name, version=1, as_frame=False, parser="auto")
            X, y = data.data, data.target.astype(int)

            rng = np.random.default_rng(43)
            idx = rng.choice(len(X), min(n_sub, len(X)), replace=False)
            X, y = X[idx], y[idx]

            X = PCA(n_components=50, random_state=42).fit_transform(X.astype(float))

            row = _save_dataset(ds_id, X, y, "image", desc)
            new_rows.append(row)
            logger.info(f"  Saved {ds_id}: n={X.shape[0]}, d={X.shape[1]}, k={len(np.unique(y))}")
        except Exception as e:
            logger.warning(f"  Failed {ds_id}: {e}")

    logger.info(f"Collected {len(new_rows)} image datasets")
    return new_rows


def collect_text_datasets():
    """Add text clustering datasets (20Newsgroups with TF-IDF)."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    existing = set()
    if REGISTRY_PATH.exists():
        reg = pd.read_csv(REGISTRY_PATH)
        existing = set(reg["dataset_id"].tolist())

    new_rows = []

    # Full 20 newsgroups → TF-IDF → SVD-50
    text_configs = [
        (
            "text_20ng_all",
            None,
            5000,
            "20NG all categories TF-IDF SVD-50",
        ),
        (
            "text_20ng_science",
            ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
            2000,
            "20NG science TF-IDF SVD-50",
        ),
        (
            "text_20ng_politics",
            ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
             "talk.religion.misc"],
            2000,
            "20NG politics+religion TF-IDF SVD-50",
        ),
        (
            "text_20ng_comp",
            ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
             "comp.sys.mac.hardware", "comp.windows.x"],
            2000,
            "20NG computers TF-IDF SVD-50",
        ),
        (
            "text_20ng_rec",
            ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
            2000,
            "20NG recreation TF-IDF SVD-50",
        ),
    ]

    for ds_id, categories, n_sub, desc in text_configs:
        if ds_id in existing:
            logger.info(f"  {ds_id} already exists, skipping")
            continue
        try:
            logger.info(f"  Fetching 20newsgroups for {ds_id}...")
            data = fetch_20newsgroups(
                subset="all",
                categories=categories,
                remove=("headers", "footers", "quotes"),
            )
            texts, y = data.data, data.target

            # TF-IDF
            tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
            X_tfidf = tfidf.fit_transform(texts)

            # SVD to 50 dims
            svd = TruncatedSVD(n_components=50, random_state=42)
            X = svd.fit_transform(X_tfidf)

            # Subsample if needed
            if len(X) > n_sub:
                rng = np.random.default_rng(44)
                idx = rng.choice(len(X), n_sub, replace=False)
                X, y = X[idx], y[idx]

            row = _save_dataset(ds_id, X, y, "text", desc)
            new_rows.append(row)
            logger.info(f"  Saved {ds_id}: n={X.shape[0]}, d={X.shape[1]}, k={len(np.unique(y))}")
        except Exception as e:
            logger.warning(f"  Failed {ds_id}: {e}")

    logger.info(f"Collected {len(new_rows)} text datasets")
    return new_rows


# ──────────────────────────────────────────────────────────────────────
# 2. DIVERSE SYNTHETIC DATASETS (MDCGEN-inspired)
# ──────────────────────────────────────────────────────────────────────

def generate_mdcgen_style_synthetics():
    """
    Generate diverse synthetic datasets inspired by Simpson et al.'s use of MDCGEN.
    Systematically vary: dimensionality, n_clusters, overlap, noise, imbalance, shape.
    """
    from sklearn.datasets import make_blobs, make_moons, make_circles

    existing = set()
    if REGISTRY_PATH.exists():
        reg = pd.read_csv(REGISTRY_PATH)
        existing = set(reg["dataset_id"].tolist())

    rng = np.random.default_rng(2024)
    new_rows = []

    def _try_save(ds_id, X, y, desc):
        if ds_id in existing:
            return None
        row = _save_dataset(ds_id, X, y, "synthetic", desc)
        logger.info(f"  Generated {ds_id}: n={X.shape[0]}, d={X.shape[1]}, k={len(np.unique(y))}")
        return row

    # ---- A. Controlled overlap grid (like MDCGEN) ----
    # Vary (dim, k, overlap_std) systematically
    for d in [2, 5, 15, 30]:
        for k in [3, 6, 10]:
            for std_level, std_name in [(0.5, "low"), (1.5, "med"), (3.0, "high")]:
                ds_id = f"synth_grid_d{d}_k{k}_{std_name}overlap"
                n = min(k * 200, 3000)
                X, y = make_blobs(
                    n_samples=n, n_features=d, centers=k,
                    cluster_std=std_level,
                    random_state=rng.integers(0, 2**31),
                )
                row = _try_save(ds_id, X, y, f"grid d={d} k={k} overlap={std_name}")
                if row:
                    new_rows.append(row)

    # ---- B. Ellipsoidal clusters with random covariance ----
    for d in [3, 10, 20]:
        for k in [3, 5, 8]:
            ds_id = f"synth_ellipsoidal_d{d}_k{k}"
            n = k * 200
            centers = rng.standard_normal((k, d)) * 5
            X_parts, y_parts = [], []
            for i in range(k):
                # Random covariance matrix
                A = rng.standard_normal((d, d)) * 0.5
                cov = A @ A.T + np.eye(d) * 0.1
                Xi = rng.multivariate_normal(centers[i], cov, size=n // k)
                X_parts.append(Xi)
                y_parts.append(np.full(n // k, i))
            X = np.vstack(X_parts)
            y = np.concatenate(y_parts)
            row = _try_save(ds_id, X, y, f"ellipsoidal d={d} k={k}")
            if row:
                new_rows.append(row)

    # ---- C. Manifold datasets (Swiss roll, S-curve variants) ----
    from sklearn.datasets import make_swiss_roll, make_s_curve

    # Swiss roll with clusters
    for n_clusters in [3, 5]:
        ds_id = f"synth_swiss_roll_k{n_clusters}"
        X, t = make_swiss_roll(n_samples=1500, noise=0.5, random_state=rng.integers(0, 2**31))
        # Bin the parameter t into clusters
        y = pd.qcut(t, n_clusters, labels=False)
        row = _try_save(ds_id, X, y, f"swiss roll {n_clusters} clusters")
        if row:
            new_rows.append(row)

    # S-curve with clusters
    for n_clusters in [3, 4]:
        ds_id = f"synth_s_curve_k{n_clusters}"
        X, t = make_s_curve(n_samples=1500, noise=0.1, random_state=rng.integers(0, 2**31))
        y = pd.qcut(t, n_clusters, labels=False)
        row = _try_save(ds_id, X, y, f"s-curve {n_clusters} clusters")
        if row:
            new_rows.append(row)

    # ---- D. Varying imbalance levels ----
    for ratio_name, sizes in [
        ("mild", [200, 300, 500]),
        ("moderate", [50, 150, 800]),
        ("extreme", [20, 30, 50, 900]),
        ("many_small", [30, 30, 30, 30, 30, 800]),
    ]:
        for d in [5, 15]:
            ds_id = f"synth_imb_{ratio_name}_d{d}"
            k = len(sizes)
            centers = rng.standard_normal((k, d)) * 4
            X, y = make_blobs(
                n_samples=sizes, n_features=d, centers=centers,
                cluster_std=1.0,
                random_state=rng.integers(0, 2**31),
            )
            row = _try_save(ds_id, X, y, f"imbalanced {ratio_name} d={d}")
            if row:
                new_rows.append(row)

    # ---- E. Noise injection grid ----
    for noise_pct in [0.05, 0.15, 0.25]:
        for d in [2, 10]:
            ds_id = f"synth_noisy_{int(noise_pct*100)}pct_d{d}"
            n_total = 1000
            n_clean = int(n_total * (1 - noise_pct))
            n_noise = n_total - n_clean
            k = 4
            X_clean, y_clean = make_blobs(
                n_samples=n_clean, n_features=d, centers=k,
                cluster_std=1.0,
                random_state=rng.integers(0, 2**31),
            )
            lo = X_clean.min(axis=0) - 3
            hi = X_clean.max(axis=0) + 3
            X_noise = rng.uniform(lo, hi, size=(n_noise, d))
            y_noise = np.full(n_noise, k)
            X = np.vstack([X_clean, X_noise])
            y = np.concatenate([y_clean, y_noise])
            row = _try_save(ds_id, X, y, f"noisy {int(noise_pct*100)}% d={d}")
            if row:
                new_rows.append(row)

    # ---- F. Subspace clusters (high d, few relevant) ----
    for d_rel, d_total in [(2, 50), (3, 100), (5, 200)]:
        ds_id = f"synth_subspace_{d_rel}of{d_total}"
        n, k = 1000, 4
        X_rel, y = make_blobs(
            n_samples=n, n_features=d_rel, centers=k,
            cluster_std=0.8,
            random_state=rng.integers(0, 2**31),
        )
        X_noise = rng.standard_normal((n, d_total - d_rel)) * 0.5
        X = np.hstack([X_rel, X_noise])
        row = _try_save(ds_id, X, y, f"subspace {d_rel}/{d_total}")
        if row:
            new_rows.append(row)

    # ---- G. Mixed-shape clusters (blobs + moons + rings in same dataset) ----
    ds_id = "synth_mixed_shapes_2d"
    if ds_id not in existing:
        # Cluster 0: blob
        X0, _ = make_blobs(n_samples=200, n_features=2, centers=[[5, 5]],
                           cluster_std=0.5, random_state=100)
        y0 = np.zeros(200, dtype=int)
        # Cluster 1-2: moons
        X12, y12 = make_moons(n_samples=400, noise=0.08, random_state=101)
        X12 = X12 * 2 - np.array([1, 0.5])
        y12 = y12 + 1
        # Cluster 3-4: circles
        X34, y34 = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=102)
        X34 = X34 * 1.5 + np.array([5, -3])
        y34 = y34 + 3
        X = np.vstack([X0, X12, X34])
        y = np.concatenate([y0, y12, y34])
        row = _try_save(ds_id, X, y, "mixed shapes 2D")
        if row:
            new_rows.append(row)

    # ---- H. Density-varying clusters ----
    for d in [2, 5]:
        ds_id = f"synth_density_varying_d{d}"
        stds = [0.3, 0.8, 1.5, 2.5, 4.0]
        k = len(stds)
        centers = rng.standard_normal((k, d)) * 6
        X_parts, y_parts = [], []
        for i, s in enumerate(stds):
            Xi, _ = make_blobs(
                n_samples=200, n_features=d,
                centers=[centers[i]], cluster_std=s,
                random_state=rng.integers(0, 2**31),
            )
            X_parts.append(Xi)
            y_parts.append(np.full(200, i))
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        row = _try_save(ds_id, X, y, f"density varying d={d}")
        if row:
            new_rows.append(row)

    logger.info(f"Generated {len(new_rows)} new synthetic datasets")
    return new_rows


# ──────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING & CLUSTERING
# ──────────────────────────────────────────────────────────────────────

def preprocess_new_datasets(new_ids):
    """Preprocess only the new datasets."""
    from src.data.preprocessing import preprocess_dataset

    for i, ds_id in enumerate(new_ids):
        out_dir = PROCESSED_DIR / ds_id
        if (out_dir / "X.npy").exists():
            logger.info(f"  [{i+1}/{len(new_ids)}] {ds_id} already preprocessed, skipping")
            continue
        try:
            logger.info(f"  [{i+1}/{len(new_ids)}] Preprocessing {ds_id}...")
            preprocess_dataset(ds_id)
        except Exception as e:
            logger.warning(f"  Failed to preprocess {ds_id}: {e}")
        gc.collect()


def run_clustering_on_new(new_ids):
    """Run clustering on new datasets."""
    from src.data.clustering_runner import run_all

    # Filter to only datasets that have been preprocessed
    ready_ids = [
        ds_id for ds_id in new_ids
        if (PROCESSED_DIR / ds_id / "X.npy").exists()
    ]
    logger.info(f"Running clustering on {len(ready_ids)} new datasets...")
    run_all(dataset_ids=ready_ids)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["generate", "preprocess", "cluster", "all"],
                        default="all")
    args = parser.parse_args()

    new_rows = []
    new_ids = []

    if args.step in ("generate", "all"):
        logger.info("=" * 60)
        logger.info("STEP 1: Generating new datasets")
        logger.info("=" * 60)

        # 1a. Extra OpenML
        logger.info("\n--- Extra OpenML datasets ---")
        rows_openml = collect_extra_openml()
        new_rows.extend(rows_openml)

        # 1b. Image datasets
        logger.info("\n--- Image datasets ---")
        rows_img = collect_image_datasets()
        new_rows.extend(rows_img)

        # 1c. Text datasets
        logger.info("\n--- Text datasets ---")
        rows_text = collect_text_datasets()
        new_rows.extend(rows_text)

        # 1d. Diverse synthetics
        logger.info("\n--- MDCGEN-style synthetic datasets ---")
        rows_synth = generate_mdcgen_style_synthetics()
        new_rows.extend(rows_synth)

        # Update registry
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if REGISTRY_PATH.exists():
                existing = pd.read_csv(REGISTRY_PATH)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset="dataset_id", keep="first")
            else:
                combined = new_df
            combined.to_csv(REGISTRY_PATH, index=False)
            logger.info(f"\nRegistry updated: {len(combined)} total datasets")
            new_ids = [r["dataset_id"] for r in new_rows]
        else:
            logger.info("No new datasets to add")

    if args.step in ("preprocess", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Preprocessing new datasets")
        logger.info("=" * 60)

        if not new_ids:
            # Figure out which datasets need preprocessing
            reg = pd.read_csv(REGISTRY_PATH)
            new_ids = [
                ds_id for ds_id in reg["dataset_id"]
                if not (PROCESSED_DIR / ds_id / "X.npy").exists()
                and (RAW_DIR / ds_id / "X.npy").exists()
            ]
            logger.info(f"Found {len(new_ids)} datasets needing preprocessing")

        preprocess_new_datasets(new_ids)

    if args.step in ("cluster", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Running clustering on new datasets")
        logger.info("=" * 60)

        if not new_ids:
            # Figure out which datasets need clustering
            reg = pd.read_csv(REGISTRY_PATH)
            master_path = DATA_DIR / "features" / "runs_master.csv"
            if master_path.exists():
                master = pd.read_csv(master_path)
                clustered = set(master["dataset_id"].unique())
            else:
                clustered = set()

            new_ids = [
                ds_id for ds_id in reg["dataset_id"]
                if ds_id not in clustered
                and (PROCESSED_DIR / ds_id / "X.npy").exists()
            ]
            logger.info(f"Found {len(new_ids)} datasets needing clustering")

        run_clustering_on_new(new_ids)

    # Final summary
    if REGISTRY_PATH.exists():
        reg = pd.read_csv(REGISTRY_PATH)
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total datasets in registry: {len(reg)}")
        logger.info(f"By source:\n{reg['source'].value_counts().to_string()}")

        master_path = DATA_DIR / "features" / "runs_master.csv"
        if master_path.exists():
            master = pd.read_csv(master_path)
            logger.info(f"Total clustering runs: {len(master)}")
            logger.info(f"Datasets with runs: {master['dataset_id'].nunique()}")


if __name__ == "__main__":
    main()
