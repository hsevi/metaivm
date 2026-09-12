"""MetaIVM: a scikit-learn-style interface for label-free clustering model selection.

Typical use (no ground-truth labels needed at deployment):

    from src.metaivm import MetaIVM
    from sklearn.cluster import KMeans, DBSCAN

    selector = MetaIVM.from_pretrained()              # trained on the 223-dataset benchmark
    candidates = [KMeans(k).fit_predict(X) for k in (2, 5, 10)]
    candidates.append(DBSCAN(eps=0.5).fit_predict(X))

    best = selector.select(X, candidates)             # -> the best label array
    scores = selector.score(X, candidates)            # -> predicted quality per candidate

The predicted "quality" is MetaIVM's estimate of external agreement (AMI) between a
candidate partition and the (unavailable) ground truth; higher is better. Selection
picks the candidate with the highest predicted quality.

Design notes
------------
* Features are computed fully in memory from (X, labels): the 28-dimensional
  `partition_x_graph` map used as the paper's default (12 partition-structural + 10
  partition-data interaction + 6 kNN-graph features). X is standardized and a
  symmetric kNN graph (k=10) is built exactly as in the training pipeline
  (`src/data/preprocessing.py`), so deployment features match training features.
* The default model is gradient-boosted trees (XGBoost, the paper's 0.071 model).
  If XGBoost is not installed, MetaIVM transparently falls back to Ridge (0.088),
  which is dependency-light and fully deterministic.
* `from_pretrained()` trains once on the bundled benchmark and caches the fitted
  selector under `models/`, so subsequent calls load instantly.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.features.partition_features import compute_partition_features
from src.features.partition_x_features import compute_partition_x_features
from src.features.partition_graph_features import compute_partition_graph_features
from src.models.tabular_models import RidgeModel, XGBoostModel, get_feature_columns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "all_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_SET = "partition_x_graph"   # the paper's default 28-feature map
KNN_K = 10


def _knn_graph(X_std: np.ndarray, k: int = KNN_K) -> sparse.csr_matrix:
    """Symmetric binary kNN adjacency, matching src/data/preprocessing.py."""
    n = X_std.shape[0]
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(X_std)
    _, indices = nn.kneighbors(X_std)
    rows = np.repeat(np.arange(n), k_eff)
    cols = indices.ravel()
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    return ((A + A.T) > 0).astype(np.float32)


def _partition_row(X_std: np.ndarray, labels: np.ndarray, A: sparse.csr_matrix) -> dict:
    """The 28-feature row for one (dataset, partition) pair, matching all_features.csv."""
    row = {}
    row.update(compute_partition_features(labels))
    row.update(compute_partition_x_features(X_std, labels))
    row.update({f"graph_{key}": val for key, val in compute_partition_graph_features(A, labels).items()})
    return row


class MetaIVM:
    """Label-free clustering model selector (scikit-learn-style)."""

    def __init__(self, model: str = "xgboost"):
        if model not in ("xgboost", "ridge"):
            raise ValueError("model must be 'xgboost' or 'ridge'")
        self.model = model
        self._estimator = None          # fitted RidgeModel / XGBoostModel
        self.feature_cols_: Optional[List[str]] = None

    # ------------------------------------------------------------------ fit
    def fit(self, feature_df: Optional[pd.DataFrame] = None,
            val_frac: float = 0.15, seed: int = 0) -> "MetaIVM":
        """Train the selector on a labeled benchmark of (dataset, partition) feature rows.

        Args:
            feature_df: rows with the 28 `partition_x_graph` feature columns, a `dataset_id`
                column, and an `ami` target. If None, the bundled 223-dataset benchmark
                (`data/features/all_features.csv`) is used.
            val_frac: dataset-level fraction held out for early stopping / alpha selection.
            seed: split seed.
        """
        if feature_df is None:
            feature_df = pd.read_csv(FEATURES_CSV)

        est = self._make_estimator()

        # dataset-level split so early stopping / alpha selection never leaks partitions
        ds_ids = feature_df["dataset_id"].unique().tolist()
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ds_ids))
        n_val = max(1, int(len(ds_ids) * val_frac))
        val_ids = {ds_ids[i] for i in perm[:n_val]}
        is_val = feature_df["dataset_id"].isin(val_ids)
        train_df, val_df = feature_df[~is_val], feature_df[is_val]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(train_df, val_df)
        self._estimator = est
        self.feature_cols_ = get_feature_columns(feature_df, FEATURE_SET)
        return self

    def _make_estimator(self):
        if self.model == "xgboost":
            try:
                import xgboost  # noqa: F401
                return XGBoostModel(feature_set=FEATURE_SET)
            except Exception:
                logger.warning("xgboost unavailable; falling back to Ridge.")
                self.model = "ridge"
        return RidgeModel(feature_set=FEATURE_SET)

    # ---------------------------------------------------------- from_pretrained
    @classmethod
    def from_pretrained(cls, model: str = "xgboost", cache: bool = True) -> "MetaIVM":
        """Return a selector trained on the bundled 223-dataset benchmark.

        Fits once and caches the fitted selector under `models/`; later calls load it.
        Falls back to Ridge if XGBoost is unavailable.
        """
        import joblib
        path = MODELS_DIR / f"metaivm_{model}.joblib"
        if cache and path.exists():
            try:
                obj = joblib.load(path)
                if isinstance(obj, cls) and obj._estimator is not None:
                    return obj
            except Exception:
                logger.warning("Could not load cached model at %s; refitting.", path)
        obj = cls(model=model).fit()
        if cache:
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(obj, MODELS_DIR / f"metaivm_{obj.model}.joblib")
            except Exception:
                logger.warning("Could not cache fitted model under %s.", MODELS_DIR)
        return obj

    # --------------------------------------------------------------- score / select
    def _features(self, X: np.ndarray, partitions: Sequence[np.ndarray],
                  knn_k: int = KNN_K, standardize: bool = True) -> pd.DataFrame:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
        for p in partitions:
            if len(p) != X.shape[0]:
                raise ValueError("each partition must have one label per row of X.")
        X_std = StandardScaler().fit_transform(X) if standardize else X
        A = _knn_graph(X_std, k=knn_k)
        rows = [_partition_row(X_std, np.asarray(labels), A) for labels in partitions]
        df = pd.DataFrame(rows)
        # The underlying tabular predict() reads (and discards) a target column;
        # supply a placeholder so it never needs ground-truth labels at deployment.
        df["ami"] = 0.0
        return df

    def score(self, X: np.ndarray, partitions: Sequence[np.ndarray],
              knn_k: int = KNN_K, standardize: bool = True) -> np.ndarray:
        """Predicted external quality (higher = better) for each candidate partition."""
        if self._estimator is None:
            raise RuntimeError("MetaIVM is not fitted. Call fit() or from_pretrained() first.")
        df = self._features(X, partitions, knn_k=knn_k, standardize=standardize)
        return np.asarray(self._estimator.predict(df), dtype=np.float64)

    def select(self, X: np.ndarray, partitions: Sequence[np.ndarray],
               knn_k: int = KNN_K, standardize: bool = True) -> np.ndarray:
        """Return the candidate partition with the highest predicted quality."""
        if len(partitions) == 0:
            raise ValueError("partitions is empty.")
        scores = self.score(X, partitions, knn_k=knn_k, standardize=standardize)
        return np.asarray(partitions[int(np.argmax(scores))])

    def best_index(self, X: np.ndarray, partitions: Sequence[np.ndarray],
                   knn_k: int = KNN_K, standardize: bool = True) -> int:
        """Index of the best candidate partition (argmax of predicted quality)."""
        return int(np.argmax(self.score(X, partitions, knn_k=knn_k, standardize=standardize)))
