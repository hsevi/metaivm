"""
Tabular meta-evaluator models for Neural IVM.

Train regression models: features -> predicted AMI.
Each model implements fit(train_df, val_df) and predict(test_df).
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature set definitions — maps feature set name to column prefixes/names
FEATURE_SETS = {
    "algo_hyperparams_only": {
        "prefixes": ["algo_", "hp_"],
        "columns": [],
    },
    "run_feats_only": {
        "prefixes": ["algo_", "hp_"],
        "columns": [],
    },
    "dataset_desc_only": {
        "prefixes": ["ds_"],
        "columns": [],
    },
    "partition_only": {
        "prefixes": [],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
        ],
    },
    "partition_algo": {
        "prefixes": ["algo_", "hp_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
        ],
    },
    "partition_dataset": {
        "prefixes": ["ds_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
        ],
    },
    "partition_x": {
        "prefixes": [],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
            "within_cluster_var_mean", "within_cluster_var_std",
            "between_cluster_var", "dispersion_ratio",
            "cluster_density_mean", "cluster_density_std", "cluster_density_max",
            "centroid_sep_min", "centroid_sep_mean", "centroid_sep_std",
        ],
    },
    "partition_x_graph": {
        "prefixes": ["graph_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
            "within_cluster_var_mean", "within_cluster_var_std",
            "between_cluster_var", "dispersion_ratio",
            "cluster_density_mean", "cluster_density_std", "cluster_density_max",
            "centroid_sep_min", "centroid_sep_mean", "centroid_sep_std",
        ],
    },
    "ivm_only": {
        "prefixes": ["ivm_"],
        "columns": [],
    },
    "ivm_plus_partition": {
        "prefixes": ["ivm_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
        ],
    },
    "all_features": {
        "prefixes": ["ds_", "graph_", "ivm_", "algo_", "hp_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
            "within_cluster_var_mean", "within_cluster_var_std",
            "between_cluster_var", "dispersion_ratio",
            "cluster_density_mean", "cluster_density_std", "cluster_density_max",
            "centroid_sep_min", "centroid_sep_mean", "centroid_sep_std",
        ],
    },
    # ---- Feature-ablation sets (Table 6 / tab:ablation) ----
    # Resolved sizes on all_features.csv: graph_=6, ds_=14, ivm_=3, algo_=6, hp_=5,
    # partition-stat cols=12, partition_x cols (partition+interaction)=22.
    "graph_only": {  # 6 features
        "prefixes": ["graph_"],
        "columns": [],
    },
    "partition_x_dataset": {  # 22 + 14 = 36 features
        "prefixes": ["ds_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
            "within_cluster_var_mean", "within_cluster_var_std",
            "between_cluster_var", "dispersion_ratio",
            "cluster_density_mean", "cluster_density_std", "cluster_density_max",
            "centroid_sep_min", "centroid_sep_mean", "centroid_sep_std",
        ],
    },
    "no_ivm_no_algo": {  # 22 + ds_(14) + graph_(6) = 42 features
        "prefixes": ["ds_", "graph_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
            "within_cluster_var_mean", "within_cluster_var_std",
            "between_cluster_var", "dispersion_ratio",
            "cluster_density_mean", "cluster_density_std", "cluster_density_max",
            "centroid_sep_min", "centroid_sep_mean", "centroid_sep_std",
        ],
    },
    "no_ivm": {  # all_features minus ivm_ : 22 + ds_(14) + graph_(6) + algo_(6) + hp_(5) = 53
        "prefixes": ["ds_", "graph_", "algo_", "hp_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
            "within_cluster_var_mean", "within_cluster_var_std",
            "between_cluster_var", "dispersion_ratio",
            "cluster_density_mean", "cluster_density_std", "cluster_density_max",
            "centroid_sep_min", "centroid_sep_mean", "centroid_sep_std",
        ],
    },
    "no_algo": {  # all_features minus algo_/hp_ : 22 + ds_(14) + graph_(6) + ivm_(3) = 45
        "prefixes": ["ds_", "graph_", "ivm_"],
        "columns": [
            "n_clusters", "noise_fraction", "cluster_size_min", "cluster_size_max",
            "cluster_size_mean", "cluster_size_std", "cluster_size_median",
            "size_ratio", "entropy", "gini", "imbalance", "singleton_fraction",
            "within_cluster_var_mean", "within_cluster_var_std",
            "between_cluster_var", "dispersion_ratio",
            "cluster_density_mean", "cluster_density_std", "cluster_density_max",
            "centroid_sep_min", "centroid_sep_mean", "centroid_sep_std",
        ],
    },
    # Compact sets: the 15 / 8 features selected by gain importance stable across
    # training seeds (selection protocol and the exact lists are in Appendix L).
    "compact_15": {
        "prefixes": [],
        "columns": [
            "ivm_calinski_harabasz", "ds_mean_of_kurtosis", "graph_conductance_min",
            "entropy", "ds_mean_of_skewness", "ds_n_samples", "ds_pairwise_dist_std",
            "cluster_size_mean", "graph_conductance_mean", "ds_pairwise_dist_mean",
            "between_cluster_var", "ds_pairwise_dist_median", "ivm_silhouette",
            "ds_d_over_n", "noise_fraction",
        ],
    },
    "compact_8": {
        "prefixes": [],
        "columns": [
            "ivm_calinski_harabasz", "ds_mean_of_kurtosis", "graph_conductance_min",
            "entropy", "ds_mean_of_skewness", "ds_n_samples", "ds_pairwise_dist_std",
            "cluster_size_mean",
        ],
    },
}

# Columns that are not features
NON_FEATURE_COLS = {"dataset_id", "run_id", "algo", "ami", "hyperparams_json"}


def get_feature_columns(df, feature_set_name):
    """Get the list of feature columns for a given feature set."""
    spec = FEATURE_SETS[feature_set_name]
    cols = []

    # Add explicit columns that exist in df
    for c in spec["columns"]:
        if c in df.columns:
            cols.append(c)

    # Add prefix-matched columns
    for prefix in spec["prefixes"]:
        for c in df.columns:
            if c.startswith(prefix) and c not in NON_FEATURE_COLS:
                cols.append(c)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            result.append(c)

    return result


def _prepare_xy(df, feature_cols):
    """Extract feature matrix X and target y from a DataFrame."""
    X = df[feature_cols].values.astype(np.float64)
    y = df["ami"].values.astype(np.float64)

    # Replace NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


class RidgeModel:
    """Ridge regression meta-evaluator."""

    def __init__(self, feature_set="partition_x_graph", alpha=1.0):
        self.feature_set = feature_set
        self.alpha = alpha
        self.model = None
        self.scaler = None
        self.feature_cols = None

    def fit(self, train_df, val_df):
        self.feature_cols = get_feature_columns(train_df, self.feature_set)
        if not self.feature_cols:
            logger.warning(f"RidgeModel: no features for set '{self.feature_set}'")
            self.model = None
            return

        X_train, y_train = _prepare_xy(train_df, self.feature_cols)

        # Tune alpha on val if multiple options
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha, best_score = self.alpha, -np.inf

        X_val, y_val = _prepare_xy(val_df, self.feature_cols)

        for a in alphas:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_v_s = scaler.transform(X_val)

            model = Ridge(alpha=a)
            model.fit(X_tr_s, y_train)
            score = model.score(X_v_s, y_val)
            if score > best_score:
                best_score = score
                best_alpha = a

        # Refit on train+val with best alpha
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        self.scaler = StandardScaler()
        X_all_s = self.scaler.fit_transform(X_all)

        self.model = Ridge(alpha=best_alpha)
        self.model.fit(X_all_s, y_all)

    def predict(self, test_df):
        if self.model is None or not self.feature_cols:
            return np.zeros(len(test_df))

        X, _ = _prepare_xy(test_df, self.feature_cols)
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)


class MLPModel:
    """MLP regression meta-evaluator."""

    def __init__(self, feature_set="partition_x_graph",
                 hidden_layers=(256, 128), lr=0.001, max_iter=200, patience=20):
        self.feature_set = feature_set
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.max_iter = max_iter
        self.patience = patience
        self.model = None
        self.scaler = None
        self.feature_cols = None

    def fit(self, train_df, val_df):
        self.feature_cols = get_feature_columns(train_df, self.feature_set)
        if not self.feature_cols:
            self.model = None
            return

        X_train, y_train = _prepare_xy(train_df, self.feature_cols)
        X_val, y_val = _prepare_xy(val_df, self.feature_cols)

        # Scale
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        # Grid search over architectures and learning rates.
        # NOTE: previously this pooled train+val and let sklearn's early_stopping
        # randomly split at the row (partition) level, which leaked across the
        # dataset-level train/val boundary. We now train on X_train_s only and
        # use the dataset-level val fold (X_val_s, y_val) for BOTH inner early
        # stopping and outer arch/lr model selection, preserving dataset-level
        # separation throughout.
        arch_grid = [
            (256, 128),
            (512, 256),
            (256, 128, 64),
            (512, 256, 128),
        ]
        lr_grid = [1e-3, 3e-3]

        best_model, best_val_loss = None, np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for arch in arch_grid:
                for lr in lr_grid:
                    # early_stopping=False so sklearn never splits X_train_s at
                    # the row (partition) level. Model selection uses the outer
                    # dataset-level (X_val_s, y_val) fold.
                    model = MLPRegressor(
                        hidden_layer_sizes=arch,
                        learning_rate_init=lr,
                        max_iter=self.max_iter,
                        early_stopping=False,
                        random_state=0,
                    )
                    model.fit(X_train_s, y_train)
                    val_pred = model.predict(X_val_s)
                    val_loss = np.mean((val_pred - y_val) ** 2)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model

        self.model = best_model

    def predict(self, test_df):
        if self.model is None or not self.feature_cols:
            return np.zeros(len(test_df))

        X, _ = _prepare_xy(test_df, self.feature_cols)
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)


class XGBoostModel:
    """XGBoost regression meta-evaluator."""

    def __init__(self, feature_set="partition_x_graph",
                 max_depth=5, n_estimators=300, learning_rate=0.05):
        self.feature_set = feature_set
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols = None

    def fit(self, train_df, val_df):
        import xgboost as xgb

        self.feature_cols = get_feature_columns(train_df, self.feature_set)
        if not self.feature_cols:
            self.model = None
            return

        X_train, y_train = _prepare_xy(train_df, self.feature_cols)
        X_val, y_val = _prepare_xy(val_df, self.feature_cols)

        # Grid search over key hyperparams
        best_model, best_val_score = None, np.inf
        for md in [3, 5, 7]:
            for lr in [0.01, 0.05, 0.1]:
                model = xgb.XGBRegressor(
                    max_depth=md,
                    n_estimators=500,
                    learning_rate=lr,
                    early_stopping_rounds=20,
                    eval_metric="rmse",
                    random_state=0,
                    verbosity=0,
                    n_jobs=-1,
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                val_pred = model.predict(X_val)
                val_mse = np.mean((val_pred - y_val) ** 2)
                if val_mse < best_val_score:
                    best_val_score = val_mse
                    best_model = model

        # Refit best config on train+val
        if best_model is not None:
            X_all = np.vstack([X_train, X_val])
            y_all = np.concatenate([y_train, y_val])

            self.model = xgb.XGBRegressor(
                max_depth=best_model.get_params()["max_depth"],
                n_estimators=best_model.best_iteration + 1 if hasattr(best_model, "best_iteration") else 300,
                learning_rate=best_model.get_params()["learning_rate"],
                random_state=0,
                verbosity=0,
                n_jobs=-1,
            )
            self.model.fit(X_all, y_all)

    def predict(self, test_df):
        if self.model is None or not self.feature_cols:
            return np.zeros(len(test_df))

        X, _ = _prepare_xy(test_df, self.feature_cols)
        return self.model.predict(X)
