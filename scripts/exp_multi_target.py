"""
Experiment: Multi-target robustness.
Train XGBoost/Ridge/MLP predicting each of AMI, ARI, NMI, V-measure.
Also evaluate classical IVMs against each target.
Shows the method is not specific to AMI.
"""
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
# MLP removed — results from separate grid search experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import make_splits
from src.models.tabular_models import get_feature_columns, _prepare_xy


def evaluate_ivm_baseline(test_df, ivm_col, target_col):
    """Evaluate a classical IVM as a selector for a given target."""
    per_ds = []
    for ds_id, grp in test_df.groupby("dataset_id"):
        if len(grp) < 2:
            continue
        true_vals = grp[target_col].values
        ivm_vals = grp[ivm_col].values

        # Handle NaN
        valid = ~np.isnan(ivm_vals) & ~np.isnan(true_vals)
        if valid.sum() < 2:
            continue

        true_v = true_vals[valid]
        ivm_v = ivm_vals[valid]

        # Note: the DB column here is `ivm_davies_bouldin_neg`, which is ALREADY
        # negated (higher = better) upstream in ivm_features.py, so no further
        # sign flip is applied. (Silhouette / CH are natively higher = better.)

        best_true = true_v.max()
        top1_idx = np.argmax(ivm_v)
        regret = best_true - true_v[top1_idx]

        rho, _ = spearmanr(true_v, ivm_v)
        if np.isnan(rho):
            rho = 0.0

        per_ds.append({"regret": regret, "spearman": rho})

    if not per_ds:
        return {"regret_mean": np.nan, "spearman_mean": np.nan}
    df = pd.DataFrame(per_ds)
    return {
        "regret_mean": df["regret"].mean(),
        "regret_std": df["regret"].std(),
        "spearman_mean": df["spearman"].mean(),
        "spearman_std": df["spearman"].std(),
    }


def evaluate_model(model, scaler, feat_cols, test_df, target_col):
    """Evaluate a trained model on test datasets for a given target."""
    per_ds = []
    per_ds_regrets = []
    for ds_id, grp in test_df.groupby("dataset_id"):
        if len(grp) < 2:
            continue
        X_t, _ = _prepare_xy(grp, feat_cols)
        if scaler is not None:
            X_t = scaler.transform(X_t)
        pred = model.predict(X_t)
        true_vals = grp[target_col].values

        best_true = true_vals.max()
        top1_idx = np.argmax(pred)
        regret = best_true - true_vals[top1_idx]

        rho, _ = spearmanr(true_vals, pred)
        if np.isnan(rho):
            rho = 0.0

        per_ds.append({"regret": regret, "spearman": rho})
        per_ds_regrets.append(regret)

    if not per_ds:
        return {"regret_mean": np.nan, "spearman_mean": np.nan}, []
    df = pd.DataFrame(per_ds)
    return {
        "regret_mean": df["regret"].mean(),
        "regret_std": df["regret"].std(),
        "spearman_mean": df["spearman"].mean(),
        "spearman_std": df["spearman"].std(),
    }, per_ds_regrets


def main():
    logger.info("Loading data...")
    features_df = pd.read_csv(PROJECT_ROOT / "data" / "features" / "all_features.csv")
    master = pd.read_csv(PROJECT_ROOT / "data" / "features" / "runs_master.csv")
    logger.info(f"Loaded {len(features_df)} rows")

    # Ensure hyperparams_json is available
    if "hyperparams_json" not in features_df.columns:
        features_df = features_df.merge(
            master[["dataset_id", "run_id", "hyperparams_json"]],
            on=["dataset_id", "run_id"],
            how="left",
        )

    feat_cols = get_feature_columns(features_df, "partition_x_graph")
    dataset_ids = features_df["dataset_id"].unique().tolist()
    logger.info(f"Features: {len(feat_cols)}, Datasets: {len(dataset_ids)}")

    targets = ["ami", "ari", "nmi", "v_measure"]
    ivm_cols = ["ivm_silhouette", "ivm_calinski_harabasz", "ivm_davies_bouldin_neg"]
    ivm_names = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]

    seeds = [0, 1, 2, 3, 4]
    results = []

    for target in targets:
        logger.info(f"\n{'='*60}")
        logger.info(f"TARGET: {target}")
        logger.info(f"{'='*60}")

        all_xgb_regrets = []
        all_ridge_regrets = []
        seed_results = {
            "XGBoost": [], "Ridge": [],
            **{n: [] for n in ivm_names}
        }

        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            splits = make_splits(dataset_ids, n_splits=1, train_frac=0.6,
                                val_frac=0.2, test_frac=0.2, seed=seed)
            train_ids, val_ids, test_ids = splits[0]

            train_df = features_df[features_df["dataset_id"].isin(train_ids)]
            val_df = features_df[features_df["dataset_id"].isin(val_ids)]
            test_df = features_df[features_df["dataset_id"].isin(test_ids)]

            X_train, _ = _prepare_xy(train_df, feat_cols)
            y_train = train_df[target].values
            X_val, _ = _prepare_xy(val_df, feat_cols)
            y_val = val_df[target].values

            # --- XGBoost ---
            import xgboost as xgb
            X_trv = np.vstack([X_train, X_val])
            y_trv = np.concatenate([y_train, y_val])
            model_xgb = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=30, random_state=seed,
                verbosity=0,
            )
            model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            res, regrets = evaluate_model(model_xgb, None, feat_cols, test_df, target)
            seed_results["XGBoost"].append(res)
            all_xgb_regrets.extend(regrets)
            logger.info(f"    XGBoost done: regret={res['regret_mean']:.4f}")

            # --- Ridge ---
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_trv)
            model_ridge = Ridge(alpha=1.0)
            model_ridge.fit(X_train_s, y_trv)
            res, regrets = evaluate_model(model_ridge, scaler, feat_cols, test_df, target)
            seed_results["Ridge"].append(res)
            all_ridge_regrets.extend(regrets)
            logger.info(f"    Ridge done: regret={res['regret_mean']:.4f}")

            # --- IVM baselines ---
            for ivm_col, ivm_name in zip(ivm_cols, ivm_names):
                res = evaluate_ivm_baseline(test_df, ivm_col, target)
                seed_results[ivm_name].append(res)

        # Aggregate across seeds
        logger.info(f"\n{'Method':<25s} {'Regret':>12s} {'Spearman':>12s}")
        logger.info("-" * 55)
        for method in ["XGBoost", "Ridge"] + ivm_names:
            regrets = [r["regret_mean"] for r in seed_results[method] if not np.isnan(r["regret_mean"])]
            spears = [r["spearman_mean"] for r in seed_results[method] if not np.isnan(r["spearman_mean"])]
            r_mean = np.mean(regrets) if regrets else np.nan
            r_std = np.std(regrets) if regrets else np.nan
            s_mean = np.mean(spears) if spears else np.nan
            s_std = np.std(spears) if spears else np.nan
            logger.info(f"  {method:<23s} {r_mean:.4f}±{r_std:.4f}  {s_mean:.4f}±{s_std:.4f}")
            results.append({
                "target": target, "method": method,
                "regret_mean": r_mean, "regret_std": r_std,
                "spearman_mean": s_mean, "spearman_std": s_std,
            })

    # Save results
    results_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "results" / "aggregated" / "exp_multi_target.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"\nSaved to {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("MULTI-TARGET ROBUSTNESS SUMMARY")
    print("=" * 80)
    pivot = results_df.pivot(index="method", columns="target", values="regret_mean")
    pivot = pivot.reindex(["XGBoost", "Ridge", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])
    print(pivot.round(4).to_string())
    print()
    pivot_rho = results_df.pivot(index="method", columns="target", values="spearman_mean")
    pivot_rho = pivot_rho.reindex(["XGBoost", "Ridge", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])
    print("Spearman ρ:")
    print(pivot_rho.round(4).to_string())


if __name__ == "__main__":
    main()
