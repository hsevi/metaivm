"""
Evaluation metrics for Neural IVM.

All metrics are computed per dataset (given true AMIs and predicted scores
for all clustering runs on that dataset), then aggregated across datasets.
"""

import numpy as np
from scipy import stats


def selection_regret(true_amis, predicted_scores):
    """
    Regret = best_true_ami - true_ami_of_argmax(predicted_scores).

    Args:
        true_amis: Array of true AMI values for all runs on one dataset.
        predicted_scores: Array of predicted scores (higher = better).

    Returns:
        Float regret value (>= 0). Lower is better.
    """
    true_amis = np.asarray(true_amis)
    predicted_scores = np.asarray(predicted_scores)
    best_true = true_amis.max()
    selected_idx = np.argmax(predicted_scores)
    return float(best_true - true_amis[selected_idx])


def eps_success(true_amis, predicted_scores, eps=0.01):
    """
    1 if regret <= eps, else 0.

    Args:
        true_amis: Array of true AMI values.
        predicted_scores: Array of predicted scores.
        eps: Tolerance threshold.

    Returns:
        1.0 or 0.0.
    """
    regret = selection_regret(true_amis, predicted_scores)
    return 1.0 if regret <= eps else 0.0


def ndcg_at_k(true_amis, predicted_scores, k=5):
    """
    NDCG@k treating true AMIs as relevance scores.

    Args:
        true_amis: Array of true AMI values (used as relevance).
        predicted_scores: Array of predicted scores.
        k: Number of top positions to evaluate.

    Returns:
        Float NDCG@k value in [0, 1].
    """
    true_amis = np.asarray(true_amis, dtype=np.float64)
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64)

    k = min(k, len(true_amis))

    # Ranking by predicted scores (descending)
    pred_order = np.argsort(-predicted_scores)[:k]
    # Ideal ranking by true AMIs (descending)
    ideal_order = np.argsort(-true_amis)[:k]

    # DCG
    discounts = np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ..., log2(k+1)
    dcg = np.sum(true_amis[pred_order] / discounts)
    idcg = np.sum(true_amis[ideal_order] / discounts)

    if idcg < 1e-12:
        return 1.0  # All items have zero relevance

    return float(dcg / idcg)


def spearman_corr(true_amis, predicted_scores):
    """
    Spearman rank correlation between true AMIs and predicted scores.

    Returns:
        Float correlation in [-1, 1].
    """
    true_amis = np.asarray(true_amis)
    predicted_scores = np.asarray(predicted_scores)

    if len(true_amis) < 3:
        return np.nan
    if np.std(true_amis) < 1e-12 or np.std(predicted_scores) < 1e-12:
        return np.nan

    corr, _ = stats.spearmanr(true_amis, predicted_scores)
    return float(corr)


def kendall_tau(true_amis, predicted_scores):
    """
    Kendall tau rank correlation between true AMIs and predicted scores.

    Returns:
        Float correlation in [-1, 1].
    """
    true_amis = np.asarray(true_amis)
    predicted_scores = np.asarray(predicted_scores)

    if len(true_amis) < 3:
        return np.nan
    if np.std(true_amis) < 1e-12 or np.std(predicted_scores) < 1e-12:
        return np.nan

    corr, _ = stats.kendalltau(true_amis, predicted_scores)
    return float(corr)


def compute_all_metrics(true_amis, predicted_scores, eps=0.01, k=5):
    """
    Compute all metrics for a single dataset.

    Returns:
        Dict of metric name -> value.
    """
    return {
        "regret": selection_regret(true_amis, predicted_scores),
        "eps_success": eps_success(true_amis, predicted_scores, eps=eps),
        "ndcg_at_k": ndcg_at_k(true_amis, predicted_scores, k=k),
        "spearman": spearman_corr(true_amis, predicted_scores),
        "kendall": kendall_tau(true_amis, predicted_scores),
    }


def aggregate_metrics(per_dataset_metrics):
    """
    Aggregate per-dataset metrics to mean +/- std.

    Args:
        per_dataset_metrics: List of dicts from compute_all_metrics.

    Returns:
        Dict of metric_name -> {"mean": float, "std": float}.
    """
    import pandas as pd

    df = pd.DataFrame(per_dataset_metrics)
    result = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            result[col] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return result
