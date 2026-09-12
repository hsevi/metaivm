"""
Same-class kNN rate diagnostic across the full 223-dataset benchmark.

For each dataset, computes the fraction of each point's 10 nearest neighbors
(in standardized raw feature space) that share its ground-truth class label.
This is the diagnostic that predicted the spirals failure mode in Stage 0:
when this rate is at or near random, no method that builds on local Euclidean
distance can recover the classes.

Random baseline depends on the number of classes K and the class proportions:
  - K=2 balanced: random ≈ 0.50
  - K=3 balanced: random ≈ 0.33
  - K=10 balanced: random ≈ 0.10
We report both the raw rate and a normalized "rate above random" so different
class counts are comparable.

Output:
  results/aggregated/nn_rate_diagnostic.csv
  stdout                                   histogram + summary

Run:
  cd /Users/harrysevi/altrove_projects/Neural_IVM
  python -u -m scripts.phi_nn_rate_diagnostic
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REGISTRY = DATA_DIR / "dataset_registry.csv"
OUT_DIR = PROJECT_ROOT / "results" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KNN_K = 10
N_CAP = 5000   # subsample only the very largest datasets


def random_baseline_nn(y: np.ndarray) -> float:
    """Expected same-class NN rate under random neighbor assignment."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # E[same class | random neighbor] = sum_c p_c * (count_c - 1) / (N - 1)
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def nn_rate(X: np.ndarray, y: np.ndarray, k: int = KNN_K) -> float:
    n = X.shape[0]
    if n <= k:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return float((y[idx[:, 1:]] == y[:, None]).mean())


def main() -> None:
    reg = pd.read_csv(REGISTRY)
    rng = np.random.default_rng(0)

    rows: List[Dict] = []
    for i, r in reg.iterrows():
        ds = r["dataset_id"]
        src = r["source"]
        ds_dir = PROCESSED_DIR / ds
        if not (ds_dir / "X.npy").exists():
            continue
        X = np.load(ds_dir / "X.npy").astype(np.float64)
        y = np.load(ds_dir / "y_true.npy").astype(np.int64)

        # Subsample only if very large.
        if X.shape[0] > N_CAP:
            idx = []
            for c in np.unique(y):
                ci = np.where(y == c)[0]
                take = max(1, int(N_CAP * len(ci) / len(y)))
                picks = rng.choice(ci, size=min(take, len(ci)), replace=False)
                idx.extend(picks.tolist())
            idx = np.array(sorted(idx))
            X, y = X[idx], y[idx]

        try:
            Xs = StandardScaler().fit_transform(X)
            rate = nn_rate(Xs, y, KNN_K)
            rnd = random_baseline_nn(y)
            rows.append(dict(
                dataset_id=ds, source=src,
                n=int(X.shape[0]), d=int(X.shape[1]),
                n_classes=int(len(np.unique(y))),
                nn_rate=rate,
                random_baseline=rnd,
                excess_over_random=rate - rnd,
            ))
        except Exception as e:
            logger.warning(f"[{ds}] failed: {e}")
        if (i + 1) % 30 == 0:
            logger.info(f"processed {i+1}/{len(reg)}")

    df = pd.DataFrame(rows).sort_values("nn_rate")
    out_path = OUT_DIR / "nn_rate_diagnostic.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}  ({len(df)} datasets)")

    # ============================================================
    # Print histogram + summary
    # ============================================================
    print()
    print("=" * 80)
    print("SAME-CLASS kNN RATE DIAGNOSTIC — 223-dataset benchmark")
    print("=" * 80)
    print()
    print(f"Total datasets processed: {len(df)}")
    print()

    # Buckets relative to random baseline
    df["above_random_bucket"] = pd.cut(
        df["excess_over_random"],
        bins=[-1, -0.01, 0.05, 0.15, 0.30, 0.50, 1.0],
        labels=["≤ random", "barely +", "+0.05..0.15", "+0.15..0.30", "+0.30..0.50", "+0.50..1.0"],
    )
    print("Same-class NN rate vs random baseline (excess above random):")
    print()
    overall_counts = df["above_random_bucket"].value_counts().sort_index()
    for bucket, count in overall_counts.items():
        bar = "█" * int(40 * count / max(1, overall_counts.max()))
        print(f"  {bucket:<15s}  {count:>3d}  {bar}")
    print()

    # Per-source breakdown
    print("Breakdown by source:")
    print()
    for src in ["synthetic", "openml", "text", "image"]:
        sub = df[df["source"] == src]
        if len(sub) == 0:
            continue
        below = int((sub["excess_over_random"] <= 0.01).sum())
        barely = int(((sub["excess_over_random"] > 0.01) & (sub["excess_over_random"] <= 0.15)).sum())
        good = int((sub["excess_over_random"] > 0.15).sum())
        print(f"  {src:<10s}  n={len(sub):>3d}   below/at random: {below:>3d}   barely above (+0.01..+0.15): {barely:>3d}   well above (>+0.15): {good:>3d}")
    print()

    # The "spirals problem" datasets — bottom of the distribution
    print("10 hardest datasets (lowest excess over random):")
    print()
    fmt = "{:<40} {:<10} {:>6} {:>8} {:>10} {:>10} {:>10}"
    print(fmt.format("dataset", "source", "n", "classes", "NN_rate", "random", "excess"))
    print("-" * 100)
    for _, r in df.head(10).iterrows():
        print(fmt.format(
            r["dataset_id"][:38],
            r["source"][:9],
            int(r["n"]),
            int(r["n_classes"]),
            f"{r['nn_rate']:.3f}",
            f"{r['random_baseline']:.3f}",
            f"{r['excess_over_random']:+.3f}",
        ))
    print()

    # Verdict
    fraction_below = float((df["excess_over_random"] <= 0.01).mean())
    fraction_barely = float(((df["excess_over_random"] > 0.01) & (df["excess_over_random"] <= 0.15)).mean())
    print(f"Fraction of datasets with NN rate ≤ random:               {fraction_below:.1%}")
    print(f"Fraction with NN rate barely above random (≤ +0.15):     {fraction_barely:.1%}")
    print()
    if fraction_below + fraction_barely > 0.20:
        verdict = ("WIDESPREAD — more than 20% of datasets have weak local class signal. "
                   "The 'spirals problem' is a real concern; investigate further.")
    elif fraction_below + fraction_barely > 0.05:
        verdict = ("MILD — 5-20% of datasets have weak local signal. The spirals problem is "
                   "not negligible but is bounded. Worth a paragraph in the paper limits section.")
    else:
        verdict = ("RARE — under 5% of datasets have weak local signal. The spirals case is "
                   "a synthetic stress test, not a representative real-world failure mode. "
                   "Park as a clean 'open problem' in the paper.")
    print(f"VERDICT: {verdict}")
    print("=" * 80)


if __name__ == "__main__":
    main()
