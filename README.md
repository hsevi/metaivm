# MetaIVM: Meta-Learned Surrogates for Clustering Model Selection

Reference implementation and benchmark for the paper *Meta-Learned Surrogates for
Clustering Model Selection*, published in **Transactions on Machine Learning Research
(TMLR), 2026**. [[OpenReview]](https://openreview.net/forum?id=5r3WODcU47)

MetaIVM selects among candidate clusterings **without ground-truth labels** and is
usable as a drop-in scikit-learn-style tool (see Quickstart below).

## Overview

MetaIVM is a meta-learned surrogate for external-agreement-based clustering model selection.
Trained offline on labeled benchmarks and deployed without labels, it predicts the quality of
individual (dataset, partition) pairs from observable features of partition structure, dataset
statistics, and graph topology.

## Quickstart: using MetaIVM on your own data

MetaIVM selects among candidate clusterings **without ground-truth labels**, with a
scikit-learn-style interface:

```python
from src.metaivm import MetaIVM
from sklearn.cluster import KMeans, DBSCAN

# 1. Load a selector trained on the bundled 223-dataset benchmark
#    (fits in a few seconds on first call, then caches locally).
selector = MetaIVM.from_pretrained()          # model="xgboost" (default) or "ridge"

# 2. You have data X and several candidate clusterings.
candidates = [KMeans(k, n_init=10).fit_predict(X) for k in (2, 5, 10)]
candidates.append(DBSCAN(eps=0.5).fit_predict(X))

# 3. Let MetaIVM pick the best one — no labels needed.
best_labels = selector.select(X, candidates)  # the chosen label array
scores      = selector.score(X, candidates)   # predicted quality per candidate (higher = better)
```

`score` returns MetaIVM's estimate of external agreement (AMI) with the unavailable
ground truth; `select` returns the highest-scoring partition. Features (the paper's
default 28-dimensional `partition_x_graph` map) are computed in memory from `(X, labels)`;
`X` is standardized and a symmetric kNN graph is built internally, matching the training
pipeline. If XGBoost is not installed, MetaIVM falls back to Ridge automatically.

To train on your **own** labeled benchmark instead of the bundled one, pass a feature
table (rows = (dataset, partition) pairs with the 28 feature columns, a `dataset_id`
column, and an `ami` target) to `MetaIVM(model="xgboost").fit(feature_df)`.

## Reviewer quick-start

The entry point for **every table in the paper** is the shipped feature table:

```
data/features/all_features.csv    # 16,889 (dataset, partition) pairs x 56 features
data/features/runs_master.csv     # matched hyperparameter descriptions
data/features/pymfe_features.csv  # PyMFE dataset-level meta-features (used by PoAC-style)
```

You do **not** need to run the ~25 GB from-scratch pipeline to reproduce the paper's tables.
Reproduction takes minutes-to-a-few-hours from the shipped features.

## Repository structure

```
src/                        # Core library code
  data/                     # Dataset collection, preprocessing, clustering runner
  features/                 # Feature extraction (partition, dataset, graph, IVM)
  models/                   # Tabular models (Ridge, MLP, XGBoost) + shared helpers
  evaluation/               # Splits, metrics, regret computation
  experiments/              # Canonical experiment scripts (exp1-exp7, exp_gnn)
scripts/                    # Standalone scripts for the revision-added experiments
configs/                    # Clustering algorithm hyperparameter grids
data/features/              # Precomputed feature tables (entry point for reviewers)
results/aggregated/         # Aggregated experiment results (CSV outputs cited in the paper)
paper/                      # LaTeX source and compiled PDF of the paper
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10-3.12. The revision added new dependencies for the density-based
CVI baselines (`s_dbw`), PoAC dataset meta-features (`pymfe`), and the graph community
detection extension (`networkx`, `python-igraph`, `python-louvain`, `leidenalg`).

## Reproducing the main results

### Original experiments (unchanged since first submission)

```bash
python -m src.experiments.exp1_main_table       # Table 2 (main comparison)
python -m src.experiments.exp3_leave_one_algo   # Table 19 (LOO algorithm family)
python -m src.experiments.exp6_cross_domain     # Table 18 (cross-domain transfer)
python -m src.experiments.exp7_zero_engineering # Table 22 (zero-engineering / GNN)
```

### Revision-added experiments and baselines

| Script | Reproduces |
| --- | --- |
| `scripts/ablation_table6.py` | Table 6 (`tab:ablation`): feature-set ablation, all 11 rows (canonical generator; supersedes `src/experiments/exp2_feature_ablation.py`) |
| `scripts/compute_density_cvis.py` | Per-partition DBCV / S_Dbw values + degenerate-partition coverage (Table 17) |
| `scripts/density_cvi_regret.py` | DBCV (0.219), S_Dbw (0.334) selection-regret rows of Table 2 (`tab:main`), on the same test folds as every other method |
| `scripts/autoclust_poac_baselines.py` | AutoClust-style (0.195), PoAC-style (0.139) rows of Table 4 (`tab:baselines`); Appendix I |
| `scripts/mlp_5seed_postfix.py` | MLP row of Table 2 under the leakage-fixed training procedure (0.082 +/- 0.115) |
| `scripts/permutation_importance.py` | Table 12 (`tab:app_perm_importance`), Appendix B: protocol-specific bootstrap CIs |
| `scripts/logfo_experiment.py` | Leave-one-generator-family-out (Section 6.5) |
| `scripts/controlled_shifts.py` | Table 7 (`tab:shifts`, Section 6.7): regret binned by k, d, n, imbalance, source |
| `scripts/exp_multi_target.py` | Table 23 (`tab:multitarget`, Appendix): AMI / ARI / NMI / V-measure |
| `scripts/exp_graph_community.py` | Table 9 (`tab:graph_community`, Section 6.8): SBM graph community detection |
| `scripts/exp_attributed_graphs.py` | Table 21 (`tab:attr_graph`, Appendix): preliminary attributed-graph LOO (requires `torch_geometric` dataset downloads) |
| `scripts/generate_tables.py` | Assembles the raw CSVs into LaTeX tables |

Fixed-IVM baselines that need no training (CH, Silhouette, DB) are evaluated
inside `exp1_main_table.py`. All learned methods and all fixed IVMs are scored on
the **same** 46-test-datasets-per-seed folds (appearance order of
`all_features.csv`); `make_splits` permutes indices, so every script uses that
ordering to guarantee identical folds. For bit-reproducible MLP/XGBoost numbers,
run single-threaded (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`);
multi-threaded BLAS can perturb the last digit over many non-convex iterations.

### Test / evaluation protocol

All experiments use the shared 5-seed 60/20/20 dataset-level split (`src/evaluation/splits.py`):
133 train + 44 val + 46 test datasets per seed, 230 seed-dataset test evaluations
across the 5 seeds, 158 unique held-out datasets. See Section 3.3 and Appendix J.

The MLP leakage fix is in `src/models/tabular_models.py::MLPModel`: sklearn's internal
`early_stopping` is disabled and the MLP is fit on the dataset-level train fold only,
with outer arch/lr model selection on the dataset-level held-out val fold. See Section 4
of the paper and the response to Reviewer g9yy for details.

## Reproducing the full benchmark from scratch

Building the benchmark from raw data requires several hours and ~25 GB of disk space:

```bash
# 1. Dataset collection (downloads OpenML datasets, generates synthetic)
python -m src.data.dataset_collection

# 2. Preprocessing (kNN graphs, standardization)
python -m src.data.preprocessing

# 3. Run clustering algorithms across the hyperparameter grid (91 configs per dataset)
python -m src.data.clustering_runner

# 4. Feature extraction (28 default + 28 extended features per partition)
python -m src.features.assemble_features
```

Reviewers should not need this; the shipped feature tables suffice for every reported table.

## Citation

```bibtex
@article{sevi2026metaivm,
  title   = {Meta-Learned Surrogates for Clustering Model Selection},
  author  = {Sevi, Harry},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=5r3WODcU47}
}
```

## License

Released under the MIT License (see `LICENSE`).
