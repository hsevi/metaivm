#!/usr/bin/env bash
# End-to-end pipeline: data collection -> experiments -> tables
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Neural IVM — Full Pipeline"
echo "=========================================="

echo ""
echo "=== Phase 1.1: Dataset Collection ==="
python -m src.data.dataset_collection

echo ""
echo "=== Phase 1.2: Preprocessing ==="
python -m src.data.preprocessing

echo ""
echo "=== Phase 1.3: Clustering Runs ==="
python -m src.data.clustering_runner

echo ""
echo "=== Phase 2: Feature Assembly ==="
python -m src.features.assemble_features

echo ""
echo "=== Experiment 1: Main Comparison Table ==="
python -m src.experiments.exp1_main_table

echo ""
echo "=== Experiment 2: Feature Ablation ==="
python -m src.experiments.exp2_feature_ablation

echo ""
echo "=== Experiment 3: Leave-One-Algo-Out ==="
python -m src.experiments.exp3_leave_one_algo

echo ""
echo "=== Experiment 5: Diagnostics ==="
python -m src.experiments.exp5_analysis

echo ""
echo "=== Generate LaTeX Tables ==="
python scripts/generate_tables.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results in: results/aggregated/"
echo "=========================================="

# Note: Experiment 4 (stability) is slow and optional.
# Run separately with: python -m src.experiments.exp4_stability
