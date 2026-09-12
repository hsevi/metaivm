#!/usr/bin/env bash
# Run Phase 1-2 pipeline: dataset collection -> preprocessing -> clustering -> features
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

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
echo "=== Done ==="
echo "Feature table: data/features/all_features.csv"
echo "Master table:  data/features/runs_master.csv"
echo "Registry:      data/dataset_registry.csv"
