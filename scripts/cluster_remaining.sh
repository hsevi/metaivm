#!/bin/bash
# Run clustering on remaining datasets, one subprocess per dataset.
# Each subprocess is memory-isolated — if one OOMs, others continue.

cd /Users/harrysevi/altrove_projects/Neural_IVM

# Remaining datasets sorted by size (smallest first)
# Skip openml_3, openml_46, openml_50 — failed preprocessing (all NaN columns)
DATASETS=(
    openml_1040
    openml_1471
    openml_1120
    openml_41147
    openml_1596
    openml_40927
    openml_40668
    openml_1169
    openml_23517
    openml_4135
    openml_6
    openml_40685
    openml_1486
    openml_41027
    openml_1461
    openml_151
    openml_1590
)

TOTAL=${#DATASETS[@]}
for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    echo "=== [$((i+1))/$TOTAL] $ds ==="
    # Run in subprocess with memory limit (ulimit -v = 32GB virtual)
    ulimit -v 33554432 2>/dev/null
    python scripts/cluster_one.py "$ds"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "*** FAILED: $ds (exit code $STATUS) ***"
    fi
    echo ""
done

echo "=== DONE ==="
python -c "
import pandas as pd
m = pd.read_csv('data/features/runs_master.csv')
print(f'Total runs: {len(m)}, Datasets: {m[\"dataset_id\"].nunique()}')
"
