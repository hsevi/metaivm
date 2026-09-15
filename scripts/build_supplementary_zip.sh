#!/usr/bin/env bash
# Build the TMLR supplementary code zip.
#
# What ships:
#   - src/, scripts/, configs/, data/features/, data/raw/ (small metadata only),
#     results/aggregated/ (paper-cited CSVs only), models/ (none — all learned
#     models are refit from the code), notebooks/ (empty placeholder), paper/main.tex,
#     README.md, requirements.txt.
#
# What is excluded (preserved locally in the working tree, but not in the zip):
#   - _local_only/ (exploratory artifacts kept for future work)
#   - .DS_Store, .claude/, paper/author_response*.md, paper/main.pdf, paper/*.aux etc.
#   - Any existing metaivm_supplementary*.zip (the previous submission)
#   - __pycache__/ everywhere
#   - Exploratory scripts and results not cited in main.tex (phi_*, layer1_*, tier0*,
#     poc_clusterpfn, exp_gsc_metaivm, exp_local_search, scale_up_*, cluster_one,
#     cluster_large_safe, cluster_remaining, run_phase1_2, select_datasets,
#     rerun_mlp_after_fix.py which is now superseded by scripts/mlp_5seed_postfix.py).
#   - Large raw dataset caches under data/raw/, data/pyg_cache/, data/clustering_runs/,
#     data/processed/ (reviewers should regenerate from the pipeline or use the shipped
#     features tables data/features/all_features.csv and data/features/runs_master.csv).
#
# Usage:
#   bash scripts/build_supplementary_zip.sh
#
# Produces: metaivm_supplementary_v2.zip in the repo root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="metaivm_supplementary_v2.zip"
rm -f "$OUT"

# --- Files/dirs to include (whitelist approach) ---
INCLUDE=(
    "README.md"
    "LICENSE"
    "requirements.txt"
    "pyproject.toml"
    "uv.lock"
    "configs/"
    "src/"
    "scripts/"
    "data/features/all_features.csv"
    "data/features/runs_master.csv"
    "data/features/pymfe_features.csv"
    "results/aggregated/"
    "paper/main.tex"
    "paper/main.pdf"
    "paper/figure1_regret_boxplot.pdf"
    "paper/figure2_scatter.pdf"
    "paper/figure3_impossibility.pdf"
    "paper/tmlr.sty"
    "paper/tmlr.bst"
    "paper/references.bib"
    "paper/math_commands.tex"
    "notebooks/"
)

# Only include entries that actually exist so the script does not crash on
# missing optional files (e.g. paper/tmlr.sty, paper/main.bib may or may not
# be present depending on the template used).
EXISTING=()
for p in "${INCLUDE[@]}"; do
    if [[ -e "$p" ]]; then
        EXISTING+=("$p")
    fi
done

# --- Patterns to exclude everywhere ---
EXCLUDE=(
    # OS / editor
    "*.DS_Store"
    ".DS_Store"
    "*/__pycache__/*"
    "__pycache__/*"
    "*.pyc"
    "*.pyo"

    # Local MetaIVM model cache (not portable across xgboost versions;
    # from_pretrained() retrains on first call). Do not ship.
    "models/*.joblib"

    # Author-response markdown (revision documents, not code)
    "paper/author_response*"
    "paper/*.md"

    # LaTeX build artifacts
    "paper/*.aux"
    "paper/*.log"
    "paper/*.out"
    "paper/*.toc"
    "paper/*.bbl"
    "paper/*.blg"
    "paper/*.synctex.gz"

    # Local-only files
    "_local_only/*"
    ".claude/*"
    "metaivm_supplementary.zip"
    "metaivm_supplementary_v*.zip"

    # Exploratory scripts (follow-up research, not part of this paper)
    "scripts/phi_*"
    "scripts/layer1_*"
    "scripts/tier0*"
    "scripts/poc_clusterpfn.py"
    "scripts/exp_gsc_metaivm.py"
    "scripts/exp_local_search.py"
    "scripts/scale_up_*"
    "scripts/cluster_one.py"
    "scripts/cluster_large_safe.py"
    "scripts/cluster_remaining.sh"
    "scripts/run_phase1_2.sh"
    "scripts/select_datasets.py"
    "scripts/rerun_mlp_after_fix.py"

    # Exploratory result CSVs (paired with the excluded scripts)
    "results/aggregated/phi_stage*"
    "results/aggregated/phi_umap_manifold_results.csv"
    "results/aggregated/nn_rate_diagnostic.csv"
    "results/aggregated/tier0_*"
    "results/aggregated/tier0b_*"
    "results/aggregated/layer1_probe_*"
    "results/aggregated/clusterpfn_20k_results.csv"
    "results/aggregated/poc_clusterpfn.csv"
    "results/aggregated/table5_seed0_recompute.csv"
    "results/aggregated/mlp_rerun_after_fix.csv"
    "results/aggregated/xgb_5seed_authoritative.csv"

    # Precompiled LaTeX intermediate figures (already covered by main.pdf figures)
    "results/aggregated/figure*"

    # Session ephemera
    ".pytest_cache/*"
    ".mypy_cache/*"
)

EXCLUDE_ARGS=()
for p in "${EXCLUDE[@]}"; do
    EXCLUDE_ARGS+=("-x" "$p")
done

echo "Building $OUT ..."
zip -r "$OUT" "${EXISTING[@]}" "${EXCLUDE_ARGS[@]}" >/dev/null

SIZE_MB=$(du -m "$OUT" | cut -f1)
COUNT=$(unzip -l "$OUT" | tail -1 | awk '{print $2}')
echo ""
echo "Built $OUT  ($SIZE_MB MB, $COUNT files)."
echo ""
echo "Top-level entries:"
unzip -l "$OUT" | head -25
echo ""
echo "Sanity checks (should all pass):"
for bad in .DS_Store .claude "author_response" "__pycache__" "phi_stage" "tier0" "layer1_probe" "_local_only"; do
    if unzip -l "$OUT" | grep -q "$bad"; then
        echo "  [FAIL] $bad found in zip"
    else
        echo "  [ok]   no $bad"
    fi
done
