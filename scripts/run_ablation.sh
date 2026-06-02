#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Ablation studies: module / feature_grouping / aggregation / guardian.
#
# Usage:
#   bash scripts/run_ablation.sh                 # all, fast mode
#   bash scripts/run_ablation.sh guardian        # a single ablation
#
# Env vars: ABLATION_TYPE (default: all) · FAST (default: --fast) · N_REPEATS (default: 3)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ABLATION_TYPE="${1:-${ABLATION_TYPE:-all}}"
FAST="${FAST---fast}"
N_REPEATS="${N_REPEATS:-3}"

echo ">> Ablation: type=${ABLATION_TYPE} n_repeats=${N_REPEATS} ${FAST}"
python prime/main.py --mode ablation --ablation_type "${ABLATION_TYPE}" --n_repeats "${N_REPEATS}" ${FAST}
echo ">> Done. Results under outputs/."
