#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Main experiments: market suite, ETF generalization, cross-geography, case study.
#
# Usage:
#   bash scripts/run_main_experiments.sh                 # all, fast mode
#   bash scripts/run_main_experiments.sh market_suite    # a single category
#   MAIN_EXP_TYPE=cross_geography FAST= bash scripts/run_main_experiments.sh
#
# Env vars: MAIN_EXP_TYPE (default: all) · FAST (default: --fast; set empty for full)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_EXP_TYPE="${1:-${MAIN_EXP_TYPE:-all}}"
FAST="${FAST---fast}"

echo ">> Main experiments: type=${MAIN_EXP_TYPE} ${FAST}"
python prime/main.py --mode main_exp --main_exp_type "${MAIN_EXP_TYPE}" ${FAST}
echo ">> Done. Results under outputs/."
