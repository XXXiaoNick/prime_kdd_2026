#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Baselines: market_index, lstm, alstm, gru, transformer, patchtst, gpt4ts,
#            time_llm, alphastock, deeptrader.
#
# Usage:
#   bash scripts/run_baselines.sh                          # all baselines, fast
#   bash scripts/run_baselines.sh lstm                     # a single baseline
#   bash scripts/run_baselines.sh all csi500 stock         # choose market / asset
#
# Args: $1 baseline (default all) · $2 market_profile (default csi500) · $3 asset_type (default stock)
# Env vars: FAST (default: --fast)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASELINE="${1:-all}"
MARKET="${2:-csi500}"
ASSET="${3:-stock}"
FAST="${FAST---fast}"

echo ">> Baseline: name=${BASELINE} market=${MARKET} asset=${ASSET} ${FAST}"
echo "   (note: GBDT/LLM baselines may need extra deps, e.g. lightgbm)"
python prime/main.py --mode baseline --baseline_name "${BASELINE}" \
    --market_profile "${MARKET}" --asset_type "${ASSET}" ${FAST}
echo ">> Done. Results under outputs_baselines/ (baseline_summary.csv)."
