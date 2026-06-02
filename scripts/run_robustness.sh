#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Robustness studies: noise / topk / lr / epochs / crash_label / rolling_window /
# macro_degradation / energy_bounds / stress_period.
#
# Usage:
#   bash scripts/run_robustness.sh              # all, fast mode
#   bash scripts/run_robustness.sh noise        # a single study
#
# Env vars: ROBUSTNESS_TYPE (default: all) · FAST (default: --fast)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROBUSTNESS_TYPE="${1:-${ROBUSTNESS_TYPE:-all}}"
FAST="${FAST---fast}"

echo ">> Robustness: type=${ROBUSTNESS_TYPE} ${FAST}"
python prime/main.py --mode robustness --robustness_type "${ROBUSTNESS_TYPE}" ${FAST}
echo ">> Done. Results under outputs/."
