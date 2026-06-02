#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fast end-to-end smoke check of the whole PRIME system.
# Covers every experiment category with tiny isolated data + light training,
# auto-generating mock data (no real data required). Finishes in a few minutes.
#
# Usage:  bash scripts/smoke_test.sh [extra args passed to integrity_check.py]
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ">> PRIME smoke test (mock data, baselines skipped)"
python prime/integrity_check.py --quick --skip_baselines "$@"
echo ">> Done. See outputs/integrity_check/ for the report."
