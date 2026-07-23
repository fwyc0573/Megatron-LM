#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SENDRECV_DIR="${SENDRECV_DIR:-data/h800_dgx_roce_sendrecv}"
THRESHOLD_PCT="${THRESHOLD_PCT:-10}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

python tests/performance/p2p_alignment_suite.py \
  --sendrecv-dir "${SENDRECV_DIR}" \
  --collective-sim-repo-root src/core/cc_backend/collective-sim \
  --threshold-pct "${THRESHOLD_PCT}" \
  --json-out task_memory/task_2026-02-27_sim_restructure/p2p_alignment_suite_2026-02-28.json \
  --report-path task_memory/task_2026-02-27_sim_restructure/test_report_2026-02-28_p2p_semantic_alignment_round3.md
