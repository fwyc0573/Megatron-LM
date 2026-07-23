#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRACE_DIR="${TRACE_DIR:-simulation_inputs/megatron_operation_log/moe_6.7b_2pp_1tp_2dp/global_ranks_profile}"
WORLD_SIZE="${WORLD_SIZE:-4}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
EXP_SIZE="${EXP_SIZE:-1}"
LOCAL_SIZE="${LOCAL_SIZE:-4}"
PP_DOMAIN_DIM="${PP_DOMAIN_DIM:-DP}"
REPORT_PATH="${REPORT_PATH:-task_memory/task_2026-02-27_sim_restructure/test_report_2026-02-28_cc_backend_cross_validation.md}"
JSON_OUT="${JSON_OUT:-task_memory/task_2026-02-27_sim_restructure/cc_backend_cross_validation_2026-02-28.json}"

if [[ ! -d "${REPO_ROOT}/${TRACE_DIR}" && ! -d "${TRACE_DIR}" ]]; then
  echo "ERROR: TRACE_DIR not found: ${TRACE_DIR}"
  exit 1
fi

if [[ -d "${REPO_ROOT}/${TRACE_DIR}" ]]; then
  TRACE_DIR="${REPO_ROOT}/${TRACE_DIR}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

python tests/performance/compare_cc_backends.py \
  --trace-dir "${TRACE_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --pp-domain-dim "${PP_DOMAIN_DIM}" \
  --report-path "${REPORT_PATH}" \
  --json-out "${JSON_OUT}"

echo "CC backend comparison finished."
echo "Report: ${REPORT_PATH}"
echo "JSON:   ${JSON_OUT}"
