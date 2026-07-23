#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# This script demonstrates a 16-GPU profile-mode invocation.
# Required input directories can be overridden by environment variables.
TRACE_DIR="${TRACE_DIR:-simulation_inputs/megatron_operation_log/quick_evaluate_16gpus_4pp_2pp_13b/global_ranks_profile}"
DATABASE_DIR="${DATABASE_DIR:-}"
WORLD_SIZE="${WORLD_SIZE:-16}"
PP_SIZE="${PP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-2}"
EXP_SIZE="${EXP_SIZE:-1}"
LOCAL_SIZE="${LOCAL_SIZE:-8}"

if [[ -z "${DATABASE_DIR}" ]]; then
  echo "ERROR: DATABASE_DIR is required."
  echo "Set DATABASE_DIR to a matching database_profile directory before running."
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/${TRACE_DIR}" && ! -d "${TRACE_DIR}" ]]; then
  echo "ERROR: TRACE_DIR not found: ${TRACE_DIR}"
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/${DATABASE_DIR}" && ! -d "${DATABASE_DIR}" ]]; then
  echo "ERROR: DATABASE_DIR not found: ${DATABASE_DIR}"
  exit 1
fi

if [[ -d "${REPO_ROOT}/${TRACE_DIR}" ]]; then
  TRACE_DIR="${REPO_ROOT}/${TRACE_DIR}"
fi
if [[ -d "${REPO_ROOT}/${DATABASE_DIR}" ]]; then
  DATABASE_DIR="${REPO_ROOT}/${DATABASE_DIR}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

echo "Running 16-GPU quick-eval profile case..."
python simu_main.py \
  --framework megatron-lm \
  --mode profile \
  --trace-dir "${TRACE_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --no-visualize

echo "16-GPU quick-eval run finished successfully."
