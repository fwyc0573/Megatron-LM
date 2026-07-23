#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Required. Must contain: schedule/, database_profile/, global_ranks_profile/
# Example:
#   MIXTRAL_INPUT_DIR=simulation_inputs/megatron_operation_log/<your_mixtral_case>
MIXTRAL_INPUT_DIR="${MIXTRAL_INPUT_DIR:-}"

WORLD_SIZE="${WORLD_SIZE:-16}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
EXP_SIZE="${EXP_SIZE:-1}"
LOCAL_SIZE="${LOCAL_SIZE:-8}"

if [[ -z "${MIXTRAL_INPUT_DIR}" ]]; then
  echo "ERROR: MIXTRAL_INPUT_DIR is required."
  echo "It must point to a directory containing schedule/, database_profile/, and global_ranks_profile/."
  exit 1
fi

if [[ -d "${REPO_ROOT}/${MIXTRAL_INPUT_DIR}" ]]; then
  MIXTRAL_INPUT_DIR="${REPO_ROOT}/${MIXTRAL_INPUT_DIR}"
fi

if [[ ! -d "${MIXTRAL_INPUT_DIR}" ]]; then
  echo "ERROR: MIXTRAL_INPUT_DIR not found: ${MIXTRAL_INPUT_DIR}"
  exit 1
fi

SCHEDULE_DIR="${MIXTRAL_INPUT_DIR}/schedule"
DATABASE_DIR="${MIXTRAL_INPUT_DIR}/database_profile"
TRACE_DIR="${MIXTRAL_INPUT_DIR}/global_ranks_profile"

if [[ ! -d "${SCHEDULE_DIR}" ]]; then
  echo "ERROR: missing schedule directory: ${SCHEDULE_DIR}"
  exit 1
fi
if [[ ! -d "${DATABASE_DIR}" ]]; then
  echo "ERROR: missing database_profile directory: ${DATABASE_DIR}"
  exit 1
fi
if [[ ! -d "${TRACE_DIR}" ]]; then
  echo "ERROR: missing global_ranks_profile directory: ${TRACE_DIR}"
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

echo "[1/2] Mixtral 16-GPU 2-node simulate mode"
python simu_main.py \
  --framework megatron-lm \
  --mode simulate \
  --schedule-dir "${SCHEDULE_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --no-visualize

echo
echo "[2/2] Mixtral 16-GPU 2-node profile mode"
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

echo
echo "Mixtral template run finished successfully."
