#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

CASE_DIR="${CASE_DIR:-}"
NO_OVERLAP_CASE_DIR="${NO_OVERLAP_CASE_DIR:-}"
WORLD_SIZE="${WORLD_SIZE:-4}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
EXP_SIZE="${EXP_SIZE:-1}"
LOCAL_SIZE="${LOCAL_SIZE:-4}"

if [[ -z "${CASE_DIR}" ]]; then
  echo "ERROR: CASE_DIR is required. It must contain schedule/, database_profile/, global_ranks_profile/, and slowdown_assets/."
  exit 1
fi

SCHEDULE_DIR="${SCHEDULE_DIR:-${CASE_DIR}/schedule}"
DATABASE_DIR="${DATABASE_DIR:-${CASE_DIR}/database_profile}"
TRACE_DIR="${TRACE_DIR:-${CASE_DIR}/global_ranks_profile}"
SLOWDOWN_ASSETS_DIR="${SLOWDOWN_ASSETS_DIR:-${CASE_DIR}/slowdown_assets}"

echo "[1/4] Baseline simulate run with slowdown disabled (default)"
python simu_main.py \
  --framework megatron-lm \
  --mode simulate \
  --trace-dir "${TRACE_DIR}" \
  --schedule-dir "${SCHEDULE_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --no-visualize

echo
echo "[2/4] Simulate run with slowdown explicitly enabled"
python simu_main.py \
  --framework megatron-lm \
  --mode simulate \
  --trace-dir "${TRACE_DIR}" \
  --schedule-dir "${SCHEDULE_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --enable-slowdown \
  --slowdown-assets-dir "${SLOWDOWN_ASSETS_DIR}" \
  --no-visualize

echo
echo "[3/4] Fail-fast case: force overlap off while trace contains DDP overlap metadata"
if python simu_main.py \
  --framework megatron-lm \
  --mode simulate \
  --trace-dir "${TRACE_DIR}" \
  --schedule-dir "${SCHEDULE_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --overlap-mode off \
  --no-visualize; then
  echo "ERROR: overlap-mode=off was expected to fail fast for overlap-bearing traces."
  exit 1
fi
echo "Observed expected fail-fast behavior for overlap-mode=off."

echo
echo "[4/4] Warning-only case: slowdown requested on traces without overlap metadata"
if [[ -n "${NO_OVERLAP_CASE_DIR}" ]]; then
  python simu_main.py \
    --framework megatron-lm \
    --mode simulate \
    --trace-dir "${NO_OVERLAP_CASE_DIR}/global_ranks_profile" \
    --schedule-dir "${NO_OVERLAP_CASE_DIR}/schedule" \
    --database-dir "${NO_OVERLAP_CASE_DIR}/database_profile" \
    --world-size "${WORLD_SIZE}" \
    --pp-size "${PP_SIZE}" \
    --tp-size "${TP_SIZE}" \
    --exp-size "${EXP_SIZE}" \
    --local-size "${LOCAL_SIZE}" \
    --enable-slowdown \
    --no-visualize
else
  echo "Skip warning-only case. Set NO_OVERLAP_CASE_DIR to a trace set without DDP overlap metadata."
fi

echo
echo "DDP overlap / slowdown example cases finished."
