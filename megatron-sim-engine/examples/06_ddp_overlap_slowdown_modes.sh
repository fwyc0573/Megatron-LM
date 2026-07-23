#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TRACE_DIR=${TRACE_DIR:-}
TRACE_DIR_NO_OVERLAP=${TRACE_DIR_NO_OVERLAP:-}
SCHEDULE_DIR=${SCHEDULE_DIR:-}
DATABASE_DIR=${DATABASE_DIR:-}
WORLD_SIZE=${WORLD_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
TP_SIZE=${TP_SIZE:-1}
EXP_SIZE=${EXP_SIZE:-1}
LOCAL_SIZE=${LOCAL_SIZE:-4}
SLOWDOWN_ASSETS_DIR=${SLOWDOWN_ASSETS_DIR:-}

if [[ -z "${TRACE_DIR}" || -z "${SCHEDULE_DIR}" || -z "${DATABASE_DIR}" ]]; then
  echo "[ERROR] TRACE_DIR, SCHEDULE_DIR, and DATABASE_DIR must be set." >&2
  exit 1
fi

BASE_CMD=(
  python simu_main.py
  --framework megatron-lm
  --mode simulate
  --trace-dir "${TRACE_DIR}"
  --schedule-dir "${SCHEDULE_DIR}"
  --database-dir "${DATABASE_DIR}"
  --world-size "${WORLD_SIZE}"
  --pp-size "${PP_SIZE}"
  --tp-size "${TP_SIZE}"
  --exp-size "${EXP_SIZE}"
  --local-size "${LOCAL_SIZE}"
  --no-visualize
)

cd "${REPO_ROOT}"

echo "[1/3] Slowdown disabled by default"
"${BASE_CMD[@]}"

if [[ -z "${SLOWDOWN_ASSETS_DIR}" ]]; then
  echo "[ERROR] SLOWDOWN_ASSETS_DIR must be set for slowdown-enabled examples." >&2
  exit 1
fi

echo
echo "[2/3] Slowdown explicitly enabled"
"${BASE_CMD[@]}" \
  --enable-slowdown \
  --slowdown-assets-dir "${SLOWDOWN_ASSETS_DIR}"

echo
echo "[3/3] Force overlap off and expect fail-fast when trace carries overlap metadata"
set +e
"${BASE_CMD[@]}" --overlap-mode off
status=$?
set -e
if [[ ${status} -eq 0 ]]; then
  echo "[ERROR] Expected overlap-mode off to fail fast for overlap-aware traces." >&2
  exit 1
fi
echo "[INFO] Observed expected fail-fast with overlap-aware trace (exit=${status})."

if [[ -n "${TRACE_DIR_NO_OVERLAP}" ]]; then
  echo
  echo "[4/4] Optional warning-only path: no-overlap trace keeps simulation running"
  python simu_main.py \
    --framework megatron-lm \
    --mode simulate \
    --trace-dir "${TRACE_DIR_NO_OVERLAP}" \
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
fi
