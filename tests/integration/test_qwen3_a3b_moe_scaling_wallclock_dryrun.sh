#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
TARGET_SCRIPT="${PROJECT_ROOT}/examples/qwen3_a3b_moe_scaling_wallclock_scan.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "[FAIL] Target script not found: ${TARGET_SCRIPT}" >&2
  exit 1
fi

ARTIFACT_ROOT="${PROJECT_ROOT}/tests/integration/artifacts"
mkdir -p "${ARTIFACT_ROOT}"

RUN_TAG="dryrun_qwen3_a3b_moe_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_TAG}"
mkdir -p "${RUN_DIR}"

OUTPUT_CSV="${RUN_DIR}/qwen3_a3b_moe_scaling_wallclock_timing.csv"
LOG_ROOT="${RUN_DIR}/logs"
DRYRUN_LOG="${RUN_DIR}/dryrun_stdout.log"

SCALE_GPU=0 \
DRY_RUN=1 \
OUTPUT_CSV="${OUTPUT_CSV}" \
LOG_ROOT="${LOG_ROOT}" \
bash "${TARGET_SCRIPT}" > "${DRYRUN_LOG}" 2>&1

if [[ ! -f "${OUTPUT_CSV}" ]]; then
  echo "[FAIL] Output CSV not generated: ${OUTPUT_CSV}" >&2
  exit 1
fi

expected_header="world_size,pp_size,tp_size,ep_size,dp_size,measured_ranks_count,single_iter_wallclock_seconds,estimated_5_iters_seconds"
actual_header=$(head -n1 "${OUTPUT_CSV}")
if [[ "${actual_header}" != "${expected_header}" ]]; then
  echo "[FAIL] CSV header mismatch." >&2
  echo "[FAIL] expected: ${expected_header}" >&2
  echo "[FAIL] actual  : ${actual_header}" >&2
  exit 1
fi

row_count=$(tail -n +2 "${OUTPUT_CSV}" | wc -l | awk "{print \$1}")
if [[ "${row_count}" != "4" ]]; then
  echo "[FAIL] Expected 4 data rows, got ${row_count}." >&2
  exit 1
fi

mapfile -t measured_counts < <(tail -n +2 "${OUTPUT_CSV}" | cut -d"," -f6)
expected_counts=(32 128 512 1024)
for idx in "${!expected_counts[@]}"; do
  if [[ "${measured_counts[idx]}" != "${expected_counts[idx]}" ]]; then
    echo "[FAIL] measured_ranks_count mismatch at row $((idx + 1)): expected ${expected_counts[idx]}, got ${measured_counts[idx]}" >&2
    exit 1
  fi
done

if ! awk -F"," "NR > 1 { if (NF != 8) { exit 1 } }" "${OUTPUT_CSV}"; then
  echo "[FAIL] CSV row has invalid field count (expect 8)." >&2
  exit 1
fi

if ! awk -F"," "NR > 1 { expected=sprintf(\"%.6f\", \$7 * 5); if (\$8 != expected) { exit 1 } }" "${OUTPUT_CSV}"; then
  echo "[FAIL] estimated_5_iters_seconds != single_iter_wallclock_seconds * 5." >&2
  exit 1
fi

echo "[PASS] Qwen3-A3B MoE dry-run integration checks passed."
echo "[INFO] Artifacts: ${RUN_DIR}"
