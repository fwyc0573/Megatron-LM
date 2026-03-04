#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
TARGET_SCRIPT="${PROJECT_ROOT}/examples/gpt175b_scaling_wallclock_scan.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "[FAIL] Target script not found: ${TARGET_SCRIPT}" >&2
  exit 1
fi

declare -a CONFIGS=(
  "256 16 8 2"
  "1024 16 8 8"
  "4096 32 8 16"
  "8192 32 8 32"
)

for config in "${CONFIGS[@]}"; do
  read -r world_size pp_size tp_size dp_size <<< "${config}"

  if (( world_size != pp_size * tp_size * dp_size )); then
    echo "[FAIL] Invalid config arithmetic: ${config}" >&2
    exit 1
  fi

  step=$((tp_size * dp_size))
  measured_count=0
  first_rank=-1
  last_rank=-1

  for ((pp_stage=0; pp_stage<pp_size; pp_stage++)); do
    rank=$((pp_stage * step))
    expected_rank=$((pp_stage * step))

    if (( rank != expected_rank )); then
      echo "[FAIL] Rank mapping mismatch for config ${config} at pp_stage=${pp_stage}" >&2
      exit 1
    fi

    if (( pp_stage == 0 )); then
      first_rank=${rank}
    fi
    last_rank=${rank}
    measured_count=$((measured_count + 1))
  done

  if (( measured_count != pp_size )); then
    echo "[FAIL] measured_count(${measured_count}) != pp_size(${pp_size}) for ${config}" >&2
    exit 1
  fi

  if (( first_rank != 0 )); then
    echo "[FAIL] first selected rank must be 0 for ${config}, got ${first_rank}" >&2
    exit 1
  fi

  expected_last_rank=$(((pp_size - 1) * step))
  if (( last_rank != expected_last_rank )); then
    echo "[FAIL] last rank mismatch for ${config}: expected ${expected_last_rank}, got ${last_rank}" >&2
    exit 1
  fi

  global_batch_size=$((1 * dp_size))
  if (( global_batch_size != dp_size )); then
    echo "[FAIL] global_batch_size mismatch for ${config}: expected ${dp_size}, got ${global_batch_size}" >&2
    exit 1
  fi
done

if ! grep -q '"8192 32 8 32"' "${TARGET_SCRIPT}"; then
  echo "[FAIL] Corrected 8192 configuration not found in script." >&2
  exit 1
fi

if grep -q '"8192 32 8 64"' "${TARGET_SCRIPT}"; then
  echo "[FAIL] Invalid 8192 DP=64 configuration still present in script." >&2
  exit 1
fi

echo "[PASS] GPT-175B scaling wall-clock config checks passed."
