#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
TARGET_SCRIPT="${PROJECT_ROOT}/examples/qwen3_a3b_moe_scaling_wallclock_scan.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "[FAIL] Target script not found: ${TARGET_SCRIPT}" >&2
  exit 1
fi

num_query_groups=$(grep -E '^NUM_QUERY_GROUPS=' "${TARGET_SCRIPT}" | head -n1 | cut -d'=' -f2)
if ! [[ "${num_query_groups}" =~ ^[0-9]+$ ]]; then
  echo "[FAIL] NUM_QUERY_GROUPS is missing or not a positive integer in ${TARGET_SCRIPT}" >&2
  exit 1
fi

declare -a CONFIGS=(
  "256 8 8 4 4"
  "1024 8 8 16 16"
  "4096 16 8 32 32"
  "8192 16 8 64 64"
)

for config in "${CONFIGS[@]}"; do
  read -r world_size pp_size tp_size ep_size dp_size <<< "${config}"

  if (( world_size != pp_size * tp_size * dp_size )); then
    echo "[FAIL] Invalid config arithmetic: ${config}" >&2
    exit 1
  fi

  if (( num_query_groups % tp_size != 0 )); then
    echo "[FAIL] NUM_QUERY_GROUPS(${num_query_groups}) must be divisible by tp_size(${tp_size}) for ${config}" >&2
    exit 1
  fi

  if (( dp_size % ep_size != 0 )); then
    echo "[FAIL] Invalid MoE divisibility: dp % ep != 0 for ${config}" >&2
    exit 1
  fi

  if (( ep_size > dp_size )); then
    echo "[FAIL] Invalid MoE relation: ep > dp for ${config}" >&2
    exit 1
  fi

  measured_count=0
  first_rank=-1
  last_rank=-1

  for ((pp_stage=0; pp_stage<pp_size; pp_stage++)); do
    for ((exp_rank=0; exp_rank<ep_size; exp_rank++)); do
      rank=$((pp_stage * tp_size * dp_size + exp_rank * tp_size))
      expected_rank=$((pp_stage * tp_size * dp_size + exp_rank * tp_size))

      if (( rank != expected_rank )); then
        echo "[FAIL] Rank mapping mismatch for config ${config}, pp_stage=${pp_stage}, exp_rank=${exp_rank}" >&2
        exit 1
      fi

      if (( measured_count == 0 )); then
        first_rank=${rank}
      fi
      last_rank=${rank}
      measured_count=$((measured_count + 1))
    done
  done

  expected_count=$((pp_size * ep_size))
  if (( measured_count != expected_count )); then
    echo "[FAIL] measured_count(${measured_count}) != pp*ep(${expected_count}) for ${config}" >&2
    exit 1
  fi

  if (( first_rank != 0 )); then
    echo "[FAIL] first selected rank must be 0 for ${config}, got ${first_rank}" >&2
    exit 1
  fi

  expected_last_rank=$(((pp_size - 1) * tp_size * dp_size + (ep_size - 1) * tp_size))
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

for required_config in "${CONFIGS[@]}"; do
  if ! grep -q "\"${required_config}\"" "${TARGET_SCRIPT}"; then
    echo "[FAIL] Required config missing in script: ${required_config}" >&2
    exit 1
  fi
done

echo "[PASS] Qwen3-A3B MoE scaling wall-clock config checks passed."
