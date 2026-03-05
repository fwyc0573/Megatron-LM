#!/bin/bash
# =============================================================================
# Sequential Scaling-Mode Trace Collection — 6 Cases on a Single GPU
#
# Runs all 6 scaling-mode configurations sequentially on one GPU card.
# Traces (profiler_log/ and memory_traces_scaling/) are written into a single
# timestamped output directory.
#
# Cases:
#   1. Qwen3-MoE   PP=4 EP=4 (FAKE_WORLD_SIZE=16)
#   2. Qwen3-MoE   PP=2 EP=8 (FAKE_WORLD_SIZE=16)
#   3. Qwen3-MoE   PP=8 EP=2 (FAKE_WORLD_SIZE=16)
#   4. DeepSeek-V3  PP=2 EP=8 (FAKE_WORLD_SIZE=16)
#   5. DeepSeek-V3  PP=2 EP=4 (FAKE_WORLD_SIZE=16)
#   6. DeepSeek-V3  PP=4 EP=4 (FAKE_WORLD_SIZE=16)
#
# Usage (background):
#   nohup bash examples/run_all_scaling_traces_h800.sh > /dev/null 2>&1 &
#
# Optional environment overrides:
#   SCALE_GPU=3           — pin to a specific GPU (default: auto-select idlest)
#   OUTPUT_DIR=/path/...  — custom output directory (default: auto-timestamped)
#   FAKE_WORLD_SIZE=16    — number of fake ranks (default: 16)
#   TRAIN_ITERS=10        — training iterations (default: 10)
#   TRACE_START=10        — iteration to start tracing (default: 10)
# =============================================================================

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

# ---------------------------------------------------------------------------
# Auto-select the idlest GPU (used once, shared across all cases)
# ---------------------------------------------------------------------------
pick_idle_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. Please set SCALE_GPU explicitly." >&2
    return 1
  fi
  local candidate
  candidate=$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t',' -k2,2n \
      | head -n1 \
      | cut -d',' -f1 \
      | tr -d ' '
  )
  if [[ -z "${candidate}" ]]; then
    echo "[ERROR] Failed to auto-select idle GPU. Please set SCALE_GPU explicitly." >&2
    return 1
  fi
  echo "${candidate}"
}

SCALE_GPU=${SCALE_GPU:-}
if [[ -z "${SCALE_GPU}" ]]; then
  SCALE_GPU=$(pick_idle_gpu)
fi

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/scaling_traces_h800_${TIMESTAMP}"}
mkdir -p "${OUTPUT_DIR}/logs"

# Redirect all stdout/stderr to master log
exec > >(tee -a "${OUTPUT_DIR}/logs/master.log") 2>&1

echo "============================================================"
echo " Scaling-Mode Trace Collection — 6 Cases"
echo " GPU:     ${SCALE_GPU}"
echo " Output:  ${OUTPUT_DIR}"
echo " Started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Common parameters (exported so sub-scripts inherit them)
# ---------------------------------------------------------------------------
export MODE=scaling
export FAKE_WORLD_SIZE=${FAKE_WORLD_SIZE:-16}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export TRACE_START=${TRACE_START:-10}
export SEQ_LEN=${SEQ_LEN:-2048}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRACE_SUBOP_SYNC_MODE=${TRACE_SUBOP_SYNC_MODE:-global}
export TRACE_MEMORY=1
export TRACE_MEMORY_INTERVAL=0.01
export SCALE_GPU

# ---------------------------------------------------------------------------
# Change to output directory so all relative-path traces land here
# (profiler_log/, memory_traces_scaling/, profiler_log/scaling_replay_cache/)
# ---------------------------------------------------------------------------
cd "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
TOTAL_CASES=6
PASSED=0
FAILED=0
FAILED_LIST=()

# ---------------------------------------------------------------------------
# run_case — run one scaling-mode case
#   $1: case label (e.g. "case1_qwen3_pp4_ep4")
#   $2: script path (absolute)
#   $3+: extra env vars (e.g. PP=4 EP=4)
# ---------------------------------------------------------------------------
run_case() {
  local case_name="$1"
  local script="$2"
  shift 2
  local extra_env=("$@")

  echo ""
  echo "------------------------------------------------------------"
  echo " [${case_name}] Started:  $(date)"
  echo " [${case_name}] Script:   ${script}"
  echo " [${case_name}] Env:      ${extra_env[*]}"
  echo "------------------------------------------------------------"

  local log_file="${OUTPUT_DIR}/logs/${case_name}.log"
  local start_ts
  start_ts=$(date +%s)

  # Run the scaling script; do NOT abort the outer script on failure
  env "${extra_env[@]}" bash "${script}" > "${log_file}" 2>&1
  local rc=$?

  local end_ts
  end_ts=$(date +%s)
  local elapsed=$(( end_ts - start_ts ))

  if [[ ${rc} -eq 0 ]]; then
    echo " [${case_name}] ✓ PASSED  (${elapsed}s)"
    PASSED=$((PASSED + 1))
  else
    echo " [${case_name}] ✗ FAILED  (exit ${rc}, ${elapsed}s)  → check ${log_file}"
    FAILED=$((FAILED + 1))
    FAILED_LIST+=("${case_name}")
  fi
  echo " [${case_name}] Finished: $(date)"
}

# =====================================================================
# CASE 1: Qwen3-MoE  PP=4 EP=4
# =====================================================================
run_case "case1_qwen3_pp4_ep4" \
  "${PROJECT_ROOT}/examples/pretrain_qwen3_30b_a3b_moe.sh" \
  PP=4 EP=4

# =====================================================================
# CASE 2: Qwen3-MoE  PP=2 EP=8
# =====================================================================
run_case "case2_qwen3_pp2_ep8" \
  "${PROJECT_ROOT}/examples/pretrain_qwen3_30b_a3b_moe.sh" \
  PP=2 EP=8

# =====================================================================
# CASE 3: Qwen3-MoE  PP=8 EP=2
# =====================================================================
run_case "case3_qwen3_pp8_ep2" \
  "${PROJECT_ROOT}/examples/pretrain_qwen3_30b_a3b_moe.sh" \
  PP=8 EP=2

# =====================================================================
# CASE 4: DeepSeek-V3  PP=2 EP=8
# =====================================================================
run_case "case4_dsv3_pp2_ep8" \
  "${PROJECT_ROOT}/examples/pretrain_deepseek_v3_moe_aligned.sh" \
  PP=2 EP=8

# =====================================================================
# CASE 5: DeepSeek-V3  PP=2 EP=4
# =====================================================================
run_case "case5_dsv3_pp2_ep4" \
  "${PROJECT_ROOT}/examples/pretrain_deepseek_v3_moe_aligned.sh" \
  PP=2 EP=4

# =====================================================================
# CASE 6: DeepSeek-V3  PP=4 EP=4
# =====================================================================
run_case "case6_dsv3_pp4_ep4" \
  "${PROJECT_ROOT}/examples/pretrain_deepseek_v3_moe_aligned.sh" \
  PP=4 EP=4

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "============================================================"
echo " ALL CASES COMPLETED"
echo "============================================================"
echo ""
echo " GPU used:       ${SCALE_GPU}"
echo " Passed / Total: ${PASSED} / ${TOTAL_CASES}"
echo " Failed / Total: ${FAILED} / ${TOTAL_CASES}"
if (( FAILED > 0 )); then
  echo " Failed cases:   ${FAILED_LIST[*]}"
fi
echo ""
echo " Output directory: ${OUTPUT_DIR}"
echo ""
echo " Profiler trace directories:"
if [[ -d "${OUTPUT_DIR}/profiler_log" ]]; then
  for d in "${OUTPUT_DIR}/profiler_log"/*/; do
    if [[ -d "${d}" && "$(basename "${d}")" != "scaling_replay_cache" ]]; then
      local_count=$(find "${d}" -maxdepth 1 -name '*.txt' | wc -l)
      echo "   $(basename "${d}")/ — ${local_count} trace files"
    fi
  done
else
  echo "   (none)"
fi
echo ""
echo " Memory trace files:"
if [[ -d "${OUTPUT_DIR}/memory_traces_scaling" ]]; then
  mem_count=$(find "${OUTPUT_DIR}/memory_traces_scaling" -name '*.json' | wc -l)
  echo "   ${mem_count} JSON files in memory_traces_scaling/"
else
  echo "   (none)"
fi
echo ""
echo " Per-case logs:    ${OUTPUT_DIR}/logs/"
echo " Master log:       ${OUTPUT_DIR}/logs/master.log"
echo ""
echo " Finished: $(date)"
echo "============================================================"
