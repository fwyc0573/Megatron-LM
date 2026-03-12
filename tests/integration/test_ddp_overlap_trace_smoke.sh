#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if ! command -v torchrun >/dev/null 2>&1; then
  echo "[FAIL] torchrun not found in PATH." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[FAIL] nvidia-smi not found in PATH." >&2
  exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | awk '{print $1}')
if (( GPU_COUNT < 2 )); then
  echo "[FAIL] DDP overlap smoke requires at least 2 visible GPUs, got ${GPU_COUNT}." >&2
  exit 1
fi

ARTIFACT_ROOT="${PROJECT_ROOT}/tests/integration/artifacts"
mkdir -p "${ARTIFACT_ROOT}"
RUN_TAG="ddp_overlap_trace_smoke_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_TAG}"
mkdir -p "${RUN_DIR}"

DIST_LOG="${RUN_DIR}/distributed.log"
SCALE_LOG="${RUN_DIR}/scaling.log"
DIST_GPUS=${TRACE_SMOKE_DIST_GPUS:-0,1}
SCALE_GPU=${TRACE_SMOKE_SCALE_GPU:-0}

run_distributed() {
  echo "[INFO] Running distributed DDP overlap smoke on GPUs ${DIST_GPUS}" | tee -a "${DIST_LOG}"
  CUDA_VISIBLE_DEVICES="${DIST_GPUS}" \
  torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=26207 \
    pretrain_llama.py \
    --distributed-backend nccl \
    --use-mcore-models \
    --transformer-impl local \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 4096 \
    --make-vocab-size-divisible-by 128 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --num-layers 2 \
    --hidden-size 128 \
    --ffn-hidden-size 512 \
    --num-attention-heads 4 \
    --seq-length 32 \
    --max-position-embeddings 32 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --train-iters 3 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 3 \
    --lr-warmup-iters 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --eval-interval 10000 \
    --bf16 \
    --do-trace True \
    --trace-start 2 \
    --overlap-grad-reduce \
    --trace-ddp-grad-overlap \
    > "${DIST_LOG}" 2>&1
}

run_scaling() {
  echo "[INFO] Running scaling DDP overlap smoke on GPU ${SCALE_GPU}" | tee -a "${SCALE_LOG}"
  CUDA_VISIBLE_DEVICES="${SCALE_GPU}" \
  torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=26217 \
    pretrain_llama.py \
    --distributed-backend nccl \
    --use-mcore-models \
    --transformer-impl local \
    --mock-data \
    --dataloader-type cyclic \
    --tokenizer-type NullTokenizer \
    --vocab-size 4096 \
    --make-vocab-size-divisible-by 128 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --num-layers 2 \
    --hidden-size 128 \
    --ffn-hidden-size 512 \
    --num-attention-heads 4 \
    --seq-length 32 \
    --max-position-embeddings 32 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --train-iters 3 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 3 \
    --lr-warmup-iters 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --eval-interval 10000 \
    --bf16 \
    --num-experts 1 \
    --moe-layer-freq 1 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 1 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --disable-bias-linear \
    --do-trace True \
    --trace-start 2 \
    --overlap-grad-reduce \
    --trace-ddp-grad-overlap \
    --is-scaling-mode \
    --fake-world-size 2 \
    --fake-wrank 0 \
    --fake-gpus-per-node 2 \
    --fake-local-rank 0 \
    --fake-pp 1 \
    --fake-dp 2 \
    --fake-tp 1 \
    --fake-exp 1 \
    --fake-num-experts 1 \
    --fake-current-rank-id 0 \
    > "${SCALE_LOG}" 2>&1
}

latest_file() {
  local pattern_root=$1
  find "${pattern_root}" -type f -name '*.txt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

assert_contains() {
  local file=$1
  local pattern=$2
  if ! rg -q "${pattern}" "${file}"; then
    echo "[FAIL] Missing pattern '${pattern}' in ${file}" >&2
    exit 1
  fi
}

assert_not_contains() {
  local file=$1
  local pattern=$2
  if rg -q "${pattern}" "${file}"; then
    echo "[FAIL] Unexpected pattern '${pattern}' in ${file}" >&2
    exit 1
  fi
}

run_distributed
DIST_TRACE_DIR="${PROJECT_ROOT}/realistic_trace/pp1_tp1_exp1_expnNone_dp2_nl2_hs128_sl32"
DIST_TRACE_FILE=$(latest_file "${DIST_TRACE_DIR}")
if [[ -z "${DIST_TRACE_FILE}" || ! -f "${DIST_TRACE_FILE}" ]]; then
  echo "[FAIL] Distributed trace file not found under ${DIST_TRACE_DIR}" >&2
  exit 1
fi
assert_contains "${DIST_TRACE_FILE}" 'ddp_grad_comm\('
assert_contains "${DIST_TRACE_FILE}" 'completion_observed_timestamp_ms=[0-9]'
assert_contains "${DIST_TRACE_FILE}" 'wait_cmd_uid=cmd-'
assert_contains "${DIST_TRACE_FILE}" 'op_semantics=wait_flush_only'

run_scaling
SCALE_TRACE_DIR="${PROJECT_ROOT}/profiler_log/pp1_tp1_ep1_expn1_dp2_nl2_hs128_sl32"
SCALE_TRACE_FILE=$(latest_file "${SCALE_TRACE_DIR}")
if [[ -z "${SCALE_TRACE_FILE}" || ! -f "${SCALE_TRACE_FILE}" ]]; then
  echo "[FAIL] Scaling trace file not found under ${SCALE_TRACE_DIR}" >&2
  exit 1
fi
assert_contains "${SCALE_TRACE_FILE}" 'ddp_grad_comm\('
assert_contains "${SCALE_TRACE_FILE}" 'metadata_only=True'
assert_contains "${SCALE_TRACE_FILE}" 'status=launch_only'
assert_contains "${SCALE_TRACE_FILE}" 'trigger_op=backward_step'
assert_not_contains "${SCALE_TRACE_FILE}" 'trigger_op=loss_func'
assert_contains "${SCALE_TRACE_FILE}" 'op_semantics=metadata_placeholder'
assert_contains "${SCALE_TRACE_FILE}" 'finalize_base_duration_ms=[0-9]'

echo "[PASS] DDP overlap trace smoke test passed."
echo "[INFO] Artifacts: ${RUN_DIR}"
echo "[INFO] Distributed trace: ${DIST_TRACE_FILE}"
echo "[INFO] Scaling trace: ${SCALE_TRACE_FILE}"
