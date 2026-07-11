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
if ! command -v nsys >/dev/null 2>&1; then
  echo "[FAIL] nsys not found in PATH." >&2
  exit 1
fi

ARTIFACT_ROOT="${PROJECT_ROOT}/tests/e2e/artifacts"
mkdir -p "${ARTIFACT_ROOT}"
RUN_TAG="ddp_slowdown_simulate_smoke_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_TAG}"
TRACE_DIR="${RUN_DIR}/trace_db"
SCHEDULE_DIR="${RUN_DIR}/schedule"
NSYS_DIR="${RUN_DIR}/nsys"
SLOWDOWN_ASSETS_DIR="${RUN_DIR}/slowdown_assets"
mkdir -p "${TRACE_DIR}" "${SCHEDULE_DIR}" "${NSYS_DIR}" "${SLOWDOWN_ASSETS_DIR}"

SCALE_GPU=${SLOWDOWN_E2E_SCALE_GPU:-7}
WORLD_SIZE=4
LOCAL_SIZE=4
PP_SIZE=1
TP_SIZE=1
EXP_SIZE=1
FAKE_DP=4
NUM_LAYERS=12
HIDDEN_SIZE=256
FFN_HIDDEN_SIZE=1024
NUM_HEADS=8
SEQ_LEN=256
MBS=1
GBS=4
TRAIN_ITERS=3
TRACE_START=2
TRACE_ROOT="${PROJECT_ROOT}/profiler_log/pp1_tp1_ep1_expnNone_dp4_nl12_hs256_sl256"
MODEL_PATH="${PROJECT_ROOT}/Echo-slowdown/training_testing/output/xgb_model.json"
SCALER_PATH="${SLOWDOWN_SCALER_PATH:-${PROJECT_ROOT}/Echo-slowdown/training_testing/output/standard_scaler.json}"
NCU_METRICS_CSV="${PROJECT_ROOT}/Echo-slowdown/merge/input/kernel_metric_output.csv"
COMPARE_JSON="${RUN_DIR}/simulate_compare.json"
SUMMARY_MD="${RUN_DIR}/summary.md"
RUN_ALL_SCRIPT="${RUN_DIR}/run_all_scaling_ranks.sh"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[FAIL] Slowdown model not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SCALER_PATH}" ]]; then
  echo "[FAIL] Slowdown scaler spec not found: ${SCALER_PATH}" >&2
  exit 1
fi
if [[ ! -f "${NCU_METRICS_CSV}" ]]; then
  echo "[FAIL] Kernel metrics CSV not found: ${NCU_METRICS_CSV}" >&2
  exit 1
fi

latest_rank_file() {
  local dir=$1
  local rank_id=$2
  find "${dir}" -maxdepth 1 -type f -name "*rank${rank_id}_*.txt" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

cat > "${RUN_ALL_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${SCALE_GPU}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}"
for fake_rank_id in 0 1 2 3; do
  torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="\$((26420 + fake_rank_id))" \
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
    --num-layers "${NUM_LAYERS}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
    --num-attention-heads "${NUM_HEADS}" \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --train-iters "${TRAIN_ITERS}" \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters "${TRAIN_ITERS}" \
    --lr-warmup-iters 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --eval-interval 10000 \
    --bf16 \
    --do-trace True \
    --trace-start "${TRACE_START}" \
    --trace-subop-sync-mode global \
    --trace-kernel-ground-truth \
    --trace-kernel-ground-truth-prefix cmd_trace \
    --trace-kernel-ground-truth-phase \
    --trace-kernel-boundary-sync-mode global \
    --overlap-grad-reduce \
    --trace-ddp-grad-overlap \
    --is-scaling-mode \
    --fake-world-size "${WORLD_SIZE}" \
    --fake-wrank 0 \
    --fake-gpus-per-node "${LOCAL_SIZE}" \
    --fake-local-rank 0 \
    --fake-pp "${PP_SIZE}" \
    --fake-dp "${FAKE_DP}" \
    --fake-tp "${TP_SIZE}" \
    --fake-exp "${EXP_SIZE}" \
    --fake-current-rank-id "\${fake_rank_id}"
done
EOF
chmod +x "${RUN_ALL_SCRIPT}"

NSYS_REPORT_BASE="${NSYS_DIR}/all_ranks_scaling"
NSYS_LOG="${RUN_DIR}/all_ranks_nsys.log"
CUDA_VISIBLE_DEVICES="${SCALE_GPU}" \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --wait=all \
  --trace-fork-before-exec=true \
  --force-overwrite true \
  --output "${NSYS_REPORT_BASE}" \
  bash "${RUN_ALL_SCRIPT}" \
  > "${NSYS_LOG}" 2>&1

if [[ ! -f "${NSYS_REPORT_BASE}.nsys-rep" ]]; then
  echo "[FAIL] Expected nsys report was not generated: ${NSYS_REPORT_BASE}.nsys-rep" >&2
  exit 1
fi
nsys export -t sqlite --force-overwrite true -o "${NSYS_DIR}/all_ranks_scaling.sqlite" "${NSYS_REPORT_BASE}.nsys-rep" >/dev/null

for rank_id in 0 1 2 3; do
  rank_trace=$(latest_rank_file "${TRACE_ROOT}" "${rank_id}")
  if [[ -z "${rank_trace}" || ! -f "${rank_trace}" ]]; then
    echo "[FAIL] Rank${rank_id} scaling trace not found under ${TRACE_ROOT}" >&2
    exit 1
  fi
  cp "${rank_trace}" "${TRACE_DIR}/$(basename "${rank_trace}")"
done

if [[ "${PP_SIZE}" -eq 1 ]]; then
  cat > "${SCHEDULE_DIR}/stage0_manual_scheduling_plan.txt" <<'EOF'
stage:0:get_batch(batch_id=0, mg_state=steady, duration=None, description=None, group_kind=None, input__shape=None, input__dtype=None)
stage:0:forward_step(batch_id=0, mg_state=steady, duration=None, description=None, group_kind=None, input__shape=None, input__dtype=None)
stage:0:loss_func(batch_id=0, mg_state=steady, duration=None, description=None, group_kind=None, input__shape=None, input__dtype=None)
stage:0:backward_step(batch_id=0, mg_state=steady, duration=None, description=None, group_kind=None, input__shape=None, input__dtype=None)
stage:0:dp_allreduce(batch_id=0, mg_state=finalize, duration=None, description=model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas, group_kind=dp, input__shape=None, input__dtype=None)
stage:0:optimizer_step(batch_id=0, mg_state=finalize, duration=None, description=None, group_kind=None, input__shape=None, input__dtype=None)
EOF
  printf 'Generated manual no-pipelining schedule for PP=1
' > "${RUN_DIR}/scheduler.log"
else
  pushd "${PROJECT_ROOT}/megatron-sim-engine/src/scheduler/mg_scheduling" >/dev/null
  python mg_test.py     --local-size "${LOCAL_SIZE}"     --world-size "${WORLD_SIZE}"     --micro-batch-size "${MBS}"     --global-batch-size "${GBS}"     --seq-length "${SEQ_LEN}"     --hidden-size "${HIDDEN_SIZE}"     --train-iters "${TRAIN_ITERS}"     --model-size tiny     -pp "${PP_SIZE}"     -tp "${TP_SIZE}"     --trace-start "${TRACE_START}"     > "${RUN_DIR}/scheduler.log" 2>&1
  popd >/dev/null

  SCHED_SRC_DIR="${PROJECT_ROOT}/megatron-sim-engine/simulation_inputs/scheduling_plans/mg_scheduling_plan_log/MODELtiny_pp1_tp1_dp4_exp1_seq256_mbs1_gbs4_fp32"
  SCHED_FILE=$(find "${SCHED_SRC_DIR}" -maxdepth 1 -type f -name 'stage0_*_scheduling_plan.txt' -printf '%T@ %p
' | sort -nr | head -n 1 | cut -d' ' -f2-)
  if [[ -z "${SCHED_FILE}" || ! -f "${SCHED_FILE}" ]]; then
    echo "[FAIL] Generated schedule file not found under ${SCHED_SRC_DIR}" >&2
    exit 1
  fi
  cp "${SCHED_FILE}" "${SCHEDULE_DIR}/$(basename "${SCHED_FILE}")"
fi

python megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py \
  --trace-dir "${TRACE_DIR}" \
  --nsys-sqlite "${NSYS_DIR}/all_ranks_scaling.sqlite" \
  --ncu-metrics-csv "${NCU_METRICS_CSV}" \
  --label-prefix cmd_trace \
  --output-dir "${SLOWDOWN_ASSETS_DIR}" \
  --model-path "${MODEL_PATH}" \
  --scaler-path "${SCALER_PATH}" \
  > "${RUN_DIR}/build_assets.log" 2>&1

python tests/e2e/run_ddp_slowdown_compare.py \
  --trace-dir "${TRACE_DIR}" \
  --database-dir "${TRACE_DIR}" \
  --schedule-dir "${SCHEDULE_DIR}" \
  --slowdown-assets-dir "${SLOWDOWN_ASSETS_DIR}" \
  --slowdown-model-path "${MODEL_PATH}" \
  --slowdown-scaler-path "${SCALER_PATH}" \
  --world-size "${WORLD_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --strategy no-pipelining \
  --wrank-id 0 \
  --output-json "${COMPARE_JSON}" \
  > "${RUN_DIR}/simulate_compare.log" 2>&1

python - <<'PY' "${COMPARE_JSON}" "${SUMMARY_MD}" "${RUN_DIR}"
import json
import sys
from pathlib import Path
compare_json = Path(sys.argv[1])
summary_md = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
summary = json.loads(compare_json.read_text())
text = f"""## E2E Slowdown Smoke Summary

- Run dir: `{run_dir}`
- `backward_step` off: `{summary['backward_duration_ms_off']:.4f} ms`
- `backward_step` on: `{summary['backward_duration_ms_on']:.4f} ms`
- Delta: `{summary['backward_duration_delta_ms']:.4f} ms`
- Shared comm uids: `{len(summary['shared_comm_uids'])}`
- Delayed comm uids: `{len(summary['delayed_comm_uids'])}`
- Processed backward cmd_uids: `{summary['slowdown_on']['slowdown_processed_backward_cmd_uids']}`
"""
summary_md.write_text(text, encoding='utf-8')
print(text)
PY

echo "[PASS] DDP slowdown simulate smoke passed."
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Summary: ${SUMMARY_MD}"
