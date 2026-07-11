#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

for tool_name in torchrun nsys ncu python; do
  if ! command -v "${tool_name}" >/dev/null 2>&1; then
    echo "[FAIL] ${tool_name} not found in PATH." >&2
    exit 1
  fi
done

WORLD_SIZE=4
LOCAL_SIZE=4
PP_SIZE=2
TP_SIZE=1
EXP_SIZE=1
FAKE_DP=2
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=16384
NUM_HEADS=32
SEQ_LEN=256
MBS=1
GBS=8
TRACE_TRAIN_ITERS=3
TRACE_START=2
NCU_TRAIN_ITERS=1
DDP_BUCKET_SIZE="${GPT67B_SLOWDOWN_DDP_BUCKET_SIZE:-10000000}"
DRYRUN_GPU="${GPT67B_SLOWDOWN_DRYRUN_GPU:-7}"
NSYS_GPU="${GPT67B_SLOWDOWN_NSYS_GPU:-4}"
NCU_STAGE0_GPU="${GPT67B_SLOWDOWN_NCU_STAGE0_GPU:-5}"
NCU_STAGE1_GPU="${GPT67B_SLOWDOWN_NCU_STAGE1_GPU:-6}"
SIM_GPU="${GPT67B_SLOWDOWN_SIM_GPU:-7}"
TRACE_ROOT="${PROJECT_ROOT}/profiler_log/pp2_tp1_ep1_expnNone_dp2_nl32_hs4096_sl256"
MODEL_PATH="${PROJECT_ROOT}/Echo-slowdown/training_testing/output/xgb_model.json"
SCALER_PATH="${GPT67B_SLOWDOWN_SCALER_PATH:-${PROJECT_ROOT}/Echo-slowdown/training_testing/output/standard_scaler.json}"
VOCAB_FILE="${PROJECT_ROOT}/data/output_prefix_gpt2/gpt2-vocab.json"
MERGE_FILE="${PROJECT_ROOT}/data/output_prefix_gpt2/gpt2-merges.txt"
DATA_PATH="${PROJECT_ROOT}/data/output_prefix_gpt2/my-gpt2_text_document"
RUN_TAG="gpt67b_ddp_slowdown_e2e_bucket${DDP_BUCKET_SIZE}_$(date +%Y%m%d_%H%M%S)_$$"
ARTIFACT_ROOT="${PROJECT_ROOT}/tests/e2e/artifacts"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_TAG}"
CASE_DIR="${PROJECT_ROOT}/megatron-sim-engine/simulation_inputs/megatron_operation_log/${RUN_TAG}"
GLOBAL_TRACE_DIR="${CASE_DIR}/global_ranks_profile"
DATABASE_DIR="${CASE_DIR}/database_profile"
SCHEDULE_DIR="${CASE_DIR}/schedule"
SLOWDOWN_ASSETS_DIR="${CASE_DIR}/slowdown_assets"
NSYS_DIR="${RUN_DIR}/nsys"
NCU_STAGE0_DIR="${RUN_DIR}/ncu_stage0_rank0"
NCU_STAGE1_DIR="${RUN_DIR}/ncu_stage1_rank2"
COMPARE_DIR="${RUN_DIR}/compare"
SUMMARY_MD="${RUN_DIR}/summary.md"
SUMMARY_JSON="${RUN_DIR}/summary.json"
SCHED_MODEL_SIZE="gpt67b_dense"
SCHED_SRC_DIR="${PROJECT_ROOT}/megatron-sim-engine/simulation_inputs/scheduling_plans/mg_scheduling_plan_log/MODEL${SCHED_MODEL_SIZE}_pp${PP_SIZE}_tp${TP_SIZE}_dp${FAKE_DP}_exp${EXP_SIZE}_seq${SEQ_LEN}_mbs${MBS}_gbs${GBS}_fp16"

mkdir -p "${ARTIFACT_ROOT}" "${RUN_DIR}" "${CASE_DIR}" "${GLOBAL_TRACE_DIR}" "${DATABASE_DIR}" \
  "${SCHEDULE_DIR}" "${SLOWDOWN_ASSETS_DIR}" "${NSYS_DIR}" "${NCU_STAGE0_DIR}/temp" \
  "${NCU_STAGE0_DIR}/output" "${NCU_STAGE1_DIR}/temp" "${NCU_STAGE1_DIR}/output" "${COMPARE_DIR}"

for required_file in "${MODEL_PATH}" "${SCALER_PATH}" "${VOCAB_FILE}" "${MERGE_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[FAIL] Required file not found: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -e "${DATA_PATH}.bin" || ! -e "${DATA_PATH}.idx" ]]; then
  echo "[FAIL] Expected GPT data prefix missing: ${DATA_PATH}.bin / ${DATA_PATH}.idx" >&2
  exit 1
fi

latest_rank_file() {
  local dir=$1
  local rank_id=$2
  find "${dir}" -maxdepth 1 -type f -name "*rank${rank_id}_*.txt" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

build_torchrun_cmd() {
  local fake_rank_id=$1
  local train_iters=$2
  local trace_mode=$3
  local -a cmd=(
    torchrun
    --nproc_per_node=1
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    --master_port="$((29900 + fake_rank_id))"
    pretrain_llama.py
    --distributed-backend nccl
    --use-mcore-models
    --transformer-impl local
    --data-path "${DATA_PATH}"
    --split 949,50,1
    --tokenizer-type GPT2BPETokenizer
    --vocab-file "${VOCAB_FILE}"
    --merge-file "${MERGE_FILE}"
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --num-attention-heads "${NUM_HEADS}"
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --train-iters "${train_iters}"
    --lr 1.5e-4
    --min-lr 1e-5
    --lr-decay-style cosine
    --lr-decay-iters "${train_iters}"
    --weight-decay 1e-2
    --clip-grad 1.0
    --log-interval 1
    --eval-interval 1000
    --eval-iters 1
    --fp16
    --overlap-grad-reduce
    --ddp-bucket-size "${DDP_BUCKET_SIZE}"
    --is-scaling-mode
    --fake-world-size "${WORLD_SIZE}"
    --fake-wrank 0
    --fake-gpus-per-node "${LOCAL_SIZE}"
    --fake-local-rank 0
    --fake-pp "${PP_SIZE}"
    --fake-dp "${FAKE_DP}"
    --fake-tp "${TP_SIZE}"
    --fake-exp "${EXP_SIZE}"
    --fake-current-rank-id "${fake_rank_id}"
  )
  if [[ "${trace_mode}" == "traced" ]]; then
    cmd+=(
      --do-trace True
      --trace-start "${TRACE_START}"
      --trace-subop-sync-mode global
      --trace-kernel-ground-truth
      --trace-kernel-ground-truth-prefix cmd_trace
      --trace-kernel-ground-truth-phase
      --trace-kernel-boundary-sync-mode global
      --trace-ddp-grad-overlap
    )
  fi
  printf '%q ' "${cmd[@]}"
  printf '\n'
}

count_buckets_in_log() {
  local log_file=$1
  python - <<'PY' "${log_file}"
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(errors='ignore')
print(sum(1 for line in text.splitlines() if 'Params for bucket ' in line))
PY
}

run_dryrun_rank() {
  local fake_rank_id=$1
  local gpu_id=$2
  local log_file=$3
  CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "cd ${PROJECT_ROOT@Q} && export PYTHONPATH=${PYTHONPATH@Q} && export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS@Q} && $(build_torchrun_cmd "${fake_rank_id}" 1 untraced)" > "${log_file}" 2>&1
}

run_ncu_rank() {
  local fake_rank_id=$1
  local gpu_id=$2
  local out_dir=$3
  local log_file=$4
  local report_base="${out_dir}/temp/output"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    ncu -o "${report_base}" \
      -f \
      --replay-mode application \
      --target-processes all \
      --device 0 \
      --section SpeedOfLight \
      --section LaunchStats \
      --section Occupancy \
      --section MemoryWorkloadAnalysis \
      bash -lc "cd ${PROJECT_ROOT@Q} && export PYTHONPATH=${PYTHONPATH@Q} && export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS@Q} && $(build_torchrun_cmd "${fake_rank_id}" "${NCU_TRAIN_ITERS}" untraced)" \
      > "${log_file}" 2>&1
  if [[ ! -f "${report_base}.ncu-rep" ]]; then
    echo "[FAIL] Missing NCU report: ${report_base}.ncu-rep" >&2
    exit 1
  fi
  ncu -i "${report_base}.ncu-rep" --page details --csv --log-file "${report_base}_details.csv" >/dev/null
  ncu -i "${report_base}.ncu-rep" --print-kernel-base function --csv > "${report_base}_kshortname.csv"
  (
    cd "${out_dir}"
    python "${PROJECT_ROOT}/Echo-slowdown/kernel_metric/ncu_report_process.py"
  ) >/dev/null
  if [[ ! -f "${out_dir}/output/kernel_metric_output.csv" ]]; then
    echo "[FAIL] Missing processed NCU metrics: ${out_dir}/output/kernel_metric_output.csv" >&2
    exit 1
  fi
}

echo "[INFO] Step 1/7: GPU${DRYRUN_GPU} dry-run with smaller DDP bucket size ${DDP_BUCKET_SIZE}."
DRYRUN_RANK0_LOG="${RUN_DIR}/dryrun_rank0.log"
DRYRUN_RANK2_LOG="${RUN_DIR}/dryrun_rank2.log"
run_dryrun_rank 0 "${DRYRUN_GPU}" "${DRYRUN_RANK0_LOG}"
run_dryrun_rank 2 "${DRYRUN_GPU}" "${DRYRUN_RANK2_LOG}"
DRYRUN_BUCKETS_RANK0=$(count_buckets_in_log "${DRYRUN_RANK0_LOG}")
DRYRUN_BUCKETS_RANK2=$(count_buckets_in_log "${DRYRUN_RANK2_LOG}")

cat > "${RUN_DIR}/run_all_scaling_ranks.sh" <<RUNALL
#!/usr/bin/env bash
set -euo pipefail
cd ${PROJECT_ROOT@Q}
export CUDA_VISIBLE_DEVICES=${NSYS_GPU@Q}
export PYTHONPATH=${PYTHONPATH@Q}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS@Q}
$(for fake_rank_id in 0 1 2 3; do build_torchrun_cmd "${fake_rank_id}" "${TRACE_TRAIN_ITERS}" traced; done)
RUNALL
chmod +x "${RUN_DIR}/run_all_scaling_ranks.sh"

echo "[INFO] Step 2/7: GPU${NSYS_GPU} sequential scaling profiling under nsys."
NSYS_REPORT_BASE="${NSYS_DIR}/all_ranks_scaling"
set +e
CUDA_VISIBLE_DEVICES="${NSYS_GPU}" \
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --wait=all \
    --trace-fork-before-exec=true \
    --force-overwrite true \
    --output "${NSYS_REPORT_BASE}" \
    bash "${RUN_DIR}/run_all_scaling_ranks.sh" \
    > "${RUN_DIR}/all_ranks_nsys.log" 2>&1
NSYS_EXIT_CODE=$?
set -e
if [[ ! -f "${NSYS_REPORT_BASE}.nsys-rep" ]]; then
  echo "[FAIL] Expected nsys report missing: ${NSYS_REPORT_BASE}.nsys-rep (nsys exit=${NSYS_EXIT_CODE})" >&2
  exit 1
fi
if [[ "${NSYS_EXIT_CODE}" -ne 0 && "${NSYS_EXIT_CODE}" -ne 143 ]]; then
  echo "[FAIL] nsys exited with ${NSYS_EXIT_CODE} and report validation did not permit it." >&2
  exit 1
fi
nsys export -t sqlite --force-overwrite true -o "${NSYS_DIR}/all_ranks_scaling.sqlite" "${NSYS_REPORT_BASE}.nsys-rep" >/dev/null

echo "[INFO] Step 3/7: Copy latest traced ranks into simulation input case directory."
for rank_id in 0 1 2 3; do
  rank_trace=$(latest_rank_file "${TRACE_ROOT}" "${rank_id}")
  if [[ -z "${rank_trace}" || ! -f "${rank_trace}" ]]; then
    echo "[FAIL] Rank${rank_id} trace not found under ${TRACE_ROOT}" >&2
    exit 1
  fi
  cp "${rank_trace}" "${GLOBAL_TRACE_DIR}/$(basename "${rank_trace}")"
done
for rank_id in 0 2; do
  rank_trace=$(latest_rank_file "${GLOBAL_TRACE_DIR}" "${rank_id}")
  cp "${rank_trace}" "${DATABASE_DIR}/$(basename "${rank_trace}")"
done

echo "[INFO] Step 4/7: GPU${NCU_STAGE0_GPU} and GPU${NCU_STAGE1_GPU} collect NCU metrics for representative stages."
run_ncu_rank 0 "${NCU_STAGE0_GPU}" "${NCU_STAGE0_DIR}" "${RUN_DIR}/ncu_rank0.log" &
pid_stage0=$!
run_ncu_rank 2 "${NCU_STAGE1_GPU}" "${NCU_STAGE1_DIR}" "${RUN_DIR}/ncu_rank2.log" &
pid_stage1=$!
wait "${pid_stage0}"
wait "${pid_stage1}"
MERGED_NCU_CSV="${RUN_DIR}/kernel_metric_output_rank0_rank2_merged.csv"
python - <<'PY' \
  "${NCU_STAGE0_DIR}/output/kernel_metric_output.csv" \
  "${NCU_STAGE1_DIR}/output/kernel_metric_output.csv" \
  "${MERGED_NCU_CSV}"
import pandas as pd
import sys
rank0_csv, rank2_csv, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.concat([pd.read_csv(rank0_csv), pd.read_csv(rank2_csv)], ignore_index=True)
df.to_csv(out_csv, index=False)
print(f"merged_rows={len(df)} unique_kernels={df['Kernel Name'].nunique()} out={out_csv}")
PY

echo "[INFO] Step 5/7: Generate pp2 schedule and build slowdown assets."
pushd "${PROJECT_ROOT}/megatron-sim-engine/src/scheduler/mg_scheduling" >/dev/null
python mg_test.py \
  --local-size "${LOCAL_SIZE}" \
  --world-size "${WORLD_SIZE}" \
  --micro-batch-size "${MBS}" \
  --global-batch-size "${GBS}" \
  --seq-length "${SEQ_LEN}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --train-iters "${TRACE_TRAIN_ITERS}" \
  --trace-start "${TRACE_START}" \
  --model-size "${SCHED_MODEL_SIZE}" \
  --fp16 \
  -pp "${PP_SIZE}" \
  -tp "${TP_SIZE}" \
  > "${RUN_DIR}/scheduler.log" 2>&1
popd >/dev/null
for stage_id in 0 1; do
  sched_file=$(find "${SCHED_SRC_DIR}" -maxdepth 1 -type f -name "stage${stage_id}_*_scheduling_plan.txt" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  if [[ -z "${sched_file}" || ! -f "${sched_file}" ]]; then
    echo "[FAIL] Stage${stage_id} schedule not found under ${SCHED_SRC_DIR}" >&2
    exit 1
  fi
  cp "${sched_file}" "${SCHEDULE_DIR}/$(basename "${sched_file}")"
done
python megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py \
  --trace-dir "${GLOBAL_TRACE_DIR}" \
  --nsys-sqlite "${NSYS_DIR}/all_ranks_scaling.sqlite" \
  --ncu-metrics-csv "${MERGED_NCU_CSV}" \
  --label-prefix cmd_trace \
  --output-dir "${SLOWDOWN_ASSETS_DIR}" \
  --model-path "${MODEL_PATH}" \
  --scaler-path "${SCALER_PATH}" \
  > "${RUN_DIR}/build_assets.log" 2>&1

echo "[INFO] Step 6/7: Run slowdown off/on compare for wrank0 and wrank2."
CUDA_VISIBLE_DEVICES="${SIM_GPU}" python tests/e2e/run_ddp_slowdown_compare.py \
  --trace-dir "${GLOBAL_TRACE_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --schedule-dir "${SCHEDULE_DIR}" \
  --slowdown-assets-dir "${SLOWDOWN_ASSETS_DIR}" \
  --slowdown-model-path "${MODEL_PATH}" \
  --slowdown-scaler-path "${SCALER_PATH}" \
  --world-size "${WORLD_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --strategy 1F1B-none_interleaved \
  --wrank-id 0 \
  --output-json "${COMPARE_DIR}/wrank0.json" \
  > "${COMPARE_DIR}/wrank0.log" 2>&1
CUDA_VISIBLE_DEVICES="${SIM_GPU}" python tests/e2e/run_ddp_slowdown_compare.py \
  --trace-dir "${GLOBAL_TRACE_DIR}" \
  --database-dir "${DATABASE_DIR}" \
  --schedule-dir "${SCHEDULE_DIR}" \
  --slowdown-assets-dir "${SLOWDOWN_ASSETS_DIR}" \
  --slowdown-model-path "${MODEL_PATH}" \
  --slowdown-scaler-path "${SCALER_PATH}" \
  --world-size "${WORLD_SIZE}" \
  --local-size "${LOCAL_SIZE}" \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --exp-size "${EXP_SIZE}" \
  --strategy 1F1B-none_interleaved \
  --wrank-id 2 \
  --output-json "${COMPARE_DIR}/wrank2.json" \
  > "${COMPARE_DIR}/wrank2.log" 2>&1

echo "[INFO] Step 7/7: Summarize artifacts, overlap growth, and slowdown evidence."
python - <<'PY' \
  "${GLOBAL_TRACE_DIR}" \
  "${SLOWDOWN_ASSETS_DIR}" \
  "${COMPARE_DIR}/wrank0.json" \
  "${COMPARE_DIR}/wrank2.json" \
  "${SUMMARY_MD}" \
  "${SUMMARY_JSON}" \
  "${RUN_DIR}" \
  "${CASE_DIR}" \
  "${DDP_BUCKET_SIZE}" \
  "${DRYRUN_BUCKETS_RANK0}" \
  "${DRYRUN_BUCKETS_RANK2}" \
  "${NSYS_GPU}" \
  "${NCU_STAGE0_GPU}" \
  "${NCU_STAGE1_GPU}" \
  "${DRYRUN_GPU}" \
  "${SIM_GPU}"
import json
from pathlib import Path
import sys

global_trace_dir = Path(sys.argv[1])
assets_dir = Path(sys.argv[2])
wrank0_json = Path(sys.argv[3])
wrank2_json = Path(sys.argv[4])
summary_md = Path(sys.argv[5])
summary_json = Path(sys.argv[6])
run_dir = Path(sys.argv[7])
case_dir = Path(sys.argv[8])
ddp_bucket_size = int(sys.argv[9])
dryrun_buckets_rank0 = int(sys.argv[10])
dryrun_buckets_rank2 = int(sys.argv[11])
nsys_gpu = sys.argv[12]
ncu_stage0_gpu = sys.argv[13]
ncu_stage1_gpu = sys.argv[14]
dryrun_gpu = sys.argv[15]
sim_gpu = sys.argv[16]

def count_ddp(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if 'ddp_grad_comm' in line)

def load_json(path: Path):
    return json.loads(path.read_text())

trace_counts = {}
for path in sorted(global_trace_dir.glob('*.txt')):
    trace_counts[path.name] = {'ddp_grad_comm_count': count_ddp(path)}
manifest = load_json(assets_dir / 'manifest.json')
kernel_features = load_json(assets_dir / 'kernel_features.json')
blueprints = load_json(assets_dir / 'backward_kernel_blueprints.json')
wrank0 = load_json(wrank0_json)
wrank2 = load_json(wrank2_json)
summary = {
    'run_dir': str(run_dir),
    'case_dir': str(case_dir),
    'gpus': {
        'dryrun_gpu': dryrun_gpu,
        'nsys_gpu': nsys_gpu,
        'ncu_stage0_gpu': ncu_stage0_gpu,
        'ncu_stage1_gpu': ncu_stage1_gpu,
        'sim_gpu': sim_gpu,
    },
    'ddp_bucket_size': ddp_bucket_size,
    'baseline_default_bucket_trace_ddp_counts': {'rank0': 49, 'rank2': 50},
    'dryrun_bucket_counts': {'rank0': dryrun_buckets_rank0, 'rank2': dryrun_buckets_rank2},
    'trace_counts': trace_counts,
    'slowdown_assets': {
        'scope': manifest.get('scope'),
        'kernel_feature_count': len(kernel_features),
        'backward_blueprint_count': len(blueprints),
        'model_path': manifest.get('model_path'),
    },
    'wrank0': wrank0,
    'wrank2': wrank2,
}
summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
md = f"""## GPT-6.7B DDP Slowdown E2E Summary

- Run dir: `{run_dir}`
- Case dir: `{case_dir}`
- GPUs: dry-run=`{dryrun_gpu}`, nsys=`{nsys_gpu}`, ncu(stage0/rank0)=`{ncu_stage0_gpu}`, ncu(stage1/rank2)=`{ncu_stage1_gpu}`, simulate=`{sim_gpu}`
- `--ddp-bucket-size`: `{ddp_bucket_size}`
- Baseline default trace DDP counts: rank0=`49`, rank2=`50`
- Dry-run bucket counts with smaller bucket size: rank0=`{dryrun_buckets_rank0}`, rank2=`{dryrun_buckets_rank2}`
- Built slowdown assets: kernel_features=`{len(kernel_features)}`, backward_blueprints=`{len(blueprints)}`

### wrank0
- `backward_step` off/on: `{wrank0['backward_duration_ms_off']:.6f} -> {wrank0['backward_duration_ms_on']:.6f} ms`
- Delta: `{wrank0['backward_duration_delta_ms']:.6f} ms`
- Shared DDP comm uids: `{len(wrank0['shared_comm_uids'])}`
- Delayed DDP comm uids: `{len(wrank0['delayed_comm_uids'])}`
- Processed backward cmd_uids: `{wrank0['slowdown_on']['slowdown_processed_backward_cmd_uids']}`

### wrank2
- `backward_step` off/on: `{wrank2['backward_duration_ms_off']:.6f} -> {wrank2['backward_duration_ms_on']:.6f} ms`
- Delta: `{wrank2['backward_duration_delta_ms']:.6f} ms`
- Shared DDP comm uids: `{len(wrank2['shared_comm_uids'])}`
- Delayed DDP comm uids: `{len(wrank2['delayed_comm_uids'])}`
- Processed backward cmd_uids: `{wrank2['slowdown_on']['slowdown_processed_backward_cmd_uids']}`
"""
summary_md.write_text(md, encoding='utf-8')
print(md)
PY

echo "[PASS] GPT-6.7B DDP slowdown E2E completed."
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Case dir: ${CASE_DIR}"
echo "[INFO] Summary: ${SUMMARY_MD}"
