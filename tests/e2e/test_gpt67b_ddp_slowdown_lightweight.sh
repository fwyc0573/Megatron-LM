#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/megatron-sim-engine:${PYTHONPATH:-}"
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
DDP_BUCKET_SIZE="${GPT67B_LIGHT_DDP_BUCKET_SIZE:-10000000}"
DRYRUN_GPU="${GPT67B_LIGHT_DRYRUN_GPU:-7}"
NSYS_GPU="${GPT67B_LIGHT_NSYS_GPU:-4}"
NCU_STAGE0_GPU="${GPT67B_LIGHT_NCU_STAGE0_GPU:-5}"
NCU_STAGE1_GPU="${GPT67B_LIGHT_NCU_STAGE1_GPU:-6}"
REAL_REFERENCE_GPUS="${GPT67B_LIGHT_REAL_REFERENCE_GPUS:-4,5,6,7}"
SIM_GPU="${GPT67B_LIGHT_SIM_GPU:-7}"
MODEL_PATH="${GPT67B_LIGHT_MODEL_PATH:-${PROJECT_ROOT}/Echo-slowdown/training_testing/output/xgb_model.json}"
SCALER_PATH="${GPT67B_LIGHT_SCALER_PATH:-${PROJECT_ROOT}/Echo-slowdown/training_testing/output/standard_scaler.json}"
VOCAB_FILE="${PROJECT_ROOT}/data/output_prefix_gpt2/gpt2-vocab.json"
MERGE_FILE="${PROJECT_ROOT}/data/output_prefix_gpt2/gpt2-merges.txt"
DATA_PATH="${PROJECT_ROOT}/data/output_prefix_gpt2/my-gpt2_text_document"
SCALING_TRACE_ROOT="${PROJECT_ROOT}/profiler_log/pp2_tp1_ep1_expnNone_dp2_nl32_hs4096_sl256"
REAL_TRACE_ROOT="${PROJECT_ROOT}/realistic_trace/pp2_tp1_exp1_expnNone_dp2_nl32_hs4096_sl256"
RUN_TAG="gpt67b_ddp_slowdown_lightweight_bucket${DDP_BUCKET_SIZE}_$(date +%Y%m%d_%H%M%S)_$$"
ARTIFACT_ROOT="${PROJECT_ROOT}/tests/e2e/artifacts"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_TAG}"
CASE_DIR="${PROJECT_ROOT}/megatron-sim-engine/simulation_inputs/megatron_operation_log/${RUN_TAG}"
GLOBAL_TRACE_DIR="${CASE_DIR}/global_ranks_profile"
DATABASE_DIR="${CASE_DIR}/database_profile"
SCHEDULE_DIR="${CASE_DIR}/trace_shaped_schedule"
SLOWDOWN_ASSETS_DIR="${CASE_DIR}/slowdown_assets"
NSYS_DIR="${RUN_DIR}/nsys"
NCU_PASS1_STAGE0_DIR="${RUN_DIR}/ncu_pass1_stage0_rank0"
NCU_PASS1_STAGE1_DIR="${RUN_DIR}/ncu_pass1_stage1_rank2"
NCU_PASS2_STAGE0_DIR="${RUN_DIR}/ncu_pass2_stage0_rank0"
NCU_PASS2_STAGE1_DIR="${RUN_DIR}/ncu_pass2_stage1_rank2"
COMPARE_DIR="${RUN_DIR}/compare"
REFERENCE_DIR="${RUN_DIR}/reference"
REFERENCE_TRACE_DIR="${REFERENCE_DIR}/trace"
REFERENCE_NSYS_DIR="${REFERENCE_DIR}/nsys"
SUMMARY_MD="${RUN_DIR}/summary.md"
SUMMARY_JSON="${RUN_DIR}/summary.json"

mkdir -p "${RUN_DIR}" "${CASE_DIR}" "${GLOBAL_TRACE_DIR}" "${DATABASE_DIR}" \
  "${SCHEDULE_DIR}" "${SLOWDOWN_ASSETS_DIR}" "${NSYS_DIR}" \
  "${NCU_PASS1_STAGE0_DIR}/temp" "${NCU_PASS1_STAGE0_DIR}/output" \
  "${NCU_PASS1_STAGE1_DIR}/temp" "${NCU_PASS1_STAGE1_DIR}/output" \
  "${NCU_PASS2_STAGE0_DIR}/temp" "${NCU_PASS2_STAGE0_DIR}/output" \
  "${NCU_PASS2_STAGE1_DIR}/temp" "${NCU_PASS2_STAGE1_DIR}/output" \
  "${COMPARE_DIR}" "${REFERENCE_TRACE_DIR}" "${REFERENCE_NSYS_DIR}"

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
  python - <<'PY2' "${dir}" "${rank_id}"
from pathlib import Path
import sys

trace_dir = Path(sys.argv[1])
rank_id = sys.argv[2]
candidates = sorted(
    trace_dir.glob(f'*rank{rank_id}_*.txt'),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if candidates:
    print(candidates[0])
PY2
}

build_scaling_torchrun_cmd() {
  local fake_rank_id=$1
  local train_iters=$2
  local trace_mode=$3
  local -a cmd=(
    torchrun
    --nproc_per_node=1
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    --master_port="$((30700 + fake_rank_id))"
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

build_distributed_torchrun_cmd() {
  local train_iters=$1
  local trace_mode=$2
  local -a cmd=(
    torchrun
    --nproc_per_node="${WORLD_SIZE}"
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    --master_port=30990
    pretrain_llama.py
    --distributed-backend nccl
    --use-mcore-models
    --transformer-impl local
    --data-path "${DATA_PATH}"
    --split 949,50,1
    --tokenizer-type GPT2BPETokenizer
    --vocab-file "${VOCAB_FILE}"
    --merge-file "${MERGE_FILE}"
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
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
  CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "cd ${PROJECT_ROOT@Q} && export PYTHONPATH=${PYTHONPATH@Q} && export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS@Q} && $(build_scaling_torchrun_cmd "${fake_rank_id}" 1 untraced)" > "${log_file}" 2>&1
}

run_targeted_ncu_rank() {
  local fake_rank_id=$1
  local gpu_id=$2
  local kernel_list_path=$3
  local out_dir=$4
  local log_file=$5
  mkdir -p "${out_dir}/temp" "${out_dir}/output"
  : > "${log_file}"
  local kernel_index=0
  while IFS= read -r kernel_name || [[ -n "${kernel_name}" ]]; do
    if [[ -z "${kernel_name}" ]]; then
      continue
    fi
    local report_base="${out_dir}/temp/output_${kernel_index}"
    echo "[INFO] kernel_index=${kernel_index} kernel_name=${kernel_name}" >> "${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}"       ncu -o "${report_base}"         -f         --replay-mode application         --app-replay-mode relaxed         --target-processes all         --device 0         --kernel-name-base function         -k "${kernel_name}"         -c 1         --kill yes         --section SpeedOfLight         --section Occupancy         --section MemoryWorkloadAnalysis         bash -lc "cd ${PROJECT_ROOT@Q} && export PYTHONPATH=${PYTHONPATH@Q} && export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS@Q} && $(build_scaling_torchrun_cmd "${fake_rank_id}" "${NCU_TRAIN_ITERS}" untraced)"         >> "${log_file}" 2>&1
    if [[ ! -f "${report_base}.ncu-rep" ]]; then
      echo "[FAIL] Missing NCU report: ${report_base}.ncu-rep" >&2
      exit 1
    fi
    ncu -i "${report_base}.ncu-rep" --page details --csv --log-file "${report_base}_details.csv" >/dev/null
    ncu -i "${report_base}.ncu-rep" --print-kernel-base function --csv > "${report_base}_kshortname.csv"
    kernel_index=$((kernel_index + 1))
  done < "${kernel_list_path}"
  if [[ "${kernel_index}" -eq 0 ]]; then
    echo "[FAIL] No kernel names found in ${kernel_list_path}" >&2
    exit 1
  fi
  python - <<'PY2' "${out_dir}"
import pandas as pd
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])
temp_dir = out_dir / 'temp'
output_dir = out_dir / 'output'
output_dir.mkdir(parents=True, exist_ok=True)
frames = []
for details_csv in sorted(temp_dir.glob('*_details.csv')):
    prefix = details_csv.name[:-len('_details.csv')]
    kshortname_csv = temp_dir / f'{prefix}_kshortname.csv'
    if not kshortname_csv.exists():
        raise SystemExit(f'Missing companion kshortname csv for {details_csv}')
    details_df = pd.read_csv(details_csv)
    kshortname_df = pd.read_csv(kshortname_csv)
    unique_ids = details_df['ID'].unique()
    output_rows = []
    for uid in unique_ids:
        df_kernel = details_df[details_df['ID'] == uid]
        def get_metric_value(section_name, metric_name):
            filtered_df = df_kernel[(df_kernel['Section Name'] == section_name) & (df_kernel['Metric Name'] == metric_name)].reset_index(drop=True)
            if not filtered_df.empty:
                return filtered_df.at[0, 'Metric Value']
            return None
        output_rows.append({
            'ID': df_kernel.reset_index(drop=True).at[0, 'ID'],
            'Compute throughput': get_metric_value('GPU Speed Of Light Throughput', 'Compute (SM) Throughput'),
            'SM': get_metric_value('Launch Statistics', '# SMs'),
            'Memory throughput': get_metric_value('GPU Speed Of Light Throughput', 'Memory Throughput'),
            'DRAM throughput': get_metric_value('GPU Speed Of Light Throughput', 'DRAM Throughput'),
            'Achieved occupancy': get_metric_value('Occupancy', 'Achieved Occupancy'),
            'Maximum occupancy': get_metric_value('Occupancy', 'Theoretical Occupancy'),
            'L1 hit rate': get_metric_value('Memory Workload Analysis', 'L1/TEX Hit Rate'),
            'L2 hit rate': get_metric_value('Memory Workload Analysis', 'L2 Hit Rate'),
        })
    df_output = pd.DataFrame(output_rows)
    kshortname_df = kshortname_df.drop_duplicates(subset='ID', keep='first')
    merged_df = pd.merge(kshortname_df[['ID', 'Kernel Name']], df_output, on='ID', how='inner')
    frames.append(merged_df)
if not frames:
    raise SystemExit(f'No NCU csv fragments generated under {temp_dir}')
merged_all = pd.concat(frames, ignore_index=True)
merged_all = merged_all.drop_duplicates(subset='Kernel Name', keep='first')
merged_all.to_csv(output_dir / 'kernel_metric_output.csv', index=False)
PY2
  if [[ ! -f "${out_dir}/output/kernel_metric_output.csv" ]]; then
    echo "[FAIL] Missing processed NCU metrics: ${out_dir}/output/kernel_metric_output.csv" >&2
    exit 1
  fi
}

write_required_kernels_report() {
  local trace_dir=$1
  local sqlite_path=$2
  local required_json=$3
  local report_json=$4
  python - <<'PY2' "${trace_dir}" "${sqlite_path}" "${required_json}" "${report_json}"
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path('megatron-sim-engine').resolve()))
from tools.data_prep.slowdown.prepare_case_kernel_metrics import _collect_required_kernels_by_rank
trace_dir = Path(sys.argv[1])
sqlite_path = Path(sys.argv[2])
required_json = Path(sys.argv[3])
report_json = Path(sys.argv[4])
required_by_rank = _collect_required_kernels_by_rank(
    trace_dir=trace_dir,
    nsys_sqlite=sqlite_path,
    label_prefix='cmd_trace',
)
required = sorted({name for names in required_by_rank.values() for name in names})
required_json.write_text(json.dumps(required, indent=2), encoding='utf-8')
report_json.write_text(
    json.dumps(
        {
            'required_kernels': required,
            'required_kernels_by_rank': required_by_rank,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding='utf-8',
)
print(f'required_kernel_count={len(required)}')
for rank, names in required_by_rank.items():
    print(f'rank={rank} kernel_count={len(names)}')
    for name in names:
        print(f'  {name}')
PY2
}

validate_required_kernel_representatives() {
  local report_json=$1
  python - <<'PY2' "${report_json}"
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
required = report['required_kernels_by_rank']
for rank in ('0', '1', '2', '3'):
    if rank not in required:
        raise SystemExit(f'Missing required_kernels_by_rank entry for rank {rank}')
if required['0'] != required['1']:
    raise SystemExit('Representative-rank assumption failed: rank0 kernels differ from rank1 kernels')
if required['2'] != required['3']:
    raise SystemExit('Representative-rank assumption failed: rank2 kernels differ from rank3 kernels')
print('representative_rank_validation=passed')
print(f"rank0_rank1_kernel_count={len(required['0'])}")
print(f"rank2_rank3_kernel_count={len(required['2'])}")
PY2
}

write_rank_kernel_regex() {
  local report_json=$1
  local field_name=$2
  local rank_id=$3
  local output_path=$4
  python - <<'PY2' "${report_json}" "${field_name}" "${rank_id}" "${output_path}"
import json
import re
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
field_name = sys.argv[2]
rank_id = sys.argv[3]
output_path = Path(sys.argv[4])
values = report.get(field_name)
if not isinstance(values, dict):
    raise SystemExit(f'Missing dict field {field_name!r} in {sys.argv[1]}')
names = values.get(rank_id)
if names is None:
    raise SystemExit(f'Missing {field_name}[{rank_id!r}] in {sys.argv[1]}')
if not names:
    output_path.write_text('', encoding='utf-8')
    print('')
    raise SystemExit(0)
regex = 'regex:^(' + '|'.join(re.escape(name) for name in names) + ')$'
output_path.write_text(regex, encoding='utf-8')
print(regex)
PY2
}

write_rank_kernel_list() {
  local report_json=$1
  local field_name=$2
  local rank_id=$3
  local output_path=$4
  python - <<'PY2' "${report_json}" "${field_name}" "${rank_id}" "${output_path}"
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
field_name = sys.argv[2]
rank_id = sys.argv[3]
output_path = Path(sys.argv[4])
values = report.get(field_name)
if not isinstance(values, dict):
    raise SystemExit(f'Missing dict field {field_name!r} in {sys.argv[1]}')
names = values.get(rank_id)
if names is None:
    raise SystemExit(f'Missing {field_name}[{rank_id!r}] in {sys.argv[1]}')
output_path.write_text(''.join(f'{name}\n' for name in names), encoding='utf-8')
print(f'kernel_list_count={len(names)}')
for name in names:
    print(name)
PY2
}

count_report_list_items() {
  local report_json=$1
  local field_name=$2
  local rank_id=$3
  python - <<'PY2' "${report_json}" "${field_name}" "${rank_id}"
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
field_name = sys.argv[2]
rank_id = sys.argv[3]
values = report.get(field_name)
if not isinstance(values, dict):
    raise SystemExit(f'Missing dict field {field_name!r} in {sys.argv[1]}')
items = values.get(rank_id)
if items is None:
    raise SystemExit(f'Missing {field_name}[{rank_id!r}] in {sys.argv[1]}')
print(len(items))
PY2
}

count_report_total_items() {
  local report_json=$1
  local field_name=$2
  python - <<'PY2' "${report_json}" "${field_name}"
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
items = report.get(sys.argv[2])
if not isinstance(items, list):
    raise SystemExit(f'Missing list field {sys.argv[2]!r} in {sys.argv[1]}')
print(len(items))
PY2
}

prepare_case_kernel_metrics_report() {
  local output_csv=$1
  local report_json=$2
  shift 2
  local candidate_csvs=("$@")
  local cmd=(
    python megatron-sim-engine/tools/data_prep/slowdown/prepare_case_kernel_metrics.py
    --trace-dir "${GLOBAL_TRACE_DIR}"
    --nsys-sqlite "${NSYS_DIR}/all_ranks_scaling.sqlite"
    --label-prefix cmd_trace
    --output-csv "${output_csv}"
    --report-json "${report_json}"
    --alias ln_bwd_general_kernel=ln_bwd_tuned_kernel
    --alias ln_bwd_finalize_general_kernel=ln_bwd_finalize_tuned_kernel
  )
  for csv_path in "${candidate_csvs[@]}"; do
    cmd+=(--candidate-csv "${csv_path}")
  done
  "${cmd[@]}"
}

echo "[INFO] Step 1/9: dry-run on GPU${DRYRUN_GPU} to validate smaller DDP bucket size ${DDP_BUCKET_SIZE}."
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
$(for fake_rank_id in 0 1 2 3; do build_scaling_torchrun_cmd "${fake_rank_id}" "${TRACE_TRAIN_ITERS}" traced; done)
RUNALL
chmod +x "${RUN_DIR}/run_all_scaling_ranks.sh"

echo "[INFO] Step 2/9: collect traced scaling run under nsys on GPU${NSYS_GPU}."
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
  echo "[FAIL] Missing nsys report: ${NSYS_REPORT_BASE}.nsys-rep (exit=${NSYS_EXIT_CODE})" >&2
  exit 1
fi
if [[ "${NSYS_EXIT_CODE}" -ne 0 && "${NSYS_EXIT_CODE}" -ne 143 ]]; then
  echo "[FAIL] nsys exited with unexpected code ${NSYS_EXIT_CODE}." >&2
  exit 1
fi
nsys export -t sqlite --force-overwrite true -o "${NSYS_DIR}/all_ranks_scaling.sqlite" "${NSYS_REPORT_BASE}.nsys-rep" >/dev/null
for rank_id in 0 1 2 3; do
  rank_trace=$(latest_rank_file "${SCALING_TRACE_ROOT}" "${rank_id}")
  if [[ -z "${rank_trace}" || ! -f "${rank_trace}" ]]; then
    echo "[FAIL] Missing scaling trace for rank${rank_id} under ${SCALING_TRACE_ROOT}" >&2
    exit 1
  fi
  cp "${rank_trace}" "${GLOBAL_TRACE_DIR}/$(basename "${rank_trace}")"
done
for rank_id in 0 2; do
  rank_trace=$(latest_rank_file "${GLOBAL_TRACE_DIR}" "${rank_id}")
  cp "${rank_trace}" "${DATABASE_DIR}/$(basename "${rank_trace}")"
done

echo "[INFO] Step 3/9: auto-generate trace-shaped PP schedule from compressed trace."
python megatron-sim-engine/tools/data_prep/schedule/build_trace_shaped_pp_schedule.py \
  --trace-dir "${GLOBAL_TRACE_DIR}" \
  --output-dir "${SCHEDULE_DIR}" \
  --pp-size "${PP_SIZE}" \
  --seq-length "${SEQ_LEN}" \
  --micro-batch-size "${MBS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --pipeline-dtype torch.float16 \
  > "${RUN_DIR}/schedule_builder.log" 2>&1

echo "[INFO] Step 4/9: derive required backward kernel short names from trace + nsys."
REQUIRED_KERNELS_JSON="${RUN_DIR}/required_kernels.json"
REQUIRED_KERNELS_REPORT_JSON="${RUN_DIR}/required_kernels_report.json"
REQUIRED_KERNELS_RANK0_REGEX="${RUN_DIR}/required_kernels_rank0.regex"
REQUIRED_KERNELS_RANK2_REGEX="${RUN_DIR}/required_kernels_rank2.regex"
write_required_kernels_report "${GLOBAL_TRACE_DIR}" "${NSYS_DIR}/all_ranks_scaling.sqlite" "${REQUIRED_KERNELS_JSON}" "${REQUIRED_KERNELS_REPORT_JSON}" > "${RUN_DIR}/required_kernels.log" 2>&1
validate_required_kernel_representatives "${REQUIRED_KERNELS_REPORT_JSON}" > "${RUN_DIR}/required_kernels_by_rank.log" 2>&1
REQUIRED_KERNELS_RANK0_LIST="${RUN_DIR}/required_kernels_rank0.list"
REQUIRED_KERNELS_RANK2_LIST="${RUN_DIR}/required_kernels_rank2.list"
write_rank_kernel_list "${REQUIRED_KERNELS_REPORT_JSON}" required_kernels_by_rank 0 "${REQUIRED_KERNELS_RANK0_LIST}" > "${RUN_DIR}/required_kernels_rank0_list.log" 2>&1
write_rank_kernel_list "${REQUIRED_KERNELS_REPORT_JSON}" required_kernels_by_rank 2 "${REQUIRED_KERNELS_RANK2_LIST}" > "${RUN_DIR}/required_kernels_rank2_list.log" 2>&1

echo "[INFO] Step 5/9: targeted NCU collection on GPU${NCU_STAGE0_GPU} then GPU${NCU_STAGE1_GPU} (sequential replay, one launch per kernel short name)."
echo "[INFO] Step 5/9a: pass-1 representative rank0 on GPU${NCU_STAGE0_GPU}."
run_targeted_ncu_rank 0 "${NCU_STAGE0_GPU}" "${REQUIRED_KERNELS_RANK0_LIST}" "${NCU_PASS1_STAGE0_DIR}" "${RUN_DIR}/ncu_pass1_rank0.log"
echo "[INFO] Step 5/9b: pass-1 representative rank2 on GPU${NCU_STAGE1_GPU}."
run_targeted_ncu_rank 2 "${NCU_STAGE1_GPU}" "${REQUIRED_KERNELS_RANK2_LIST}" "${NCU_PASS1_STAGE1_DIR}" "${RUN_DIR}/ncu_pass1_rank2.log"
NCU_PASS1_STAGE0_CSV="${NCU_PASS1_STAGE0_DIR}/output/kernel_metric_output.csv"
NCU_PASS1_STAGE1_CSV="${NCU_PASS1_STAGE1_DIR}/output/kernel_metric_output.csv"
PASS1_PREPARED_NCU_CSV="${RUN_DIR}/kernel_metric_output_targeted_pass1.csv"
PASS1_KERNEL_REPORT_JSON="${RUN_DIR}/kernel_metrics_report_pass1.json"
prepare_case_kernel_metrics_report "${PASS1_PREPARED_NCU_CSV}" "${PASS1_KERNEL_REPORT_JSON}"   "${NCU_PASS1_STAGE0_CSV}"   "${NCU_PASS1_STAGE1_CSV}"   > "${RUN_DIR}/merge_metrics_pass1.log" 2>&1
MISSING_RANK0_COUNT=$(count_report_list_items "${PASS1_KERNEL_REPORT_JSON}" missing_kernels_by_rank 0)
MISSING_RANK2_COUNT=$(count_report_list_items "${PASS1_KERNEL_REPORT_JSON}" missing_kernels_by_rank 2)
MERGED_NCU_CSV="${RUN_DIR}/kernel_metric_output_targeted_merged.csv"
KERNEL_METRICS_REPORT_JSON="${RUN_DIR}/kernel_metrics_report.json"
if [[ "${MISSING_RANK0_COUNT}" -gt 0 || "${MISSING_RANK2_COUNT}" -gt 0 ]]; then
  echo "[INFO] Step 5b/9: second targeted NCU pass for missing kernels (rank0=${MISSING_RANK0_COUNT}, rank2=${MISSING_RANK2_COUNT})."
  PASS2_CSVS=()
  if [[ "${MISSING_RANK0_COUNT}" -gt 0 ]]; then
    MISSING_KERNELS_RANK0_LIST="${RUN_DIR}/missing_kernels_pass1_rank0.list"
    write_rank_kernel_list "${PASS1_KERNEL_REPORT_JSON}" missing_kernels_by_rank 0 "${MISSING_KERNELS_RANK0_LIST}" > "${RUN_DIR}/missing_kernels_rank0_list.log" 2>&1
    echo "[INFO] Step 5b/9a: pass-2 rank0 missing kernels on GPU${NCU_STAGE0_GPU}."
    run_targeted_ncu_rank 0 "${NCU_STAGE0_GPU}" "${MISSING_KERNELS_RANK0_LIST}" "${NCU_PASS2_STAGE0_DIR}" "${RUN_DIR}/ncu_pass2_rank0.log"
    PASS2_CSVS+=("${NCU_PASS2_STAGE0_DIR}/output/kernel_metric_output.csv")
  fi
  if [[ "${MISSING_RANK2_COUNT}" -gt 0 ]]; then
    MISSING_KERNELS_RANK2_LIST="${RUN_DIR}/missing_kernels_pass1_rank2.list"
    write_rank_kernel_list "${PASS1_KERNEL_REPORT_JSON}" missing_kernels_by_rank 2 "${MISSING_KERNELS_RANK2_LIST}" > "${RUN_DIR}/missing_kernels_rank2_list.log" 2>&1
    echo "[INFO] Step 5b/9b: pass-2 rank2 missing kernels on GPU${NCU_STAGE1_GPU}."
    run_targeted_ncu_rank 2 "${NCU_STAGE1_GPU}" "${MISSING_KERNELS_RANK2_LIST}" "${NCU_PASS2_STAGE1_DIR}" "${RUN_DIR}/ncu_pass2_rank2.log"
    PASS2_CSVS+=("${NCU_PASS2_STAGE1_DIR}/output/kernel_metric_output.csv")
  fi
  prepare_case_kernel_metrics_report "${MERGED_NCU_CSV}" "${KERNEL_METRICS_REPORT_JSON}"     "${NCU_PASS1_STAGE0_CSV}"     "${NCU_PASS1_STAGE1_CSV}"     "${PASS2_CSVS[@]}"     > "${RUN_DIR}/merge_metrics_pass2.log" 2>&1
else
  cp "${PASS1_PREPARED_NCU_CSV}" "${MERGED_NCU_CSV}"
  cp "${PASS1_KERNEL_REPORT_JSON}" "${KERNEL_METRICS_REPORT_JSON}"
fi
FINAL_MISSING_COUNT=$(count_report_total_items "${KERNEL_METRICS_REPORT_JSON}" missing_kernels)
if [[ "${FINAL_MISSING_COUNT}" -gt 0 ]]; then
  echo "[FAIL] Targeted NCU collection still misses ${FINAL_MISSING_COUNT} required kernels after pass2." >&2
  cat "${KERNEL_METRICS_REPORT_JSON}" >&2
  exit 1
fi

echo "[INFO] Step 6/9: build slowdown assets and compare slowdown off/on."
python megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py \
  --trace-dir "${GLOBAL_TRACE_DIR}" \
  --nsys-sqlite "${NSYS_DIR}/all_ranks_scaling.sqlite" \
  --ncu-metrics-csv "${MERGED_NCU_CSV}" \
  --label-prefix cmd_trace \
  --output-dir "${SLOWDOWN_ASSETS_DIR}" \
  --model-path "${MODEL_PATH}" \
  --scaler-path "${SCALER_PATH}" \
  > "${RUN_DIR}/build_assets.log" 2>&1
for wrank_id in 0 2; do
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
    --wrank-id "${wrank_id}" \
    --output-json "${COMPARE_DIR}/wrank${wrank_id}.json" \
    > "${COMPARE_DIR}/wrank${wrank_id}.log" 2>&1
done

cat > "${RUN_DIR}/run_real_reference.sh" <<RUNREAL
#!/usr/bin/env bash
set -euo pipefail
cd ${PROJECT_ROOT@Q}
export PYTHONPATH=${PYTHONPATH@Q}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS@Q}
$(build_distributed_torchrun_cmd "${TRACE_TRAIN_ITERS}" traced)
RUNREAL
chmod +x "${RUN_DIR}/run_real_reference.sh"

echo "[INFO] Step 7/9: collect real 4-GPU hardware reference under nsys on GPUs ${REAL_REFERENCE_GPUS}."
REAL_NSYS_REPORT_BASE="${REFERENCE_NSYS_DIR}/distributed_reference"
set +e
CUDA_VISIBLE_DEVICES="${REAL_REFERENCE_GPUS}" \
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --wait=all \
    --trace-fork-before-exec=true \
    --force-overwrite true \
    --output "${REAL_NSYS_REPORT_BASE}" \
    bash "${RUN_DIR}/run_real_reference.sh" \
    > "${RUN_DIR}/real_reference_nsys.log" 2>&1
REAL_NSYS_EXIT_CODE=$?
set -e
if [[ ! -f "${REAL_NSYS_REPORT_BASE}.nsys-rep" ]]; then
  echo "[FAIL] Missing real-reference nsys report: ${REAL_NSYS_REPORT_BASE}.nsys-rep (exit=${REAL_NSYS_EXIT_CODE})" >&2
  exit 1
fi
if [[ "${REAL_NSYS_EXIT_CODE}" -ne 0 && "${REAL_NSYS_EXIT_CODE}" -ne 143 ]]; then
  echo "[FAIL] Real-reference nsys exited with unexpected code ${REAL_NSYS_EXIT_CODE}." >&2
  exit 1
fi
nsys export -t sqlite --force-overwrite true -o "${REFERENCE_NSYS_DIR}/distributed_reference.sqlite" "${REAL_NSYS_REPORT_BASE}.nsys-rep" >/dev/null
for rank_id in 0 1 2 3; do
  rank_trace=$(latest_rank_file "${REAL_TRACE_ROOT}" "${rank_id}")
  if [[ -z "${rank_trace}" || ! -f "${rank_trace}" ]]; then
    echo "[FAIL] Missing realistic trace for rank${rank_id} under ${REAL_TRACE_ROOT}" >&2
    exit 1
  fi
  cp "${rank_trace}" "${REFERENCE_TRACE_DIR}/$(basename "${rank_trace}")"
done

echo "[INFO] Step 8/9: build slowdown off/on vs hardware error table."
python tests/e2e/compare_ddp_slowdown_reference.py \
  --reference-trace-dir "${REFERENCE_TRACE_DIR}" \
  --reference-nsys-sqlite "${REFERENCE_NSYS_DIR}/distributed_reference.sqlite" \
  --sim-json "${COMPARE_DIR}/wrank0.json" \
  --sim-json "${COMPARE_DIR}/wrank2.json" \
  --output-json "${COMPARE_DIR}/reference_compare.json" \
  --output-md "${COMPARE_DIR}/reference_compare.md" \
  > "${COMPARE_DIR}/reference_compare.log" 2>&1

echo "[INFO] Step 9/9: summarize schedule, metrics, slowdown evidence, and hardware error table."
python - <<'PY' \
  "${RUN_DIR}" \
  "${CASE_DIR}" \
  "${GLOBAL_TRACE_DIR}" \
  "${SLOWDOWN_ASSETS_DIR}" \
  "${COMPARE_DIR}/wrank0.json" \
  "${COMPARE_DIR}/wrank2.json" \
  "${COMPARE_DIR}/reference_compare.json" \
  "${SUMMARY_JSON}" \
  "${SUMMARY_MD}" \
  "${DDP_BUCKET_SIZE}" \
  "${DRYRUN_BUCKETS_RANK0}" \
  "${DRYRUN_BUCKETS_RANK2}" \
  "${REQUIRED_KERNELS_JSON}" \
  "${MERGED_NCU_CSV}" \
  "${REFERENCE_NSYS_DIR}/distributed_reference.sqlite"
import json
import pandas as pd
from pathlib import Path
import sys
run_dir = Path(sys.argv[1])
case_dir = Path(sys.argv[2])
global_trace_dir = Path(sys.argv[3])
assets_dir = Path(sys.argv[4])
wrank0_json = Path(sys.argv[5])
wrank2_json = Path(sys.argv[6])
reference_compare_json = Path(sys.argv[7])
out_json = Path(sys.argv[8])
out_md = Path(sys.argv[9])
ddp_bucket_size = int(sys.argv[10])
dryrun_rank0 = int(sys.argv[11])
dryrun_rank2 = int(sys.argv[12])
required_kernels_json = Path(sys.argv[13])
merged_ncu_csv = Path(sys.argv[14])
reference_nsys_sqlite = Path(sys.argv[15])
summary = {
    'run_dir': str(run_dir),
    'case_dir': str(case_dir),
    'ddp_bucket_size': ddp_bucket_size,
    'baseline_default_bucket_trace_ddp_counts': {'rank0': 49, 'rank2': 50},
    'dryrun_bucket_counts': {'rank0': dryrun_rank0, 'rank2': dryrun_rank2},
    'trace_ddp_counts': {},
    'required_kernels': json.loads(required_kernels_json.read_text(encoding='utf-8')),
    'merged_ncu_unique_kernel_count': int(pd.read_csv(merged_ncu_csv)['Kernel Name'].astype(str).nunique()),
    'assets': {
        'kernel_feature_count': len(json.loads((assets_dir / 'kernel_features.json').read_text(encoding='utf-8'))),
        'blueprint_count': len(json.loads((assets_dir / 'backward_kernel_blueprints.json').read_text(encoding='utf-8'))),
    },
    'reference_compare': json.loads(reference_compare_json.read_text(encoding='utf-8')),
}
for path in sorted(global_trace_dir.glob('*.txt')):
    summary['trace_ddp_counts'][path.name] = sum(1 for line in path.read_text(encoding='utf-8').splitlines() if 'ddp_grad_comm' in line)
summary['wrank0'] = json.loads(wrank0_json.read_text(encoding='utf-8'))
summary['wrank2'] = json.loads(wrank2_json.read_text(encoding='utf-8'))
out_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
rows = summary['reference_compare']['rows']
ref_lines = []
for row in rows:
    ref_lines.append(
        f"- wrank{row['wrank_id']} hardware backward: `{row['hardware_backward_duration_ms']:.6f}` ms; "
        f"off/on abs err: `{row['backward_abs_err_ms_off']:.6f}` / `{row['backward_abs_err_ms_on']:.6f}` ms; "
        f"launch MAE off/on: `{row['ddp_launch_mae_ms_off']}` / `{row['ddp_launch_mae_ms_on']}`"
    )
out_md.write_text(
    "## GPT-6.7B Lightweight Slowdown E2E\n\n"
    f"- Run dir: `{run_dir}`\n"
    f"- Case dir: `{case_dir}`\n"
    f"- `--ddp-bucket-size`: `{ddp_bucket_size}`\n"
    f"- Dry-run bucket counts: `{summary['dryrun_bucket_counts']}`\n"
    f"- Trace DDP counts: `{summary['trace_ddp_counts']}`\n"
    f"- Required kernel short names: `{len(summary['required_kernels'])}`\n"
    f"- Merged NCU unique kernels: `{summary['merged_ncu_unique_kernel_count']}`\n"
    f"- Slowdown assets: `{summary['assets']}`\n"
    f"- wrank0 backward off/on: `{summary['wrank0']['backward_duration_ms_off']:.6f} -> {summary['wrank0']['backward_duration_ms_on']:.6f}` ms\n"
    f"- wrank2 backward off/on: `{summary['wrank2']['backward_duration_ms_off']:.6f} -> {summary['wrank2']['backward_duration_ms_on']:.6f}` ms\n"
    f"- Reference nsys sqlite: `{reference_nsys_sqlite}`\n"
    + '\n'.join(ref_lines)
    + '\n',
    encoding='utf-8',
)
print(out_md.read_text(encoding='utf-8'))
PY

echo "[PASS] GPT-6.7B lightweight slowdown E2E completed."
echo "[INFO] Summary: ${SUMMARY_MD}"
