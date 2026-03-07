## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initial comprehensive test report for WS256 dense simulation phase |

# Test Report: WS256 Dense (H800) Simulation Phase

**Date**: 2026-03-04  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Host Python**: `/opt/anaconda/envs/myenv_yc/bin/python`

## 1. Test Script Information

### 1.1 Data Organization Commands
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
EXP_DIR="megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2"
mkdir -p "${EXP_DIR}/database_profile" "${EXP_DIR}/schedule"
cp -av profiler_log/my_pp16_tp8_ep1_expnNone_dp2_nl96_hs12288_sl2048/. "${EXP_DIR}/database_profile/"
```

### 1.2 Schedule Generation Commands
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/src/scheduler/mg_scheduling
python mg_test.py \
  --local-size 8 \
  --world-size 256 \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 16 \
  --expert-model-parallel-size 1 \
  --num-experts 1 \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --seq-length 2048 \
  --hidden-size 12288 \
  --model-size 175 \
  --train-iters 10 \
  --trace-start 10 \
  --fp16

cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
SCHED_PARENT="megatron-sim-engine/simulation_inputs/scheduling_plans/mg_scheduling_plan_log/MODEL175_pp16_tp8_dp2_exp1_seq2048_mbs1_gbs128_fp16"
TS=$(ls -1 "${SCHED_PARENT}"/stage0_*_scheduling_plan.txt | sed -E 's/.*stage0_([0-9]{8}_[0-9]{6})_scheduling_plan\.txt/\1/' | sort | tail -1)
for sid in $(seq 0 15); do
  cp -av "${SCHED_PARENT}/stage${sid}_${TS}_scheduling_plan.txt" "megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/schedule/"
done
```

### 1.3 E2E Simulation Command (collective-sim)
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine
export PYTHONPATH=.
export SIMULATOR_HARDWARE_TYPE=H800_SXM
python simu_main.py \
  --framework megatron-lm \
  --mode simulate \
  --schedule-dir simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/schedule \
  --database-dir simulation_inputs/megatron_operation_log/h800_256gpus_gpt175b_tp8_pp16_dp2/database_profile \
  --world-size 256 \
  --pp-size 16 \
  --tp-size 8 \
  --exp-size 1 \
  --local-size 8 \
  --strategy 1F1B-none_interleaved \
  --cc-backend collective-sim \
  --cc-backend-options-json '{"collective-sim":{"placement_mode":"group_size"}}' \
  --collective-sim-repo-root src/core/cc_backend/collective-sim \
  --no-visualize
```

### 1.4 Metrics Extraction Command
- Scripted runtime extraction from timeline manager was executed and saved to:
  - `task_memory/task_2026-03-04_ws256_dense_simulation_phase/logs/ws256_dense_simulation_metrics_20260304_140456.json`

## 2. Validation Criteria

- Data organization validity:
  - `database_profile` file count == 16
  - rank ids cover `{0,16,...,240}`
- Schedule validity:
  - `schedule` file count == 16
  - stage ids cover `{0..15}`
- Simulation validity:
  - process exit code == 0
  - log contains `sim load time` and `sim execution time`
- Output metrics validity:
  - iteration time and throughput are computable
  - per-rank and per-stage breakdown is present in metrics JSON
- Measured comparison:
  - explicitly marked as `N/A (skipped by task decision)`

## 3. Test Results and Evidence

### 3.1 Organization Validation
- `DB_FILE_COUNT=16` (PASS)
- `SCH_FILE_COUNT=16` (PASS)
- rank coverage: `0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240`
- stage coverage: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`

### 3.2 E2E Simulation Result (PASS)
- Run log: `megatron-sim-engine/log/mg_scheduling/ws256_dense_simulate_group_size_20260304_140255.log`
- Key lines:
  - `sim load time: 2.894034s`
  - `world_size: 256, sim load time: 2.894034s, sim execution time: 45.604053s`

### 3.3 Aggregated Simulation Metrics (PASS)
- Metrics file:
  - `task_memory/task_2026-03-04_ws256_dense_simulation_phase/logs/ws256_dense_simulation_metrics_20260304_140456.json`
- Key outputs:
  - `sim_load_time_s = 2.837538`
  - `sim_execution_time_s = 46.70788`
  - `simulated_iteration_time_ms = 3587.99`
  - `tokens_per_iter = 262144`
  - `simulated_throughput_tokens_per_s = 73061.519124`

### 3.4 Slowest Ranks (Top-10)
| Rank | Stage | sum_time (ms) |
|------|-------|---------------|
| 16   | 1     | 3587.99 |
| 32   | 2     | 3563.87 |
| 48   | 3     | 3538.02 |
| 64   | 4     | 3519.70 |
| 0    | 0     | 3516.95 |
| 80   | 5     | 3489.64 |
| 96   | 6     | 3468.36 |
| 112  | 7     | 3444.57 |
| 128  | 8     | 3412.38 |
| 144  | 9     | 3396.67 |

### 3.5 Per-Stage Summary
| Stage | Count | min (ms) | mean (ms) | max (ms) |
|-------|-------|----------|-----------|----------|
| 0 | 1 | 3516.95 | 3516.95 | 3516.95 |
| 1 | 1 | 3587.99 | 3587.99 | 3587.99 |
| 2 | 1 | 3563.87 | 3563.87 | 3563.87 |
| 3 | 1 | 3538.02 | 3538.02 | 3538.02 |
| 4 | 1 | 3519.70 | 3519.70 | 3519.70 |
| 5 | 1 | 3489.64 | 3489.64 | 3489.64 |
| 6 | 1 | 3468.36 | 3468.36 | 3468.36 |
| 7 | 1 | 3444.57 | 3444.57 | 3444.57 |
| 8 | 1 | 3412.38 | 3412.38 | 3412.38 |
| 9 | 1 | 3396.67 | 3396.67 | 3396.67 |
| 10 | 1 | 3376.11 | 3376.11 | 3376.11 |
| 11 | 1 | 3359.58 | 3359.58 | 3359.58 |
| 12 | 1 | 3322.79 | 3322.79 | 3322.79 |
| 13 | 1 | 3285.87 | 3285.87 | 3285.87 |
| 14 | 1 | 3297.46 | 3297.46 | 3297.46 |
| 15 | 1 | 3250.42 | 3250.42 | 3250.42 |

### 3.6 Measured Comparison
- `N/A (skipped by task decision)`

## 4. Failure Handling Record
- No final blocking failure in the committed result path.
- Intermediate long-running attempts were replaced by a practical backend option (`placement_mode=group_size`) while preserving `collective-sim` backend type.
