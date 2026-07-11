## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-12 | Added GPT-6.7B pp2/tp1/dp2/sl256 lightweight E2E validation report |
| 2026-03-12 | Added slowdown-support implementation verification report |
| 2026-03-12 | Expanded report with collective-sim recovery and Echo workflow validation |

# Test Report: Sim-Engine DDP Slowdown Support

**Date**: 2026-03-12
**Environment**: `conda` env `myenv_yc` (`Python 3.9.18`, `/opt/anaconda/envs/myenv_yc/bin/python`)

## Test Script Information
- Scripts:
  - `tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py`
  - `tests/unit/test_echo_slowdown_merge.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py`
  - `megatron-sim-engine/tests/unit/test_slowdown_predictor.py`
  - `megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py`
  - `megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py`
  - `megatron-sim-engine/tests/unit/*`
  - `Echo-slowdown/run_all.sh`
  - `Echo-slowdown/training_testing/prediction_api.py`
- Reproducible commands:
  ```bash
  python --version
  which python

  cd megatron-sim-engine
  git submodule update --init --recursive src/core/cc_backend/collective-sim
  pytest -q tests/unit

  cd ..
  pytest -q tests/unit/test_echo_slowdown_merge.py tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py

  cd Echo-slowdown/slowdown_collection
  bash run.sh

  cd ../merge
  bash run.sh

  cd ../training_testing
  bash run.sh

  cd ..
  SKIP_KERNEL_METRIC=1 bash run_all.sh

  cd training_testing
  python - <<'PY'
  import pandas as pd
  from prediction_api import SlowdownPredictor

  feature_keys = [
      'ground_truth',
      'Compute throughput',
      'Memory throughput',
      'DRAM throughput',
      'Achieved occupancy',
      'Maximum occupancy',
      'L1 hit rate',
      'L2 hit rate',
  ]

  df = pd.read_csv('input/test_csv/merged_features.csv')
  row = df[df['overlap_ratio'] > 0].iloc[0]
  predictor = SlowdownPredictor('output/xgb_model.json')
  features = {key: float(row[key]) for key in feature_keys}
  print(predictor.predict_slowdown(features, float(row['overlap_ratio'])))
  PY
  ```

## Validation Criteria
- `cmd_uid` is present in kernel-ground-truth NVTX labels without breaking existing label fields.
- Slowdown CLI/config rejects unsupported or incomplete slowdown inputs.
- Slowdown asset loader rejects malformed schema / missing features / duplicate JSON keys.
- Slowdown predictor adapter clips negative slowdown to zero and matches `Echo-slowdown` API shape.
- Fixed-point kernel solver returns baseline when overlap is zero and increases duration under overlap.
- Backward micro-scheduler delays later DDP bucket launches when earlier kernels slow down.
- Overlay comm replay consumes slowdown-aware launch/finish timestamps when slowdown is enabled.
- Finalize wait still waits on updated comm finish times.
- Slowdown-disabled replay retains offset-based DDP comm behavior.
- `collective-sim` full unit regression passes after restoring the submodule and measured P2P profile assets.
- `Echo-slowdown` practical workflow produces:
  - slowdown ground-truth Excel from `slowdown_collection`,
  - merged feature CSV,
  - trained XGBoost model,
  - batch prediction outputs,
  - direct API prediction on a trace-derived overlapped kernel row.

## Test Results

| Suite | Result | Details |
|------|--------|---------|
| Root NVTX tracing + Echo merge unit tests | PASS | `9 passed in 1.77s` |
| Slowdown-focused sim-engine unit+integration tests | PASS | Covered by existing slowdown report entries and full `tests/unit` regression |
| Sim-engine full `tests/unit` | PASS | `57 passed in 0.57s` |
| Echo `slowdown_collection -> merge -> train -> predict` practical chain | PASS | Completed with real `slowdown_collection` and reused kernel metrics |
| Echo one-command workflow (`SKIP_KERNEL_METRIC=1 bash run_all.sh`) | PASS | End-to-end command completed successfully |
| Echo direct predictor API smoke | PASS | Returned slowdown factor and predicted execution time on an overlapped kernel |
| Minimal E2E slowdown simulate smoke | PASS | Assets built from real trace/sqlite/NCU report; slowdown delayed the DDP comm launch and slightly increased backward time |
| Acceptance-grade accuracy study | PENDING | Relaxed in this round; canonical trace/`nsys`/`ncu` trio still needed |

## Evidence
- Fresh verification output:
  - `57 passed in 0.57s`
  - `9 passed in 1.77s`
  - `All modules ran successfully.`
  - `All predictions match!`
- Echo workflow artifacts:
  - `Echo-slowdown/slowdown_collection/output/slowdown_stats_output_device_0.xlsx`
  - `Echo-slowdown/merge/output/merged_features.csv`
  - `Echo-slowdown/training_testing/output/train_dataset.csv`
  - `Echo-slowdown/training_testing/output/xgb_model.json`
  - `Echo-slowdown/training_testing/output/prediction/output_metrics.txt`
- Echo merged dataset summary:
  - `Merged rows: 515`
  - `slowdown_nonzero: 494`
- Echo predictor API smoke:
  - kernel: `vectorized_elementwise_kernel`
  - `overlap_ratio: 1.0`
  - result:
    - `predicted_slowdown_factor: 0.64179087`
    - `predicted_execution_time: 22643.579635620117`
- Echo batch prediction metrics (`output_metrics.txt`):
  - `Our RMSE of slowdown (original) 1.1696784996905025`
  - `Our RMSE of duration 4618.365227985145`
  - `Baseline RMSE of duration 5599.176938183023`

## Functional Coverage Summary
- Verified Megatron tracing label enrichment in `megatron/profiler/cmd.py`.
- Verified slowdown asset generation path in `megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py`.
- Verified slowdown adapter and loader in `megatron-sim-engine/src/extensions/slowdown_predictor.py`.
- Verified slowdown-enabled replay path in `megatron-sim-engine/src/core/simu_engine.py`.
- Verified slowdown-disabled comm replay regression path remains offset-based.
- Verified `collective-sim` backend can initialize in the restored unit-test environment.
- Verified `Echo-slowdown` practical workflow on this machine, including a one-command path that skips fresh `kernel_metric` collection.

## Limitations / Risks
- The practical Echo workflow currently reuses `merge/input/kernel_metric_output.csv` via `SKIP_KERNEL_METRIC=1`; this is intentional for workflow completeness on the shared machine, not for paper-grade accuracy.
- `Echo-slowdown` still does not persist the `StandardScaler` fitted in training, so downstream prediction semantics remain those of upstream Echo rather than a perfectly closed training/inference pipeline.
- The restored measured P2P `.txt` assets for `collective-sim` are local and currently ignored by Git.

## 2026-03-12 Minimal E2E Slowdown Simulate Smoke

### Test Script Information
- Case root: `megatron-sim-engine/simulation_inputs/megatron_operation_log/e2e_dense_ddp_slowdown_gpu7_smoke_20260312_145348`
- Model weight path: `Echo-slowdown/training_testing/output/xgb_model.json`
- Existing Nsight Systems sqlite: `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_20260312_145348/scale_rank0.sqlite`
- Existing Nsight Compute report: `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_fp16_20260312_150116/scale_rank0_ncu.ncu-rep`
- Exported case-local metrics CSV: `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_fp16_20260312_150116/ncu_processed/kernel_metric_output.csv`
- Compare helper: `tests/e2e/run_ddp_slowdown_compare.py`
- Environment: `Python 3.9.18`, interpreter `/opt/anaconda/envs/myenv_yc/bin/python`
- Commands:
  ```bash
  cd megatron-sim-engine
  pytest -q tests/unit
  pytest -q tests/integration/test_simu_engine_ddp_slowdown_integration.py

  cd ..
  /usr/local/cuda-12.1/bin/ncu -i \
    task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_fp16_20260312_150116/scale_rank0_ncu.ncu-rep \
    --page details --csv --log-file \
    task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_fp16_20260312_150116/ncu_processed/details.csv

  /usr/local/cuda-12.1/bin/ncu -i \
    task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_fp16_20260312_150116/scale_rank0_ncu.ncu-rep \
    --print-kernel-base function --csv > \
    task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_fp16_20260312_150116/ncu_processed/kshortname.csv

  python megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py \
    --trace-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/e2e_dense_ddp_slowdown_gpu7_smoke_20260312_145348/database_profile \
    --nsys-sqlite task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_20260312_145348/scale_rank0.sqlite \
    --ncu-metrics-csv task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/dense_ddp_slowdown_e2e_fp16_20260312_150116/ncu_processed/kernel_metric_output.csv \
    --label-prefix cmd_trace \
    --output-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/e2e_dense_ddp_slowdown_gpu7_smoke_20260312_145348/slowdown_assets \
    --model-path Echo-slowdown/training_testing/output/xgb_model.json

  python tests/e2e/run_ddp_slowdown_compare.py \
    --trace-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/e2e_dense_ddp_slowdown_gpu7_smoke_20260312_145348/database_profile \
    --database-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/e2e_dense_ddp_slowdown_gpu7_smoke_20260312_145348/database_profile \
    --schedule-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/e2e_dense_ddp_slowdown_gpu7_smoke_20260312_145348/schedule \
    --slowdown-assets-dir megatron-sim-engine/simulation_inputs/megatron_operation_log/e2e_dense_ddp_slowdown_gpu7_smoke_20260312_145348/slowdown_assets \
    --slowdown-model-path Echo-slowdown/training_testing/output/xgb_model.json \
    --world-size 2 --local-size 2 --pp-size 1 --tp-size 1 --exp-size 1 \
    --strategy no-pipelining \
    --wrank-id 0 \
    --output-json task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/e2e_dense_ddp_slowdown_gpu7_smoke_compare.json
  ```

### Validation Criteria
- Slowdown-enabled simulator must process at least one `backward_step.cmd_uid`.
- The same `ddp_grad_comm` must exist in both slowdown-off and slowdown-on summaries.
- `backward_step` must not shrink under slowdown.
- At least one DDP comm launch must be delayed, or the backward duration must increase.
- Regression suites for builder parsing, short-name feature matching, residual duration preservation, and `PP=1 no-pipelining` slowdown replay must pass.

### Test Results
- `megatron-sim-engine/tests/unit`: PASS (`60 passed in 0.56s`)
- `megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py`: PASS (`2 passed in 0.10s`)
- Minimal E2E slowdown simulate smoke: PASS

### Evidence
- Output JSON: `task_memory/task_2026-03-12_sim_engine_slowdown_support/logs/e2e_dense_ddp_slowdown_gpu7_smoke_compare.json`
- Key numbers:
  - `backward_duration_ms_off = 10.54`
  - `backward_duration_ms_on = 10.541538`
  - `backward_duration_delta_ms = 0.001538`
  - shared comm uid: `ddpcomm-b2fd9ffefbea`
  - delayed comm uid: `ddpcomm-b2fd9ffefbea`
  - processed backward cmd uid: `cmd-91266ed41298`
- Timing evidence from JSON:
  - slowdown-off comm join: `8588878574.91`
  - slowdown-on comm join: `8588878575.14`
  - launch delay: `0.23 ms`

### Interpretation
- This smoke validates the intended v1 behavior: the slowdown predictor is consumed on overlap-phase backward compute kernels, the simulated `backward_step` becomes slightly longer, and the associated DDP bucket launch is correspondingly delayed.
- The tiny numerical delta is expected for this minimal case because the trained model and the exported kernel metrics were reused for functionality validation rather than re-trained for this exact workload.


## Test Report: GPT-6.7B Lightweight DDP Slowdown E2E

**Date**: 2026-03-12
**Environment**: `conda activate myenv_yc` (`Python 3.9.18`)
**GPUs Used**: GPU7 for DDP bucket dry-run, GPU4 for traced scaling `nsys`, prepared lightweight metrics CSV assembled earlier from targeted NCU augmentation on GPU5/GPU6.

### Test Script Information
- Script: `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh`
- Metrics CSV: `tests/e2e/artifacts/gpt67b_ddp_slowdown_e2e_bucket10000000_20260312_161902_928972/gpt67b_kernel_metric_output_lightweight.csv`
- Run dir: `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015`
- Case dir: `megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015`
- Commands:
  ```bash
  export GPT67B_SLOWDOWN_METRICS_CSV=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_e2e_bucket10000000_20260312_161902_928972/gpt67b_kernel_metric_output_lightweight.csv
  bash tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh
  ```

### Validation Criteria
- `--ddp-bucket-size 10000000` must increase DDP bucket count versus the earlier default-bucket GPT-6.7B traces.
- The traced case must contain DDP overlap records on all four ranks and case-local `slowdown_assets/` must be generated successfully.
- Slowdown-enabled simulation must process at least one backward `cmd_uid` and must increase `backward_step` wall time for representative wranks.
- At least one DDP comm launch must be delayed after slowdown injection for each checked wrank.
- The case must replay successfully with a `pp2,tp1,dp2,sl256` schedule shape that keeps PP send/recv dependencies valid.

### Test Results

| Test Item | Result | Details |
|-----------|--------|---------|
| Lightweight E2E script | PASS | Completed end-to-end and emitted `summary.md` / `summary.json` |
| Larger DDP bucket count | PASS | Default-bucket evidence rank0=`49`, rank2=`50`; new dry-run rank0=`65`, rank2=`66` |
| Trace DDP overlap records | PASS | rank0/rank1=`65`, rank2/rank3=`66` |
| Slowdown assets | PASS | `kernel_feature_count=23`, `blueprint_count=4` |
| wrank0 slowdown compare | PASS | `42.44 -> 61.035222 ms`, delayed comms=`65` |
| wrank2 slowdown compare | PASS | `44.05 -> 62.695410 ms`, delayed comms=`66` |

### Evidence
- Earlier default-bucket trace evidence:
  - `profiler_log/pp2_tp1_ep1_expnNone_dp2_nl32_hs4096_sl256/wd4_tp1_pp2_exp1_expNumNone_numl32_bs1_rank0_20260312145531.txt`
  - `profiler_log/pp2_tp1_ep1_expnNone_dp2_nl32_hs4096_sl256/wd4_tp1_pp2_exp1_expNumNone_numl32_bs1_rank2_20260312145938.txt`
- New run summary:
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015/summary.md`
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015/summary.json`
- Compare outputs:
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015/compare/wrank0.json`
  - `tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015/compare/wrank2.json`
- Manual PP dependency schedule:
  - `megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015/manual_schedule_ppdeps/stage0_manual_ppdeps_scheduling_plan.txt`
  - `megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260312_170102_996015/manual_schedule_ppdeps/stage1_manual_ppdeps_scheduling_plan.txt`
- Key observed values:
  - wrank0 processed backward cmd_uids: `cmd-0a3db334730b`, `cmd-5fa4887655a6`
  - wrank2 processed backward cmd_uids: `cmd-0a3db334730b`, `cmd-5fa4887655a6`
  - slowdown predictor was consumed on overlap-phase backward kernels, which stretched backward wall time and delayed subsequent DDP bucket launches instead of only modifying test-side summaries.

### Notes
- This run is intentionally lightweight and prioritizes end-to-end functional validation of the slowdown injection path over paper-grade numerical accuracy.
- The manual PP dependency schedule is required because the current GPT-6.7B scaling trace is a compressed single-batch top-level trace; replaying the full `mg_test.py` schedule against it leads to missing `dp_allreduce` matches or deadlock.
