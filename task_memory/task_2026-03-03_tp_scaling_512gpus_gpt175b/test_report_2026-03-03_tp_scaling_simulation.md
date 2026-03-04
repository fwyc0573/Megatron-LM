## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-03 | Added end-to-end execution report for PP=8 TP scaling simulation |
| 2026-03-03 | Added fake_tp TP partition validation (weight-shape + unit test) |
| 2026-03-03 | Added bug3 minimal-fix regression tests and before/after log comparison |

# Test Report: A800 GPT-175B TP Scaling Simulation (PP=8, world=1024)

**Date**: 2026-03-03  
**Environment**: `conda env = myenv_yc`, `Python 3.9.18`, `SIMULATOR_HARDWARE_TYPE=A800_SXM`

## 1. Test Script Information

### 1.1 Step0 (backend capability)
- Static evidence scripts: shell inspection + source references
- Dynamic probe command: internal Python probe (collective-sim `predict_collective_time`) for TP=`8/16/32/64`, world=`1024`
- Output:
  - `logs/step0_static_evidence.md`
  - `logs/step0_quick_probe.md`
  - `logs/step0_quick_probe.json`

### 1.2 Step1 profiling (Megatron scaling mode)
- Scripted run log: `logs/step1_profiling.log`
- Per-rank logs:
  - `logs/profiling_runs/tp8_pp8_dp16/rank_{0,128,256,384,512,640,768,896}.log`
  - `logs/profiling_runs/tp16_pp8_dp8/rank_{0,128,256,384,512,640,768,896}.log`
  - `logs/profiling_runs/tp32_pp8_dp4/rank_{0,128,256,384,512,640,768,896}.log`

### 1.3 Step2-3 data/schedule
- Database copy summary: `logs/step2_database_copy_summary.json`
- Schedule summary: `logs/step3_schedule_summary.json`

### 1.4 Step4 simulate
- Commands (representative):
```bash
SIMULATOR_HARDWARE_TYPE=A800_SXM \
python tests/performance/run_simu_with_collective_cache.py \
  --framework megatron-lm --mode simulate \
  --schedule-dir <config>/schedule \
  --database-dir <config>/database_profile \
  --world-size 1024 --pp-size 8 --tp-size <8|16|32> --exp-size 1 \
  --local-size <8|16|32> --no-visualize --cc-backend collective-sim \
  --cc-backend-options-json '{"collective-sim":{"placement_mode":"global","strict_mpu_alignment":true,"gpus_per_server":<node_size>}}'
```
- Logs:
  - `logs/step4_simulate.log`
  - `logs/simulate_tp8_pp8_dp16.log`
  - `logs/simulate_tp16_pp8_dp8.log`
  - `logs/simulate_tp32_pp8_dp4.log`

### 1.5 Result extraction
- Generated outputs:
  - `results_tp_scaling_512gpus_gpt175b.csv`
  - `results_tp_scaling_512gpus_gpt175b.md`
  - `logs/step5_metrics_extraction.json`

## 2. Validation Criteria

1. Step0 static evidence chain complete with source path/line references.  
2. Step0 dynamic probe: TP `8/16/32/64` has `g_in_group == TP` and monotonic `predicted_time_ms`.  
3. Step1: each config has exactly 8 representative rank profiles (`0,128,256,384,512,640,768,896`).  
4. Step2: each `database_profile/` contains exactly 8 copied files.  
5. Step3: each `schedule/` contains exactly 8 stage plan files (`stage0..stage7`).  
6. Step4: simulate for TP=`8/16/32` exits `0` and topology semantics check passes (`TP intra-node`, `DP/PP inter-node`).  
7. Step5: comparative table generated with fixed formula and fixed `GBS=256`.  
8. Step6: report includes backend limitations, comm semantic verification, TP64 strict-blocking, interpretation caveats.

## 3. Test Results and Evidence

| Check | Result | Evidence |
|---|---|---|
| Step0-Static | PASS | `logs/step0_static_evidence.md` |
| Step0-Dynamic | PASS | `logs/step0_quick_probe.md`, `logs/step0_quick_probe.json` |
| Step1-Profiling | PASS | `logs/step1_profiling.log` |
| Step2-Database | PASS | `logs/step2_database_copy_summary.json` |
| Step3-Schedule | PASS | `logs/step3_schedule_summary.json` |
| Step4-Topology/Comm Semantics | PASS | `logs/step4_topology_validation.md`, `logs/step4_comm_semantics.md` |
| Step4-Simulate Exit | PASS | `logs/step4_simulate.log` (three configs completed) |
| Step5-Comparative Table | PASS | `results_tp_scaling_512gpus_gpt175b.csv`, `results_tp_scaling_512gpus_gpt175b.md`, `logs/step5_metrics_extraction.json` |
| TP64 strict GPT-175B block | PASS (expected block) | `logs/tp64_blocking_evidence.md` |

## 4. Failure Handling / Runtime Blockers

### 4.1 Observed blocker
- `tp32_pp8_dp4` with `collective-sim` in strict global placement produced very long `htsim_ndp` calls (`-rail 32 ... -nodes 1024`).

### 4.2 Root cause
- Repeated communication predictions over structurally equivalent participant groups caused many high-cost htsim invocations.

### 4.3 Mitigation applied
- Added runtime wrapper script `tests/performance/run_simu_with_collective_cache.py`:
  - No simulator public API change.
  - Adds in-process memoization for `predict_collective_time`.
  - Canonicalizes `participant_ranks` to topology-equivalent IDs to improve cache hit-rate while preserving server-local structure.

### 4.4 Verification after mitigation
- TP=`8/16/32` simulations all completed with expected logs and final result table generated.

## 5. Notes on Metric Extraction

- Because Step4 runs used `--no-visualize`, direct `sum_time` prints were not emitted.
- Stage time extraction uses max observed `last_operation_time` per representative stage-rank from simulate log.
- This extraction method is documented in `results_tp_scaling_512gpus_gpt175b.md` and consistently applied across all three executed configs.

## 6. Additional Validation: fake_tp Partition Effectiveness

### 6.1 Validation intent
- Verify whether TP partition is really effective in scaling mode via **weight shape** and TE attention `tp_size`, instead of inferring from runtime only.

### 6.2 Test scripts and commands
- Script:
  - `logs/tp_weight_partition_validation.py`
- Commands:
```bash
python task_memory/task_2026-03-03_tp_scaling_512gpus_gpt175b/logs/tp_weight_partition_validation.py \
  > task_memory/task_2026-03-03_tp_scaling_512gpus_gpt175b/logs/tp_weight_partition_validation.log 2>&1

pytest -q tests/unit_tests/transformer/test_transformer_engine_tp_size_resolution.py
```

### 6.3 Validation criteria
1. Control case (`tensor_model_parallel_size=1`, `fake_tp=16`, scaling mode): no TP partition in TE layer.
2. Scaling run-path case (`tensor_model_parallel_size=fake_tp`): TE layer weight shape scales with TP.
3. TE attention module `tp_size` equals overridden TP size in scaling run-path case.
4. Unit tests pass.

### 6.4 Results
- Weight-shape evidence (`logs/tp_weight_partition_validation.log`):
  - control: `linear_weight_shape=(1024, 256)`, `attention_tp_size=1`
  - scaling tp16: `linear_weight_shape=(64, 256)`, `attention_tp_size=16`
  - scaling tp8: `linear_weight_shape=(128, 256)`, `attention_tp_size=8`
- Unit test result:
  - `tests/unit_tests/transformer/test_transformer_engine_tp_size_resolution.py`
  - `3 passed` (warnings only), exit code `0`

### 6.5 Interpretation
- Scaling mode TP partition is effective when run-path override is applied (`pretrain_llama.py` sets `config.tensor_model_parallel_size = args.fake_tp`).
- Therefore, TP compute-time nonlinearity should not be directly attributed to missing TP partition activation.

## 7. Bug3 Minimal Fix Validation

### 7.1 Code changes under test
- File:
  - `megatron-sim-engine/src/core/simu_engine.py`
- Scope:
  - In `process_mg_profile_files()`, remove parser-time legacy allreduce estimator usage for:
    - `dp_allreduce`
    - `ep_allreduce`
    - `exp_dp_allreduce`
  - Keep tensor metadata and safe placeholder/profile duration only.

### 7.2 Test scripts and commands
- Unit tests:
```bash
cd megatron-sim-engine
pytest -q \
  tests/unit/test_simu_engine_profile_parser_legacy_free.py \
  tests/unit/test_simu_engine_cc_semantics.py \
  tests/unit/test_simu_engine_ep_exp_semantics.py
```
- Parser-path regression capture:
```bash
python - <<'PY' > task_memory/task_2026-03-03_tp_scaling_512gpus_gpt175b/logs/bug3_parser_after_patch.log 2>&1
...
from src.core.simu_engine import process_mg_profile_files
...
PY
```

### 7.3 Validation criteria
1. `process_mg_profile_files()` does not call `get_comm_op_exc_time` for parser-time allreduce operations.
2. Existing communication semantic tests continue to pass.
3. After patch, parser-path log contains no `Comm time calculation` legacy estimator output.

### 7.4 Results
- Unit tests: `13 passed in 0.14s`
  - Includes new test: `test_simu_engine_profile_parser_legacy_free.py`
- Regression log comparison:
  - Before: `logs/simulate_tp16_pp8_dp8.log` contains legacy `Comm time calculation` before `CC backend initialized`.
  - After: `logs/bug3_parser_after_patch.log` contains parser metadata lines, and no `Comm time calculation`.
  - Comparison doc: `logs/bug3_regression_compare_2026-03-03.md`
