## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-12 | Added implementation notes for sim-engine slowdown support |
| 2026-03-12 | Added Echo workflow validation notes and practical execution policy |
| 2026-03-15 | Added example index and Phase 8c/8d workflow entrypoints |

# Notes: Sim-Engine DDP Slowdown Support

## Scope
- DDP overlap + `backward_step` only.
- `simulate` mode only.
- Offline slowdown assets only.
- Practical workflow validation may reuse existing kernel metrics instead of collecting fresh `ncu` output every run.

## Relevant Code Paths
- `megatron/profiler/cmd.py`
- `megatron-sim-engine/src/core/simu_engine.py`
- `megatron-sim-engine/src/core/simulator_config.py`
- `megatron-sim-engine/simu_main.py`
- `megatron-sim-engine/src/extensions/slowdown_predictor.py`
- `Echo-slowdown/slowdown_collection/run.sh`
- `Echo-slowdown/slowdown_collection/run-nsys.sh`
- `Echo-slowdown/merge/merge_script.py`
- `Echo-slowdown/training_testing/prediction_api.py`
- `Echo-slowdown/run_all.sh`

## Example and Workflow Index
- Qwen stage-1 baseline entry: `examples/pretrain_qwen3_30b_a3b_moe.sh`
- Qwen DDP overlap trace wrapper: `examples/pretrain_qwen3_30b_a3b_moe_ddp_overlap_trace.sh`
- DeepSeek DDP overlap trace wrapper: `examples/pretrain_deepseek_v3_moe_ddp_overlap_trace.sh`
- Sim-engine overlap/slowdown walkthrough: `megatron-sim-engine/examples/06_ddp_overlap_slowdown_modes.sh`
- Auto trace-shaped PP schedule builder (Phase 8c): `megatron-sim-engine/tools/data_prep/schedule/build_trace_shaped_pp_schedule.py`
- Case-local kernel metrics preparer (Phase 8d): `megatron-sim-engine/tools/data_prep/slowdown/prepare_case_kernel_metrics.py`
- Self-contained lightweight E2E workflow: `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh`

## Phase 8c/8d Closure Notes
- Phase 8c is functionally closed by the trace-shaped PP schedule builder plus unit coverage. The builder consumes compressed trace files and emits replayable `stage*_trace_shaped_scheduling_plan.txt` plans for simulator input.
- Phase 8d is functionally closed by the case-local targeted `NCU` workflow in `tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh` together with `prepare_case_kernel_metrics.py`. The flow now derives required kernel short names from `trace + nsys`, runs serialized targeted `NCU`, performs optional second-pass missing-kernel collection, and feeds the merged case-local CSV into slowdown-asset generation.
- Remaining work is no longer workflow completeness; it is acceptance-quality accuracy tuning and wall-clock practicality for fresh `NCU` recollection on this machine.

## Ground Truth Observations
- `Echo-slowdown/` model input is compute-kernel metrics + `ground_truth` only.
- `overlap_ratio` is applied outside the XGBoost model.
- Existing simulator already replays DDP overlap comm/wait semantics from trace overlay.
- The practical Echo workflow on this machine is limited more by tool/runtime compatibility than by code-path completeness.

## Constraints
- No fallback logic inside the simulator runtime.
- Keep slowdown disabled by default.
- Preserve existing DDP overlap replay behavior when slowdown is off.
- Only patch `Echo-slowdown/` when a clear workflow bug or environment mismatch blocks the requested validation.

## Practical Workflow Policy
- Fresh `kernel_metric` collection is optional for workflow validation.
- Preferred practical command on the current machine:
  - `SKIP_KERNEL_METRIC=1 bash Echo-slowdown/run_all.sh`
- This still exercises:
  - `slowdown_collection`
  - `merge`
  - `training_testing/create_dataset.py`
  - `training_testing/train.py`
  - `training_testing/predict.py`
- Direct API smoke can be verified through `Echo-slowdown/training_testing/prediction_api.py` using a row from `input/test_csv/merged_features.csv`.


## Frozen Canonical Trio (2026-03-14)
- Canonical refreshed acceptance artifact: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518`
- Canonical simulator trace dir: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/megatron-sim-engine/simulation_inputs/megatron_operation_log/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/global_ranks_profile`
- Canonical baseline scaling `nsys` sqlite: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/nsys/all_ranks_scaling.sqlite`
- Canonical targeted `ncu` metrics csv: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/kernel_metric_output_targeted_merged.csv`
- Canonical hardware reference `nsys` sqlite: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/reference/nsys/distributed_reference.sqlite`
- Canonical compare outputs:
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/wrank0.json`
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/wrank2.json`
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/reference_compare.json`
  - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/e2e/artifacts/gpt67b_ddp_slowdown_lightweight_bucket10000000_20260313_165901_1876518/compare/reference_compare.md`
