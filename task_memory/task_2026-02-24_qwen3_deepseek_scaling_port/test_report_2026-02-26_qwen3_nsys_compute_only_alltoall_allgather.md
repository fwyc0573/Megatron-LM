## Test Report: Qwen3 trace/kernel-ground-truth (alltoall vs allgather)

**Date**: 2026-02-26  
**Environment**: conda `myenv_yc`, Python 3.9, CUDA 12.1, GPUs 0-7 (distributed) + GPU7 single-card loop (scaling)  

### 1) Test Script Information

- Code paths:
  - `megatron/profiler/cmd.py`
  - `megatron/training/arguments.py`
  - `megatron/core/tensor_parallel/mappings.py`
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/performance/compare_qwen_nsys_compute_only.py`
- Unit tests:
  - `tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py`
  - `tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py`
  - `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py`
- Core commands (reproducible):
  ```bash
  # Unit tests
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29624 \
    PYTHONPATH=$(pwd) \
    pytest -q \
      tests/unit_tests/tensor_parallel/test_mappings_moe_api.py \
      tests/unit_tests/profiler/test_cmd_subop_sync_mode.py \
      tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py \
      tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
      tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py

  # Nsight capture (short window) - alltoall distributed/scaling
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nsys profile -w true -t cuda,nvtx,osrt --sample=none \
    --force-overwrite=true --trace-fork-before-exec=true \
    -o logs/nsys_kernel_gt/qwen_alltoall_dist_trace4 \
    bash -lc "MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TP=1 PP=4 EP=2 MICRO_BATCH_SIZE=8 SEQ_LEN=2048 TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=event TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PREFIX=cmd_trace MOE_TOKEN_DISPATCHER_TYPE=alltoall MASTER_PORT=6790 bash examples/pretrain_qwen3_30b_a3b_moe.sh"

  CUDA_VISIBLE_DEVICES=7 nsys profile -w true -t cuda,nvtx,osrt --sample=none \
    --force-overwrite=true --trace-fork-before-exec=true \
    -o logs/nsys_kernel_gt/qwen_alltoall_scale_trace4 \
    bash -lc "MODE=scaling MODEL_PROFILE=smoke FAKE_RANK_ORDER=0,7 SCALE_GPU=0 TP=1 PP=4 EP=2 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 MICRO_BATCH_SIZE=8 SEQ_LEN=2048 TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=event TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PREFIX=cmd_trace MOE_TOKEN_DISPATCHER_TYPE=alltoall MASTER_PORT=6800 bash examples/pretrain_qwen3_30b_a3b_moe.sh"

  # Nsight export + kernel breakdown
  /usr/local/cuda-12.1/bin/nsys export --type sqlite --force-overwrite=true \
    --output logs/nsys_kernel_gt/qwen_alltoall_dist_trace4_sqlite \
    logs/nsys_kernel_gt/qwen_alltoall_dist_trace4.nsys-rep
  /usr/local/cuda-12.1/bin/nsys export --type sqlite --force-overwrite=true \
    --output logs/nsys_kernel_gt/qwen_alltoall_scale_trace4_sqlite \
    logs/nsys_kernel_gt/qwen_alltoall_scale_trace4.nsys-rep
  python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
    --sqlite logs/nsys_kernel_gt/qwen_alltoall_dist_trace4_sqlite \
    --label-prefix cmd_trace --ranks 0,7 \
    --ops forward_step,backward_step,optimizer_step \
    --json-path logs/nsys_kernel_gt/qwen_alltoall_dist_trace4_kernel_breakdown.json \
    --report-path logs/nsys_kernel_gt/qwen_alltoall_dist_trace4_kernel_breakdown.md
  python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
    --sqlite logs/nsys_kernel_gt/qwen_alltoall_scale_trace4_sqlite \
    --label-prefix cmd_trace --ranks 0,7 \
    --ops forward_step,backward_step,optimizer_step \
    --json-path logs/nsys_kernel_gt/qwen_alltoall_scale_trace4_kernel_breakdown.json \
    --report-path logs/nsys_kernel_gt/qwen_alltoall_scale_trace4_kernel_breakdown.md
  python tests/performance/compare_qwen_nsys_compute_only.py \
    --distributed-json logs/nsys_kernel_gt/qwen_alltoall_dist_trace4_kernel_breakdown.json \
    --scaling-json logs/nsys_kernel_gt/qwen_alltoall_scale_trace4_kernel_breakdown.json \
    --ranks 0,7 --ops forward_step,optimizer_step \
    --report-path logs/compare_nsys/qwen_alltoall_nsys_compute_only_fwd_optim.log

  # Trace A/B full-8 reruns (alltoall / allgather)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TP=1 PP=4 EP=2 \
    MICRO_BATCH_SIZE=8 SEQ_LEN=2048 TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=event \
    TRACE_KERNEL_GROUND_TRUTH=0 MOE_TOKEN_DISPATCHER_TYPE=alltoall MASTER_PORT=6830 \
    bash examples/pretrain_qwen3_30b_a3b_moe.sh
  MODE=scaling MODEL_PROFILE=smoke SCALE_GPU=7 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 TP=1 PP=4 EP=2 \
    MICRO_BATCH_SIZE=8 SEQ_LEN=2048 TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=event \
    TRACE_KERNEL_GROUND_TRUTH=0 MOE_TOKEN_DISPATCHER_TYPE=alltoall MASTER_PORT=6840 \
    bash examples/pretrain_qwen3_30b_a3b_moe.sh

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TP=1 PP=4 EP=2 \
    MICRO_BATCH_SIZE=8 SEQ_LEN=2048 TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=event \
    TRACE_KERNEL_GROUND_TRUTH=0 MOE_TOKEN_DISPATCHER_TYPE=allgather MASTER_PORT=6870 \
    bash examples/pretrain_qwen3_30b_a3b_moe.sh
  MODE=scaling MODEL_PROFILE=smoke SCALE_GPU=7 FAKE_WORLD_SIZE=8 FAKE_PP=4 FAKE_TP=1 FAKE_EXP=2 TP=1 PP=4 EP=2 \
    MICRO_BATCH_SIZE=8 SEQ_LEN=2048 TRACE_START=4 TRAIN_ITERS=6 TRACE_SUBOP_SYNC_MODE=event \
    TRACE_KERNEL_GROUND_TRUTH=0 MOE_TOKEN_DISPATCHER_TYPE=allgather MASTER_PORT=6880 \
    bash examples/pretrain_qwen3_30b_a3b_moe.sh
  ```

### 2) Validation Criteria

- Kernel-ground-truth pipeline works end-to-end:
  - Nsight sqlite export succeeds.
  - NVTX parsing and kernel overlap aggregation produce non-empty forward/backward/optimizer rows.
- compare helper works with robust stats:
  - per-op `trimmed_mean + median-of-runs` report is generated.
- A/B dispatcher validation:
  - alltoall and allgather both run without OOM/crash under the same config (`TP1 PP4 EP2 DP2`, `seq=2048`, `mbs=8`).
  - compare output is available for both trace-level and nsys-kernel-level views.

### 3) Test Results and Evidence

#### 3.1 Unit tests

| Suite | Result | Evidence |
|---|---|---|
| profiler/performance/tensor_parallel new tests | PASS | `7 passed` |
| broader regression around new files | PASS | `29 passed` (targeted test matrix) |

#### 3.2 Failure encountered and fix

- Failure:
  - allgather distributed run initially failed:
    - `TypeError: gather_from_sequence_parallel_region_to_moe() got an unexpected keyword argument 'use_global_buffer'`
- Fix (minimal, fail-fast compatible):
  - `megatron/core/tensor_parallel/mappings.py`
    - make `gather_from_sequence_parallel_region_to_moe(input_, use_global_buffer=False)` accept the kwarg.
  - Added unit guard:
    - `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py`
- Re-run after fix:
  - allgather distributed/scaling both PASS (no OOM, no crash).

#### 3.3 Nsight kernel-level compute-only (rank0/rank7)

- alltoall:
  - report: `logs/compare_nsys/qwen_alltoall_nsys_compute_only_fwd_optim.log`
  - forward diff: rank0 `0.48%`, rank7 `14.00%` (median `7.24%`)
  - optimizer diff: rank0 `16.45%`, rank7 `22.31%`
- allgather:
  - report: `logs/compare_nsys/qwen_allgather_nsys_compute_only_fwd_optim.log`
  - forward diff: rank0 `23.11%`, rank7 `28.94%`
  - optimizer diff: rank0 `21.51%`, rank7 `21.63%`

Interpretation: kernel-level compute-only口径下，alltoall显著优于allgather，但仍有rank/stage依赖的残余差异（尤其optimizer）。

#### 3.4 Trace-level A/B (full 8 ranks)

- alltoall full-8:
  - report: `logs/compare_trace/qwen_alltoall_trace_compare_full8.log`
  - `op_rank_median_aux_summary`:
    - `forward_step`: `5.70%`
    - `backward_step`: `13.60%`
    - `optimizer_step`: `3.40%`
- allgather full-8 (after comm trace fix):
  - report: `logs/compare_trace/qwen_allgather_trace_compare_full8_after_comm_tracefix.log`
  - `op_rank_median_aux_summary`:
    - `forward_step`: `25.15%`
    - `backward_step`: `23.45%`
    - `optimizer_step`: `5.98%`

Interpretation: 在当前Qwen3 smoke配置下，alltoall的scaling-vs-realistic一致性明显优于allgather；allgather forward/backward残差仍然系统性偏高。

#### 3.5 Memory/OOM risk (A/B)

- Logs checked:
  - `logs/compare_trace/qwen_alltoall_dist_full8.log`
  - `logs/compare_trace/qwen_alltoall_scale_full8.log`
  - `logs/compare_trace/qwen_allgather_dist_full8_after_comm_tracefix.log`
  - `logs/compare_trace/qwen_allgather_scale_full8_after_comm_tracefix.log`
- Result:
  - No `OOM` / `CUDA out of memory` / traceback in final reruns.
  - Distributed theoretical footprint (both dispatchers): `~18.0 GB` total per rank (same model/settings).

### 4) Conclusion (current stage)

- B-path minimal pipeline (NVTX->nsys sqlite->kernel-level compute-only compare) is implemented and validated.
- alltoall A/B is runnable and relatively stable; allgather path has been unblocked and traced, but mismatch remains large.
- Current evidence supports:
  1. **论文主口径优先使用 robust 指标**：`op_rank_median + median_of_runs`（trace口径）；
  2. kernel-level B口径作为“物理解释与交叉验证”而非单一 acceptance gate；
  3. allgather 在当前实现下不建议作为主展示配置。
