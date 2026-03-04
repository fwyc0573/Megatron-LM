## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added deep-dive report for backward residual source attribution and scaling comm-adjacent emulation feasibility test (`--scaling-comm-adjacent-copy-iters`) |

## Test Report: Backward Residual Source Deep-dive + Scaling Comm-adjacent Emulation

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Test Scope

1. Re-check whether current backward residual is dominated by comm-adjacent/data-movement kernels.
2. Implement and validate a scaling-side emulation knob:
   - `--scaling-comm-adjacent-copy-iters` (default `0`),
   - intended to inject extra copy kernels in scaling all_to_all backward path.
3. Verify if this emulation materially improves distributed/scaling backward `comp_only` alignment.

### 2) Code Paths Modified

- `megatron/training/arguments.py`
  - Added `--scaling-comm-adjacent-copy-iters` (int, default `0`).
- `megatron/core/tensor_parallel/mappings.py`
  - Added `_emulate_comm_adjacent_copies(...)`.
  - Extended `_AllToAll.apply` signature to carry `scaling_comm_adjacent_copy_iters`.
  - Injected optional copy emulation in `_AllToAll.backward` when scaling mode is enabled.
- `examples/pretrain_deepseek_v3_moe.sh`
  - Added `SCALING_COMM_ADJACENT_COPY_ITERS` env passthrough + non-negative validation.
- `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - Added `SCALING_COMM_ADJACENT_COPY_ITERS` env passthrough + non-negative validation.
- `tests/unit_tests/test_training.py`
  - Added argument parse tests for new flag (default/custom).
- `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py`
  - Added helper tests for emulation function behavior.

### 3) Reproducible Commands

```bash
# Unit tests (with required env)
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29620 PYTHONPATH=$(pwd) \
python -m pytest \
  tests/unit_tests/tensor_parallel/test_mappings_moe_api.py \
  tests/unit_tests/test_training.py::TestTraining::test_scaling_comm_adjacent_copy_iters_default \
  tests/unit_tests/test_training.py::TestTraining::test_scaling_comm_adjacent_copy_iters_custom -q

# Static/syntax checks
python -m py_compile \
  megatron/core/tensor_parallel/mappings.py \
  megatron/training/arguments.py \
  tests/unit_tests/tensor_parallel/test_mappings_moe_api.py \
  tests/unit_tests/test_training.py
bash -n examples/pretrain_deepseek_v3_moe.sh
bash -n examples/pretrain_qwen3_30b_a3b_moe.sh

# x1 scaling capture with emulation=2
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy2 \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=1024 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=2 MASTER_PORT=6520 bash examples/pretrain_deepseek_v3_moe.sh"

nsys export --type sqlite --force-overwrite=true \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy2 \
  task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy2.nsys-rep

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy2 \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy2_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy2_breakdown.md

python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_postfix_breakdown.json \
  --scaling-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy2_breakdown.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --compute-metric pure_primary_union \
  --kernel-scope shared \
  --shared-kernel-source primary_stream \
  --require-low-contamination-pct 1 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_compare_copy2_pure_primary_union_shared.log

# x1 scaling capture with emulation=8 (sensitivity)
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy8 \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=1024 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=8 MASTER_PORT=6530 bash examples/pretrain_deepseek_v3_moe.sh"

nsys export --type sqlite --force-overwrite=true \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy8 \
  task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy8.nsys-rep

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy8 \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy8_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy8_breakdown.md

python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_postfix_breakdown.json \
  --scaling-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_copy8_breakdown.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --compute-metric pure_primary_union \
  --kernel-scope shared \
  --shared-kernel-source primary_stream \
  --require-low-contamination-pct 1 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_compare_copy8_pure_primary_union_shared.log
```

### 4) Source Attribution Evidence (Round68 seq8192 run5, stage1 backward ranks 4-7)

Using kernel-name delta decomposition on existing run5 phasecheck JSON:

- Dist total primary-union compute: `575.107 ms`
- Scale total primary-union compute: `336.820 ms`
- Gap (`dist - scale`): `238.288 ms`

Positive delta contributions:

- total positive delta: `258.332 ms`
- comm-adjacent/data-movement classified share: `70.559 ms` (`27.31%`)
- top contributor is non-comm-adjacent:
  - `fmha_cutlassB...`: `149.201 ms` (dominant)
- two largest grouped-GEMM kernels remain almost equal:
  - `ampere_bf16...256x128...64x3_nn`: delta `0.085 ms`
  - `ampere_bf16...256x128...64x3_nt`: delta `0.000 ms`

Interpretation:

- comm-adjacent/data-movement contributes part of the backward gap, but is not the dominant term in this dataset.
- residual is dominated by non-comm-adjacent kernels (especially `fmha_cutlassB` family) in backward steady stage1.

### 5) Emulation Feasibility Results (`SEQ_LEN=1024` x1 protocol)

All runs keep contamination clean (`dist/scale contamination = 0.00%`).

`op_rank_median_aux_summary` comparison:

| Setup | forward | backward | optimizer |
|---|---:|---:|---:|
| copy_iters=0 (postfix baseline) | 12.25% | 19.72% | 5.32% |
| copy_iters=2 | 12.27% | 19.56% | 4.86% |
| copy_iters=8 | 12.26% | 19.67% | 3.95% |

Observations:

1. The emulation knob is **feasible** (works end-to-end and is trace-visible).
2. On this x1 workload, backward improvement is tiny/non-monotonic (`19.72 -> 19.56 -> 19.67`), i.e. no material closure.
3. Optimizer median improves, but p75 remains unstable and full gate still FAIL.
4. Therefore, scaling-side copy emulation alone is insufficient to solve current backward residual.

### 6) Test Results Summary

| Check | Result | Evidence |
|------|--------|----------|
| new arg parse tests | PASS | `test_scaling_comm_adjacent_copy_iters_default/custom` |
| mapping helper tests | PASS | `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py` |
| static/syntax checks | PASS | `py_compile` + `bash -n` |
| scaling copy2/copy8 capture/export/analyze | PASS | artifacts generated under `logs/nsys_phase_sanity/` |
| contamination gate | PASS | all compare rows `contam=0.00%` |
| fidelity gate (`<=5%`) | FAIL | backward remains around `~19.6-19.7%` |

### 7) Conclusion

- Your PS direction (“scaling mode补齐comm-adjacent成本”) is technically feasible and now has a controllable switch.
- But current x1 evidence shows it is **not** a dominant fix for backward residual.
- Next step should prioritize deeper decomposition for non-comm-adjacent dominant kernels (especially attention backward family) under fixed protocol (`seq8192 + rank7-cap + repeat x5`).
