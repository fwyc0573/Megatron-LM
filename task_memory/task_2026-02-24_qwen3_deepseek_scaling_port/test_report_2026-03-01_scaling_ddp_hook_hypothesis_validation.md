## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added DDP-hook hypothesis validation report: scaling debug flag, x1 NSYS A/B (`ddp-hook on/off`), and evidence-based verdict |

## Test Report: Scaling DDP-hook Hypothesis Validation

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Test Scope

1. Validate whether backward residual is primarily caused by DDP backward-hook overhead mismatch.
2. Add a scaling debug switch to disable DDP hook accumulation path (without breaking existing DDP wrapper interfaces).
3. Compare `ddp-hook on` vs `ddp-hook off` in scaling NSYS x1 protocol.
4. Re-check whether this direction materially closes distributed/scaling backward residual.

### 2) Code Paths Modified

- `megatron/training/arguments.py`
  - Added `--scaling-disable-ddp-wrap` (debug-only; now used to disable DDP param-hook accumulation path in scaling mode).
- `megatron/core/distributed/distributed_data_parallel.py`
  - Added `disable_param_hook_accumulation` runtime switch (scaling + debug flag).
  - In `param_hook`, skip accumulation path (`param.main_grad.add_`) when the switch is enabled.
- `megatron/training/training.py`
  - Kept standard DDP wrapping path (for interface compatibility).
- `megatron/core/optimizer/__init__.py`
  - Hardened grad-buffer collection for non-DDP-compatible wrappers:
    - only collect when both `buffers` and `expert_parallel_buffers` attributes are present.
- `examples/pretrain_deepseek_v3_moe.sh`
  - Added env passthrough and validation: `SCALING_DISABLE_DDP_WRAP` (`0|1`).
- `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - Added env passthrough and validation: `SCALING_DISABLE_DDP_WRAP` (`0|1`).
- `tests/unit_tests/test_training.py`
  - Added parser tests for `--scaling-disable-ddp-wrap` default/custom.

### 3) Validation Criteria

1. New debug flag and scripts parse/validate correctly.
2. Scaling run remains runnable with debug flag enabled.
3. Backward kernel profile under `ddp-hook off` shows measurable change if hooks are active.
4. Determine whether the change magnitude is sufficient to explain major backward residual.

### 4) Reproducible Commands

```bash
# Unit tests (parser + related perf/profiler suites)
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29622 PYTHONPATH=$(pwd) \
python -m pytest \
  tests/unit_tests/test_training.py::TestTraining::test_scaling_disable_ddp_wrap_default \
  tests/unit_tests/test_training.py::TestTraining::test_scaling_disable_ddp_wrap_enabled \
  tests/unit_tests/tensor_parallel/test_mappings_moe_api.py \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py \
  tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py -q

# Static checks
python -m py_compile \
  megatron/training/arguments.py \
  megatron/training/training.py \
  megatron/core/distributed/distributed_data_parallel.py \
  megatron/core/optimizer/__init__.py \
  tests/unit_tests/test_training.py
bash -n examples/pretrain_deepseek_v3_moe.sh
bash -n examples/pretrain_qwen3_30b_a3b_moe.sh

# Scaling x1 NSYS capture (DDP hook accumulation disabled)
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_ddp_off \
  bash -lc "MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=1024 TRAIN_ITERS=3 TRACE_START=1 TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PHASE=1 TRACE_KERNEL_BOUNDARY_SYNC_MODE=event TRACE_SUBOP_SYNC_MODE=global SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 SCALING_COMM_ADJACENT_COPY_ITERS=0 SCALING_DISABLE_DDP_WRAP=1 MASTER_PORT=6540 bash examples/pretrain_deepseek_v3_moe.sh"

nsys export --type sqlite --force-overwrite=true \
  --output task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_ddp_off \
  task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_ddp_off.nsys-rep

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_ddp_off \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_ddp_off_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_ddp_off_breakdown.md

# Dist-vs-scaling compare under DDP-hook-off scaling
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_postfix_breakdown.json \
  --scaling-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_ddp_off_breakdown.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --compute-metric pure_primary_union \
  --kernel-scope shared \
  --shared-kernel-source primary_stream \
  --require-low-contamination-pct 1 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_compare_ddp_off_pure_primary_union_shared.log
```

### 5) Results and Evidence

#### 5.1 Test pass/fail

| Check | Result | Evidence |
|---|---|---|
| Unit tests | PASS | `31 passed, 4 warnings` |
| Static checks | PASS | `py_compile` + `bash -n` exit code 0 |
| Scaling NSYS capture (`ddp-hook off`) | PASS | full fake-rank loop finished, `.nsys-rep` generated |
| Analyzer parse + contamination | PASS | `phase_window_parents=48`, contamination `0.00%` |
| Compare gate (`<=5%`) | FAIL | backward/forward remain above threshold |

#### 5.2 Scaling `ddp-hook on` vs `ddp-hook off` (stage1 backward steady, ranks 4..7)

Source JSON:

- on: `deepseek_phase_sanity_scaling_postfix_breakdown.json`
- off: `deepseek_phase_sanity_scaling_ddp_off_breakdown.json`

Aggregated across 12 events (`rank4..7 x 3 iterations`):

- `compute_pure_primary_union_ms`: `78.899 -> 71.230` (**-9.72%**)
- `kernel_count`: `7576 -> 6988` (**-7.76%**)

Interpretation:

- scaling path does include DDP hook-related compute overhead;
- but magnitude is moderate (single-digit / low double-digit), not enough to explain large residuals alone.

#### 5.3 Kernel-family delta (`on - off`) in scaling stage1 backward

Top reduction:

- `CUDAFunctor_add<float>` family: `-7.597 ms` (dominant)

Attention kernel sensitivity in this A/B:

- `fmha_cutlassB` delta: `-0.007 ms` (negligible)

Interpretation:

- disabling hook path mostly removes elementwise add kernels;
- no meaningful `fmha_cutlassB` speedup observed in this x1 A/B, so the “DDP hook causes major fmha slowdown” claim is not supported by this run.

#### 5.4 Cross-check on historical high-gap set (`round68 run5`, `seq8192`)

From existing kernel delta decomposition (`dist - scale`, stage1 backward ranks4..7):

- `fmha_cutlassB` delta: `149.201 ms`
- `CUDAFunctor_add*` total delta: `7.946 ms`

Interpretation:

- even in high-gap dataset, add-kernel delta is much smaller than fmha delta.

#### 5.5 Dist-vs-scaling compare impact

`deepseek_phase_sanity_compare_ddp_off_pure_primary_union_shared.log` summary:

- `backward_step` rank-median diff: `20.08%` (FAIL)
- previous `ddp-hook on` reference (`postfix`): `19.72%`

Interpretation:

- turning off scaling DDP-hook accumulation does **not** improve backward gap in this x1 protocol; it slightly worsens it.

### 6) Failure Diagnosis During Execution

During early attempt, directly skipping DDP wrapper caused runtime failures:

1. missing `expert_parallel_buffers` in optimizer setup;
2. missing `zero_grad_buffer` in training loop.

Resolution:

- switched to a hook-path disable strategy while preserving DDP wrapper interfaces;
- hardened optimizer buffer-collection condition for safer non-DDP compatibility.

### 7) Verdict

1. The hypothesis “scaling backward effectively skips DDP hook behavior” is **not supported**.
2. DDP hook accumulation contributes measurable overhead in scaling backward, but it is **not the dominant residual source**.
3. Current evidence still points to non-comm dominant residual components (especially attention family in historical high-gap data).
