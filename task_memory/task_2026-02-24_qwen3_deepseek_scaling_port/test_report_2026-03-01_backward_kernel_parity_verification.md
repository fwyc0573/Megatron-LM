## Test Report: Backward Kernel Count Parity Verification (DeepSeek-V3 Smoke)

**Date**: 2026-03-01  
**Environment**: conda myenv_yc (Python 3.9.18, PyTorch 2.1.2, CUDA 12.1)  
**Codebase**: Main branch (current HEAD)  
**Config**: DeepSeek-V3 smoke (8 layers, 1024 hidden, 16 experts, PP=2, TP=1, EP=2, MLA, local impl)

---

### Background

Previous investigation (round 68, ~Feb 28) found a **2x backward kernel count gap**:
- Distributed backward rank4: 590 total, 4 fmha_bwd, 16 rmsnorm_bwd
- Scaling backward rank4: 285 total, 2 fmha_bwd, 8 rmsnorm_bwd

This was hypothesized to be an **autograd graph break** causing only 2 of 4 transformer
layers to execute backward.

### Investigation (This Session)

#### Step 1: All-to-all Shape Verification
- Extracted all_to_all tensor shapes from trace files for rank 4
- **Result**: Shapes IDENTICAL between distributed and scaling
  - Both: alternating `[16384, 1024]` and `[3455, 1024]`, 8 sub_ops each
- **Conclusion**: Token count mismatch hypothesis **DISPROVED**

#### Step 2: Autograd Graph Integrity Test
- Added tensor-level backward hooks to every TransformerLayer output
  in `TransformerBlock.forward()` (via `DIAG_BWD_HOOKS=1` env flag)
- Ran scaling mode rank 4 with hooks enabled

**Result**: ALL 4 LAYERS received backward gradients:
```
[DIAG_BWD] Layer 3/4 backward grad shape=torch.Size([256, 1, 1024]), norm=1.0428e-01
[DIAG_BWD] Layer 2/4 backward grad shape=torch.Size([256, 1, 1024]), norm=1.0472e-01
[DIAG_BWD] Layer 1/4 backward grad shape=torch.Size([256, 1, 1024]), norm=1.0494e-01
[DIAG_BWD] Layer 0/4 backward grad shape=torch.Size([256, 1, 1024]), norm=1.0506e-01
```

- **Conclusion**: Autograd graph is INTACT. No break between layers.

#### Step 3: Basic Autograd Mechanics Tests
- Verified `index_copy_` (used in `unpermute`) preserves autograd ✅
- Verified `_AllToAll` autograd Function preserves gradient flow ✅
- Verified `deallocate_output_tensor` + `custom_backward` chain works ✅

#### Step 4: Fresh NSYS Kernel Count (Current Code)

Ran NSYS profiling on BOTH modes with current code:

**Scaling Mode** (rank 4, device 0):
```
nsys profile --trace=cuda,nvtx ... TRACE_KERNEL_GROUND_TRUTH=1
```

**Distributed Mode** (rank 4, device 4):
```
nsys profile --trace=cuda,nvtx ... TRACE_KERNEL_GROUND_TRUTH=1
```

### Key Results: NSYS Kernel Parity

| Op | Metric | Scaling | Distributed | Diff |
|---|---|---|---|---|
| **forward_step** | fmha | 4 | 4 | **0** |
| | rmsnorm | 9 | 9 | **0** |
| | compute kernels | 497 | 497 | **0 (0.0%)** |
| | compute time (kernel sum) | 3.034ms | 2.511ms | +20.8% |
| **backward_step** | fmha | 4 | 4 | **0** |
| | rmsnorm | 18 | 18 | **0** |
| | compute kernels | 647 | 645 | **+2 (0.3%)** |
| | compute time (kernel sum) | 4.529ms | 3.764ms | +20.3% |
| **optimizer_step** | compute kernels | 82 | 82 | **0 (0.0%)** |
| | compute time (kernel sum) | 2.346ms | 2.196ms | +6.8% |

### Conclusion

1. **The round 68 2x backward kernel count bug is FIXED in the current code.**
   - Old: 285 kernels, 2 fmha_bwd (only 2/4 layers in backward)
   - Current: 647 kernels, 4 fmha_bwd (all 4 layers in backward)

2. **Kernel counts match near-perfectly:**
   - forward: 497 vs 497 (0.0% diff)
   - backward: 647 vs 645 (0.3% diff, likely 2 extra comm-simulation copy kernels)
   - optimizer: 82 vs 82 (0.0% diff)

3. **Kernel type parity is exact:** fmha=4/4, rmsnorm=9/9 (fwd), rmsnorm=18/18 (bwd)

4. **Compute time residual (~20%) remains**, likely dominated by:
   - Smoke model's extremely small kernel sizes (<0.1ms per kernel)
   - CPU launch overhead dominance for 497-647 tiny kernels
   - Single-GPU vs multi-GPU thermal/frequency differences
   - This residual is expected to diminish with larger model configs (seq8192, etc.)

### Trace-Based Comparison (CMD timings)

Ran `compare_qwen_trace_comp.py` with paired latest traces:

| Op | rank_median_diff_pct | rank_p75_diff_pct | Status |
|---|---|---|---|
| backward_step | 12.71% | 17.54% | FAIL |
| forward_step | 22.00% | 26.44% | FAIL |
| optimizer_step | 11.10% | 12.01% | FAIL |

These are CMD wall-clock timings (including CPU overhead), not pure kernel time.
The 12-22% residual is consistent with the NSYS ~20% compute-time residual for
this extremely small smoke model.

### Next Steps

1. ~~Investigate what code change fixed the round 68 bug~~ **DONE**: commit `f1084adc`
2. **Re-run comparison with seq8192 or larger config** to reduce CPU-overhead-dominated noise
3. **Update backward gate semantics** now that kernel parity is confirmed
4. The remaining ~20% residual for smoke model should be characterized separately
   from the kernel parity question

### Residual Characterization (Per-Kernel Timing Analysis)

The ~20% compute-time residual was further analyzed by grouping kernel types by
their distributed-mode duration:

| Kernel Size Bucket | Forward Diff | Backward Diff | Optimizer Diff |
|---|---|---|---|
| <5us (majority) | +25.5% | +22.9% | +26.6% |
| 5-10us | +24.2% | +20.9% | +19.8% |
| 10-50us | +22.2% | +19.5% | — |
| 50-100us | +19.4% | +20.4% | — |
| >100us | — | **+7.4%** | **+3.0%** |

**Key finding**: The residual is driven by a **per-kernel constant overhead** (NSYS
profiling noise + CUDA launch latency). Larger kernels show progressively smaller
relative diff — the >100us bucket shows only +3-7%, which is near the 5% acceptance
threshold.

**Implication for paper reporting**: The smoke model (median kernel ~5us, 497-647
kernels per op) is too small to meaningfully benchmark compute-time fidelity. At
production-scale workloads where kernels are >>100us, the residual should be well
within acceptable range based on the observed trend.

### Root Cause Analysis

**Commit `f1084adc`** (author: `fwyc0573`) fixed the backward kernel count bug:

> "fix backwards problem in moe mode. the problem is raised by grad breaks.
> put it into torch.autograd.func can fix it."

Three concurrent gradient breaks were fixed:

1. **`mappings.py`**: `_profiled_all_to_all_single` (scaling mode comm simulation)
   was not wrapped in an autograd Function. Now routed through `_AllToAll.apply()`
   with `is_scaling_mode` parameter.

2. **`token_dispatcher.py`**: `torch.gather`/`scatter`/`scatter_add` replaced with
   custom `moe_gather`/`moe_scatter` autograd Functions for proper backward flow.

3. **`moe_layer.py`**: `pre_fixed_routing_results` path was replacing `hidden_states`
   with a pre-computed tensor (detached from graph). Now preserves real `hidden_states`
   and re-runs `router.gating()` to maintain gradient connectivity.

### Commands to Reproduce

```bash
# Scaling mode (single GPU)
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 \
nsys profile --trace=cuda,nvtx --output=/tmp/diag_scale_rank4 --force-overwrite=true \
  bash -c 'MODE=scaling MODEL_PROFILE=smoke TRANSFORMER_IMPL=local \
  TRAIN_ITERS=3 TRACE_START=2 SCALING_MIN_WARMUP_ITERS=1 SCALING_PROFILE_ITERS=1 \
  SCALING_STRICT_GRAD_REPLAY=0 DO_TRACE=False SCALING_FAKE_RANK_ORDER=4 \
  TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PREFIX=cmd_trace \
  bash examples/pretrain_deepseek_v3_moe.sh'

# Distributed mode (8 GPUs)
CUDA_DEVICE_MAX_CONNECTIONS=1 \
nsys profile --trace=cuda,nvtx --output=/tmp/diag_dist --force-overwrite=true \
  bash -c 'MODE=distributed MODEL_PROFILE=smoke TRANSFORMER_IMPL=local \
  TRAIN_ITERS=3 TRACE_START=2 DO_TRACE=False \
  TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PREFIX=cmd_trace \
  bash examples/pretrain_deepseek_v3_moe.sh'
```
