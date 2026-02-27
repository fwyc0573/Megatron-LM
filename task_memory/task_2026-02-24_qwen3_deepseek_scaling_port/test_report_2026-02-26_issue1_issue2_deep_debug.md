## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-26 | Added Issue1/Issue2 deep debugging report (NSYS compute-only discrepancy + profiler before/after validation) |

## Test Report: Qwen3 Issue1/Issue2 Deep Debug (NSYS vs Trace, before/after profiler)

**Date**: 2026-02-26  
**Environment**: conda `myenv_yc`, Python 3.9, CUDA 12.1, 8x A100 80GB (shared cluster)

### 1) Test Script Information

#### 1.1 Source files reviewed

- `megatron/profiler/cmd.py`
- `megatron/core/tensor_parallel/mappings.py`
- `megatron/core/transformer/moe/token_dispatcher.py`
- `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`
- `tests/performance/compare_qwen_nsys_compute_only.py`
- `tests/performance/compare_qwen_trace_comp.py`

#### 1.2 Reproducible analysis commands

```bash
# Before/after compare (trace comp) for Issue2
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir ../Megatron-LM_wt_c3a77a33/realistic_trace/pp4_tp1_exp2_expn128_dp2_nl48_hs2048_sl2048 \
  --scaling-dir ../Megatron-LM_wt_c3a77a33/profiler_log/pp4_tp1_ep2_expn128_dp2_nl48_hs2048_sl2048 \
  --ranks 0,7 --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260226180100 \
  --report-path logs/compare_trace/qwen_issue2_before_c3a77a33_full_pp4_seq2048_mbs1_event_rank0_7.log

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp4_tp1_exp2_expn128_dp2_nl48_hs2048_sl2048 \
  --scaling-dir profiler_log/pp4_tp1_ep2_expn128_dp2_nl48_hs2048_sl2048 \
  --ranks 0,7 --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260226175600 \
  --report-path logs/compare_trace/qwen_issue2_full_pp4_seq2048_mbs1_event_rank0_7.log

# NSYS JSON compare (compute-only)
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json logs/nsys_kernel_gt/qwen_alltoall_dist_trace4_kernel_breakdown.json \
  --scaling-json logs/nsys_kernel_gt/qwen_alltoall_scale_trace4_kernel_breakdown.json \
  --ranks 0,7 --ops forward_step,optimizer_step \
  --report-path logs/compare_nsys/qwen_alltoall_nsys_compute_only_fwd_optim.log

python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json logs/nsys_kernel_gt/qwen_allgather_dist_trace4_kernel_breakdown.json \
  --scaling-json logs/nsys_kernel_gt/qwen_allgather_scale_trace4_kernel_breakdown.json \
  --ranks 0,7 --ops forward_step,optimizer_step \
  --report-path logs/compare_nsys/qwen_allgather_nsys_compute_only_fwd_optim.log
```

#### 1.3 New deep-dive evidence artifacts

- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_issue1_nsys_kernel_overlap_deep_debug.log`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_issue1_nsys_ratio_overview.log`
- `logs/compare_trace/qwen_issue2_before_c3a77a33_full_pp4_seq2048_mbs1_event_rank0_7.stdout`
- `logs/compare_trace/qwen_issue2_full_pp4_seq2048_mbs1_event_rank0_7.stdout`

### 2) Validation Criteria

1. **Issue1 (NSYS discrepancy):**
   - Verify whether distributed/scaling execute identical compute kernels and stream patterns.
   - Quantify kernel-set overlap and stream-count differences for representative ops.
   - Check whether current NSYS aggregation can amplify discrepancy under multi-stream overlap.
2. **Issue2 (profiler code change suspicion):**
   - Compare before (`c3a77a33`) vs after (`0094c239`) using same workload/config.
   - Determine whether profiler modifications significantly increase comp error.
3. **Large-workload check:**
   - Attempt to push per-op fwd/bwd toward >=300ms; if blocked, provide root-cause evidence.

### 3) Test Results and Evidence

#### 3.1 Large-workload attempts (Issue2 Step1)

Multiple 8-GPU full-profile attempts with larger sequence/batch were run. Key blockers:

- OOM under several configs:
  - `task_memory/.../logs/qwen_issue2_large_dist_pp1_tp1_ep2_dp4_seq4096_mbs1_event.log`
  - `task_memory/.../logs/qwen_issue2_large_dist_pp2_tp1_ep2_dp2_seq4096_mbs1_event.log`
  - `task_memory/.../logs/qwen_issue2_large_dist_pp4_tp1_ep2_dp1_seq4096_mbs1_event.log`
- Shape mismatch under `pp1,tp2,ep2,seq4096`:
  - `RuntimeError: expected index [4096, 8] to be smaller than self [2048, 128]`
  - log: `task_memory/.../logs/qwen_issue2_large_dist_pp1_tp2_ep2_dp2_seq4096_mbs1_event.log`

**Runnable upper bound (stable)** in this cluster window: `full, seq=2048, mbs=1, TP1 PP4 EP2 DP2`.

#### 3.2 Before/after profiler comparison (Issue2 Step2)

Using identical compare setup (rank0/rank7, `forward/backward/optimizer`, `trace_subop_sync_mode=event`):

| Version | forward op-rank-median | backward op-rank-median | optimizer op-rank-median |
|---|---:|---:|---:|
| before (`c3a77a33`) | 13.665% | 2.27% | 13.715% |
| after (`0094c239`) | 13.74% | 2.995% | 13.30% |

Evidence:

- before: `logs/compare_trace/qwen_issue2_before_c3a77a33_full_pp4_seq2048_mbs1_event_rank0_7.stdout`
- after: `logs/compare_trace/qwen_issue2_full_pp4_seq2048_mbs1_event_rank0_7.stdout`

**Conclusion**: no significant regression attributable to profiler modifications in `0094c239`.

#### 3.3 Issue1 root-cause evidence (NSYS compute-only discrepancy)

##### A) Kernel set/order is **not always identical** between distributed and scaling

From `qwen_issue1_nsys_kernel_overlap_deep_debug.log`:

- alltoall, rank7, backward:
  - compute diff = **79.44%**
  - top30 compute-kernel Jaccard = **0.073** (inter=3, union=41)
  - streams: dist=`[7,40,333,334,335,336]`, scale=`[7]`
- allgather, rank0, forward:
  - compute diff = **23.07%**
  - top30 Jaccard = **0.935** (kernel family mostly similar) but per-kernel time weight differs strongly.

This directly refutes “both modes always execute the same compute kernels in same order”.

##### B) Multi-stream behavior differs and affects NSYS-aggregated compute-only metric

From the same log and `qwen_issue1_nsys_ratio_overview.log`:

- distributed often uses more streams than scaling for the same op (especially backward).
- `total_kernel_ms / wall_ms` in distributed backward can exceed 1.0:
  - alltoall dist rank0 backward: **1.036**
  - alltoall dist rank7 backward: **1.012**

This indicates overlap accumulation across streams in the current extraction method.

##### C) Current NSYS extractor can amplify discrepancy by design

`tests/performance/analyze_nsys_cmd_kernel_breakdown.py` computes per-window kernel overlap by summing each kernel’s overlapped duration (`summarize_nvtx_ranges`, overlap accumulation loop), which is sensitive to stream concurrency and can double-count timeline occupancy under overlap-heavy distributed windows.

#### 3.4 Additional context on realistic in-file variance

For the full-profile stable run (`seq2048, mbs1, pp4,tp1,ep2,dp2`), realistic per-file variance is low for fwd/bwd, but not for optimizer:

- `task_memory/.../logs/qwen_pp4tp1_8gpu_seq2048_mbs1_trace4_event_full_single_vs_avg_analysis.log`
  - forward avg CV ≈ **1.37%**
  - backward avg CV ≈ **0.33%**
  - optimizer avg CV ≈ **7.07%**

So “distributed mean ≈ single-step” holds better for forward/backward than optimizer in this workload.

### 4) Root-Cause Determination (Issue2 Step3)

**Decision: Option B** (profiler modifications are logically correct; dominant error source is elsewhere).

- No evidence that `0094c239` introduced a logic bug that materially amplifies comp error.
- Main contributors are:
  1. distributed vs scaling execution-path divergence for some ops/stages,
  2. stream-level concurrency differences,
  3. NSYS overlap-sum statistic sensitivity under multi-stream windows.

### 5) Prioritized Fix Plan (minimal-risk)

1. **Keep trace robust metric as primary acceptance/reporting metric**
   - Use `op_rank_median + median_of_runs` for paper-facing stability.
2. **Add NSYS auxiliary metric with overlap de-dup semantics**
   - Add a non-gating view for timeline-union compute time (avoid pure overlap-sum bias).
3. **Targeted fidelity alignment for problematic op/stage only**
   - Focus on alltoall rank7 backward and allgather forward windows first.
4. **Large-workload reproducibility track**
   - Keep current stable profile as baseline; attempt >=300ms windows only in dedicated non-contention GPU slots.

### 6) Final Status

- Issue1 explanation is established with concrete kernel/stream evidence.
- Issue2 before/after suspicion is not supported by data.
- Remaining work is methodology refinement (NSYS auxiliary metric) plus focused fidelity tuning, not rollback of profiler changes.
