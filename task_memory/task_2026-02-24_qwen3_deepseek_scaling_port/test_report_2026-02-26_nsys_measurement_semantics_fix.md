## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-26 | Added NSYS measurement semantics fix report: union/primary-stream/shared-kernel compare and rank-total comp validation |

## Test Report: NSYS Measurement Semantics Fix (Qwen3)

**Date**: 2026-02-26  
**Environment**: conda `myenv_yc`, Python 3.9, CUDA 12.1, Nsight sqlite traces reused from 8-GPU Qwen3 TRACE_START=4 captures.

### 1) Test Script Information

- Modified scripts:
  - `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/performance/compare_qwen_nsys_compute_only.py`
- Updated unit tests:
  - `tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py`

### 2) Measurement/Statistics Fixes Implemented

1. **Fix overlap-sum bias (extractor side)**
   - Added union-time metrics to avoid multi-stream overlap double counting:
     - `compute_kernel_union_ms`
     - `comm_kernel_union_ms`
     - `total_kernel_union_ms`
2. **Align with scaling single-stream semantics**
   - Added dominant compute-stream metric:
     - `compute_primary_stream_union_ms`
     - `primary_compute_stream_id`
     - `compute_stream_count`
3. **Reduce distributed-only helper-kernel bias**
   - Added per-event kernel-name attribution maps:
     - `compute_kernel_name_overlap_ms`
     - `primary_stream_compute_kernel_name_overlap_ms`
   - Compare script supports `--kernel-scope shared` (intersection of kernel names between distributed/scaling).
4. **Add rank-total comp metric (user-requested)**
   - New compare section: `rank_total_comp_summary`
   - Sums all selected `forward_step/backward_step/optimizer_step` rows per rank, then computes total diff.

### 3) Reproducible Commands

```bash
# 1) Unit validation
pytest -q \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py

# 2) Re-extract JSON with new metrics
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

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite logs/nsys_kernel_gt/qwen_allgather_dist_trace4_sqlite \
  --label-prefix cmd_trace --ranks 0,7 \
  --ops forward_step,backward_step,optimizer_step \
  --json-path logs/nsys_kernel_gt/qwen_allgather_dist_trace4_kernel_breakdown.json \
  --report-path logs/nsys_kernel_gt/qwen_allgather_dist_trace4_kernel_breakdown.md

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite logs/nsys_kernel_gt/qwen_allgather_scale_trace4_sqlite \
  --label-prefix cmd_trace --ranks 0,7 \
  --ops forward_step,backward_step,optimizer_step \
  --json-path logs/nsys_kernel_gt/qwen_allgather_scale_trace4_kernel_breakdown.json \
  --report-path logs/nsys_kernel_gt/qwen_allgather_scale_trace4_kernel_breakdown.md

# 3) Compare with recommended semantics
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json logs/nsys_kernel_gt/qwen_alltoall_dist_trace4_kernel_breakdown.json \
  --scaling-json logs/nsys_kernel_gt/qwen_alltoall_scale_trace4_kernel_breakdown.json \
  --ranks 0,7 --ops forward_step,backward_step,optimizer_step \
  --compute-metric primary_stream_union \
  --kernel-scope shared --shared-kernel-source primary_stream \
  --dist-reducer trimmed_mean --scale-reducer median --trim-ratio 0.2 \
  --report-path logs/compare_nsys/qwen_alltoall_nsys_compute_only_shared_primary_total_fbo.log

python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json logs/nsys_kernel_gt/qwen_allgather_dist_trace4_kernel_breakdown.json \
  --scaling-json logs/nsys_kernel_gt/qwen_allgather_scale_trace4_kernel_breakdown.json \
  --ranks 0,7 --ops forward_step,backward_step,optimizer_step \
  --compute-metric primary_stream_union \
  --kernel-scope shared --shared-kernel-source primary_stream \
  --dist-reducer trimmed_mean --scale-reducer median --trim-ratio 0.2 \
  --report-path logs/compare_nsys/qwen_allgather_nsys_compute_only_shared_primary_total_fbo.log
```

### 4) Validation Criteria

1. extractor provides overlap + union + primary-stream metrics together;
2. compare supports shared-kernel filtering and rank-total aggregation;
3. metric-mode comparison should show reduced gap versus legacy overlap-sum in at least part of problematic ranks.

### 5) Test Results and Evidence

#### 5.1 Unit tests

- `12 passed`:
  - `tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py`

#### 5.2 Mode comparison summary (rank-total comp error)

Evidence source:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_issue1_metric_mode_comparison_summary.log`

##### alltoall (rank-total over fwd+bwd+optimizer)

| Metric mode | rank0 diff | rank7 diff |
|---|---:|---:|
| overlap_all (legacy) | 40.46% | 61.88% |
| union_all | 38.74% | 60.13% |
| primary_all | 35.70% | 56.89% |
| **shared_primary (new)** | **35.04%** | **14.27%** |

##### allgather (rank-total over fwd+bwd+optimizer)

| Metric mode | rank0 diff | rank7 diff |
|---|---:|---:|
| overlap_all (legacy) | 22.09% | 26.00% |
| union_all | 22.48% | 26.00% |
| primary_all | 21.54% | 26.00% |
| **shared_primary (new)** | **20.31%** | **18.62%** |

#### 5.3 Op-level robust summary change (op-rank-median)

- alltoall:
  - backward: `70.98% -> 39.68%` (legacy overlap_all -> shared_primary)
  - optimizer: `19.38% -> 8.84%`
  - forward: `7.24% -> 6.19%`
- allgather:
  - backward: `22.32% -> 13.24%`
  - optimizer: `21.57% -> 5.12%`
  - forward: remains high (`~26%`), indicating residual path-fidelity mismatch not solved by statistics alone.

### 6) Conclusion

- 我认同“overlap-sum 会把多-stream并发差异放大成计算差异”的判断；该问题已通过 union/primary-stream/shared-kernel 三层策略在统计口径上得到修复与缓解。
- 新口径下，误差在多个关键项显著下降（尤其 alltoall rank7、allgather backward/optimizer）。
- 但残余高误差（特别 allgather forward、alltoall rank0 backward）仍存在，说明还有真实执行路径差异（不仅是统计偏差）。

### 7) Current Recommended Metric for Paper

在 NSYS 口径中，建议使用：

- `compute-metric=primary_stream_union`
- `kernel-scope=shared`
- `shared-kernel-source=primary_stream`
- 报告 `op_rank_median` + `rank_total_comp_summary`

并将 legacy overlap-sum 作为附录对照，不作为主结论口径。
