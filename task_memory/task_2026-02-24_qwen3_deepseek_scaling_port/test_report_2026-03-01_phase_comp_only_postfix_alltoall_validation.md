## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added postfix validation for all_to_all comm-adjacent attribution fix (unit + NSYS analyze/compare + pre/post residual check) |

## Test Report: Postfix Validation for all_to_all Comm-adjacent Attribution

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Test Script Information

- Modified code paths (this postfix check):
  - `megatron/core/tensor_parallel/mappings.py`
  - `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py`
- Existing capture artifacts used for postfix analyze/compare:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_postfix.nsys-rep`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_postfix.nsys-rep`

### 2) Validation Criteria

1. all_to_all scaling bypass no longer returns input alias directly in equal/none-split branches.
2. Unit tests cover new materialization behavior and non-scaling contiguous input contract.
3. Postfix NSYS analyzer can parse phase windows and keep contamination near zero.
4. Postfix compare uses `pure_primary_union` + contamination gate and reports residual shape.
5. Pre/postfix residual trend is explicit (detect improvement/no-improvement).
6. `_AllToAll.apply` autograd graph parity is checked directly for `is_scaling_mode=False/True`.

### 3) Reproducible Commands

```bash
# Unit tests for mappings MoE API behavior
python -m pytest tests/unit_tests/tensor_parallel/test_mappings_moe_api.py -q

# Static syntax check
python -m py_compile megatron/core/tensor_parallel/mappings.py tests/unit_tests/tensor_parallel/test_mappings_moe_api.py

# Analyze postfix sqlite (phase-aware)
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_postfix \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_postfix_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_postfix_breakdown.md

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_postfix \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_postfix_breakdown.json \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_postfix_breakdown.md

# Compare postfix (official candidate metric + contamination gate)
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_dist_postfix_breakdown.json \
  --scaling-json task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_scaling_postfix_breakdown.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --compute-metric pure_primary_union \
  --kernel-scope shared \
  --shared-kernel-source primary_stream \
  --require-low-contamination-pct 1 \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_sanity/deepseek_phase_sanity_compare_postfix_pure_primary_union_shared.log

# Local autograd graph parity micro-repro (monkeypatched all_to_all_single)
python - <<'PY'
import torch
from megatron.core.tensor_parallel.mappings import _AllToAll

orig_get_world_size = torch.distributed.get_world_size
orig_all_to_all_single = torch.distributed.all_to_all_single

def fake_get_world_size(group=None):
    return 2

def fake_all_to_all_single(output, input_, output_split_sizes=None, input_split_sizes=None, group=None):
    output.copy_(input_)

def count_nodes(grad_fn):
    stack, seen, names = [grad_fn], set(), set()
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        names.add(type(fn).__name__)
        for nxt, _ in fn.next_functions:
            if nxt is not None:
                stack.append(nxt)
    return len(seen), sorted(names)

torch.distributed.get_world_size = fake_get_world_size
torch.distributed.all_to_all_single = fake_all_to_all_single

for mode in (False, True):
    x = torch.randn(8, 4, device='cuda', requires_grad=True)
    y = _AllToAll.apply(object(), x, None, None, 'exp', mode)
    loss = (y * y).sum()
    print(mode, count_nodes(loss.grad_fn))
    loss.backward()

torch.distributed.get_world_size = orig_get_world_size
torch.distributed.all_to_all_single = orig_all_to_all_single
PY
```

### 4) Results and Evidence

| Check | Result | Evidence |
|------|--------|----------|
| mappings unit tests | PASS | `5 passed` |
| mappings static syntax check | PASS | `py_compile` exit code 0 |
| postfix distributed analyzer | PASS | `phase_window_parents=48`, `event_rows=72`, contamination `0.00%` |
| postfix scaling analyzer | PASS | `phase_window_parents=48`, `event_rows=72`, contamination `0.00%` |
| postfix contamination gate (`<=1%`) | PASS | compare rows show `dist_contam_pct=0.00`, `scale_contam_pct=0.00` |
| postfix fidelity gate (`<=5%`) | FAIL | op-rank median: `forward=12.25%`, `backward=19.72%`, `optimizer=5.32%` |
| `_AllToAll.apply` autograd graph parity | PASS | `is_scaling_mode=False/True` both `node_count=3`, both include `_AllToAllBackward` |

Pre/post residual delta (same x1 sanity protocol, `pure_primary_union + shared(primary_stream)`):

| Metric | Pre (dist_rerun) | Postfix | Delta |
|---|---:|---:|---:|
| forward rank-median diff | 10.25% | 12.25% | +2.00pp |
| backward rank-median diff | 17.97% | 19.72% | +1.75pp |
| optimizer rank-median diff | 5.75% | 5.32% | -0.43pp |

Stage1 backward focus (`rank=4..7`, `mg_state=steady`, event-row pair median):

- pre: `21.62%`
- postfix: `21.73%`
- delta: `+0.11pp`

### 5) Interpretation

- The postfix change is functionally correct and test-covered (no alias return in scaling all_to_all bypass, contiguous-in-wrapper for non-scaling path).
- Phase-level purity remains intact (postfix contamination still `0.00%`), so no regression in semantic path.
- Residual gap is not materially reduced in this x1 sanity check, so this postfix is **not** the dominant fix for backward residual.
- Next decisive validation remains fixed-protocol `seq8192 + rank7-cap + repeat-x5` under phase labels.

### 6) Exit Codes

- `pytest`: 0
- `py_compile`: 0
- analyzer commands: 0
- compare command: 1 (threshold gate fail, expected behavior for gating mode)
