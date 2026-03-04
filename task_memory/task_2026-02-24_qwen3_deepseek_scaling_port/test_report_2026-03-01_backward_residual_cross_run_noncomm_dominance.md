## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added cross-run validation report for backward residual source (`round68 seq8192 run1..5`) and `_AllToAll` autograd-graph parity check |

## Test Report: Backward Residual Cross-run Non-comm Dominance Validation

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1)

### 1) Test Script Information

- Kernel-family cross-run analysis:
  - Input artifacts:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run{1..5}_{dist,scaling}_kernel_breakdown.json`
  - Command:
    ```bash
    python - <<'PY'
    import json, os, statistics
    from collections import defaultdict

    base = "task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics"

    def load_rows(path):
        with open(path, "r") as f:
            return json.load(f)["event_rows"]

    def filt(rows):
        return [
            r
            for r in rows
            if r["op"] == "backward_step"
            and int(r["stage_id"]) == 1
            and r["mg_state"] == "steady"
            and int(r["rank"]) in (4, 5, 6, 7)
        ]

    def kernel_map(rows):
        acc = defaultdict(float)
        for r in rows:
            for k, v in r.get("primary_stream_compute_kernel_name_overlap_ms", {}).items():
                acc[k] += float(v)
        return acc

    for run in range(1, 6):
        dist = kernel_map(filt(load_rows(os.path.join(base, f"deepseek_round68_sl8192_run{run}_dist_kernel_breakdown.json"))))
        scale = kernel_map(filt(load_rows(os.path.join(base, f"deepseek_round68_sl8192_run{run}_scaling_kernel_breakdown.json"))))
        deltas = sorted(
            ((k, dist.get(k, 0.0) - scale.get(k, 0.0)) for k in (set(dist) | set(scale))),
            key=lambda kv: kv[1],
            reverse=True,
        )
        print(run, sum(dist.values()) - sum(scale.values()), deltas[:3])
    PY
    ```

- `_AllToAll` autograd-graph parity check:
  - Command:
    ```bash
    python - <<'PY'
    import torch
    from megatron.core.tensor_parallel import mappings

    orig_get_world_size = torch.distributed.get_world_size
    orig_all_to_all_single = torch.distributed.all_to_all_single

    def fake_get_world_size(group=None):
        return 1

    def fake_all_to_all_single(output, input_, output_split_sizes=None, input_split_sizes=None, group=None):
        output.copy_(input_)

    torch.distributed.get_world_size = fake_get_world_size
    torch.distributed.all_to_all_single = fake_all_to_all_single
    try:
        x = torch.arange(12.0, requires_grad=True).reshape(3, 4)

        def graph_types(t):
            names, q, seen = [], [t.grad_fn], set()
            while q:
                fn = q.pop(0)
                if fn is None or id(fn) in seen:
                    continue
                seen.add(id(fn))
                names.append(type(fn).__name__)
                for nxt, _ in fn.next_functions:
                    if nxt is not None:
                        q.append(nxt)
            return names

        y_dist = mappings._AllToAll.apply(object(), x, None, None, "exp", False, 0)
        y_scale = mappings._AllToAll.apply(object(), x, None, None, "exp", True, 0)
        d = graph_types((y_dist * 1.23).sum())
        s = graph_types((y_scale * 1.23).sum())
        print("dist_nodes", len(d), sorted(d))
        print("scale_nodes", len(s), sorted(s))
    finally:
        torch.distributed.get_world_size = orig_get_world_size
        torch.distributed.all_to_all_single = orig_all_to_all_single
    PY
    ```

- Regression/unit tests:
  - Command:
    ```bash
    CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29620 PYTHONPATH=$(pwd) \
    python -m pytest \
      tests/unit_tests/tensor_parallel/test_mappings_moe_api.py \
      tests/unit_tests/test_training.py::TestTraining::test_scaling_comm_adjacent_copy_iters_default \
      tests/unit_tests/test_training.py::TestTraining::test_scaling_comm_adjacent_copy_iters_custom \
      tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
      tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py \
      tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py -q
    ```

### 2) Validation Criteria

1. Validate whether backward residual source is stable across `run1..5`, not a single-run artifact.
2. Verify whether `_AllToAll` scaling/distributed branches create different autograd-node topology at minimal reproduction level.
3. Confirm existing phase/compare/scaling-emulation code paths remain test-green.

### 3) Test Results and Evidence

#### 3.1 Cross-run kernel-family evidence (`round68 seq8192`, stage1 backward steady, ranks 4..7)

- Per-run `dist-scale` gap (ms): `240.586`, `238.740`, `225.564`, `237.983`, `238.288`
- For all 5 runs, top-1 positive delta is consistently:
  - `fmha_cutlassB_bf16_aligned_...AttentionBackwardKernel...`
  - contribution range: `146.581 ms` to `153.787 ms`
- Median top deltas across runs:
  - `fmha_cutlassB...`: `150.686 ms`
  - `indexFuncLargeIndex...`: `21.246 ms`
  - `indexSelectLargeIndex...`: `9.410 ms`

Interpretation:
- backward residual has strong cross-run consistency;
- dominant source remains non-comm kernel family (`fmha_cutlassB`), consistent with previous single-run deep-dive.

#### 3.2 `_AllToAll` autograd parity evidence

Observed output:

- `dist_nodes 3 ['MulBackward0', 'SumBackward0', '_AllToAllBackward']`
- `scale_nodes 3 ['MulBackward0', 'SumBackward0', '_AllToAllBackward']`

Interpretation:
- at `_AllToAll.apply` micro-repro level, distributed/scaling autograd node counts are equal;
- “scaling bypass removes `_AllToAll` backward node” is not supported by this evidence.

#### 3.3 Regression tests

- Result: `31 passed, 3 warnings in 7.87s`
- Exit code: `0`

### 4) Conclusion

1. The residual source pattern is stable across run1..5 and dominated by non-comm kernels (`fmha_cutlassB` family), not only comm-adjacent/data-movement.
2. `_AllToAll` node-topology mismatch is not observed in micro-repro, so it should not be treated as primary root cause without further evidence.
3. Existing phase-comp-only + contamination-gate + scaling comm-adjacent emulation code paths remain stable in unit tests.
