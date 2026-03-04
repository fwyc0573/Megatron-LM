# GPT-175B TP Scaling Simulation — Root Cause Analysis Report

## Modification History

| Date       | Summary of Changes                                      |
|------------|----------------------------------------------------------|
| 2026-03-04 | Initial root cause analysis report created               |

---

## Executive Summary

The GPT-175B TP scaling simulation on A800 GPUs (1024 GPUs, PP=8) exhibits **anomalous inverse scaling**: iteration time increases dramatically with higher TP (34.4s → 60.5s → 106.2s for TP8/16/32), instead of the expected decrease. This report identifies **4 root causes** ranked by impact, with code-level evidence.

---

## Simulation Configurations

| Config | TP | PP | DP | Micro-batches | Expected Behavior |
|--------|----|----|----|--------------|--------------------|
| TP8    | 8  | 8  | 16 | 16           | Baseline           |
| TP16   | 16 | 8  | 8  | 32           | ~1.7x faster comp per microbatch |
| TP32   | 32 | 8  | 4  | 64           | ~3.3x faster comp per microbatch |

---

## 🔴 BUG #1 [CRITICAL]: Profiled Computation Time Does NOT Scale with TP

### Symptom

Per-microbatch forward_step computation time is **virtually identical** across all TP configurations:

| Config | Parameters/GPU | Forward Duration (raw) | Forward sub_comp total | Expected Duration |
|--------|---------------|----------------------|----------------------|------------------|
| TP8    | 2.82B         | 871.03 ms            | 663.39 ms            | baseline         |
| TP16   | 1.42B (-50%)  | 958.53 ms (+10%!)    | 761.23 ms (+15%!)    | ~435 ms (-50%)   |
| TP32   | 0.73B (-74%)  | 873.24 ms (+0.3%)    | 673.49 ms (+1.5%)    | ~218 ms (-75%)   |

### Root Cause

The `fake_tp` mechanism correctly partitions model weights (verified by parameter counts), **but single-GPU GEMM execution time does not decrease proportionally** for smaller matrix partitions. This is because:

1. **GPU underutilization**: On a single A800 GPU, a GEMM like `[2048, 12288] × [12288, 384]` (TP32 partition) runs almost as fast as `[2048, 12288] × [12288, 1536]` (TP8 partition) because both are memory-bandwidth-bound, not compute-bound at these dimensions.
2. **Fixed overhead dominance**: Kernel launch overhead, memory allocation, and synchronization dominate over actual FLOPS for these small matrix sizes.
3. **The simulation engine trusts raw profiled values with zero correction/scaling**.

### Evidence: Parameter Counts (from profiling logs)

```
tp8_pp8_dp16:  > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 2822928384
tp16_pp8_dp8:  > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 1424489472
tp32_pp8_dp4:  > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 725270016
```

### Evidence: sub_comp_2 (Dominant Computation Block — 90%+ of Forward)

```
TP8:  sub_comp_2 = 620.79 ms
TP16: sub_comp_2 = 703.73 ms (+13.4% — LARGER than TP8!)
TP32: sub_comp_2 = 635.25 ms (+2.3%)
```

### Code Path

1. **Profiling**: Single-GPU profiling with `fake_tp={8,16,32}` and `tensor_model_parallel_size=1`
   - `megatron/core/tensor_parallel/layers.py`: `ColumnParallelLinear.__init__()` — `world_size = config.fake_tp`, `output_size_per_partition = divide(output_size, world_size)` ✅ (correctly partitions weights)
   
2. **Database Profile Loading**: `megatron-sim-engine/src/core/simu_engine.py`
   - `process_mg_profile_files()` (lines 2503-2625): Reads raw durations from database_profile logs
   - `extract_sub_operations()` (lines 3743-3877): Computes `sub_comp` durations from **timestamp differences**: `current_duration = current_finish_time - current_start_time`
   - These raw profiled timestamps reflect actual single-GPU execution, which **doesn't scale with matrix partition size**

3. **Simulation**: `add_sub_ops_according_to_profile_dict()` (lines 3879-3937): Deep-copies sub_ops_list from database and assigns to each microbatch **without any TP-scaling correction**

### Impact Calculation

With computation not scaling but microbatches increasing linearly:
```
TP8:  16 batches × (663 + 685) ms ≈ 21,568 ms
TP16: 32 batches × (761 + 650) ms ≈ 45,152 ms (2.1x)
TP32: 64 batches × (673 + 593) ms ≈ 81,024 ms (3.8x)
```

This is the **primary driver** of the anomalous scaling behavior.

---

## 🟡 BUG #2 [HIGH]: DP Allreduce NOT Overlapped with Backward Computation

### Symptom

`dp_allreduce` (gradient synchronization) appears as a **single monolithic operation** that starts only **after all backward computation completes**. In real Megatron-LM, dp_allreduce uses bucketed allreduce that overlaps with backward computation.

### Evidence (from simulation log, TP8 rank 0)

```
last_operation_time-> 34000.37      # All backward+TP_allreduce done
dp_allreduce wrank_id=0
  join_time=34000.37, duration=431.506, finish_time=34431.88
```

The dp_allreduce adds **431.5 ms** (TP8) of pure serial time that should be mostly hidden.

### Root Cause

In `megatron-sim-engine/simu_main.py` (line 310):
```python
simulator_engine = SimulatorEngine(
    ...  # can_overlap NOT passed, defaults to False
)
```

`SimulatorEngine.__init__()` at `simu_engine.py` line 4027: `can_overlap=False`

In `_add_operation_to_timeline()` at lines 644-648:
```python
if self.can_overlap:
    _, timeline_last_operation = timeline._get_last_operation_time_and_op([target_timeline])
else:
    _, timeline_last_operation = timeline._get_last_operation_time_and_op([timeline.comp_timeline, timeline.comm_timeline])
```

With `can_overlap=False`, **every operation waits for both comp and comm timelines**, so everything is fully serialized.

### dp_allreduce Duration Breakdown

| Config | dp_allreduce (initial) | dp_allreduce (cc-predicted) | DP size |
|--------|----------------------|---------------------------|---------|
| TP8    | 717.59 ms (nccl_comm) | 431.51 ms (collective-sim) | 16      |
| TP16   | 539.32 ms            | ~539 ms                    | 8       |
| TP32   | 270.08 ms            | ~270 ms                    | 4       |

Note: dp_allreduce correctly scales down with DP size, but is NOT overlapped.

---

## 🟡 BUG #3 [MEDIUM]: nccl_comm.py GPUS_PER_MACHINE=8 Hardcode

### Symptom

The legacy communication estimator in `nccl_comm.py` hardcodes `GPUS_PER_MACHINE = 8`, which is incorrect for TP16 (16 GPUs/node) and TP32 (32 GPUs/node via NVSwitch) configurations.

### Code Location

`megatron-sim-engine/src/core/comm_sim/nccl_comm.py`:
- Line 15: `GPUS_PER_MACHINE = 8`
- `set_gpus_per_machine()` exists (line ~26) but is **never called** in the main simulation pipeline
- `_get_machine_distribution()` (lines 483-501) uses this value to determine cross-machine vs. intra-machine

### Impact Assessment: **LIMITED**

This legacy estimator is used in two places:
1. `extract_sub_operations()` — for **initial** sub_comm (tp_allreduce) duration estimation during database_profile loading
2. `process_mg_profile_files()` — for dp_allreduce/ep_allreduce initial duration

**However**, the `collective-sim` CC backend **re-predicts** all communication durations during the simulation phase via `_calculate_comm_duration()` → `_predict_comm_duration_ms()`. The collective-sim backend receives the **correct** `gpus_per_server` via `--cc-backend-options-json`.

### Verification

From `collective_sim_backend.py`: `gpus_per_server` is correctly resolved per config:
- TP8: gpus_per_server=8 (via --cc-backend-options-json)
- TP16: gpus_per_server=16
- TP32: gpus_per_server=32

TP allreduce times after collective-sim prediction are **reasonable**:
- TP8: ~0.66 ms/allreduce × 24 = ~15.7 ms per forward_step
- TP16: ~0.72 ms/allreduce × 24 = ~17.3 ms per forward_step
- TP32: ~0.81 ms/allreduce × 24 = ~19.5 ms per forward_step

---

## 🟢 BUG #4 [LOW]: Python Boolean Logic Bug in Duration Assignment

### Code Location

`megatron-sim-engine/src/core/simu_engine.py`, line 3915:
```python
if operation.name == "dp_allreduce" or "ep_allreduce" or "tp_allreduce":
```

### Issue

This condition is **always True** because Python evaluates `"ep_allreduce"` as a truthy string. The correct code should be:
```python
if operation.name in ("dp_allreduce", "ep_allreduce", "tp_allreduce"):
```

### Impact

The print statement on line 3916 fires for **every** operation that reaches this branch, not just allreduce operations. This is a logging bug only and does not affect simulation accuracy since the duration assignment on line 3913 has already occurred.

---

## Summary: Iteration Time Decomposition

### TP8 (TP=8, DP=16, 16 micro-batches)
```
Total fwd sub_comp:   10,614 ms (368 sub_comp ops)
Total bwd sub_comp:   10,964 ms (368 sub_comp ops)
TP allreduce:            ~251 ms (768 ops × ~0.33 ms)
dp_allreduce:            432 ms
optimizer_step:          161 ms
Pipeline overhead:      ~10 ms
─────────────────────────────────
Iteration time:       34,432 ms
```

### TP16 (TP=16, DP=8, 32 micro-batches)
```
Total fwd sub_comp:   24,359 ms (736 sub_comp ops)  ← 2.3x TP8!
Total bwd sub_comp:   20,794 ms (736 sub_comp ops)  ← 1.9x TP8!
TP allreduce:            ~554 ms (1536 ops × ~0.36 ms)
dp_allreduce:            539 ms
optimizer_step:           79 ms
Pipeline overhead:       ~10 ms
─────────────────────────────────
Iteration time:       60,503 ms (1.76x TP8)
```

### TP32 (TP=32, DP=4, 64 micro-batches)
```
Total fwd sub_comp:   43,103 ms (1472 sub_comp ops) ← 4.1x TP8!
Total bwd sub_comp:   37,932 ms (1472 sub_comp ops) ← 3.5x TP8!
TP allreduce:          ~1,156 ms (3072 ops × ~0.38 ms)
dp_allreduce:            270 ms
optimizer_step:           55 ms
Pipeline overhead:       ~10 ms
─────────────────────────────────
Iteration time:      106,202 ms (3.1x TP8)
```

---

## Root Cause Priority Ranking

| Rank | Bug | Impact | Estimated Error Contribution |
|------|-----|--------|------------------------------|
| 1    | **Sub_comp does NOT scale with TP** | CRITICAL | ~95% of total error |
| 2    | **dp_allreduce not overlapped with backward** | HIGH | ~3-5% of total error |
| 3    | **nccl_comm.py GPUS_PER_MACHINE=8 hardcode** | MEDIUM | <1% (overridden by collective-sim) |
| 4    | **Python boolean logic bug** | LOW | 0% (logging only) |

---

## Recommended Fixes (NOT implemented — review only)

### For Bug #1 (CRITICAL):
**Option A**: Add a TP scaling correction factor to `add_sub_ops_according_to_profile_dict()`. When assigning sub_comp durations, apply: `corrected_duration = raw_duration * (baseline_tp / target_tp)` for compute-bound operations.

**Option B**: Use FLOPS-based analytical model for sub_comp instead of raw profiling. For each sub_comp block, compute theoretical FLOPS based on known matrix dimensions and apply a GPU efficiency curve.

**Option C**: Profile at the target TP size on actual multi-GPU hardware instead of single-GPU fake_tp profiling.

### For Bug #2 (HIGH):
Enable `can_overlap=True` for dp_allreduce or implement bucketed gradient allreduce that overlaps with backward computation, matching Megatron-LM's actual behavior.

### For Bug #3 (MEDIUM):
Call `set_gpus_per_machine(config.local_size)` before using `nccl_comm.py`, or remove dependency on the hardcoded value entirely.

### For Bug #4 (LOW):
Fix the Python boolean expression: `if operation.name in ("dp_allreduce", "ep_allreduce", "tp_allreduce"):`
