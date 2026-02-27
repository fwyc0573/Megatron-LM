## Test Report: DeepSeek-V3 Stage-2 Protocol Alignment (Round-6/7/8)

**Date**: 2026-02-27  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Host**: 8x A800-SXM4-80GB

### 1) Test Script Information

- Working directory: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`
- Idle GPU pre-check:
  - `nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader`
- Serena call check:
  - `list_mcp_resources(server="serena")`
  - `mcp__serena__activate_project(...)`

- Main run commands (fixed protocol family):

```bash
# scaling run (example: protocolfix8 interleave + warmup=1)
MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global \
SCALE_GPU=5 MASTER_ADDR=127.0.0.1 MASTER_PORT=9400 \
SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALING_MIN_WARMUP_ITERS=1 SCALING_PROFILE_ITERS=3 \
SCALING_REPLAY_CACHE_TAG=stage2_protocolfix8_interleave_w1_run1 \
bash examples/pretrain_deepseek_v3_moe.sh

# compare on explicit pairset (fixed distributed ts=20260227145522)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir <pairset_dist_dir> \
  --scaling-dir <pairset_scale_dir> \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_protocolfix8_vs_dist145522_subtract.jsonl \
  --report-path <compare_log>
```

### 2) Validation Criteria

- Protocol alignment constraints:
  1. fixed port segment,
  2. fixed fake rank order,
  3. repeated pairing with archived evidence.
- Accuracy criteria (trace compare, subtract-comm):
  - target op-rank-median diff <= 5% for `forward_step`, `backward_step`, `optimizer_step`.
- Evidence requirements:
  - provide **single-run** evidence,
  - provide **repeat aggregation** (median-of-runs) evidence.

### 3) Test Results and Evidence

#### 3.1 Single-run evidence (current best in this round)

- Report: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_trace4_iter6_protocolfix7_run1_vs_dist145522_subtract.log`
- `op_rank_median_aux_summary`:
  - `forward_step`: **3.06%** (PASS)
  - `backward_step`: **7.51%** (FAIL)
  - `optimizer_step`: **5.97%** (FAIL)

#### 3.2 Additional protocol variants (all fixed-port + fixed-order family)

- `...protocolfix8_interleave_w1_run1_vs_dist145522_subtract.log`
  - forward `10.99%` (FAIL), backward `5.66%` (FAIL, near threshold), optimizer `7.42%` (FAIL)
- `...protocolfix8_seq_w0_run1_vs_dist145522_subtract.log`
  - forward `9.18%` (FAIL), backward `7.70%` (FAIL), optimizer `7.29%` (FAIL)
- `...protocolfix8_interleave_w0_p2_run1_vs_dist145522_subtract.log`
  - forward `2.97%` (PASS), backward `7.44%` (FAIL), optimizer `9.35%` (FAIL)

#### 3.3 Repeat aggregation evidence (median-of-runs)

- Repeat series (3 runs):
  - run1: `...protocolfix6_run1_subtract.log`
  - run2: `...protocolfix6_run2_subtract.log`
  - run3: `...protocolfix6_run3_subtract.log`
- Repeat JSONL:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_fidelityfix6_subtract.jsonl`
- Median-of-runs (on op-rank-median values):
  - `forward_step`: **8.48%** (FAIL)
  - `backward_step`: **15.02%** (FAIL)
  - `optimizer_step`: **6.69%** (FAIL)

### 4) Failure Diagnosis (current round)

- Target `backward_step <= 5%` is **not achieved** in stable repeated protocol.
- Root-cause signal from current evidence:
  - strongest residual error集中在 `backward_step@steady` 的 rank4/rank6 区段，且 run-to-run 漂移较大；
  - fixed-port + fixed-order + explicit pairset 已显著收敛到可解释区间，但仍未跨过 5% gate；
  - `optimizer_step` remains above threshold (typically 5.9%~9%+).

### 5) Exit Status

- Protocol alignment execution: **PASS** (commands complete, logs complete, pairsets complete).
- Fidelity target gate (`<=5%` on three ops): **FAIL** (not yet met).
