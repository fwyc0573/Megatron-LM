## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-27 | Initial version: stage-2 optimizer microphase protocolfix8 fidelity evidence (single + repeat) |

## Test Report: DeepSeek-V3 Stage-2 Microphase Fidelity (Protocolfix8)

**Date**: 2026-02-27  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Host**: 8x A800-SXM4-80GB

### 1) Test Script Information

- Working directory: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`
- Idle GPU pre-check:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader
```

- Distributed run (microphase enabled):

```bash
MODE=distributed MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=9530 \
bash examples/pretrain_deepseek_v3_moe.sh
```

- Scaling runs (fixed protocol, microphase enabled):

```bash
# run1
MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
SCALE_GPU=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=9630 \
SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 \
SCALING_REPLAY_CACHE_TAG=stage2_microphase_protocolfix8_run1 \
bash examples/pretrain_deepseek_v3_moe.sh

# run2
MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
SCALE_GPU=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=9631 \
SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 \
SCALING_REPLAY_CACHE_TAG=stage2_microphase_protocolfix8_run2 \
bash examples/pretrain_deepseek_v3_moe.sh

# run3
MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
SCALE_GPU=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=9632 \
SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 \
SCALING_REPLAY_CACHE_TAG=stage2_microphase_protocolfix8_run3 \
bash examples/pretrain_deepseek_v3_moe.sh
```

- Compare commands (phase-aware ops):

```bash
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step,optimizer_main_update,optimizer_state_update,optimizer_post_update \
  --threshold-pct 5 \
  --pair-timestamp <run_timestamp_cap> \
  --report-path <run_report_log> \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_microphase_protocolfix8_subtract.jsonl
```

### 2) Validation Criteria

- Functional criteria:
  - distributed/scaling runs complete with exit code `0`.
  - microphase traces appear in both modes:
    - `optimizer_main_update`
    - `optimizer_state_update`
    - `optimizer_post_update`
- Fidelity criteria:
  - per-run `op_rank_median_aux_summary` target <= `5%`.
- Aggregation criteria:
  - provide repeat evidence and median-of-runs summary for both top-level and microphase ops.

### 3) Test Results and Evidence

#### 3.1 Run status

| Item | Result | Evidence |
|---|---|---|
| Distributed microphase run | PASS | `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6.log` |
| Scaling run1 | PASS | `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_protocolfix8_run1.log` |
| Scaling run2 | PASS | `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_protocolfix8_run2.log` |
| Scaling run3 | PASS | `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_protocolfix8_run3.log` |

#### 3.2 Single-run phase-aware summaries (`op_rank_median_aux_summary`)

| Run | forward | backward | optimizer_step | optimizer_main_update | optimizer_state_update | optimizer_post_update |
|---|---:|---:|---:|---:|---:|---:|
| run1 | 8.34% | 6.81% | 11.19% | 10.86% | 33.33% | 0.00% |
| run2 | 10.53% | 13.39% | 10.43% | 9.75% | 27.78% | 16.67% |
| run3 | 7.79% | 8.17% | 13.22% | 12.52% | 30.00% | 12.50% |

Evidence logs:
- `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_protocolfix8_run1.log`
- `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_protocolfix8_run2.log`
- `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_protocolfix8_run3.log`

#### 3.3 Repeat aggregation (median-of-runs on op-rank-median)

| Op | Median-of-runs | Status |
|---|---:|---|
| forward_step | 8.34% | FAIL |
| backward_step | 8.17% | FAIL |
| optimizer_step | 11.19% | FAIL |
| optimizer_main_update | 10.86% | FAIL |
| optimizer_state_update | 30.00% | FAIL |
| optimizer_post_update | 12.50% | FAIL |

Repeat JSONL:
- `logs/deepseek_v3_stage2_repeat_microphase_protocolfix8_subtract.jsonl` (3 records)

### 4) Diagnosis and Interpretation

- `optimizer_main_update` remains high across all runs (~9.75% to 12.52%), close to top-level `optimizer_step`, indicating the dominant residual is still in main optimizer update compute path.
- `optimizer_state_update` / `optimizer_post_update` show very high relative error percentages but their absolute durations are very small (typically `0.01~0.05ms`), so relative percent alone is unstable for gating.
- Current protocolfix8 + microphase instrumentation improves observability, but does not yet bring fidelity to `<=5%`.

### 5) Exit Status

- Execution protocol and microphase tracing: **PASS**
- Stage-2 fidelity gate (`<=5%`): **FAIL**
- Next action needed: propose next semantic-touching optimization design and wait for explicit confirmation before code changes.
