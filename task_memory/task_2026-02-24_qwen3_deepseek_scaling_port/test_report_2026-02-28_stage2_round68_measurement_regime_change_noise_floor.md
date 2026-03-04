## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Added Round6-8 baseline measurement-regime change report: OOM sweep (model/seq scaling), fixed rank7-cap repeat x5 at seq8192, and gate/noise-floor verdict |

## Test Report: Stage-2 Round6-8 Measurement Regime Change (No Code Change)

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18)  
**Execution workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_regime`  
**Code state**: detached `3a50265d` (Round6-8 baseline line; no new code patches)  

---

### 1) Goal

Validate whether the current smoke workload is too small for reliable <=5% gating by first changing **measurement regime only** (no new code hypothesis):

1. OOM detection sweep for larger workload settings.
2. Pick maximum feasible configuration.
3. Run fixed protocol `repeat x5` with **rank7 end-of-run cap**.
4. Decide whether noise floor is <5% and whether code-level A/B is statistically meaningful.

---

### 2) Protocol (fixed)

- Compare protocol retained from prior validated flow:
  - rank7 end-of-run `pair_timestamp` cap
  - repeated pairing `x5`
  - `--distributed-subtract-comm`
  - metric source: `op_rank_median_aux_summary`
- Base runtime params:
  - `TRACE_START=4`
  - `TRAIN_ITERS=6`
  - `TRACE_SUBOP_SYNC_MODE=global`
  - `TRACE_CMD_SYNC_MODE=global`
  - `TRACE_OPTIMIZER_MICROPHASES=1`
  - scaling rank order: `0,4,1,5,2,6,3,7`
  - `SCALING_MIN_WARMUP_ITERS=0`, `SCALING_PROFILE_ITERS=3`

---

### 3) Step-1 OOM Detection Sweep

#### 3.1 Sweep method

- **No code edits**; script-parameter-only sweep.
- Tried increasing model/sequence regime via existing script knobs:
  - `MODEL_PROFILE=full` (raises `NUM_LAYERS/HIDDEN_SIZE`), with reduced `SEQ_LEN` attempts.
  - `MODEL_PROFILE=smoke`, increasing `SEQ_LEN` up to script max (`8192`).

#### 3.2 OOM summary

| profile | seq_len | distributed | scaling | verdict |
|---|---:|---|---|---|
| full | 256 | OOM | - | rejected |
| full | 192 | OOM | - | rejected |
| full | 128 | OOM | - | rejected |
| full | 96 | OOM | - | rejected |
| smoke | 256~8192 | PASS | (8192) PASS | feasible |

Key evidence:
- Dist OOM summaries: `logs/deepseek_v3_stage2_round68_regime_oom_sweep_dist_summary.tsv`
- Scaling OOM summary: `logs/deepseek_v3_stage2_round68_regime_oom_sweep_scaling_summary.tsv`
- Full-profile OOM logs:
  - `logs/deepseek_v3_stage2_round68_regime_oom_dist_full_sl256.log`
  - `logs/deepseek_v3_stage2_round68_regime_oom_dist_full_sl192.log`
  - `logs/deepseek_v3_stage2_round68_regime_oom_dist_full_sl128.log`
  - `logs/deepseek_v3_stage2_round68_regime_oom_dist_full_sl96.log`

**Confirmed feasible config for noise-floor run**: `MODEL_PROFILE=smoke`, `SEQ_LEN=8192`.

---

### 4) Step-2 Noise Floor Measurement (repeat x5, seq8192)

#### 4.1 Run artifacts

- Distributed logs:
  - `logs/deepseek_v3_stage2_dist_round68_regime_sl8192_run1.log`
  - `logs/deepseek_v3_stage2_dist_round68_regime_sl8192_run2.log`
  - `logs/deepseek_v3_stage2_dist_round68_regime_sl8192_run3.log`
  - `logs/deepseek_v3_stage2_dist_round68_regime_sl8192_run4.log`
  - `logs/deepseek_v3_stage2_dist_round68_regime_sl8192_run5.log`
- Scaling logs:
  - `logs/deepseek_v3_stage2_scaling_round68_regime_sl8192_run1.log`
  - `logs/deepseek_v3_stage2_scaling_round68_regime_sl8192_run2.log`
  - `logs/deepseek_v3_stage2_scaling_round68_regime_sl8192_run3.log`
  - `logs/deepseek_v3_stage2_scaling_round68_regime_sl8192_run4.log`
  - `logs/deepseek_v3_stage2_scaling_round68_regime_sl8192_run5.log`
- Compare logs:
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run1.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run2.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run3.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run4.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run5.log`
- Compare stdout (with pair timestamps):
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run1.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run2.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run3.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run4.stdout.log`
  - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run5.stdout.log`
- Repeat aggregate:
  - `logs/deepseek_v3_stage2_repeat_round68_regime_sl8192_subtract.jsonl`
- Derived summary:
  - `logs/deepseek_v3_stage2_round68_regime_sl8192_metrics_summary.json`
  - `logs/deepseek_v3_stage2_round68_regime_sl8192_metrics_summary.md`

#### 4.2 Single-run and robust summary

| Run | Pair Timestamp | forward_step | backward_step | optimizer_step | mean_3ops | max_3ops |
|---|---|---:|---:|---:|---:|---:|
| run1 | 20260228152645 | 4.46% | 87.68% | 8.56% | 33.57% | 87.68% |
| run2 | 20260228152924 | 1.97% | 59.21% | 4.58% | 21.92% | 59.21% |
| run3 | 20260228153204 | 3.51% | 62.73% | 8.93% | 25.06% | 62.73% |
| run4 | 20260228153444 | 1.23% | 66.05% | 6.72% | 24.67% | 66.05% |
| run5 | 20260228153723 | 2.00% | 53.27% | 6.17% | 20.48% | 53.27% |

| Metric | Median | Min | Max | Range | Q1 | Q3 | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
| forward | 2.00% | 1.23% | 4.46% | 3.23% | 1.97% | 3.51% | 1.54% |
| backward | 62.73% | 53.27% | 87.68% | 34.41% | 59.21% | 66.05% | 6.84% |
| optimizer | 6.72% | 4.58% | 8.93% | 4.35% | 6.17% | 8.56% | 2.39% |
| mean_3ops | 24.67% | 20.48% | 33.57% | 13.09% | 21.92% | 25.06% | 3.14% |
| max_3ops | 62.73% | 53.27% | 87.68% | 34.41% | 59.21% | 66.05% | 6.84% |

---

### 5) Comparison vs previous smoke noise floor (seq256, repeat x5)

Reference (`test_report_2026-02-28_stage2_round68_noise_floor_repeat5.md`):
- median: `forward=7.16%`, `backward=11.67%`, `optimizer=8.28%`
- range: `forward=5.79%`, `backward=5.31%`, `optimizer=6.24%`

Current seq8192:
- median: `forward=2.00%`, `backward=62.73%`, `optimizer=6.72%`
- range: `forward=3.23%`, `backward=34.41%`, `optimizer=4.35%`

Delta (seq8192 - seq256 median):
- `forward -5.16%` (improved)
- `backward +51.06%` (severely worse)
- `optimizer -1.56%` (slight improvement)

---

### 6) Verdict

1. **Partially confirmed**: increasing workload did reduce forward variability and median mismatch.
2. **Not sufficient for <=5% gate**:
   - `optimizer` median still `6.72%` (>5%),
   - `backward` median exploded to `62.73%` with very large spread.
3. Therefore, we cannot conclude that smoke-size noise alone explains the current gate failure. Evidence indicates a **measurement/systematic bias** (especially `backward_step` under `distributed_subtract_comm`) dominates at long sequence regime.
4. **Decision**: do **not** start new code-level single-variable A/B yet. First validate measurement semantics for backward while preserving rank7-cap + repeat x5.

---

### 7) Repro Commands (exact pattern)

```bash
# OOM sweep (distributed)
MODE=distributed MODEL_PROFILE=<smoke|full> SEQ_LEN=<...> TRACE_START=1 TRAIN_ITERS=2 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
DO_TRACE=False GPUS_PER_NODE=8 MASTER_PORT=<port> \
bash examples/pretrain_deepseek_v3_moe.sh

# Feasible-config repeat x5 (distributed + scaling + compare)
MODE=distributed MODEL_PROFILE=smoke SEQ_LEN=8192 TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
DO_TRACE=True GPUS_PER_NODE=8 MASTER_PORT=<port> \
bash examples/pretrain_deepseek_v3_moe.sh

MODE=scaling MODEL_PROFILE=smoke SEQ_LEN=8192 TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 \
SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 MASTER_PORT=<port> \
bash examples/pretrain_deepseek_v3_moe.sh

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl8192 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl8192 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp <rank7_end_ts> \
  --distributed-subtract-comm \
  --repeat-report task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_repeat_round68_regime_sl8192_subtract.jsonl \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run<k>.log
```
