## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Completed attention-core SDPA deep-segment formal repeat-x5 (`dist + scaling_on + scaling_off`), generated robust summary (`median/IQR`) and added variability bucketing (`rank/state/iter` with fmha-vs-adjacency coupling) |

## Test Report: Seq8192 Attention-Core SDPA Deep-Segment Repeat-x5

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Test Script Information

- Capture/analysis artifact root:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_deepseg_repeat5_v1/`
- Main workflow (executed in one resumable pipeline, run1..5):
  1. NSYS profile for `distributed`
  2. NSYS profile for `scaling_on` (`SCALING_DISABLE_DDP_WRAP=0`)
  3. NSYS profile for `scaling_off` (`SCALING_DISABLE_DDP_WRAP=1`)
  4. `nsys export --sqlite`
  5. NVTX structure health gate
  6. phase-pure kernel breakdown + compare (`pure_primary_union`, `shared primary-stream`)
  7. deep-segment diagnosis (`attn_core_bwd`, `attn_core_sdpa_bwd`, `attn_core_sdpa_prefmha_bwd`, `attn_core_sdpa_fmha_bwd`, `attn_core_sdpa_postfmha_bwd`)
- Fixed runtime protocol:
  - `SEQ_LEN=8192`, `TRAIN_ITERS=3`, `TRACE_START=1`
  - `TRACE_KERNEL_GROUND_TRUTH=1`
  - `TRACE_KERNEL_GROUND_TRUTH_PHASE=1`
  - `TRACE_KERNEL_BOUNDARY_SYNC_MODE=event`
  - `TRACE_ATTENTION_BACKWARD_SEGMENTS=1`
- Post-run robust aggregation command (executed in this round):

```bash
python - <<PY
# Parse run1..5 compare logs + segment diagnosis JSON,
# then output:
#   deepseek_phase_sl8192_attncore_deepseg_repeat5_summary.json
#   deepseek_phase_sl8192_attncore_deepseg_repeat5_summary.md
PY
```

### 2) Validation Criteria

1. NVTX structure gate must pass for every run/branch:
   - `open_forward_step == 0`
   - `open_backward_step == 0`
   - `forward_backward_overlap_count == 0`
2. contamination gate must pass (`max contamination == 0.00%` in this protocol).
3. stage1 backward (`rank4..7`) deep-segment定位要稳定：
   - top1 kernel 稳定
   - `fmha_gap_share` 稳定
4. 若 backward 仍高波动，补充 `rank/state/iter` 波动分桶并检查与 pre-fmha 邻接强度耦合。

### 3) Test Results and Evidence

#### 3.1 Gate Results (run1..5, all branches)

- NVTX structure gate: **PASS (all)**
  - `open_forward=0`, `open_backward=0`, `overlap=0`
- contamination: **PASS (all)**
  - max contamination across all runs/branches = `0.00%`

Evidence:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_deepseg_repeat5_v1/deepseek_phase_sl8192_attncore_deepseg_repeat5_summary.md`

#### 3.2 Op-level Robust Results (rank-median diff%)

- scaling_on:
  - backward (all ranks): runs `[19.51, 11.68, 10.28, 12.36, 18.90]` → median/IQR `12.36 / 7.22`
  - backward (stage1 rank4..7): runs `[20.57, 15.915, 10.275, 10.530, 18.905]` → median/IQR `15.915 / 8.375`
- scaling_off:
  - backward (all ranks): runs `[19.27, 16.20, 19.03, 10.33, 16.68]` → median/IQR `16.68 / 2.83`
  - backward (stage1 rank4..7): runs `[19.995, 17.50, 19.495, 10.335, 14.05]` → median/IQR `17.50 / 5.445`

结论：backward 仍显著高于 freeze 目标（<=5%），且 stage1 residual 未收敛。

#### 3.3 Deep-Segment Robust Localization (`stage1/backward/steady/rank4..7`)

- `attn_core_sdpa_fmha_bwd` remains dominant in both branches:
  - scaling_on:
    - `gap_ms` median/IQR = `41.935 / 16.329`
    - `fmha_gap_share_pct` median/IQR = `97.62 / 0.33`
  - scaling_off:
    - `gap_ms` median/IQR = `43.605 / 8.525`
    - `fmha_gap_share_pct` median/IQR = `97.87 / 0.79`
- top1 kernel stability:
  - both branches: top1 `fmha_cutlassB...` = **5/5 runs**
- non-dominant segment behavior:
  - `attn_core_sdpa_prefmha_bwd` is small but non-zero (`~0.43ms on`, `~0.55ms off`)
  - `attn_core_sdpa_postfmha_bwd` remains `0.0ms`
- pre-fmha adjacency asymmetry remains:
  - distributed pre-adj ms median `~1.901`
  - scaling pre-adj ms median `~1.150` (on), `~1.017` (off)

#### 3.4 波动来源分桶（rank/state/iter，fmha 时长 vs 邻接强度耦合）

- 分桶对象：`attn_core_sdpa_fmha_bwd` paired windows（run1..5 × rank4..7 × iter0/1/2）
- 样本量：`240`（on/off 各 240）
- 关键模式（on/off一致）：
  - `iter=1` 是高残差桶：
    - on: `fmha_gap median ≈ 1.766ms`
    - off: `fmha_gap median ≈ 1.777ms`
  - `iter=0` 与 `iter=2` 接近低残差桶：
    - on: `~0.012ms`, `~0.023ms`
    - off: `~0.000ms`, `~0.010ms`
  - `iter=1` 同时出现稳定邻接偏移：
    - `small_pre_count_gap_median = -1.0`
    - `small_pre_gap_ms_median ≈ -0.05ms`

解释：当前波动主结构更像“iter1 上下文相位”而非随机噪声；这也是为何总体 segment total gap 维持高位，而窗口级分布呈现双峰（近0 与 ~1.8ms）。

### 4) Final Verdict

- 语义洁净性（phase-pure + NVTX structure）本轮已稳定通过。
- residual 主因定位进一步收敛：**`attn_core_sdpa_fmha_bwd` + `fmha_cutlassB`**（5/5 稳定）。
- 但 backward freeze 仍不可发布：
  - residual 水平仍远超 <=5%
  - run-level variability 由 iter-context bucket 主导，仍需单独约束/解释。

### 5) Key Artifacts

- Summary:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_deepseg_repeat5_v1/deepseek_phase_sl8192_attncore_deepseg_repeat5_summary.md`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_deepseg_repeat5_v1/deepseek_phase_sl8192_attncore_deepseg_repeat5_summary.json`
- Per-run compare logs:
  - `.../deepseek_phase_sl8192_attncore_deepseg_repeat5_run{1..5}_compare_on_pure_primary_union_shared.log`
  - `.../deepseek_phase_sl8192_attncore_deepseg_repeat5_run{1..5}_compare_off_pure_primary_union_shared.log`
- Per-run deep-segment diagnosis JSON/MD:
  - `.../deepseek_phase_sl8192_attncore_deepseg_repeat5_run{1..5}_{on|off}_attn_core_sdpa_fmha_bwd_diag.{json,md}`
