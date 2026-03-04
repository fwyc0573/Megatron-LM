## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Executed seq8192 formal repeat-x5 with attention-core SDPA subsegment diagnostics (`dist + scaling_on + scaling_off`), then aggregated robust `median/IQR + top-k` evidence for stage1 backward rank4..7 |

## Test Report: Seq8192 Attention-Core SDPA Subsegment Repeat-x5

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Scope

1. Keep code and training semantics unchanged; run only measurement/analysis workflow.
2. Under `SEQ_LEN=8192`, execute formal repeat-x5 for:
   - distributed
   - scaling DDP-on (`SCALING_DISABLE_DDP_WRAP=0`)
   - scaling DDP-off (`SCALING_DISABLE_DDP_WRAP=1`)
3. Reuse phase-pure gate:
   - NVTX structure health (`open_forward/open_backward/overlap`)
   - contamination gate (`compute_pure` contamination <= 1%, observed target 0%).
4. For each run, collect `attn_core_sdpa_bwd` diagnosis and produce robust `median/IQR` + per-run top-k evidence.

### 2) Reproducible Commands

```bash
BASE=task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_repeat5_v1
mkdir -p "$BASE"

# repeat x5 capture + export + gate + analyze + compare + sdpa diag
# (dist + scaling_on + scaling_off)
# executed via scripted loop; artifacts are fully persisted under $BASE
```

实际执行脚本（等价命令）在本轮 shell 历史中已完整记录，核心参数保持与 x1 micro-segment 轮一致：
- `TRACE_KERNEL_GROUND_TRUTH=1`
- `TRACE_KERNEL_GROUND_TRUTH_PHASE=1`
- `TRACE_KERNEL_BOUNDARY_SYNC_MODE=event`
- `TRACE_ATTENTION_BACKWARD_SEGMENTS=1`
- `TRACE_SUBOP_SYNC_MODE=global`
- `SEQ_LEN=8192`, `TRAIN_ITERS=3`

聚合命令：
```bash
python - <<PY
# aggregate run1..5 compare logs + sdpa diag json
# output:
#   deepseek_phase_sl8192_attncore_repeat5_summary.json
#   deepseek_phase_sl8192_attncore_repeat5_summary.md
PY
```

### 3) Validation Criteria

1. 每个 run/branch 的 NVTX 结构 gate 必须通过；
2. contamination gate 必须通过（max contamination_pct_median == 0.0）；
3. `attn_core_sdpa_bwd` top1 kernel 在 repeat-x5 需稳定；
4. 输出 stage1 backward rank4..7 的 robust 统计：`top-k + median/IQR`。

### 4) Results and Evidence

#### 4.1 Gate 状态

- **NVTX structure gate**: run1..5 / dist+scaling_on+scaling_off 全部通过；
- **contamination gate**: run1..5 三分支 aggregate max 均为 `0.000%`。

证据：
- `logs/nsys_phase_attn_core_microseg_repeat5_v1/deepseek_phase_sl8192_attncore_repeat5_summary.md`

#### 4.2 `attn_core_sdpa_bwd` 稳健统计（stage1/backward/steady/rank4..7）

`scaling_on`:
- `gap_ms` median/IQR: `22.515 / 7.021`
- `fmha_gap_ms` median/IQR: `22.012 / 6.898`
- `fmha_gap_share_pct` median/IQR: `98.022 / 0.283`
- `dist_pre_count` median/IQR: `35 / 3`
- `scale_pre_count` median/IQR: `21 / 4`
- `dist_pre_ms` median/IQR: `1.751 / 0.151`
- `scale_pre_ms` median/IQR: `1.050 / 0.168`

`scaling_off`:
- `gap_ms` median/IQR: `23.725 / 20.024`
- `fmha_gap_ms` median/IQR: `23.263 / 19.963`
- `fmha_gap_share_pct` median/IQR: `98.041 / 0.703`
- `dist_pre_count` median/IQR: `35 / 3`
- `scale_pre_count` median/IQR: `21 / 4`
- `dist_pre_ms` median/IQR: `1.751 / 0.151`
- `scale_pre_ms` median/IQR: `1.051 / 0.201`

#### 4.3 Top-k 稳定性（每轮）

- `attn_core_sdpa_bwd` 的 top1 kernel 在 `scaling_on` 与 `scaling_off` **均为 5/5**:
  - `fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80(...)`
- pre-fmha 邻接 top-name 在两侧 5/5 均稳定为：
  - `vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`
- stream-set mismatch 仍为 0。

#### 4.4 Op-level（来自 compare logs 的 op-rank median diff%）

`scaling_on` run1..5:
- forward: `[15.83, 14.40, 14.51, 8.02, 6.60]` → median/IQR `14.40 / 6.49`
- backward: `[12.10, 4.27, 11.25, 14.83, 13.77]` → median/IQR `12.10 / 2.52`

`scaling_off` run1..5:
- forward: `[15.02, 14.86, 11.64, 10.64, 9.81]` → median/IQR `11.64 / 4.22`
- backward: `[4.05, 11.08, 18.17, 8.30, 14.84]` → median/IQR `11.08 / 6.54`

### 5) Verdict

1. `attn_core_sdpa_bwd` 作为 residual 主来源在 repeat-x5 上已稳健复现，top1 `fmha_cutlassB` 为 5/5。
2. pre-fmha 邻接 `FillFunctor<unsigned char>` 仍在 distributed 侧稳定更高，且在 on/off 两分支一致。
3. contamination 与 NVTX 结构均 clean，说明该 residual 不是 phase 语义污染造成。
4. backward 官方 freeze 仍未达成（run-level 波动仍存在，尤其 scaling_off IQR 较大），下一步应继续 SDPA 邻域 targeted 诊断而非扩展 comm-adjacent emulation。

### 6) Key Artifacts

- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_repeat5_v1/`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_repeat5_v1/deepseek_phase_sl8192_attncore_repeat5_summary.md`
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_core_microseg_repeat5_v1/deepseek_phase_sl8192_attncore_repeat5_summary.json`
