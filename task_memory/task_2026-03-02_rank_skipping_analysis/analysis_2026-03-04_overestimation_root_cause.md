## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initial root cause analysis: Qwen3 + DeepSeek-V3-variant systematic overestimation |

# Root Cause Analysis: H800 16-GPU Qwen3 + DeepSeek-V3-variant Systematic Overestimation

## 1. Problem Statement

2026-03-04 的 6-case simulate/profile 比对出现系统性 overestimation（mean gap +104.66%），
与 2026-03-02 Mixtral 8-case 的良好结果（mean abs gap 4.18%）形成鲜明对比。
两次测试均针对 H800 16-GPU 2-node 环境，仅更换了模型。

## 2. Root Cause: Schedule / Profile GBS 不匹配（PRIMARY）

### 2.1 证据

所有 6 个 Qwen3 + DeepSeek-V3-variant 案例的 schedule 均使用 `--global-batch-size 64` 生成，
但实际 profiling（`global_ranks_profile`）采用的是 **GBS=32**。

#### 完整一致性检查表

| Model | Case | PP | Schedule fwd | Profile fwd | DB iters | Match? |
|-------|------|----|-------------|------------|---------|--------|
| **Mixtral** | pp2_tp1_exp2_expn8_dp8 | 2 | 8 | 8 | 1 | **OK** |
| **Mixtral** | pp2_tp1_exp4_expn16_dp8 | 2 | 8 | 8 | 1 | **OK** |
| **Mixtral** | pp2_tp1_exp4_expn8_dp8 | 2 | 8 | 8 | 1 | **OK** |
| **Mixtral** | pp2_tp1_exp8_expn16_dp8 | 2 | 8 | 8 | 1 | **OK** |
| **Mixtral** | pp2_tp1_exp8_expn8_dp8 | 2 | 8 | 8 | 1 | **OK** |
| **Mixtral** | pp4_tp1_exp2_expn8_dp4 | 4 | 16 | 16 | 1 | **OK** |
| **Mixtral** | pp4_tp1_exp4_expn16_dp4 | 4 | 16 | 16 | 1 | **OK** |
| **Mixtral** | pp4_tp1_exp4_expn8_dp4 | 4 | 16 | 16 | 1 | **OK** |
| **Qwen3** | pp2_tp1_exp8_expn128_dp8 | 2 | 8 | **4** | 1 | **MISMATCH** |
| **Qwen3** | pp4_tp1_exp4_expn128_dp4 | 4 | 16 | **8** | 1 | **MISMATCH** |
| **Qwen3** | pp8_tp1_exp2_expn128_dp2 | 8 | 32 | **16** | 1 | **MISMATCH** |
| **DS-V3** | pp2_tp1_exp4_expn32_dp8 | 2 | 8 | **4** | 3 | **MISMATCH** |
| **DS-V3** | pp2_tp1_exp8_expn32_dp8 | 2 | 8 | **4** | 3 | **MISMATCH** |
| **DS-V3** | pp4_tp1_exp4_expn32_dp4 | 4 | 16 | **8** | 3 | **MISMATCH** |

所有 MISMATCH 案例的 ratio = schedule_fwd / profile_fwd = **2.0x**。

### 2.2 GBS 推算

- Mixtral profiling GBS = profile_fwd × MBS × DP = 8 × 1 × 8 = **64** → schedule GBS=64 **一致**
- Qwen3/DS-V3 profiling GBS = profile_fwd × MBS × DP = 4 × 1 × 8 = **32** → schedule GBS=64 **不一致**

### 2.3 schedule 生成命令溯源

来自 `step1_schedule_summary_2026-03-04.json`，所有 6 个 case 的 `mg_test_cmd` 均包含：
```
--global-batch-size 64
```

### 2.4 影响链

1. **Simulate 模式**: 使用 schedule → 2x forward/backward/send/recv ops
2. **Profile 模式**: 使用 global_ranks_profile → 正确的 microbatch 数量
3. **结果**: simulate 系统性高估 E2E

### 2.5 量化验证

通过 1F1B 模型 `E2E ∝ (PP-1 + num_micro)` 估算 2x microbatch 的预期 gap：

| Model | Case | PP | Expected gap from 2x | Actual gap | Corrected gap |
|-------|------|----|---------------------|-----------|--------------|
| Qwen3 | pp2_exp8 | 2 | +80% | +111.7% | +17.6% |
| Qwen3 | pp4_exp4 | 4 | +72.7% | +52.4% | -11.8% |
| Qwen3 | pp8_exp2 | 8 | +69.6% | +42.3% | -16.1% |
| DS-V3 | pp2_exp4 | 2 | +80% | +242.4% | +90.2% |
| DS-V3 | pp2_exp8 | 2 | +80% | +141.3% | +34.1% |
| DS-V3 | pp4_exp4 | 4 | +72.7% | +37.8% | -20.2% |

- Qwen3 案例：校正后 gap 在 -16% ~ +18%，与 Mixtral 基线（mean abs 4.18%）同数量级
- DS-V3 PP=2 案例：校正后仍有 +34% ~ +90% 残差 → 存在额外问题

## 3. Secondary Finding: DS-V3 database_profile 多迭代

### 3.1 现象

DS-V3 的 database_profile 每个 rank 文件包含 **3 轮迭代**（18 行），
而 Mixtral 和 Qwen3 仅 1 轮（6 行）。

### 3.2 影响

- Engine `process_mg_profile_files` 通过 dict overwrite 保留最后一轮
- 第 1 轮 forward_step=135.85ms（warmup 膨胀），第 3 轮=56.94ms（稳定值）
- Engine 正确使用了最后一轮的值 → **不是 gap 的直接原因**
- 但建议数据预处理时只保留最后一轮，避免潜在歧义

### 3.3 真实 vs database 计算时间对比（DS-V3 pp2_exp4 rank0）

| Op | database_profile (scaling) | global_ranks_profile (realistic) | Ratio |
|----|-------------------------:|-------------------------------:|------:|
| forward_step | 56.94 ms | ~66.9 ms | 0.85x |
| backward_step | 79.74 ms | ~93.4 ms | 0.85x |
| optimizer_step | 14.99 ms | 14.96 ms | ~1.0x |

Database 的 forward/backward 比真实分布式低约 15%。这本应导致 simulate **低估**，
但实际观测到的是大幅 **高估**，进一步确认 GBS 不匹配是主因。

## 4. Tertiary Finding: Sub-op 数量差异放大 CC 预测误差

| Model | fwd sub_ops | bwd sub_ops | Total comm ops/microbatch |
|-------|------------|------------|--------------------------|
| Mixtral (nl=8) | 12 | 8 | 20 |
| DS-V3 (nl=32) | 33 | 22 | 55 |
| Qwen3 (nl=48) | 72 | 48 | 120 |

更多的 MoE layer → 更多的 `exp_all_to_all`/`exp_allgather` sub-ops → CC predictor 误差累积更大。
这可能是 DS-V3 PP=2 案例在 GBS 校正后仍有较大残差的部分原因。

## 5. 修复计划

### Fix 1: 用正确的 GBS 重新生成 schedule（CRITICAL）

```bash
# 所有 Qwen3 + DS-V3 案例应使用 --global-batch-size 32
# Example for Qwen3 pp2:
python mg_test.py --local-size 8 --world-size 16 --micro-batch-size 1 \
  --global-batch-size 32 --seq-length 2048 --hidden-size 2048 \
  --train-iters 3 --trace-start 3 --model-size Custom_128x_hs2048_nl48 \
  -pp 2 -tp 1 -exp 8 --num-experts 128 --untie-embeddings-and-output-weights
```

正确的 GBS 计算：`GBS = profile_fwd_count × MBS × DP`
- PP=2, DP=8, 4 fwd → GBS = 4 × 1 × 8 = 32
- PP=4, DP=4, 8 fwd → GBS = 8 × 1 × 4 = 32
- PP=8, DP=2, 16 fwd → GBS = 16 × 1 × 2 = 32

### Fix 2: 重新运行 6-case 比对

用新 schedule 重新执行 simulate/profile comparison，预期 gap 回到 Mixtral 基线水平（mean abs gap < 10%）。

### Fix 3: 增加 GBS 一致性校验（STRUCTURAL）

在 `SimulatorEngine._init_3d_parallel_all_ranks` 或 `generate_stages_and_cmds_info_from_datasets_and_schedules` 中增加：

```python
# Validate schedule microbatch count matches profile trace
if trace_stages_dict and stages_dict:
    for stage_id in stages_dict:
        sched_fwd = sum(1 for op in stages_dict[stage_id].operations_list if op.name == 'forward_step')
        if stage_id in trace_stages_dict:
            trace_fwd = sum(1 for op in trace_stages_dict[stage_id].operations_list if op.name == 'forward_step')
            if sched_fwd != trace_fwd:
                raise ValueError(
                    f"Microbatch count mismatch: schedule has {sched_fwd} forward_step "
                    f"but profile trace has {trace_fwd} for stage {stage_id}. "
                    f"Check --global-batch-size parameter."
                )
```

### Fix 4: DS-V3 database_profile 数据清理（OPTIONAL）

将 3 轮迭代的 database_profile 裁剪为仅保留最后 1 轮（6 行/rank），
消除 warmup 迭代可能引起的歧义。

### Fix 5: 残差诊断（AFTER Fix 1）

GBS 修复后若 DS-V3 PP=2 案例仍有较大残差（>10%），需进一步做：
- Per-op effective latency 分解（simulate vs profile）
- CC predictor 对 `exp_all_to_all` 在该消息规模下的精度标定
- Profile trace vs schedule replay 的 `mg_state`/batch 对齐审计

## 6. 预期修复效果

基于量化分析，修复 GBS 后预期 gap：

| Model | Case | Current gap | Predicted corrected gap |
|-------|------|-----------|----------------------|
| Qwen3 | pp2_exp8 | +111.7% | ~+15%~+20% |
| Qwen3 | pp4_exp4 | +52.4% | ~-10%~0% |
| Qwen3 | pp8_exp2 | +42.3% | ~-15%~-5% |
| DS-V3 | pp2_exp4 | +242.4% | ~+80%~+100% (需额外诊断) |
| DS-V3 | pp2_exp8 | +141.3% | ~+30%~+40% (需额外诊断) |
| DS-V3 | pp4_exp4 | +37.8% | ~-20%~-10% |

DS-V3 PP=2 案例预计仍需进一步优化。
