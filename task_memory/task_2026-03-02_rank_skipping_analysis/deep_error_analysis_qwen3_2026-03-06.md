# Qwen3 MoE 深入误差分析报告

## Modification History

| Date       | Summary of Changes                                      |
|------------|----------------------------------------------------------|
| 2026-03-06 | Initial version: comprehensive deep error analysis for Qwen3 MoE |

---

## 1. 概述

本报告对 Qwen3-30B-A3B MoE (128 experts) 在 H800 16-GPU 集群上的三种并行配置的模拟误差进行深入剖析。

**数据来源**：`megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_qwen3_moe`

| Case | PP | TP | EP | DP | Layers | E2E Gap |
|------|----|----|----|----|--------|---------|
| pp2_tp1_exp8 | 2 | 1 | 8 | 8 | 48 | **+19.31%** |
| pp4_tp1_exp4 | 4 | 1 | 4 | 4 | 48 | **-10.21%** |
| pp8_tp1_exp2 | 8 | 1 | 2 | 2 | 48 | **-15.11%** |

已修复的 GBS mismatch 问题不在本报告范围内。本报告聚焦于 **GBS 修复后的残差误差 (residual error)**。

---

## 2. 核心发现：四层误差来源

### 2.1 误差源 #1：Scaling Mode 纯计算开销 (+7% ~ +18%)

**证据：optimizer_step 直接对比**

optimizer_step 没有任何 comm sub-operations，其 DB vs GR 差异直接反映 scaling mode 的计算开销：

| Case | Mean Opt Bias |
|------|---------------|
| PP2_EP8 | **+7.4%** |
| PP4_EP4 | **+14.6%** |
| PP8_EP2 | **+16.2%** |

**Per-rank 详细数据 (PP2 示例)**：

| Rank | DB_opt (ms) | GR_opt (ms) | Bias |
|------|-------------|-------------|------|
| 0 | 48.64 | 43.77 | +11.1% |
| 1 | 49.32 | 43.81 | +12.6% |
| 8 | 41.62 | 43.58 | -4.5% |
| 10 | 43.76 | 43.48 | +0.6% |

**原因分析**：
- **GPU 热管理**：单 GPU 顺序执行 16 个 fake rank，持续高负载导致温度升高和频率降低
- **CUDA kernel 调度差异**：无并发 comm stream 竞争时，GPU SM 调度行为不同
- **Memory subsystem 差异**：scaling mode 的内存访问模式（无 NCCL buffer 竞争）与 distributed 不同
- **PP 度数越高，影响越大**：PP8 每个 stage 只有 6 层，optimizer_step 更轻量，相对误差更敏感

**性质**：systematic overestimation，不可忽略但影响有限。

---

### 2.2 误差源 #2：Comm-Compute Overlap 未建模 (最核心误差)

**这是导致系统性 sub_comp 高估的根本原因。**

#### 2.2.1 数学推导

Simulator 的 forward_step 计算逻辑（代码位于 `simu_engine.py:extract_sub_operations`）：

```
sim_forward = sum(sub_comp_i) + sum(CC_comm_j)
            = D_DB + sum(CC_comm_j)        // 因为 sum(sub_comp_i) = D_DB
```

其中：
- `D_DB` = database_profile 的 forward_step wall-clock duration（scaling mode，comm=0）
- `CC_comm_j` = collective-sim 预测的各 comm sub-op duration
- 关键设定：`can_overlap=False`（`simu_engine.py:630`），comp 和 comm **串行排列**

现实中（global_ranks_profile）：
```
D_GR = f(comp, comm, overlap)   // 真实分布式模式
     ≈ comp_real + comm_real - overlap
```

因此误差为：
```
error = sim_forward - D_GR
      = D_DB + CC_comm - D_GR
```

如果 CC_comm 预测完美（= C_GR = sum of real comm sub-op durations）：
```
error = D_DB + C_GR - D_GR
      = D_DB - (D_GR - C_GR)
      = D_DB - naive_comp
```

**关键洞察**：当 comm 预测完美时，E2E 误差恰好等于 `D_DB - naive_comp`，即 comm-compute overlap 在真实分布式执行中"省下"的时间。

#### 2.2.2 Implied Overlap Ratio (α) 量化

定义 `α = (D_GR - D_DB) / C_GR`，即需要多少 overlap 才能使 D_DB = real_comp。

**Forward Step α 分布**：

| Case | α mean | α median | 解读 |
|------|--------|----------|------|
| PP2_EP8 | 0.25 | 0.30 | 低 overlap (但 comm 量大: ~48ms/mb) |
| PP4_EP4 | 0.43 | 0.47 | 中等 overlap |
| PP8_EP2 | 0.01 | 0.01 | 几乎无 overlap (comm 量小: ~7ms/mb) |

**Backward Step α 分布**：

| Case | α mean | α median | 解读 |
|------|--------|----------|------|
| PP2_EP8 | 0.65 | 0.67 | 高 overlap |
| PP4_EP4 | 0.59 | 0.62 | 中高 overlap |
| PP8_EP2 | 0.39 | 0.40 | 中低 overlap |

**解读**：
- **PP2** forward α=0.25 表示：在分布式模式下，约 25% 的 MoE comm 时间与 compute 并行执行
- **PP8** forward α≈0 表示：comm 几乎不与 compute overlap（因为 comm 量小: ~7ms vs 总时间 ~30ms）
- **Backward 普遍比 Forward 有更高 overlap**：原因是 backward MoE all-to-all 的 kernel launch 更容易与梯度计算 overlap

#### 2.2.3 代表性 Rank 实例分析 (PP2 Rank0 Forward)

| 指标 | DB | GR |
|------|----|----|
| forward_step duration | 105.64ms | 134.31ms (batch0) |
| Comm sub-ops count | 72 (all duration=0.0) | 72 (total=47.46ms) |
| Naive comp | 105.64ms | 86.85ms |
| **naive comp bias** | | **+21.6%** |

Simulator 路径：`105.64 + CC_comm(≈47ms) = ≈153ms` → 比真实 134ms **高 14%**。

这 14% 完全来自 **overlap 未建模**：真实分布式中 comp 和 comm 部分并行执行，wall-clock 小于两者之和。

---

### 2.3 误差源 #3：Per-microbatch Comp Naive Bias (全量化)

将 DB_comp 与 GR_naive_comp = (GR_total - GR_comm) 做 per-microbatch 比较：

**Forward Step Naive Bias (mean across 16 ranks)**：

| Case | Mean Bias | Median Bias |
|------|-----------|-------------|
| PP2_EP8 | **+44.1%** | +41.2% |
| PP4_EP4 | **+27.2%** | +26.8% |
| PP8_EP2 | **+29.6%** | +27.5% |

**Backward Step Naive Bias (mean across 16 ranks)**：

| Case | Mean Bias | Median Bias |
|------|-----------|-------------|
| PP2_EP8 | **+36.1%** | +37.5% |
| PP4_EP4 | **+38.3%** | +36.8% |
| PP8_EP2 | **+43.9%** | +43.8% |

**重要注意**：这些 "+27-44%" 的 naive bias **并非全部是"错误"**。它包含两个成分：

1. **真实的 scaling mode 计算开销** (≈+10-18%, 由 optimizer_step 佐证)
2. **Comm-compute overlap 导致的 naive_comp 低估** (≈+10-25%, 真实 comp 高于 naive_comp)

因此：`naive_bias ≈ scaling_overhead + overlap_illusion`

---

### 2.4 误差源 #4：Comm 预测准确度

从 residual_per_op 报告（GBS 修复后）：

**exp_all_to_all delta (sim - prof)**:

| Case | Anchor Rank | delta (ms) | 方向 |
|------|-------------|------------|------|
| PP2_EP8 | rank2 | -133.63 | 低估 |
| PP4_EP4 | rank2 | -178.55 | 低估 |
| PP8_EP2 | rank3 | -209.53 | 低估 |

MoE all-to-all 通信在所有 case 中均被 **系统性低估**。

**GR forward_step comm 构成 (rank0, per-microbatch)**:

| Case | exp_allgather | exp_all_to_all | Total Comm |
|------|---------------|----------------|------------|
| PP2_EP8 | 12.91ms | 11.02ms | ~24ms |
| PP4_EP4 | 6.49ms | 4.69ms | ~11ms |
| PP8_EP2 | 1.84ms | 1.64ms | ~3.5ms |

**Comm 量随 EP 减少而显著下降**（EP=8→4→2 时，per-layer comm 次数不变但 group_size 变小）。

---

## 3. 误差交互效应与 E2E 净影响

### 3.1 E2E Gap 分解

对于每个 case 的 E2E gap，误差源的贡献方向：

| 误差源 | PP2 (+19.3%) | PP4 (-10.2%) | PP8 (-15.1%) |
|--------|-------------|-------------|-------------|
| Scaling comp overhead | ↑ +7-12% | ↑ +15-18% | ↑ +16-18% |
| Overlap not modeled | ↑ +10-15% | ↑ +5-8% | ↑ ~0% (comm 太小) |
| Comm prediction error | ↓ (all-to-all 低估) | ↓↓ (更严重) | ↓↓↓ (最严重) |
| Recv/send PP timing | ↑ (recv_bwd +162ms) | ↓ | ↓↓ |

**关键洞察**：
- **PP2 (+19.3%)**：overlap 高估 + scaling overhead → 显著正偏，PP timing 误差也贡献正偏
- **PP4 (-10.2%)**：comm 低估超过了 comp 高估，净负偏
- **PP8 (-15.1%)**：comm 低估最严重（EP=2 的 all-to-all 预测最差），且 recv_backward P2P timing 误差极大 (-221ms)

### 3.2 误差的 sign 翻转机制

三个 case 展现了 **正→负** 的 gap 翻转，机制是：

1. PP 增加 → microbatch 数增加 → comm 预测误差被放大（每个 mb 都有 comm 子操作）
2. EP 减少 → group_size 变小 → CC 预测模型可能对小 group all-to-all 适配性差
3. PP 增加 → pipeline bubble 比例增加 → recv_backward/send_forward P2P timing 变得关键

---

## 4. 根因归因总结

### 4.1 核心根因排序 (按影响大小)

| 排序 | 根因 | 影响量级 | 性质 | 可修复性 |
|------|------|----------|------|----------|
| **#1** | **Comm-Compute Overlap 未建模** | +10-15% E2E (PP2/PP4) | Systematic overestimation | 需 simulator 架构改进 |
| **#2** | **Scaling mode 计算开销** | +7-18% per-op | Systematic overestimation | 需 calibration 或多次测量取均 |
| **#3** | **MoE all-to-all 预测误差** | -130~-210ms per E2E | Systematic underestimation | 需改进 CC backend |
| **#4** | **P2P pipeline timing** | ±90-220ms per E2E | Case-dependent | 需改进 scheduling 匹配 |
| **#5** | **Rank 异常值** (如 PP2 rank10 DB_fwd=191ms) | Sporadic | Outlier | 需多次测量取中位数 |

### 4.2 关键机制图示

```
Scaling Mode (single GPU):
  forward_step = [==== comp ==== 0_comm 0_comm ==== comp ====]
                  |<----------- D_DB = 105.64ms ----------->|
  comm sub-ops: 72 × duration=0.0ms (bypassed)
  DB 记录的是 "如果 comm 免费" 的 wall-clock

Realistic Mode (16 GPUs distributed):
  forward_step = [=== comp === |comm| === comp === |comm| ... ]
                  |<------------ D_GR = 134.31ms ------------>|
  comm sub-ops: 72 × avg ~0.66ms = 47.46ms total
  但 comp 和 comm 有 partial overlap → total < comp + comm

Simulator (can_overlap=False):
  sim_forward = [sub_comp_0][CC_comm_0][sub_comp_1][CC_comm_1]...
                |<---- D_DB = 105.64ms ---->|<-- CC ~47ms -->|
                |<----------- sim_total ≈ 153ms ----------->|
  
  vs reality 134ms → overestimate by ~19ms (14%)
```

---

## 5. 建议的改进方向

### 5.1 短期优化 (Impact: High, Effort: Medium)

1. **引入 overlap factor (α)**：
   - 在 `extract_sub_operations` 或 timeline 排布时，允许部分 comm 与 comp overlap
   - 可以基于经验 α 值（forward: 0.3, backward: 0.6）或者基于 CUDA stream 分析自动推断
   - 修改 `IndividualTimeline` 的 `can_overlap` 机制，使 comp_timeline 和 comm_timeline 可以部分并行

2. **Scaling mode 计算校准**：
   - 多次 warm-up 后取中位数，减轻热管理影响
   - 或者引入 per-op 校准系数（calibration factor from optimizer_step baseline）

### 5.2 中期优化 (Impact: High, Effort: High)

3. **改进 MoE all-to-all CC 预测模型**：
   - 当前模型对不同 group_size 的 all-to-all 预测系统性偏低
   - 需要 profiling 不同 group_size (2/4/8) 的 all-to-all bandwidth curve

4. **P2P pipeline timing 改进**：
   - recv_backward 的预测在 PP8 中误差极大 (-221ms)
   - 可能需要更精确的 pipeline bubble / dependency 建模

### 5.3 长期架构改进

5. **Dual-stream timeline 模型**：
   - 将 comp 和 comm 放在独立的 timeline stream 上
   - 模拟 CUDA 的多 stream 并发行为
   - 这是解决 overlap 问题的根本方案

---

## 6. 附录：完整数据

### A. Per-Rank Forward Step Breakdown (PP2_EP8)

| Rank | DB_comp | GR_total | GR_comm | GR_naive | Naive Bias | α |
|------|---------|----------|---------|----------|------------|---|
| 0 | 105.64 | 133.28 | 47.92 | 85.36 | +23.8% | 0.58 |
| 1 | 123.30 | 133.52 | 53.09 | 80.43 | +53.3% | 0.19 |
| 2 | 117.97 | 133.74 | 52.93 | 80.82 | +46.0% | 0.30 |
| 3 | 103.09 | 133.45 | 51.73 | 81.72 | +26.2% | 0.59 |
| 4 | 116.47 | 133.23 | 51.54 | 81.69 | +42.6% | 0.33 |
| 5 | 126.51 | 133.39 | 48.86 | 84.53 | +49.7% | 0.14 |
| 6 | 125.66 | 133.39 | 47.53 | 85.86 | +46.4% | 0.16 |
| 7 | 136.76 | 133.19 | 46.99 | 86.19 | +58.7% | -0.08 |
| 8 | 115.44 | 141.43 | 51.46 | 89.97 | +28.3% | 0.51 |
| 9 | 121.32 | 140.33 | 47.77 | 92.56 | +31.1% | 0.40 |
| 10 | 191.59 | 141.59 | 53.98 | 87.62 | +118.7% | -0.93 |
| 11 | 112.98 | 140.46 | 51.93 | 88.53 | +27.6% | 0.53 |
| 12 | 127.09 | 140.53 | 49.70 | 90.84 | +39.9% | 0.27 |
| 13 | 117.68 | 140.02 | 51.74 | 88.28 | +33.3% | 0.43 |
| 14 | 126.59 | 141.02 | 46.57 | 94.45 | +34.0% | 0.31 |
| 15 | 128.27 | 140.36 | 52.55 | 87.81 | +46.1% | 0.23 |

### B. Optimizer Step Bias (所有 Case)

| Case | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| PP2_EP8 | +7.4% | 5.3% | -4.5% | +12.6% |
| PP4_EP4 | +14.6% | 6.0% | -0.1% | +18.8% |
| PP8_EP2 | +16.2% | 4.6% | -0.2% | +18.3% |

### C. Comm Breakdown per Case (Rank0, per-microbatch)

**Forward:**

| Case | exp_allgather | exp_all_to_all | Total |
|------|---------------|----------------|-------|
| PP2 (24 layers/stage) | 12.91ms | 11.02ms | ~24ms |
| PP4 (12 layers/stage) | 6.49ms | 4.69ms | ~11ms |
| PP8 (6 layers/stage) | 1.84ms | 1.64ms | ~3.5ms |

**Backward:**

| Case | exp_all_to_all | Total |
|------|----------------|-------|
| PP2 | 23.53ms | ~24ms |
| PP4 | 10.21ms | ~10ms |
| PP8 | 3.72ms | ~4ms |

### D. Simulator 代码关键路径

| 代码位置 | 功能 | 影响 |
|----------|------|------|
| `simu_engine.py:extract_sub_operations` | 将 DB profile 拆为 sub_comp + comm | sum(sub_comp) = D_DB |
| `simu_engine.py:630 (can_overlap)` | comp/comm 是否可并行 | 默认 False → 串行 |
| `simu_engine.py:IndividualTimeline` | Timeline 管理 | comp_timeline + comm_timeline 分离但串行 |
| `simu_engine.py:add_sub_ops_according_to_profile_dict` | 从 DB dict 填充 op duration | operation.hidden_duration = D_DB |
