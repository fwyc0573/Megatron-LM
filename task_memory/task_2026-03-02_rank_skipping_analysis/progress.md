## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Add execution progress log for MoE engine debug, testing, and 4-case replay |
| 2026-03-02 | Append H800 16-GPU full 8-case simulate/profile comparison validation progress |
| 2026-03-03 | Consolidate dense/MoE rank-skipping exploratory analysis with 8-GPU case communication semantics and simulator guidance |
| 2026-03-04 | Added H800 16-GPU Qwen3 + DeepSeek-V3-variant Step1/Step2 execution progress with organization/scheduling evidence and 6-case collective-sim comparison |
| 2026-03-04 | Added 6-case error attribution analysis (schedule A/B, comp-bias, waiting decomposition) for systematic overestimation |
| 2026-03-04 | Applied Step1 GBS fix from runtime scripts, replaced official rerun report, and added post-fix residual per-op report |
| 2026-03-04 | Added DeepSeek 3-case comp re-verification using last-iteration scaling reference and distributed sub-op subtraction |
| 2026-03-04 | Added DeepSeek rank0/stage-aligned comp re-check and rank0 comp-gap decomposition to isolate comp vs comm-adjacent effects |

# Progress

## 2026-03-02 Session Log

### Completed
1. 实现 Step A~G（engine-only）
   - 完成 MoE 检测扩展、simulate fail-fast、MoE all ranks、TP barrier 保留、EP/EXP 显式映射、group-size fail-fast、filename token 校验、文案去歧义
2. 新增测试并通过
   - `test_simu_engine_moe_detection.py` (5/5)
   - `test_simu_engine_moe_rank_selection.py` (3/3)
   - `test_simu_engine_ep_exp_semantics.py` (7/7)
   - `test_moe_simulate_all_ranks_tp_barrier.py` (1/1)
   - Combined: `16/16 passed`
3. 现有回归测试通过
   - `test_simu_engine_cc_semantics.py` (4/4)
4. 4-case 模板脚本回放完成
   - 所有 case 均显示 MoE all ranks（8/8）

### Quantitative Outcome (current)
- a800_2ep: `806.62 / 939.36 ms` -> `-14.13%`
- a800_4ep: `836.63 / 981.14 ms` -> `-14.73%`
- h800_2ep: `358.99 / 431.77 ms` -> `-16.86%`
- h800_4ep: `334.51 / 357.74 ms` -> `-6.49%`

### Baseline Comparison
- a800_2ep: `-16.56% -> -14.13%`（+2.43 pct-point）
- a800_4ep: `-39.13% -> -14.73%`（+24.40 pct-point）
- h800_2ep: `-18.64% -> -16.86%`（+1.78 pct-point）
- h800_4ep: `-29.05% -> -6.49%`（+22.56 pct-point）

### Main Finding
- 语义级根因（MoE误判Dense + rank裁剪 + TP barrier shortcut + EP/EXP 混淆）已修复
- 残余 gap 主要来自 comm realism mismatch（`recv_backward`、`exp_all_to_all`、部分 `dp_allreduce`）

### References
- Main code: `megatron-sim-engine/src/core/simu_engine.py`
- Full report: `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-02_moe_engine_simulate_debug.md`

## 2026-03-02 H800 16-GPU Supplement

### Completed
1. 完成 `h800_16gpus_moe` 全 8 个 case 的 simulate/profile 对比验证（`collective-sim` backend）
2. 对每个 case 统一按 timeline `max(finish_time)` 提取 E2E
3. 结果与统计已落盘：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_moe_collective_sim_comparison_2026-03-02.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-02_h800_16gpus_moe_collective_sim.md`

### Quantitative Outcome (H800 16-GPU, 8 cases)
- Mean gap: `-1.86%`
- Mean absolute gap: `4.18%`
- Median absolute gap: `3.77%`
- Worst underestimation: `-7.93%`
- Worst overestimation: `+9.17%`

### Main Finding
- 16-GPU 数据集上，修复后的 engine 已不再呈现此前“MoE误判Dense+rank裁剪”导致的系统性低估。
- 误差主导项转为 comm realism 偏差，且在高 `EXP` + 特定 `expn` 组合下出现 overestimation（`+9.17%`）。

## 2026-03-03 Documentation Consolidation

### Completed
1. 将 Dense/MoE 的 rank-skipping exploratory analysis 统一整理进单文档：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/analysis.md`
2. 文档新增并固化以下内容：
   - Dense 与 MoE 的最终推荐最小测量集合
   - `PP=2,TP=1,EP=2,DP=4,world=8` 的 rank 映射与代表 rank 选择
   - DP/EP/EXP_DP 通信在训练流程中的发生时机、参与 rank、通信目的
   - 对“routing 导致 duplicate 失效”的风险分析与 spot-check 回退策略
   - 与 `megatron-sim-engine` 当前 rank 选择逻辑的对齐说明

### Main Finding
- 在当前 fork 语义下，Dense 可按 PP 代表 rank 测量；MoE 推荐 `PP × EP` 主测并辅以 DP spot-check。
- 对于用户关注的 `rank0` vs `rank2`（同 `(PP,EP)` 不同 DP）问题：当前 Scaling 路径存在 pre-fixed routing 机制，duplicate 假设更容易成立；但若目标是逼近真实动态 routing，必须保留一致性校验门槛。

## 2026-03-04 H800 16-GPU Qwen3 + DeepSeek-V3-variant Execution

### Completed
1. 完成 Step1 数据组织（按模型拆分）并校验：
   - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_qwen3_moe/*`
   - `megatron-sim-engine/simulation_inputs/megatron_operation_log/h800_16gpus_deepseek_v3_variant_moe/*`
   - 每个 case 均满足：`database_profile=16`、`global_ranks_profile=16`、rank 覆盖 `0..15`
2. 完成 Step1 调度生成与校验：
   - 逐 case 调用 `mg_test.py` 生成 stage plan
   - 每个 case 的 `schedule/*.txt` 数量等于 `PP`，且 stage 覆盖完整（`stage0..stage(PP-1)`）
3. 完成 Step2 simulate/profile 对比（6-case, collective-sim）：
   - 统一口径：`E2E = max(op.finish_time)` across all ranks
   - 结果落盘：
     - `task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_qwen3_deepseek_v3_variant_collective_sim_comparison_2026-03-04.json`
     - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-04_h800_16gpus_qwen3_deepseek_v3_variant_collective_sim.md`
4. 证据文件落盘：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_data_organization_summary_2026-03-04.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_data_organization_summary_2026-03-04.md`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04.md`

### Quantitative Outcome (H800 16-GPU, 6 cases)
- Mean gap: `+104.66%`
- Mean absolute gap: `104.66%`
- Median absolute gap: `82.04%`
- Max gap: `+242.45%` (`pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048`)
- Min gap: `+37.82%` (`pp4_tp1_exp4_expn32_dp4_nl32_hs2048_sl2048`)

### Main Finding
- 本轮 6-case 全部可运行并完成比对，但结果呈现系统性 overestimation（所有 case gap 为正）。
- Qwen3 与 DeepSeek-V3-variant 的分模型统计差异明显，DeepSeek-V3-variant overestimation 更高（均值 `+140.53%`）。
- 在本次执行中启用 `collective-sim` 预测缓存以缩短运行时间；缓存统计：hits=`18895`, misses=`55`。

## 2026-03-04 Error Attribution (Issue-07 Deep Dive)

### Completed
1. 完成 6-case 全量误差归因复算（simulate/profile + schedule A/B）：
   - 基线：当前 schedule（`GBS=4*PP*DP`）
   - A/B：替代 schedule（`GBS=2*PP*DP`，其余参数保持一致）
2. 完成 trace 级别 comp 偏差分析（distributed-comp vs scaling-comp）：
   - 使用 `compare_qwen_trace_comp.py` 的解析逻辑（distributed 侧扣 comm 子操作）
3. 完成 slowest-rank timeline 分解：
   - 输出 `comp_total_ms`、`comm_total_ms`、`comm_duration_ms`、`comm_waiting_ms`
   - 提取 top-k comm 操作（按 total/waiting）
4. 新增证据产物：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/error_attribution_qwen3_deepseek_collective_sim_2026-03-04.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/error_attribution_qwen3_deepseek_collective_sim_2026-03-04.md`

### Quantitative Outcome
- 基线 6-case mean gap：`+104.66%`
- A/B（`GBS=2*PP*DP`）后 mean gap：`+14.41%`
- baseline gap 中由 schedule 口径解释的占比（均值）：`106.95%`
- trace 级 comp 偏差（scaling vs distributed-comp，6-case 均值）：`+38.28%`

### Main Finding
- 误差主因已定位为混合型：
  1. **Primary**: schedule 生成口径与真实 trace 微批次/迭代口径不一致（本轮使用 `4*PP*DP` 明显偏大）。
  2. **Secondary**: scaling trace 的 comp 基线系统性偏高（Qwen3 约 `+28.60%`，DeepSeek-V3-variant 约 `+47.95%`）。
  3. **Tertiary**: simulate 慢 rank 的 comm waiting 占比高（尤其 `recv_backward/send_forward`），存在等待链路放大现象。

## 2026-03-04 Official Rerun After Step1 GBS Fix

### Completed
1. 根据实际运行脚本口径修复 Step1 schedule：
   - 脚本来源：`examples/pretrain_qwen3_30b_a3b_moe.sh`, `examples/pretrain_deepseek_v3_moe.sh`
   - 公式：`NUM_MICBATCH=4*PP`, `GBS=NUM_MICBATCH*MICRO_BATCH_SIZE*(GPUS_PER_NODE/TP/PP)`
   - 在本批 6-case 条件（`MICRO_BATCH_SIZE=1`, `GPUS_PER_NODE=8`, `TP=1`）下，得到 `GBS=32`（全 case）
2. 重新生成并覆盖 6-case schedule（stage 覆盖校验 PASS）：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04_gbs_fix.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04_gbs_fix.md`
3. 用修正 schedule 完成 Step2 正式复跑，并替换官方结果文件：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_qwen3_deepseek_v3_variant_collective_sim_comparison_2026-03-04.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-04_h800_16gpus_qwen3_deepseek_v3_variant_collective_sim.md`
4. 产出 post-fix residual per-op 报告：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/residual_per_op_after_gbs_fix_2026-03-04.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/residual_per_op_after_gbs_fix_2026-03-04.md`

### Quantitative Outcome (Official Rerun, 6 cases)
- Mean gap: `+14.41%`（from `+104.66%`）
- Mean absolute gap: `30.36%`
- Median absolute gap: `20.92%`
- Worst underestimation: `-22.54%`
- Worst overestimation: `+86.58%`

### Main Finding
- Step1 的 GBS 口径修复后，“全 case 同向过估”现象已消失（case 结果出现正负混合）。
- residual 误差仍显著，且 DeepSeek-V3-variant 子集残差更大（均值 `+30.82%`），后续需在 comp 偏差与 waiting 链路上继续收敛。

## 2026-03-04 DeepSeek Comp Error Deep-Dive (3 cases)

### Completed
1. 对 `h800_16gpus_deepseek_v3_variant_moe` 的 3 个 case 执行 comp 误差复核：
   - `pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048`
   - `pp2_tp1_exp8_expn32_dp8_nl32_hs2048_sl2048`
   - `pp4_tp1_exp4_expn32_dp4_nl32_hs2048_sl2048`
2. 明确并执行以下口径：
   - simulated comp：`database_profile` 每个 rank 中 `forward_step/backward_step` 的**最后一次**记录（last iteration）
   - actual comp：`global_ranks_profile` 中 `comp = total_step_duration - sum(all sub-op durations)`
3. 核验 scaling/distributed trace 语义：
   - scaling rank0 为 `fwd=3, bwd=3, opt=3`，且无 PP p2p op；
   - distributed rank0 存在 PP p2p op，且 `fwd/bwd` 次数符合 pipeline 微批行为。
4. 产出证据文件：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_last_iter_verification_2026-03-04.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_last_iter_verification_2026-03-04.md`

### Quantitative Outcome (steady-state actual as primary baseline)
- `pp2_tp1_exp4...`:
  - forward: sim `116.08ms` vs actual `64.91ms` (`+78.83%`)
  - backward: sim `115.00ms` vs actual `56.96ms` (`+101.88%`)
- `pp2_tp1_exp8...`:
  - forward: sim `84.21ms` vs actual `63.76ms` (`+32.07%`)
  - backward: sim `88.84ms` vs actual `55.07ms` (`+61.32%`)
- `pp4_tp1_exp4...`:
  - forward: sim `30.89ms` vs actual `29.51ms` (`+4.68%`)
  - backward: sim `39.01ms` vs actual `29.54ms` (`+32.04%`)

### Main Finding
- 对比 `pp2_exp4` 与 `pp2_exp8`，distributed actual comp 差异很小（forward 约 `1.8%`，backward 约 `3.4%`），但 simulated comp 差异明显更大（forward 约 `37.8%`，backward 约 `29.5%`）。
- 这说明 residual 并非仅来自 distributed 侧口径，DeepSeek 在 scaling comp 基线本身存在结构性偏高与配置敏感性。

## 2026-03-04 DeepSeek Comp Re-Check (Rank0/Stage Aligned)

### Completed
1. 按用户指定口径执行三项强约束复核：
   - 验证 scaling 来源为 `database_profile`（`description=simulation`，comm sub-op duration 全 0）
   - 验证 simulated comp 仅使用每个 rank 的最后一次 `forward_step/backward_step`
   - 验证 actual comp 使用 distributed `total - sum(comm sub-op durations)`
2. 新增 `rank0 exact-stage` 与 `stage-wise` 两套对齐视图，避免全 rank 混合均值误导。
3. 新增 rank0 误差分解：
   - `comp_gap = (sim_total - dist_total) + (dist_comm - sim_comm)`
   - 用于区分 total 差值与 comm 口径差值贡献。
4. 产出证据文件：
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_rank0_stage_aligned_2026-03-04.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_rank0_stage_aligned_2026-03-04.md`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_decomposition_rank0_2026-03-04.json`
   - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_decomposition_rank0_2026-03-04.md`

### Quantitative Outcome (rank0 exact-stage)
- `pp2_tp1_exp4...`:
  - forward: sim `56.94ms` vs actual `59.48ms` (`-4.26%`)
  - backward: sim `79.74ms` vs actual `61.23ms` (`+30.23%`)
- `pp2_tp1_exp8...`:
  - forward: sim `55.73ms` vs actual `57.77ms` (`-3.53%`)
  - backward: sim `73.75ms` vs actual `59.10ms` (`+24.78%`)
- `pp4_tp1_exp4...`:
  - forward: sim `23.34ms` vs actual `23.35ms` (`-0.03%`)
  - backward: sim `38.41ms` vs actual `34.85ms` (`+10.20%`)

### Main Finding
- 争议点已确认：`rank0/stage` 对齐下，forward 基本对齐；主要偏差集中在 backward。
- rank0 分解显示 backward 的 `missing_comm_term`（`dist_comm - sim_comm`）显著大于 `total_gap`，说明误差不只是纯 compute mismatch，还包含 scaling 侧 comm-adjacent 开销被吸入 comp 的影响。
