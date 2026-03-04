## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Add issue tracker for MoE engine debug session |
| 2026-03-02 | Append H800 16-GPU comparison issues and residual error pattern |
| 2026-03-04 | Added Qwen3 + DeepSeek-V3-variant H800 16-GPU comparison issue (systematic overestimation) |
| 2026-03-04 | Added Issue-07 deep-dive attribution: schedule microbatch mismatch (primary), comp bias (secondary), waiting amplification (tertiary) |
| 2026-03-04 | Updated Issue-07/08 after official GBS-fix rerun and residual per-op evidence |
| 2026-03-04 | Added Issue-11 for DeepSeek comp mismatch after enforcing last-iteration scaling reference |
| 2026-03-04 | Added Issue-12 for DeepSeek rank0/stage-aligned discrepancy and comm-adjacent comp contamination evidence |

# Issues

## Closed Issues

### Issue-01: MoE simulate mis-detected as Dense
- Status: Closed
- Symptom: MoE case 出现 `Dense 模型，选择了 2 个ranks`
- Root Cause: 旧检测逻辑依赖有限信号，simulate 场景易误判
- Resolution:
  - 扩展检测来源（topology + stage ops + database ops）
  - simulate + `exp_size>1` 且检测 Dense 时 fail-fast

### Issue-02: MoE simulate not using all ranks
- Status: Closed
- Symptom: timeline/barrier 不完整，E2E 系统性偏低
- Resolution:
  - MoE 直接强制 `selected_ranks = all ranks`

### Issue-03: TP barrier skipped in MoE simulate
- Status: Closed
- Symptom: TP peer 被清空，依赖链被过度放松
- Resolution:
  - MoE simulate 禁用 TP shortcut，保留 TP peer matching

### Issue-04: EP/EXP semantic ambiguity
- Status: Closed
- Symptom: group_kind/group_size 推断存在歧义风险
- Resolution:
  - comm op 显式映射表
  - schedule allreduce group-size 显式估算
  - `database_profile` filename token (`expX/epX`) 语义校验 fail-fast

## Open Issues

### Issue-05: Residual E2E gap after semantic fixes (2ep cases especially)
- Status: Open
- Impact: a800_2ep/h800_2ep 仍有 `~14%/~17%` 低估
- Evidence:
  - 慢 rank 有效通信耗时对比中，`recv_backward` / `exp_all_to_all`（以及部分 case 的 `dp_allreduce`）仍偏差较大
- Suspected Cause:
  - profile 中真实同步开销与当前 comm predictor/engine wait 重建仍有差
- Proposed Follow-up:
  1. 对 `collective-sim` 按 A800/H800 拓扑和消息规模做 `exp_all_to_all`/`dp_allreduce` 标定
  2. 增强 `recv_backward` 等 PP 关键路径等待建模
  3. 增加自动化 per-op effective latency 对比报表用于回归

### Issue-06: H800 16-GPU high-EXP configuration shows overestimation tail
- Status: Open
- Impact: `pp2_tp1_exp8_expn8_dp8_nl8_hs4096_sl1024` case 出现 `+9.17%` simulate overestimation
- Evidence:
  - `task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_moe_collective_sim_comparison_2026-03-02.json`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-02_h800_16gpus_moe_collective_sim.md`
- Suspected Cause:
  - 高 `EXP` 下 `exp_all_to_all` / `exp_dp_allreduce` 与 PP 等待链条组合时，collective predictor + engine wait replay 仍有偏差
- Proposed Follow-up:
  1. 对该 case 做 per-op effective latency 分解（top-k 路径：`exp_all_to_all`, `exp_dp_allreduce`, `recv_backward`）
  2. 对 `collective-sim` 对应消息规模执行 H800 16-GPU 点校准并回归
  3. 增加 high-EXP case 的 CI regression threshold（例如 `abs(gap)<=6%`）

### Issue-07: Qwen3 + DeepSeek-V3-variant H800 16-GPU shows systematic overestimation
- Status: Open
- Impact:
  - 初始症状（修复前）：6-case 全为正 gap，整体 `mean gap=+104.66%`。
  - GBS 修复后（官方复跑）：`mean gap=+14.41%`，系统性同向过估已消失，但 residual 仍超出论文级 fidelity 目标。
- Evidence:
  - `task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_qwen3_deepseek_v3_variant_collective_sim_comparison_2026-03-04.json`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-04_h800_16gpus_qwen3_deepseek_v3_variant_collective_sim.md`
  - Attribution report:
    - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/error_attribution_qwen3_deepseek_collective_sim_2026-03-04.json`
    - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/error_attribution_qwen3_deepseek_collective_sim_2026-03-04.md`
  - Post-fix residual per-op:
    - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/residual_per_op_after_gbs_fix_2026-03-04.json`
    - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/residual_per_op_after_gbs_fix_2026-03-04.md`
  - Step1 inputs/schedule evidence:
    - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_data_organization_summary_2026-03-04.json`
    - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04.json`
    - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04_gbs_fix.json`
- Suspected Cause:
  - Root-cause attribution (quantified):
    1. **Primary**: schedule microbatch 口径不一致。当前 schedule 使用 `GBS=4*PP*DP`，A/B 改为 `GBS=2*PP*DP` 后，6-case mean gap 从 `+104.66%` 降至 `+14.41%`。
    2. **Secondary**: scaling trace comp 偏高（distributed-comp vs scaling-comp 均值 `+38.28%`，DeepSeek 子集更高）。
    3. **Tertiary**: simulate 慢 rank 的通信等待（`comm_waiting`）占比高，top 路径集中在 `recv_backward`/`send_forward`/`exp_all_to_all`。
- Proposed Follow-up:
  1. 继续收敛 residual gap（`+14.41%`）：
     - 在修正 schedule 后执行 per-op effective latency 分解（重点 `forward_step/backward_step/optimizer_step`）。
  2. 对 comm waiting 链路做针对性审计：
     - 重点检查 `recv_backward` / `send_forward` 的匹配与等待传播是否过于保守。
  3. 深入 comp 偏差来源：
     - 对 scaling/distributed 的 trace 采样窗口、迭代口径和 router 负载一致性做复核（尤其 DeepSeek-V3-variant）。

### Issue-08: Schedule generation uses mismatched global_batch_size formula for this dataset
- Status: Open
- Impact:
  - 导致 simulate 的 microbatch 数偏大，直接放大 E2E。
- Evidence:
  - rank0 trace 计数显示 distributed 侧 `forward_step` 次数分别为 `4/8/16`（对应 `PP=2/4/8`），与 `num_microbatches=2*PP` 一致；
  - 当前 schedule 生成参数为 `GBS=4*PP*DP`，对应 `num_microbatches=4*PP`，与 trace 口径不一致；
  - A/B 结果显示该项解释 baseline gap 的主要部分（平均 `106.95%`）。
- Proposed Follow-up:
  1. 将 Step1 schedule 生成默认口径切换到 `GBS=2*PP*DP`（针对该批次数据）。
  2. 在 schedule summary 中增加 `num_microbatches` 与 trace 计数一致性校验（fail-fast）。

### Issue-10: Post-fix residual remains high for DeepSeek-V3-variant subset
- Status: Open
- Impact:
  - GBS 修复后 Qwen3 子集均值接近收敛（`-2.00%`），但 DeepSeek-V3-variant 子集仍有较高残差（均值 `+30.82%`，最差 `+86.58%`）。
- Evidence:
  - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-04_h800_16gpus_qwen3_deepseek_v3_variant_collective_sim.md`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/residual_per_op_after_gbs_fix_2026-03-04.md`
- Suspected Cause:
  - anchor rank residual 显示 `sub_comp` 与 `exp_allgather/exp_all_to_all` 路径仍有明显偏差，且 waiting 成分未完全解释。
- Proposed Follow-up:
  1. 对 DeepSeek 的 3 个 case 做逐 rank per-op residual（不仅限 slowest rank）；
  2. 用 Nsight compute-only 指标交叉验证 `sub_comp` 偏差是否为真实计算差异；
  3. 对 `exp_*` 通信相关 op 的 message-size 映射和等待传播做专项审计。

### Issue-11: DeepSeek trace-level comp mismatch persists even with last-iteration scaling reference
- Status: Open
- Impact:
  - 在严格使用 last-iteration scaling comp、并按 distributed `total - all_subop_sum` 计算 actual comp 后，DeepSeek 3-case 仍存在较大 comp 偏差。
- Evidence:
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_last_iter_verification_2026-03-04.json`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_last_iter_verification_2026-03-04.md`
  - 关键数据：
    - `pp2_tp1_exp4...`: fwd `+78.83%`, bwd `+101.88%`
    - `pp2_tp1_exp8...`: fwd `+32.07%`, bwd `+61.32%`
    - `pp4_tp1_exp4...`: fwd `+4.68%`, bwd `+32.04%`
- Clarification:
  - 引擎读取 `database_profile` 时，`process_mg_profile_files()` 对相同 `cmd_name` 采用后写覆盖（即最后一次记录生效），因此“仅最后一轮作为 comp reference”在当前路径中是成立的。
- Proposed Follow-up:
  1. 对 `sub_comp` 在 `forward_step/backward_step` 内的组成做分段归因（attention/mlp/router）；
  2. 对 DeepSeek `PP2` 两个 `EP` 配置进行同 rank、同 stage 的 kernel-level compute-only 对齐验证；
  3. 审计 scaling run 的 router/dispatch 负载稳定性，确认 last-iter 是否仍有分布漂移。

### Issue-09: Scaling comp trace bias remains after schedule correction
- Status: Open
- Impact:
  - schedule 修正后仍有 residual gap（6-case mean 约 `+14.41%`；DeepSeek 子集均值 `+30.82%`）。
- Evidence:
  - distributed-comp vs scaling-comp 的 trace 级偏差均值：Qwen3 `+28.60%`，DeepSeek-V3-variant `+47.95%`。
- Proposed Follow-up:
  1. 固定 schedule 后重新采样 scaling/distributed，统一 trace 窗口与迭代口径。
  2. 对 DeepSeek case 增加 Nsight compute-only 交叉验证，判断是否为纯计算基线偏高。

### Issue-12: DeepSeek rank0/stage-aligned view shows backward-specific positive bias and comm-adjacent comp contamination
- Status: Open
- Impact:
  - 当按 `rank0 + same stage` 严格对齐后，forward 误差明显收敛（约 `0% ~ -4%`），但 backward 仍持续正偏（`+10% ~ +30%`）。
  - 说明争议主要不在“是否使用 last-iteration”，而在 comp 口径中是否混入了未计入 comm 的 comm-adjacent 开销。
- Evidence:
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_rank0_stage_aligned_2026-03-04.json`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_rank0_stage_aligned_2026-03-04.md`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_decomposition_rank0_2026-03-04.json`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/deepseek_comp_error_decomposition_rank0_2026-03-04.md`
  - 分解结果（rank0）显示：`comp_gap = total_gap + (dist_comm - sim_comm)` 中，backward 的 `(dist_comm - sim_comm)` 项主导正偏。
- Suspected Cause:
  - scaling trace 中 `all_to_all` 子操作 duration 记录为 0，但 scaling 分支仍执行 `contiguous()/copy_` 等张量搬运，导致 comm-adjacent work 被计入 parent op comp。
  - 相关实现位于 `megatron/core/tensor_parallel/mappings.py` 的 `_profiled_all_to_all_single()` scaling 分支与 `_AllToAll.backward()`。
- Proposed Follow-up:
  1. 增加 `all_to_all` scaling 分支 comm-adjacent kernel 的可观测时延口径（避免 0-duration sub-op + parent comp 吸收）。
  2. 用 kernel-ground-truth（Nsight/CMD labels）对 `forward_step/backward_step` 做 compute-only 对齐，隔离 comm-adjacent 拷贝。
  3. 在报告中固定三层口径（global aggregate / stage aggregate / rank0 exact-stage）并禁止混用解释。
