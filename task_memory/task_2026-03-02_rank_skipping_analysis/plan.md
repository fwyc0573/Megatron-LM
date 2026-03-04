## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Create structured plan for MoE engine simulating debug/verification (A~G + validation matrix) |
| 2026-03-02 | Extend plan with H800 16-GPU dataset supplemental validation |
| 2026-03-04 | Extend plan with Qwen3 + DeepSeek-V3-variant H800 16-GPU Step1/Step2 execution and reporting |

# Plan: Engine 层 MoE Simulating Debug（全 ranks + TP barrier + EXP/EP 语义校正）

## Scope
- 仅修改 `megatron-sim-engine` 的 engine 路径：`src/core/simu_engine.py`
- 不改训练侧 trace/database 数据生成逻辑
- 本轮目标：修复 MoE simulate 语义错误并完成可复现验证

## Objectives
1. simulate 模式下 MoE 不再误判 Dense
2. MoE simulate 强制 all ranks 参与 timeline
3. MoE simulate 保留 TP barrier
4. engine 内 EXP/EP 语义与 `database_profile` 文件 token 语义明确区分并 fail-fast
5. 4 个 Mixtral case 的 simulate/profile E2E gap 相比 baseline 明显收敛

## Implementation Steps
- [x] Step A: 扩展 `_detect_model_type(...)` 检测来源并接入调用点，增加 simulate + `exp_size>1` fail-fast
- [x] Step B: `_select_optimization_ranks(...)` 对 MoE 返回 all ranks
- [x] Step C: TP matching 在 MoE simulate 下禁止 shortcut，保留 barrier
- [x] Step D: `_get_comm_operation_kind_and_parallel_dimension(...)` 改为显式 op-name 映射并做语义一致性校验
- [x] Step E: `process_mg_files(...)` schedule 分支 allreduce group-size 估算显式化并 fail-fast
- [x] Step F: 增加 `database_profile` 文件名 `expX/epX` 解析与校验，并接入 init 流程
- [x] Step G: 可视化标题与文件名文案改为 `EXP`/`EP` 双显示，去歧义

## Validation Matrix
- [x] Unit tests: detection / rank selection / EP-EXP semantics
- [x] Integration test: MoE simulate all-ranks + TP barrier
- [x] Existing semantic regression test (`test_simu_engine_cc_semantics.py`)
- [x] 4-case replay: a800_2ep, a800_4ep, h800_2ep, h800_4ep

## Acceptance Criteria
- [x] MoE case 不再出现 `Dense 模型，选择了 2 个ranks`
- [x] MoE simulate `selected_ranks == world_size`
- [x] TP 通信在 MoE simulate 下保留 peer 匹配
- [x] EP/EXP 映射覆盖通过测试
- [x] 4 case 的 E2E gap 对 baseline 全部改善（其中 4ep case 显著改善）

## Deliverables
- 代码变更：`megatron-sim-engine/src/core/simu_engine.py`
- 新增测试：
  - `megatron-sim-engine/tests/unit/test_simu_engine_moe_detection.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_moe_rank_selection.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_ep_exp_semantics.py`
  - `megatron-sim-engine/tests/integration/test_moe_simulate_all_ranks_tp_barrier.py`
- 测试报告：`task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-02_moe_engine_simulate_debug.md`

## Supplemental Validation (H800 16-GPU)
- [x] 对 `h800_16gpus_moe` 完成 8-case simulate/profile 对比
- [x] 输出汇总 JSON：
  - `task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_moe_collective_sim_comparison_2026-03-02.json`
- [x] 输出补充测试报告：
  - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-02_h800_16gpus_moe_collective_sim.md`

## Supplemental Validation (H800 16-GPU, Qwen3 + DeepSeek-V3-variant)
- [x] 完成 Step1 数据组织（模型分目录）并校验输入完整性
- [x] 完成 Step1 调度生成与 stage 覆盖校验
- [x] 完成 Step2 simulate/profile（collective-sim）6-case 对比
- [x] 输出证据与报告：
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_data_organization_summary_2026-03-04.json`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/logs/step1_schedule_summary_2026-03-04.json`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/h800_16gpus_qwen3_deepseek_v3_variant_collective_sim_comparison_2026-03-04.json`
  - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-04_h800_16gpus_qwen3_deepseek_v3_variant_collective_sim.md`
