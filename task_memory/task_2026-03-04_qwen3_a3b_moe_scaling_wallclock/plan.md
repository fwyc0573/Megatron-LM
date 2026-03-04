## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Created implementation plan for Qwen3-A3B MoE scaling wall-clock task |

# 任务计划：Qwen3-A3B MoE Scaling Wall-clock 扫描

## Goal
在单卡 scaling mode 下，完成 Qwen3-A3B MoE 四组并行配置的 wall-clock 测量，采用 `PP × EP` rank skipping，并输出单次迭代与 `×5` 估算值。

## Scope
- 新增执行脚本：`examples/qwen3_a3b_moe_scaling_wallclock_scan.sh`
- 新增测试脚本：
  - `tests/unit/test_qwen3_a3b_moe_scaling_wallclock_config.sh`
  - `tests/integration/test_qwen3_a3b_moe_scaling_wallclock_dryrun.sh`
- 产出 CSV：`docs/data/qwen3_a3b_moe_scaling_wallclock_timing.csv`
- 产出测试报告与过程记录

## Acceptance Criteria
1. 四组配置均通过 MoE 约束校验（`ws == pp*tp*dp` 且 `dp % ep == 0`）。
2. rank skipping 按公式 `rank = pp_stage * tp * dp + exp_rank * tp` 选择，`measured_ranks_count = pp * ep`。
3. CSV 列顺序与字段名称符合要求：包含 `ep_size` 与 `estimated_5_iters_seconds`。
4. 通过静态检查、unit test、dry-run integration test。
5. 尝试真实测量；若受环境或时长限制，记录阻塞证据与影响。
