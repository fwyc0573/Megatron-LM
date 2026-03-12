## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-08 | Added Config3 partial-duration evidence and resumed-rank chunk commands |
| 2026-03-07 | Added setsid-managed full-sweep commands, PIDs, and log targets |
| 2026-03-04 | Added implementation assumptions and workload notes |
| 2026-03-04 | Updated Qwen3 TP=8 compatible query-group setting and runtime blocker notes |
| 2026-03-04 | Added runtime fixes for TE RMSNorm contiguous input and MoE scaling row restore |
| 2026-03-04 | Added resume-run command and interim real timing results (Config1/Config2) |
| 2026-03-06 | Added rank1400 repro result and megatron-sim-engine partial-rank simulation notes |
| 2026-03-06 | Added background full-sweep command/output locations for Config3 and Config4 |

# Notes

## 固定参数
- Model: Qwen3-A3B MoE (stage-1 full profile)
- Precision: BF16
- hidden size=2048, num layers=48, num heads=32, num query groups=8
- ffn hidden size=6144
- num experts=128, moe ffn hidden size=768, moe router topk=8
- seq length=2048
- micro batch size=1
- global batch size=`dp`

## 配置列表
- `256/8/8/4/4` (ws/pp/tp/ep/dp)
- `1024/8/8/16/16`
- `4096/16/8/32/32`
- `8192/16/8/64/64`

## Rank Skipping 公式
MoE 代表 rank 选择：
`world_rank = pp_stage * tp_size * dp_size + exp_rank * tp_size`

含义：
- 对每个 `pp_stage` 保留所有 `exp_rank`
- 固定 `tp_rank=0` 且选择 DP 代表位
- 总数为 `pp_size * ep_size`

## 计时口径
- 配置级 wall-clock：从该配置第一个代表 rank 开始，到最后一个代表 rank 完成。
- 单次 iteration：`train-iters=1`、`scaling-min-warmup-iters=0`、`scaling-profile-iters=1`。
- 估算 5 次：`estimated_5_iters_seconds = single_iter_wallclock_seconds * 5`。

## Resume 运行命令
```bash
SCALE_GPU=0 \
APPEND_CSV=1 \
CONFIG_START_INDEX=1 \
CONFIG_END_INDEX=3 \
OUTPUT_CSV=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/docs/data/qwen3_a3b_moe_scaling_wallclock_timing.csv \
LOG_ROOT=/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/log/qwen3_a3b_moe_scaling_wallclock \
bash /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/examples/qwen3_a3b_moe_scaling_wallclock_scan.sh \
2>&1 | tee /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_resume_20260304_092804.log
```

## 已落盘真实结果（截至 2026-03-04 10:04 UTC）
- `256,8,8,4,4,32,536.436879,2682.184395`
- `1024,8,8,16,16,128,2147.262902,10736.314510`

## 当前运行阻塞
- 先前阻塞 1（已修复）：`transformer_engine RMSNorm` 非连续张量 `view` 异常。
- 先前阻塞 2（已修复）：MoE `token_unpermutation` 在 scaling mode 下 `output.view(self.hidden_shape)` 行数不匹配。

## 运行时修复摘要
- `megatron/core/transformer/custom_layers/transformer_engine.py`
  - TENorm forward wrapper 在 dtype cast 后增加 `contiguous()`，避免 TE RMSNorm `view` 崩溃。
- `megatron/core/transformer/moe/token_dispatcher.py`
  - 新增 `_restore_scaling_token_rows(...)`，在 scaling mode 下对 `token_unpermutation` 输出做确定性行数恢复，再 `view` 回 `hidden_shape`。

## 2026-03-06 Rank1400 Repro Validation
- Repro target: `ws4096_pp16_tp8_ep32_dp32`, `fake_current_rank_id=1400`.
- Environment prerequisite: `CUDA_DEVICE_MAX_CONNECTIONS=1`, `NCCL_DEBUG=WARN`, `CUDA_VISIBLE_DEVICES=0`.
- Result: warmup, `forward_step`, `backward_step`, and `optimizer_step` all completed successfully.
- Evidence log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/repro_rank1400_20260306.log`.

## 2026-03-06 Megatron-Sim-Engine Partial-Rank Run
- Engine switch: `--moe-rank-selection pp-ep` (representative ranks = `PP * EP`, `tp_rank=0`, DP representative).
- Comm backend: `collective-sim` with prediction cache enabled via `cache_path`.
- Batch result CSV: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/megatron_sim_engine_partial_ranks_20260306.csv`.
- Batch test report: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/test_report_2026-03-06_megatron_sim_engine_partial_ranks.md`.

## 2026-03-06 Dedicated Full-Sweep Runs
- `Config3` official run: `CUDA_DEVICE_MAX_CONNECTIONS=1 SCALE_GPU=0 CONFIG_START_INDEX=2 CONFIG_END_INDEX=2 APPEND_CSV=0 OUTPUT_CSV=task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config3_20260306.csv LOG_ROOT=log/qwen3_a3b_moe_scaling_wallclock_final_20260306 bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh`
- `Config4` official run: `CUDA_DEVICE_MAX_CONNECTIONS=1 SCALE_GPU=1 CONFIG_START_INDEX=3 CONFIG_END_INDEX=3 APPEND_CSV=0 OUTPUT_CSV=task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config4_20260306.csv LOG_ROOT=log/qwen3_a3b_moe_scaling_wallclock_final_20260306 bash examples/qwen3_a3b_moe_scaling_wallclock_scan.sh`
- Config3 live log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_20260306.log`
- Config4 live log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config4_20260306.log`
- Config3 live CSV target: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config3_20260306.csv`
- Config4 live CSV target: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config4_20260306.csv`
- A duplicate local `Config3` retry was intentionally cancelled after it was found to overlap with the official `GPU0` run.

## 2026-03-07 Setsid Full-Sweep Runs
- Root-cause observation: earlier `20260306` background attempts did not finish and left no surviving worker processes; outer logs stopped without an explicit Python traceback, consistent with harness-level cleanup of non-independent detached jobs.
- `setsid` verification: a detached `sleep 60` test was re-parented to PID 1 and survived after the launching shell exited.
- `Config3` official run: `CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_DEBUG=WARN SCALE_GPU=0 MASTER_PORT_BASE=26000 CONFIG_START_INDEX=2 CONFIG_END_INDEX=2 APPEND_CSV=0 OUTPUT_CSV=task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config3_setsid_20260307.csv LOG_ROOT=log/qwen3_a3b_moe_scaling_wallclock_final_20260307_setsid setsid bash -lc ...`
- `Config4` official run: `CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_DEBUG=WARN SCALE_GPU=1 MASTER_PORT_BASE=26000 CONFIG_START_INDEX=3 CONFIG_END_INDEX=3 APPEND_CSV=0 OUTPUT_CSV=task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config4_setsid_20260307.csv LOG_ROOT=log/qwen3_a3b_moe_scaling_wallclock_final_20260307_setsid setsid bash -lc ...`
- Config3 PID file: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_setsid_20260307.pid`
- Config4 PID file: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config4_setsid_20260307.pid`
- Config3 live log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_setsid_20260307.log`
- Config4 live log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config4_setsid_20260307.log`
- Config3 per-rank log root: `log/qwen3_a3b_moe_scaling_wallclock_final_20260307_setsid/ws4096_pp16_tp8_ep32_dp32/`
- Config4 per-rank log root: `log/qwen3_a3b_moe_scaling_wallclock_final_20260307_setsid/ws8192_pp16_tp8_ep64_dp64/`

## 2026-03-08 Config3 Transient Crash Audit
- Config3 `20260307` run status at audit: `425/512` representative ranks completed; failure point was after rank `3392` finished all profiled phases.
- Evidence line cluster: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_setsid_20260307.log`.
- Log-file wall-clock for completed chunk (`birth -> mtime`): start `2026-03-07 03:18:57.730793206 +0000`, end `2026-03-07 11:13:30.624781655 +0000`.
- Direct repro command for rank `3392` completed successfully; evidence log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/repro_rank3392_20260308.log`.
- Resume chunk command target: ranks `3400, 3408, ..., 4088` (`87` ranks) on `GPU0`.
- Resume chunk PID file: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_resume_after3392_setsid_20260308.pid`.
- Resume chunk live log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/run_qwen3_a3b_moe_wallclock_config3_resume_after3392_setsid_20260308.log`.
- Resume chunk per-rank log root: `log/qwen3_a3b_moe_scaling_wallclock_config3_resume_20260308/`.
- Resume chunk CSV target: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/qwen3_a3b_moe_scaling_wallclock_config3_resume_after3392_20260308.csv`.
