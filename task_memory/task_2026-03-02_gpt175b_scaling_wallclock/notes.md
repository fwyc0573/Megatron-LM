## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Added implementation assumptions and environment observations |
| 2026-03-02 | Added measured wall-clock results from real run |

# Notes

## 固定参数
- Model: GPT-175B dense
- FP16
- hidden size=12288, num layers=96, num heads=96
- seq length=2048
- micro batch size=1
- global batch size=dp

## 配置列表
- 256/16/8/2
- 1024/16/8/8
- 4096/32/8/16
- 8192/32/8/32

## Rank 优化公式
`world_rank = pp_stage * tp_size * dp_size`，`pp_stage=0..pp_size-1`。

## 计时口径
- 配置级计时（进入配置到该配置所有代表 rank 完成）
- 单次 iteration（train-iters=1, scaling-min-warmup-iters=0, scaling-profile-iters=1）
- 保留 trace 开销（do-trace=True, trace-subop-sync-mode=global）

## 实测结果（real run）
- CSV: `docs/data/gpt175b_scaling_wallclock_timing.csv`
- 日志: `task_memory/task_2026-03-02_gpt175b_scaling_wallclock/logs/run_gpt175b_wallclock_20260302_115651.log`
- 关键 wall-clock:
  - ws256/pp16/tp8/dp2: 269.702444s
  - ws1024/pp16/tp8/dp8: 270.106722s
  - ws4096/pp32/tp8/dp16: 537.983920s
  - ws8192/pp32/tp8/dp32: 699.711289s
