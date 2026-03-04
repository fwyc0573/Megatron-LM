## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Added implementation assumptions and workload notes |

# Notes

## 固定参数
- Model: Qwen3-A3B MoE (stage-1 full profile)
- Precision: BF16
- hidden size=2048, num layers=48, num heads=32, num query groups=4
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
