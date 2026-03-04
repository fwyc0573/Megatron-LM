## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-03 | Added PP=8, world_size=1024 TP scaling simulation comparative results |

# A800 GPT-175B TP Scaling Results (PP=8, world=1024)

## Metric Definition
- `stage_i_ms`: from simulate log, per representative rank (`[0,128,256,384,512,640,768,896]`) using max observed `last_operation_time` for that rank.
- `iteration_ms`: `max(stage0_ms..stage7_ms)`.
- `tokens_per_sec = (global_batch_size * seq_len) / (iteration_ms / 1000)`.
- `samples_per_sec = global_batch_size / (iteration_ms / 1000)`.
- Fixed: `global_batch_size=256`, `seq_len=2048`.

## Comparative Table

| config | tp | pp | dp | node_size | stage0_ms | stage1_ms | stage2_ms | stage3_ms | stage4_ms | stage5_ms | stage6_ms | stage7_ms | iteration_ms | tokens/s | samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tp8_pp8_dp16 | 8 | 8 | 16 | 8 | 34431.88 | 33299.28 | 32592.64 | 31929.50 | 31174.36 | 30409.64 | 29736.05 | 34431.88 | 34431.88 | 15226.82 | 7.4350 |
| tp16_pp8_dp8 | 16 | 8 | 8 | 16 | 60503.28 | 59633.07 | 58984.00 | 58367.40 | 57677.22 | 57050.24 | 56386.89 | 60503.28 | 60503.28 | 8665.45 | 4.2312 |
| tp32_pp8_dp4 | 32 | 8 | 4 | 32 | 106201.94 | 105502.13 | 104886.02 | 104252.30 | 103550.68 | 102930.89 | 102207.25 | 106201.94 | 106201.94 | 4936.71 | 2.4105 |
| tp64_pp8_dp2 | 64 | 8 | 2 | 64 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## TP64 Blocked Reason
- Strict GPT-175B (`num_attention_heads=96`) fails `96 % 64 == 0` divisibility requirement.
- Evidence file: `logs/tp64_blocking_evidence.md`.

## Key Observation
- Under fixed `GBS=256`, throughput decreases as TP increases from 8 -> 16 -> 32.
- In this setup, larger TP increases communication cost faster than any compute-side gain from additional intra-node NVLink grouping.

## Caveats
1. Node size 16/32 full NVLink is a hypothetical topology assumption.
2. Stage time extraction used `last_operation_time` maxima from simulator runtime log because `--no-visualize` suppresses direct `sum_time` printouts.
3. For TP32 runtime, a runtime-only canonical participant cache was used in wrapper script `tests/performance/run_simu_with_collective_cache.py` to reduce repeated equivalent collective-sim calls.
