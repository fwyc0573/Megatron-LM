## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-03 | Initialized notes with locked assumptions and run matrix |
| 2026-03-03 | Updated execution matrix to PP=8 world_size=1024 |
| 2026-03-03 | Added final extraction method and TP32 runtime mitigation notes |

# Notes

## Locked Assumptions
- Model: GPT-175B dense (strict), FP16
- Hidden size: 12288
- Num layers: 96
- Num attention heads: 96
- Sequence length: 2048
- Micro batch size: 1
- Global batch size: 256 (fixed across configs)
- World size: 1024
- PP: 8
- EXP: 1

## Execution Matrix
- TP=8, DP=16, node_size=8
- TP=16, DP=8, node_size=16
- TP=32, DP=4, node_size=32
- TP=64, DP=2, node_size=64 (blocked: heads divisibility)

## Representative Rank Set (Dense)
- Selected ranks: [0, 128, 256, 384, 512, 640, 768, 896]
- Formula: `rank = pp_stage * tp * dp`, `pp_stage in [0..7]`

## Step4 Runtime Note (TP32)
- For TP32 global placement, collective-sim produced repeated long htsim invocations.
- Mitigation used for successful completion:
  - wrapper script `tests/performance/run_simu_with_collective_cache.py`
  - in-process memoization + canonical participant mapping for topology-equivalent groups
  - no public API modification

## Metric Extraction Note
- Since simulate was executed with `--no-visualize`, direct `sum_time` lines are absent.
- Stage time was extracted as per-rank max `last_operation_time` from simulate logs for ranks `[0,128,256,384,512,640,768,896]`.
- `iteration_ms = max(stage0..stage7)`.

## Key Evidence Files
- Step0 static evidence: `logs/step0_static_evidence.md`
- Step0 dynamic probe: `logs/step0_quick_probe.md` and `logs/step0_quick_probe.json`
- TP64 divisibility evidence: `logs/tp64_blocking_evidence.md`
- Step1 run log: `logs/step1_profiling.log`
- Step2 summary: `logs/step2_database_copy_summary.json`
- Step3 summary: `logs/step3_schedule_summary.json`
- Step4 semantic checks: `logs/step4_topology_validation.md`, `logs/step4_comm_semantics.md`
- Final simulate logs: `logs/simulate_tp8_pp8_dp16.log`, `logs/simulate_tp16_pp8_dp8.log`, `logs/simulate_tp32_pp8_dp4.log`
