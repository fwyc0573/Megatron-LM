## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-06 | Added long-sequence routing-skew validation (`SEQ_LEN` 1024--4096 probe) |

# Long-Sequence Routing-Skew Validation

## Goal

Evaluate whether longer sequence lengths between `1024` and `4096` produce a cleaner reviewer-facing MoE routing-skew signal than the earlier `SEQ_LEN=256/512` smoke studies.

## Setup

- Script: `examples/pretrain_deepseek_v3_moe_aligned.sh`
- Model profile: `MODEL_PROFILE=smoke`
- Parallelism: `PP=2`, `TP=1`, `EP=4`, `DP=4`, `FAKE_WORLD_SIZE=8`
- Tracing: `TRAIN_ITERS=6`, `TRACE_START=4`, `TRACE_SUBOP_SYNC_MODE=event`
- Batch shape: `MICRO_BATCH_SIZE=1`
- Routing profiles: `balanced`, `moderate_skew`, `strong_skew`
- Metric: the same `critical-path proxy` used by `tests/performance/report_deepseek_routing_skew.py`

## Feasibility

| Seq len | Distributed | Scaling | Notes |
|---:|---|---|---|
| 1024 | PASS | PASS | Full 3-profile paired validation completed |
| 2048 | PASS | PASS | Full 3-profile paired validation completed |
| 3072 | PASS | PASS | Full 3-profile paired validation completed |
| 4096 | FAIL | Not run | Distributed run OOM in MLA SDPA (`torch.cuda.OutOfMemoryError`) |

## Long-Sequence Results

| Seq len | Skew | Hottest/median expert load | GT straggler ratio | Moye straggler ratio | Critical-path proxy error |
|---:|---|---:|---:|---:|---:|
| 1024 | Balanced | 1.00 | 1.08 | 1.09 | 13.22% |
| 1024 | Moderate | 1.48 | 1.02 | 1.02 | 0.27% |
| 1024 | Strong | 3.00 | 1.02 | 1.05 | 1.10% |
| 2048 | Balanced | 1.00 | 1.04 | 1.02 | 5.70% |
| 2048 | Moderate | 1.50 | 1.03 | 1.04 | 5.08% |
| 2048 | Strong | 3.00 | 1.02 | 1.05 | 5.76% |
| 3072 | Balanced | 1.00 | 1.01 | 1.01 | 3.84% |
| 3072 | Moderate | 1.50 | 1.00 | 1.02 | 0.51% |
| 3072 | Strong | 3.00 | 1.03 | 1.02 | 4.90% |

## Key Findings

- `SEQ_LEN=4096` is **not feasible** in the current distributed smoke configuration: the run fails with `torch.cuda.OutOfMemoryError` inside MLA scaled-dot-product attention.
- `SEQ_LEN=1024/2048/3072` all complete in both distributed and scaling modes, so these lengths are valid paired-validation points for this setup.
- However, these longer lengths do **not** provide a stronger routing-skew rebuttal signal than the earlier short-sequence study:
  - GT straggler ratios compress into the `1.00`--`1.08` range.
  - The strongest skew regime is no longer cleanly separated from balanced at `2048/3072`.
  - This suggests that, in the current smoke configuration, longer-sequence compute dominates enough of the critical path to partially mask routing-skew effects.
- On the positive side, `Moye` remains reasonably stable on these longer-sequence points:
  - all `2048/3072` cases stay within `<= 5.76%` critical-path proxy error,
  - while `1024/balanced` is the only clear outlier at `13.22%`.

## Recommendation for Rebuttal Use

- Keep the existing `SEQ_LEN=256` routing-skew mini-table as the main paper-facing evidence, because it still exposes the clearest strong-skew stress regime.
- Use the long-sequence validation in this document as **supporting evidence** that:
  1. the routing-skew path continues to run correctly at larger token lengths up to `3072`, and
  2. the absence of a stronger skew signal at long sequence length is itself informative: the smoke setup becomes more compute-dominated.
- If we want a stronger long-sequence reviewer response, the next step should likely increase both sequence length and runtime signal strength together, e.g. larger `TRAIN_ITERS`, slightly larger `MICRO_BATCH_SIZE` if memory permits, or a configuration with more exposed EP imbalance.

## Reproducibility Notes

- Probe logs: `task_memory/task_2026-03-06_isca_rebuttal_moe/logs/long_seq_probe`
- Full paired-run logs: `task_memory/task_2026-03-06_isca_rebuttal_moe/logs/long_seq_runs`
- Per-case summaries: `task_memory/task_2026-03-06_isca_rebuttal_moe/logs/seq*_routing_skew_summary.json`
- Paired artifact directories: `task_memory/task_2026-03-06_isca_rebuttal_moe/long_seq_artifacts`
