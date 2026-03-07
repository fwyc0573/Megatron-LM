## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-06 | Initialized progress tracker |
| 2026-03-06 | Implemented routing-skew support, ran paired validation, and patched rebuttal text |
| 2026-03-07 | Patched GPU memory simulation rebuttal wording in workload_const.tex |

# Progress

## 2026-03-06
- Created task directory and baseline tracking files.
- Confirmed the main target paper files: `evaluation.tex`, `comm_pred.tex`, `workload_const.tex`, and `discussion.tex`.
- Confirmed current paper already contains latest MoE model results and `all-to-all` validation, but lacks direct routing-skew evidence.
- Implemented controlled routing profiles (`balanced`, `moderate_skew`, `strong_skew`) and threaded them through `sim_routing`, CLI parsing, and the DeepSeek aligned example script.
- Added `tests/unit_tests/profiler/test_sim_routing_skew.py` to validate load-profile generation and fail-fast behavior.
- Added `tests/performance/report_deepseek_routing_skew.py` to summarize paired skew runs into Markdown/JSON artifacts.
- Completed paired distributed/scaling runs for `SEQ_LEN=256` and a larger `SEQ_LEN=512` recheck.
- Decided to use the `SEQ_LEN=256` skew table for the paper because it shows the clearest strong-skew stress signal, while still reporting the `SEQ_LEN=512` rerun as a robustness check in task memory.
- Patched the rebuttal draft to explicitly answer reviewer concerns on modern MoE generalization, MoE `all-to-all` modeling, routing-skew sensitivity, and capacity-factor scope.
- Ran final verification: unit tests passed, Python syntax checks passed, shell syntax checks passed, and the strong-skew report recheck regenerated successfully.
- Completed long-sequence routing-skew validation for `SEQ_LEN=1024/2048/3072` and documented a `SEQ_LEN=4096` distributed OOM boundary.
- Found that long-sequence runs remain feasible up to `3072`, but they do not sharpen the skew-separation signal beyond the earlier `SEQ_LEN=256` study.

## 2026-03-07
- Re-read `workload_const.tex`, `evaluation.tex`, reviewer comments, and the Megatron scaling-mode memory-tracing implementation.
- Confirmed that scaling mode launches each fake rank as a separate single-process `torchrun`, which provides process-level CUDA allocator isolation across ranks.
- Confirmed that memory tracing is runtime sampled in-process and records `allocated`, `reserved`, `peak`, and `theoretical` memory fields.
- Patched `docs/isca-rebuttal/rebuttal/workload_const.tex` to answer reviewer concerns on OOM risk, minimum memory requirement scaling, and sequential tracing isolation.
- Removed the over-strong implication that current memory validation includes faithful NCCL communication-time memory replay, keeping the method text consistent with `evaluation.tex` (`excluding communication time`).

