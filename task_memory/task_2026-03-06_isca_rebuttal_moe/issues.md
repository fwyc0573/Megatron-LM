## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-06 | Initialized issue tracker |
| 2026-03-06 | Updated issues after routing-skew validation and paper patching |

# Issues

## Open
- The current smoke-scale DeepSeek setup does not separate the moderate-skew regime cleanly from balanced; system noise still masks part of the runtime shift.
- The skew table uses a critical-path proxy rather than a full end-to-end simulator metric; this is now stated explicitly in the paper wording and test report.

## Resolved
- Found a minimal shared injection point for controlled routing profiles that covers both distributed and scaling runs via `pre_fixed_routing_results`.
- Added a deterministic routing-skew control path without modifying the simulator core logic.
- Produced paired skew summaries and integrated them into the rebuttal draft with reviewer-facing framing.
- Added explicit paper text to scope out token-dropping / capacity-aware admission control as future work rather than over-claiming support.
- Additional issue from long-sequence validation: `SEQ_LEN=4096` is not feasible in the current distributed smoke setup because MLA attention OOMs before paired validation can complete.
- Additional observation: longer sequence lengths up to `3072` make the skew signal less separable in this smoke configuration, likely because the workload becomes more compute-dominated.

