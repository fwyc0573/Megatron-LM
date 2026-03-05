## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Initial plan for DeepSeek-V3 scaling vs distributed comp latency gap analysis |
| 2026-03-05 | Updated phase status after trace comparison, root-cause analysis, and reproduction attempt |
| 2026-03-05 | Updated status after stable rerun with metadata-comm timing patch |
| 2026-03-05 | Updated status after fix-option comparison and final patch selection |

# Plan

## Goal
Produce a 16-rank comp-only comparison (fwd/bwd/optimizer) between scaling and distributed traces, identify discrepancy ranks, locate root cause in source code, and propose sharding-semantics unification recommendations.

## Scope
1. Parse distributed trace and isolate comp by subtracting communication sub-operation durations from fwd/bwd.
2. Parse scaling trace and compute same metrics.
3. Align results by rank and report discrepancy magnitude.
4. Analyze PP partitioning/MoE semantics in codepaths for both modes.
5. Assess reproducibility on 8-GPU A800 and provide runnable adaptation/debug plan.

## Acceptance Criteria
- 16-rank comparison table for `forward_step`, `backward_step`, `optimizer_step`.
- Identified discrepancy ranks with absolute and relative gaps.
- Root cause analysis tied to concrete source locations.
- Proposed fix/recommendation with explicit implementation points.

## Status
- [x] Setup task docs
- [x] Parse and compare traces
- [x] Source-code root cause analysis
- [x] Reproduce/debug evaluation on local 8-GPU machine (stable rerun completed)
- [x] Compare alternative fixes (`comm attribution` vs `remove comm-adjacent copy`) and choose final direction
- [x] Final report
