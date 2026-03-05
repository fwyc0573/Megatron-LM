## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Initialized issues tracker |
| 2026-03-05 | Added runtime blocker discovered during reproduction |
| 2026-03-05 | Updated blocker status after stable rerun and added residual-gap issue |
| 2026-03-05 | Added remove-comm-adjacent-copy validation outcome and remaining gap tracking |

# Issues

## Open
- Residual optimizer threshold issue after selected fix (`remove comm-adjacent copy`):
  - Under preferred accounting (`distSub + scaleNoSub`), op-rank median diff:
    - `forward_step`: `2.78%` (PASS)
    - `backward_step`: `2.21%` (PASS)
    - `optimizer_step`: `5.31%` (slightly above threshold)
  - Impact:
    - Major fwd/bwd discrepancy is resolved.
    - Strict all-op `<=5%` target still narrowly missed for optimizer median.
  - Candidate follow-up:
    - Increase profiling iterations / repeated runs for optimizer noise reduction.
    - Audit optimizer-specific timing window asymmetry (without reintroducing comm-adjacent compute pollution).

## Resolved
- Previous blocker resolved: 8-GPU distributed rerun OOM due external occupancy.
  - Resolution: reran when GPUs became idle and completed distributed + scaling (baseline/patched) traces.
- Previous root-cause uncertainty largely resolved:
  - Evidence supports that scaling metadata-only comm path included comm-adjacent kernels (`contiguous/copy_`) in `all_to_all` path.
  - Removing those kernels in scaling mode substantially reduced comp mismatch on forward/backward.
