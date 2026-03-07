## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initialized issue tracker and recorded known blocker handling |

# Issues

## Resolved
1. `mg_test.py` cannot generate schedule when `PP=1`.
   - Root cause: scheduler returns `None` for no-pipeline path and still invokes callable.
   - Resolution: runner generates `PP=1` stage schedule manually with required ops.
   - Impact: no source patch required in engine; execution remains unblocked.
2. Simulator default strategy fails for `PP=1`.
   - Root cause: default `1F1B-none_interleaved` dependency mapping requires `pp_size > 1`.
   - Resolution: runner sets strategy to `no-pipelining` when `PP=1`, and keeps `1F1B-none_interleaved` for `PP>1`.
   - Impact: scale-8 execution succeeds while preserving original strategy for multi-stage runs.

## Open
1. None at current stage.
