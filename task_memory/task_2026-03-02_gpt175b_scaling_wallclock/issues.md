## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Initialized issues tracker |
| 2026-03-02 | Updated issue status after successful real run |

# Issues

## Open
1. `do-trace` currently remains enabled in scaling timing path by design.
2. Real run stdout is very large because each rank prints full Megatron arguments.

## Resolved
- 8192 config mismatch resolved to `8192/32/8/32`.
- Real wall-clock run on single GPU completed across all 4 target configurations.
