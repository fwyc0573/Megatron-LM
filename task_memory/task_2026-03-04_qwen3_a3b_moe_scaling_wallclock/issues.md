## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initialized issues tracker |

# Issues

## Open
1. Full real measurement includes `1024` representative ranks for `ws=8192`, so total runtime may be very long.
2. Real measurement logs are expected to be large because each rank run writes a dedicated torchrun log.

## Resolved
- All target configurations satisfy MoE constraints (`ws == pp*tp*dp` and `dp % ep == 0`).
- `PP × EP` rank skipping logic is implemented with deterministic rank mapping.
