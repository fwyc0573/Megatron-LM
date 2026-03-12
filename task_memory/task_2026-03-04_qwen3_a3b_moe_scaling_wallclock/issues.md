## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-08 | Recorded transient Config3 rank3392 SIGSEGV evidence and remainder-resume mitigation |
| 2026-03-07 | Recorded nohup detachment failure symptom and setsid-based mitigation |
| 2026-03-04 | Initialized issues tracker |
| 2026-03-04 | Added real-run blocker details and query-group fix status |
| 2026-03-04 | Updated blocker status after runtime fixes and rerun |
| 2026-03-04 | Updated runtime status after Config2 completion in resume run |
| 2026-03-04 | Added new blocker: Config3 failed at rank 1400 with TE RMSNorm view error |
| 2026-03-06 | Resolved rank1400 blocker and updated remaining long-run issue status |
| 2026-03-06 | Recorded dedicated Config3/Config4 background sweep status and duplicate-run cancellation |

# Issues

## Open
1. Full real measurement includes `1024` representative ranks for `ws=8192`, so total runtime may be very long.
2. Real measurement logs are expected to be large because each rank run writes a dedicated torchrun log.
3. Config3/Config4 full sequential wall-clock sweep remains time-consuming on one physical GPU.
4. Final CSV validation is still blocked on completing the remaining long-run configurations (`ws4096`, `ws8192`).
5. Dedicated background sweeps are running for `Config3@GPU0` and `Config4@GPU1`; until they finish, task completion remains pending on final CSV merge/verification.
6. `nohup`-style detached runs did not persist reliably in the current execution harness; detached shell wrappers disappeared without explicit runtime errors in the outer task log.
7. `Config3` encountered a transient `SIGSEGV` after rank `3392` completed profiling; because the crash happened post-profile and direct repro of rank `3392` succeeded, the remaining risk is resumability/accounting rather than deterministic correctness of rank `3392` itself.

## Resolved
- All target configurations satisfy MoE constraints (`ws == pp*tp*dp` and `dp % ep == 0`).
- `PP × EP` rank skipping logic is implemented with deterministic rank mapping.
- Initial TP compatibility issue fixed: `NUM_QUERY_GROUPS` changed from `4` to `8` for `fake_tp=8`.
- TE RMSNorm non-contiguous `view` runtime error fixed via TENorm contiguous cast.
- MoE token dispatcher scaling reshape mismatch fixed via deterministic row restore before `view`.
- Resume-run functional blocker cleared: Config2 finished successfully and CSV row was written.
- Config3 rank `1400` TE RMSNorm `view` blocker resolved in current workspace; repro passed with `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Duplicate local `Config3` retry was cancelled after confirming a separate official `GPU0` sweep was already running; this avoids contaminating single-GPU wall-clock measurements.
- Previous `nohup`-style detached full sweeps were replaced by `setsid`-managed sessions on 2026-03-07 after confirming `setsid` re-parented the wrapper shells to PID 1 and kept `torchrun` + `tee` children alive.
- Rank `3392` for `Config3` was re-run successfully on 2026-03-08; the previous `SIGSEGV` is currently treated as transient/post-profile-exit behavior rather than a stable model-path crash.
