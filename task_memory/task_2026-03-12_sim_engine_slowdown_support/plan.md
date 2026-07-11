## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-15 | Reviewed overlap/slowdown follow-up gaps, fixed CLI/runtime semantics, and added example coverage |
| 2026-03-15 | Closed Phase 8c/8d workflow status and documented remaining non-blocking limitations |
| 2026-03-12 | Added auto trace-shaped schedule, self-contained NCU workflow, and hardware-reference validation phase |
| 2026-03-12 | Added GPT-6.7B pp2/tp1/dp2/sl256 lightweight E2E validation with larger DDP buckets |
| 2026-03-12 | Created execution plan for sim-engine DDP slowdown support |
| 2026-03-12 | Marked implementation/testing status after slowdown v1 integration |
| 2026-03-12 | Updated plan status after collective-sim recovery and Echo workflow validation |
| 2026-03-12 | Added E2E smoke simulate validation phase and schedule plan |
| 2026-03-12 | Completed minimal E2E slowdown simulate smoke validation |

# Task Plan: Sim-Engine DDP Slowdown Support

## Goal
Implement DDP-overlap slowdown prediction support for `megatron-sim-engine/` simulate mode with offline slowdown assets and minimal Megatron tracing changes, then validate the practical `Echo-slowdown` collection/train/predict workflow.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Megatron tracing metadata update
- [x] Phase 3: Slowdown asset generator implementation
- [x] Phase 4: Sim-engine runtime slowdown integration
- [x] Phase 5: Tests and validation for slowdown runtime
- [x] Phase 6: Restore `collective-sim` submodule/assets and rerun `megatron-sim-engine/tests/unit`
- [x] Phase 7: Validate practical `Echo-slowdown` workflow (`slowdown_collection -> merge -> train -> predict`)
- [x] Phase 8: E2E smoke simulate validation on trace-backed DDP overlap case
- [x] Phase 8b: GPT-6.7B pp2/tp1/dp2/sl256 lightweight E2E with enlarged DDP buckets
- [x] Phase 8c: Auto trace-shaped PP schedule generation from compressed traces
- [x] Phase 8d: Self-contained targeted NCU workflow for GPT-6.7B slowdown assets
- [ ] Phase 8e: Real multi-GPU reference `nsys` validation and slowdown off/on vs hardware error table
- [ ] Phase 9: Acceptance-grade accuracy study on a canonical trace/`nsys`/`ncu` trio

## Key Questions
1. How do we stably bind kernel-level baseline assets to simulator top-level ops?
2. How do we preserve existing DDP overlap replay semantics while injecting slowdown?
3. How do we make the Echo workflow reproducible on the current environment without blocking on heavy `ncu` collection?

## Decisions Made
- For GPT-6.7B `pp2,tp1,dp2,sl256` lightweight E2E, use `--ddp-bucket-size 10000000` to increase DDP overlap markers and validate a paper-like overlap regime without moving to a full heavy reprofiling flow.
- Only DDP overlap slowdown is in scope for v1.
- Use offline-generated slowdown assets; simulator never spawns Nsight tools at runtime.
- Treat `Echo-slowdown/` implementation as the ground truth API/feature set.
- Bind backward kernel blueprints to top-level ops via `cmd_uid`.
- Keep same-rank DDP comm serialization semantics; only launch timing is recomputed by the backward kernel micro-scheduler.
- Fail fast on sub-op-expanded `backward_step`; v1 only supports undecomposed top-level backward replay.
- For practical workflow validation, reuse existing `kernel_metric_output.csv` through `SKIP_KERNEL_METRIC=1` instead of insisting on a fresh heavy `ncu` run.
- Fix clear Echo workflow bugs only when needed to unblock the requested end-to-end validation.
- Megatron now auto-enables `trace_ddp_grad_overlap` whenever `overlap_grad_reduce` and tracing are both active; sim-engine now uses `--overlap-mode {auto,on,off}` and disables slowdown with a warning when overlap metadata is absent.
- Phase 8c/8d are functionally closed; remaining work is accuracy-quality follow-up rather than workflow completeness.

## Errors Encountered
- The generated full `pp2` schedule from `mg_test.py` did not align with the compressed single-batch trace shape; replay hit missing `dp_allreduce` matches or deadlock until we switched to a trace-shaped manual PP dependency schedule.
- `Echo-slowdown/slowdown_collection/run.sh` invoked `run-nsys.sh` via `sh`, which broke `set -o pipefail`.
- The installed `nsys` (`2023.1.2.43`) rejects `--python-backtrace=cuda`.
- `nsys profile` produced `.nsys-rep` successfully but returned exit code `143` on the multi-process training script.
- `Echo-slowdown/merge/merge_script.py` originally used a brittle pointer walk and produced an empty merged CSV on newly collected slowdown data.

## E2E Smoke Plan
1. Collect scaling-mode DDP-overlap traces for fake ranks `0..3` on GPU7 with `--trace-kernel-ground-truth`, `--trace-kernel-ground-truth-phase`, and `--trace-ddp-grad-overlap`.
2. Wrap the sequential fake-rank run with `nsys profile`, then export the combined report to sqlite so all ranks share one kernel-ground-truth source.
3. Generate the simulator schedule plan with `megatron-sim-engine/src/scheduler/mg_scheduling/mg_test.py` for `pp=1,tp=1,dp=4,seq=256,mbs=1,gbs=4`.
4. Build `slowdown_assets/` from the copied scaling traces, the combined `nsys` sqlite, and the trained Echo model plus kernel metrics CSV.
5. Run `megatron-sim-engine` in `simulate` mode with slowdown off/on and assert that:
   - at least one `backward_step` is processed by the slowdown path,
   - runtime DDP comm schedules are emitted,
   - `backward_step` duration does not shrink,
   - at least one DDP comm launch is delayed or the backward duration visibly increases.

## Status
**In Progress** - Core slowdown integration, unit regression recovery, practical Echo workflow validation, the minimal dense smoke, the GPT-6.7B `pp2,tp1,dp2,sl256` lightweight E2E with enlarged DDP buckets, the 2026-03-15 overlap/slowdown semantics review fixes, and the Phase 8c/8d auto-schedule + self-contained `NCU` workflow closure are complete. Acceptance-grade accuracy study remains open; fresh `NCU` recollection wall-clock cost is a known non-blocking limitation.
