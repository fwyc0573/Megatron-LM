# Test Report: I72 Timeline-Construction Representation Conflict

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-22 | Recorded fact-corrected independent WATCH authorizing comm-only projection after precise RED→GREEN preservation, collective-matching, DDP, and fail-fast tests |
| 2026-07-22 | Added the four-rank overlap experiment: legacy and comm-only representations produced identical TP/EXP matching, zero pending collectives, and 13.0 ms makespan |
| 2026-07-22 | Added the first reopened-I72 independent review premise audit: 37/37 Scaling comm durations are zero, so the claimed residual-is-NCCL contract remains unproven |
| 2026-07-22 | Reopened I72 after real rank-224 reconstruction showed the passing fixture missed 37 TP/EXP communication children deleted by the current bypass |
| 2026-07-22 | Recorded the final requested continuation review: focused 7/7, affected 34/34, post-I72 old-conflict count 0, and distinct I73 count 1 |
| 2026-07-22 | Reconciled the qualification boundary after the nested simulator and outer producer commits completed |
| 2026-07-22 | Added fresh committed-state re-review and the seven-case/34-test verification on the I73-descendant simulator HEAD |
| 2026-07-22 | Recorded I72 comm-only projection GREEN, affected regression, static checks, and pending independent post-fix review |
| 2026-07-22 | Recorded independent I72 post-fix Claude APPROVE with artifact hash and no required remediation |
| 2026-07-22 | Recorded I72 TDD RED/GREEN, independent pre-implementation WATCH, affected regression, and construction-to-replay numeric evidence |

**Date:** 2026-07-22
**Result:** LOCAL FIX VERIFIED; independent post-fix review and nested/outer commits remain pending
**Qualification boundary:** CPU-only object-level RCA evidence only. No Fresh recapture may start until the corrected contract has independent review, RED→GREEN coverage, regression evidence, and new producer commits.

## 1. Test Script Information

### Modified implementation

- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/src/core/simu_engine.py`

### Test scripts

- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py`
- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py`
- Regression dependencies:
  - `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py`
  - `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_slowdown_predictor.py`
  - `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_simulator_config_cpu.py`

### Environment

| Item | Actual value |
|------|--------------|
| Working directory | `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine` |
| Conda environment | None on the controller |
| Python | `/usr/bin/python`, Python `3.12.3` |
| pytest | `9.1.1` |
| torch | `2.5.1+cu124` |
| CUDA available | `False` |
| Temporary root | `/data/ycfeng/sc26-ae-test-tmp` |

The controller does not contain the worker-image interpreter
`/opt/conda/envs/megatron_env/bin/python`. That fixed interpreter remains required for the later GPU
worker workflow; `/usr/bin/python` is used only for these CPU-only simulator tests.

## 2. Validation Criteria

The repair is accepted locally only if all of the following hold:

1. The pre-fix tests reproduce I72 by showing that a trace-backed slowdown `backward_step` still
   receives legacy `hidden_duration` and legacy sub-operations.
2. The repaired construction path bypasses legacy expansion only when all reviewed gate conditions
   hold: `MODE_SIMULATE`, requested slowdown, matched trace-backed backward, non-null `cmd_uid`, and
   at least one paired DDP overlay.
3. A bypassed backward keeps the matched trace duration as its temporary construction baseline,
   keeps `hidden_duration=None`, and is followed by the paired DDP overlays in source order.
4. Schedule-only backward, slowdown-disabled backward, requested slowdown without a paired overlay,
   and forward operations retain the legacy profile representation.
5. Missing matched-trace duration raises an explicit `ValueError`.
6. Construction-to-replay consumes the backward kernel blueprint, preserves the two DDP launch
   overlays, leaves no pending runtime comm schedules, and does not weaken the existing replay
   fail-fast checks.
7. All affected unit/integration tests pass, `py_compile` succeeds, and `git diff --check` reports no
   whitespace errors.

## 3. Commands

### TDD RED command, before the production edit

```bash
cd /data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine
export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp
export TMPDIR=/data/ycfeng/sc26-ae-test-tmp

/usr/bin/python -m pytest \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py \
  -q
```

Observed pre-fix result:

```text
3 failed, 18 passed in 1.30s
exit code=1
```

The three failures matched the expected missing behavior:

- construction retained actual `hidden_duration=25.63 ms` instead of `None`;
- missing trace duration did not raise;
- construction-to-replay reached the unchanged fail-fast with
  `cmd_uid=cmd-bwd-1` and `hidden_duration=8.0 ms`.

The first attempted RED command used the worker-only path
`/opt/conda/envs/megatron_env/bin/python` and exited `127` before pytest. It is not counted as RED;
the command above is the valid controller RED run.

### Focused GREEN command

```bash
/usr/bin/python -m pytest \
  tests/unit/test_simu_engine_ddp_slowdown.py::test_trace_backed_slowdown_backward_bypasses_legacy_profile_expansion \
  tests/unit/test_simu_engine_ddp_slowdown.py::test_trace_backed_slowdown_backward_requires_trace_duration \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py::test_constructed_trace_backed_backward_replays_without_legacy_conflict \
  -vv
```

Observed result:

```text
3 passed in 0.97s
exit code=0
```

### Fresh final static and affected-regression command

```bash
cd /data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine
export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp
export TMPDIR=/data/ycfeng/sc26-ae-test-tmp

git diff --check

/usr/bin/python -m py_compile \
  src/core/simu_engine.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py

/usr/bin/python -m pytest \
  tests/unit/test_build_ddp_slowdown_assets.py \
  tests/unit/test_slowdown_predictor.py \
  tests/unit/test_simulator_config_cpu.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py \
  -q
```

An initial timing wrapper referenced absent `/usr/bin/time` and exited `127` before pytest. Root
cause: the controller provides Bash's `time` keyword but not the external GNU-time binary. The
test command itself was unchanged and was rerun with Bash timing.

## 4. Test Results and Evidence

### Outcome summary

| Validation | Result | Actual evidence |
|------------|--------|-----------------|
| TDD RED | PASS | Expected reproduction: `3 failed, 18 passed`, exit `1` |
| Focused GREEN | PASS | `3 passed`, exit `0` |
| Full changed unit + integration files | PASS | `21 passed`, exit `0` |
| Affected regression | PASS | `32 passed in 3.64s`, exit `0` |
| Affected regression wall time | PASS | Bash real time `4.096s`; user `3.484s`; sys `0.163s` |
| Python compilation | PASS | `3/3` changed Python files compiled, exit `0` |
| Diff hygiene | PASS | `git diff --check` produced no output, exit `0` |
| Independent post-fix review | PASS | StepCode Claude Opus 4.6 verdict `APPROVE`; no Critical or Important findings |

### Construction-to-replay numeric evidence

The integration fixture constructs the legacy-conflict shape, invokes
`SimulatorEngine._init_3d_parallel_all_ranks()`, and then runs
`TimelinesManager._replay_profile_no_pipelining()`.

| Metric | Expected / baseline | Actual | Delta / derived value |
|--------|---------------------|--------|-----------------------|
| Constructed operation count | 3 | 3 | 0 |
| Constructed operation order | backward, comm-1, comm-2 | `cmd-bwd-1`, `comm-1`, `comm-2` | Exact match |
| Legacy `SubOperation` count | 0 | 0 | 0 |
| Constructed backward duration | trace baseline `8.0 ms` | `8.0 ms` | `0.0 ms` |
| Constructed backward hidden duration | `None` | `None` | Exact match |
| Replayed backward duration | greater than `8.0 ms` | `14.0 ms` | `+6.0 ms`, `+75.0%` |
| Replayed backward interval | starts at `0.0 ms` | `0.0 → 14.0 ms` | `14.0 ms` |
| DDP overlay count | 2 | 2 | 0 |
| `comm-1` schedule | paired to `cmd-bwd-1` | join `3.5 ms`, duration `3.0 ms`, finish `6.5 ms` | Exact expected schedule |
| `comm-2` schedule | paired to `cmd-bwd-1` | join `10.5 ms`, duration `3.0 ms`, finish `13.5 ms` | Exact expected schedule |
| Processed backward IDs | `cmd-bwd-1` | `cmd-bwd-1` | Exact match |
| Remaining runtime comm schedules | 0 | 0 | 0 |

### Independent post-fix review evidence

- Artifact:
  `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/artifacts/claude-act-as-an-independent-read-only-post-fix-code-reviewer-for-s-2026-07-22T04-39-30-862Z.md`
- Provider: StepCode Claude Opus 4.6 through `omx ask claude`
- Provider exit code: `0`
- Size: `13,366` bytes
- SHA256: `69eff9c591c30b1066797bff6eed44e4db2af8e8083fe95ecfbf8e2bf722896b`
- Verdict: `APPROVE`
- Required changes: none
- Optional findings only: inherited double `deepcopy` of overlays and inherited long call line; both
  remain outside this narrow root-cause repair.

## 5. Root-Cause Resolution Assessment

The evidence supports the reviewed attribution: I72 was caused during timeline construction, where
one trace-backed backward received both the legacy profile representation and the slowdown
kernel-blueprint identity. The repair prevents that mutually exclusive double representation at its
source. It does not clear `hidden_duration` after legacy expansion, weaken replay checks, add a
fallback, alter manifests, or change global slowdown enablement.

This report closes only the local construction-layer defect. It does **not** qualify the old failed
Fresh Task3 run. The nested simulator and outer producer commits have changed the producer identity;
new Fresh Task1 and exactly-two-GPU Task2 artifacts are therefore required before Fresh CPU-only
Task3 can provide valid report, manifest, and marker evidence.

## 6. Fresh Committed-State Re-review

I72 was rechecked on simulator HEAD
`0b889ed6763d040187dc3e42ff33577544e9bebf`, which contains I72 commit
`1a62d1ea54e9e0a1c53cf70e9841c54b5fd36094` and the later independent I73 builder repair. The
worktree was clean. Environment values were Python `3.12.3`, pytest `9.1.1`, torch
`2.5.1+cu124`, `torch.cuda.is_available() == False`, and `torch.cuda.device_count() == 0`.

Fresh command results:

| Validation | Expected | Actual | Delta |
|---|---:|---:|---:|
| I72 changed/control/error/e2e cases | `7` pass | `7 passed in 0.91s` | `0` failures |
| Combined I72/I73 affected regression | `34` pass | `34 passed in 3.11s` | `0` failures |
| Python compilation | `5` files | `5/5` compiled | `0` failures |
| Diff hygiene | no errors | `git diff --check` PASS | `0` errors |
| Post-I72 old hidden-duration conflict | `0` | `0` | `0` |
| Post-I72 distinct I73 monotonicity failure | `1` | `1` | `0` |

The original production evidence also remains internally consistent: the trace-backed backward was
`25.63 ms`, its kernel blueprint baseline was `25.92 ms`, and two DDP launch markers were paired to
the same `cmd_uid`. Legacy construction had overwritten the top level with `duration=0.01 ms` and
`hidden_duration=25.63 ms`; therefore the replay fail-fast was rejecting a real double compute
representation, not a missing asset. The subsequent full child-type audit below supersedes the
earlier conclusion that no additional I72 production change was needed.

## 7. Reopened Full-Representation Audit

The earlier fixture modeled every legacy child as compute and asserted that a repaired
trace-backed backward contained zero `SubOperation` objects. Reconstructing the actual failed
rank-224 trace with production parsing shows that this assertion deletes required communication.

### Numeric evidence

| Metric | Expected / source | Actual reconstruction | Delta / implication |
|--------|-------------------|-----------------------|---------------------|
| Parent backward duration | trace source | `25.63 ms` | baseline |
| Legacy child count | production parser output | `75` | exact object count |
| Legacy compute children | duplicate compute representation | `38` | must be removed |
| Legacy compute duration sum | parent `25.63 ms` | `25.63 ms` | `0.00 ms` delta |
| Legacy communication children | must remain represented | `37` | current bypass keeps `0`; deficit `37` |
| TP communication | source trace | `25` | current bypass deficit `25` |
| EXP communication | source trace | `12` | current bypass deficit `12` |
| all-to-all | source trace | `24` | current bypass deficit `24` |
| allgather | source trace | `6` | current bypass deficit `6` |
| reduce-scatter | source trace | `6` | current bypass deficit `6` |
| allreduce | source trace | `1` | current bypass deficit `1` |
| Blueprint compute kernels | slowdown asset | `759` | compute-only representation |
| Blueprint DDP launch markers | slowdown asset | `2` | DDP overlay representation |
| Blueprint TP/EXP collectives | must not be assumed | `0` | cannot restore dropped communication |

### Corrected assessment

The original RCA is only partially closed: the committed bypass correctly prevents duplicate
legacy compute from reaching slowdown replay, but it also silently drops 37 TP/EXP communication
children. The prior `7 passed` and `34 passed` results remain historically valid for the old test
contract, yet they cannot establish correctness of the real production representation. I72 is
therefore reopened and the report result is no longer PASS.

Required next evidence is a new independent design verdict, a RED fixture containing legacy
compute plus TP/EXP communication and two DDP overlays, a minimal root fix, focused and affected
GREEN runs, static checks, and an independent post-fix review. The replay fail-fast must remain
unchanged.

## 8. First Reopened-I72 Independent Review Premise Audit

The first new independent review completed successfully and returned `APPROVE`, but approved
retaining the current bypass. Its decisive numeric argument was:

```text
blueprint baseline                       = 25.920000 ms
sum of projected compute-kernel baseline =  4.269357 ms
residual                                 = 21.650643 ms
claimed interpretation                   = TP/EXP NCCL time + gaps
```

The subtraction is numerically correct; the interpretation is not yet supported by the producer
contract. Direct inspection provides the following counter-evidence:

| Metric / path | Expected for a residual that represents real TP/EXP network time | Actual | Delta / implication |
|---|---:|---:|---|
| Scaling collective execution | executed | early return / skipped | no real distributed NCCL latency measured |
| Target TP/EXP trace records | non-zero network durations | `0.0 ms` for `37/37` | `37` zero-duration records |
| Blueprint TP/EXP records | at least one explicit representation | `0` | no explicit network object |
| Blueprint NCCL kernel names | at least one | `0` | no NCCL kernel timing |
| Residual schema field declaring network time | present | absent | residual meaning is not network-specific |
| Task3 overlap mode | serial if comm must be added after compute | `on` | separate comp/comm predecessor lanes |

Generic communication scheduling under overlap uses
`_get_last_schedulable_comm_operation()` and ignores DDP overlay records for predecessor selection.
The removed legacy compute chunks therefore do not automatically become predecessors of the
generic TP/EXP communication. Whether the candidate projection preserves full multi-rank behavior
must still be proven by an executable collective-matching experiment; this report does not yet
claim the candidate is correct.

Independent artifact:

```text
path   = .omx/artifacts/claude-act-as-an-independent-read-only-design-and-root-cause-review-2026-07-22T08-46-21-221Z.md
bytes  = 16873
sha256 = 6cdba5d7e100442bbaae742268aeec6a5230238df81c3e054baa476b365376c9
exit   = 0
verdict= APPROVE (recommendation not adopted pending fact-corrected follow-up)
```

## 9. Four-Rank Legacy-versus-Comm-Only Semantic Experiment

### Test script information

The experiment was an inline CPU-only Python invocation in:

```text
/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine
```

It constructed `Operation` / `SubOperation` objects and executed the current production
`TimelinesManager` scheduling and collective-matching paths. It did not add a temporary script or
modify repository files. Environment: `/usr/bin/python` `3.12.3`, torch `2.5.1+cu124`, CUDA
unavailable.

### Validation criteria

1. Legacy and candidate representations have identical generic TP/EXP source order, participant
   sets, matching count, join times, finish times, and final makespan.
2. Global unmatched-collective waiting-pool size is `0` in both scenarios.
3. The candidate processes every slowdown-backed backward and consumes exactly two DDP overlays
   per rank.
4. The candidate does not serialize backward compute and generic communication into
   `8.0 + 13.0 = 21.0 ms`.

### Test results and numeric evidence

| Metric | Legacy expected / actual | Candidate actual | Delta / result |
|---|---:|---:|---:|
| Rank count | `4` | `4` | `0` |
| Backward compute baseline per rank | `8.0 ms` | `8.0 ms` | `0.0 ms` |
| TP groups | `[[0,1],[2,3]]` | same | exact |
| EXP groups | `[[0,2],[1,3]]` | same | exact |
| TP join / duration / finish | `0.0 / 6.0 / 6.0 ms` | `0.0 / 6.0 / 6.0 ms` | exact |
| EXP join / duration / finish | `6.0 / 7.0 / 13.0 ms` | `6.0 / 7.0 / 13.0 ms` | exact |
| Generic collective calls | `4` | `4` | `0` |
| Global waiting-pool size | `0` | `0` | `0` |
| Final makespan | `13.0 ms` | `13.0 ms` | `0.0 ms` |
| Incorrect serial makespan | not observed | not observed | `21.0 ms` refuted |
| Processed backward `cmd_uid` values | not applicable | `4 / 4` | complete |
| DDP overlays | not applicable | `2 / rank` | exact |
| DDP joins | not applicable | `1.0 ms`, `3.0 ms` | exact |
| DDP finishes | not applicable | `2.0 ms`, `4.0 ms` | exact |
| Remaining DDP runtime schedules | not applicable | `0` | complete |

### Interpretation boundary

This experiment proves that comm-only projection preserves the simulator's current generic
communication scheduling and matching semantics under overlap. It does not prove physical
per-collective trigger accuracy: fixture `SubOperation.start_time` metadata `9000/10000` did not
determine the actual generic joins `0.0/6.0 ms`. Changing that inherited behavior would require a
separate schema/design decision and is outside I72.

## 10. Fact-Corrected Independent Pre-Implementation Review

StepCode Claude Opus 4.6 completed with provider exit `0` and verdict
`WATCH with precise required tests`.

```text
artifact = .omx/artifacts/claude-act-as-an-independent-read-only-design-and-root-cause-review-2026-07-22T09-09-10-856Z.md
bytes    = 29996
sha256   = 8902e846c04f217c0e0d729e0fb58e5679fcde8c695b6e9dd3a9da29f6864f81
```

### Verdict findings

| Question | Earlier premise | Fact-corrected verdict |
|---|---|---|
| Residual meaning | `21.650643 ms` is TP/EXP NCCL time | No code/schema evidence; residual is opaque |
| Network double count | Restored comm is necessarily counted twice | No concrete code path exists |
| Overlap serialization | backward `8.0 ms` precedes comm `13.0 ms` | No; comm predecessor is the comm lane |
| Four-rank experiment | insufficient to refute serialization | It refutes inevitable `21.0 ms` behavior |
| Candidate semantics | requires composite schema redesign | Preserves current legacy semantics |
| Trigger offsets | must be redesigned within I72 | Inherited limitation; explicitly out of scope |

### Required RED→GREEN contract

1. Retain the top-level backward with matched trace duration and `hidden_duration=None`.
2. Remove all legacy compute children; preserve all communication children in source communication
   order.
3. Assert every preserved child has `op_kind == "comm"` and retains name, group, tensor shape and
   dtype, batch, stage, `mg_state`, and description metadata.
4. Retain the two paired DDP overlays exactly once.
5. Prove two TP groups and two EXP groups collectively match, use TP `0.0→6.0 ms`, EXP
   `6.0→13.0 ms`, and leave global waiting-pool size `0`.
6. Preserve the existing `hidden_duration` replay fail-fast without relaxation.

The verdict authorizes only this narrow source projection. Residual math, slowdown replay,
predictor, collective matching, DDP matching, manifests, producer schema, epsilon, and trigger
offset behavior remain unchanged.

## 11. Corrected TDD RED Evidence

### Environment

| Item | Actual |
|---|---|
| Python | `/usr/bin/python` `3.12.3` |
| pytest | `9.1.1` |
| Host | CPU-only controller |
| Production source | unchanged at nested HEAD `0b889ed6763d040187dc3e42ff33577544e9bebf` |

### Focused RED commands

```bash
cd /data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine
export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp
export TMPDIR=/data/ycfeng/sc26-ae-test-tmp

/usr/bin/python -m pytest \
  tests/unit/test_simu_engine_ddp_slowdown.py::test_trace_backed_slowdown_backward_preserves_only_legacy_communication \
  tests/unit/test_simu_engine_ddp_slowdown.py::test_trace_backed_slowdown_backward_rejects_unknown_legacy_child_kind \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py::test_constructed_trace_backed_backward_replays_without_legacy_conflict \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py::test_comm_only_projection_schedules_four_rank_tp_exp_collectives \
  -vv
```

### Results

| Validation | Expected pre-fix failure | Actual | Result |
|---|---:|---:|---|
| Unit projected operation count | `5` | `3` | RED |
| Preserved generic comm children | `2` | `0` | RED |
| Unknown child kind | `ValueError` | no exception | RED |
| Single-rank integration operation count | `5` | `3` | RED |
| Four-rank stages with five operations | `4 / 4` | `0 / 4` | RED |
| Combined focused run | failures required | `4 failed in 1.07s` | valid RED |

The separate six-case unit matrix produced `2 failed, 4 passed in 1.58s`: all four unchanged
legacy control branches passed. These failures demonstrate missing behavior rather than syntax,
fixture, import, or environment errors.

## 12. Post-fix GREEN and static verification

### Commands and environment

Environment remained `/usr/bin/python` Python `3.12.3`, pytest `9.1.1`, torch `2.5.1+cu124`,
CPU-only controller, with `SC26_AE_TMP_ROOT` and `TMPDIR` set to
`/data/ycfeng/sc26-ae-test-tmp`. No temporary file, log, or cache was written under `/tmp`.

```bash
cd /data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine
export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp
export TMPDIR=/data/ycfeng/sc26-ae-test-tmp

/usr/bin/python -m pytest \
  tests/unit/test_simu_engine_ddp_slowdown.py::test_trace_backed_slowdown_backward_preserves_only_legacy_communication \
  tests/unit/test_simu_engine_ddp_slowdown.py::test_trace_backed_slowdown_backward_rejects_unknown_legacy_child_kind \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py::test_constructed_trace_backed_backward_replays_without_legacy_conflict \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py::test_comm_only_projection_schedules_four_rank_tp_exp_collectives \
  -vv

/usr/bin/python -m pytest tests/unit/test_simu_engine_ddp_slowdown.py -q
/usr/bin/python -m pytest tests/integration/test_simu_engine_ddp_slowdown_integration.py -q
/usr/bin/python -m pytest \
  tests/unit/test_build_ddp_slowdown_assets.py \
  tests/unit/test_slowdown_predictor.py \
  tests/unit/test_simulator_config_cpu.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py -q

/usr/bin/python -m py_compile \
  src/core/simu_engine.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py
git diff --check
```

### Results and numeric evidence

| Validation | Expected | Actual | Result |
|---|---:|---:|---|
| Focused I72 cases | 4 pass | 4 pass in 4.70 s | PASS |
| Full unit file | 19 pass | 19 pass in 1.02 s | PASS |
| Full integration file | 4 pass | 4 pass in 0.84 s | PASS |
| Affected five-file regression | 36 pass | 36 pass in 17.24 s | PASS |
| Python compilation | exit 0 | exit 0 | PASS |
| Git diff check | exit 0 | exit 0 | PASS |

The implementation preserves the matched trace-backed parent (`duration=25.63 ms` in the unit
fixture; `8.0 ms` in the four-rank fixture), sets `hidden_duration=None`, removes all legacy
compute children, and preserves TP/EXP communication children with source order and metadata.
The four-rank scheduler evidence remains TP `0.0→6.0 ms`, EXP `6.0→13.0 ms`, four collective
requests, waiting pool `0`, and makespan `13.0 ms`; DDP overlays are consumed exactly twice per
rank. These are local CPU semantic/regression results, not Fresh GPU qualification.

One initial wrapper command exited `127` because `/usr/bin/time` is unavailable on this controller;
pytest was rerun directly and passed. No package installation or source fallback was used.

### Evidence boundary

The local fix is not yet I72 final/closed: independent post-fix review, nested/outer Lore commits,
Fresh Qwen Task1 recapture, exactly-two-GPU Task2, and CPU-only Fresh Task3 remain pending. Existing
pre-fix Fresh artifacts and manifests remain RCA-only and must not be reused.

## 13. Independent post-fix review

StepCode Claude Opus 4.6 was run through `omx ask claude` in read-only mode after all local tests
passed. Provider exit code was `0`; verdict was `APPROVE`. Artifact:

```text
.omx/artifacts/claude-perform-an-independent-read-only-post-fix-code-review-of-the-2026-07-22T11-06-26-630Z.md
bytes=9038
sha256=dafec79ca150e2d32c0a8536d4caca044ffdb520c4a3b52633933643551e4550
```

The reviewer confirmed the root cause, selective comp-only filtering, communication metadata and
source order, unknown-kind fail-fast, unchanged `hidden_duration` replay fail-fast, DDP exactly-once
behavior, four-rank fixture assumptions, and narrow scope. All observations were INFO-level and no
remediation was requested. This clears the local I72 commit gate but does not qualify Fresh GPU
artifacts or close the broader AE workflow.

## 14. Commit and post-commit evidence

The nested simulator was committed as
`2b18afc9ad3b860de2f46b9fc4b364313a21647a`; the outer producer gitlink was committed as
`df940b09c25537add927441594664c71ce01d473`. Both working trees were clean after their respective
commits. A fresh post-commit run of the affected five-file regression returned `36 passed in 8.69s`
with exit `0`.

This evidence closes the local I72 implementation/commit gate only. Fresh Qwen Task1 must be
recaptured from the new outer identity before Task2 or CPU-only Task3. The main development checkout
still contains pre-existing dirty changes and was deliberately not staged or committed.
