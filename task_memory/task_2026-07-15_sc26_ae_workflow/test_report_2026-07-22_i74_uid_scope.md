# Test Report: I74 Rank-Scoped Replay UID Repair

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-22 | Added TDD RED/GREEN evidence and affected simulator regression for the Fresh Qwen3 duplicate-UID scope repair |

**Date:** 2026-07-22
**Result:** PASS for the local simulator identity-scope repair; Fresh Task3 recapture remains pending.
**Qualification boundary:** CPU-only nested simulator tests. The prior failed Fresh Task3 run remains RCA-only and no functional/prebaked bundle may be built until a new producer identity passes Fresh Task3.

## 1. Test Script Information

### Modified implementation and tests

- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/src/core/simu_engine.py`
- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py`
- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py`

### Environment

| Item | Actual value |
|------|--------------|
| Working directory | `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine` |
| Python | `/usr/bin/python`, Python `3.12.3` |
| pytest | `9.1.1` |
| torch | `2.5.1+cu124` |
| CUDA available / device count | `False` / `0` |
| Temporary root | `/data/ycfeng/tmp/qwen3_i74_*` |

## 2. Validation Criteria

1. Two fake ranks may consume copied representative backward and communication UIDs independently.
2. Each rank must consume its own slowdown communication schedule even when both ranks share the same `comm_uid`.
3. A second use of the same `(wrank_id, cmd_uid)` must still fail fast with `Backward slowdown already processed`.
4. Rank-0 shared slowdown blueprints and trigger-presence validation must continue to use bare UIDs.
5. Existing I72/I73 construction, replay, communication matching, residual, and CPU configuration tests must remain green.
6. Python compilation and diff hygiene must pass.

## 3. Commands and TDD Evidence

All commands explicitly set `TMPDIR` below `/data/ycfeng/tmp`; no `/tmp` scratch path was used.

### RED (before production edit)

```bash
cd /data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine
export TMPDIR=/data/ycfeng/tmp/qwen3_i74_uid_red
mkdir -p "$TMPDIR"
/usr/bin/python -m pytest -q \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py \
  -k 'cross_rank_reused_uid or same_rank_reused_uid'
```

Observed result:

```text
1 failed, 1 passed, 4 deselected in 0.91s
exit code=1
```

The expected RED failure occurred at the second rank's backward operation:

```text
ValueError: Backward slowdown already processed for cmd_uid=cmd-shared
```

This proves the old global guard rejects legal cross-rank UID reuse while the same-rank duplicate control remains green.

### Focused GREEN

```bash
export TMPDIR=/data/ycfeng/tmp/qwen3_i74_uid_green
mkdir -p "$TMPDIR"
/usr/bin/python -m pytest -q \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py \
  -k 'cross_rank_reused_uid or same_rank_reused_uid'
```

Observed result:

```text
2 passed, 4 deselected in 0.91s
exit code=0
```

Numeric assertions from the fixture:

| Metric | Expected | Actual |
|--------|----------|--------|
| Shared backward UID consumed | ranks `0` and `1` | `(0, cmd-shared)` and `(1, cmd-shared)` |
| Shared communication UIDs consumed | one schedule per rank | runtime schedule map empty after `4` overlays |
| Rank-0 synthetic comm duration | `3.0 ms` for each overlay | `[3.0, 3.0] ms` |
| Rank-1 synthetic comm duration | `4.0 ms` for each overlay | `[4.0, 4.0] ms` |
| Pending wait identity | rank-local | `(0, wait-shared)` and `(1, wait-shared)` |
| Same-rank second backward | raise `ValueError` | raised exact duplicate-guard error |

### Affected regression and static checks

```bash
export TMPDIR=/data/ycfeng/tmp/qwen3_i74_affected
mkdir -p "$TMPDIR"
/usr/bin/python -m pytest -q \
  tests/unit/test_build_ddp_slowdown_assets.py \
  tests/unit/test_slowdown_predictor.py \
  tests/unit/test_simulator_config_cpu.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py
/usr/bin/python -m py_compile \
  src/core/simu_engine.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py
git diff --check
```

Observed result:

```text
38 passed in 5.95s
exit code=0
py_compile exit code=0
git diff --check exit code=0
```

## 4. Root-Cause Resolution

The Fresh failure was caused by a global-versus-rank-local identity mismatch. Representative trace
expansion intentionally copied `32` source backward UIDs across `256` fake ranks. The repair scopes
only runtime replay state as `(wrank_id, uid)`:

- `completed_cmd_operations_by_uid`
- `pending_ddp_wait_finish_times_by_uid`
- `slowdown_runtime_comm_schedules_by_uid`
- `slowdown_processed_backward_cmd_uids`

The shared rank-0 slowdown blueprint lookup and trigger-presence set remain bare UID by design. The
same-rank duplicate guard is preserved; no fallback, duplicate suppression, epsilon change, or
blueprint duplication was added.

## 5. Review and Qualification Boundary

The saved independent RCA review
`/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/.omx/artifacts/ask-claude-qwen3-uid-scope-20260722.md`
returned `APPROVE` for the rank-scoped identity design. A separate post-edit provider review was
attempted but did not return a completed artifact in the local provider session; the local gate is
therefore the focused RED/GREEN evidence plus the full `38`-test regression and static checks.

This report does not qualify Fresh Task3. A new nested/outer producer commit and a new Fresh Task3
run producing `report.json`, `report.md`, `artifact_manifest.json`, and `run_marker.json` are still
required before functional/prebaked packaging or AE-ready discussion.
