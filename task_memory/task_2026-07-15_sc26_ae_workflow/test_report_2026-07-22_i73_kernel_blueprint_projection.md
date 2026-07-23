# Test Report: I73 Multi-Stream Kernel Blueprint Projection

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-22 | Updated the fresh committed affected-regression evidence to the final 34-test rerun |
| 2026-07-22 | Removed Markdown-only trailing whitespace during the final report structure check |
| 2026-07-22 | Recorded raw-SQLite RCA, TDD repair, real-asset reconstruction, independent review, commits, and fresh committed regression |

**Date:** 2026-07-22
**Result:** PASS for the local I73 builder repair
**Qualification boundary:** CPU-only builder/replay validation using real prior Qwen inputs for
RCA. Producer-bound Fresh Task1, exactly-two-GPU Task2, and CPU-only Fresh Task3 remain required.

## 1. Test Script Information

### Modified implementation and tests

- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py`
- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_build_ddp_slowdown_assets.py`

### Affected regression scripts

- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_slowdown_predictor.py`
- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_simulator_config_cpu.py`
- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/unit/test_simu_engine_ddp_slowdown.py`
- `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine/tests/integration/test_simu_engine_ddp_slowdown_integration.py`

### Environment

| Item | Actual value |
|---|---|
| Working directory | `/data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine` |
| Conda environment | None on controller |
| Python | `/usr/bin/python`, Python `3.12.3` |
| pytest | `9.1.1` |
| torch | `2.5.1+cu124` |
| CUDA available / device count | `False` / `0` |
| Simulator HEAD | `0b889ed6763d040187dc3e42ff33577544e9bebf` |
| Outer producer HEAD | `dbea3e6dd3d5aaa0e4cb9f5d5ec70b33715729c5` |

### Real RCA inputs

- SQLite:
  `SC26-AE/output_gpu_20260722T1245_qwen3_i72_recap/qwen3_a30b/task1/runs/qwen3_a30b-20260722T044701Z/nsys/qwen3_a30b.sqlite`
- SQLite size: `140,226,560 bytes`
- Source blueprint `cmd_uid`: `cmd-c99b1200693b`
- Retained rebuilt assets:
  `/data/ycfeng/sc26-ae-test-tmp/i73-real-assets-20260722T150032-3660690`

## 2. Validation Criteria

1. Prove the monotonicity error comes from real cross-stream overlap, not duplicate windows,
   duplicate kernels, builder duplication, same-stream overlap, or formatting/rounding.
2. Preserve replay's existing monotonic fail-fast; do not increase epsilon or silently accept
   overlap.
3. Project source intervals deterministically onto their interval union and preserve source stream,
   source offset, source duration, and trimmed-prefix provenance.
4. Fail explicitly when a source interval is fully covered; do not silently drop a kernel.
5. Preserve total interval-union wall time and DDP launch markers.
6. Pass focused RED→GREEN coverage, affected regression, compilation, and diff hygiene.

## 3. Commands

### Focused TDD matrix

```bash
cd /data/ycfeng/sc26_ae_task3_qwen/megatron-sim-engine
export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp
export TMPDIR=/data/ycfeng/sc26-ae-test-tmp

/usr/bin/python -m pytest \
  tests/unit/test_build_ddp_slowdown_assets.py \
  -q
```

The three new cases were observed RED before production modification and GREEN afterward.

### Fresh committed affected regression

```bash
/usr/bin/python -m pytest \
  tests/unit/test_build_ddp_slowdown_assets.py \
  tests/unit/test_slowdown_predictor.py \
  tests/unit/test_simulator_config_cpu.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py \
  -q
```

### Static checks

```bash
git diff --check 0d24a5fc1f5938e9f9056688615e105989686d7d..HEAD
/usr/bin/python -m py_compile \
  src/core/simu_engine.py \
  tools/data_prep/slowdown/build_ddp_slowdown_assets.py \
  tests/unit/test_simu_engine_ddp_slowdown.py \
  tests/integration/test_simu_engine_ddp_slowdown_integration.py \
  tests/unit/test_build_ddp_slowdown_assets.py
```

## 4. RCA Evidence

| Diagnostic | Expected if duplication bug | Actual | Assessment |
|---|---:|---:|---|
| `phase=compute` windows | more than one/overlap | `1`, overlap `0` | excluded |
| Blueprint entries | may exceed source kernels | `759` | exact |
| Unique CUPTI kernel rows | less than entries | `759` | no duplication |
| Duplicate physical kernels | positive | `0` | excluded |
| Same-stream overlaps | positive | `0` | excluded |
| CUDA compute streams | one for serial source | `[7, 144, 145, 146, 147]` | multi-stream input |
| Cross-stream overlaps | positive | `2` | root-cause evidence |

The two overlaps were:

| Earlier interval | Later interval | Actual overlap |
|---|---|---:|
| stream `146`, `[10706790187, 10706794762] ns` | stream `144`, `[10706794698, 10706799082] ns` | `64 ns` |
| stream `146`, `[10710742083, 10710746627] ns` | stream `144`, `[10710746339, 10710750915] ns` | `288 ns` |

The old builder read `streamId` but discarded it from the blueprint schema. Slowdown v1 replay
then interpreted the multi-stream intervals as a single serial sequence. This producer/consumer
timeline-model mismatch is the root cause.

## 5. Test Results and Numeric Evidence

### Outcome summary

| Validation | Result | Actual evidence |
|---|---|---|
| TDD RED | PASS | `3 failed`; missing provenance, replay monotonicity failure, and missing fully-covered fail-fast |
| Focused GREEN | PASS | `3 passed in 2.56s` |
| Pre-commit affected regression | PASS | `33 passed in 6.11s` |
| Fresh committed affected regression | PASS | `34 passed in 3.11s`, exit `0` |
| Diff hygiene | PASS | no output, exit `0` |
| Python compilation | PASS | `5/5` files compiled, exit `0` |
| Independent post-fix review | PASS | `APPROVE`; Critical=`0`, Important=`0`, mandatory changes=`0` |

### Real reconstructed blueprint metrics

| Metric | Source / expected | Actual projected | Delta |
|---|---:|---:|---:|
| Kernel count | `759` | `759` | `0` |
| Launch marker count | `4` | `4` | `0` |
| Trimmed kernel count | two overlaps | `2` | `0` |
| Trim values | `64 ns`, `288 ns` | `0.000064 ms`, `0.000288 ms` | exact |
| Kernel duration sum | `4.291150 ms` | `4.290798 ms` | `-0.000352 ms` |
| Expected interval union | `4.290798 ms` | `4.290798 ms` | `0.000000 ms` |
| Monotonic violations | `0` | `0` | `0` |
| Blueprint baseline duration | trace-backed wall time | `27.080000 ms` | n/a |
| Replay residual duration | uncovered baseline remainder | `0.048475 ms` | n/a |

Manifest evidence:

```text
generator_version=v3
kernel_timeline_model=serial_interval_union_projection_v1
```

## 6. Independent Review and Scope Assessment

Pre-fix review approved builder-side deterministic serial projection and rejected epsilon
relaxation or accepting overlap. Post-fix StepCode Claude Opus 4.6 review returned `APPROVE` with
no Critical, Important, or mandatory changes.

- Design artifact SHA256:
  `1c5d41f5bc55a0b776717fe46ceabbf1c1b3bd91c77707c30464849592b005bc`
- Post-fix artifact SHA256:
  `6dad9916ef99f7f389c62ffac2d482817890eee2640adef8087f547080a50442`

The local root-cause defect is resolved in simulator commit
`0b889ed6763d040187dc3e42ff33577544e9bebf`. True multi-stream replay remains future slowdown-v2
work and is not needed for the functional AE target. All old manifests and failed runs remain RCA
evidence only; they cannot qualify the new producer.
