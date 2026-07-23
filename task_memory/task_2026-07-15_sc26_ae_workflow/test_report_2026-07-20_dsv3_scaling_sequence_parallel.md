# Test Report — DeepSeek-V3 Scaling Sequence-Parallel Contract

## Modification History

| Date | Summary of Changes |
|------|--------------------|
| 2026-07-20 | Added fake-TP fixed-routing cardinality tests, 8/8 topology-matched Scaling runtime metrics, trace hashes, and the explicit non-qualifying comparison FAIL |
| 2026-07-20 | Added fresh reconciliation validation, corrected audit-script failure evidence, and independent Claude `APPROVE` with future-only WATCH items |
| 2026-07-21 | Added D34 r5 fake/synthetic Task1→Task2→Task3 chain evidence, numeric metrics, worker prerequisites, and explicit non-qualification boundary |
| 2026-07-20 | Added topology-paired 8-GPU smoke trace comparison; recorded failure metrics and topology/sampling limits |
| 2026-07-20 | Added corrected `LOCAL_RANK` regression evidence (`51 passed`, 2 deselected) and documented the two unrelated legacy `LinearWithFrozenWeight` test mismatches |
| 2026-07-20 | Added 8-GPU physical mapping RED/GREEN, `_reduce`/interception repair, and final 48-test H800 regression |
| 2026-07-20 | Added final 47-test regression, mapping RED/GREEN, MoE effective-TP validation, targeted diff/compile evidence, and current-host import boundary |
| 2026-07-20 | Added H800 RED/GREEN evidence for the DSV3 `seq_aux_loss` sequence-shape failure, effective-TP argument validation, fake sequence split/replay, MLA projection, router, and six representative Scaling ranks |

## Scope and status boundary

This report covers the narrow DeepSeek-V3 full (MLA + MoE) Scaling Mode repair for the
sequence-parallel shape contract. It does **not** claim Task1 accuracy qualification,
Realistic-vs-Scaling accuracy, full 256-rank coverage, exact-two-H800 qualification, `AE-ready`, or
release readiness. D34's later fake/synthetic Task1→Task2→Task3 chain is complete only for its
explicit non-qualifying evidence classes; real Task2 and paper-level Task3 remain pending. Global
workflow status remains `INCOMPLETE`.

The original runtime RED was the router reshape failure:

```text
RuntimeError: shape '[256, 1, 32]' is invalid for input of size 1024
```

The root cause was physical TP=1 disabling `sequence_parallel` even when Scaling Mode had
`fake_tp=8`. The resulting contract was `hidden_states=(256,1,2048)`, precomputed
`indices=(32,2)`, `probs elements=8192`, and `mask elements=1024`.

## 1. Test Script Information

### Repository and H800 environment

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Image: `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`
- Python: `3.9.18` (`/opt/conda/envs/megatron_env/bin/python`)
- PyTorch: `2.1.2`
- CUDA: `12.1`
- GPU: `NVIDIA H800`
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- rlaunch resources: `--gpu=1 --cpu=8|16 --memory=32768|65536`,
  `--charged-group=codesign --private-machine=group --positive-tags=h800 --backoff-limit=1`

### Exact commands

Predict-only resource check:

```bash
rlaunch --predict-only \
  --charged-group=codesign --private-machine=group --positive-tags=h800 \
  --gpu=1 --cpu=8 --memory=32768 --backoff-limit=1 \
  --image hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae \
  --volume /data:/data \
  --workdir /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717 \
  -- bash -lc 'true'
```

Focused H800 unit suites (scripts are retained under `SC26-AE/output_gpu_20260720_dsv3_r6/`):

```bash
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_unit_tests.sh
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_unit_tests_retry.sh
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_unit_tests_mla.sh
```

The first unit wrapper intentionally exposed an environment precondition (`LOCAL_RANK` missing)
and exited non-zero during collection. The retry set `LOCAL_RANK=0`, `RANK=0`, and `WORLD_SIZE=1`
before rerunning the affected suites.

Representative DSV3 Scaling runs:

```bash
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_dsv3_scaling_rank0.sh
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_dsv3_scaling_rank1.sh
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_dsv3_scaling_rank64.sh
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_dsv3_scaling_rank65.sh
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_dsv3_scaling_rank128.sh
bash SC26-AE/output_gpu_20260720_dsv3_r6/run_dsv3_scaling_rank192.sh
```

Each run used `MODEL_PROFILE=smoke`, `SEQ_LEN=256`, `MICRO_BATCH_SIZE=1`,
`FAKE_WORLD_SIZE=256`, `FAKE_PP=4`, `FAKE_TP=8`, `FAKE_DP=8`, `FAKE_EXP=8`,
`SCALING_MIN_WARMUP_ITERS=1`, `SCALING_PROFILE_ITERS=1`, and a shared unique replay-cache tag
`dsv3-gpu-20260720T203800Z-r6`.

## 2. Validation Criteria

1. Scaling argument validation preserves `sequence_parallel=True` for physical TP=1 and
   `fake_tp=8`.
2. Scaling argument validation rejects `fake_tp<=0` explicitly and retains Realistic Mode behavior.
3. Fake sequence split/gather/reduce-scatter and replay input shape use fake TP/rank only in
   Scaling Mode; Realistic Mode remains physical-topology based.
4. MLA projection input partition and RowParallelLinear input partition match fake TP.
5. Router `seq_aux_loss` receives shape-consistent local precomputed routing results without
   changing the routing contract to full sequence or using reshape/repeat/truncate fallbacks.
6. Representative fake ranks complete warmup, forward, backward, and optimizer profile with no
   traceback or shape error and produce trace/replay artifacts.

## 3. Test Results and Evidence

### 3.1 H800 unit tests

| Suite | Result | Numeric evidence |
|------|--------|------------------|
| `tests/unit_tests/test_scaling_sequence_parallel_args.py` | PASS | `6 passed`, `1 warning`, `5.60s` |
| `test_mappings_scaling_mode.py` + `test_scaling_input_shape.py` + selected RowParallel tests | PASS | `18 passed`, `2 deselected`, `1 warning`, `6.28s` |
| Selected MoE router tests | PASS | `5 passed`, `3 deselected`, `1 warning`, `6.45s` |
| TransformerConfig Scaling tests | PASS | `3 passed`, `1 warning`, `4.57s` |
| MLA projection/forward shape tests | PASS | `3 passed`, `3 deselected`, `1 warning`, `5.90s` |

The initial combined unit wrapper (`rjob=ws-56153d316be61e0f-jlaunch-krn7j`, H800
`gpu-h800-0376`) exited `2` during collection because `tests.unit_tests.test_utilities` requires
`LOCAL_RANK`. This was an environment invocation defect, not a product failure. The corrected
wrapper (`rjob=ws-56153d316be61e0f-jlaunch-5f6nt`, same node) passed the mapping/replay/layer and
router suites. The MLA wrapper used `rjob=ws-56153d316be61e0f-jlaunch-7752j` on
`gpu-h800-0606`.

### 3.2 DSV3 Scaling rank evidence

All six representative runs exited `0` at the rlaunch outer layer and reached all three profile
markers. The first run (`rjob=...-txxz9`, `gpu-h800-0376`) also confirmed the corrected argument
state in its log:

```text
fake_tp ......................................... 8
sequence_parallel ............................... True
```

| Fake rank | PP/TP role covered | RJob | H800 node | Forward (ms) | Backward (ms) | Optimizer (ms) | Profile markers |
|----------:|--------------------|------|-----------|-------------:|--------------:|---------------:|-----------------|
| 0 | PP0 / TP0 | `...-txxz9` | `gpu-h800-0376` | 32.26 | 31.83 | 2.63 | 3/3 |
| 1 | PP0 / TP1 | `...-rgqvj` | `gpu-h800-0606` | 29.54 | 27.40 | 2.34 | 3/3 |
| 64 | PP1 / TP0 | `...-8gnrq` | `gpu-h800-0376` | 55.76 | 51.30 | 3.02 | 3/3 |
| 65 | PP1 / TP1 | `...-8mz5r` | `gpu-h800-0606` | 47.78 | 42.40 | 2.45 | 3/3 |
| 128 | PP2 / TP0 | `...-9hrjq` | `gpu-h800-0606` | 47.22 | 42.86 | 2.44 | 3/3 |
| 192 | PP3 / TP0 | `...-j5llr` | `gpu-h800-0606` | 52.39 | 43.73 | 2.55 | 3/3 |

The shared replay cache contains, among others:

```text
activation_to_rank64_iter1.pt   263483 bytes
activation_to_rank128_iter1.pt  263488 bytes
grad_to_rank0_iter1.pt           132312 bytes
```

The trace directory contains one trace per completed representative rank under:

```text
profiler_log/pp4_tp8_ep8_expn32_dp8_nl32_hs2048_sl256/
```

### 3.3 Root-cause RED and corrected GREEN

- Historical failed DSV3 Scaling runs `r4`/`r5`: exit `1`, exact reshape error above, with
  `sequence_parallel=False` despite `fake_tp=8`.
- Current argument unit test: `6/6` pass, including Scaling fake TP preservation, fake TP=1
  behavior, Realistic physical TP behavior, and `fake_tp=0/-1` fail-fast cases.
- Current representative Scaling runs: `6/6` complete all profile phases; no `RuntimeError`,
  `Traceback`, or shape mismatch marker appears in their logs.

## 4. Evidence classification and remaining gaps

This is real H800 execution evidence for a representative subset, but it is **not** full Task1
qualification. The following remain open:

- no Realistic Mode ground-truth traces were collected for this DSV3 run;
- no Scaling-vs-Realistic `comp` relative-error calculation was performed;
- fake ranks `2..63`, `66..127`, `129..191`, and `193..255` were not run;
- the acceptance target of full 8-GPU/8-fake-rank validation and the 5% accuracy threshold is not
  established by this report;
- Task2 statistical aggregation and Task3 workload/configuration recommendations have not begun;
- Gate B1, issuer/interpreter authority, external sealing, and release evidence remain unchanged.

Therefore this report records only representative H800 runtime observations; it does not create or
promote an AE evidence class, and the workflow remains `INCOMPLETE`.

## 6. Topology-paired 8-GPU smoke comparison — 2026-07-20

### Test Script Information

- Distributed trace directory: `realistic_trace/pp2_tp1_exp2_expn32_dp4_nl32_hs2048_sl256`
- Scaling trace directory: `profiler_log/pp2_tp1_ep2_expn32_dp4_nl32_hs2048_sl256`
- Comparison script: `tests/performance/compare_qwen_trace_comp.py`
- Exact command:

  ```bash
  python tests/performance/compare_qwen_trace_comp.py \
    --distributed-dir realistic_trace/pp2_tp1_exp2_expn32_dp4_nl32_hs2048_sl256 \
    --scaling-dir profiler_log/pp2_tp1_ep2_expn32_dp4_nl32_hs2048_sl256 \
    --ranks 0,1,2,3,4,5,6,7 \
    --ops forward_step,backward_step,optimizer_step \
    --threshold-pct 5 \
    --distributed-subtract-comm \
    --report-path task_memory/task_2026-07-15_sc26_ae_workflow/logs/compare_dsv3_smoke_tp1pp2.log
  ```

- Runtime evidence: H800 image `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`, Python
  3.9.18, PyTorch 2.1.2, CUDA 12.1. The trace pair uses world size 8, `TP=1`, `PP=2`,
  `EP=2`, `DP=4`, `NUM_LAYERS=32`, `HIDDEN_SIZE=2048`, and `SEQ_LEN=256` on both modes.

### Validation Criteria

- The topology and model-shape fields must match before computing relative error.
- Primary compute comparison subtracts distributed CMD communication sub-operations and uses the
  script's state-bucket mean; the auxiliary trimmed summary is non-gating.
- Every requested `(rank, op, mg_state)` row must be at or below 5% relative error for a
  qualification claim.

### Test Results and Evidence

The command found eight distributed files (`20260720131909`) and eight scaling files
(`20260720132026` through `20260720132150`) with matching topology tags. It exited `1` with
`23` primary rows above the 5% threshold. The script's operation-level rank-median auxiliary
summary was:

| Operation | Rank samples | Median relative error | P75 relative error | Status |
|---|---:|---:|---:|---|
| `forward_step` | 8 | 83.48% | 84.01% | FAIL |
| `backward_step` | 8 | 19.06% | 20.35% | FAIL |
| `optimizer_step` | 8 | 93.35% | 93.44% | FAIL |

Representative primary rows (distributed compute after communication subtraction versus scaling
duration) were:

| Rank/op/state | Distributed total (ms) | Distributed comm (ms) | Distributed comp (ms) | Scaling (ms) | Relative error |
|---|---:|---:|---:|---:|---:|
| 0 / `forward_step` / warmup | 482.5367 | 32.9567 | 449.5800 | 71.9000 | 84.01% |
| 0 / `backward_step` / cooldown | 60.5733 | 4.1567 | 56.4167 | 59.1200 | 4.79% |
| 4 / `forward_step` / steady | 117.1988 | 13.8038 | 103.3950 | 189.2300 | 83.02% |
| 7 / `forward_step` / steady | 116.8525 | 12.9583 | 103.8942 | 95.9500 | 7.65% |
| 7 / `backward_step` / steady | 91.3904 | 5.8558 | 85.5346 | 69.4900 | 18.76% |
| 7 / `optimizer_step` / finalize | 306.7533 | 0.0000 | 306.7533 | 20.7200 | 93.25% |

The Realistic files contain 3 warmup or cooldown samples and 21 steady-state samples per rank,
whereas each Scaling file contains exactly one profiled sample (`scaling_profile_iters=1`).
Realistic rank-0 `forward_step` warmup values include `1323.51 ms`, `70.27 ms`, and `53.83 ms`,
and rank-0 optimizer-finalize values include `811.52 ms`, `20.43 ms`, and `20.05 ms`; therefore
the primary mean is visibly contaminated by first-iteration initialization. This is a measurement
limitation, not evidence that the model implementation meets the 5% target. A rerun with matched
`SCALING_PROFILE_ITERS>=3` and robust state-level pairing is required before any accuracy claim.

The metric is also sensitive to communication subtraction. For rank 0, the median distributed
`forward_step` total is `70.27 ms` and the scaling duration is `71.90 ms` (a `2.32%` total-time
difference when `--no-distributed-subtract-comm` is used), but the median distributed communication
sub-operations sum to `11.16 ms`; subtracting that sum produces `59.11 ms` compute and a `21.64%`
difference. This demonstrates that CMD communication sub-operations may overlap with the enclosing
compute window, so subtraction is an analysis choice rather than a directly observed disjoint
interval. Both views must be reported and neither supports a 5% qualification claim for this run.

### Status Boundary

This is a valid topology-paired 8-GPU smoke dataset, but it is **FAIL / non-qualifying** for the
5% compute criterion. It does not validate the required 256-rank (`TP=8`, `PP=4`, `EP=8`,
`DP=8`) target topology, does not establish full fake-rank coverage, and does not promote any
Task1 evidence class. Task2 and Task3 remain pending; workflow status stays `INCOMPLETE`.

## 5. Final regression and static validation addendum — 2026-07-20

### Test Script Information

- H800 regression wrapper: `SC26-AE/output_gpu_20260720_dsv3_r6/run_full_regression.sh`
- H800 mapping RED/GREEN wrappers: `run_mapping_red.sh`, `run_mapping_green.sh`
- Static checks (current checkout): targeted `git diff --check` and
  `python -m py_compile` over all modified DSV3 production/test paths
- Current host import smoke: `PYTHONPATH=. python -c 'from megatron.core.tensor_parallel import mappings'`

### Validation Criteria

- Every selected DSV3 shape/routing/mapping/MLA/argument test passes.
- The MoE fake-TP validation rejects physical TP=1 + fake TP>1 + missing sequence parallel.
- Physical mapping helpers remain usable before Megatron global args initialization.
- Changed Python files compile and contain no new whitespace errors.
- Host-only dependency failures are reported without fallback logic.

### Test Results and Evidence

| Check | Result | Numeric evidence |
|---|---|---|
| H800 final targeted regression | PASS | `47 passed`, `5 deselected`, `2 warnings`, `6.58s`; selected pass rate `47/47 = 100%` |
| Mapping RED (pre-fix) | EXPECTED RED | `1 failed`; `AssertionError: args is not initialized.` |
| Mapping GREEN (post-fix) | PASS | `7 passed`, `1 warning`, `5.00s` |
| Targeted diff check | PASS | exit code `0` |
| Python compilation | PASS | exit code `0`; one pre-existing `SyntaxWarning` |
| Current-host import smoke | ENVIRONMENT BLOCK | `ModuleNotFoundError: transformer_engine`; no product test executed |

### Key Metrics

| Metric | Scaling evidence | Ground truth / comparison | Error |
|---|---:|---:|---:|
| Representative forward duration | `29.54–55.76 ms` across six fake ranks | Not collected | Not computable |
| Representative backward duration | `27.40–51.30 ms` across six fake ranks | Not collected | Not computable |
| Representative optimizer duration | `2.34–3.02 ms` across six fake ranks | Not collected | Not computable |
| Profile completion | `6/6` ranks with `3/3` markers | N/A | N/A |
| Full fake-rank coverage | `6/256` ranks | Required `256/256` | `250` ranks pending |

### Evidence and Status Boundary

The H800 regression and representative Scaling runs validate the local sequence-parallel/fake-TP/MoE
contract only. They do not produce `realistic_trace/`, do not establish a compute relative error,
and do not select Task2 aggregation or Task3 paper configurations. The workflow therefore remains
`INCOMPLETE`; no AE qualification or release claim is made.

## 6. Physical mapping and final regression addendum — 2026-07-20

### Test Script Information

- Physical mapping RED/GREEN: `torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/tensor_parallel/test_mappings.py -vv`
- Final focused H800 regression: the command retained in
  `SC26-AE/output_gpu_20260720_dsv3_r6/logs/full_regression_final.log`, covering all DSV3
  shape/routing/MLA/mapping/interception tests.
- H800 image: `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`; Python `3.9.18`, PyTorch
  `2.1.2`, CUDA `12.1`; `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### Validation Criteria

- Physical mapping tests run at their required world size `TP=4 × PP=2 = 8`.
- `_reduce` and interception preserve physical process-group behavior before global args init.
- Scaling interception still skips collectives when `is_scaling_mode=True`.
- Final targeted DSV3 regression passes all selected tests.

### Test Results and Evidence

| Check | Result | Numeric evidence |
|---|---|---|
| Physical mapping RED | EXPECTED RED | `2 failed, 5 passed, 5 warnings, 15.97s`; both failures were `AssertionError: args is not initialized.` in `_reduce` |
| Physical mapping GREEN | PASS | 8-GPU torchrun, `7 passed` on every rank, `5 warnings`, `15.03s`; RJob `ws-56153d316be61e0f-jlaunch-94jwp` |
| Interception focused H800 | PASS | `4 passed`, `1 warning`, `4.58s`; RJob `ws-56153d316be61e0f-jlaunch-mt9xm` |
| Final focused H800 regression | PASS | `48 passed`, `5 deselected`, `1 warning`, `7.57s`; selected pass rate `48/48 = 100%` |
| Current-host interception collection | ENVIRONMENT BLOCK | `ModuleNotFoundError: transformer_engine`; no fallback added |

### Key Metrics

| Metric | Value | Acceptance interpretation |
|---|---:|---|
| Physical mapping ranks exercised | `8/8` | Complete for the TP=4, PP=2 unit contract |
| Final selected unit pass rate | `48/48 = 100%` | Local DSV3 contract GREEN |
| Representative fake-rank coverage | `6/256` | `250` fake ranks remain pending |
| Realistic-vs-Scaling comp error | Not available | No matching `realistic_trace/` yet |

The `_reduce`/interception repair is local physical-semantics validation only. It does not provide a
Realistic trace, a compute relative-error table, full fake-rank qualification, or Task2/Task3 output.
The workflow remains `INCOMPLETE` and no AE/release claim is made.

## 7. Corrected affected-suite regression addendum — 2026-07-20

### Test Script Information

- H800 wrapper: `SC26-AE/output_gpu_20260720_dsv3_r6/run_unit_tests_retry.sh`
- Environment: image `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`, Python `3.9.18`,
  PyTorch `2.1.2`, CUDA `12.1`, with `LOCAL_RANK=0`, `RANK=0`, `WORLD_SIZE=1`, and
  `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- RJob: `ws-56153d316be61e0f-jlaunch-mbgn5`.

### Validation Criteria

- Exercise the affected argument, fake-input-shape, mapping, interception, RowParallelLinear,
  MoE-router, MLA, and TransformerConfig paths in the pinned H800 image.
- Keep unrelated legacy API/world-size failures separate from the DSV3 repair verdict; do not
  convert an unfiltered collection result into a product failure or a false all-pass claim.

### Test Results and Evidence

| Check | Result | Numeric evidence |
|---|---|---|
| Corrected affected-suite regression | PASS | `51 passed`, `2 deselected`, `1 warning`, `6.25s`; selected pass rate `51/51 = 100%` |
| Unfiltered collection context | EXPECTED SCOPE MISMATCH | `53` collected; the only two failures were legacy `test_LinearWithFrozenWeight` cases (8 positional arguments against the current 6–7 argument API, and TP=8 under `WORLD_SIZE=1`) |
| Initial wrapper environment | INVOCATION ERROR | Three collection errors because `LOCAL_RANK` was not set; rerun explicitly supplied `LOCAL_RANK=0`, `RANK=0`, and `WORLD_SIZE=1` |
| Interception focused rerun | PASS | `4 passed`, `1 warning`, `4.69s`; RJob `...-f79j6` |

The two `LinearWithFrozenWeight` failures are pre-existing test/API and world-size mismatches,
not failures in the changed Scaling Mode paths. They remain recorded as evidence and were not
masked by production fallbacks or test deletion.

### Status Boundary

This corrected regression raises confidence in the local DSV3 fake-TP/sequence-parallel contract
only. It does not create a matching `realistic_trace/`, compute relative-error table, full-rank
coverage, Task2 aggregation decision, or Task3 recommendation. Workflow status remains
`INCOMPLETE`.

## 6. Physical mapping and final regression addendum — 2026-07-20

### Test Script Information

- Physical mapping RED/GREEN: `torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/tensor_parallel/test_mappings.py -vv`
- Final focused H800 regression: the command retained in
  `SC26-AE/output_gpu_20260720_dsv3_r6/logs/full_regression_final.log`, covering all DSV3
  shape/routing/MLA/mapping/interception tests.
- H800 image: `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`; Python `3.9.18`, PyTorch
  `2.1.2`, CUDA `12.1`; `CUDA_DEVICE_MAX_CONNECTIONS=1`.

### Validation Criteria

- Physical mapping tests run at their required world size `TP=4 × PP=2 = 8`.
- `_reduce` and interception preserve physical process-group behavior before global args init.
- Scaling interception still skips collectives when `is_scaling_mode=True`.
- Final targeted DSV3 regression passes all selected tests.

### Test Results and Evidence

| Check | Result | Numeric evidence |
|---|---|---|
| Physical mapping RED | EXPECTED RED | `2 failed, 5 passed, 5 warnings, 15.97s`; both failures were `AssertionError: args is not initialized.` in `_reduce` |
| Physical mapping GREEN | PASS | 8-GPU torchrun, `7 passed` on every rank, `5 warnings`, `15.03s`; RJob `ws-56153d316be61e0f-jlaunch-94jwp` |
| Interception focused H800 | PASS | `4 passed`, `1 warning`, `4.58s`; RJob `ws-56153d316be61e0f-jlaunch-mt9xm` |
| Final focused H800 regression | PASS | `48 passed`, `5 deselected`, `1 warning`, `7.57s`; selected pass rate `48/48 = 100%` |
| Current-host interception collection | ENVIRONMENT BLOCK | `ModuleNotFoundError: transformer_engine`; no fallback added |

### Key Metrics

| Metric | Value | Acceptance interpretation |
|---|---:|---|
| Physical mapping ranks exercised | `8/8` | Complete for the TP=4, PP=2 unit contract |
| Final selected unit pass rate | `48/48 = 100%` | Local DSV3 contract GREEN |
| Representative fake-rank coverage | `6/256` | `250` fake ranks remain pending |
| Realistic-vs-Scaling comp error | Not available | No matching `realistic_trace/` yet |

The `_reduce`/interception repair is local physical-semantics validation only. It does not provide a
Realistic trace, a compute relative-error table, full fake-rank qualification, or Task2/Task3 output.
The workflow remains `INCOMPLETE` and no AE/release claim is made.
## 8. Physical TP=2 MLA preflight failure — 2026-07-20

### Test Script Information

- Script: `examples/pretrain_deepseek_v3_proxy_moe.sh` with the Realistic-mode TP2/PP2/EP2 smoke
  environment used for the retained paired attempt.
- Command shape: `MODE=distributed MODEL_PROFILE=smoke TP=2 PP=2 EP=2 DP=2 SEQ_LEN=256
  MICRO_BATCH_SIZE=1 NUM_MICROBATCH=8 GLOBAL_BATCH_SIZE=16 TRAIN_ITERS=3 TRACE_START=1
  TRANSFORMER_IMPL=local`.
- Log: `SC26-AE/output_gpu_20260720_dsv3_r6/logs/realistic_dsv3_paired_tp2_pp2_ep2.log`.
- Runtime: H800 image `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae`; RJob
  `ws-56153d316be61e0f-jlaunch-gpptc`.

### Validation Criteria

- Physical TP=2 MLA q-up output must have local width
  `num_attention_heads_per_partition * q_head_dim`.
- For this run: `64 / 2 = 32` local heads, `64 + 32 = 96` q-head width, and expected reshape
  elements `128 * 1 * 32 * 96 = 393216`.

### Test Results and Evidence

| Check | Result | Numeric evidence |
|---|---|---|
| Realistic TP2/PP2/EP2 smoke | FAIL / expected RED | Exit code `1`; all failing ranks hit `multi_latent_attention.py:412` |
| MLA expected local q-up shape | FAIL | Expected `[128, 1, 32, 96]`, `393216` elements |
| MLA observed q-up shape | FAIL | Returned `786432` elements, exactly `256 * 1 * 32 * 96`; this was the global sequence after physical all-gather, not a full-head partition |

### Root-Cause Boundary

The initial interpretation as a physical `ColumnParallelLinear`/MLA output-partition contract
mismatch was incorrect. `ColumnParallelLinear` head/output partitioning was correct. Physical
sequence-parallel all-gather expanded q-up/kv-up from local sequence `128` to global sequence `256`,
while MLA reshaped q-up with the pre-gather local sequence; `k_pe` also required an explicit
autograd-aware gather and context required a global-sequence reshape before RowParallel
reduce-scatter. No TP2 Realistic trace was produced, so no TP2 Scaling-vs-Realistic accuracy metric
may be reported. The workflow remains `INCOMPLETE`; the local unit repair is recorded in Section 9,
while the topology-matched Realistic smoke remains pending.

## 9. Physical TP=2 MLA backward coverage — 2026-07-20

### Test Script Information

- Focused two-GPU command:
  ```bash
  torchrun --nproc_per_node=2 -m pytest -q \
    tests/unit_tests/transformer/test_multi_latent_attention.py::test_multi_latent_attention_physical_tp2_projection_partition \
    -vv -s
  ```
- Full MLA file command:
  ```bash
  torchrun --nproc_per_node=2 -m pytest -q \
    tests/unit_tests/transformer/test_multi_latent_attention.py -vv
  ```
- Runtime: H800 image pinned by digest
  `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4`,
  `megatron_env`, Python `3.9.18`, PyTorch `2.1.2`, CUDA `12.1`,
  `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- RJobs: focused `sc26-ae-mla-tp2-backward-red2-07202110`; full file
  `sc26-ae-mla-tp2-file-green2-07202112`.

### Validation Criteria

- The physical TP=2 test must execute `output.sum().backward()` after the global-sequence MLA
  forward path.
- The input gradient must be present, finite, and have the same local shape `(128, 1, 2048)`.
- The physical sequence-parallel gather/reduce-scatter autograd path must complete on both ranks.
- The complete MLA unit file must remain regression-free.

### Test Results and Evidence

| Check | Result | Numeric evidence |
|---|---|---|
| Physical TP=2 forward + backward | PASS | `1 passed` on each of 2 ranks, `4 warnings`, `6.65s`; focused RJob `sc26-ae-mla-tp2-backward-red2-07202110` |
| Input gradient shape | PASS | `hidden_states.grad.shape == (128, 1, 2048)` on both ranks; `torch.isfinite(...).all()` passed |
| Full MLA file regression | PASS | `7 passed` on each of 2 ranks, `4 warnings`, `7.06s`; full-file RJob `sc26-ae-mla-tp2-file-green2-07202112` |

### Root-Cause Correction

The earlier Session 65 wording that described a full-head projection or a
`ColumnParallelLinear` partition mismatch was incorrect. `ColumnParallelLinear` head/output
partitioning was already correct. The actual failure was that physical sequence-parallel
all-gather expanded q-up/kv-up from local sequence `128` to global sequence `256`, while MLA
reshaped q-up with the pre-gather local sequence. `k_pe` also needed an autograd-aware physical
sequence gather because it bypasses the ColumnParallelLinear path. The implementation now derives
sequence and batch dimensions from the gathered projection outputs, gathers `k_pe` in physical
mode, and reshapes context with the global sequence before RowParallel reduce-scatter.

This evidence closes only the physical TP=2 MLA forward/backward unit contract. It does not
establish a topology-matched Realistic smoke, paired Scaling traces, a `<=5%` compute comparison,
full fake-rank coverage, Task2 aggregation, or Task3 paper configuration. Workflow status remains
`INCOMPLETE`.

## 10. Physical TP2/EP2 MoE fixed-routing row contract — 2026-07-20

### Test Script Information

- Script: `tests/unit_tests/transformer/moe/test_token_dispatcher.py`
- Focused fixed-routing command:
  ```bash
  torchrun --standalone --nproc_per_node=4 -m pytest -q \
    tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAlltoAllDispatcher::test_tp2_ep2_sequence_parallel_fixed_routing_forward_backward \
    -vv -s
  ```
- Focused regular-router command:
  ```bash
  torchrun --standalone --nproc_per_node=4 -m pytest -q \
    tests/unit_tests/transformer/moe/test_token_dispatcher.py::TestAlltoAllDispatcher::test_tp2_ep2_sequence_parallel_forward_backward \
    -vv -s
  ```
- Environment: H800 image
  `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4`,
  conda environment `megatron_env`, Python `3.9.18`, PyTorch `2.1.2`, CUDA `12.1`,
  `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- RJobs: fixed/local `ws-56153d316be61e0f-jlaunch-bwbrx`; regular/global
  `ws-56153d316be61e0f-jlaunch-57r9c`.

### Validation Criteria

- For local hidden shape `(4, 1, 128)` and physical `TP=2`, fixed local routing metadata `(4, 2)`
  must be gathered to `8` rows before all-to-all permutation.
- For regular router metadata already at `(8, 2)`, no second gather may occur.
- Restored output shape must equal `(4, 1, 128)` and contain `4*1*128 = 512` elements.
- `restored.sum().backward()` must produce a gradient with shape `(4, 1, 128)` and all finite values.
- No padding, truncation, repetition, zero-fill, or numeric correction is allowed on the physical path.

### Test Results and Evidence

| Test | Result | Numeric evidence |
|---|---|---|
| TP2/EP2 fixed local routing metadata | PASS | `1 passed` on each of 4 ranks, `3 warnings`, `7.70s` |
| TP2/EP2 regular global routing metadata | PASS | `1 passed` on each of 4 ranks, `3 warnings`, `7.79s` |
| Static compile | PASS | `python -m py_compile` exit `0` |
| Diff hygiene | PASS | `git diff --check` exit `0` |

### Root-Cause Evidence

The pre-fix 8-GPU Realistic smoke reported an expected output of `(128, 1, 2048)` with
`262144` elements but produced `(64, 2048)` with `131072` elements. The exact `0.5` row/element
ratio follows from local fixed metadata `(128, 2)` being used after physical `sp2hp` expanded the
hidden rows to `256`. The dispatcher now gathers local physical metadata before permutation and
checks the exact global row count, so the focused tests pass without masking the mismatch.

### Qualification Boundary

This report section validates only the physical MoE dispatcher contract. It does not qualify a new
8-GPU Realistic trace, paired Scaling trace, `forward_step`/`backward_step`/`optimizer_step` compute
relative error, full 256 fake-rank coverage, Task2 aggregation, or Task3 paper configuration. The
workflow remains `INCOMPLETE`.

## 11. D34 r5 fake/synthetic Task1→Task2→Task3 chain — 2026-07-21

### Scope and qualification boundary

This section is the current user-scoped result for fake-level Task1, Task2, and Task3 operation. It
does not run or claim Realistic multi-node/multi-GPU equivalence, Scaling-vs-Realistic `<=5%`
compute accuracy, full 256-rank qualification, exact-two-H800 Echo qualification, AE readiness, or
release readiness. The evidence classes remain `local_synthetic_not_gpu_qualification` for Task1
and Task3 and `local_synthetic_not_two_gpu_qualification` for Task2.

### Test Script Information

- Repository: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`
- Public fake/synthetic entries: `SC26-AE/task1_dsv3.sh`, `SC26-AE/task2_dsv3.sh`, and
  `SC26-AE/task3_dsv3.sh`.
- User-provided reference shell entries inspected: `examples/update_pretrain_gpt_moe-copy2.sh`,
  `examples/update_pretrain_gpt-copy.sh`, `examples/pretrain_gpt_moe-copy2.sh`,
  `examples/pretrain_qwen3_30b_a3b_moe.sh`, and `examples/pretrain_gpt.sh`.
- GPU RJob: `ws-56153d316be61e0f-jlaunch-9rgs7` (worker completed successfully; client polling was
  interrupted after submission).
- Worker log:
  `task_memory/task_2026-07-15_sc26_ae_workflow/logs/d34_task1_dsv3_fake_gpu_r5_nsys_rlaunch_20260720.log`.
- Image:
  `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4`.
- Runtime: conda environment `megatron_env`, Python `3.9.18`, PyTorch `2.1.2`, CUDA `12.1`, one
  visible NVIDIA H800 GPU. The worker installed `nvidia-ml-py==12.535.133` and used the fixed
  interpreter `/opt/conda/envs/megatron_env/bin/python` and `/opt/conda/envs/megatron_env/bin/torchrun`.
- Task1 configuration: `MODE=scaling`, `CUDA_VISIBLE_DEVICES=0`, `FAKE_WORLD_SIZE=256`,
  `FAKE_PP=4`, `FAKE_TP=8`, `FAKE_DP=8`, `FAKE_EXP=8`, `QUICK=1`, selected ranks `0,64,128,192`,
  `SCALING_MIN_WARMUP_ITERS=3`, `SCALING_PROFILE_ITERS=1`, `TRACE_MEMORY=1`, and `CAPTURE_NSYS=1`.
- Task1 output root:
  `SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/dsv3/task1/runs/dsv3-d34-fakegpu-r5-20260720T000000Z/`.
- Task2 output root:
  `SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/_shared/task2/runs/d34-dsv3-synthetic-predictor-r5-20260721T000000Z/`.
- Task3 output root:
  `SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/dsv3/task3/runs/d34-dsv3-synthetic-fresh-r5-20260721T000000Z/`.

### Validation Criteria

1. Task1 must produce four verified selected-rank trace files, four memory JSON files with non-empty
   samples and positive memory metrics, a verified capture marker, a verified artifact manifest,
   and Nsight `.nsys-rep` plus `.sqlite` artifacts.
2. Task2 must produce a verified predictor marker/manifest, finite metrics, a deterministic reload
   delta, and a synthetic evidence class.
3. Task3 must consume the r5 Task1/Task2 identities with `artifact_source=fresh`, produce a verified
   marker/manifest, report rank0 step components and simulator timings, and retain a synthetic
   evidence class.
4. Final local checks must pass: all requested shell files parse, the existing Task1 contract and
   fresh-chain e2e tests pass, all three canonical manifests verify, and `git diff --check` is clean.

### Test Results and Evidence

| Check | Result | Evidence |
|---|---|---|
| Task1 fake capture | PASS | `TASK1_STATUS=verified`; selected ranks `0,64,128,192`; trace files `4`; memory JSON `4` |
| Task1 artifact manifest | PASS | `MANIFEST_STATUS=verified`; manifest file count `33` |
| Task1 Nsight artifacts | PASS | `dsv3.nsys-rep` `7,038,752` bytes; `dsv3.sqlite` `23,699,456` bytes |
| Task1 fresh-capture timing gate | PASS | rank0 `16.550779585 s`; 256-rank estimate `4236.999573760 s`; threshold `7200 s` |
| Task2 synthetic predictor | PASS | manifest file count `13`; dataset rows `2`; average validation MSE `3.0`; test MSE `0.5` |
| Task2 reload stability | PASS | maximum absolute prediction delta `0.0` |
| Task3 fresh synthetic simulation | PASS | manifest file count `17`; artifact source `fresh`; marker `verified=true` |

### Task1 numeric metrics

| Fake rank | Forward (ms) | Backward (ms) | Optimizer (ms) | Peak allocated (MB) |
|---:|---:|---:|---:|---:|
| 0 | 33.94 | 38.46 | 2.82 | 917.52 |
| 64 | 55.29 | 56.72 | 3.43 | 679.86 |
| 128 | 54.58 | 55.29 | 3.61 | 679.86 |
| 192 | 55.37 | 58.30 | 3.08 | 836.11 |

Each memory JSON contains a non-empty `samples` array, positive `peak_allocated_MB`, and positive
`theoretical_memory_MB`. The maximum observed peak allocation is `917.52 MB`.

### Task2 numeric metrics

| Metric | Actual |
|---|---:|
| Dataset row count | 2 |
| Validation MSE by fold | `[1.0, 2.0, 3.0, 4.0, 5.0]` |
| Average validation MSE | 3.0 |
| Test MSE | 0.5 |
| Model reload max absolute prediction delta | 0.0 |
| Scaler feature/mean/scale/nonzero counts | `2/2/2/2` |
| Original execution time | 1.0 |
| Predicted execution time | 6.2 |
| Predicted slowdown factor | 10.4 |
| Task2 run elapsed | 0.039398512 s |

### Task3 numeric metrics

| Metric | Actual |
|---|---:|
| Rank0 step time | 24.5 ms |
| Forward duration sum | 6.5 ms |
| Backward duration sum | 12.0 ms |
| Optimizer duration sum | 3.0 ms |
| Comp-plus-comm diagnostic | 21.75 ms |
| Simulator load time | 0.125 s |
| Simulator execution time | 0.375 s |
| Simulator wall clock | 0.5 s |

Task3 also generated four pipeline-stage schedule files, slowdown assets, input and resolved
provenance manifests, and the verified run marker.

### Historical failure root causes and resolution

| Attempt | Failure | Root cause | Resolution |
|---|---|---|---|
| r1 | Memory inventory mismatch | Pinned image lacked `pynvml`/NVML provider | Install `nvidia-ml-py==12.535.133` in `megatron_env` |
| r2 | Provenance failed with `detected dubious ownership` | Worker UID differed from mounted repository owner | Add the exact Git `safe.directory` entry |
| r3 | `ModuleNotFoundError: torch` before Task1 | Worker default `python` was not `megatron_env` | Bind Python and torchrun absolute paths |
| r4 | Task1 had no SQLite | `CAPTURE_NSYS=0` | Rerun r5 with `CAPTURE_NSYS=1` and use the same-source fresh chain |

### Commands for final local verification

The following commands are reproducible from the repository root and are recorded with their
observed results in the continuation log:

```bash
bash -n \
  SC26-AE/task1_dsv3.sh SC26-AE/task2_dsv3.sh SC26-AE/task3_dsv3.sh \
  SC26-AE/lib/common.sh SC26-AE/lib/task1_trace.sh SC26-AE/lib/task2_echo.sh \
  SC26-AE/lib/task3_simulation.sh \
  examples/update_pretrain_gpt_moe-copy2.sh examples/update_pretrain_gpt-copy.sh \
  examples/pretrain_gpt_moe-copy2.sh examples/pretrain_qwen3_30b_a3b_moe.sh \
  examples/pretrain_gpt.sh examples/pretrain_deepseek_v3_moe.sh

bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh

python3 SC26-AE/tools/artifact_manifest.py verify --root \
  SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/dsv3/task1/runs/dsv3-d34-fakegpu-r5-20260720T000000Z \
  --manifest \
  SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/dsv3/task1/runs/dsv3-d34-fakegpu-r5-20260720T000000Z/artifact_manifest.json

python3 SC26-AE/tools/artifact_manifest.py verify --root \
  SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/_shared/task2/runs/d34-dsv3-synthetic-predictor-r5-20260721T000000Z \
  --manifest \
  SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/_shared/task2/runs/d34-dsv3-synthetic-predictor-r5-20260721T000000Z/artifact_manifest.json

python3 SC26-AE/tools/artifact_manifest.py verify --root \
  SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/dsv3/task3/runs/d34-dsv3-synthetic-fresh-r5-20260721T000000Z \
  --manifest \
  SC26-AE/output_gpu_20260720_dsv3_fake_e2e_r5/dsv3/task3/runs/d34-dsv3-synthetic-fresh-r5-20260721T000000Z/artifact_manifest.json

git diff --check -- SC26-AE task_memory/env_handbook.md \
  task_memory/task_2026-07-15_sc26_ae_workflow tests
```

### Final interpretation

`D34 fake-only Task1 → Task2 → Task3 chain: PASS.` This is fake/synthetic contract evidence only.
It is **not** Realistic distributed qualification, **not** `<=5%` accuracy evidence, **not**
exact-two-H800 Echo qualification, and **not** AE-ready/release qualification. Global workflow
status remains `INCOMPLETE`.

### Observed local verification results

| Command/check | Result | Numeric evidence / exit code |
|---|---|---|
| `bash -n` over requested entries | PASS | `13/13` scripts; exit `0` |
| `bash tests/integration/test_sc26_ae_task1_contracts.sh` | PASS | `PASS_COUNT=38`; exit `0` |
| `bash tests/e2e/test_sc26_ae_fresh_chain.sh` | PASS | `CHAIN_PASS_COUNT=1`; trace `4`; memory `4`; rows `2`; exit `0` |
| r5 Task1 `artifact_manifest.py verify` | PASS | `MANIFEST_FILE_COUNT=33`; exit `0` |
| r5 Task2 `artifact_manifest.py verify` | PASS | `MANIFEST_FILE_COUNT=13`; exit `0` |
| r5 Task3 `artifact_manifest.py verify` | PASS | `MANIFEST_FILE_COUNT=17`; exit `0` |
| Scoped `git diff --check` | PASS | no diagnostics; exit `0` |

The e2e command above is a local synthetic regression fixture and reports its own rank0 values
(`22.5 ms` step, `6.0/11.0/2.5 ms` forward/backward/optimizer). The r5 live fake evidence in the
tables above remains the authoritative D34 chain result (`24.5 ms` synthetic Task3 fresh report),
and the two numeric sets must not be conflated.

## 12. Fake-TP fixed-routing cardinality and 8-rank Scaling evidence — 2026-07-20

### Test Script Information

- Production paths:
  - `megatron/profiler/moe/sim_routing.py`
  - `megatron/profiler/moe/sim_dispatching.py`
  - `megatron/core/transformer/moe/moe_layer.py`
  - `megatron/core/transformer/moe/token_dispatcher.py`
- Test paths:
  - `tests/unit_tests/transformer/moe/test_routers.py`
  - `tests/unit_tests/transformer/moe/test_token_dispatcher.py`
  - `tests/unit_tests/transformer/moe/test_token_dispatcher_shape_restore.py`
- Environment: image
  `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4`,
  conda environment `megatron_env`, Python `3.9.18`, PyTorch `2.1.2`, CUDA `12.1`, NVIDIA H800.
- Focused routing/dispatch command:

  ```bash
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
  pytest -q tests/unit_tests/transformer/moe/test_routers.py \
    -k "scaling_sim_routing_expands_fake_tp_sequence_rows or \
        expand_scaling_fixed_routing_rows_validates_cardinality or \
        scaling_sim_dispatching_counts_expanded_fake_tp_assignments"
  ```

- Scaling dispatcher integration command:

  ```bash
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
  pytest -q tests/unit_tests/transformer/moe/test_token_dispatcher.py \
    -k scaling_fake_tp2_fixed_routing_preserves_rows_without_restore -vv
  ```

- Strict restore/router regression command:

  ```bash
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
  pytest -q \
    tests/unit_tests/transformer/moe/test_token_dispatcher_shape_restore.py \
    tests/unit_tests/transformer/moe/test_routers.py \
    -k "not test_tp2_ep2_sequence_parallel_forward_backward and \
        not test_tp2_ep2_sequence_parallel_fixed_routing_forward_backward"
  ```

- Runtime configuration: `MODE=scaling`, `MODEL_PROFILE=smoke`, `TRANSFORMER_IMPL=local`,
  `TP=2`, `PP=2`, `EP=2`, `FAKE_WORLD_SIZE=8`, `FAKE_TP=2`, `FAKE_PP=2`, `FAKE_EXP=2`,
  `FAKE_DP=2`, `SEQ_LEN=256`, microbatch `1`, global batch `16`, warmup `3`, profile `1`.
- Runtime log:
  `task_memory/task_2026-07-15_sc26_ae_workflow/logs/scaling_dsv3_tp2_pp2_ep2_cardinality_20260720.log`.
- Comparison report:
  `task_memory/task_2026-07-15_sc26_ae_workflow/logs/compare_dsv3_tp2_pp2_ep2_cardinality_20260720.log`.

### Validation Criteria

1. For local routing rows `L`, fake TP `T`, and router top-k `K`, fixed routing must contain exactly
   `L*T` rows and `L*T*K` assignments before token dispatch.
2. Scores and indices must both be two-dimensional, have identical shapes, use the configured top-k
   width, and contain only valid expert IDs.
3. Missing or extra output rows must raise `ValueError`; padding, truncation, repetition, zero-fill,
   fallback, and numeric correction are forbidden.
4. The fake-TP dispatcher must restore the original local hidden shape through its real compute path
   and produce a finite backward gradient of the same shape.
5. Fake ranks `0..7` must each complete warmup and exactly one forward/backward/optimizer profile,
   write one trace, and contain no traceback or row-contract error.
6. The historical comparison must remain `FAIL` unless every raw primary check is within `5%`; an
   auxiliary median/trimmed statistic may not promote it.

### Test Results and Evidence

| Suite/check | Result | Numeric evidence |
|---|---|---|
| Routing expansion and dispatch counts | PASS | `3 passed`, `7 deselected`, `1 warning`, `5.77s`; RJob `ws-56153d316be61e0f-jlaunch-snzd6` |
| Scaling dispatcher forward/backward | PASS | `1 passed`, `7 deselected`, `1 warning`, `5.72s`; RJob `ws-56153d316be61e0f-jlaunch-blqrr` |
| Strict restore/router regression | PASS | `15 passed`, `1 warning`, `5.99s`; RJob `ws-56153d316be61e0f-jlaunch-xs9pf` |
| Topology-matched Scaling runtime | PASS | `8/8` fake ranks; each trace has forward/backward/optimizer counts `1/1/1`; RJob `ws-56153d316be61e0f-jlaunch-pjndk` |
| Runtime error scan | PASS | `5,697` log lines; `0` `Traceback`/`RuntimeError`/`ValueError`/row-count failures |
| Historical primary comparison | FAIL | `24/24` raw checks above `5%` or invalid for qualification |

The fake-TP expansion unit case uses `L=32`, `T=8`, and `K=2`, so expected routing rows are
`32*8=256` and expected assignments are `32*8*2=512`. The integration case uses hidden rows `2`,
`fake_tp=2`, routing rows `4`, and `topk=2`; its restored output and gradient both retain the
original hidden shape and all gradient values are finite.

### Runtime Metrics

| Fake rank | Forward (ms) | Backward (ms) | Optimizer (ms) | Trace timestamp |
|---:|---:|---:|---:|---|
| 0 | 58.43 | 60.10 | 10.71 | `20260720152210` |
| 1 | 57.03 | 57.99 | 10.44 | `20260720152224` |
| 2 | 58.49 | 57.96 | 10.65 | `20260720152236` |
| 3 | 57.91 | 59.44 | 10.55 | `20260720152252` |
| 4 | 71.38 | 70.03 | 12.21 | `20260720152304` |
| 5 | 72.43 | 71.24 | 11.95 | `20260720152315` |
| 6 | 72.72 | 71.14 | 12.22 | `20260720152332` |
| 7 | 71.56 | 69.50 | 11.84 | `20260720152343` |

| Operation | Scaling minimum (ms) | Scaling maximum (ms) | Scaling mean (ms) | Realistic comparison rank-median error |
|---|---:|---:|---:|---:|
| `forward_step` | 57.03 | 72.72 | 64.9938 | 63.38% |
| `backward_step` | 57.96 | 71.24 | 64.6750 | 23.15% |
| `optimizer_step` | 10.44 | 12.22 | 11.3213 | 95.49% |

Runtime log size is `5,697` lines with SHA256
`ed4aed6a5a4bba12020f96be887e3fbd7bc8c429ce0d6b2571c4c70f83c17938`. Comparison report SHA256
is `c58abc3661f9e9bef7337276f5f3ae1fbe5457bf4a410494762c13fadac381dc`.

### Failure Interpretation and Status Boundary

The comparison is not a valid one-iteration pairing: Realistic buckets contain either `3`
optimizer/warmup/cooldown samples or `24` steady microbatch samples, while each Scaling bucket has
one sample. The first Realistic optimizer sample is approximately `696–766 ms`, whereas later
samples are approximately `10–13 ms` and Scaling is `10–12 ms`; averaging these produces the
reported `~95%` optimizer error. The trace text does not serialize an iteration identity that can
justify silently selecting only the last sample. Communication sub-operation durations also cannot
be assumed disjoint from their enclosing CMD window without interval evidence.

Therefore the comparison verdict remains `FAIL`; trimmed mean, rank median, p75, or p90 are
diagnostics only. No startup sample was silently removed, no threshold was changed, and no
calibration factor or fallback was added. A startup-clean controller submission attempt failed
before RJob creation with Go `newosproc`, `errno=11`, which is an environment resource failure, not
product evidence. D34 later made Realistic execution and paired accuracy out of scope for the
current fake-only gate, so no new Realistic job is part of this report. Global Task1 accuracy,
Qwen3.5-MoE, full 256-rank coverage, real Task2 aggregation, paper Task3 configuration, and AE/
release readiness remain open. Global workflow status is `INCOMPLETE`.

### Fresh Local Reconciliation Verification

The documentation and retained-evidence audit was rerun locally with `/usr/bin/python3` `3.12.3`,
pytest `9.1.1`, no active conda environment, and an empty `PYTHONPATH`. This local environment was
used only for shell, manifest, comparator-unit, trace-text, document, and diff checks; it does not
replace the pinned H800 runtime evidence above.

| Fresh check | Result | Actual numeric output |
|---|---|---|
| Shell syntax | PASS | `13/13` scripts; exit `0` |
| Task1 integration | PASS | `PASS_COUNT=38`; exit `0` |
| Fresh synthetic e2e | PASS | `CHAIN_PASS_COUNT=1`; traces `4`; memory JSON `4`; Task2 rows `2`; exit `0` |
| Synthetic numeric metrics | PASS | validation MSE `3.0`; test MSE `0.5`; reload delta `0.0`; Task3 step `22.5 ms`; simulator wall `0.5 s` |
| Retained Task1/Task2/Task3 manifests | PASS | file counts `33/13/17`; all exit `0` |
| Comparator unit regression | PASS | `39 passed in 0.93s`; exit `0` |
| Comparator compile | PASS | `python3 -m py_compile`; exit `0` |
| Retained trace audit | PASS | ranks `8/8`; operation samples `24/24`; both SHA256 values matched |
| Document contract audit | PASS | four files, three focused assertions per file |
| Scoped diff hygiene | PASS | no diagnostics; exit `0` |

The first retained-evidence audit attempt failed inside its Python snippet with
`re.error: missing ), unterminated subpattern`: the temporary audit regex over-escaped `\\(`. This
was a test-script defect, not a trace/product failure. The surrounding first wrapper lacked
`set -e`, so its later exit `0` was explicitly rejected as completion evidence. The corrected retry
used direct string counting under `set -e -o pipefail`, reran the failed audit plus all downstream
document/diff gates, and exited `0`. Evidence log:
`task_memory/task_2026-07-15_sc26_ae_workflow/logs/session72_fake_cardinality_reconciliation_20260720.log`,
SHA256 `21cf3f00ffe15b1b45d7fc65ebb48768a4d03d14941b02cdfad93d067b99fb10`.

### Independent Review Evidence

StepCode Claude independently reviewed the production diff, focused tests, all four Session 72
documents, and all three retained logs. Provider exit was `0`; verdict was `APPROVE`; blockers,
numeric discrepancies, and CRITICAL/HIGH findings were all `0`. It independently confirmed all
eight timestamps and file sizes, the recorded means/ranges and SHA256 values, the `24` primary
comparison failures, the D34 fake-only boundary, the initial regex-audit failure and corrected
`set -e` retry, and the absence of false qualification claims.

The artifact is
`.omx/artifacts/claude-act-as-an-independent-read-only-reviewer-for-the-current-ses-2026-07-20T18-39-09-554Z.md`
(`5,540` bytes; SHA256
`f532ffdd85b4c20d983ce372748e7c174422426186e5b34131d64192614708ac`). Its non-blocking WATCH items
cover report numbering, additional direct branch/unit coverage, an explicit repeated-score
assertion, a pre-existing live-dispatcher histogram, and later physical TP runtime confirmation.
They are recorded in `future.md` and do not promote any qualification state.

The first post-review document audit used a line-contiguous assertion for text that wraps normally
in Markdown and failed on `issues.md`. The corrected verifier used a stable semantic fragment,
passed all five document groups with `25` focused assertions, rechecked all four hashes, and passed
scoped `git diff --check` under `set -e -o pipefail`. This second verifier defect did not change any
source, artifact, metric, evidence class, or qualification result.
