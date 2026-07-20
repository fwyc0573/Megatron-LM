# Test Report: Task1 D16 Independent Rank-0 Preflight Contracts

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Added the fresh GREEN report for the strict D16 preflight behavior matrix, stale-count remediation, and synthetic/controller evidence boundary |

## 1. Test Script Information

**Date:** 2026-07-20  
**Repository:** `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`  
**Environment:** controller-only shell; GNU bash 5.2.21; Python 3.12.3; no conda, GPU, RJob, or Docker workload was used.

### Scripts and exact commands

- `SC26-AE/lib/task1_trace.sh`
- `tests/unit/test_sc26_ae_task1_d16_timing.sh`
- `tests/integration/test_sc26_ae_task1_contracts.sh`
- `tests/e2e/test_sc26_ae_fresh_chain.sh`
- `tests/e2e/test_sc26_ae_task1_smoke.sh`
- `tests/e2e/test_sc26_ae_clean_clone_replay.sh` (current count consumer; exercised by the clean-clone regression in the repository's follow-up suite)
- `tests/integration/sc26_ae_v21_verifier.py` and `tests/integration/test_sc26_ae_v21_verifier.sh` (historical/current evidence split)

```bash
cd /data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717
bash -n SC26-AE/lib/task1_trace.sh
bash -n tests/unit/test_sc26_ae_task1_d16_timing.sh
bash -n tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/unit/test_sc26_ae_task1_d16_timing.sh
bash tests/integration/test_sc26_ae_task1_contracts.sh
bash tests/e2e/test_sc26_ae_fresh_chain.sh
bash tests/e2e/test_sc26_ae_task1_smoke.sh
git diff --check
```

Fresh command logs:

| Log | Bytes | SHA256 |
|-----|------:|--------|
| `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-d16-unit-green-20260720.log` | 2,504 | `66cc74096a6e2c38f9487f9f5b4faee7281fd66dc66fd81c7f0b288ba09b9a5b` |
| `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-d16-integration-green-20260720.log` | 2,595 | `d1d2e06b9afe66e9b96c66b819f363bbfad284dd0a844ee1bc7bdf84f7e4ced6` |
| `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-d16-fresh-chain-current-20260720.log` | 702 | `ae5402501dcacd3e495d760b5a0959c89a6d4217cdf23968dcc75566733cbc8f` |
| `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-d16-task1-smoke-current-20260720.log` | 2,685 | `a735d5c2ef88854ff0027651f36809f0f653e2db61ceb627c1600508c6e54b80` |
| `task_memory/task_2026-07-15_sc26_ae_workflow/logs/i53-d16-diff-current-20260720.log` | 20 | `a1758210a4f81cd6bb68e4406fbfe4970b63541fd7067d52d304b0a63cab3cb4` |

## 2. Validation Criteria

1. The D16 unit suite must pass all strict schema, type, interval, provenance-copy, manifest-isolation, and gate-boundary cases: `49/49`.
2. The Task1 integration suite must pass all behavior and negative cases: `38/38`.
3. The frozen D16 constants must remain unchanged:
   - MoE estimate rank count: `256`;
   - threshold: `7,200` seconds;
   - exact-threshold result: `pass`;
   - above-threshold result: `prebaked_required`.
4. QUICK behavior must be observation-only (`d16_gate_enforced=false`, `gate_decision_applied=false`) and continue the four-rank smoke path even when the estimate is above threshold.
5. Full behavior must enforce the gate before selected capture:
   - pass: one preflight call plus `256` selected-rank calls;
   - above threshold: exit `2`, zero selected-rank calls, no full root, no marker, and no source switch.
6. Current replay and V21 consumers must distinguish current counts from immutable historical transcripts.
7. No synthetic/controller result may be labeled H800 qualification or used to close I53/Gate B1.

## 3. Test Results and Evidence

| Test suite / check | Result | Observed value |
|--------------------|--------|----------------|
| D16 unit | **PASS** | `PASS_COUNT=49`, exit `0` |
| Task1 integration | **PASS** | `PASS_COUNT=38`, exit `0` |
| Fake torchrun matrix | **PASS** | `281` total invocations |
| Full-pass branch | **PASS** | `1` preflight + `256` selected = `257` calls |
| Full-above-threshold branch | **PASS** | exit `2`; `1` preflight; `0` selected-loop calls; full root absent; marker absent |
| QUICK-above-threshold branch | **PASS** | observation continued four-rank smoke; no D16 gate failure |
| Fresh Task1→Task2→Task3 chain | **PASS** | `CHAIN_PASS_COUNT=1`; trace/memory `4/4` |
| Task1 public smoke | **PASS** | `SMOKE_PASS_COUNT=1` |
| `git diff --check` | **PASS** | exit `0` |

### Key numeric D16 values

| Quantity | Expected / rule | Actual synthetic-controller value | Delta |
|----------|-----------------|----------------------------------:|------:|
| Rank-0 elapsed time (QUICK above-threshold fixture) | positive finite | `30,000 s` | `0 s` |
| Estimated full time | `30,000 × 256` | `7,680,000 s` | `0 s` |
| Gate threshold | frozen | `7,200 s` | `0 s` |
| Estimate / threshold | `> 1` means above threshold | `1,066.6667×` | — |
| Full-pass selected calls | exactly `256` | `256` | `0` |
| Full-pass total calls | `1 + 256` | `257` | `0` |
| Full-above-threshold selected calls | exactly `0` | `0` | `0` |
| Total fake torchrun calls | matrix-derived | `281` | `0` |
| Current unit cases | `49` | `49` | `0` |
| Current integration cases | `38` | `38` | `0` |

### Fresh-chain numeric context

The fresh synthetic chain additionally recorded trace files=`4`, memory JSON=`4`, Task2 dataset
rows=`2`, validation MSE=`3.0`, test MSE=`0.5`, reload max absolute prediction delta=`0.0`, and
Task3 rank-0 step/forward/backward/optimizer=`22.5/6.0/11.0/2.5 ms`; simulator wall time was
`0.5 s`. These values are wiring/regression metrics only.

## 4. Root Cause and Fix Record

The first integration rerun failed after all behavior cases with `expected '18', got '281'`.
The behavior matrix intentionally added independent preflight calls and full selected-rank calls,
but two per-invocation warmup/profile assertions still used the historical count. The clean-clone
replay also searched for the old `PASS_COUNT=35`, while the V21 verifier had only a historical
`PASS_COUNT=31` check.

The minimal root-cause fix was to update current per-invocation assertions to `281`, update replay
to `PASS_COUNT=38`, and add separate V21 checks for current `PASS_COUNT=49` and `PASS_COUNT=38`
without rewriting the historical Session 47 transcript. The rerun then passed all checks above.
No fallback, source switching, threshold change, rank-count change, or qualification-label change
was introduced.

## 5. Evidence Boundary and Open Status

All values in this report come from fake `torchrun` fixtures and local controller tests. The run did
not execute CUDA, H800, RJob, Docker, real Nsight, real SQLite/NVTX, or an actual independent rank-0
preflight. Therefore:

```text
Evidence class = local_synthetic_not_gpu_qualification
I53 = OPEN / HIGH / WATCH
Gate B1 = BLOCKED
real_pre_dataset = NOT QUALIFIED
release_pre_dataset = NOT QUALIFIED
AE-ready = NO
```

The report proves the local orchestration contract and regression behavior only; it is not a real
performance or release qualification result.
