# Test Report: I51 Task1 Producer Provenance

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-20 | Recorded the focused I51 RED→GREEN cycle, Task1 regression, and exact-snapshot producer-blob verification |

**Date:** 2026-07-20  
**Repository:** `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717`  
**Baseline HEAD:** `0e0d8c4c57c8c241b46b36390eeec632b6f40492`  
**Environment:** no active conda env; Bash `5.2.21(1)-release`; `/usr/bin/python3` `3.12.3`  
**Evidence class:** `local_synthetic_not_gpu_qualification`

## 1. Test Script Information

### Modified implementation and test

- Implementation: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/SC26-AE/lib/task1_trace.sh`
- Unit test: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/unit/test_sc26_ae_task1_source_provenance.sh`
- Integration test: `/data/ycfeng/Megatron-LM-sc26-ae-exec-clean-20260717/tests/integration/test_sc26_ae_task1_contracts.sh`

### Exact commands

RED and GREEN use the same focused command before and after the implementation change:

```bash
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp \
  bash tests/unit/test_sc26_ae_task1_source_provenance.sh
```

Affected Task1 regression:

```bash
SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp \
  bash tests/integration/test_sc26_ae_task1_contracts.sh
```

Syntax and diff checks:

```bash
bash -n SC26-AE/lib/task1_trace.sh
bash -n tests/unit/test_sc26_ae_task1_source_provenance.sh
git diff --check
```

Exact-snapshot verification used a temporary Git index without changing the real index:

```bash
SNAP_DIR=$(mktemp -d /data/ycfeng/sc26-ae-test-tmp/session57-i51-index-v2.XXXXXX)
INDEX_FILE="${SNAP_DIR}/index"
GIT_INDEX_FILE="${INDEX_FILE}" git read-tree HEAD
GIT_INDEX_FILE="${INDEX_FILE}" git add -- SC26-AE/lib/task1_trace.sh
TREE=$(GIT_INDEX_FILE="${INDEX_FILE}" git write-tree)
SNAPSHOT_COMMIT=$(printf 'Session 57 I51 producer verification snapshot\n' | \
  GIT_AUTHOR_NAME=sc26-ae-verifier \
  GIT_AUTHOR_EMAIL=sc26-ae-verifier@example.invalid \
  GIT_COMMITTER_NAME=sc26-ae-verifier \
  GIT_COMMITTER_EMAIL=sc26-ae-verifier@example.invalid \
  git commit-tree "${TREE}" -p HEAD)
source SC26-AE/lib/task1_trace.sh
for source_path in \
    examples/update_pretrain_gpt.sh \
    examples/pretrain_qwen3_30b_a3b_moe.sh \
    examples/pretrain_deepseek_v3_moe.sh; do
  AE_T1_SOURCE_SCRIPT="${PWD}/${source_path}"
  ae_task1_assert_source_provenance "${PWD}" "${SNAPSHOT_COMMIT}"
  ae_task1_assert_source_provenance "${PWD}" "${SNAPSHOT_COMMIT}"
done
```

## 2. Validation Criteria

1. The clean 11-file Task1 producer surface must match the pinned Git snapshot.
2. Mutation of the existing representative core file or any of the six newly bound AE producer
   files must fail with `Task1 tracked HEAD blob mismatch`.
3. The three public Task1 workflows must retain all `38` existing integration behaviors.
4. The three model source selections must pass both pre- and post-capture-equivalent snapshot
   checks: `3 × 2 = 6` helper invocations, each comparing `11` blobs, for `66` successful blob
   comparisons.
5. Both modified shell files must pass syntax validation; the real Git index must remain unstaged;
   `git diff --check` must pass.
6. No local result may promote Gate B1, real/release datasets, issuer authentication, or
   `AE-ready`.

## 3. Test Results and Evidence

### Summary

| Test / metric | Expected | Actual | Delta | Result |
|---------------|---------:|-------:|------:|--------|
| Initial omitted-wrapper rejection | `1` rejection | `0` rejections; mutation was accepted | `-1` | RED (expected) |
| Unit positive cases | `1` | `1` | `0` | PASS |
| Unit mutation rejection cases | `7` | `7` | `0` | PASS |
| Unit total `PASS_COUNT` | `8` | `8` | `0` | PASS |
| Task1 integration `PASS_COUNT` | `38` | `38` | `0` | PASS |
| Snapshot helper invocations | `6` | `6` | `0` | PASS |
| Snapshot blob comparisons | `66` | `66` | `0` | PASS |
| Shell syntax files | `2` | `2` | `0` | PASS |
| Real-index staged paths | `0` | `0` | `0` | PASS |
| Diff-check failures | `0` | `0` | `0` | PASS |

### RED evidence

- Log: `logs/session57-i51-producer-surface-red-20260720.log`
- Exit code: `1`
- Bytes: `73`
- SHA256: `b19efec532edb3091c2b30c58d4f46a8d350fd08a494eb33d60fc81adba64972`
- Output:

```text
mutated Task1 source was unexpectedly accepted: SC26-AE/task1_gpt175b.sh
```

**Root cause:** `ae_task1_assert_source_provenance` bound only the selected model script and four
core Megatron files. It omitted the three public wrappers, `common.sh`, `task1_trace.sh`, and
`artifact_manifest.py`, even though those files determine capture orchestration and publication.

### GREEN evidence

| Evidence | Exit | Bytes | SHA256 | Key output |
|----------|-----:|------:|--------|------------|
| `logs/session57-i51-producer-surface-green-20260720.log` | `0` | `13` | `a1e3f7a9b68a6ef5b20d75a3ecd9229755e1d9a12ec212a0c92ca7a0a69580fb` | `PASS_COUNT=8` |
| `logs/session57-i51-task1-integration-green-20260720.log` | `0` | `2,512` | `e3c53f116a0735b7537c861006d7feb57548061aa4626d556b99bc33253dc6c9` | `PASS_COUNT=38` |
| `logs/session57-i51-current-snapshot-check-v2-20260720.log` | `0` | `597` | `54c0cf1ec783b8b8756190167959a85467aa305f8b72a71d89b0c8d610259463` | `SOURCE_CHECK_COUNT=6`, `GIT_DIFF_CHECK=PASS` |

The verification snapshot was:

```text
SNAPSHOT_TREE=1d8fb120ec53575267e44de3ff0c69dd1425d442
SNAPSHOT_COMMIT=68b8bd23ec1b375ebc57f5bcb384a0a0468ddb2a
```

The first broad temporary-index orchestration attempt exited before creating a log and did not
alter the real index. The v2 command reduced the snapshot to the single modified producer file,
set an explicit ephemeral commit identity, and passed. This was a command-harness correction, not
a product fallback.

### Independent review

StepCode Claude returned `WATCH` and explicitly found the minimal repair sufficient for local I51
closure. It did not require a second outer-tree field, per-file SHA256 artifact, or git-exported
execution. The recorded WATCH items are: maintain the fixed producer list when the producer surface
changes, and retain focused unit mutation coverage because `AE_TASK1_TEST_MODE=1` does not execute
the real-mode provenance bracket.

- Artifact: `.omx/artifacts/claude-act-as-an-independent-read-only-reviewer-for-sc26-ae-i51-in--2026-07-20T07-19-39-728Z.md`
- Provider exit: `0`
- Bytes: `7,905`
- SHA256: `9e403ad1b7ffb1f77c0957007805bc03286b5ae8bea33cbc50c0f6995d97f680`

## 4. Conclusion and Qualification Boundary

The focused implementation and verification satisfy I51's local tracked-snapshot and pre/post byte
identity requirements. The result proves producer reproducibility/integrity against a recorded Git
commit; it is not adversarial code signing and does not authenticate an external issuer. Gate B1
remains `BLOCKED`, `real_pre_dataset` and `release_pre_dataset` remain `NOT QUALIFIED`,
`AE-ready=NO`, and the overall workflow remains `INCOMPLETE`.
