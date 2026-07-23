# Cross-Worktree `task_memory` Merge Audit

## Modification History

| Date | Summary of Changes |
|------|--------------------|
| 2026-07-23 | Compared canonical and six related worktrees; merged 18 missing current-workflow test reports with source commits and SHA256 provenance. |

## Scope and decision

The canonical delivery branch is `sc26-ae-functional` in `/data/ycfeng/sc26_ae_task3_qwen`; the pre-merge base commit was `b2c4376076b8509fae85823103bd82f5dd936045`. This audit intentionally merges only direct test reports for the formal `task_2026-07-15_sc26_ae_workflow`; it does not copy nested runtime logs, caches, OMX state, or unrelated historical tasks.

## Inventory evidence

| Source | Files scanned | Missing relative paths | Same hash | Conflicts | Decision |
|--------|--------------:|----------------------:|----------:|---------:|----------|
| `qwen-c368` | 27 | 27 | 0 | 0 | Excluded: unrelated/legacy task or duplicate runtime tree |
| `qwen-outer` | 27 | 27 | 0 | 0 | Excluded: unrelated/legacy task or duplicate runtime tree |
| `sc26-ae` | 5223 | 4975 | 234 | 14 | Merged direct current-workflow reports only |
| `sc26-ae-exec-clean-20260717` | 740 | 409 | 319 | 12 | Merged direct current-workflow reports only |
| `task3-clean2` | 331 | 0 | 320 | 11 | Excluded: unrelated/legacy task or duplicate runtime tree |
| `task3-provenance` | 331 | 0 | 320 | 11 | Excluded: unrelated/legacy task or duplicate runtime tree |

The raw machine-readable inventory used for this decision was written outside the repository at `/data/ycfeng/tmp/sc26_ae_task_memory_inventory.json` and `/data/ycfeng/tmp/sc26_ae_task_memory_diff.json`; no `/tmp` path was used.

## Merged reports

| Relative path | Source worktree | Source branch/commit | SHA256 | Size (bytes) |
|----------------|-----------------|----------------------|--------|-------------:|
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-17_gate_b_recon_checkpoint.md` | `sc26-ae` | `sc26-ae` / `3c91d15bc035` | `29747dddfb8d1a642deb30a3bd467d73a0010c4d6b5fe70728c9aefaac2fbd38` | 39113 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-18_gate_b2_predict_quota_block.md` | `sc26-ae` | `sc26-ae` / `3c91d15bc035` | `0db1e70a615c67bd29abbb294333abc47bb81aff56959ae0d5c5395d5c7c3810` | 30413 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-18_gate_b2_source_truth_schema_fix.md` | `sc26-ae` | `sc26-ae` / `3c91d15bc035` | `97520aacb3638690dcba97930f93663b51fbdc54dd5878d4a0b31336f00b54f0` | 7556 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-18_gate_b2_static_validator_binding_failure.md` | `sc26-ae` | `sc26-ae` / `3c91d15bc035` | `4adf1dda7d47df81c81ab37e46a5fa4e22dff27e6d4fbf41525b2760e5d7db38` | 9025 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-18_gate_b2_worker_authority_path_failure.md` | `sc26-ae` | `sc26-ae` / `3c91d15bc035` | `691518db41c7383567651d343e62b4b19c93f1ca402f391f099e63b59d0ed636` | 16845 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-20_dsv3_scaling_sequence_parallel.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `8963f1c462ba2e97d2762903ea9dcfaf1c43faa69db85312849cefa65187b0d1` | 50840 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-20_i51_producer_provenance.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `73d7ea20dc7cabbac9cfe6bafc914f55cf8dbc31ee1f4d10fe0fec0da9d67cf8` | 7003 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-20_i55_outer_provenance.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `c9fa3b60363e378c30d666ae4a818f930c66686bcbaab5f05c1127e86d4ebaae` | 11057 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-20_i56_path_contract.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `7c995a0704387849c7a3032912af65be88962558d98e3798deaaa39a2849c5a0` | 9997 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-21_fake_level_readme_and_regression.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `f2475632fd85bf26bb59678b29b20a90f87bad9dce5473c44ce82f914d286ebd` | 4598 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-21_functional_prebaked.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `48688b9308a1199c95617b3797accf22bdbdda8b7c5317666a7e22794d055441` | 2248 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-21_functional_prebaked_package_tests.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `8e4ede1b52a735c418a9f0e4cd5756062f349ce6549e0d84bbb202edac34a8d0` | 2274 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-22_i72_timeline_construction.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `46b01f425d6605668ae725d841202dd96095bce218630d5ab3376878428548b2` | 27483 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-22_i73_kernel_blueprint_projection.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `d61827b0aee905cc628776346e5ea521ec12ce01c2c74b253000c2f056fdf6b8` | 7478 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-22_i74_uid_scope.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `064c84426bc7f981971d0c3428a4c32c3a12bf224781dc650e6801f2ea7c7ea9` | 5855 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-22_i75_missing_kernel_policy.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `e7b5c830c631c673bae52479023013bfc0bb0bb323e1f08000e41c2dc3053a99` | 5226 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-22_qwen3_task1_fresh.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `25068b50e7da6ba46e54bece4c270301934866d794a91e92d46635625e783463` | 5121 |
| `task_2026-07-15_sc26_ae_workflow/test_report_2026-07-22_qwen3_task2_fresh.md` | `sc26-ae-exec-clean-20260717` | `sc26-ae-exec-clean-20260717` / `7579593737e2` | `2f25a6c925265cde19abcce4571331ef612cc39d8633e240106f1431b2d938be` | 5048 |

## Exclusion rationale

- The 4,668+ nested `logs/.../runtime/task_memory/...` files on `sc26-ae` are archived worker transcripts and historical experiments, not direct records for the current canonical workflow; copying them would add thousands of duplicate or superseded reports.
- The detached `qwen-c368` and `qwen-outer` trees contain the February `sim_restructure` task and are outside the current GPT-175B/Qwen3-A3B AE delivery scope.
- Conflicting versions of `plan.md`, `progress.md`, `review.md`, `summary.md`, and other ledgers were not overwritten. The canonical ledger remains authoritative; this report records the merge boundary.
- No runtime artifact, bundle, predictor, trace, or source code was changed by this merge. Existing bundle provenance therefore remains tied to its original producer commit and must be rebuilt only if a future release requires the new documentation commit.

## Test Script Information

- Inventory script: `/data/ycfeng/tmp/inventory_task_memory.py`
- Inventory outputs: `/data/ycfeng/tmp/sc26_ae_task_memory_inventory.json`, `/data/ycfeng/tmp/sc26_ae_task_memory_diff.json`
- Verification command:
  ```bash
  git diff --check
  python - <<'PY'  # compare each merged report against its recorded SHA256
  ...
  PY
  ```
- Environment: `/data/ycfeng/sc26_ae_task3_qwen`; Python 3.12.3; Git 2.43.0; all temporary outputs under `/data/ycfeng/tmp`.

## Validation Criteria

- Every selected report exists under the canonical task directory.
- Every selected report's observed SHA256 equals the value recorded in this audit.
- No canonical ledger is overwritten and no runtime artifact is modified.
- `git diff --check` exits successfully.

## Test Results and Evidence

| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Worktree inventory | 7 sources (canonical + 6 related) | 7 sources; 6,679 non-canonical records compared | PASS |
| Selected report count | 18 | 18 copied | PASS |
| Merged SHA256 checks | 18/18 | 18/18 | PASS |
| Canonical task_memory tracked files | increase by 19 (18 reports + audit) | 334 → 353 | PASS |
| `git diff --check` | exit 0 | exit 0 | PASS |
| Runtime/source/bundle changes | 0 | 0 | PASS |

## Verification

- `git diff --check` must pass after this documentation-only merge.
- Every merged report must exist in canonical with the SHA256 shown above.
- The canonical branch remains a single branch for both formal models; DeepSeek records remain historical/deferred.
