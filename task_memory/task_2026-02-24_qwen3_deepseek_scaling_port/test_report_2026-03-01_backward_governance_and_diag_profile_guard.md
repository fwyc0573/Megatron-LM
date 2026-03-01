## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added validation report for backward-governance implementation and advanced-diagnostics opt-in guard in DeepSeek/Qwen example scripts |

## Test Report: Backward Governance + Advanced Diagnostics Guard

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

#### 1.1 Modified Files

- `examples/pretrain_deepseek_v3_moe.sh`
  - Added `ADVANCED_DIAGNOSTICS=0|1` and fail-fast guard for non-baseline diagnostic flags.
- `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - Added `ADVANCED_DIAGNOSTICS=0|1` and fail-fast guard for qwen-side non-baseline diagnostic flags.
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/plan.md`
  - Added governance addendum: official semantics freeze + diagnostics scope lock.
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/notes.md`
  - Added frozen metric and diagnostics policy note.
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/issues.md`
  - Added Issues 69-71 (semantics freeze / diagnostics opt-in / scope lock).
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/progress.md`
  - Added completion records for governance implementation.

#### 1.2 Reproducible Commands

```bash
# 1) Shell syntax checks
bash -n examples/pretrain_deepseek_v3_moe.sh
bash -n examples/pretrain_qwen3_30b_a3b_moe.sh

# 2) DeepSeek guard negative case (advanced flag without acknowledgement)
ADVANCED_DIAGNOSTICS=0 SCALING_DISABLE_DDP_WRAP=1 MODE=invalid \
  bash examples/pretrain_deepseek_v3_moe.sh

# 3) DeepSeek guard positive acknowledgement case
ADVANCED_DIAGNOSTICS=1 SCALING_DISABLE_DDP_WRAP=1 MODE=invalid \
  bash examples/pretrain_deepseek_v3_moe.sh

# 4) Qwen guard negative case
ADVANCED_DIAGNOSTICS=0 SCALING_COMM_ADJACENT_COPY_ITERS=2 MODE=invalid \
  bash examples/pretrain_qwen3_30b_a3b_moe.sh

# 5) Qwen guard positive acknowledgement case
ADVANCED_DIAGNOSTICS=1 SCALING_COMM_ADJACENT_COPY_ITERS=2 MODE=invalid \
  bash examples/pretrain_qwen3_30b_a3b_moe.sh
```

### 2) Validation Criteria

1. Scripts remain syntactically valid after introducing guard logic.
2. Advanced diagnostic flags must fail fast when `ADVANCED_DIAGNOSTICS=0`.
3. With `ADVANCED_DIAGNOSTICS=1`, advanced guard should pass and script should continue to later argument/mode checks.
4. Governance decisions are persisted in `task_memory` docs (`plan/notes/issues/progress`).

### 3) Test Results and Evidence

| Check | Result | Evidence |
|------|--------|----------|
| `bash -n` DeepSeek script | PASS | exit code `0` |
| `bash -n` Qwen script | PASS | exit code `0` |
| DeepSeek guard negative case | PASS | exits with advanced-diagnostics guard error |
| DeepSeek guard positive case | PASS | advanced guard bypassed; next error is unsupported mode |
| Qwen guard negative case | PASS | exits with advanced-diagnostics guard error |
| Qwen guard positive case | PASS | advanced guard bypassed; next error is unsupported mode |

Key log excerpts:

- DeepSeek negative case:
  - `[ERROR] Advanced diagnostics flags are set but ADVANCED_DIAGNOSTICS=0.`
  - `Active advanced flags: SCALING_DISABLE_DDP_WRAP=1`
- DeepSeek positive case:
  - `[ERROR] Unsupported MODE=invalid. Use distributed or scaling.`
- Qwen negative case:
  - `[ERROR] Advanced diagnostics flags are set but ADVANCED_DIAGNOSTICS=0.`
  - `Active advanced flags: SCALING_COMM_ADJACENT_COPY_ITERS=2`
- Qwen positive case:
  - `[ERROR] Unsupported MODE=invalid. Use distributed or scaling.`

### 4) Conclusion

- Governance implementation is complete for this round:
  - official backward metric semantics and scope-lock policy are documented;
  - example scripts now enforce explicit acknowledgement for advanced diagnostics;
  - baseline path remains default and unchanged.
- No training-semantic code path was modified in this round; changes are governance/documentation + launch-script guard behavior.
