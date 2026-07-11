## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Added unit-test environment bootstrap notes for LOCAL_RANK/NCCL-based test modules |
| 2026-02-24 | Added fix for Claude Code VSCode launch failure under root container with bypassPermissions |
| 2026-03-05 | Added checklist for shared-GPU OOM during distributed profiling runs |
| 2026-03-06 | Added `CUDA_DEVICE_MAX_CONNECTIONS=1` prerequisite for Megatron scaling-mode torchrun validation |
| 2026-03-12 | Added Echo-slowdown shell/Nsight compatibility notes for workflow validation |

# Environment Handbook

## Unit Test Bootstrap for Distributed Test Utilities

Some unit tests import `tests/unit_tests/test_utilities.py`, which reads `LOCAL_RANK` at import time and initializes NCCL with `world_size=torch.cuda.device_count()`.

### Symptoms

- `KeyError: LOCAL_RANK`
- Test hang in `torch.distributed.init_process_group` when only one test process is launched but visible GPUs > 1

### Required Environment Setup

Run these tests with explicit single-GPU distributed env:

```bash
CUDA_VISIBLE_DEVICES=0 \
LOCAL_RANK=0 \
RANK=0 \
WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=295xx \
pytest ...
```

This ensures `torch.cuda.device_count()==1` in test utilities and avoids waiting for non-existent ranks.

## Claude Code VSCode Launch Fails with Exit Code 1 in Root Container

### Symptoms

- VSCode Claude panel immediately closes with `Error: Claude Code process exited with code 1`
- Logs include:
  - `--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons`
  - `permissionMode":"bypassPermissions"`

### Root Cause

`bypassPermissions` maps to `--dangerously-skip-permissions`, which is blocked when the process runs as `root` inside the container.

### Fix

1. Set Claude default permission mode to `default` in user config:

```json
{
  "permissions": {
    "defaultMode": "default"
  }
}
```

2. Enforce safe extension defaults in VSCode machine settings:

```json
{
  "claudeCode.initialPermissionMode": "default",
  "claudeCode.allowDangerouslySkipPermissions": false
}
```

3. Create managed settings file to avoid startup ENOENT noise:

```bash
mkdir -p /etc/claude-code/.claude/skills
printf '{}\n' > /etc/claude-code/managed-settings.json
```

## Shared-GPU OOM During Distributed Profiling

### Symptoms

- `torchrun` launches normally but fails during model/DDP/optimizer initialization with `torch.cuda.OutOfMemoryError`.
- `nvidia-smi` shows each GPU already occupied by long-running external jobs (e.g., >70GB used on 80GB cards).

### Quick Diagnostic

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits
```

### Mitigation

1. Schedule profiling on reserved/idle GPUs.
2. If immediate run is required, lower memory pressure first (`MODEL_PROFILE=smoke`, shorter `SEQ_LEN`, smaller parallel degrees where valid).
3. Re-run with a new `MASTER_PORT` after resources are available.

## Scaling-Mode `torchrun` Requires `CUDA_DEVICE_MAX_CONNECTIONS=1`

### Symptoms

- Early argument validation failure before model build:
  - `RuntimeError: Using async gradient all reduce requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1`

### Fix

Run scaling-mode repro or wall-clock commands with:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=<gpu_id> torchrun ...
```

### Notes

- This is required even for single-process scaling-mode repro because Megatron argument validation still checks the async gradient all-reduce precondition.
- Keep `MASTER_PORT` unique across repeated runs.


## Echo-slowdown Workflow Compatibility on This Machine

### Symptoms

- `run.sh` or `run-nsys.sh` fails with:
  - `set: Illegal option -o pipefail`
  - `unrecognised option '--python-backtrace=cuda'`
  - `ImportError` caused by importing the wrong external `utils` package
- `nsys profile` generates `.nsys-rep` but exits with code `143` on the multi-process training script.
- A fresh `kernel_metric` run is much heavier than the practical workflow needs.

### Fix

1. Invoke Echo shell wrappers with `bash`, not `sh`.
2. Export `PYTHONPATH=<repo>/Echo-slowdown:${PYTHONPATH:-}` inside Echo wrapper scripts so the local `utils` package wins.
3. Replace `jq`-based JSON extraction with a Python helper when `jq` is absent.
4. Use `nsys` flags compatible with Nsight Systems `2023.1.2.43` by removing unsupported Python backtrace options.
5. Accept the observed `143` return code only when the expected `.nsys-rep` file was actually produced, then continue to `nsys export`.
6. For practical end-to-end validation, prefer:

```bash
SKIP_KERNEL_METRIC=1 bash Echo-slowdown/run_all.sh
```

This reuses `Echo-slowdown/merge/input/kernel_metric_output.csv` and still exercises the real `slowdown_collection -> merge -> train -> predict` chain.
