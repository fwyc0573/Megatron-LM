## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Added unit-test environment bootstrap notes for LOCAL_RANK/NCCL-based test modules |
| 2026-02-24 | Added fix for Claude Code VSCode launch failure under root container with bypassPermissions |
| 2026-03-05 | Added checklist for shared-GPU OOM during distributed profiling runs |
| 2026-03-06 | Added `CUDA_DEVICE_MAX_CONNECTIONS=1` prerequisite for Megatron scaling-mode torchrun validation |
| 2026-03-12 | Added Echo-slowdown shell/Nsight compatibility notes for workflow validation |
| 2026-07-16 | Added safe `rlaunch`/`brainctl` status handling and predict-only quota-result interpretation |
| 2026-07-23 | Added the controller CPU-only Task3 `xgboost` dependency procedure without writing to `/tmp` |

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

## Controller CPU-Only Task3 Requires XGBoost

### Symptoms

- Functional prebaked Task3 reaches the simulator and then exits with:
  `Slowdown prediction requires xgboost; install it before enabling slowdown.`
- `/usr/bin/python3` is available, but `python3 -c 'import xgboost'` fails.
- `python3 -m venv` may also fail because the controller image does not include
  `ensurepip` / `python3.12-venv`.

### Root Cause

The CPU-only simulator imports the slowdown predictor at runtime. The controller Python includes
the numerical dependencies but does not necessarily include the XGBoost version used to train the
prebaked predictor. This is an environment dependency gap; it is not a missing-kernel condition and
does not require rerunning Task2.

### Verified Procedure

Install the exact Task2 XGBoost version into a dedicated directory under `/data/ycfeng/tmp`, then
export that directory through `PYTHONPATH` for Task3:

```bash
export TMPDIR=/data/ycfeng/tmp
export TEMP=/data/ycfeng/tmp
export TMP=/data/ycfeng/tmp
export PIP_CACHE_DIR=/data/ycfeng/tmp/pip-cache
export TASK3_CPU_PYDEPS=/data/ycfeng/tmp/sc26_ae_cpu_task3_pydeps_xgboost210

eval "$(curl -fsS http://deploy.i.shaipower.com/httpproxy)"
python3 -m pip install \
  --target "${TASK3_CPU_PYDEPS}" \
  --no-deps \
  'xgboost==2.1.0'

export PYTHONPATH="${TASK3_CPU_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
python3 -c 'import xgboost; assert xgboost.__version__ == "2.1.0"'
```

The controller already provides `numpy`, `pandas`, and `scipy`; verify those imports before using
`--no-deps`. Bind both `TASK3_META_PYTHON` and `TASK3_SIMULATOR_PYTHON` explicitly to `python3` in
the CPU-only Task3 command. Do not use this controller-only dependency layer as real worker or GPU
qualification evidence.

## Safe RJob Inspection and Predict-Only Result Handling

### Symptoms

- Running `rlaunch status <rjob-id>` unexpectedly creates a new RJob instead of querying an existing one.
- A `rlaunch --predict-only` command exits `0` even though its output contains `fail to pass quota check`.

### Root Cause

- `rlaunch` does not expose a read-only `status` subcommand; unrecognized positional text is treated as launch payload.
- The platform CLI may report quota rejection in stdout/stderr without returning a nonzero process status.

### Verified Commands

Use `brainctl` for read-only inspection:

```bash
brainctl get rjob <rjob-id> -n shai-core -o yaml
brainctl get replica -n shai-core -l 'rjob.brainpp.cn/rjob-name=<rjob-id>'
brainctl logs -n shai-core replica/<replica-name>
```

Delete an accidentally created RJob with:

```bash
brainctl delete rjob <rjob-id> -n shai-core
```

After deletion, verify that `brainctl get rjob -n shai-core` no longer lists the exact ID. Also verify that no local `brainctl rjob launch status ...` process remains; a blocked client can outlive its parent shell even after the server-side RJob is deleted. Terminate only the exact erroneous client PID, then repeat both process and RJob checks.

### Acceptance Rule

For `rlaunch --predict-only`, evaluate both the exit code and output text. Any explicit quota failure, including `fail to pass quota check`, is a FAIL even when the CLI exits `0`; do not proceed to live allocation.
