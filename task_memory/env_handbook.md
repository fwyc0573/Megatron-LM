## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Added unit-test environment bootstrap notes for LOCAL_RANK/NCCL-based test modules |
| 2026-02-24 | Added fix for Claude Code VSCode launch failure under root container with bypassPermissions |

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
