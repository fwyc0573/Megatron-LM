## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Added unit-test environment bootstrap notes for LOCAL_RANK/NCCL-based test modules |

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
