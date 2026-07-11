#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODE=${MODE:-scaling}
MODEL_PROFILE=${MODEL_PROFILE:-smoke}
TRAIN_ITERS=${TRAIN_ITERS:-3}
TRACE_START=${TRACE_START:-1}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-1}
DDP_BUCKET_SIZE=${DDP_BUCKET_SIZE:-10000000}

cat <<EOF
[Example] Megatron DDP overlap tracing
- Base script: examples/pretrain_deepseek_v3_moe.sh
- overlap_grad_reduce: enabled
- DDP overlap tracing: auto-enabled by validate_args when --overlap-grad-reduce and tracing are on
- Explicit --trace-ddp-grad-overlap is intentionally omitted in this wrapper
EOF

exec bash "${SCRIPT_DIR}/pretrain_deepseek_v3_moe.sh"
