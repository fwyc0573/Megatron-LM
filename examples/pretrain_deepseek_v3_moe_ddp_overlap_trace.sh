#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-1}
export DO_TRACE=${DO_TRACE:-True}

echo "[INFO] Launching DeepSeek-V3 MoE with DDP overlap enabled."
echo "[INFO] Overlap tracing metadata is auto-enabled when overlap-grad-reduce and tracing are on."

exec bash "${SCRIPT_DIR}/pretrain_deepseek_v3_moe.sh"
