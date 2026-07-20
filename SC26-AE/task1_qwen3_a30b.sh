#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/task1_trace.sh"
ae_run_task1 qwen3_a30b
