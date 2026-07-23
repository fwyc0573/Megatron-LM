#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

echo "[1/2] Smoke simulate mode (preset: moe_tiny_2pp_1tp_2dp)"
python simu_main.py \
  --preset moe_tiny_2pp_1tp_2dp \
  --mode simulate \
  --no-visualize

echo
echo "[2/2] Smoke profile mode (same tiny dataset)"
python simu_main.py \
  --framework megatron-lm \
  --mode profile \
  --trace-dir simulation_inputs/megatron_operation_log/moe_tiny_2pp_1tp_2dp/global_ranks_profile \
  --database-dir simulation_inputs/megatron_operation_log/moe_tiny_2pp_1tp_2dp/database_profile \
  --world-size 4 \
  --pp-size 2 \
  --tp-size 1 \
  --exp-size 1 \
  --local-size 4 \
  --no-visualize

echo
echo "Smoke run finished successfully."
