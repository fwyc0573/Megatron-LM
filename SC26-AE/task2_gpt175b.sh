#!/usr/bin/env bash
# Run the shared Echo Task2 predictor workflow for GPT-175B.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TASK2_MODEL_PROFILE=${MODEL_PROFILE:-full}
exec "${SCRIPT_DIR}/lib/task2_echo.sh" gpt175b "$@"
