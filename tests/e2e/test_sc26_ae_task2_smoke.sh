#!/usr/bin/env bash
# Public-entry smoke test using the explicit synthetic fixture only.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
for entry in task2_gpt175b.sh task2_qwen3_a30b.sh task2_dsv3.sh; do
    test -x "$REPO_ROOT/SC26-AE/$entry"
    grep -q 'lib/task2_echo.sh' "$REPO_ROOT/SC26-AE/$entry"
done
bash "$REPO_ROOT/tests/integration/test_sc26_ae_task2_contract.sh"
printf '%s\n' 'PASS: public Task2 entries share the isolated synthetic contract test.'
