#!/usr/bin/env bash
# Public-entry smoke using local synthetic torchrun/Nsight fixtures only.
# It is not a real-GPU workload or pre-dataset qualification.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONTRACT_TEST="${REPO_ROOT}/tests/integration/test_sc26_ae_task1_contracts.sh"

for entry in task1_gpt175b.sh task1_qwen3_a30b.sh task1_dsv3.sh; do
    entry_path="${REPO_ROOT}/SC26-AE/${entry}"
    [[ -x "${entry_path}" ]]
    grep -Fq -- 'lib/task1_trace.sh' "${entry_path}"
done

printf 'EVIDENCE_CLASS=local_synthetic_not_gpu_qualification\n'
printf 'REAL_GPU_WORKLOAD_COUNT=0\n'
bash "${CONTRACT_TEST}"
printf '%s\n' 'PASS: all public Task1 entries satisfy the local synthetic smoke contract.'
printf 'SMOKE_PASS_COUNT=1\n'
