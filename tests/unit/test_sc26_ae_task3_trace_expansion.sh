#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

# shellcheck source=SC26-AE/lib/task3_simulation.sh
source "${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh"

TEST_ROOT=$(mktemp -d -p /data/ycfeng/tmp sc26-ae-task3-trace-expansion.XXXXXX)
TASK3_META_PYTHON=$(command -v python3)

write_trace() {
    local directory=$1
    local rank=$2
    mkdir -p "${directory}"
    printf 'rank:%s:forward_step(duration=1.0)\n' "${rank}" > \
        "${directory}/trace_rank${rank}_case.txt"
}

assert_trace_rank() {
    local directory=$1
    local rank=$2
    local path="${directory}/trace_rank${rank}_case.txt"
    [[ -f "${path}" ]]
    grep -Fq "rank:${rank}:forward_step" "${path}"
}

run_gpt_case() {
    local case_root="${TEST_ROOT}/gpt"
    TASK3_MODEL_KEY=gpt175b
    TASK3_TRACE_DIR="${case_root}/source"
    TASK3_RUN_ROOT="${case_root}/run"
    TASK3_WORLD_SIZE=1024
    TASK3_PP=8
    TASK3_TP=8
    TASK3_DP=16
    TASK3_EXP=1
    mkdir -p "${TASK3_RUN_ROOT}/provenance"
    local rank
    for rank in 0 128 256 384 512 640 768 896; do
        write_trace "${TASK3_TRACE_DIR}" "${rank}"
    done

    task3_materialize_simulator_trace

    [[ "${TASK3_SIMULATOR_TRACE_DIR}" == "${TASK3_RUN_ROOT}/simulator_trace" ]]
    [[ $(find "${TASK3_SIMULATOR_TRACE_DIR}" -maxdepth 1 -type f -name '*.txt' -printf '.' | wc -c) -eq 1024 ]]
    assert_trace_rank "${TASK3_SIMULATOR_TRACE_DIR}" 0
    assert_trace_rank "${TASK3_SIMULATOR_TRACE_DIR}" 127
    assert_trace_rank "${TASK3_SIMULATOR_TRACE_DIR}" 128
    assert_trace_rank "${TASK3_SIMULATOR_TRACE_DIR}" 1023
    python3 -B - "${TASK3_RUN_ROOT}/provenance/simulator_trace_expansion.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["source_trace_file_count"] == 8
assert payload["destination_trace_file_count"] == 1024
assert payload["representative_coordinates"] == ["pp"]
PY
}

run_qwen_case() {
    local case_root="${TEST_ROOT}/qwen"
    TASK3_MODEL_KEY=qwen3_a30b
    TASK3_TRACE_DIR="${case_root}/source"
    TASK3_RUN_ROOT="${case_root}/run"
    TASK3_WORLD_SIZE=256
    TASK3_PP=8
    TASK3_TP=8
    TASK3_DP=4
    TASK3_EXP=4
    mkdir -p "${TASK3_RUN_ROOT}/provenance"
    local pp_rank ep_rank rank
    for pp_rank in 0 1 2 3 4 5 6 7; do
        for ep_rank in 0 1 2 3; do
            rank=$((pp_rank * 32 + ep_rank * 8))
            write_trace "${TASK3_TRACE_DIR}" "${rank}"
        done
    done

    task3_materialize_simulator_trace

    [[ $(find "${TASK3_SIMULATOR_TRACE_DIR}" -maxdepth 1 -type f -name '*.txt' -printf '.' | wc -c) -eq 256 ]]
    assert_trace_rank "${TASK3_SIMULATOR_TRACE_DIR}" 7
    assert_trace_rank "${TASK3_SIMULATOR_TRACE_DIR}" 8
    assert_trace_rank "${TASK3_SIMULATOR_TRACE_DIR}" 31
    assert_trace_rank "${TASK3_SIMULATOR_TRACE_DIR}" 255
    python3 -B - "${TASK3_RUN_ROOT}/provenance/simulator_trace_expansion.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["source_trace_file_count"] == 32
assert payload["destination_trace_file_count"] == 256
assert payload["representative_coordinates"] == ["pp", "ep"]
PY
}

run_full_world_case() {
    local case_root="${TEST_ROOT}/full"
    TASK3_MODEL_KEY=gpt175b
    TASK3_TRACE_DIR="${case_root}/source"
    TASK3_RUN_ROOT="${case_root}/run"
    TASK3_WORLD_SIZE=2
    TASK3_PP=1
    TASK3_TP=1
    TASK3_DP=2
    TASK3_EXP=1
    write_trace "${TASK3_TRACE_DIR}" 0
    write_trace "${TASK3_TRACE_DIR}" 1

    task3_materialize_simulator_trace

    [[ "${TASK3_SIMULATOR_TRACE_DIR}" == "${TASK3_TRACE_DIR}" ]]
}

run_invalid_inventory_case() {
    local case_root="${TEST_ROOT}/invalid"
    TASK3_MODEL_KEY=gpt175b
    TASK3_TRACE_DIR="${case_root}/source"
    TASK3_RUN_ROOT="${case_root}/run"
    TASK3_WORLD_SIZE=1024
    TASK3_PP=8
    TASK3_TP=8
    TASK3_DP=16
    TASK3_EXP=1
    mkdir -p "${TASK3_RUN_ROOT}/provenance"
    write_trace "${TASK3_TRACE_DIR}" 0
    local status
    set +e
    (set -e; task3_materialize_simulator_trace) >"${case_root}/error.log" 2>&1
    status=$?
    set -e
    if ((status == 0)); then
        printf 'FAIL: invalid GPT trace inventory was accepted\n' >&2
        return 1
    fi
    grep -Fq 'expected world=1024, GPT PP=8, or Qwen PP×EP=8×1' \
        "${case_root}/error.log"
}

run_gpt_case
run_qwen_case
run_full_world_case
run_invalid_inventory_case

printf 'PASS_COUNT=4\n'
printf 'TEST_ROOT=%s\n' "${TEST_ROOT}"
