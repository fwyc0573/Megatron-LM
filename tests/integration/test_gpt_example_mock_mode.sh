#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
UPDATE_SCRIPT="${UPDATE_SCRIPT:-${REPO_ROOT}/examples/update_pretrain_gpt.sh}"
REALISTIC_SCRIPT="${REALISTIC_SCRIPT:-${REPO_ROOT}/examples/realistic_run_gpt.sh}"
TEST_ROOT="${TEST_ROOT:-$(mktemp -d /tmp/megatron-gpt-example-mock.XXXXXX)}"
PROJECT_ROOT="${TEST_ROOT}/project"
CAPTURE_FILE="${TEST_ROOT}/torchrun-args.txt"
FAKE_BIN="${REPO_ROOT}/tests/integration/fixtures/mock_torchrun_bin"
PASSED_CASES=0

mkdir -p "${PROJECT_ROOT}/logs"

assert_contains() {
    local expected=$1
    local file=${2:-${CAPTURE_FILE}}
    if ! grep -Fq -- "${expected}" "${file}"; then
        echo "Expected ${file} to contain: ${expected}" >&2
        echo "Observed content:" >&2
        cat "${file}" >&2
        exit 1
    fi
}

assert_not_contains() {
    local unexpected=$1
    local file=${2:-${CAPTURE_FILE}}
    if grep -Fq -- "${unexpected}" "${file}"; then
        echo "Expected ${file} to omit: ${unexpected}" >&2
        echo "Observed content:" >&2
        cat "${file}" >&2
        exit 1
    fi
}

assert_capture_line_count() {
    local expected=$1
    local actual
    actual=$(wc -l < "${CAPTURE_FILE}")
    if [[ ${actual} -ne ${expected} ]]; then
        echo "Expected ${expected} torchrun invocation(s), got ${actual}." >&2
        cat "${CAPTURE_FILE}" >&2
        exit 1
    fi
}

assert_status() {
    local expected=$1
    local actual=$2
    local context=$3
    if [[ ${actual} -ne ${expected} ]]; then
        echo "Expected ${context} to return ${expected}, got ${actual}." >&2
        exit 1
    fi
}

assert_compact_argv_sequence() {
    local -a expected=()
    local -a observed=()
    local argument encoded
    local start offset matches=0

    for argument in "$@"; do
        printf -v encoded '%q' "${argument}"
        expected+=("${encoded}")
    done

    mapfile -t observed < <(
        grep -oE 'ARGV\[[0-9]+\]=[^ ]*' "${CAPTURE_FILE}" |
            sed -E 's/^ARGV\[[0-9]+\]=//'
    )

    for ((start = 0; start + ${#expected[@]} <= ${#observed[@]}; start++)); do
        local sequence_matches=1
        for ((offset = 0; offset < ${#expected[@]}; offset++)); do
            if [[ "${observed[start + offset]}" != "${expected[offset]}" ]]; then
                sequence_matches=0
                break
            fi
        done
        if (( sequence_matches == 1 )); then
            matches=$((matches + 1))
        fi
    done

    if (( matches != 1 )); then
        echo "Expected exactly one contiguous argv sequence:" >&2
        printf '  %q\n' "$@" >&2
        echo "Observed capture:" >&2
        cat "${CAPTURE_FILE}" >&2
        exit 1
    fi
}

create_glob_data_paths() {
    local base_pattern=$1
    local sibling_base=$2
    local base

    for base in "${base_pattern}" "${sibling_base}"; do
        mkdir -p "${base}/data/output_prefix_gpt2"
        : > "${base}/data/output_prefix_gpt2/my-gpt2_text_document"
        : > "${base}/data/output_prefix_gpt2/gpt2-vocab.json"
        : > "${base}/data/output_prefix_gpt2/gpt2-merges.txt"
    done
}

pass_case() {
    local name=$1
    PASSED_CASES=$((PASSED_CASES + 1))
    echo "PASS: ${name}"
}

reset_capture() {
    : > "${CAPTURE_FILE}"
}

case_update_mock_overrides() {
    reset_capture
    CUDA_VISIBLE_DEVICES=3 \
    NCCL_DEBUG=INFO \
    BASE_PATH="${PROJECT_ROOT}" \
    LOG_ROOT="${TEST_ROOT}/update-logs" \
    MODEL_SIZE=tiny_h64 \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=1 \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-mock.log" 2>&1

    assert_capture_line_count 1
    assert_contains "--mock-data"
    assert_contains "--tokenizer-type NullTokenizer"
    assert_contains "--vocab-size 51200"
    assert_contains "--num-layers 4"
    assert_contains "--hidden-size 256"
    assert_contains "--num-attention-heads 4"
    assert_contains "--fake-world-size 1"
    assert_contains "${PROJECT_ROOT}/pretrain_llama.py"
    assert_contains "ARGV[8]=${PROJECT_ROOT}/pretrain_llama.py"
    assert_contains "CUDA_VISIBLE_DEVICES=3"
    assert_contains "NCCL_DEBUG=INFO"
    assert_not_contains "--data-path"
    assert_not_contains "--vocab-file"
    assert_not_contains "--merge-file"
    assert_not_contains "syntax error" "${TEST_ROOT}/update-mock.log"
    if [[ ! -d "${TEST_ROOT}/update-logs" ]]; then
        echo "Expected update script to honor LOG_ROOT." >&2
        exit 1
    fi
    pass_case "update mock mode and environment overrides"
}

case_update_legacy_data() {
    reset_capture
    CUDA_VISIBLE_DEVICES=4 \
    NCCL_DEBUG=INFO \
    BASE_PATH="${PROJECT_ROOT}" \
    LOG_ROOT="${TEST_ROOT}/update-legacy-logs" \
    MODEL_SIZE=tiny_h64 \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=0 \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-legacy.log" 2>&1

    assert_capture_line_count 1
    assert_contains "--data-path ${PROJECT_ROOT}/data/output_prefix_gpt2/my-gpt2_text_document"
    assert_contains "--vocab-file ${PROJECT_ROOT}/data/output_prefix_gpt2/gpt2-vocab.json"
    assert_contains "--merge-file ${PROJECT_ROOT}/data/output_prefix_gpt2/gpt2-merges.txt"
    assert_contains "--tokenizer-type GPT2BPETokenizer"
    assert_contains "--vocab-size 51200"
    assert_not_contains "--mock-data"
    assert_not_contains "--tokenizer-type NullTokenizer"
    pass_case "update legacy data arguments"
}

case_update_legacy_glob_path() {
    reset_capture
    local base_pattern="${TEST_ROOT}/update-glob/project*"
    local sibling_base="${TEST_ROOT}/update-glob/projectX"
    create_glob_data_paths "${base_pattern}" "${sibling_base}"

    BASE_PATH="${base_pattern}" \
    LOG_ROOT="${TEST_ROOT}/update-glob-logs" \
    MODEL_SIZE=tiny_h64 \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=0 \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-glob.log" 2>&1

    assert_capture_line_count 1
    assert_compact_argv_sequence \
        --data-path "${base_pattern}/data/output_prefix_gpt2/my-gpt2_text_document" \
        --vocab-file "${base_pattern}/data/output_prefix_gpt2/gpt2-vocab.json" \
        --merge-file "${base_pattern}/data/output_prefix_gpt2/gpt2-merges.txt" \
        --split 949,50,1
    pass_case "update preserves glob characters in legacy data argv"
}

case_update_invalid_mock() {
    reset_capture
    set +e
    TP=1 \
    PP=1 \
    BASE_PATH="${PROJECT_ROOT}" \
    LOG_ROOT="${TEST_ROOT}/update-invalid-mock-logs" \
    MODEL_SIZE=tiny \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=invalid \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-invalid-mock.log" 2>&1
    local status=$?
    set -e

    assert_status 1 "${status}" "update script with invalid MOCK_DATA"
    assert_capture_line_count 0
    assert_contains "ERROR: MOCK_DATA must be 0 or 1, got invalid" "${TEST_ROOT}/update-invalid-mock.log"
    pass_case "update invalid MOCK_DATA fails before torchrun"
}

case_update_legacy_env_defaults() {
    reset_capture
    env -u CUDA_VISIBLE_DEVICES -u NCCL_DEBUG -u FAKE_TORCHRUN_EXIT \
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_ROOT="${TEST_ROOT}/update-default-env-logs" \
        MODEL_SIZE=tiny_h64 \
        FAKE_WORLD_SIZE=1 \
        FAKE_PP=1 \
        FAKE_TP=1 \
        MOCK_DATA=1 \
        bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-default-env.log" 2>&1

    assert_capture_line_count 1
    assert_contains "CUDA_VISIBLE_DEVICES=7"
    assert_contains "NCCL_DEBUG=WARN"
    assert_contains "${PROJECT_ROOT}/pretrain_llama.py"
    pass_case "update legacy CUDA and NCCL defaults"
}

case_update_failure_status() {
    reset_capture
    set +e
    TP=1 \
    PP=1 \
    CUDA_VISIBLE_DEVICES=3 \
    BASE_PATH="${PROJECT_ROOT}" \
    LOG_ROOT="${TEST_ROOT}/update-failure-logs" \
    MODEL_SIZE=tiny_h64 \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=1 \
    FAKE_TORCHRUN_EXIT=23 \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-failure.log" 2>&1
    local status=$?
    set -e

    assert_status 1 "${status}" "update script when torchrun returns 23"
    assert_capture_line_count 1
    assert_contains "Python script failed for rank 0 with status 23" "${TEST_ROOT}/update-failure.log"
    assert_contains "ERROR: 1 configuration(s) failed." "${TEST_ROOT}/update-failure.log"
    pass_case "update aggregates a torchrun failure into status 1"
}

case_update_without_external_seq() {
    reset_capture
    local fake_seq_bin="${TEST_ROOT}/fake-seq-bin"
    local seq_marker="${TEST_ROOT}/fake-seq-invoked"
    mkdir -p "${fake_seq_bin}"
    cat > "${fake_seq_bin}/seq" <<'SH'
#!/bin/bash
: > "${SEQ_MARKER}"
exit 127
SH
    chmod +x "${fake_seq_bin}/seq"

    PATH="${fake_seq_bin}:${PATH}" \
    SEQ_MARKER="${seq_marker}" \
    BASE_PATH="${PROJECT_ROOT}" \
    LOG_ROOT="${TEST_ROOT}/update-without-external-seq-logs" \
    MODEL_SIZE=tiny_h64 \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=1 \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-without-external-seq.log" 2>&1

    assert_capture_line_count 1
    assert_contains "--fake-current-rank-id 0"
    assert_contains "Total ranks to simulate: 1" "${TEST_ROOT}/update-without-external-seq.log"
    if [[ -e "${seq_marker}" ]]; then
        echo "Expected update script to calculate PP ranks without invoking external seq." >&2
        exit 1
    fi
    pass_case "update calculates PP ranks without external seq"
}

case_update_invalid_topology() {
    local invalid_configs=(
        "negative_pp_and_tp 1 -1 -1 positive integer"
        "negative_world -1 1 1 positive integer"
        "zero_world 0 1 1 positive integer"
        "negative_pp 1 -1 1 positive integer"
        "negative_tp 1 1 -1 positive integer"
        "zero_pp 1 0 1 positive integer"
        "zero_tp 1 1 0 positive integer"
        "non_integer_world invalid 1 1 positive integer"
        "non_integer_pp 1 invalid 1 positive integer"
        "non_integer_tp 1 1 invalid positive integer"
        "expression_world 1+0 1 1 positive integer"
        "expression_pp 1 1+0 1 positive integer"
        "expression_tp 1 1 1+0 positive integer"
        "out_of_range_world 2147483648 1 1 between 1 and 2147483647"
        "out_of_range_pp 1 2147483648 1 between 1 and 2147483647"
        "out_of_range_tp 1 1 2147483648 between 1 and 2147483647"
        "overflow_world 9223372036854775808 1 1 between 1 and 2147483647"
        "non_divisible 3 2 1 Invalid configuration"
    )
    local entry

    for entry in "${invalid_configs[@]}"; do
        local name world_size pp_size tp_size expected_error
        read -r name world_size pp_size tp_size expected_error <<< "${entry}"
        reset_capture
        set +e
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_ROOT="${TEST_ROOT}/update-invalid-topology-${name}-logs" \
        MODEL_SIZE=tiny_h64 \
        FAKE_WORLD_SIZE="${world_size}" \
        FAKE_PP="${pp_size}" \
        FAKE_TP="${tp_size}" \
        MOCK_DATA=1 \
        bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-invalid-topology-${name}.log" 2>&1
        local status=$?
        set -e

        assert_status 1 "${status}" "update script with ${name} topology"
        assert_capture_line_count 0
        assert_contains "${expected_error}" "${TEST_ROOT}/update-invalid-topology-${name}.log"
        assert_not_contains "ALL SIMULATIONS COMPLETED!" "${TEST_ROOT}/update-invalid-topology-${name}.log"
    done

    pass_case "update rejects invalid topology before torchrun"
}

case_update_log_directory_failure() {
    reset_capture
    local log_root_file="${TEST_ROOT}/update-log-root-file"
    : > "${log_root_file}"

    set +e
    BASE_PATH="${PROJECT_ROOT}" \
    LOG_ROOT="${log_root_file}" \
    MODEL_SIZE=tiny_h64 \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=1 \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-log-directory-failure.log" 2>&1
    local status=$?
    set -e

    assert_status 1 "${status}" "update script when the log directory cannot be created"
    assert_capture_line_count 0
    assert_contains "ERROR: Failed to create log directory" "${TEST_ROOT}/update-log-directory-failure.log"
    assert_not_contains "ALL SIMULATIONS COMPLETED!" "${TEST_ROOT}/update-log-directory-failure.log"
    pass_case "update fails before torchrun when log directory creation fails"
}

case_update_tee_failure() {
    reset_capture
    local log_root="${TEST_ROOT}/update-tee-failure-logs"
    local log_dir="${log_root}/SIM_GPT_tiny_h64_Config1_WS1_PP1_TP1_DP1_nMICROB6_MICROB_SIZE1_GLOBAL_BATCH6"
    mkdir -p "${log_dir}/fake_rank_0.log"

    set +e
    BASE_PATH="${PROJECT_ROOT}" \
    LOG_ROOT="${log_root}" \
    MODEL_SIZE=tiny_h64 \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=1 \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-tee-failure.log" 2>&1
    local status=$?
    set -e

    assert_status 1 "${status}" "update script when tee cannot write the rank log"
    assert_capture_line_count 1
    assert_contains "ERROR: Failed to write log for rank 0" "${TEST_ROOT}/update-tee-failure.log"
    assert_not_contains "ALL SIMULATIONS COMPLETED!" "${TEST_ROOT}/update-tee-failure.log"
    pass_case "update propagates tee failure"
}

case_update_base_path_whitespace() {
    reset_capture
    set +e
    BASE_PATH="${TEST_ROOT}/project with space" \
    LOG_ROOT="${TEST_ROOT}/update-base-path-whitespace-logs" \
    MODEL_SIZE=tiny_h64 \
    FAKE_WORLD_SIZE=1 \
    FAKE_PP=1 \
    FAKE_TP=1 \
    MOCK_DATA=1 \
    bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-base-path-whitespace.log" 2>&1
    local status=$?
    set -e

    assert_status 1 "${status}" "update script with whitespace in BASE_PATH"
    assert_capture_line_count 0
    assert_contains "ERROR: BASE_PATH must not contain whitespace" "${TEST_ROOT}/update-base-path-whitespace.log"
    pass_case "update rejects whitespace in BASE_PATH before torchrun"
}

case_update_log_path_whitespace() {
    reset_capture
    local log_root="${TEST_ROOT}/update log root"
    local log_file="${log_root}/SIM_GPT_tiny_h64_Config1_WS1_PP1_TP1_DP1_nMICROB6_MICROB_SIZE1_GLOBAL_BATCH6/fake_rank_0.log"

    set +e
    (
        cd "${TEST_ROOT}"
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_ROOT="${log_root}" \
        MODEL_SIZE=tiny_h64 \
        FAKE_WORLD_SIZE=1 \
        FAKE_PP=1 \
        FAKE_TP=1 \
        MOCK_DATA=1 \
        bash "${UPDATE_SCRIPT}" > "${TEST_ROOT}/update-log-path-whitespace.log" 2>&1
    )
    local status=$?
    set -e

    assert_status 0 "${status}" "update script with whitespace in LOG_ROOT"
    assert_capture_line_count 1
    if [[ ! -f "${log_file}" ]]; then
        echo "Expected update script to preserve the LOG_ROOT path boundary: ${log_file}" >&2
        exit 1
    fi
    pass_case "update preserves whitespace in LOG_ROOT"
}

case_realistic_mock_overrides() {
    reset_capture
    (
        cd "${TEST_ROOT}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        NCCL_DEBUG=INFO \
        NCCL_DEBUG_SUBSYS=COLL \
        NCCL_SOCKET_IFNAME=eth0 \
        NCCL_P2P_DISABLE=1 \
        NCCL_IB_DISABLE=0 \
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_DIR="${TEST_ROOT}/realistic-logs" \
        MOCK_DATA=1 \
        bash "${REALISTIC_SCRIPT}" \
            1 0 8 6000 localhost 2 2 tiny_h64 > "${TEST_ROOT}/realistic-mock.log" 2>&1
    )

    assert_capture_line_count 1
    assert_contains "--nproc_per_node=8"
    assert_contains "--tensor-model-parallel-size 2"
    assert_contains "--pipeline-model-parallel-size 2"
    assert_contains "--global-batch-size 4"
    assert_contains "--hidden-size 256"
    assert_contains "--num-attention-heads 4"
    assert_contains "--mock-data"
    assert_contains "--main-tokenizer-type NullTokenizer"
    assert_contains "--tokenizer-type NullTokenizer"
    assert_contains "--vocab-size 51200"
    assert_contains "${PROJECT_ROOT}/pretrain_llama.py"
    assert_contains "ARGV[8]=${PROJECT_ROOT}/pretrain_llama.py"
    assert_contains "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    assert_contains "NCCL_DEBUG=INFO"
    assert_contains "NCCL_DEBUG_SUBSYS=COLL"
    assert_contains "NCCL_SOCKET_IFNAME=eth0"
    assert_contains "NCCL_P2P_DISABLE=1"
    assert_contains "NCCL_IB_DISABLE=0"
    assert_not_contains "--data-path"
    assert_not_contains "--vocab-file"
    assert_not_contains "--merge-file"
    if [[ ! -d "${TEST_ROOT}/realistic-logs" ]]; then
        echo "Expected realistic script to honor LOG_DIR." >&2
        exit 1
    fi
    pass_case "realistic mock mode, topology, and environment overrides"
}

case_realistic_legacy_defaults() {
    reset_capture
    local workdir="${TEST_ROOT}/realistic-default-work"
    mkdir -p "${workdir}"
    (
        cd "${workdir}"
        env -u CUDA_VISIBLE_DEVICES \
            -u NCCL_DEBUG \
            -u NCCL_DEBUG_SUBSYS \
            -u NCCL_SOCKET_IFNAME \
            -u NCCL_P2P_DISABLE \
            -u NCCL_IB_DISABLE \
            -u BASE_PATH \
            -u LOG_DIR \
            -u FAKE_TORCHRUN_EXIT \
            MOCK_DATA=0 \
            bash "${REALISTIC_SCRIPT}" \
                1 0 8 6001 localhost 2 2 tiny > "${TEST_ROOT}/realistic-defaults.log" 2>&1
    )

    assert_capture_line_count 1
    assert_contains "CUDA_VISIBLE_DEVICES=0,1,2,5,6,7"
    assert_contains "NCCL_DEBUG=TRACE"
    assert_contains "NCCL_DEBUG_SUBSYS=ALL"
    assert_contains "NCCL_SOCKET_IFNAME=ens81f0"
    assert_contains "NCCL_P2P_DISABLE=0"
    assert_contains "NCCL_IB_DISABLE=1"
    assert_contains "/research/d1/gds/ytyang/yichengfeng/Megatron-LM/pretrain_llama.py"
    assert_contains "--data-path /research/d1/gds/ytyang/yichengfeng/Megatron-LM/data/output_prefix_gpt2/my-gpt2_text_document"
    assert_contains "--vocab-file /research/d1/gds/ytyang/yichengfeng/Megatron-LM/data/output_prefix_gpt2/gpt2-vocab.json"
    assert_contains "--merge-file /research/d1/gds/ytyang/yichengfeng/Megatron-LM/data/output_prefix_gpt2/gpt2-merges.txt"
    assert_contains "--main-tokenizer-type GPT2BPETokenizer"
    assert_contains "--vocab-size 3200"
    assert_not_contains "--mock-data"
    if [[ ! -d "${workdir}/run_log" ]]; then
        echo "Expected realistic script to use the legacy run_log directory." >&2
        exit 1
    fi
    pass_case "realistic legacy data and environment defaults"
}

case_realistic_legacy_glob_path() {
    reset_capture
    local base_pattern="${TEST_ROOT}/realistic-glob/project*"
    local sibling_base="${TEST_ROOT}/realistic-glob/projectX"
    create_glob_data_paths "${base_pattern}" "${sibling_base}"

    (
        cd "${TEST_ROOT}"
        BASE_PATH="${base_pattern}" \
        LOG_DIR="${TEST_ROOT}/realistic-glob-logs" \
        MOCK_DATA=0 \
        bash "${REALISTIC_SCRIPT}" \
            1 0 8 6009 localhost 2 2 tiny_h64 > "${TEST_ROOT}/realistic-glob.log" 2>&1
    )

    assert_capture_line_count 1
    assert_compact_argv_sequence \
        --data-path "${base_pattern}/data/output_prefix_gpt2/my-gpt2_text_document" \
        --vocab-file "${base_pattern}/data/output_prefix_gpt2/gpt2-vocab.json" \
        --merge-file "${base_pattern}/data/output_prefix_gpt2/gpt2-merges.txt" \
        --split 949,50,1
    pass_case "realistic preserves glob characters in legacy data argv"
}

case_realistic_invalid_mock() {
    reset_capture
    set +e
    (
        cd "${TEST_ROOT}"
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_DIR="${TEST_ROOT}/realistic-invalid-mock-logs" \
        MOCK_DATA=invalid \
        bash "${REALISTIC_SCRIPT}" \
            1 0 8 6002 localhost 2 2 tiny > "${TEST_ROOT}/realistic-invalid-mock.log" 2>&1
    )
    local status=$?
    set -e

    assert_status 1 "${status}" "realistic script with invalid MOCK_DATA"
    assert_capture_line_count 0
    assert_contains "ERROR: MOCK_DATA must be 0 or 1, got invalid" "${TEST_ROOT}/realistic-invalid-mock.log"
    pass_case "realistic invalid MOCK_DATA fails before torchrun"
}

case_realistic_invalid_model() {
    reset_capture
    set +e
    (
        cd "${TEST_ROOT}"
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_DIR="${TEST_ROOT}/realistic-invalid-model-logs" \
        MOCK_DATA=1 \
        bash "${REALISTIC_SCRIPT}" \
            1 0 8 6003 localhost 2 2 invalid_model > "${TEST_ROOT}/realistic-invalid-model.log" 2>&1
    )
    local status=$?
    set -e

    assert_status 1 "${status}" "realistic script with invalid MODEL_SIZE"
    assert_capture_line_count 0
    assert_contains "Invalid MODEL_SIZE: invalid_model" "${TEST_ROOT}/realistic-invalid-model.log"
    pass_case "realistic invalid model fails before torchrun"
}

case_realistic_failure_status() {
    reset_capture
    set +e
    (
        cd "${TEST_ROOT}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        NCCL_SOCKET_IFNAME=eth0 \
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_DIR="${TEST_ROOT}/realistic-failure-logs" \
        MOCK_DATA=1 \
        FAKE_TORCHRUN_EXIT=23 \
        bash "${REALISTIC_SCRIPT}" \
            1 0 8 6004 localhost 2 2 tiny > "${TEST_ROOT}/realistic-failure.log" 2>&1
    )
    local status=$?
    set -e

    assert_status 23 "${status}" "realistic script when torchrun returns 23"
    assert_capture_line_count 1
    pass_case "realistic propagates torchrun status 23"
}

case_realistic_invalid_topology() {
    local invalid_configs=(
        "negative_pp_and_tp 1 8 -2 -2 positive integer"
        "negative_nnodes -1 8 2 2 positive integer"
        "negative_gpus 1 -1 2 2 positive integer"
        "negative_pp 1 8 -1 2 positive integer"
        "negative_tp 1 8 2 -1 positive integer"
        "zero_nnodes 0 8 2 2 positive integer"
        "zero_gpus 1 0 2 2 positive integer"
        "zero_pp 1 8 0 2 positive integer"
        "zero_tp 1 8 2 0 positive integer"
        "expression_nnodes 1+0 8 2 2 positive integer"
        "expression_gpus 1 8+0 2 2 positive integer"
        "expression_pp 1 8 2+0 2 positive integer"
        "expression_tp 1 8 2 2+0 positive integer"
        "out_of_range_nnodes 2147483648 8 2 2 between 1 and 2147483647"
        "out_of_range_gpus 1 2147483648 2 2 between 1 and 2147483647"
        "out_of_range_pp 1 8 2147483648 2 between 1 and 2147483647"
        "out_of_range_tp 1 8 2 2147483648 between 1 and 2147483647"
        "overflow_nnodes 9223372036854775808 8 2 2 between 1 and 2147483647"
        "non_divisible 1 7 2 2 Invalid topology"
    )
    local entry

    for entry in "${invalid_configs[@]}"; do
        local name nnodes gpus_per_node pp_size tp_size expected_error
        read -r name nnodes gpus_per_node pp_size tp_size expected_error <<< "${entry}"
        reset_capture
        set +e
        (
            cd "${TEST_ROOT}"
            BASE_PATH="${PROJECT_ROOT}" \
            LOG_DIR="${TEST_ROOT}/realistic-invalid-topology-${name}-logs" \
            MOCK_DATA=1 \
            bash "${REALISTIC_SCRIPT}" \
                "${nnodes}" 0 "${gpus_per_node}" 6005 localhost "${pp_size}" "${tp_size}" tiny_h64 \
                > "${TEST_ROOT}/realistic-invalid-topology-${name}.log" 2>&1
        )
        local status=$?
        set -e

        assert_status 1 "${status}" "realistic script with ${name} topology"
        assert_capture_line_count 0
        assert_contains "${expected_error}" "${TEST_ROOT}/realistic-invalid-topology-${name}.log"
    done

    pass_case "realistic rejects invalid topology before torchrun"
}

case_realistic_log_directory_failure() {
    reset_capture
    local log_dir_file="${TEST_ROOT}/realistic-log-directory-file"
    : > "${log_dir_file}"

    set +e
    (
        cd "${TEST_ROOT}"
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_DIR="${log_dir_file}" \
        MOCK_DATA=1 \
        bash "${REALISTIC_SCRIPT}" \
            1 0 8 6008 localhost 2 2 tiny_h64 > "${TEST_ROOT}/realistic-log-directory-failure.log" 2>&1
    )
    local status=$?
    set -e

    assert_status 1 "${status}" "realistic script when the log directory cannot be created"
    assert_capture_line_count 0
    assert_contains "ERROR: Failed to create log directory" "${TEST_ROOT}/realistic-log-directory-failure.log"
    pass_case "realistic fails before torchrun when log directory creation fails"
}

case_realistic_base_path_whitespace() {
    reset_capture
    set +e
    (
        cd "${TEST_ROOT}"
        BASE_PATH="${TEST_ROOT}/project with space" \
        LOG_DIR="${TEST_ROOT}/realistic-base-path-whitespace-logs" \
        MOCK_DATA=1 \
        bash "${REALISTIC_SCRIPT}" \
            1 0 8 6006 localhost 2 2 tiny_h64 > "${TEST_ROOT}/realistic-base-path-whitespace.log" 2>&1
    )
    local status=$?
    set -e

    assert_status 1 "${status}" "realistic script with whitespace in BASE_PATH"
    assert_capture_line_count 0
    assert_contains "ERROR: BASE_PATH must not contain whitespace" "${TEST_ROOT}/realistic-base-path-whitespace.log"
    pass_case "realistic rejects whitespace in BASE_PATH before torchrun"
}

case_realistic_log_path_whitespace() {
    reset_capture
    local log_dir="${TEST_ROOT}/realistic log directory"
    local log_file="${log_dir}/realistic_log_1nodes_8gpus_2pp_2tp_tiny_h64model.log"

    set +e
    (
        cd "${TEST_ROOT}"
        BASE_PATH="${PROJECT_ROOT}" \
        LOG_DIR="${log_dir}" \
        MOCK_DATA=1 \
        bash "${REALISTIC_SCRIPT}" \
            1 0 8 6007 localhost 2 2 tiny_h64 > "${TEST_ROOT}/realistic-log-path-whitespace.log" 2>&1
    )
    local status=$?
    set -e

    assert_status 0 "${status}" "realistic script with whitespace in LOG_DIR"
    assert_capture_line_count 1
    if [[ ! -f "${log_file}" ]]; then
        echo "Expected realistic script to preserve the LOG_DIR path boundary: ${log_file}" >&2
        exit 1
    fi
    pass_case "realistic preserves whitespace in LOG_DIR"
}

CASES=(
    update_mock_overrides
    update_legacy_data
    update_legacy_glob_path
    update_invalid_mock
    update_legacy_env_defaults
    update_failure_status
    update_without_external_seq
    update_invalid_topology
    update_log_directory_failure
    update_tee_failure
    update_base_path_whitespace
    update_log_path_whitespace
    realistic_mock_overrides
    realistic_legacy_defaults
    realistic_legacy_glob_path
    realistic_invalid_mock
    realistic_invalid_model
    realistic_failure_status
    realistic_invalid_topology
    realistic_log_directory_failure
    realistic_base_path_whitespace
    realistic_log_path_whitespace
)

if [[ ! -f "${UPDATE_SCRIPT}" ]]; then
    echo "Update script not found: ${UPDATE_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${REALISTIC_SCRIPT}" ]]; then
    echo "Realistic script not found: ${REALISTIC_SCRIPT}" >&2
    exit 1
fi

export PATH="${FAKE_BIN}:${PATH}"
export CAPTURE_FILE

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [case-name]" >&2
    exit 1
fi

selected_cases=("${CASES[@]}")
if [[ $# -eq 1 ]]; then
    selected_cases=()
    for case_name in "${CASES[@]}"; do
        if [[ ${case_name} == "$1" ]]; then
            selected_cases=("${case_name}")
            break
        fi
    done
    if [[ ${#selected_cases[@]} -eq 0 ]]; then
        echo "Unknown case: $1" >&2
        echo "Available cases: ${CASES[*]}" >&2
        exit 1
    fi
fi

for case_name in "${selected_cases[@]}"; do
    "case_${case_name}"
done

echo "PASS: ${PASSED_CASES}/${#selected_cases[@]} selected GPT example integration cases."
echo "Artifacts: ${TEST_ROOT}"
