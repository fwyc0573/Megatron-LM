#!/usr/bin/env bash
# Task3 provenance contract: real producers must use clean canonical files
# whose bytes match the outer repository's pinned sim-engine commit.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COMMON_SH="${REPO_ROOT}/SC26-AE/lib/common.sh"
TASK3_SH="${REPO_ROOT}/SC26-AE/lib/task3_simulation.sh"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "${TMP_PARENT}"
ROOT=$(mktemp -d "${TMP_PARENT%/}/sc26-ae-task3-provenance.XXXXXX")
FAKE_BIN="${ROOT}/bin"
mkdir -p "${FAKE_BIN}"
mkdir -p "${ROOT}/isolated-tools"

cat > "${FAKE_BIN}/git" <<'SH'
#!/usr/bin/env bash
if [[ " $* " == *" status --short --untracked-files=all "* ]]; then
    if [[ "${FAKE_GIT_STATUS:-dirty}" == dirty ]]; then
        printf ' M producer.py\n'
    fi
    exit 0
fi
if [[ -n "${FAKE_GIT_SIM_HEAD:-}" && " $* " == *" rev-parse HEAD "* && " $* " == *megatron-sim-engine* ]]; then
    printf '%s\n' "${FAKE_GIT_SIM_HEAD}"
    exit 0
fi
if [[ -n "${FAKE_GIT_PRODUCER_BLOB:-}" && " $* " == *" hash-object --no-filters "* ]]; then
    printf '%s\n' "${FAKE_GIT_PRODUCER_BLOB}"
    exit 0
fi
exec /usr/bin/git "$@"
SH
chmod +x "${FAKE_BIN}/git"

run_current_commits() {
    local mode=$1
    local producer_root=$2
    local status_mode=$3
    local simulated_head=${4:-}
    local builder=${5:-"${producer_root}/tools/data_prep/slowdown/build_ddp_slowdown_assets.py"}
    local scheduler=${6:-"${producer_root}/src/scheduler/mg_scheduling/mg_test.py"}
    local simulator=${7:-"${producer_root}/simu_main.py"}
    local producer_blob=${8:-}
    set +e
    OUTPUT=$(env PATH="${FAKE_BIN}:${PATH}" \
        FAKE_GIT_STATUS="${status_mode}" \
        FAKE_GIT_SIM_HEAD="${simulated_head}" \
        FAKE_GIT_PRODUCER_BLOB="${producer_blob}" \
        bash -c '
    set -euo pipefail
    source "$1"
    source "$2"
    TASK3_REPO_ROOT="$3"
    TASK3_SIM_ENGINE_ROOT="$4"
    TASK3_EXECUTION_MODE="$5"
    TASK3_BUILDER="$6"
    TASK3_SCHEDULER="$7"
    TASK3_SIMULATOR="$8"
    task3_current_commits
' -- "${COMMON_SH}" "${TASK3_SH}" "${REPO_ROOT}" \
        "${producer_root}" "${mode}" "${builder}" "${scheduler}" \
        "${simulator}" 2>&1)
    STATUS=$?
    set -e
}

run_current_commits real "${REPO_ROOT}/megatron-sim-engine" dirty
[[ ${STATUS} -ne 0 ]] || {
    printf 'FAIL: dirty Task3 producer was accepted\n%s\n' "${OUTPUT}" >&2
    exit 1
}
grep -Fq -- 'dirty' <<<"${OUTPUT}" || {
    printf '%s\n' "${OUTPUT}" >&2
    printf 'FAIL: dirty-producer error did not identify the root cause\n' >&2
    exit 1
}
printf 'PASS: dirty canonical Task3 producer fails before provenance resolution\n'

run_current_commits synthetic "${ROOT}/isolated-tools" dirty
[[ ${STATUS} -eq 0 ]] || {
    printf 'FAIL: explicit synthetic producer fixture was rejected\n%s\n' "${OUTPUT}" >&2
    exit 1
}
printf 'PASS: explicit synthetic producer fixture remains isolated from canonical status\n'

run_current_commits real "${ROOT}/isolated-tools" clean
[[ ${STATUS} -ne 0 ]] || {
    printf 'FAIL: real Task3 accepted a non-canonical producer\n%s\n' "${OUTPUT}" >&2
    exit 1
}
grep -Fq -- 'canonical' <<<"${OUTPUT}" || {
    printf '%s\n' "${OUTPUT}" >&2
    printf 'FAIL: non-canonical producer error did not identify the root cause\n' >&2
    exit 1
}
printf 'PASS: real Task3 rejects a non-canonical producer path\n'

run_current_commits real "${REPO_ROOT}/megatron-sim-engine" clean
[[ ${STATUS} -eq 0 ]] || {
    printf 'FAIL: clean canonical Task3 producer was rejected\n%s\n' "${OUTPUT}" >&2
    exit 1
}
printf 'PASS: clean canonical producer matching the outer gitlink is accepted\n'

for producer_name in builder scheduler simulator; do
    printf 'external producer fixture\n' >"${ROOT}/isolated-tools/${producer_name}.py"
done
for producer_variable in TASK3_BUILDER TASK3_SCHEDULER TASK3_SIMULATOR; do
    external_builder="${REPO_ROOT}/megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py"
    external_scheduler="${REPO_ROOT}/megatron-sim-engine/src/scheduler/mg_scheduling/mg_test.py"
    external_simulator="${REPO_ROOT}/megatron-sim-engine/simu_main.py"
    case "${producer_variable}" in
        TASK3_BUILDER)
            external_builder="${ROOT}/isolated-tools/builder.py"
            ;;
        TASK3_SCHEDULER)
            external_scheduler="${ROOT}/isolated-tools/scheduler.py"
            ;;
        TASK3_SIMULATOR)
            external_simulator="${ROOT}/isolated-tools/simulator.py"
            ;;
    esac
    run_current_commits real "${REPO_ROOT}/megatron-sim-engine" clean "" \
        "${external_builder}" "${external_scheduler}" "${external_simulator}"
    [[ ${STATUS} -ne 0 ]] || {
        printf 'FAIL: real Task3 accepted external %s\n%s\n' \
            "${producer_variable}" "${OUTPUT}" >&2
        exit 1
    }
    grep -Fq -- "${producer_variable}" <<<"${OUTPUT}" || {
        printf '%s\n' "${OUTPUT}" >&2
        printf 'FAIL: external producer error did not identify %s\n' \
            "${producer_variable}" >&2
        exit 1
    }
    printf 'PASS: real Task3 rejects external %s before producer execution\n' \
        "${producer_variable}"
done

run_current_commits synthetic "${REPO_ROOT}/megatron-sim-engine" clean "" \
    "${ROOT}/isolated-tools/builder.py" \
    "${ROOT}/isolated-tools/scheduler.py" \
    "${ROOT}/isolated-tools/simulator.py"
[[ ${STATUS} -eq 0 ]] || {
    printf 'FAIL: synthetic Task3 rejected explicit producer fixtures\n%s\n' \
        "${OUTPUT}" >&2
    exit 1
}
printf 'PASS: synthetic Task3 retains explicit producer fixture paths\n'

run_current_commits real "${REPO_ROOT}/megatron-sim-engine" clean "" \
    "${REPO_ROOT}/megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py" \
    "${REPO_ROOT}/megatron-sim-engine/src/scheduler/mg_scheduling/mg_test.py" \
    "${REPO_ROOT}/megatron-sim-engine/simu_main.py" \
    0000000000000000000000000000000000000000
[[ ${STATUS} -ne 0 ]] || {
    printf 'FAIL: real Task3 accepted producer bytes outside the pinned commit\n%s\n' \
        "${OUTPUT}" >&2
    exit 1
}
grep -Fq -- 'bytes differ from pinned commit' <<<"${OUTPUT}" || {
    printf '%s\n' "${OUTPUT}" >&2
    printf 'FAIL: producer-byte mismatch did not identify the root cause\n' >&2
    exit 1
}
printf 'PASS: real Task3 producer bytes must match the pinned commit\n'

run_current_commits real "${REPO_ROOT}/megatron-sim-engine" clean \
    0000000000000000000000000000000000000000
[[ ${STATUS} -ne 0 ]] || {
    printf 'FAIL: mismatched Task3 producer commit was accepted\n%s\n' "${OUTPUT}" >&2
    exit 1
}
grep -Fq -- 'differs from outer gitlink' <<<"${OUTPUT}" || {
    printf '%s\n' "${OUTPUT}" >&2
    printf 'FAIL: commit mismatch error did not identify the root cause\n' >&2
    exit 1
}
printf 'PASS: checked-out producer commit must match the outer gitlink\n'

printf 'PROVENANCE_TEST_STATUS=PASS\n'
