#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INSTALLER="$REPO_ROOT/tools/ae/setup_grouped_gemm_v1.sh"
SETUP_ENTRY="$REPO_ROOT/SC26-AE/setup.sh"
TMP_PARENT=${SC26_AE_TMP_ROOT:-${TMPDIR:-/tmp}}
mkdir -p -- "$TMP_PARENT"
EXPECTED_VCS_URL="git+https://github.com/fanshiqing/grouped_gemm@v1.0"
EXPECTED_GROUPED_GEMM_SHA256="c80276f32455f7b216c53bab33a050bd3b699415c70098342d0549235326a26f"
EXPECTED_CUTLASS_SHA256="163146409c12f5cab6fae1218b4a702ab90713c2f363d8170179033d148c704e"
EXPECTED_ABSL_PY_CONSTRAINT="absl-py==2.3.1"

TESTS_RUN=0

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_contains() {
    local file=$1
    local expected=$2
    grep -Fq -- "$expected" "$file" || fail "Expected '$expected' in $file"
}

assert_not_contains() {
    local file=$1
    local unexpected=$2
    if grep -Fq -- "$unexpected" "$file"; then
        fail "Did not expect '$unexpected' in $file"
    fi
}

assert_no_command() {
    local file=$1
    local command_name=$2
    if grep -Eq "^${command_name}([[:space:]]|$)" "$file"; then
        fail "Did not expect command '$command_name' in $file"
    fi
}

assert_exact_file() {
    local file=$1
    local expected=$2
    local actual
    [[ -f $file ]] || fail "Expected file does not exist: $file"
    actual=$(cat "$file")
    [[ $actual == "$expected" ]] || fail "Expected exact content '$expected' in $file, got '$actual'"
    [[ $(wc -l < "$file") -eq 1 ]] || fail "Expected exactly one line in $file"
}

make_fixture() {
    local root=$1
    mkdir -p "$root/bin" "$root/logs" "$root/state" "$root/tmp"

    cat > "$root/bin/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "python $*" >> "$FAKE_CALL_LOG"

if [[ ${1:-} == "-m" && ${2:-} == "pip" && ${3:-} == "--version" ]]; then
    echo "pip 24.0 from /fake/site-packages/pip (python 3.9)"
    exit 0
fi

if [[ ${1:-} == "-m" && ${2:-} == "pip" && ${3:-} == "show" ]]; then
    exit "${FAKE_PACKAGE_PRESENT:-1}"
fi

if [[ ${1:-} == "-m" && ${2:-} == "pip" && ${3:-} == "install" ]]; then
    if [[ " $* " == *" git+https://github.com/fanshiqing/grouped_gemm@v1.0 "* ]]; then
        echo "pip_vcs mode=${GROUPED_GEMM_BUILD_MODE:-unset} arch=${TORCH_CUDA_ARCH_LIST-unset} constraint=${PIP_CONSTRAINT-unset}" >> "$FAKE_CALL_LOG"
        echo "FAKE_PIP_STREAM=vcs"
        exit "${FAKE_VCS_STATUS:-0}"
    fi
    echo "pip_source mode=${GROUPED_GEMM_BUILD_MODE:-unset} arch=${TORCH_CUDA_ARCH_LIST-unset} constraint=${PIP_CONSTRAINT-unset}" >> "$FAKE_CALL_LOG"
    echo "FAKE_PIP_STREAM=source"
    exit "${FAKE_SOURCE_STATUS:-0}"
fi

if [[ ${1:-} == "-c" ]]; then
    code=${2:-}
    case $code in
        *AE_ENVIRONMENT_QUERY*)
            echo "python_version=${FAKE_PYTHON_VERSION:-3.9.18}"
            echo "torch_version=${FAKE_TORCH_VERSION:-2.1.2}"
            echo "torch_cuda_version=${FAKE_TORCH_CUDA_VERSION:-12.1}"
            echo "cuda_available=${FAKE_CUDA_AVAILABLE:-True}"
            ;;
        *AE_IMPORT_QUERY*)
            echo "backend_so=${FAKE_BACKEND_SO}"
            echo "absl_py_version=${FAKE_ABSL_PY_VERSION:-2.3.1}"
            exit "${FAKE_IMPORT_STATUS:-0}"
            ;;
        *AE_VERIFY_QUERY*)
            echo "package_version=${FAKE_PACKAGE_VERSION:-0.0.1}"
            echo "absl_py_version=${FAKE_ABSL_PY_VERSION:-2.3.1}"
            echo "package_dir=/fake/site-packages/grouped_gemm"
            echo "backend_so=${FAKE_BACKEND_SO}"
            echo "direct_url_commit=${FAKE_DIRECT_URL_COMMIT:-7a7f0189797889e926a30b3487512f9539161060}"
            ;;
        *)
            echo "Unknown fake Python code query" >&2
            exit 90
            ;;
    esac
    exit 0
fi

echo "Unexpected fake python invocation: $*" >&2
exit 91
EOF

    cat > "$root/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
echo "nvcc $*" >> "$FAKE_CALL_LOG"
echo "Cuda compilation tools, release 12.1, V12.1.105"
EOF

    cat > "$root/bin/g++" <<'EOF'
#!/usr/bin/env bash
echo "g++ $*" >> "$FAKE_CALL_LOG"
echo "g++ (Ubuntu 11.4.0) 11.4.0"
EOF

    cat > "$root/bin/ninja" <<'EOF'
#!/usr/bin/env bash
echo "ninja $*" >> "$FAKE_CALL_LOG"
echo "1.11.1"
EOF

    cat > "$root/bin/nproc" <<'EOF'
#!/usr/bin/env bash
echo "${FAKE_NPROC:-32}"
EOF

    cat > "$root/bin/tee" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
input=$(cat)
printf '%s\n' "$input" | /usr/bin/tee "$@"
if [[ $input == *"FAKE_PIP_STREAM=source"* && ${FAKE_SOURCE_TEE_STATUS:-0} != 0 ]]; then
    exit "$FAKE_SOURCE_TEE_STATUS"
fi
exit "${FAKE_TEE_STATUS:-0}"
EOF

    cat > "$root/bin/timeout" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "timeout $*" >> "$FAKE_CALL_LOG"
if [[ ${FAKE_TIMEOUT_STATUS:-0} != 0 ]]; then
    exit "$FAKE_TIMEOUT_STATUS"
fi
shift 3
exec "$@"
EOF

    cat > "$root/bin/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "curl $*" >> "$FAKE_CALL_LOG"
output=""
while (($#)); do
    if [[ $1 == "-o" ]]; then
        output=$2
        shift 2
        continue
    fi
    shift
done
[[ -n $output ]] || exit 92
: > "$output"
EOF

    cat > "$root/bin/sha256sum" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "sha256sum \$*" >> "\$FAKE_CALL_LOG"
if [[ \${1:-} == "-c" ]]; then
    input=\$(cat)
    echo "hash_input=\$input" >> "\$FAKE_CALL_LOG"
    if [[ \$input == *"$EXPECTED_GROUPED_GEMM_SHA256"* ]]; then
        status=\${FAKE_GROUPED_GEMM_HASH_STATUS:-\${FAKE_HASH_STATUS:-0}}
    elif [[ \$input == *"$EXPECTED_CUTLASS_SHA256"* ]]; then
        status=\${FAKE_CUTLASS_HASH_STATUS:-\${FAKE_HASH_STATUS:-0}}
    else
        exit 93
    fi
    if [[ \$status != 0 ]]; then
        echo "hash mismatch" >&2
        exit "\$status"
    fi
    if [[ \$input == *"$EXPECTED_GROUPED_GEMM_SHA256"* || \$input == *"$EXPECTED_CUTLASS_SHA256"* ]]; then
        echo "archive: OK"
        exit 0
    fi
fi
echo "\${FAKE_BACKEND_SHA256:-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}  \${1:-unknown}"
EOF

    cat > "$root/bin/tar" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "tar $*" >> "$FAKE_CALL_LOG"
destination=""
while (($#)); do
    if [[ $1 == "-C" ]]; then
        destination=$2
        shift 2
        continue
    fi
    shift
done
[[ -n $destination ]] || exit 94
mkdir -p "$destination"
if [[ $destination == */source ]]; then
    mkdir -p "$destination/third_party/cutlass" "$destination/csrc"
    cat > "$destination/setup.py" <<'PY'
name="grouped_gemm"
PY
    echo '::cutlass::arch::Sm80' > "$destination/csrc/grouped_gemm.cu"
elif [[ $destination == */third_party/cutlass ]]; then
    mkdir -p "$destination/include/cutlass"
    : > "$destination/include/cutlass/cutlass.h"
fi
EOF

    cat > "$root/bin/mktemp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
path="$FAKE_TMP_ROOT/build-$RANDOM"
mkdir -p "$path"
echo "$path"
EOF

    chmod +x "$root/bin/"*
    : > "$root/calls.log"
    : > "$root/backend.so"
}

run_installer() {
    local root=$1
    shift
    env \
        PATH="$root/bin:/usr/bin:/bin" \
        GROUPED_GEMM_SOURCE=vcs \
        GROUPED_GEMM_PYTHON="$root/bin/python" \
        GROUPED_GEMM_LOG_DIR="$root/logs" \
        GROUPED_GEMM_STATE_DIR="$root/state" \
        TMPDIR="$root/tmp" \
        FAKE_TMP_ROOT="$root/tmp" \
        FAKE_CALL_LOG="$root/calls.log" \
        FAKE_BACKEND_SO="$root/backend.so" \
        "$@" \
        bash "$INSTALLER"
}

run_installer_without_source() {
    local root=$1
    shift
    env -u GROUPED_GEMM_SOURCE \
        PATH="$root/bin:/usr/bin:/bin" \
        GROUPED_GEMM_PYTHON="$root/bin/python" \
        GROUPED_GEMM_LOG_DIR="$root/logs" \
        GROUPED_GEMM_STATE_DIR="$root/state" \
        TMPDIR="$root/tmp" \
        FAKE_TMP_ROOT="$root/tmp" \
        FAKE_CALL_LOG="$root/calls.log" \
        FAKE_BACKEND_SO="$root/backend.so" \
        "$@" \
        bash "$INSTALLER"
}

run_setup() {
    local root=$1
    shift
    env \
        PATH="$root/bin:/usr/bin:/bin" \
        GROUPED_GEMM_SOURCE=vcs \
        GROUPED_GEMM_PYTHON="$root/bin/python" \
        GROUPED_GEMM_LOG_DIR="$root/logs" \
        GROUPED_GEMM_STATE_DIR="$root/state" \
        TMPDIR="$root/tmp" \
        FAKE_TMP_ROOT="$root/tmp" \
        FAKE_CALL_LOG="$root/calls.log" \
        FAKE_BACKEND_SO="$root/backend.so" \
        "$@" \
        /usr/bin/bash -c '
            set -euo pipefail
            source "$1"
            # The runtime contract has a dedicated deterministic verifier test.
            # This unit lane isolates setup source forwarding and installer
            # status propagation without discovering a host runtime.
            sc26_ae_setup_verify_runtime_contract() { :; }
            sc26_ae_setup_verify_grouped_gemm() { :; }
            sc26_ae_setup_main
        ' -- "$SETUP_ENTRY"
}

run_setup_without_source() {
    local root=$1
    shift
    env -u GROUPED_GEMM_SOURCE \
        PATH="$root/bin:/usr/bin:/bin" \
        GROUPED_GEMM_PYTHON="$root/bin/python" \
        GROUPED_GEMM_LOG_DIR="$root/logs" \
        GROUPED_GEMM_STATE_DIR="$root/state" \
        TMPDIR="$root/tmp" \
        FAKE_TMP_ROOT="$root/tmp" \
        FAKE_CALL_LOG="$root/calls.log" \
        FAKE_BACKEND_SO="$root/backend.so" \
        "$@" \
        bash "$SETUP_ENTRY"
}

run_case() {
    local name=$1
    shift
    TESTS_RUN=$((TESTS_RUN + 1))
    "$@"
    echo "PASS: $name"
}

test_missing_source_fails_before_pip_or_curl() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-missing-source.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer_without_source "$root" > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Missing GROUPED_GEMM_SOURCE returned success"
    assert_contains "$root/stderr" "GROUPED_GEMM_SOURCE must be set to 'vcs' or 'archive'"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/calls.log" "curl "
}

test_invalid_source_fails_before_pip_or_curl() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-invalid-source.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" GROUPED_GEMM_SOURCE=automatic > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Invalid GROUPED_GEMM_SOURCE returned success"
    assert_contains "$root/stderr" "Unsupported GROUPED_GEMM_SOURCE='automatic'; expected 'vcs' or 'archive'"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/calls.log" "curl "
}

test_explicit_vcs_multiarch_success() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-default.XXXXXX")
    make_fixture "$root"
    run_installer "$root" GROUPED_GEMM_SOURCE=vcs FAKE_VCS_STATUS=0 > "$root/stdout" 2> "$root/stderr"
    assert_contains "$root/calls.log" "pip_vcs mode=multiarch arch=8.0;8.6;8.9;9.0"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/calls.log" "curl "
    assert_contains "$root/state/manifest.env" "ABSL_PY_VERSION=2.3.1"
    assert_contains "$root/state/manifest.env" "SOURCE_METHOD=vcs"
    assert_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=success"
}

test_selected_vcs_failure_does_not_call_archive_tools() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-source.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" GROUPED_GEMM_SOURCE=vcs FAKE_VCS_STATUS=17 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -eq 17 ]] || fail "Expected selected VCS status 17, got $status"
    assert_contains "$root/calls.log" "pip_vcs mode=multiarch arch=8.0;8.6;8.9;9.0"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/calls.log" "curl "
    assert_contains "$root/stdout" "VCS_INSTALL_EXIT_STATUS=17"
    assert_not_contains "$root/stdout" "SOURCE_RECOVERY_USED"
    [[ ! -e $root/state/manifest.env ]] || fail "Selected VCS failure created a manifest"
    assert_not_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=success"
}

test_explicit_archive_multiarch_success() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-archive.XXXXXX")
    make_fixture "$root"
    run_installer "$root" GROUPED_GEMM_SOURCE=archive > "$root/stdout" 2> "$root/stderr"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_no_command "$root/calls.log" timeout
    assert_contains "$root/calls.log" "pip_source mode=multiarch arch=8.0;8.6;8.9;9.0"
    assert_contains "$root/calls.log" "$EXPECTED_GROUPED_GEMM_SHA256"
    assert_contains "$root/calls.log" "$EXPECTED_CUTLASS_SHA256"
    assert_contains "$root/state/manifest.env" "SOURCE_METHOD=archive"
    assert_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=success"
}

test_vcs_install_uses_exact_absl_constraint() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-vcs-constraint.XXXXXX")
    make_fixture "$root"
    run_installer "$root" FAKE_VCS_STATUS=0 > "$root/stdout" 2> "$root/stderr"
    assert_exact_file "$root/state/constraints.txt" "$EXPECTED_ABSL_PY_CONSTRAINT"
    assert_contains "$root/calls.log" "pip_vcs mode=multiarch arch=8.0;8.6;8.9;9.0 constraint=$root/state/constraints.txt"
}

test_archive_install_uses_exact_absl_constraint() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-source-constraint.XXXXXX")
    make_fixture "$root"
    run_installer "$root" GROUPED_GEMM_SOURCE=archive > "$root/stdout" 2> "$root/stderr"
    assert_exact_file "$root/state/constraints.txt" "$EXPECTED_ABSL_PY_CONSTRAINT"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_contains "$root/calls.log" "pip_source mode=multiarch arch=8.0;8.6;8.9;9.0 constraint=$root/state/constraints.txt"
}

test_existing_exact_constraint_is_reused() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-existing-constraint.XXXXXX")
    make_fixture "$root"
    printf '%s\n' "$EXPECTED_ABSL_PY_CONSTRAINT" > "$root/state/constraints.txt"
    run_installer "$root" FAKE_VCS_STATUS=0 > "$root/stdout" 2> "$root/stderr"
    assert_exact_file "$root/state/constraints.txt" "$EXPECTED_ABSL_PY_CONSTRAINT"
    assert_contains "$root/calls.log" "pip_vcs mode=multiarch arch=8.0;8.6;8.9;9.0 constraint=$root/state/constraints.txt"
}

test_existing_constraint_mismatch_fails_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-constraint-mismatch.XXXXXX")
    make_fixture "$root"
    printf '%s\n' 'absl-py==2.2.0' > "$root/state/constraints.txt"
    set +e
    run_installer "$root" > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Existing constraint mismatch returned success"
    assert_contains "$root/stderr" "Existing pip constraint mismatch"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
}

test_existing_constraint_extra_line_fails_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-constraint-extra-line.XXXXXX")
    make_fixture "$root"
    printf '%s\n\n' "$EXPECTED_ABSL_PY_CONSTRAINT" > "$root/state/constraints.txt"
    set +e
    run_installer "$root" > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Constraint with an extra line returned success"
    assert_contains "$root/stderr" "Existing pip constraint must contain exactly one line"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
}

test_existing_constraint_without_trailing_newline_fails_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-constraint-no-newline.XXXXXX")
    make_fixture "$root"
    printf '%s' "$EXPECTED_ABSL_PY_CONSTRAINT" > "$root/state/constraints.txt"
    set +e
    run_installer "$root" > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Constraint without a trailing newline returned success"
    assert_contains "$root/stderr" "Existing pip constraint must contain exactly one line"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
}

test_native_mode_is_explicit() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-native.XXXXXX")
    make_fixture "$root"
    run_installer "$root" GROUPED_GEMM_BUILD_MODE=native FAKE_VCS_STATUS=0 > "$root/stdout" 2> "$root/stderr"
    assert_contains "$root/calls.log" "pip_vcs mode=native arch=unset"
    assert_contains "$root/stdout" "BUILD_MODE=native"
}

test_invalid_mode_fails_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-invalid.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" GROUPED_GEMM_BUILD_MODE=automatic > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Invalid build mode returned success"
    assert_contains "$root/stderr" "Unsupported GROUPED_GEMM_BUILD_MODE"
    assert_not_contains "$root/calls.log" "pip_vcs"
}

test_broken_ninja_prerequisite_fails_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-prereq.XXXXXX")
    make_fixture "$root"
    printf '#!/usr/bin/env bash\nexit 127\n' > "$root/bin/ninja"
    chmod +x "$root/bin/ninja"
    set +e
    run_installer "$root" FAKE_VCS_STATUS=0 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Missing prerequisite returned success"
    assert_contains "$root/stderr" "ninja"
    assert_not_contains "$root/calls.log" "pip_vcs"
}

test_hash_mismatch_stops_source_install() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-hash.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" GROUPED_GEMM_SOURCE=archive FAKE_HASH_STATUS=1 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Hash mismatch returned success"
    assert_contains "$root/stderr" "integrity"
    assert_not_contains "$root/calls.log" "pip_source"
}

test_cutlass_hash_mismatch_stops_source_install() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-cutlass-hash.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" \
        GROUPED_GEMM_SOURCE=archive \
        FAKE_CUTLASS_HASH_STATUS=1 \
        > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "CUTLASS hash mismatch returned success"
    assert_contains "$root/stderr" "CUTLASS source integrity"
    assert_contains "$root/calls.log" "$EXPECTED_GROUPED_GEMM_SHA256"
    assert_contains "$root/calls.log" "$EXPECTED_CUTLASS_SHA256"
    assert_not_contains "$root/calls.log" "pip_source"
}

test_exact_manifest_skips_rebuild() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-idempotent.XXXXXX")
    make_fixture "$root"
    cat > "$root/state/manifest.env" <<'EOF'
GROUPED_GEMM_TAG=v1.0
GROUPED_GEMM_COMMIT=7a7f0189797889e926a30b3487512f9539161060
CUTLASS_COMMIT=8783c41851cd3582490e04e69e0cd756a8c1db7f
BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
PYTHON_VERSION=3.9.18
TORCH_VERSION=2.1.2
TORCH_CUDA_VERSION=12.1
ABSL_PY_VERSION=2.3.1
SOURCE_METHOD=vcs
BACKEND_SO=BACKEND_SO_PLACEHOLDER
BACKEND_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EOF
    sed -i "s|BACKEND_SO_PLACEHOLDER|$root/backend.so|" "$root/state/manifest.env"
    env \
        PATH="$root/bin:/usr/bin:/bin" \
        GROUPED_GEMM_SOURCE=vcs \
        GROUPED_GEMM_PYTHON="$root/bin/python" \
        GROUPED_GEMM_LOG_DIR="$root/logs" \
        GROUPED_GEMM_STATE_DIR="$root/state" \
        TMPDIR="$root/tmp" \
        FAKE_TMP_ROOT="$root/tmp" \
        FAKE_CALL_LOG="$root/calls.log" \
        FAKE_BACKEND_SO="$root/backend.so" \
        bash "$INSTALLER" > "$root/stdout" 2> "$root/stderr"
    assert_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=already_satisfied"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
}

test_archive_manifest_skips_rebuild() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-idempotent-archive.XXXXXX")
    make_fixture "$root"
    cat > "$root/state/manifest.env" <<EOF
GROUPED_GEMM_TAG=v1.0
GROUPED_GEMM_COMMIT=7a7f0189797889e926a30b3487512f9539161060
CUTLASS_COMMIT=8783c41851cd3582490e04e69e0cd756a8c1db7f
BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
PYTHON_VERSION=3.9.18
TORCH_VERSION=2.1.2
TORCH_CUDA_VERSION=12.1
ABSL_PY_VERSION=2.3.1
SOURCE_METHOD=archive
BACKEND_SO=$root/backend.so
BACKEND_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EOF
    run_installer "$root" GROUPED_GEMM_SOURCE=archive > "$root/stdout" 2> "$root/stderr"
    assert_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=already_satisfied"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/calls.log" "curl "
}

test_manifest_source_mismatch_fails_before_install() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-manifest-source.XXXXXX")
    make_fixture "$root"
    cat > "$root/state/manifest.env" <<EOF
GROUPED_GEMM_TAG=v1.0
GROUPED_GEMM_COMMIT=7a7f0189797889e926a30b3487512f9539161060
CUTLASS_COMMIT=8783c41851cd3582490e04e69e0cd756a8c1db7f
BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
PYTHON_VERSION=3.9.18
TORCH_VERSION=2.1.2
TORCH_CUDA_VERSION=12.1
ABSL_PY_VERSION=2.3.1
SOURCE_METHOD=archive
BACKEND_SO=$root/backend.so
BACKEND_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EOF
    set +e
    run_installer "$root" GROUPED_GEMM_SOURCE=vcs > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Manifest source mismatch returned success"
    assert_contains "$root/stderr" "Existing manifest mismatch for SOURCE_METHOD"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/calls.log" "curl "
}

test_manifest_absl_version_mismatch_fails() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-manifest-absl.XXXXXX")
    make_fixture "$root"
    cat > "$root/state/manifest.env" <<EOF
GROUPED_GEMM_TAG=v1.0
GROUPED_GEMM_COMMIT=7a7f0189797889e926a30b3487512f9539161060
CUTLASS_COMMIT=8783c41851cd3582490e04e69e0cd756a8c1db7f
BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
PYTHON_VERSION=3.9.18
TORCH_VERSION=2.1.2
TORCH_CUDA_VERSION=12.1
ABSL_PY_VERSION=2.2.0
SOURCE_METHOD=vcs
BACKEND_SO=$root/backend.so
BACKEND_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EOF
    set +e
    run_installer "$root" > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Manifest absl-py version mismatch returned success"
    assert_contains "$root/stderr" "Existing manifest mismatch for ABSL_PY_VERSION"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=already_satisfied"
}

test_idempotent_live_absl_version_mismatch_fails() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-live-absl.XXXXXX")
    make_fixture "$root"
    cat > "$root/state/manifest.env" <<EOF
GROUPED_GEMM_TAG=v1.0
GROUPED_GEMM_COMMIT=7a7f0189797889e926a30b3487512f9539161060
CUTLASS_COMMIT=8783c41851cd3582490e04e69e0cd756a8c1db7f
BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
PYTHON_VERSION=3.9.18
TORCH_VERSION=2.1.2
TORCH_CUDA_VERSION=12.1
ABSL_PY_VERSION=2.3.1
SOURCE_METHOD=vcs
BACKEND_SO=$root/backend.so
BACKEND_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EOF
    set +e
    run_installer "$root" \
        FAKE_ABSL_PY_VERSION=2.2.0 \
        > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Live absl-py version mismatch returned idempotent success"
    assert_contains "$root/stderr" "Existing absl-py version mismatch: expected '2.3.1', got '2.2.0'"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=already_satisfied"
}

test_manifest_backend_hash_mismatch_fails() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-manifest-hash.XXXXXX")
    make_fixture "$root"
    cat > "$root/state/manifest.env" <<EOF
GROUPED_GEMM_TAG=v1.0
GROUPED_GEMM_COMMIT=7a7f0189797889e926a30b3487512f9539161060
CUTLASS_COMMIT=8783c41851cd3582490e04e69e0cd756a8c1db7f
BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
PYTHON_VERSION=3.9.18
TORCH_VERSION=2.1.2
TORCH_CUDA_VERSION=12.1
ABSL_PY_VERSION=2.3.1
SOURCE_METHOD=vcs
BACKEND_SO=$root/backend.so
BACKEND_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EOF
    set +e
    run_installer "$root" \
        FAKE_BACKEND_SHA256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
        > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Changed backend hash returned success"
    assert_contains "$root/stderr" "BACKEND_SHA256"
    assert_not_contains "$root/calls.log" "pip_vcs"
}

test_max_jobs_above_cpu_limit_fails_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-max-jobs.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" MAX_JOBS=33 FAKE_NPROC=32 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "MAX_JOBS above CPU count returned success"
    assert_contains "$root/stderr" "MAX_JOBS"
    assert_not_contains "$root/calls.log" "pip_vcs"
}

test_wrong_environment_fails_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-wrong-env.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" FAKE_TORCH_VERSION=2.2.0 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Wrong PyTorch version returned success"
    assert_contains "$root/stderr" "Expected PyTorch 2.1.2"
    assert_not_contains "$root/calls.log" "pip_vcs"
}

test_unverified_existing_package_fails() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-existing-package.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" FAKE_PACKAGE_PRESENT=0 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Unverified existing package returned success"
    assert_contains "$root/stderr" "already installed without the AE manifest"
    assert_not_contains "$root/calls.log" "pip_vcs"
}

test_matching_manifest_with_broken_import_fails() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-broken-import.XXXXXX")
    make_fixture "$root"
    cat > "$root/state/manifest.env" <<EOF
GROUPED_GEMM_TAG=v1.0
GROUPED_GEMM_COMMIT=7a7f0189797889e926a30b3487512f9539161060
CUTLASS_COMMIT=8783c41851cd3582490e04e69e0cd756a8c1db7f
BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
PYTHON_VERSION=3.9.18
TORCH_VERSION=2.1.2
TORCH_CUDA_VERSION=12.1
ABSL_PY_VERSION=2.3.1
SOURCE_METHOD=vcs
BACKEND_SO=$root/backend.so
BACKEND_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
EOF
    set +e
    run_installer "$root" FAKE_IMPORT_STATUS=19 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Broken manifest import returned success"
    assert_contains "$root/stderr" "package import is broken"
    assert_not_contains "$root/calls.log" "pip_vcs"
}

test_selected_archive_install_failure_propagates() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-source-failure.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" GROUPED_GEMM_SOURCE=archive FAKE_SOURCE_STATUS=23 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -eq 23 ]] || fail "Expected source install status 23, got $status"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_contains "$root/calls.log" "pip_source mode=multiarch"
    assert_not_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=success"
}

test_selected_archive_log_write_failure_fails_explicitly() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-source-log-failure.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" \
        GROUPED_GEMM_SOURCE=archive \
        FAKE_SOURCE_TEE_STATUS=74 \
        > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Source log write failure returned success"
    assert_contains "$root/stderr" "Failed to persist the source installation log"
    assert_not_contains "$root/calls.log" "pip_vcs"
    [[ ! -e $root/state/manifest.env ]] || fail "Source log write failure created a manifest"
    assert_not_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=success"
}

test_vcs_wrong_commit_fails_verification() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-wrong-commit.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" \
        FAKE_VCS_STATUS=0 \
        FAKE_DIRECT_URL_COMMIT=0000000000000000000000000000000000000000 \
        > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Wrong VCS commit returned success"
    assert_contains "$root/stderr" "VCS install resolved unexpected commit"
}

test_post_install_absl_version_mismatch_fails() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-absl-version.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" \
        FAKE_VCS_STATUS=0 \
        FAKE_ABSL_PY_VERSION=2.4.0 \
        > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Unexpected absl-py version returned success"
    assert_contains "$root/stderr" "Expected absl-py 2.3.1"
    [[ ! -e $root/state/manifest.env ]] || fail "Unexpected absl-py version created a manifest"
    assert_not_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=success"
}

test_vcs_log_write_failure_fails() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-log-failure.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" FAKE_VCS_STATUS=0 FAKE_TEE_STATUS=74 > "$root/stdout" 2> "$root/stderr"
    local status=$?
    set -e
    [[ $status -ne 0 ]] || fail "VCS log write failure returned success"
    [[ ! -e $root/state/manifest.env ]] || fail "VCS log write failure created a manifest"
}

test_python_environment_bin_is_added_to_path() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-env-path.XXXXXX")
    make_fixture "$root"
    mkdir -p "$root/env/bin" "$root/toolbin"
    cp "$root/bin/python" "$root/env/bin/python"
    cp "$root/bin/ninja" "$root/env/bin/ninja"
    local tool
    for tool in curl g++ mktemp nproc nvcc sha256sum tar tee timeout; do
        cp "$root/bin/$tool" "$root/toolbin/$tool"
    done
    set +e
    env \
        PATH="$root/toolbin:/usr/bin:/bin" \
        GROUPED_GEMM_SOURCE=vcs \
        GROUPED_GEMM_PYTHON="$root/env/bin/python" \
        GROUPED_GEMM_LOG_DIR="$root/logs" \
        GROUPED_GEMM_STATE_DIR="$root/state" \
        TMPDIR="$root/tmp" \
        FAKE_TMP_ROOT="$root/tmp" \
        FAKE_CALL_LOG="$root/calls.log" \
        FAKE_BACKEND_SO="$root/backend.so" \
        FAKE_VCS_STATUS=0 \
        bash "$INSTALLER" > "$root/stdout" 2> "$root/stderr"
    local status=$?
    set -e
    [[ $status -eq 0 ]] || fail "Python environment bin was not added to PATH"
    assert_contains "$root/calls.log" "pip_vcs mode=multiarch arch=8.0;8.6;8.9;9.0"
}

test_selected_vcs_timeout_is_final() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-vcs-timeout.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" GROUPED_GEMM_SOURCE=vcs FAKE_TIMEOUT_STATUS=124 > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -eq 124 ]] || fail "Expected selected VCS timeout status 124, got $status"
    assert_contains "$root/calls.log" "timeout --signal=TERM --kill-after=30s 600s"
    assert_contains "$root/stdout" "VCS_INSTALL_EXIT_STATUS=124"
    assert_not_contains "$root/stdout" "SOURCE_RECOVERY_USED"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/calls.log" "curl "
    [[ ! -e $root/state/manifest.env ]] || fail "Selected VCS timeout created a manifest"
}

test_invalid_vcs_timeout_fails_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-invalid-timeout.XXXXXX")
    make_fixture "$root"
    set +e
    run_installer "$root" \
        GROUPED_GEMM_VCS_TIMEOUT_SECONDS=invalid \
        > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Invalid VCS timeout returned success"
    assert_contains "$root/stderr" "GROUPED_GEMM_VCS_TIMEOUT_SECONDS must be a positive integer"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
}

test_log_directory_creation_failure_stops_before_pip() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-log-dir.XXXXXX")
    make_fixture "$root"
    : > "$root/not-a-directory"
    set +e
    run_installer "$root" \
        GROUPED_GEMM_LOG_DIR="$root/not-a-directory/logs" \
        > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "Log-directory creation failure returned success"
    assert_contains "$root/stderr" "not-a-directory"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
}

test_sc26_setup_requires_explicit_source() {
    local root status
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-setup-missing-source.XXXXXX")
    make_fixture "$root"
    set +e
    run_setup_without_source "$root" > "$root/stdout" 2> "$root/stderr"
    status=$?
    set -e
    [[ $status -ne 0 ]] || fail "SC26-AE/setup.sh accepted a missing source"
    assert_contains "$root/stderr" "GROUPED_GEMM_SOURCE must be set to 'vcs' or 'archive'"
    assert_not_contains "$root/calls.log" "pip_vcs"
    assert_not_contains "$root/calls.log" "pip_source"
    assert_not_contains "$root/calls.log" "curl "
}

test_sc26_setup_forwards_selected_archive() {
    local root
    root=$(mktemp -d "${TMP_PARENT%/}/grouped-gemm-unit-setup-archive.XXXXXX")
    make_fixture "$root"
    # The setup wrapper owns fixed runtime verification; this unit case keeps
    # that verifier stubbed and models only the installer process boundary.
    # The installer-specific behavior is covered by the cases above.
    cat > "$root/bin/bash" <<'EOF'
#!/usr/bin/bash
set -euo pipefail
printf 'setup_installer source=%s python=%s args=%s\n' \
    "${GROUPED_GEMM_SOURCE-}" "${GROUPED_GEMM_PYTHON-}" "$*" >> "$FAKE_CALL_LOG"
printf 'GROUPED_GEMM_INSTALL_STATUS=success\n'
EOF
    chmod +x "$root/bin/bash"
    run_setup "$root" GROUPED_GEMM_SOURCE=archive > "$root/stdout" 2> "$root/stderr"
    assert_contains "$root/calls.log" \
        "setup_installer source=archive python=/opt/conda/envs/megatron_env/bin/python"
    assert_not_contains "$root/calls.log" "source=vcs"
    assert_contains "$root/stdout" "GROUPED_GEMM_INSTALL_STATUS=success"
}

run_case "missing source fails before pip or curl" test_missing_source_fails_before_pip_or_curl
run_case "invalid source fails before pip or curl" test_invalid_source_fails_before_pip_or_curl
run_case "explicit VCS multiarch succeeds" test_explicit_vcs_multiarch_success
run_case "selected VCS failure does not call archive tools" test_selected_vcs_failure_does_not_call_archive_tools
run_case "explicit archive multiarch succeeds" test_explicit_archive_multiarch_success
run_case "VCS install uses exact absl-py constraint" test_vcs_install_uses_exact_absl_constraint
run_case "archive install uses exact absl-py constraint" test_archive_install_uses_exact_absl_constraint
run_case "existing exact absl-py constraint is reused" test_existing_exact_constraint_is_reused
run_case "existing absl-py constraint mismatch fails before pip" test_existing_constraint_mismatch_fails_before_pip
run_case "existing absl-py constraint extra line fails before pip" test_existing_constraint_extra_line_fails_before_pip
run_case "existing absl-py constraint without trailing newline fails before pip" test_existing_constraint_without_trailing_newline_fails_before_pip
run_case "native mode is explicit" test_native_mode_is_explicit
run_case "invalid mode fails before pip" test_invalid_mode_fails_before_pip
run_case "broken Ninja prerequisite fails before pip" test_broken_ninja_prerequisite_fails_before_pip
run_case "hash mismatch stops source install" test_hash_mismatch_stops_source_install
run_case "CUTLASS hash mismatch stops source install" test_cutlass_hash_mismatch_stops_source_install
run_case "exact VCS manifest skips rebuild" test_exact_manifest_skips_rebuild
run_case "exact archive manifest skips rebuild" test_archive_manifest_skips_rebuild
run_case "manifest source mismatch fails before install" test_manifest_source_mismatch_fails_before_install
run_case "manifest absl-py version mismatch fails" test_manifest_absl_version_mismatch_fails
run_case "idempotent live absl-py version mismatch fails" test_idempotent_live_absl_version_mismatch_fails
run_case "manifest backend hash mismatch fails" test_manifest_backend_hash_mismatch_fails
run_case "MAX_JOBS above CPU count fails before pip" test_max_jobs_above_cpu_limit_fails_before_pip
run_case "wrong environment fails before pip" test_wrong_environment_fails_before_pip
run_case "unverified existing package fails" test_unverified_existing_package_fails
run_case "matching manifest with broken import fails" test_matching_manifest_with_broken_import_fails
run_case "selected archive install failure propagates" test_selected_archive_install_failure_propagates
run_case "selected archive log write failure fails explicitly" test_selected_archive_log_write_failure_fails_explicitly
run_case "VCS wrong commit fails verification" test_vcs_wrong_commit_fails_verification
run_case "post-install absl-py version mismatch fails" test_post_install_absl_version_mismatch_fails
run_case "VCS log write failure fails" test_vcs_log_write_failure_fails
run_case "Python environment bin is added to PATH" test_python_environment_bin_is_added_to_path
run_case "selected VCS timeout is final" test_selected_vcs_timeout_is_final
run_case "invalid VCS timeout fails before pip" test_invalid_vcs_timeout_fails_before_pip
run_case "log directory creation failure stops before pip" test_log_directory_creation_failure_stops_before_pip
run_case "SC26-AE setup requires explicit source" test_sc26_setup_requires_explicit_source
run_case "SC26-AE setup forwards selected archive" test_sc26_setup_forwards_selected_archive

echo "PASS: $TESTS_RUN/$TESTS_RUN grouped_gemm setup unit cases."
