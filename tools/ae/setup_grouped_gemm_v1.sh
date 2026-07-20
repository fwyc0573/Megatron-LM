#!/usr/bin/env bash

set -euo pipefail

readonly GROUPED_GEMM_TAG="v1.0"
readonly GROUPED_GEMM_COMMIT="7a7f0189797889e926a30b3487512f9539161060"
readonly GROUPED_GEMM_VCS_URL="git+https://github.com/fanshiqing/grouped_gemm@v1.0"
readonly GROUPED_GEMM_ARCHIVE_URL="https://codeload.github.com/fanshiqing/grouped_gemm/tar.gz/refs/tags/v1.0"
readonly GROUPED_GEMM_ARCHIVE_SHA256="c80276f32455f7b216c53bab33a050bd3b699415c70098342d0549235326a26f"
readonly CUTLASS_COMMIT="8783c41851cd3582490e04e69e0cd756a8c1db7f"
readonly CUTLASS_ARCHIVE_URL="https://codeload.github.com/NVIDIA/cutlass/tar.gz/8783c41851cd3582490e04e69e0cd756a8c1db7f"
readonly CUTLASS_ARCHIVE_SHA256="163146409c12f5cab6fae1218b4a702ab90713c2f363d8170179033d148c704e"
readonly MULTIARCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
readonly EXPECTED_PYTHON_VERSION="3.9.18"
readonly EXPECTED_TORCH_VERSION="2.1.2"
readonly EXPECTED_TORCH_CUDA_VERSION="12.1"
readonly EXPECTED_ABSL_PY_VERSION="2.3.1"

BUILD_MODE=${GROUPED_GEMM_BUILD_MODE:-multiarch}
SOURCE=${GROUPED_GEMM_SOURCE:-}
PYTHON_BIN=${GROUPED_GEMM_PYTHON:-/opt/conda/envs/megatron_env/bin/python}
LOG_DIR=${GROUPED_GEMM_LOG_DIR:-/tmp/grouped_gemm_v1_ae_logs}
STATE_DIR=${GROUPED_GEMM_STATE_DIR:-/opt/conda/envs/megatron_env/share/grouped_gemm_ae}
MAX_JOBS=${MAX_JOBS:-8}
VCS_TIMEOUT_SECONDS=${GROUPED_GEMM_VCS_TIMEOUT_SECONDS:-600}
MANIFEST_PATH="$STATE_DIR/manifest.env"
CONSTRAINTS_PATH="$STATE_DIR/constraints.txt"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    local command_name=$1
    command -v "$command_name" >/dev/null 2>&1 || fail "Required command is missing: $command_name"
}

require_manifest_value() {
    local key=$1
    local expected=$2
    local actual
    actual=$(sed -n "s/^${key}=//p" "$MANIFEST_PATH")
    [[ $actual == "$expected" ]] || fail "Existing manifest mismatch for $key: expected '$expected', got '$actual'"
}

case $SOURCE in
    vcs|archive)
        ;;
    "")
        fail "GROUPED_GEMM_SOURCE must be set to 'vcs' or 'archive'"
        ;;
    *)
        fail "Unsupported GROUPED_GEMM_SOURCE='$SOURCE'; expected 'vcs' or 'archive'"
        ;;
esac
export GROUPED_GEMM_SOURCE="$SOURCE"

case $BUILD_MODE in
    multiarch)
        export TORCH_CUDA_ARCH_LIST="$MULTIARCH_CUDA_ARCH_LIST"
        ;;
    native)
        unset TORCH_CUDA_ARCH_LIST
        ;;
    *)
        fail "Unsupported GROUPED_GEMM_BUILD_MODE='$BUILD_MODE'; expected 'multiarch' or 'native'"
        ;;
esac
export GROUPED_GEMM_BUILD_MODE="$BUILD_MODE"

[[ $MAX_JOBS =~ ^[1-9][0-9]*$ ]] || fail "MAX_JOBS must be a positive integer, got '$MAX_JOBS'"
[[ $VCS_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] || fail "GROUPED_GEMM_VCS_TIMEOUT_SECONDS must be a positive integer, got '$VCS_TIMEOUT_SECONDS'"
export MAX_JOBS

mkdir -p "$LOG_DIR" "$STATE_DIR"
LOG_FILE="$LOG_DIR/setup_grouped_gemm_v1_$(date +%Y%m%d_%H%M%S).log"

echo "GROUPED_GEMM_TAG=$GROUPED_GEMM_TAG"
echo "GROUPED_GEMM_COMMIT=$GROUPED_GEMM_COMMIT"
echo "CUTLASS_COMMIT=$CUTLASS_COMMIT"
echo "GROUPED_GEMM_SOURCE=$SOURCE"
echo "BUILD_MODE=$BUILD_MODE"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST-unset}"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "MAX_JOBS=$MAX_JOBS"
echo "VCS_TIMEOUT_SECONDS=$VCS_TIMEOUT_SECONDS"
echo "LOG_FILE=$LOG_FILE"

[[ -x $PYTHON_BIN ]] || fail "Python executable is missing or not executable: $PYTHON_BIN"
PYTHON_BIN_DIR=$(cd "$(dirname "$PYTHON_BIN")" && pwd)
export PATH="$PYTHON_BIN_DIR:$PATH"
echo "PYTHON_BIN_DIR=$PYTHON_BIN_DIR"
for command_name in sha256sum nvcc g++ ninja nproc; do
    require_command "$command_name"
done
case $SOURCE in
    vcs)
        require_command timeout
        ;;
    archive)
        for command_name in curl tar mktemp; do
            require_command "$command_name"
        done
        ;;
esac

cpu_count=$(nproc)
[[ $cpu_count =~ ^[1-9][0-9]*$ ]] || fail "nproc returned an invalid CPU count: '$cpu_count'"
((MAX_JOBS <= cpu_count)) || fail "MAX_JOBS=$MAX_JOBS exceeds available CPU count $cpu_count"
echo "CPU_COUNT=$cpu_count"

environment_output=$(
    "$PYTHON_BIN" -c '
# AE_ENVIRONMENT_QUERY
import sys
import torch

print(f"python_version={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print(f"torch_version={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
'
)
echo "$environment_output"

python_version=$(sed -n 's/^python_version=//p' <<< "$environment_output")
torch_version=$(sed -n 's/^torch_version=//p' <<< "$environment_output")
torch_cuda_version=$(sed -n 's/^torch_cuda_version=//p' <<< "$environment_output")
cuda_available=$(sed -n 's/^cuda_available=//p' <<< "$environment_output")

[[ $python_version == "$EXPECTED_PYTHON_VERSION" ]] || fail "Expected Python $EXPECTED_PYTHON_VERSION, got '$python_version'"
[[ $torch_version == "$EXPECTED_TORCH_VERSION" ]] || fail "Expected PyTorch $EXPECTED_TORCH_VERSION, got '$torch_version'"
[[ $torch_cuda_version == "$EXPECTED_TORCH_CUDA_VERSION" ]] || fail "Expected PyTorch CUDA $EXPECTED_TORCH_CUDA_VERSION, got '$torch_cuda_version'"
[[ $cuda_available == "True" ]] || fail "PyTorch reports CUDA unavailable"

pip_version_output=$("$PYTHON_BIN" -m pip --version)
echo "$pip_version_output"
pip_major=$(sed -E 's/^pip ([0-9]+).*/\1/' <<< "$pip_version_output")
[[ $pip_major =~ ^[0-9]+$ ]] || fail "Unable to parse pip version: $pip_version_output"
((pip_major >= 21)) || fail "pip >=21 is required, got: $pip_version_output"

if ! nvcc_version_output=$(nvcc --version); then
    fail "nvcc --version failed"
fi
if ! gxx_version_output=$(g++ --version | head -n 1); then
    fail "g++ --version failed"
fi
if ! ninja_version_output=$(ninja --version); then
    fail "ninja --version failed"
fi
echo "$nvcc_version_output"
echo "$gxx_version_output"
echo "ninja_version=$ninja_version_output"
grep -Fq "release 12.1" <<< "$nvcc_version_output" || fail "nvcc must report CUDA 12.1"
[[ -n $gxx_version_output ]] || fail "g++ did not report a version"
[[ -n $ninja_version_output ]] || fail "ninja did not report a version"

if [[ -f $CONSTRAINTS_PATH ]]; then
    constraints_content=$(cat "$CONSTRAINTS_PATH")
    [[ $constraints_content == "absl-py==$EXPECTED_ABSL_PY_VERSION" ]] || fail "Existing pip constraint mismatch: expected 'absl-py==$EXPECTED_ABSL_PY_VERSION', got '$constraints_content'"
    [[ $(wc -l < "$CONSTRAINTS_PATH") -eq 1 ]] || fail "Existing pip constraint must contain exactly one line: $CONSTRAINTS_PATH"
else
    printf 'absl-py==%s\n' "$EXPECTED_ABSL_PY_VERSION" > "$CONSTRAINTS_PATH"
fi
export PIP_CONSTRAINT="$CONSTRAINTS_PATH"
echo "PIP_CONSTRAINT=$PIP_CONSTRAINT"

if [[ -f $MANIFEST_PATH ]]; then
    require_manifest_value GROUPED_GEMM_TAG "$GROUPED_GEMM_TAG"
    require_manifest_value GROUPED_GEMM_COMMIT "$GROUPED_GEMM_COMMIT"
    require_manifest_value CUTLASS_COMMIT "$CUTLASS_COMMIT"
    require_manifest_value BUILD_MODE "$BUILD_MODE"
    require_manifest_value TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST-unset}"
    require_manifest_value PYTHON_VERSION "$python_version"
    require_manifest_value TORCH_VERSION "$torch_version"
    require_manifest_value TORCH_CUDA_VERSION "$torch_cuda_version"
    require_manifest_value ABSL_PY_VERSION "$EXPECTED_ABSL_PY_VERSION"
    require_manifest_value SOURCE_METHOD "$SOURCE"
    import_output=$("$PYTHON_BIN" -c '
# AE_IMPORT_QUERY
import importlib.metadata
import pathlib
import grouped_gemm
import grouped_gemm_backend

assert grouped_gemm.ops is not None
assert grouped_gemm_backend.__file__
print(f"backend_so={pathlib.Path(grouped_gemm_backend.__file__).resolve()}")
print(f"absl_py_version={importlib.metadata.version('absl-py')}")
') || fail "Existing grouped_gemm manifest matches but the package import is broken"
    echo "$import_output"
    installed_backend_so=$(sed -n 's/^backend_so=//p' <<< "$import_output")
    installed_absl_py_version=$(sed -n 's/^absl_py_version=//p' <<< "$import_output")
    [[ $installed_absl_py_version == "$EXPECTED_ABSL_PY_VERSION" ]] || fail "Existing absl-py version mismatch: expected '$EXPECTED_ABSL_PY_VERSION', got '$installed_absl_py_version'"
    [[ -n $installed_backend_so ]] || fail "Existing grouped_gemm import did not report its backend shared library"
    require_manifest_value BACKEND_SO "$installed_backend_so"
    [[ -f $installed_backend_so ]] || fail "Manifest backend shared library is missing: $installed_backend_so"
    installed_backend_sha256=$(sha256sum "$installed_backend_so" | awk '{print $1}')
    require_manifest_value BACKEND_SHA256 "$installed_backend_sha256"
    echo "GROUPED_GEMM_INSTALL_STATUS=already_satisfied"
    exit 0
fi

if "$PYTHON_BIN" -m pip show grouped_gemm >/dev/null 2>&1; then
    fail "grouped_gemm is already installed without the AE manifest; refusing to overwrite an unverified package"
fi

source_method=$SOURCE
case $SOURCE in
    vcs)
        echo "VCS_INSTALL_COMMAND=$PYTHON_BIN -m pip install $GROUPED_GEMM_VCS_URL"
        set +e
        PIP_NO_BUILD_ISOLATION=1 PIP_NO_CACHE_DIR=1 \
            timeout --signal=TERM --kill-after=30s "${VCS_TIMEOUT_SECONDS}s" \
            "$PYTHON_BIN" -m pip install "$GROUPED_GEMM_VCS_URL" 2>&1 | tee -a "$LOG_FILE"
        vcs_pipeline_status=("${PIPESTATUS[@]}")
        vcs_status=${vcs_pipeline_status[0]}
        vcs_tee_status=${vcs_pipeline_status[1]}
        set -e
        echo "VCS_INSTALL_EXIT_STATUS=$vcs_status"
        echo "VCS_LOG_EXIT_STATUS=$vcs_tee_status"
        ((vcs_tee_status == 0)) || fail "Failed to persist the VCS installation log: $LOG_FILE"
        ((vcs_status == 0)) || exit "$vcs_status"
        ;;
    archive)
        build_root=$(mktemp -d "${TMPDIR:-/tmp}/grouped_gemm_v1_build.XXXXXX")
        source_dir="$build_root/source"
        grouped_gemm_archive="$build_root/grouped_gemm-v1.0.tar.gz"
        cutlass_archive="$build_root/cutlass-$CUTLASS_COMMIT.tar.gz"
        mkdir -p "$source_dir/third_party/cutlass"

        curl -L --fail --connect-timeout 10 --max-time 300 --retry 2 --retry-delay 2 \
            "$GROUPED_GEMM_ARCHIVE_URL" -o "$grouped_gemm_archive"
        if ! printf '%s  %s\n' "$GROUPED_GEMM_ARCHIVE_SHA256" "$grouped_gemm_archive" | sha256sum -c -; then
            fail "grouped_gemm source integrity verification failed"
        fi
        tar -xzf "$grouped_gemm_archive" --strip-components=1 -C "$source_dir"

        curl -L --fail --connect-timeout 10 --max-time 300 --retry 2 --retry-delay 2 \
            "$CUTLASS_ARCHIVE_URL" -o "$cutlass_archive"
        if ! printf '%s  %s\n' "$CUTLASS_ARCHIVE_SHA256" "$cutlass_archive" | sha256sum -c -; then
            fail "CUTLASS source integrity verification failed"
        fi
        tar -xzf "$cutlass_archive" --strip-components=1 -C "$source_dir/third_party/cutlass"

        [[ -f $source_dir/setup.py ]] || fail "Exact grouped_gemm source is missing setup.py"
        [[ -f $source_dir/third_party/cutlass/include/cutlass/cutlass.h ]] || fail "Pinned CUTLASS source is incomplete"
        grep -Fq 'name="grouped_gemm"' "$source_dir/setup.py" || fail "Unexpected grouped_gemm setup.py content"
        grep -Fq '::cutlass::arch::Sm80' "$source_dir/csrc/grouped_gemm.cu" || fail "Unexpected grouped_gemm CUDA source content"

        echo "SOURCE_INSTALL_COMMAND=$PYTHON_BIN -m pip install --no-build-isolation --no-cache-dir $source_dir"
        set +e
        "$PYTHON_BIN" -m pip install --no-build-isolation --no-cache-dir "$source_dir" 2>&1 | tee -a "$LOG_FILE"
        source_pipeline_status=("${PIPESTATUS[@]}")
        source_install_status=${source_pipeline_status[0]}
        source_tee_status=${source_pipeline_status[1]}
        set -e
        echo "SOURCE_INSTALL_EXIT_STATUS=$source_install_status"
        echo "SOURCE_LOG_EXIT_STATUS=$source_tee_status"
        ((source_tee_status == 0)) || fail "Failed to persist the source installation log: $LOG_FILE"
        ((source_install_status == 0)) || exit "$source_install_status"
        ;;
esac

verification_output=$(
    "$PYTHON_BIN" -c '
# AE_VERIFY_QUERY
import importlib.metadata
import json
import pathlib
import grouped_gemm
import grouped_gemm_backend

distribution = importlib.metadata.distribution("grouped-gemm")
direct_url_text = distribution.read_text("direct_url.json")
direct_url_commit = ""
if direct_url_text:
    direct_url = json.loads(direct_url_text)
    direct_url_commit = direct_url.get("vcs_info", {}).get("commit_id", "")

print(f"package_version={distribution.version}")
print(f"absl_py_version={importlib.metadata.version('absl-py')}")
print(f"package_dir={pathlib.Path(grouped_gemm.__file__).resolve().parent}")
print(f"backend_so={pathlib.Path(grouped_gemm_backend.__file__).resolve()}")
print(f"direct_url_commit={direct_url_commit}")
'
)
echo "$verification_output"

package_version=$(sed -n 's/^package_version=//p' <<< "$verification_output")
absl_py_version=$(sed -n 's/^absl_py_version=//p' <<< "$verification_output")
package_dir=$(sed -n 's/^package_dir=//p' <<< "$verification_output")
backend_so=$(sed -n 's/^backend_so=//p' <<< "$verification_output")
direct_url_commit=$(sed -n 's/^direct_url_commit=//p' <<< "$verification_output")

[[ $package_version == "0.0.1" ]] || fail "Expected grouped_gemm package version 0.0.1, got '$package_version'"
[[ $absl_py_version == "$EXPECTED_ABSL_PY_VERSION" ]] || fail "Expected absl-py $EXPECTED_ABSL_PY_VERSION, got '$absl_py_version'"
[[ -n $package_dir ]] || fail "Unable to locate grouped_gemm package directory"
[[ -n $backend_so ]] || fail "Unable to locate grouped_gemm backend shared library"
if [[ $source_method == vcs && $direct_url_commit != "$GROUPED_GEMM_COMMIT" ]]; then
    fail "VCS install resolved unexpected commit '$direct_url_commit'"
fi

backend_sha256=$(sha256sum "$backend_so" | awk '{print $1}')
installed_at=$(date --iso-8601=seconds)
cat > "$MANIFEST_PATH" <<EOF
GROUPED_GEMM_TAG=$GROUPED_GEMM_TAG
GROUPED_GEMM_COMMIT=$GROUPED_GEMM_COMMIT
CUTLASS_COMMIT=$CUTLASS_COMMIT
BUILD_MODE=$BUILD_MODE
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST-unset}
PYTHON_VERSION=$python_version
TORCH_VERSION=$torch_version
TORCH_CUDA_VERSION=$torch_cuda_version
ABSL_PY_VERSION=$absl_py_version
NVCC_VERSION=$(tr '\n' ' ' <<< "$nvcc_version_output")
GXX_VERSION=$gxx_version_output
NINJA_VERSION=$ninja_version_output
SOURCE_METHOD=$source_method
PACKAGE_VERSION=$package_version
PACKAGE_DIR=$package_dir
BACKEND_SO=$backend_so
BACKEND_SHA256=$backend_sha256
INSTALLED_AT=$installed_at
EOF

echo "BACKEND_SO=$backend_so"
echo "BACKEND_SHA256=$backend_sha256"
echo "MANIFEST_PATH=$MANIFEST_PATH"
echo "GROUPED_GEMM_INSTALL_STATUS=success"
