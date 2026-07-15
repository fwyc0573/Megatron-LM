## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-13 | Made the one-command contract explicit: the script now pins, installs, verifies, and records absl-py 2.3.1 together with exact grouped_gemm and CUTLASS acquisition. |
| 2026-07-13 | Added the validated immutable-image setup, dependency inventory, architecture evidence, and failure policy for grouped_gemm v1.0. |

# AE Supplemental Setup: grouped_gemm v1.0

## 1. Scope

The AE base image is immutable and must not be rebuilt or republished:

```text
Image:  hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef
Digest: sha256:11c794efbfe166932089a7cb9fb1e68fea6fa0101d4265ab51337437184a6c9e
```

The image does not contain `grouped_gemm` or its required `absl-py` dependency. Run the supplemental installer after every new container is started. The one script automatically acquires exact grouped_gemm source, its exact CUTLASS build source, and `absl-py==2.3.1`. It fails fast on an unexpected Python, PyTorch, CUDA, compiler, source hash, package, dependency version, constraint, or manifest instead of changing to another package version or backend.

## 2. Required Command

From the repository root inside the started container, run:

```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
bash tools/ae/setup_grouped_gemm_v1.sh
```

This is the complete supplemental setup command. AE personnel do **not** need to run a separate `pip install absl-py`, `git clone CUTLASS`, `curl`, or source-build command. The script performs the following ordered actions:

1. Creates and verifies an exact pip constraint containing only `absl-py==2.3.1`.
2. Attempts the required grouped_gemm `v1.0` VCS install under that constraint.
3. If VCS/submodule transport fails, downloads and hash-verifies the exact grouped_gemm and CUTLASS archives, reconstructs `third_party/cutlass`, and builds the same selected architecture mode under the same constraint.
4. Verifies grouped_gemm distribution version `0.0.1`, `absl-py` distribution version `2.3.1`, backend import/path, and source identity before writing the manifest or reporting success.
5. On later invocations, verifies the exact constraint, manifest, live `absl-py` version, backend path, and backend hash before reporting `already_satisfied`.

The default build is the AE release configuration:

```text
GROUPED_GEMM_BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
MAX_JOBS=8
GROUPED_GEMM_VCS_TIMEOUT_SECONDS=600
PIP_CONSTRAINT=/opt/conda/envs/megatron_env/share/grouped_gemm_ae/constraints.txt
```

`PIP_CONSTRAINT` is created and exported by the script; AE personnel must not create or override it manually. Its exact content is:

```text
absl-py==2.3.1
```

The installer uses the project runtime explicitly:

```text
/opt/conda/envs/megatron_env/bin/python
```

Do not invoke it with `/opt/conda/bin/python` or an unverified environment.

## 3. Supplemental Dependency Inventory

### 3.1 Downloaded after container startup

| Item | Exact identity | Purpose | Live validation result |
|------|----------------|---------|------------------------|
| `grouped_gemm` | upstream tag `v1.0`; commit `7a7f0189797889e926a30b3487512f9539161060`; installed distribution version `0.0.1` | Megatron-Core GroupedMLP CUDA extension | Installed and imported on H800 |
| grouped_gemm source archive | `https://codeload.github.com/fanshiqing/grouped_gemm/tar.gz/refs/tags/v1.0`; `15174` bytes; SHA-256 `c80276f32455f7b216c53bab33a050bd3b699415c70098342d0549235326a26f` | Exact-tag source route when the required VCS install fails | Hash verified before extraction |
| CUTLASS source archive | commit `8783c41851cd3582490e04e69e0cd756a8c1db7f`; `https://codeload.github.com/NVIDIA/cutlass/tar.gz/8783c41851cd3582490e04e69e0cd756a8c1db7f`; `20782247` bytes; SHA-256 `163146409c12f5cab6fae1218b4a702ab90713c2f363d8170179033d148c704e` | Reconstructs the exact `third_party/cutlass` gitlink required by the tagged source | Hash verified before extraction; source-only build dependency, not an installed Python distribution |
| `absl-py` | `2.3.1` | Runtime dependency declared by grouped_gemm | Explicitly pinned by the script's pip constraint; installed by the same grouped_gemm pip operation; verified through package metadata and recorded in the AE manifest |

The successful live build did not download another PyTorch, NumPy, CUDA toolkit, compiler, or Ninja package.

### 3.2 Already present in the base image

| Item | Observed version | AE note |
|------|------------------|---------|
| Python | `3.9.18` | Must be `/opt/conda/envs/megatron_env/bin/python` |
| PyTorch | `2.1.2` | Existing package; CUDA reports `12.1` |
| NumPy | `1.26.4` | Existing package; grouped_gemm requirement was already satisfied |
| CUDA toolkit / nvcc | `12.1` / `V12.1.105` | Used to compile the extension |
| `g++` | `11.4.0` | Used with the upstream C++17 build |
| Ninja Python package | `1.13.0` | Already installed in `megatron_env`; do not download a duplicate |
| Ninja executable | `1.13.0.git.kitware.jobserver-pipe-1` | Located under `/opt/conda/envs/megatron_env/bin` |
| pip | `25.2` | The upstream package currently uses legacy `setup.py bdist_wheel` |

The container's initial `PATH` includes `/opt/conda/bin` but omits `/opt/conda/envs/megatron_env/bin`. The installer corrects this root cause by prepending the selected Python environment's `bin` directory before it checks `ninja`. A missing `ninja` result after that correction means the image/environment differs from the validated image and must be investigated; do not install a second Ninja package as an automatic workaround.

## 4. Source Selection Contract

The first install attempt is the user-required VCS command:

```bash
/opt/conda/envs/megatron_env/bin/python -m pip install \
  git+https://github.com/fanshiqing/grouped_gemm@v1.0
```

The command runs with:

```text
PIP_CONSTRAINT=/opt/conda/envs/megatron_env/share/grouped_gemm_ae/constraints.txt
```

Therefore grouped_gemm's transitive dependency resolution is fixed to `absl-py==2.3.1`; it is not left to the newest version available from the package index. The exact-tag archive recovery install uses the same exported constraint. The script does not install an unconstrained grouped_gemm first and repair or downgrade `absl-py` afterward.

The VCS operation is bounded by `GROUPED_GEMM_VCS_TIMEOUT_SECONDS` and its numeric status is recorded. The exact-image network path was able to check out grouped_gemm commit `7a7f0189797889e926a30b3487512f9539161060`, but the recursive CUTLASS submodule clone stalled on GitHub transport. The validation run therefore recorded:

```text
VCS_INSTALL_EXIT_STATUS=124
VCS_LOG_EXIT_STATUS=0
SOURCE_RECOVERY_USED=true
```

After a nonzero VCS status, the installer uses only the exact grouped_gemm `v1.0` archive and the exact CUTLASS gitlink archive listed above. Both hashes are checked before extraction. This is the explicitly approved exact-tag source route; it does not change the grouped_gemm version, build mode, or architecture list.

The exact-source build records both sides of its final `pip | tee` pipeline:

```text
SOURCE_INSTALL_EXIT_STATUS=<pip status>
SOURCE_LOG_EXIT_STATUS=<log persistence status>
```

A nonzero log status is fatal even when pip itself succeeds, because an AE installation without its retained build log is not reproducible evidence.

Any archive hash mismatch, incomplete source tree, compiler failure, installation failure, constraint mismatch, `absl-py` version mismatch, import failure, or manifest mismatch is fatal.

## 5. Architecture Build and Evidence

### 5.1 Default multi-architecture build

The default configuration requests native code for these targets:

| GPU family | CUDA target | Evidence level |
|------------|-------------|----------------|
| A800 / A100 class | `sm_80` | Static cubin inspection only |
| RTX 3090 | `sm_86` | Static cubin inspection only |
| RTX 4090 | `sm_89` | Static cubin inspection only |
| H800 / H100 class | `sm_90` | Static cubin inspection plus live H800 numerical and MoE validation |

The validated backend shared object contained four native cubins for each of `sm_80`, `sm_86`, `sm_89`, and `sm_90`, for `16` cubins total. `cuobjdump --dump-ptx` showed only `Fatbin elf code` records and no PTX program, so the validated artifact provides native cubin coverage for exactly those four targets; it does not provide a generic `compute_*` PTX JIT path.

The live backend artifact was:

```text
Size:   5139112 bytes
SHA-256: c1bd63b7af06db7144ca430476961f36b7d1eaafe33ad58df93b06dd9f67be04
```

The shared-object hash is evidence for this specific build, not a hard-coded expected hash for future compiler invocations. The installer records and verifies the hash within each container through:

```text
/opt/conda/envs/megatron_env/share/grouped_gemm_ae/manifest.env
```

### 5.2 Upstream backend semantics

The unmodified `v1.0` build uses this source-level selection:

```cpp
#if !defined(GROUPED_GEMM_DEVICE_CAPABILITY) || GROUPED_GEMM_DEVICE_CAPABILITY != 80
  CublasGroupedGemm(...);
#else
  CutlassGroupedGemm(...);
#endif
```

An explicit `TORCH_CUDA_ARCH_LIST` leaves `GROUPED_GEMM_DEVICE_CAPABILITY` undefined. Therefore, the default multiarch artifact uses the upstream cuBLAS grouped path. Do not describe the four-target build as four CUTLASS-optimized grouped kernels.

### 5.3 Explicit native backup

Native mode is retained only as an explicit backup after a default multiarch failure has been preserved and analyzed:

```bash
GROUPED_GEMM_BUILD_MODE=native \
  bash tools/ae/setup_grouped_gemm_v1.sh
```

The script never selects native mode automatically. A native SM80 build may select the upstream CUTLASS path; native H800/SM90 uses the cuBLAS path. Native mode covers only the GPU visible during that build and must not be reported as the multi-GPU-family AE artifact.

## 6. Validation Commands

After installation, verify the package and manifest:

```bash
/opt/conda/envs/megatron_env/bin/python -m pip show \
  grouped-gemm absl-py ninja

cat /opt/conda/envs/megatron_env/share/grouped_gemm_ae/manifest.env

cat /opt/conda/envs/megatron_env/share/grouped_gemm_ae/constraints.txt

/opt/conda/envs/megatron_env/bin/python - <<'PY'
import importlib.metadata
import pathlib
import torch
import grouped_gemm
import grouped_gemm_backend

print("grouped_gemm=" + importlib.metadata.version("grouped-gemm"))
print("absl_py=" + importlib.metadata.version("absl-py"))
print("backend=" + str(pathlib.Path(grouped_gemm_backend.__file__).resolve()))
PY
```

The expected supplemental state includes:

```text
# constraints.txt
absl-py==2.3.1

# manifest.env field
ABSL_PY_VERSION=2.3.1
```

Import `torch` or `grouped_gemm` before importing `grouped_gemm_backend` directly. The backend is a PyTorch extension linked against `libc10.so`; an isolated backend-only import does not establish the normal PyTorch shared-library context and is not the supported import contract. Do not add an `LD_LIBRARY_PATH` workaround unless a supported import still fails and new evidence proves that path configuration is the root cause.

Inspect compiled targets:

```bash
BACKEND_SO=$(
  /opt/conda/envs/megatron_env/bin/python - <<'PY'
import pathlib
import torch
import grouped_gemm
import grouped_gemm_backend
print(pathlib.Path(grouped_gemm_backend.__file__).resolve())
PY
)

sha256sum "$BACKEND_SO"
cuobjdump --list-elf "$BACKEND_SO"
cuobjdump --dump-ptx "$BACKEND_SO"
```

Run the H800 BF16 forward/backward reference test:

```bash
/opt/conda/envs/megatron_env/bin/python \
  tests/integration/test_grouped_gemm_v1_runtime.py
```

The test is intentionally H800-specific. It checks fixed and variable expert token counts for `trans_b=False` and `trans_b=True`, including forward output, input gradient, weight gradient, finite counts, and five-repeat determinism.

## 7. Validated Results

### 7.1 Installation

| Metric | Actual result |
|--------|---------------|
| Exact-image live installation | PASS |
| VCS attempt | Timed out with status `124`; full log persisted |
| Exact-tag source hashes | `2/2` PASS |
| Source install status | `0` |
| Log pipeline status | `0` |
| Measured install wall time | `289 s` with a validation-only `60 s` VCS timeout |
| Installed package | `grouped_gemm==0.0.1` |
| Additional Python package | `absl-py==2.3.1` |
| Second installer invocation | `GROUPED_GEMM_INSTALL_STATUS=already_satisfied`; exit `0` |
| Current exact-dependency unit contract | `30/30` PASS; both pip routes constrained; wrong post-install, manifest, and live idempotent `absl-py` versions rejected; malformed newline and source-log failure covered |

The retained live build observed `absl-py==2.3.1` in the exact image. The later exact constraint/manifest/log-integrity hardening was validated offline through the current 30-case installer suite; it did not allocate another GPU or rebuild the already validated binary. The AE default VCS timeout remains `600 s`, so wall time can exceed the validation measurement when GitHub transport stalls.

### 7.2 H800 numerical correctness

All `12` compared surfaces passed: four cases multiplied by output, input gradient, and weight gradient.

| Metric | Expected limit | Actual maximum |
|--------|----------------|----------------|
| Finite elements | All elements finite | `282624 / 282624` finite across all compared surfaces |
| Maximum absolute error | `<= 0.125` | `0.0` |
| Mean absolute error | `<= 0.01` | `0.0` |
| Maximum relative error | Recorded with denominator floor `0.01` | `0.0` |
| Five-repeat maximum delta | `<= 0.0` | `0.0` |

### 7.3 Scaled Mixtral-style MoE

After grouped_gemm installation, the same scaled Mixtral-style configuration passed the previous `MoELayer -> GroupedMLP -> assert_grouped_gemm_is_available` stop and completed:

```text
hidden_size=4096
num_layers=8
ffn_hidden_size=14336
num_attention_heads=32
seq_length=1024
topk=2
fake_world_size=16
fake_pp=2
fake_dp=8
fake_tp=1
fake_exp=2
fake_num_experts=8
```

Observed completion evidence:

```text
local parameter count: 3000139776
finish warm up
finish FWD profile iter 1/1
finish BWD profile iter 1/1
finish optimizer.step profile iter 1/1
SHELL_EXIT_STATUS=0
MIXTRAL_LOG_EXIT_STATUS=0
final H800 memory: 1 / 81559 MiB
```

This validates the requested scaled configuration on one H800 in scaling mode. It is not a live A800, RTX 3090, or RTX 4090 result.

## 8. Fail-Fast Rules for AE

1. Preserve the complete installer log and the first nonzero status.
2. Stop on source-hash, compiler, linker, import, manifest, numerical, or MoE failure.
3. Do not change grouped_gemm tag, CUTLASS commit, Python environment, PyTorch version, CUDA version, or architecture list to obtain a passing run.
4. Do not select `GROUPED_GEMM_BUILD_MODE=native` silently. Use it only after the default failure has a recorded root-cause analysis and the narrower single-GPU target is acceptable.
5. Do not remove `--moe-grouped-gemm` or switch to SequentialMLP; that changes the tested MoE contract.
6. The pip `25.2` build emits a warning that legacy `setup.py bdist_wheel` support will change in pip `25.3`. A future pip upgrade is an environment change and requires a new validation instead of an unreviewed build flag.
7. Do not edit `constraints.txt` or `ABSL_PY_VERSION` in `manifest.env`. A mismatch is evidence that the container environment drifted; the script stops rather than silently repairing or accepting it.

## 9. Evidence Locations

The retained live evidence is under:

```text
/data/ycfeng/task_memory/task_2026-07-12_megatron_image_gpu_validation/
  runtime_logs_grouped_gemm_install/
```

Key files:

```text
verify_worker_identity.txt
verify_multiarch_install.txt
verify_idempotent_install.txt
verify_package_manifest_metadata.txt
verify_backend_binary_targets.txt
verify_backend_dump_ptx.txt
verify_runtime_numerical_validation.json
verify_runtime_numerical_validation.stderr
verify_runtime_numerical_status.txt
verify_scaled_mixtral_moe.log
meg-gg-verify-20260713-0240_terminal_rjob_summary.json
meg-gg-verify-20260713-0240_terminal_replica_summary.json
```

The validation worker finished as `Succeeded`; the RJob reported `final-resource={}`, and the replica reported container exit `0` with `restartCount=0`.
