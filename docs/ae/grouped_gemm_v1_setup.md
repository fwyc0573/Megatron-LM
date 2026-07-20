## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-19 | Replaced automatic VCS-to-archive recovery with required `GROUPED_GEMM_SOURCE=vcs|archive` selection and documented the `SC26-AE/setup.sh` one-command entry point. |
| 2026-07-13 | Made the one-command contract explicit: the script now pins, installs, verifies, and records absl-py 2.3.1 together with exact grouped_gemm and CUTLASS acquisition. |
| 2026-07-13 | Added the validated immutable-image setup, dependency inventory, architecture evidence, and failure policy for grouped_gemm v1.0. |

# AE Supplemental Setup: grouped_gemm v1.0

## 1. Scope

The AE base image is immutable and must not be rebuilt or republished:

```text
Image:  hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef
Digest: sha256:11c794efbfe166932089a7cb9fb1e68fea6fa0101d4265ab51337437184a6c9e
```

The image does not contain `grouped_gemm` or its required `absl-py` dependency. Run the supplemental installer after every new container is started. The one-command entry point installs from one explicitly selected exact source and constrains the same operation to `absl-py==2.3.1`. It fails fast on a missing or invalid source selection, an unexpected Python, PyTorch, CUDA, compiler, source hash, package, dependency version, constraint, or manifest instead of changing source, package version, or backend.

## 2. Required Command

From the repository root inside the started container, run:

```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
GROUPED_GEMM_SOURCE=archive bash SC26-AE/setup.sh
```

The pinned archive source is recommended for AE reproduction because both downloaded source archives have fixed SHA-256 values. The explicit VCS alternative is:

```bash
GROUPED_GEMM_SOURCE=vcs bash SC26-AE/setup.sh
```

Exactly one source must be selected. A missing or unsupported `GROUPED_GEMM_SOURCE` fails before any pip or archive command. A failure from the selected source is final; the entry point and installer never switch to the other source.

Either command is a complete supplemental setup command. AE personnel do **not** need to run a separate `pip install absl-py`, `git clone CUTLASS`, `curl`, or source-build command. The script performs the following ordered actions:

1. Validates the explicit `GROUPED_GEMM_SOURCE=vcs|archive` selection.
2. Creates and verifies an exact pip constraint containing only `absl-py==2.3.1`.
3. Runs only the selected grouped_gemm `v1.0` route: the pinned VCS install, or the two pinned and hash-verified archives with reconstructed `third_party/cutlass`.
4. Verifies grouped_gemm distribution version `0.0.1`, `absl-py` distribution version `2.3.1`, backend import/path, and source identity before writing the manifest or reporting success.
5. On later invocations, verifies the same source method, exact constraint, manifest, live `absl-py` version, backend path, and backend hash before reporting `already_satisfied`.

The default build is the AE release configuration:

```text
GROUPED_GEMM_SOURCE=archive
GROUPED_GEMM_BUILD_MODE=multiarch
TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
MAX_JOBS=8
PIP_CONSTRAINT=/opt/conda/envs/megatron_env/share/grouped_gemm_ae/constraints.txt
```

For the explicit VCS selection, `GROUPED_GEMM_VCS_TIMEOUT_SECONDS` defaults to `600`.

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
| grouped_gemm source archive | `https://codeload.github.com/fanshiqing/grouped_gemm/tar.gz/refs/tags/v1.0`; `15174` bytes; SHA-256 `c80276f32455f7b216c53bab33a050bd3b699415c70098342d0549235326a26f` | Exact-tag source for explicit `GROUPED_GEMM_SOURCE=archive` | Hash verified before extraction |
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

`GROUPED_GEMM_SOURCE` is mandatory and accepts only `vcs` or `archive`. The installer validates it before prerequisite checks or network access. It executes exactly one source branch and records that branch as `SOURCE_METHOD` in the manifest. A later invocation with a different source selection fails on the manifest mismatch rather than silently replacing the installation.

The explicit VCS selection runs:

```bash
/opt/conda/envs/megatron_env/bin/python -m pip install \
  git+https://github.com/fanshiqing/grouped_gemm@v1.0
```

The command runs with:

```text
PIP_CONSTRAINT=/opt/conda/envs/megatron_env/share/grouped_gemm_ae/constraints.txt
```

Therefore grouped_gemm's transitive dependency resolution is fixed to `absl-py==2.3.1`; it is not left to the newest version available from the package index. The explicit archive install uses the same exported constraint. The script does not install an unconstrained grouped_gemm first and repair or downgrade `absl-py` afterward.

The VCS operation is bounded by `GROUPED_GEMM_VCS_TIMEOUT_SECONDS`, and its numeric pip and log statuses are recorded. Any nonzero VCS status is returned as the final selected-source failure. The installer does not run `curl`, extract archives, or perform a source-directory pip install after that failure.

The explicit archive selection downloads only the exact grouped_gemm `v1.0` archive and the exact CUTLASS gitlink archive listed above. Both hashes are checked before extraction. It does not invoke the VCS pip route or the VCS timeout wrapper. Its final `pip | tee` pipeline records both sides:

```text
SOURCE_INSTALL_EXIT_STATUS=<pip status>
SOURCE_LOG_EXIT_STATUS=<log persistence status>
```

A nonzero log status is fatal even when pip itself succeeds, because an AE installation without its retained build log is not reproducible evidence. Any archive hash mismatch, incomplete source tree, compiler failure, installation failure, constraint mismatch, `absl-py` version mismatch, import failure, or manifest mismatch is fatal.

### 4.1 Pre-change historical transport evidence

Before explicit source selection replaced automatic recovery, the exact-image network path checked out grouped_gemm commit `7a7f0189797889e926a30b3487512f9539161060`, but the recursive CUTLASS submodule clone stalled on GitHub transport. That historical validation run recorded:

```text
VCS_INSTALL_EXIT_STATUS=124
VCS_LOG_EXIT_STATUS=0
SOURCE_RECOVERY_USED=true
```

`SOURCE_RECOVERY_USED` is historical output only. The current installer does not emit it and does not reproduce this automatic source switch. This evidence explains why `archive` is the recommended explicit AE selection; it is not the current runtime contract.

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
GROUPED_GEMM_SOURCE=archive \
GROUPED_GEMM_BUILD_MODE=native \
  bash SC26-AE/setup.sh
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
| Pre-change historical VCS attempt | Timed out with status `124`; full log persisted before explicit source selection was introduced |
| Exact-tag source hashes | `2/2` PASS |
| Source install status | `0` |
| Log pipeline status | `0` |
| Measured install wall time | `289 s` with a validation-only `60 s` VCS timeout |
| Installed package | `grouped_gemm==0.0.1` |
| Additional Python package | `absl-py==2.3.1` |
| Second installer invocation | `GROUPED_GEMM_INSTALL_STATUS=already_satisfied`; exit `0` |
| Current exact-dependency and source-selection unit contract | `37/37` PASS; missing/invalid source rejected; VCS and archive cross-calls rejected; selected-source failures propagated; both pip routes constrained; source-specific manifests and idempotency checked |

The retained live build observed `absl-py==2.3.1` in the exact image. The later exact constraint, manifest, log-integrity, and explicit source-selection hardening was validated offline through the current 37-case installer suite; it did not allocate another GPU or rebuild the already validated binary. For an explicit VCS run, the default timeout remains `600 s`, so wall time can exceed the historical validation measurement when GitHub transport stalls.

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
2. Stop on the selected-source, source-hash, compiler, linker, import, manifest, numerical, or MoE failure. Do not switch from `vcs` to `archive`, or from `archive` to `vcs`, within the same invocation.
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
