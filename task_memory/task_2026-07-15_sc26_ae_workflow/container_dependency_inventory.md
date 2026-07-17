# Container Dependency Inventory — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-17 | Recorded D27 selection of the probe-only MemoryTracker qualification path and retained the live H800 evidence boundary |
| 2026-07-17 | Recorded controller-side isolated-loader feasibility evidence for the pending MemoryTracker probe branch |
| 2026-07-17 | Added a CPU-master conda/path inventory to distinguish controller-side environments from GPU-worker runtime paths |
| 2026-07-17 | Recorded Session 15/18 live qualification evidence: dependency gates passed, but MemoryTracker probe hit an existing profiler circular import |
| 2026-07-17 | Reconciled I32 from D26: canonical Megatron/Task1/Task3 uses runtime-minimal closure; full Echo pins remain cp310 Task2-only |
| 2026-07-17 | Recorded Session 12 cp39 package-overwrite failure and the install-only-confirmed-gaps remediation rule |
| 2026-07-17 | Recorded cp310 official manifest, offline resolver/install, pip-check, and pinned Echo import qualification evidence |
| 2026-07-16 | Added the proven Echo Python 3.10 environment requirement, fixed Task-to-interpreter bindings, and independent WATCH disposition |
| 2026-07-16 | Added D26 current-container continuation, `/opt/anaconda/envs/myenv_yc` inventory priority, and corrected pre-install classifications |
| 2026-07-16 | Added source-derived openpyxl qualification for Echo workbook generation and consumption |
| 2026-07-16 | Created the D24 dependency-gap inventory for the replacement pinned image and current validation container |

## 1. Purpose and Authority

This document is the single dependency-gap inventory requested by D24. It serves two distinct contexts:

1. **Replacement AE image:** the user will later use this inventory on another machine to build and push a new immutable internal image. The final image reference and digest are not known yet and must not be invented. D26 states that this future image is unavailable and does not block current execution.
2. **Current validation container:** the agent is explicitly required to inventory the existing conda environments, use the qualified environment when present, and download/install confirmed missing dependencies so Gate B can proceed. Every such change must record the exact command, package/tool version, source, exit status, and post-install verification. This authorization is explicit provisioning, not an automatic runtime fallback.

The old image remains historical evidence:

```text
hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef
```

It must not be described as a qualified release image. Internal images must be pushed to the approved internal registry, not to `docker.io`. The eventual replacement image must be referenced by an immutable tag and digest before AE release qualification.

## 2. Qualified Environment Findings

The earlier B1 probes did not inspect `/opt/anaconda/envs/myenv_yc`, but fresh image-wide probes now close that gap. The current image has no `/opt/anaconda` and no `myenv_yc`; the historical reports came from another environment. The task contract therefore uses two explicit role-bound interpreters. They are not fallback candidates and no task may discover or switch interpreters dynamically.

| Item | Required contract | Observed in old image | Status / required action |
|------|-------------------|-----------------------|--------------------------|
| Megatron / Task1 / Task3 Python | Exact `/opt/conda/envs/megatron_env/bin/python`, Python `3.9.18` | Live H800: torch `2.1.2`, torch CUDA `12.1`, CUDA available, torchvision `0.16.2`, torchaudio `2.1.2`, Transformer Engine `1.3.0+5b90b7f` | **Qualified base runtime.** Do not upgrade Python or torch. Install only missing Python packages from the frozen cp39 wheelhouse. |
| Echo / Task2 Python | Separate exact Python `3.10.x` conda env, torch `2.1.2`, CUDA `12.1`, torchvision `0.16.2`, torchaudio `2.1.2` | Not present in the image. Pinned Echo `prediction_api.py` fails on Python `3.9.18` with `TypeError` at `str | None` | **Confirmed required gap.** Provision from one explicit official installer plus a frozen cp310 wheelhouse; hash-verify every payload and qualify on two H800 GPUs. |
| Nsight Systems (`nsys`) | `nsys >= 2024.4.2`, with `profile` and `export` available | Image contains only `2023.1.1.0`; fixed `2024.4.2.133` plus its `56`-package Ubuntu closure passed an offline live smoke | **Confirmed required upgrade.** Final qualification repeats exact-version install, profile, and SQLite export. |
| Nsight Compute (`ncu`) | `ncu >= 2024.3` | Image contains `2023.1.1.0`; fixed `2024.3.2.0` passed version, section-list, and metric-query smoke | **Confirmed required upgrade.** Final qualification repeats the exact version and Task2 query checks. |
| `grouped_gemm` | Upstream tag `v1.0`, commit `7a7f0189797889e926a30b3487512f9539161060`, distribution `0.0.1`; import `grouped_gemm` and `grouped_gemm_backend` | The base image historically did not contain it; the repository setup path can install it, but B1 did not qualify it in the default environment | Include and verify it in the replacement image, or run the explicitly selected pinned setup source during current validation. Do not use automatic VCS-to-archive fallback. |
| `absl-py` | Exactly `2.3.1` for the pinned `grouped_gemm` build | Installed only as part of the supplemental grouped-gemm setup contract | Include/verify exactly `2.3.1`; a mismatched live version is fatal. |

## 3. Required but Not Yet Qualified

The first fail-fast probe stopped at the missing `torch` import in the base interpreter. The full inventory later selected the two fixed runtimes above. The following packages remain to be installed or fully qualified in the runtime named by each row.

| Item | Why it is required | Qualification rule |
|------|--------------------|--------------------|
| `pynvml` / NVML bindings | `megatron/profiler/trace_memory.py` otherwise returns from its tracker thread without producing memory JSON | Import in the Megatron env, execute a live NVML device query, then run a memory-trace smoke that writes a nonempty JSON with finite positive peak values. Package presence alone is insufficient. |
| `xgboost` | Echo predictor training/loading and Task3 slowdown prediction | Import and record exact version in both role-bound envs; run the pinned Echo model train/save/reload smoke in the Echo env and the sim-engine slowdown smoke in the Megatron env. |
| `numpy` | Echo training and grouped-gemm runtime dependency | Import and record exact version. Historical validated grouped-gemm environment used `1.26.4`; do not silently change it without requalification. |
| `pandas` | Echo merge, dataset, training, prediction, and Nsight analysis paths | Import and record exact version; execute current-run CSV load/merge smoke. |
| `openpyxl` | Echo slowdown analysis writes `.xlsx`, and merge reads that workbook through pandas | Import and record exact version; write and read a current-run workbook. Do not accept a tracked historical `.xlsx` as evidence. |
| `scikit-learn` | `KFold`, `train_test_split`, `StandardScaler`, and metrics in Echo training | Import `sklearn`, record exact version, and verify scaler serialization/reload. |
| `torchvision` | Echo slowdown collection imports `torchvision.models` | Preserve `0.16.2` in the Megatron env and install/verify `0.16.2` in the Echo env against torch `2.1.2`; validate the selected two-GPU collection path. |
| Build/runtime commands | Pinned grouped-gemm setup requires `curl`, `tar`, `sha256sum`, `nvcc`, `g++`, `ninja`, `nproc`, and `timeout` | Resolve every executable and record its absolute path/version. `nvcc` must report CUDA `12.1`; `pip >= 21` is required. |
| `pytest` and repository test dependencies | Required for unit/integration/e2e validation | Import/resolve exact versions in the canonical Python and run the targeted collection before Gate B continuation. |

No pinned `requirements.txt` exists in the pinned Echo-slowdown tree. Its tracked `environment.yaml` provides package pins but incorrectly declares `python=3.9`; the exact pinned source proves Python `3.10+` is required. The package import inventory is derived directly from tracked Python sources and is converted into an exact cp310 wheel manifest before installation. Do not guess versions from package names or inherit unconstrained solver output.

### 3.1 Python-version root-cause evidence

- Source: `Echo-slowdown/training_testing/prediction_api.py:9`, pinned commit `1390b4416ded08bc1b9cd0620d329d81d4470bf9`.
- Failing runtime: `/opt/conda/envs/megatron_env/bin/python`, Python `3.9.18`.
- Actual exception: `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`.
- Evidence log: `logs/d26_live_worker_echo_python39_annotation_probe_2026-07-16.log`, bytes=`6409`, SHA256=`0d987466665b72a8c37b26aa50828d3c05a32480c9a17af0224e22f8fc6033a5`.
- Resolution boundary: no Echo source edit is authorized in Gate B and D2 keeps the pinned upstream unchanged. The smallest allowed root-cause remediation is a separate exact Python `3.10.x` Echo env. Task3 remains on Python `3.9.18` because its PEP 604 annotations are guarded by `from __future__ import annotations`.

## 4. Current-Container Provisioning Contract

Current-container installation is allowed only after a read-only inventory records the pre-install state. For each confirmed gap, append a row to the table below before declaring the environment qualified.

| Item | Source / immutable identity | Install command | Exit status | Observed version/path after install | Verification result |
|------|-----------------------------|-----------------|-------------|-------------------------------------|---------------------|
| Python cp39 wheelhouse | Official PyPI payloads, `29` files, manifest SHA256 `2ede71d353ae2d7fbd9993a376b89177475efb5ac3acd635f5fc598cad344583` | Pending final worker install | Pending | `/data/ycfeng/ae_dependency_cache/sc26_ae/wheelhouse_cp39/` | Payload gate PASS; runtime gate pending |
| Nsight Ubuntu closure | Official signed Ubuntu Jammy repositories, `56` packages, manifest SHA256 `284d2ecc28ee12980f6b2f9951a304ef806317ab2651424472c6a74e59f107d1` | Pending final worker install | Pending | `/data/ycfeng/ae_dependency_cache/sc26_ae/apt_jammy_nsight_official_master_20260716/` | Offline live smoke PASS; final qualification pending |
| Nsight Systems | `2024.4.2.133-244234382004v0`, SHA256 `c208fedd0e45deb17800a75c7d47b8381763da0537ae8c03051d1818de89e468` | Pending final worker install | Pending | Cached official `.deb` | Profile/export smoke PASS; final qualification pending |
| Nsight Compute | `2024.3.2.3-1`, SHA256 `55be38ea4f345a7a81486232e315f2a81c206d9b8d36371b5cd38ce27ee97c16` | Pending final worker install | Pending | Cached official `.deb` | Version/query smoke PASS; final qualification pending |
| Echo Python `3.10.x` env | Miniconda `26.5.3-1`, installer SHA256 `4a82fe0a50a28e8a9406b3ed8e465b7009aa7d0225566802c3370df96b10d834`; cp310 wheel manifest SHA256 `d7743ee81f3bd0f8a900fd551321e5232abbf22b3191cf720130fc3ea659296c` | Offline `pip install --no-index --find-links /data/ycfeng/ae_dependency_cache/sc26_ae/wheelhouse_cp310_echo -r requirements_top_level.txt` | `0` | `/data/ycfeng/ae_dependency_cache/sc26_ae/conda_envs/echo_py310_miniconda_26_5_3_1/bin/python`, Python `3.10.20` | Resolver dry-run `0`; offline install `0`; `pip check` PASS; pinned `SlowdownPredictor` import PASS; live H800 CUDA qualification pending |

### 4.2 Session 12 cp39 qualification failure and corrected install policy

Session 12 used the exact frozen cp39 payload and reached the post-install dependency check, but the qualification script installed all supplemental wheels unconditionally. The canonical image already contained compatible versions of packages that were not confirmed gaps:

| Distribution | Image version before install | Supplemental wheel version | Consumer constraint | Result |
|--------------|-----------------------------|----------------------------|---------------------|--------|
| `datasets` | `4.0.0` | preserved | requires `huggingface-hub>=0.24.0`, `tqdm>=4.66.3` | existing package was made inconsistent by the overwrite |
| `huggingface-hub` | `0.34.4` | `0.20.3` | `datasets==4.0.0` requires `>=0.24.0` | forced downgrade caused `pip check` failure |
| `tqdm` | `4.67.1` | `4.66.2` | `datasets==4.0.0` requires `>=4.66.3` | forced downgrade caused `pip check` failure |

Evidence: RJob `ws-56153d316be61e0f-jlaunch-c2brg`, node `gpu-h800-0398.host.platform.shaipower.com`, qualification script SHA256 `eff4318e79483ed71a712d26e2d73c2144d66614fb1643902a20e56c9233b0ef`. Preflight gates passed (`CP39_PAYLOAD_GATE=PASS rows=29 bytes=313486578`, `APT_PAYLOAD_GATE=PASS rows=56 bytes=7490802`, `FIXED_BINARY_SOURCE_GATE=PASS rows=4`, `DPKG_AUDIT_BYTES=0`, PyTorch/H800 contract PASS); the first failure was `pip check` with the two `datasets==4.0.0` constraint violations.

The corrected current-container policy is fail-fast and install-only-confirmed-gaps:

1. Inventory the canonical environment with `importlib.metadata` before any pip install.
2. For each frozen manifest row, install only when the distribution is absent.
3. Preserve any present distribution and record its actual version; do not replace it with a cached wheel merely because the wheel exists.
4. If a present version conflicts with a required contract, stop and record the conflict; do not downgrade, upgrade, switch source, or use automatic fallback.
5. Run `pip check` and the full live qualification only after the preserved/new package ledger is complete.

Session 12 remains a failed immutable record. A retry must use a new artifact root and must retain the same source, versions, interpreter, image, APT payload, and resource request.

### 4.3 Canonical cp39 qualification scope (D26)

The canonical `/opt/conda/envs/megatron_env` environment is the fixed Python-3.9 runtime for Megatron, Task1, and Task3. Its current-container qualification is **runtime-minimal**, not a second copy of the full Echo `environment.yaml` pin set:

- install only confirmed missing distributions required by Task1/Task3/sim-engine, using the frozen cp39 payloads;
- preserve existing compatible distributions and record their observed versions in the package ledger;
- verify the actual imports and runtime behavior required by the current tasks (`pynvml`/NVML, MemoryTracker, NumPy, pandas, XGBoost, scikit-learn, openpyxl, torchvision, grouped-gemm, sim-engine predictor, and toolchain);
- run `pip check` after installation;
- do not downgrade or overwrite a present package merely because the Echo cp39 manifest contains an older version.

The full Echo package contract (`Python 3.10.20`, torch/torchvision/torchaudio CUDA 12.1, exact cp310 wheel manifest, pinned `SlowdownPredictor`, and Task2 train/save/reload) remains isolated to the independently provisioned cp310 environment. This scope is already authorized by D26's current-container continuation requirement; it is not an automatic fallback and does not relax the future clean replacement-image release gate.

### 4.1 Current CPU-master cp310 qualification evidence (2026-07-17)

The following gates were executed against the exact Miniconda prefix and frozen official wheelhouse. They qualify package closure and source import only; `CUDA_AVAILABLE=False` and `CUDA_DEVICE_COUNT=0` on the CPU master are expected, so live H800/T2-GPU gates remain open.

| Gate | Reproducible command / artifact | Result |
|------|----------------------------------|--------|
| Manifest integrity | `python /tmp/sc26_generate_cp310_manifest_20260717.py` followed by local size/SHA256 verification of `/data/ycfeng/ae_dependency_cache/sc26_ae/wheelhouse_cp310_echo` against `/data/ycfeng/ae_dependency_cache/sc26_ae/echo_py310/wheel_manifest_cp310.tsv` | PASS; rows=`58`, total bytes=`2,986,969,497`, manifest SHA256=`d7743ee81f3bd0f8a900fd551321e5232abbf22b3191cf720130fc3ea659296c`; every URL host=`files.pythonhosted.org` |
| Offline resolver | `/data/ycfeng/ae_dependency_cache/sc26_ae/conda_envs/echo_py310_miniconda_26_5_3_1/bin/python -m pip install --dry-run --ignore-installed --no-index --find-links /data/ycfeng/ae_dependency_cache/sc26_ae/wheelhouse_cp310_echo --report /data/ycfeng/ae_dependency_cache/sc26_ae/echo_py310/offline_install_report.json -r /data/ycfeng/ae_dependency_cache/sc26_ae/echo_py310/requirements_top_level.txt` | PASS; exit=`0`; report=`offline_install_report.json` |
| Offline install | `/data/ycfeng/ae_dependency_cache/sc26_ae/conda_envs/echo_py310_miniconda_26_5_3_1/bin/python -m pip install --no-index --find-links /data/ycfeng/ae_dependency_cache/sc26_ae/wheelhouse_cp310_echo -r /data/ycfeng/ae_dependency_cache/sc26_ae/echo_py310/requirements_top_level.txt` | PASS; exit=`0`; log=`offline_install_20260717.log`; installed torch=`2.1.2+cu121`, torchvision=`0.16.2+cu121`, torchaudio=`2.1.2+cu121` |
| Dependency consistency | `/data/ycfeng/ae_dependency_cache/sc26_ae/conda_envs/echo_py310_miniconda_26_5_3_1/bin/python -m pip check` | PASS; `No broken requirements found.` |
| Imports and pinned source | `PYTHONPATH=/data/ycfeng/Megatron-LM-sc26-ae/Echo-slowdown /data/ycfeng/ae_dependency_cache/sc26_ae/conda_envs/echo_py310_miniconda_26_5_3_1/bin/python -c 'from training_testing.prediction_api import SlowdownPredictor'` plus imports log | PASS; Python=`3.10.20`, NumPy=`1.26.4`, pandas=`2.2.0`, openpyxl=`3.1.2`, XGBoost=`2.1.0`, scikit-learn=`1.3.0`, transformers=`4.38.2`; `SlowdownPredictor` import PASS |

The current worker qualification must repeat these package checks with live CUDA, NVML, Nsight, grouped-gemm, and the two-GPU Echo train/save/reload smoke. The CPU-master result must not be promoted to a B1 PASS by itself.

### 4.5 Session 15/18 live qualification evidence (2026-07-17)

The already-submitted one-GPU RJob `sc26-ae-b1-session15-20260717` reached the live qualification probe after installing only the `17` confirmed cp39 gaps and preserving `11` compatible distributions. The following prerequisites passed in the worker: `CP39_PAYLOAD_GATE`, `APT_PAYLOAD_GATE`, `FIXED_BINARY_SOURCE_GATE`, `CP39_PACKAGE_POLICY_GATE`, `pip check`, PyTorch/CUDA/H800, NVML, Nsight Systems `2024.4.2.133`, Nsight Compute `2024.3.2.3`, and grouped-gemm prerequisites.

The first failing contract was `MEMORY TRACKER CONTRACT`, before `MemoryTracker` could run. The qualification probe's `from megatron.profiler.trace_memory import MemoryTracker` triggers an existing package-level circular import in `megatron.profiler.__init__`, producing:

```text
ImportError: cannot import name 'trace_decorator' from partially initialized module 'megatron.profiler'
```

This is recorded as a source/import-order qualification issue, not a newly missing dependency. The B1 result remains incomplete. The probe logs are immutable at:

```text
task_memory/task_2026-07-15_sc26_ae_workflow/logs/d26_session17_one_gpu_qualification_20260717.log
task_memory/task_2026-07-15_sc26_ae_workflow/logs/d26_session18_one_gpu_qualification_launch_20260717.log
```

No package downgrade, source edit, automatic fallback, or memory-contract bypass is authorized. D27 now selects the probe-only isolated-loader adjustment; another live qualification still requires a new artifact root, the canonical H800 worker interpreter, and closure of the enhanced docs-only review pause.

### 4.6 Current CPU-master environment/path inventory (2026-07-17)

The controller shell used for this enhanced plan-review checkpoint is not the H800 worker container. The `conda` executable is not on the default `PATH`, but the user-level Miniconda installation is available at `/home/i-fengyicheng/miniconda3/bin/conda`. Its environment list contains only `base`, `aello`, `aic-step-design`, `dev-vidur-feature-attn-sim-e2e`, `dev-vidur-v03-hopper-e2e`, and the previously provisioned cp310 Echo prefix. A read-only filesystem probe found no `/opt/conda/envs/megatron_env/bin/python` and no `/opt/anaconda/envs/myenv_yc/bin/python` on this controller.

| Probe | Observed value | Interpretation |
|-------|----------------|----------------|
| `conda env list` executable | `/home/i-fengyicheng/miniconda3/bin/conda` | Controller-side conda is available when invoked by absolute path; shell activation is not assumed. |
| `/opt/conda/envs/megatron_env` | Missing on controller | Not evidence that the H800 worker runtime is missing; Session 15/18 worker logs already qualified this fixed path on the GPU worker. |
| `/opt/anaconda/envs/myenv_yc` | Missing on controller | No usable `myenv_yc` environment was found in this container context; do not invent or route tasks to it. |
| Echo prefix | `/data/ycfeng/ae_dependency_cache/sc26_ae/conda_envs/echo_py310_miniconda_26_5_3_1/bin/python`, Python `3.10.20` | Available and package-qualified on the controller; CUDA/NVML and two-GPU gates remain worker-only. |
| `nsys` | `/usr/local/bin/nsys`, `2025.6.3.541-256337736014v0` | Controller tool is not the fixed worker qualification binary (`2024.4.2.133`); worker evidence remains authoritative. |
| `ncu` | Not on default `PATH`; multiple cached binaries exist | Do not select a cached binary implicitly; use the fixed `2024.3.2.3` worker source recorded in §4 and the handbook. |

This probe changes no runtime binding: Task1/Task3 remain bound to the worker's `/opt/conda/envs/megatron_env/bin/python`, and Task2 remains bound to the exact cp310 Echo prefix. It only prevents a controller-side missing path from being misreported as a missing worker dependency. No package was installed, no environment was mutated, and no RJob was submitted during this checkpoint.

### 4.7 I33/D27 probe-only feasibility evidence (controller, 2026-07-17)

Using the already provisioned cp310 prefix and an isolated module loader, the controller loaded `megatron/profiler/trace_memory.py` without executing `megatron.profiler.__init__`:

```text
isolated_loader_status=PASS
memory_tracker_module= qualification_trace_memory
pynvml_available= False
```

This proves only that the class can be loaded without the package-level circular import. The controller has no CUDA device and no `pynvml` in the cp310 prefix, so this is not a MemoryTracker or B1 pass. D27 selects this probe-only branch, but it must still run under the canonical H800 cp39 interpreter in a new artifact root, query NVML, allocate CUDA memory, and assert a non-empty JSON with positive finite metrics. B2 separately validates the real product package import/runtime path. No package, source, or runtime binding changed during this feasibility check.

Rules:

1. Inventory is complete. Use exactly two role-bound interpreters: Megatron/Task1/Task3 at `/opt/conda/envs/megatron_env/bin/python`; Echo/Task2 at the exact Python-3.10 path recorded after provisioning. These bindings are mandatory and are not fallback candidates.
2. Use exact versions or immutable sources. A selected source failure is fatal; do not switch source/version automatically.
3. Do not lower `nsys`/`ncu` version gates to match the old image.
4. Do not override platform-injected `NCCL_*` variables.
5. Do not treat a successful package install as qualification. Run the acceptance checks in §5.
6. Current-container changes do not qualify the replacement release image; the replacement image must be independently probed from a clean container instance.
7. The concrete pinned-source failure has proved Python `3.10+` necessary for Echo. Provision exactly one Python-3.10 Echo env; do not broaden the version range, patch the source, or route Task3 through it.

## 5. Replacement-Image Acceptance Checklist

The new image is acceptable only when a clean container instance records all of the following:

- [ ] Immutable internal image reference and digest.
- [ ] Megatron/Task1/Task3 Python absolute path and Python `3.9.18`.
- [ ] Echo/Task2 Python absolute path and the exact pinned Python `3.10.x` patch/build.
- [ ] In both envs, PyTorch `2.1.2`, PyTorch CUDA `12.1`, and live CUDA availability; the two-GPU Echo check must see exactly the requested two devices.
- [ ] `pynvml` import, live NVML query, and nonempty memory-trace JSON.
- [ ] `nsys >= 2024.4.2` with working `profile` and `export`.
- [ ] `ncu >= 2024.3` with the Task2 metric collection command available.
- [ ] Exact imports/versions for `numpy`, `pandas`, `openpyxl`, `xgboost`, `sklearn`, and `torchvision` in each env where the task contract uses them; `pip check` passes in both envs.
- [ ] `grouped_gemm==0.0.1`, `absl-py==2.3.1`, backend import/path, and pinned-source identity.
- [ ] Required compiler/build commands and `nvcc` CUDA `12.1`.
- [ ] The actual pinned Echo `SlowdownPredictor` imports in Python `3.10.x`; the canonical sim-engine `EchoSlowdownPredictor` imports in Python `3.9.18`.
- [ ] Echo Task2 two-GPU collection preflight and train/save/reload smoke produce current-run numeric evidence, including original prediction, reloaded prediction, and maximum absolute delta.
- [ ] One-GPU and two-GPU `rlaunch --predict-only` content-level checks pass before live jobs.
- [ ] Exact qualification commands, numeric outputs, and exit statuses are recorded in the task test report.

The historical two-GPU quota failure (`gpu: 129/128`) is independent of image dependencies and requires a fresh recheck. If the fresh check still fails, Task2/B3 remains blocked and cannot be replaced by a single-GPU run; dependency remediation continues independently.
