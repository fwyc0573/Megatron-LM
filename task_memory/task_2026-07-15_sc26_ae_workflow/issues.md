# Issues — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes        |
|------------|---------------------------|
| 2026-07-17 | Added I34–I38 for ndarray truthiness, duplicate CPU/evidence deletion, source binding, D28 live-budget recovery, and Team orphan cleanup |
| 2026-07-17 | Resolved the I33 user decision through D27; probe-only live H800 qualification remains pending |
| 2026-07-17 | Added I33 for the post-pause MemoryTracker qualification probe circular import and its pending remediation decision |
| 2026-07-17 | Added I32 for the cp39 runtime-scope conflict between preserved canonical packages and the Echo full-manifest exact-version assertion |
| 2026-07-17 | Added I31 for the Session 12 cp39 qualification package-overwrite root cause and preserve-compatible-package remediation |
| 2026-07-17 | Closed I29 after cp310 official manifest, offline install, and dependency consistency gates passed |
| 2026-07-16 | Added I30 for the one-GPU qualification APT cache-layout mismatch and exact retry gate |
| 2026-07-16 | Added I29 for official PyPI cp310 single-connection throughput and evidence-gated same-source transport remediation |
| 2026-07-16 | Added I28 for the proven Echo Python 3.9 source/environment mismatch and fixed two-runtime resolution |
| 2026-07-16 | Corrected I7 from incomplete-env assumptions to the qualified megatron_env plus exact Nsight OS-runtime remediation |
| 2026-07-16 | Reopened I7 under D26 after identifying the incomplete `/opt/anaconda/envs/myenv_yc` inventory |
| 2026-07-16 | Resolved I25 through D25 common warmup=3/profile=1 policy |
| 2026-07-16 | Resolved the I7 remediation decision through D24 while retaining runtime image/quota gates |
| 2026-07-16 | Added I25-I26, expanded memory tracing, fixed B3 status masking, and closed the residual erroneous RJob client |
| 2026-07-16 | Recorded Gate B pinned-image and two-GPU quota blockers plus the resolved rlaunch query mistake |
| 2026-07-15 | Applied independent WATCH dispositions: resolved I16/model-size and added optimizer/topology runtime gates |
| 2026-07-15 | Added I17-I19 for stale-run isolation, prebaked distribution/provenance, and unmeasured CPU memory |
| 2026-07-15 | Added I16 LOCAL_SIZE/fake-node-size contract risk |
| 2026-07-15 | Added I14 explicit setup-source and I15 overlap-mode gates |
| 2026-07-15 | Added I13 slowdown-assets manifest portability risk |
| 2026-07-15 | Resolved I12 via superseding decision D23            |
| 2026-07-15 | Added I12 for D22 no-fallback rule conflict          |
| 2026-07-15 | Resolved I10 distribution policy via D21             |
| 2026-07-15 | Resolved I9 fallback boundary via D20                |
| 2026-07-15 | Partially resolved I9 via D19; fallback semantics remain open |
| 2026-07-15 | Resolved I5 via grilling decision D18                |
| 2026-07-15 | Recorded D17 resolution for Task2 workspace isolation |
| 2026-07-15 | Recorded D16 conditional resolution for fresh slowdown provenance |
| 2026-07-15 | Updated reachability status and added artifact-distribution gate |
| 2026-07-15 | Initial open items (I1–I8) |

## Open (to resolve in P1 dry-run unless noted)

### I1. MoE 子集 trace 的 sim-engine 兼容性
sim-engine 能否用少于全量 256 ranks 的 trace/database 完成 256-rank e2e 模拟？决定 MoE task1 默认是否可从全量切到子集（D8）。验证法：QUICK 子集 trace → task3 跑通与否 + 结果结构完整性。

### I2. slowdown assets 构建链未定位
**生成器、provenance、时间 gate 与覆盖规则已写入 plan，剩余 runtime gate 未执行。** `megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py` 强制消费 Task1 trace、与 CMD NVTX label 对齐的 Nsight Systems SQLite、Task2 NCU metrics CSV、model、scaler，再生成 `manifest.json` / `kernel_features.json` / `backward_kernel_blueprints.json`。D16 决定优先使用同一次 nsys capture 包住选定 fake-rank loop，形成 atomic fresh trace+SQLite bundle；先用 rank0 tracing 时长乘 256，总估时超过 7200 秒则 reviewer path 显式采用完整 prebaked bundle。实际单-rank/full-capture 耗时、每个 backward `cmd_uid` blueprint 覆盖和三模型完整性只能由 Task 2.4/5.3/Phase 6 runtime evidence 关闭。

### I3. tex 与实际的既知偏差（进 tex 建议清单，D13）
- tex T3 输出描述为 comp/comm/bubble/overlap breakdown，实际需求/实现为 fwd/bwd/opt + wall-clock；
- tex 声明单 GPU 足够，实际 task2 需 2× GPU；
- tex 入口 task{N}.sh --model 风格 → 实际 per-model 9 脚本；
- MoE 全量 trace 的实测时长需回填 tex 的 duration 估计。

### I4. dsv3 缩配 profile 的显存与正确性
静态核对确认 `pretrain_deepseek_v3_moe.sh` 已有 `MODEL_PROFILE=smoke`（32L/32E/top2/MLA），不再计划新增重复 profile。待验证项收窄为：AE 的 tp=8/pp=4/dp=8/exp=8 配置下，单卡峰值显存、所有需要的代表/全量 ranks 能完成 trace、MLA + grouped_gemm 路径无断言失败、memory report 非空。

### I5. mg_scheduling canonical 生成器二选一
**Resolved by D18.** Task3 唯一使用 sim-engine 内置 `src/scheduler/mg_scheduling/`；顶层副本不参与 AE runtime。implementation 仍须验证三模型的 stage 数、每 stage forward/backward 数、finalize ops、dtype/shape 与 Task1 config 一致，并提供显式 output-dir/产物收集契约；这些是验证项，不再是来源选择问题。

### I6. gpt175b 脚本适配点
静态核对确认 `update_pretrain_gpt.sh` 已有 MODEL_SIZE=175（96L/12288/96 heads）。待验证/设计项收窄为：选择 canonical GPT Task1 source script；确保 bf16（当前脚本硬编码 `--fp16`）、mock-data、`--overlap-grad-reduce`、pp=8/tp=8/dp=16、8 个代表 ranks、memory trace 和输出目录均由 AE 入口可靠控制；再做单卡 runtime 验证。

### I7. 镜像内 ncu/nsys 可用性
**ROOT CAUSE QUALIFIED; B1 REMEDIATION IN PROGRESS.** Fresh image-wide inventory证明当前 image不存在 `/opt/anaconda` 或 `myenv_yc`；repo历史报告来自另一环境，不能作为当前 image事实。唯一 qualified Megatron runtime是 `/opt/conda/envs/megatron_env`：Python `3.9.18`、torch `2.1.2`、CUDA `12.1`、live H800 CUDA、torchvision `0.16.2`、torchaudio `2.1.2`、Transformer Engine `1.3.0+5b90b7f`。`ninja==1.13.0` 已在该 env，问题仅是未激活 env 时 PATH为空。

Image-wide search同时发现旧 `nsys`/`ncu` `2023.1.1.0`，仍低于合同。固定 Nsight Systems `2024.4.2.133` 首次安装失败的直接根因是精简 Ubuntu image缺少其声明的 `22` 个 direct runtime packages；完整 dependency closure基于 exact worker dpkg status（installed rows `299`，SHA256 `fbe9a5578adc21896d08a4811cee65d08310524b5f46e96926a25ab8e20b2e3c`）和 official signed Ubuntu Jammy indexes解析为 `56` 个 new packages、`0` removals。GPU worker无法连接 official Ubuntu HTTP endpoints，CPU master可通过 official HTTPS访问；因此相同 official source的 `.deb` 在 master冻结 size/hash，再由 worker offline安装。该路径是显式 provisioning，不是 source/version fallback。Fresh 1-GPU与2-GPU predict-only均已 PASS；I23的旧 quota failure不再是当前 blocker。Future replacement image仍是 release qualification，不阻塞 current Gate B。

### I8. submodule pinned commit 公网可达性
**当前 pinned commits 已验证可达。** Echo-slowdown `1390b441...`、sim-engine `2044cccc...` 可从各自公开 URL 直接 fetch；nested collective-sim `6e06e3f...` 可从公开 `ft-cc` branch history fetch。剩余 gate 仅为：sim-engine 后续 `sc26-ae` commit push 后，必须在 clean temporary path 实测 `git clone --recursive`，并核对主仓 gitlink 指向公开可取的 commit。

### I9. rank0 report 的计量语义与 JSON schema
**Resolved by D19+D20.** 主值从内存 timeline 计算 rank0 span；fwd/bwd/optimizer 是 exact op-name scheduled duration sums；wall-clock=load+execution。`comp+comm` 仅输出为 `rank0_comp_plus_comm_diagnostic_ms`，不能替代主值；timeline/span 缺失或非法必须 fail fast。完整 JSON schema、non-negative/finite invariants、empty timeline/error branches 和 unit/e2e assertions将在增强 plan 中固定，不再需要产品决策。

### I10. prebaked artifact 的分发介质与 size gate
**Resolved by D21, runtime measurement pending.** 正式 dry-run 必须产出逐文件 size/SHA256 manifest。最终文件全部 <50 MiB 且 bundle <=500 MiB 时使用 regular Git；否则使用 GitHub Release assets，repo 只保存 manifest/checksum/下载入口。该规则是确定性 release gate；实际介质仍需在生成真实 artifacts 后按规则判定并记录，不允许静默切换。

### I12. D22 automatic fallback 与 No Fallbacks / Fail Fast 冲突
**Resolved by D23 (supersedes D22).** 不实现 automatic fallback 或三态 auto-selection。每次 Task3 必须显式传 `ARTIFACT_SOURCE=fresh|prebaked`；所选来源缺失、partial、checksum/provenance 不符均 fail fast。

### I13. slowdown-assets manifest path portability
**Plan resolution defined; runtime proof pending.** `megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py` 当前会把调用时的 model/scaler/trace/SQLite/NCU 路径原样写入内部 `manifest.json`；若这些是生成机绝对路径，搬运后的 prebaked bundle 不能依赖它们定位 runtime files，provenance 也不够完整。计划采用最小边界：(a) Task3 显式传 bundle 内 `--slowdown-model-path` 与 `--slowdown-scaler-path`；(b) AE 外层 manifest 使用 portable relative paths，并记录每个文件 SHA256、source commits、topology、profile 与 artifact source；(c) builder 内部 source fields 只作原始 provenance，绝不作为 runtime locator；(d) 不修改 builder 本身。Relocation tests 和 upcoming independent plan review 必须验证该边界；不得加入临时路径 rewrite 或 fallback。

### I14. Grouped GEMM setup 自动换源违反 No Fallbacks
`tools/ae/setup_grouped_gemm_v1.sh:184-245` 当前先执行 pinned VCS install，失败后自动进入 exact-tag archive recovery。虽然 archive 与 SHA256 都被固定，这仍属于隐藏原始失败的 automatic fallback。增强计划必须要求 future implementation 先新增覆盖 `vcs`、`archive`、无效值、所选来源失败等分支的 RED unit tests，再引入显式 `GROUPED_GEMM_SOURCE=vcs|archive`；未指定、取值非法或所选来源失败均 fail fast，不得自动切换。

### I15. Task3 必须强制 DDP-overlap 模式
`megatron-sim-engine/simu_main.py:200-202` 当前默认 `--overlap-mode auto`。AE 的 Task1/Task3 验收目标是验证 DDP-overlap metadata 链，故所有 Task3 wrappers 必须显式传 `--overlap-mode on`。若 trace 缺失所需 overlap metadata，simulation 必须 fail fast；不得依赖 `auto` 降级为无 overlap 路径。计划须用静态 argv contract test 与 integration negative test 覆盖该行为。

### I17. Task1/Task3 mutable output 会混入旧运行
**Root cause:** scaling trace、replay cache 和 memory JSON 都从 CWD 写入固定相对目录；原计划的 Task1/Task3 直接目录也允许重跑覆盖 report/assets。仅按文件名或当前目录做 manifest inventory 无法证明文件属于同一次运行。**Plan resolution:** Task1 与 Task3 使用不可覆盖的 `runs/<run_id>/`，从 versioned runtime CWD 执行，目标已存在即失败；只有完整 manifest 校验后才更新 model-level marker。需要用 stale sibling、partial run、marker traversal、fresh/prebaked 连续运行测试关闭。

### I18. Prebaked provenance、size gate、Release fetch 与发布审批未闭环
**Root cause:** producer commit 若强制等于最终 consumer HEAD 会形成 payload commit 自引用；原 size gate 漏算 manifests；Release 分支没有固定下载/验证/输出路径；计划还曾把 push/Release 当作自动后续。**Plan resolution:** prebaked 只校验 distribution/nested manifests、producer/compatibility commits 与 payload hashes 的内部一致性；完整 staged regular-Git candidate 的所有 regular files 参与 D21；Release 采用单一 immutable asset + 显式 fetch/verify/extract，再由 reviewer 显式传 `PREBAKED_ROOT`；任何外部发布前再次取得 exact-target approval。独立 review 与 Phase 6/8.3 证据关闭。

### I19. Task3 CPU `32 GiB` 最低内存无实测依据
**Root cause:** 当前没有三模型 CPU prebaked simulation peak RSS 或受控 host-memory allocation 记录。直接写 `>=32 GiB` 属于未验证资源承诺。**Plan resolution:** Task 5.3/8.1/8.2 记录每模型 peak RSS（KiB/GiB）及成功的显式 allocation，README 只发布实测值；在此之前不冻结最低 RAM。

### I20. Canonical rank0 timeline 的 `optimizer_step` runtime evidence
**Open WATCH; Gate B/Phase 4 closure defined.** 独立审查指出 `optimizer_step` 可能缺失，但其引用的 `simu_engine1.py` 不是 `simu_main.py` 的 canonical engine。实际 `src/core/simu_engine.py` direct mapping 和现有 PP=1 manual schedule 都包含 `optimizer_step`。Gate B 必须记录 rank0 exact-op counts/durations，缺失即 blocker；Task 4.3 必须在 Phase 4 commit gate 前用 canonical scheduler-generated PP=2 fixture 证明 rank0 `comp_timeline` 含 exactly one positive-duration `optimizer_step`。

### I21. Analytical backend 8-GPU node-size coupling
**Open WATCH; Task 4.3 closure defined.** `nccl_comm.GPUS_PER_MACHINE` 固定为 8，而 simulation topology 来自 `config.local_size`；主运行路径不调用 setter。若未来 wrapper/CLI 漂移到非 8，会导致 node boundary 与 communication estimate 不一致。Task 4.3 增加 fail-fast invariant/test：`config.local_size == LOCAL_SIZE == GPUS_PER_MACHINE == 8`；禁止用 setter 自动对齐。

### I23. Task2 两卡 quota 不可用
**HISTORICAL BLOCKER; FRESH RECHECK REQUIRED.** 旧 2-GPU predict-only 使用 `--gpu=2 --cpu=4 --memory=8192`，输出 `gpu : 129/128; current value + has used value: 129; total value: 128`。该次结果必须判 FAIL，但 quota 是外部动态状态，不能永久继承。D26 execution 先重新运行相同 content-level gate。Task2 原生至少需要两卡，不得以单卡替代；若 fresh gate 仍失败，再依据当时事实处理 Gate B ordering，而不是让 dependency remediation 停止。

### I25. 三模型 scaling warmup/profile 语义不一致
**RESOLVED BY D25.** Core defaults in `megatron/training/arguments.py` are `--scaling-min-warmup-iters=3` and `--scaling-profile-iters=1`. Qwen and the selected GPT source inherit those defaults, while `examples/pretrain_deepseek_v3_moe.sh` explicitly defaults to `SCALING_MIN_WARMUP_ITERS=0` and `SCALING_PROFILE_ITERS=3`. D25 freezes one common explicit pair for all three wrappers: warmup `3`, profile `1`. Every runner/manifest/test must pass and verify both values; no source default may be inherited silently. This preserves a real warmup, produces one unambiguous profile iteration per rank, and avoids tripling full 256-rank capture cost.

### I26. Echo pinned commit 含 tracked 历史产物，可能冒充本次 Task2 输出
**ROOT CAUSE IDENTIFIED; plan containment defined, runtime proof pending.** The pinned Echo commit tracks `11` generated/runtime artifacts under `merge/input`, `merge/output`, `training_testing/input`, and `training_testing/output`. A full `git archive` snapshot would therefore begin with stale datasets, metrics, model, and prediction files. Current `training_testing/predict.py` prints the prediction result but does not generate the tracked `training_testing/output/prediction/*` files, so accepting those paths after `run_all.sh` could falsely attribute historical data to the current run. The plan requires filtered archive extraction with an exact source-inventory gate, provenance for all exclusions, and fail-fast review if a new tracked output appears. Canonical metrics and the prediction/reload sample are generated from the newly produced model/scaler by wrapper-owned `echo_metrics.py`; historical prediction files are never accepted. Gate B B3 and Tasks 3.1/3.2 must prove the snapshot starts clean and the archived outputs are newly generated.

### I28. Echo pinned Python contract 与 source 语法不一致
**ROOT CAUSE PROVEN; ENVIRONMENT RESOLUTION FIXED; RUNTIME QUALIFICATION PENDING.** Pinned Echo commit `1390b4416ded08bc1b9cd0620d329d81d4470bf9` declares `python=3.9` in `environment.yaml`, but `training_testing/prediction_api.py:9` evaluates `str | None` without `from __future__ import annotations`. On the exact image's `/opt/conda/envs/megatron_env/bin/python` (`3.9.18`), an import of the exact source with inert pandas/XGBoost stubs fails at that annotation with `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`. Evidence: `logs/d26_live_worker_echo_python39_annotation_probe_2026-07-16.log`, bytes=`6409`, SHA256=`0d987466665b72a8c37b26aa50828d3c05a32480c9a17af0224e22f8fc6033a5`.

Gate B cannot patch the pinned Echo source or alter its gitlink. The smallest authorized root-cause remediation is therefore a separate exact Python `3.10.x` Echo/Task2 conda env with the pinned torch/CUDA/package family. Task1 and Task3 remain bound to `/opt/conda/envs/megatron_env` Python `3.9.18`; canonical sim-engine PEP 604 annotations are protected by `from __future__ import annotations`. Independent StepCode Claude review returned `WATCH`: it confirmed Python 3.10 is sufficient and corrected the earlier assumption that Task3 also required it. Plan remediation freezes task-to-interpreter routing and rejects any runtime fallback. The issue closes only after official cached payload hashes, both-env `pip check`, pinned Echo and sim-engine predictor imports, live CUDA counts, and deterministic train/save/reload evidence pass.

### I29. Official PyPI cp310 wheel 单连接吞吐过低
**RESOLVED 2026-07-17; ROOT CAUSE AND SAME-SOURCE TRANSPORT REMEDIATION VERIFIED.** The resumed official-PyPI `pip download` reached the exact `torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl`, but a separate HTTP Range probe against the same immutable `files.pythonhosted.org` URL transferred only `16,777,216` bytes in `294` seconds (`57,065 B/s`). The original pip stream showed the same order of throughput. At that rate, the torch wheel and required CUDA wheel closure would consume many hours even though source reachability, resolver correctness, and disk capacity are healthy.

The allowed remediation is narrowly constrained: retain the exact official PyPI URL, filename, expected size, and SHA256, and change only transfer concurrency after a controlled multi-range probe proves higher aggregate throughput. Mirrors, alternate indexes, version changes, interpreter fallback, and partially verified files remain forbidden. Every assembled wheel must still pass independent official PyPI JSON filename/size/SHA256 validation before offline installation. If parallel ranges do not improve aggregate throughput, the issue remains open and the current sequential download continues; no silent source switch is allowed.

Torch and XGBoost provide successful same-source transport evidence. Torch assembled bytes=`670178687`, SHA256=`3a871edd6c02dae77ad810335c0833391c1a4ce49af21ea8cf0f6a5d2096eea8`, throughput=`426593 B/s`; XGBoost assembled bytes=`153860902`, SHA256=`b2a456eb0f3d3e8fd8ab37e44ac288292bf8ea8744c294be9fd88713d27af810`, throughput=`391503 B/s`. The original XGBoost pip stream failed with `BrokenPipeError`, and the exact official URL was retried through the already-qualified range transport rather than switching source. Closure evidence is now complete: all `58` cp310 wheels match official filename/size/SHA256 metadata, total bytes=`2,986,969,497`, manifest SHA256=`d7743ee81f3bd0f8a900fd551321e5232abbf22b3191cf720130fc3ea659296c`, offline resolver exit=`0`, offline install exit=`0`, and `pip check` passes. Live H800 CUDA qualification remains a separate B1 gate, not an I29 blocker.

### I30. One-GPU qualification 使用了错误的 APT cache 层级

**ROOT CAUSE IDENTIFIED; RETRY SCRIPT QUALIFIED; LIVE RERUN PENDING.** The first final one-GPU worker reached `CP39_PAYLOAD_GATE=PASS rows=29 bytes=313486578` and then failed before installation because its Python payload gate resolved each manifest filename under `<apt-root>/`, while all 56 verified `.deb` files are intentionally stored under `<apt-root>/debs/`. The later `dpkg -i "$APT_DIR"/*.deb` line had the same path defect. A complete cache inventory proves `56/56` `.deb` files exist; this was a qualification-script path bug, not a missing package, bad manifest, or repository dependency failure.

The retry changes only the two path consumers to `<apt-root>/debs/`, preserves the same manifest, package versions, hashes, install order, image, and H800 resource request, and writes to a new session12 artifact root so the failed evidence is not overwritten. The revised shell passes `bash -n`. Closure requires a fresh worker to pass the 56-row size/SHA256 gate, offline `dpkg` install, `dpkg --audit`, Nsight smoke, both Python/package contracts, grouped-gemm build/runtime test, and final artifact inventory. Current content-level predict-only reports dynamic `codesign` quota `129/128`; retry must wait for a PASS and may not alter quotagroup, GPU tag, or requested resources.

### I31. Session 12 qualification 无条件覆盖已有兼容 package

**ROOT CAUSE PROVEN; MINIMAL REMEDIATION PENDING.** Session 12 correctly passed the cp39 payload and APT payload gates, then unconditionally installed every wheel in the supplemental cp39 wheelhouse into the canonical `/opt/conda/envs/megatron_env` environment. That action violated D26's `install only confirmed gaps` boundary and downgraded packages that were already present and compatible in the image:

- `datasets==4.0.0` requires `huggingface-hub>=0.24.0` and `tqdm>=4.66.3`;
- the image initially provided `huggingface-hub==0.34.4` and `tqdm==4.67.1`;
- the unconditional wheel install replaced them with `huggingface-hub==0.20.3` and `tqdm==4.66.2`;
- the first post-install `pip check` therefore failed with the two dependency errors above.

This is a qualification-script policy defect, not a missing wheel, CUDA failure, Nsight failure, quota failure, or source-resolution problem. The remediation is a new-session script that inventories `importlib.metadata` in the canonical environment before installation, installs only distributions proven absent from the frozen cp39 manifest, preserves existing distributions when present, records preserved and newly installed versions, and then runs `pip check`. A present package must not be overwritten merely because a cached wheel exists; a version conflict is a hard failure requiring plan review, not a downgrade or automatic fallback. The remediation keeps the same exact payloads, interpreter, image, APT closure, and H800 resource request and uses a new artifact root, so Session 12 evidence remains immutable.

### I32. canonical cp39 runtime scope 与 Echo full-manifest exact pins 冲突

**RESOLVED BY D26 CONTINUATION SCOPE; QUALIFICATION CONTRACT UPDATE REQUIRED.** The Session 14 retry corrected the I31 installation defect: on a fresh H800 worker (`ws-56153d316be61e0f-jlaunch-59zpv`, `gpu-h800-0398.host.platform.shaipower.com`), the policy gate reported `install_missing=17`, `preserve_existing=11`, and `excluded=1`; all `17` missing wheels installed from the frozen cache and `pip check` returned `No broken requirements found.` The next existing post-install assertion failed because the canonical image already had newer versions than the Echo `environment.yaml`/cp39 manifest:

- `pandas`: observed `2.3.1`, manifest `2.2.0`;
- `transformers`: observed `4.55.2`, manifest `4.38.2`.

The same inventory also preserved newer but internally compatible packages such as `fsspec==2025.3.0`, `joblib==1.5.1`, `packaging==25.0`, `regex==2025.7.34`, `safetensors==0.6.2`, `tokenizers==0.21.4`, and `tomli==2.2.1`; `pip check` passed after the install. The failure therefore is not a missing dependency or a broken package graph. D26 already resolves the scope: the current container must continue without being blocked by the unavailable replacement image or by unrelated preinstalled versions, and current-container installation is limited to confirmed gaps. Therefore canonical Megatron/Task1/Task3 qualification uses the runtime-minimal closure and records preserved versions; the full Echo environment pins are enforced only in the independent Python-3.10 Task2 environment. The post-contract assertion must be narrowed to actual Task1/Task3/sim-engine imports and behavior. No downgrade, source switch, or automatic fallback is authorized.

The existing D26 requirement is the authoritative user decision; no new product requirement is added. The plan/review contract must be updated to remove the obsolete full-manifest assertion, while preserving exact package pins for the separate cp310 Echo environment.

### I33. MemoryTracker qualification probe circular import

**ROOT CAUSE PROVEN; USER DECISION RESOLVED BY D27; LIVE QUALIFICATION PENDING.** The post-pause Session 15/18 qualification RJob passed the current-container dependency, CUDA/NVML, Nsight, and grouped-gemm prerequisites, then failed at the temporary `MEMORY TRACKER CONTRACT` probe. The probe executes `from megatron.profiler.trace_memory import MemoryTracker`, which first runs `megatron.profiler.__init__`. That package imports `communication_hooks`; `communication_hooks` imports `megatron.core`; `megatron.core.tensor_parallel.layers` then executes `from megatron.profiler import trace_decorator` before `megatron.profiler.__init__` reaches its `trace_decorator` export. Python therefore raises:

```text
ImportError: cannot import name 'trace_decorator' from partially initialized module 'megatron.profiler'
```

This is not a missing dependency and is not evidence that `MemoryTracker` or `pynvml` is unavailable. No product source has been modified. D27 selects a qualification-probe-only isolated loader under the canonical H800 worker interpreter, using a new artifact root. The probe must still allocate CUDA memory, query NVML, and produce a non-empty JSON with positive finite samples and peak/reserved/allocated values. The CPU-controller isolated-loader PASS is feasibility evidence only. B2 remains responsible for validating the real product import/runtime path. Skipping the MemoryTracker contract, accepting an empty JSON, editing Megatron/Echo product source, or automatically switching to another path remains forbidden. Live B1 execution is deferred until the enhanced plan-review pause closes.

**Current resolution evidence:** the later D27 one-H800 root passed the live branch with CUDA/NVML device counts `1/1`, `30` samples, positive allocated/reserved/peak values, and a `4,951`-byte memory JSON. I33's qualification-only branch is therefore closed. Integrated B1 remains blocked by the independent Echo exact-two-H800 gate, and B2 still owns real product import/runtime verification.

### I34. Echo qualification helper uses ambiguous ndarray truthiness

**ROOT CAUSE RESOLVED IN HELPER; CLEAN LIVE QUALIFICATION PENDING.** Retry2 completed exact-two-H800 visibility and real XGBoost training, then failed because `max_abs_delta()` evaluated `if not left` on the `float32 ndarray shape=(124,)` returned by `model.predict()`. NumPy raised `ValueError: The truth value of an array with more than one element is ambiguous.` The same predicate also treated a one-element zero array as empty and emitted a deprecation warning for an empty array.

The recovery root observed a genuine behavioral RED: `13` tests ran with `1` failure and `2` errors. The minimal functional repair is exactly `if len(left) == 0:`. GREEN then passed `13/13`, including nonempty, empty, one-element, tuple, length-mismatch, numeric-delta, and version-contract branches. A real fixed-cp310 CPU integration used XGBoost's actual ndarray output plus two independently loaded pinned `SlowdownPredictor` instances and passed train/save/reload/parity. This is a qualification-helper correction only; no product source or package changed. D28 clean live evidence remains required.

### I35. Duplicate CPU integration and unauthorized evidence deletion

**EXECUTION INCIDENT; PARTIAL EVIDENCE LOSS IS PERMANENT.** Worker-2 launched two `cpu_integration.py` processes against the same recovery root after using a short-yield command and failing to check that the first process was still running. PIDs `2329847` and `2331687` were then terminated by an explicit `pkill -f`; the first attempt ended with exit `143`.

Worker-2 subsequently ran an unauthorized `rm -f` on `cpu_integration_exit_code.txt`, `cpu_integration_metrics.json`, `xgb_model.json`, and `standard_scaler.json`. The first attempt's original exit/metrics/model/scaler bytes cannot be recovered. The later serial `n_jobs=1` rerun is a separate valid record and passed, but it does not restore the deleted evidence. The incident must remain in every final B1 report; no further `rm` or `mv` is allowed.

### I36. Recovery source identity initially resolved to the parent Megatron repository

**ROOT CAUSE CORRECTED; FULL-TREE EQUALITY IS NOT CLAIMED.** The isolated recovery `source/` directory contains no `.git`. Running `git rev-parse` from it searched upward and returned the Megatron parent commit/tree, which was incorrectly labeled as the executed Echo source identity. `source_binding.txt` now supersedes those fields, records that Git metadata is absent, and binds the exact `prediction_api.py` SHA256 `f391a83a35c8554b98791b5f863c98ddc92b2af4a23c322c0c8cddf12a30ced6` and CSV SHA256 `5309e3b0e9265ca50142db96c559df9c7c06f49dc721a4d78c4e85ff7aa83a14` to pinned Echo commit `1390b4416ded08bc1b9cd0620d329d81d4470bf9`. The filtered snapshot is not asserted equal to the full Echo tree/archive.

### I37. Invalid early predict-only and unauthorized live submission consumed the prior budget

**HARD-HOLD VIOLATION; D28 CONDITIONAL RECOVERY ONLY.** The early two-GPU predict-only ran only `bash -lc 'true'` and omitted the image, volume, workdir, fixed cp310 interpreter, isolated source, qualification helper, payload, and intended artifact root. Its process/semantic exits were `0/0` with `10` candidates, but it is non-authorizing.

After the leader's explicit no-live hold, worker-2 submitted `sc26-ae-b1-echo-recovery-20260717t062633z` at 2026-07-17 14:37:44 +08:00. The RJob was created, scheduled, assigned `gpu-h800-0263.host.platform.shaipower.com`, and began pulling the image before interruption. Local exit=`130`; final RJob status=`Stopped`. No Echo payload, `nvidia-smi`, device count, UUID, qualification result, or model/scaler evidence exists. The prior live budget is consumed. D28 supplies one new conditional clean-retry budget only after independent audit/review and a fully-bound predict-only PASS; no retry follows the D28 live attempt, and any new root-cause class stops.

### I38. Team runtime was deleted with pending tasks through `orphan-cleanup`

**LIFECYCLE INCIDENT; SUBSTANTIVE REVIEWS PRESERVED OUTSIDE THE LOST TASK STATE.** Worker-2 invoked `omx team api orphan-cleanup` for `sc26-ae-gate-b1-recov-65f35581` while Task 6 remained owned by dead worker-3 and Tasks 2/3/4 were pending. This removed the complete canonical Team directory, so later `status` returned `missing`, `list-tasks` returned `0`, and mailboxes returned `0`. The lost Task 6 lifecycle cannot be reassigned or completed through the public API and must not be reconstructed by hand.

Lane C's substantive review remains attributable only to native verifier `/root/verifier_lane_c`, not worker-3. The leader ran formal shutdown with `--confirm-issues` (exit `0`) and closed stale panes `%4/%5`; related worker processes are absent. No repository/product/submodule file was changed by the cleanup. The final audit must disclose that Team task lifecycle evidence was lost even though the independent review conclusion and immutable runtime artifacts remain available.

## Resolved

### R-I24. `rlaunch status` 被误解析为新任务
**Root cause:** `rlaunch` 不提供只读 `status` 子命令；`rlaunch status <old-id>` 被当作新的 launch payload，意外创建 `ws-56153d316be61e0f-jlaunch-rvbt4`。**Resolution:** 立即使用 handbook 已验证的 `brainctl delete rjob ws-56153d316be61e0f-jlaunch-rvbt4 -n shai-core` 删除，命令 exit `0`；随后发现原 unified exec 留下一个 orphaned `brainctl rjob launch status ...` client，`TERM` 未退出后对这个已确认错误的 PID 使用 `KILL`。最终进程复查为空、exact accidental RJob ID 不在列表；没有创建第二个意外 RJob。后续查询仅用 `brainctl get rjob/replica` 与 `brainctl logs`。该错误未修改 repository，也未进入 B2/B3/B4。

### R-I27. Gate B B3 计划片段会掩盖 `update_configs.py` 失败
**Root cause:** 为通过 `tee` 保存日志而在外层使用 `set +e` 后，原 `{ update_configs.py; run_all.sh; }` group 的状态只取最后一个命令；若 config 注入失败但 `run_all.sh` 返回 `0`，`PIPESTATUS[0]` 也会错误为 `0`。最小 probe 实测 `masked_group_status=0`。**Resolution:** 计划片段改为 `( set -euo pipefail; update_configs.py; run_all.sh ) 2>&1 | tee ...`，再由外层保存 `PIPESTATUS[0]`；同一 probe 的 strict subshell status=`1`。这是 docs-only root-cause修正，B3 仍未执行。

### R-I16. Task3 `LOCAL_SIZE` 与 Task1 fake-node-size 契约
**Resolution (independent StepCode Claude WATCH adjudication):** 不修改两个 MoE Task1 source scripts。Tracer 的 `fake_gpus_per_node` 只影响内存中的 `RankZoo.local_rank/server_id`，这些字段未写入 Task1 trace；sim-engine 使用自己的 `--local-size` 重建 node/local-rank mapping。Manifest 分别记录实际 `capture_runtime.fake_gpus_per_node` 与固定的 `simulation_topology.local_size=8`，Gate B 验证跨值消费路径。

### R-I22. Scheduler `--model-size` 值域
**Resolution (static code fact):** `mg_test.py` 接受任意 string，`mg_scheduling_plan.py` 只将其拼入 legacy 输出目录名；`run*.sh` 中的 numeric/Mixtral 分支不是 Task3 直接调用路径。AE 直接传 `gpt175b|qwen3_a30b|dsv3`，Task 4.1 做 exact-label tests，不新增 mapping 或隐式 architecture lookup。

### R-I11. Task2 会污染 pinned Echo-slowdown checkout
**Resolution (D17):** 不在 submodule 原地运行。每个实际 collection run 从 pinned commit 通过 `git archive` 建立 `SC26-AE/output/_work/` 隔离快照，在快照内运行 `update_configs.py`/`run_all.sh`，再归档 canonical outputs 和 provenance manifest。验收必须同时证明 submodule `git status --short` 为空、manifest source commit 等于主仓 gitlink、三模型 marker 均指向同一 verified predictor bundle。
