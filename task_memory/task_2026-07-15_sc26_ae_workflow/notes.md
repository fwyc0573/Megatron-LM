# Notes — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes            |
|------------|-------------------------------|
| 2026-07-23 | Recorded the mandatory `/data/ycfeng/tmp` rule, no-Task2-rerun rule, and current clean producer/output roots |
| 2026-07-20 | Recorded the penultimate tracked-snapshot V21 PASS and local I59 closure; final identity/commit replay remains mandatory |
| 2026-07-20 | Added Session 56 staged-audit/runtime-output scope findings, task-archive whitespace handling, and independent follow-up `APPROVE` |
| 2026-07-20 | Recorded the D31 exact V21 log-tracking boundary and the D32 cleanup of reviewer-generated nested `.omc/` runtime state |
| 2026-07-19 | Recorded Session 45 audit boundaries: no GPU/RJob/publication, raw producer overlay is not a pinned release source, and qualified lifecycle changes require an approved design |
| 2026-07-19 | Recorded /tmp inode exhaustion and the verified SC26_AE_TMP_ROOT/TMPDIR execution contract |
| 2026-07-19 | Added D30 latest-user gate: any test/validation/rehearsal-exposed AE defect may be self-repaired with evidence, without weakening acceptance or release boundaries |
| 2026-07-19 | Recorded D29 test-issue self-repair scope, evidence obligations, and hard-block boundaries |
| 2026-07-17 | Added D28 Gate B1 split verdict, incident boundaries, conditional clean-retry contract, and Team lifecycle note |
| 2026-07-17 | Recorded the D27 probe-only MemoryTracker qualification boundary and B2 product-path obligation |
| 2026-07-17 | Recorded cp310 offline qualification evidence and CPU-master CUDA boundary |
| 2026-07-16 | Recorded the proven Echo Python 3.9 mismatch, fixed two-runtime task bindings, and offline conda-source constraint |
| 2026-07-16 | Recorded D26 current-container continuation and the missing `/opt/anaconda/envs/myenv_yc` inventory correction |
| 2026-07-16 | Recorded D25 common scaling warmup=3/profile=1 contract |
| 2026-07-16 | Recorded D24 replacement-image and explicit current-container provisioning boundaries |
| 2026-07-16 | Added static Gate B reconciliation for memory tracing, scaling iterations, Qwen batch flags, and Echo snapshot contamination |
| 2026-07-16 | Added Gate B pinned-image, quota, and safe RJob-query evidence |
| 2026-07-15 | Recorded independent WATCH adjudication and scheduler/simulator fact reconciliation |
| 2026-07-15 | Added author self-review facts for run isolation, database binding, provenance, distribution, and measured CPU memory |
| 2026-07-15 | Added LOCAL_SIZE and fake-node-size topology findings |
| 2026-07-15 | Added explicit-source setup and overlap-mode findings |
| 2026-07-15 | Added slowdown-assets manifest portability finding |
| 2026-07-15 | Added public commit reachability and artifact-size evidence |
| 2026-07-15 | Added enhanced-review fact-check findings |
| 2026-07-15 | Initial operational notes     |

## Terminology / naming
- System name: **Moye**（paper `main.tex:74`）= Echo，同一对象的两个名字。
- `ep` 在本 codebase 语境一般指 embedding parallel；expert parallel 用 `exp`（CLI `--fake-exp`）。目录名里 `ep2`/`expn8` 之类需按上下文甄别。

## Environment / infra quirks
- **Current AE qualification target (2026-07-19):** use `hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae` for any new preflight, predict-only, or live qualification command. Resolve and record its immutable digest before treating it as qualified; tag availability is not evidence. The `v1.1-image-11c794ef` entry immediately below is historical evidence only and must not be reused as the current target.
- Historical AE image: `hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef`；它在 Gate B B1 失败，不能作为最终 release image。现有补齐入口 `tools/ae/setup_grouped_gemm_v1.sh`（幂等，manifest at `$STATE_DIR/manifest.env`；装 grouped_gemm v1.0 + absl-py==2.3.1；期望 Python 3.9.18 / torch 2.1.2 / CUDA 12.1，失配即 fail-fast）。
- GPU worker: 按 `/data/ycfeng/stepfun-env-handbook/guidence.md`，rlaunch 验证组合 `--charged-group=codesign --private-machine=group --positive-tags=h800`，`--backoff-limit>0`，大额申请前 `--predict-only`。
- Gate B 2026-07-16 实测：1-GPU predict-only exit `0`，候选 H800 nodes=`6`，最大候选 GPUs=`8`；2-GPU predict-only 的 CLI exit 也是 `0`，但 quota 输出为 `129/128`，必须按内容判 FAIL。
- Pinned image 默认 `/opt/conda/bin/python`（Python `3.9.18`）无 torch；`/opt/conda/envs/megatron_env/bin/python` 有 torch `2.1.2` / CUDA `12.1`。`nsys` 不存在；`ncu=/usr/local/cuda/bin/ncu` 且版本 `2023.1.1.0`。
- D26 fresh image-wide inventory 已关闭遗漏：当前 image 不存在 `/opt/anaconda` 或 `myenv_yc`，历史 reports 来自其他环境；唯一 qualified Megatron runtime 是 `/opt/conda/envs/megatron_env`。Task1 与 Task3 固定使用该 Python `3.9.18` env，不升级其 Python/torch。
- D24 正式 release 路径为 `new_pinned_ae_image`。缺失/未验证依赖的单一清单是 `task_memory/task_2026-07-15_sc26_ae_workflow/container_dependency_inventory.md`；用户后续在其他机器构建并推送新 internal image。最终 image reference/digest 尚未知，禁止猜测或把旧 image 标为 qualified。
- D24 允许当前验证容器显式下载/安装确认缺失项。每次安装前后必须记录 exact command/source/version/path/exit status 和 live verification；selected source 失败即停止，禁止 automatic source/version switching、静默切 Python environment 或降低 `nsys`/`ncu` 门槛。该授权只用于当前 validation，不替代新 image 的 clean-container qualification。
- D26 明确 future replacement image 当前不可用且不阻塞 Gate B。当前工作应在 historical image 内完成 inventory、canonical env 选择、依赖安装和 live qualification；只有最终 release rehearsal 仍等待 immutable replacement image。
- `Echo-slowdown/environment.yaml` pins Python `3.9`，但 pinned `training_testing/prediction_api.py:9` 使用未延迟求值的 `str | None`。Exact Python `3.9.18` source import 已实测抛出 `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`；log bytes=`6409`，SHA256=`0d987466665b72a8c37b26aa50828d3c05a32480c9a17af0224e22f8fc6033a5`。这是 upstream source/environment contract mismatch，已满足 D26 创建独立 Python `3.10.x` env 的证据门槛。
- 固定 runtime routing：Task2 使用新建的 exact Python `3.10.x` Echo env；Task1/Task3 使用 `/opt/conda/envs/megatron_env` Python `3.9.18`。Sim-engine canonical PEP 604 annotations均有 `from __future__ import annotations`，无需 Python 3.10。两个 env 不是 fallback candidates，wrapper不得动态搜索或切换 interpreter。
- Worker 对 `repo.anaconda.com` 的 `conda search` 发生 connect timeout，而 CPU master 可访问同一 official HTTPS source。处理方式与 Ubuntu/PyPI cache一致：CPU master冻结 exact installer/wheels和 hashes，worker离线安装；不得自动切 source/version。
- 2026-07-17 cp310 offline qualification：`wheel_manifest_cp310.tsv` rows=`58`、bytes=`2,986,969,497`、SHA256=`d7743ee81f3bd0f8a900fd551321e5232abbf22b3191cf720130fc3ea659296c`；exact Python=`3.10.20` prefix 的 resolver/install/pip-check/import gates 均 PASS。CPU master 的 `CUDA_AVAILABLE=False`、`CUDA_DEVICE_COUNT=0` 只表示 live H800 qualification 尚未执行，不得提升为 B1 PASS。
- Environment probes: `ws-56153d316be61e0f-jlaunch-6t8kl` on `gpu-h800-0299.host.platform.shaipower.com`（inner probe exit `1` at torch import）；`ws-56153d316be61e0f-jlaunch-g8z9r` on `gpu-h800-0398.host.platform.shaipower.com`（inventory exit `0`）。
- `rlaunch` 没有只读 `status` 语义；不要运行 `rlaunch status <id>`。查询现有 job 使用 `brainctl get rjob/replica -n shai-core`，日志使用 `brainctl logs`；误建 job 按 handbook 用 `brainctl delete rjob <id> -n shai-core` 释放。
- 所有 scaling 脚本 export `CUDA_DEVICE_MAX_CONNECTIONS=1`。
- `update_pretrain_gpt.sh`/`realistic_run_gpt.sh` 里 `BASE_PATH` 默认硬编码旧路径（`/research/d1/.../Megatron-LM`），AE wrapper 必须显式注入。
- `realistic_run_gpt.sh` 硬编码 `NCCL_SOCKET_IFNAME=ens81f0`（AE 不用 realistic mode，仅注意别误用）。
- Echo-slowdown `global_config.json` 硬编码 `python_path=/root/miniconda3/envs/58test/bin/python`、`nsys_path` 等 → 用其自带 `update_configs.py` 运行时改写，不改 submodule。
- Echo-slowdown 采集需 `ncu`（Nsight Compute ≥2024.3）与 `nsys`（≥2024.4.2）在镜像内可用 —— dry-run 时验证，缺失则补进 setup 体系。
- **task2 需 2× GPU**：`slowdown_collection/input/train_script.py:24` `torch.cuda.set_device(rank)`，ws=2 双进程各绑一卡，NCCL 禁止双 rank 共卡 → 单卡原生不可行（2026-07-15 fact-check 结论，已获用户确认按 ws=2 执行）。
- sim-engine CPU-only 运行需 `SIMULATOR_HARDWARE_TYPE` env（`simulator_config.py:233` 的 nvidia-smi 探测会失败）。
- sim-engine `log/timeline_op_log/`、`./log/visualization_outputs` 从 CWD 写出 → task3 wrapper 控制 CWD 或收集产物。
- `tools/ae/setup_grouped_gemm_v1.sh:184-245` 当前先尝试 VCS install，失败后自动切换 exact-tag archive install；这与 No Fallbacks / Fail Fast 冲突。future implementation 必须要求调用方显式指定 `GROUPED_GEMM_SOURCE=vcs|archive`，所选来源失败即停止，禁止自动换源。
- `megatron-sim-engine/simu_main.py:200-202` 的 `--overlap-mode` 默认是 `auto`；AE Task3 必须显式传 `--overlap-mode on`，使 DDP-overlap metadata 缺失时直接失败，不能弱化为自动模式。
- `megatron-sim-engine/simu_main.py` 的 `--local-size` 表示模拟集群每节点 GPU 数，并进入 `ParallelGroupManager` / `RankManager` 的 node/local-rank 映射；canonical analytical backend `src/core/comm_sim/nccl_comm.py` 固定 `GPUS_PER_MACHINE = 8`，现有 scheduler presets 也以 8 为基线。AE Task3 因此必须显式固定 `LOCAL_SIZE=8`，并把它写入 topology manifest，不能依赖 CLI default。
- `megatron/profiler/trace_memory.py:8-11,64-75` 在 `pynvml` 缺失时让 tracker thread 直接返回，但 `stop_tracking()` 仍打印 `Data saved`；因此必须以 live NVML query 和实际非空 memory JSON 验证 image，不能只检查 package name 或 stdout。
- Session 15/18 的 `MemoryTracker` probe 失败来自 `megatron.profiler.__init__ -> communication_hooks -> megatron.core -> tensor_parallel.layers -> megatron.profiler.trace_decorator` package 初始化环，而非缺少 dependency。D27 只允许 B1 qualification probe 通过 isolated loader 直接加载 `trace_memory.py`；必须使用 canonical H800 cp39 interpreter、新 artifact root、live NVML/CUDA allocation 和 non-empty JSON assertions。该方式不得进入产品源码或替代 B2 对真实 package import/runtime path 的验证。

## Safety / boundaries
- 禁触 worktree branch `task/ddp-overlap-comprehensive-review-20260713`（其他 agent 正在优化 slowdown overlap 模块；worktree 路径 `/data/ycfeng/Megatron-LM-ddp-overlap-review-20260713`）。
- `rm`/`mv` 需用户显式授权；P0 commit 前不做任何清理性删除。
- Echo-slowdown 上游 `NetX-lab/Echo-slowdown.git`：不引入 AE 必需的本地 commit（D2）。
- `.gitignore` 现含 `profiler_log/`、`mg_scheduling_plan_log/`、`realistic_trace/` —— prebaked 产物入库需白名单条目。

## Key facts (探索结论 2026-07-15)
- Task1 产物：`profiler_log/<run_config>/`（per-rank txt）；dense 脚本只 trace 每 pp stage 1 个代表 rank，MoE 脚本默认全 rank 循环。Qwen 只读取 `FAKE_RANK_ORDER`，DeepSeek 只读取 `SCALING_FAKE_RANK_ORDER`，两个变量不可互换。
- Task2 产物：`training_testing/output/xgb_model.json` + `standard_scaler.json`；kernel-level model-agnostic，自带 micro-benchmark，不消费 task1 traces。
- Task3 输入：`--schedule-dir`（stage 级 plan）+ `--database-dir`（单卡 op 时长库）+ `--trace-dir`（仅 `--enable-slowdown` 需要）+ `--slowdown-assets-dir`（manifest/kernel_features/blueprints）+ 拓扑 flags；comm backend 注册名：`analytical` / `collective-sim`（现 CLI 默认）/ `profiling` / `cc-estimator`。
- mg_scheduling plan（`stage:<id>:`，duration=None，纯逻辑 1F1B，纯 Python 秒级、无 GPU）vs scaling trace（`rank:<id>:`，实测 duration + comm sub_operations）——前者给 timeline composer 定调度序，后者提供 per-rank 实测算子数据。
- sim-engine 现有输出：per-rank comp/comm/sum 三行 stdout + rank0 comp/comm operations JSON（`log/timeline_op_log/`）+ `sim load/execution time` stdout；无 fwd/bwd/opt 分解（P3 补）。

## Enhanced plan-review findings (2026-07-15)
- 当前主仓 branch 为 `overlap-tracing`（HEAD `a5bcd3d`）；独立 worktree `task/ddp-overlap-comprehensive-review-20260713` 位于 `/data/ycfeng/Megatron-LM-ddp-overlap-review-20260713`，本轮保持禁触。
- 主仓工作区已有 3 个 tracked 修改和多组 untracked 资产；计划审查不得把这些既有改动误判为本轮实现结果。
- AE draft `sc26-ad.tex:153` 明确承诺 T1 产物含 per-rank execution graphs、GPU memory usage reports、summary log；现有 plan acceptance criteria 只覆盖 traces/report，需补齐 memory 与 summary 验收。
- AE draft `sc26-ad.tex:112,131,143,147` 仍宣称 T2 单 GPU；已确认实际原生流程最少 2 GPU，必须列入 tex 建议与 AE README 的显式硬件 gate。
- slowdown assets 构建器已定位：`megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py`；I2 的未知项应收窄为该构建器的输入契约、与 Task1/Task2 输出的映射、以及三模型共用规则。
- nested submodule `megatron-sim-engine/src/core/cc_backend/collective-sim` 当前未初始化（`git submodule status --recursive` 前缀 `-`）；即使默认 backend 为 `analytical`，仍需验证公开 `clone --recursive` 是否会被其 URL/commit 可达性阻断。
- 根 README 只把 DeepSeek-V3-Proxy（MHA simplification）列为 stage-1 支持；AE 计划采用保留 MLA 的 DeepSeek-V3 variant，必须在 plan 中把“现有可用基线”和“待实现 AE profile”严格区分，不能将其视为已验证能力。
- `examples/pretrain_deepseek_v3_moe.sh:99-120,303` 已有 `MODEL_PROFILE=smoke`，参数正是 32L / hidden 2048 / 32 experts / top-2，并启用 `--multi-latent-attention`；因此计划中的“先新增 AE 缩配 profile 分支”是过期假设。剩余工作是把该既有 profile 作为 AE baseline 做静态契约测试和 GPU runtime 验证，而不是重复实现。
- `examples/update_pretrain_gpt.sh:105-108` 已有 GPT-175B 的 96L / hidden 12288 / 96 heads 配置；真正未满足项是当前执行参数仍硬编码 `--fp16`（约 line 275），以及 AE 所需 mock-data、bf16、DDP overlap、路径注入和输出归档是否可由 wrapper 安全控制。
- `examples/pretrain_qwen3_30b_a3b_moe.sh` 支持 `FAKE_RANK_ORDER`，`examples/pretrain_deepseek_v3_moe.sh` 支持 `SCALING_FAKE_RANK_ORDER`；两者均支持 `TRACE_MEMORY=1`。两者 `OVERLAP_GRAD_REDUCE` 默认均非强制开启，AE wrapper 必须显式设置并由静态/集成测试验证实际 argv。
- 现有配置测试主要针对 `examples/gpt175b_scaling_wallclock_scan.sh` 与 `examples/qwen3_a3b_moe_scaling_wallclock_scan.sh`，而当前计划拟包装 `update_pretrain_gpt.sh` 与模型主脚本；计划需先明确 canonical Task1 source scripts，并让测试路径与实际 AE 入口一致，避免验证错对象。
- 候选脚本矩阵对比：`gpt175b_scaling_wallclock_scan.sh` 的 1024-card 行是 pp=16/tp=8/dp=8，`qwen3_a3b_moe_scaling_wallclock_scan.sh` 的 256-card 行是 pp=8/tp=8/exp=4/dp=4；均不符合 AE 固定矩阵（GPT pp=8/tp=8/dp=16；MoE pp=4/tp=8/exp=8/dp=8），且二者没有 AE 必需的强制 DDP overlap + memory-report 契约。因此它们适合作为 rank-selection/配置测试参考，不适合直接成为 AE canonical entry。
- 当前最小改动方向是：MoE wrappers 调用 `pretrain_qwen3_30b_a3b_moe.sh` / `pretrain_deepseek_v3_moe.sh` 并显式注入 AE env；GPT wrapper 调用 `update_pretrain_gpt.sh`，只对其补充可测试的 precision/trace-memory/DDP-overlap/output knobs。此方向仍需在最终计划中列明精确修改面和 RED→GREEN 测试，不得在本轮实施。
- Echo-slowdown 的 `update_configs.py` 会原地改写 4 个 tracked `global_config.json`；`run_all.sh` 还会覆盖多项 tracked input/output 资产。直接在 pinned submodule 中运行会污染 AE checkout，并使 3 个 per-model Task2 wrappers 相互覆盖。计划需选择隔离的工作副本/工作目录契约，或明确接受并验证可恢复的 in-place 行为；当前未决。
- `build_ddp_slowdown_assets.py` 的强制输入为：Task1 trace directory、与 CMD NVTX label 对齐的 Nsight Systems SQLite、NCU metrics CSV、model 和 scaler；输出才是 `manifest.json` / `kernel_features.json` / `backward_kernel_blueprints.json`。因此 Task2 的 predictor 两文件本身不足以供 Task3 开启 slowdown。
- 现有 `tests/e2e/test_ddp_slowdown_simulate_smoke.sh` 明确通过 `--trace-kernel-ground-truth` + Nsight Systems capture 生成 SQLite，再用 Echo-slowdown 的 NCU CSV 构建 assets；这条已验证链应成为 AE 计划的接口依据。
- 关键 pipeline gap：当前 Task1 计划只承诺 execution graph + memory report，未承诺 trace-compatible Nsight SQLite；fresh Task1 + fresh Task2 无法直接构建 trace-specific blueprints。Task3 若混用 prebaked blueprints 与 fresh traces，`cmd_uid`/rank/batch 映射可能不一致。必须在计划阶段明确 fresh/prebaked provenance 和 compatibility gate。
- Builder strictness 已核实：`collect_required_kernel_names()` / `build_blueprints()` 对 trace dir 内每个 backward `cmd_uid` 都要求同一 SQLite 中存在 parent NVTX range、phase=compute windows、rank/stage/batch 一致性及 NCU kernel features；sim-engine 初始化也会对所有 trigger cmd_uids 检查 blueprint，缺失即 fail-fast。因此不能把任意 prebaked blueprint 与 fresh trace 混用。
- 已验证模式可复用：`tests/e2e/test_ddp_slowdown_simulate_smoke.sh` 用 `nsys profile ... bash run_all_scaling_ranks.sh` 包住所选 fake-rank loop，从同一 capture 导出 SQLite，再收集这些 ranks 的 trace。AE 计划若选择 fresh slowdown path，应明确复用这一捕获边界和 trace-set 原子性。
- `simu_main.py` 的 `--cc-backend` CLI 默认仍是 `collective-sim`，而 AE 决策为 `analytical`；所有 Task3 scripts 必须显式传 `--cc-backend analytical`，不能依赖内部 dataclass 的另一个默认值。
- `simu_main.run_simulation()` 当前只返回 `world_size`、`load_time`、`execution_time`，尚未返回 rank0 step/fwd/bwd/optimizer 数据；reporter 计划必须描述如何从同一个 `SimulatorEngine` / timeline manager 实例提取指标，避免事后解析 stdout 或读取不稳定的 CWD logs。
- 顶层 `mg_scheduling/*.py` 与 `megatron-sim-engine/src/scheduler/mg_scheduling/*.py` 仅部分文件哈希相同；核心 `mg_test.py`、`mg_scheduling_plan.py`、`schedules.py`、`training.py` 已漂移，不能假设输出等价。现有 slowdown e2e 使用 sim-engine 内置版本，故它是更强的 canonical 候选，但仍需用目标三模型配置验证 schedule structure/counts。
- Reporter 不能复用现有 `visualize_timelines()` 的 `comp_time + comm_time` 作为 one-step time：该和在 overlap 下重复计时，而且日志生成依赖 visualization side effect/CWD。稳定口径候选是直接读取 `timeline_manager.stages_timeline_process_dict[0].final_merge_timeline`，以 `max(finish_time) - min(join_time)` 得到 rank0 timeline span。
- forward/backward/optimizer 口径候选：在 rank0 `comp_timeline` 中按 exact operation name 分组，对每个 op 的 `finish_time - join_time` 求和。它们应在 JSON/Markdown 中标为 scheduled operation-duration sums；不得宣称三者之和等于 one-step time（pipeline idle、其他 comp、comm 与 overlap 均会造成差异）。
- D19/D20 已冻结 `simulator_wall_clock_s = load_time + execution_time`，同时保留 `load_time_s` 与 `execution_time_s` 诊断字段；report generation/visualization 不计入。rank0 timeline span 缺失时 fail fast，`comp+comm` 只能作为 diagnostic，不能替代主值。
- 公网 pinned-commit 可达性已用独立 `/tmp` shallow fetch 验证：Echo-slowdown `1390b4416ded08bc1b9cd0620d329d81d4470bf9` 可直接 fetch，且为 upstream `main`/`HEAD`；megatron-sim-engine `2044cccc8fff222172b7f91571a617886841001f` 为公开 `overlap-tracing` ref tip；nested collective-sim `6e06e3f5140cd4e2e7c12a35586ebcdc0f410df0` 可从公开 `ft-cc` history fetch，且为该 branch 祖先。当前 `clone --recursive` 的 commit-availability 风险已消除；未来 sim-engine `sc26-ae` commit 仍须在 fresh-clone gate 复核。
- 当前仓内没有计划中的三模型 prebaked SQLite/slowdown-assets bundle，故不能凭现有文件决定 D11 的最终分发介质。现有可量化参考仅为：34 个 trace txt 合计 0.165 MiB（均值 4.960 KiB），现有 `xgb_model.json` 为 0.592 MiB；这些数值不代表 AE 全量 capture。正式 dry-run 后必须生成逐文件 size manifest，再选择 regular Git、Git LFS 或 GitHub Release asset。
- GitHub 官方限制（`https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github`）：单文件超过 50 MiB 会警告，超过 100 MiB 被 regular Git push 阻止；GitHub 官方建议超限时使用 Git LFS，或用 GitHub Release 分发大文件。因此“全部 prebaked 直接 regular Git commit”必须带 size gate，不能无条件写死。
- `build_ddp_slowdown_assets.py` 当前把 `model_path`、`scaler_path`、`source_trace_dir`、`source_nsys_sqlite`、`source_ncu_metrics` 原样写入 `manifest.json`；若调用方传入绝对路径，prebaked bundle 搬到 AE 机器后这些路径不会继续可执行。sim-engine 已提供 `--slowdown-model-path` / `--slowdown-scaler-path` override，因此 Task3 不应把 builder manifest 中的生成机路径当 runtime locator。增强计划需要求 AE 外层 provenance manifest 保存 bundle-relative paths、SHA256、source commits/topology/profile，并让 Task3 始终显式传入 bundle 内 model/scaler 的实际路径；是否同时最小修改 builder 以写 relative paths，交由独立计划审查判断。
- Task1 fake-node-size 存在源脚本差异：`examples/update_pretrain_gpt.sh` 固定 `FAKE_GPUS_PER_NODE=8`，而 `examples/pretrain_qwen3_30b_a3b_moe.sh` 与 `examples/pretrain_deepseek_v3_moe.sh` 传 `--fake-gpus-per-node "${FAKE_WORLD_SIZE}"`。StepCode Claude 独立审查已裁决无需修改 MoE source scripts：tracer trace 不序列化 `local_rank` / `server_id`，sim-engine 又从自己的 `--local-size` 重建 node mapping。计划如实保留 `capture_runtime.fake_gpus_per_node`，并独立固定、测试 `simulation_topology.local_size=8`。
- `megatron/profiler/cmd.py`、`megatron/training/training.py` 与 `megatron/profiler/trace_memory.py` 分别把 scaling trace、replay cache、memory JSON 写到 CWD-relative `profiler_log/` / `memory_traces_scaling/`；若 Task1 重用同一 CWD，旧文件可被新 manifest 误收集。计划因此使用新的 `task1/runs/<capture_id>/runtime/` CWD，并在完整验证后才发布 marker。
- 当前 slowdown e2e 同时传 `--trace-dir "${TRACE_DIR}"` 与 `--database-dir "${TRACE_DIR}"`。Task3 的 operation database 与 slowdown trace 必须绑定到同一 canonical resolved directory；独立 `DATABASE_DIR` 会引入未定义来源，已在计划中禁止。
- `megatron/training/arguments.py:312-313` 在 `overlap_grad_reduce && do_trace` 时自动设置 `trace_ddp_grad_overlap=True`。因此 Task1 计划只需强制并测试 `DO_TRACE` 与 `OVERLAP_GRAD_REDUCE=1` 的最终 argv/metadata，不需要为三个 model source scripts 重复新增显式 `--trace-ddp-grad-overlap` 参数。
- `RankManager` 的 `fake_gpus_per_node` 静态消费面只定位到 tracer 内存中的 `RankZoo.local_rank/server_id`；未发现这些字段写入 Task1 trace 或被 sim-engine 读取。独立审查据此关闭 MoE source-script 修改候选，Gate B 只需验证现有 trace 能被 `LOCAL_SIZE=8` 的 Task3 消费。
- `docs/ae/grouped_gemm_v1_setup.md` 仍描述 VCS 失败后自动 archive recovery。future implementation 修改 installer 为显式 `GROUPED_GEMM_SOURCE` 时，必须原文件同步更新，不能留下与 fail-fast contract 冲突的 reviewer 文档。
- rank0 reporter 若只对 exact-name operation 过滤求和，缺失 operation 会自然得到 `0` 并掩盖不完整 schedule/timeline。计划已把 `forward_step`、`backward_step`、`optimizer_step` 各至少出现一次列为 fail-fast invariant。
- `mg_test.py --model-size` 的 parser 是无 `choices` 的任意字符串；`mg_scheduling_plan.py` 只把它用于 legacy schedule directory label。AE 可直接使用 `gpt175b|qwen3_a30b|dsv3`，不需要 model-size mapping，且不得从该 label 推导 architecture 参数。
- `simu_main.py` 的 canonical import 是 `src/core/simu_engine.py`，不是 `simu_engine1.py`。canonical engine 的 `GLOBAL_MG_DIRECT_MAPPING_LIST` 包含 `optimizer_step`，现有 slowdown PP=1 smoke 的手写 schedule 也包含该 op；所以 Gate B 缺失 `optimizer_step` 应阻断，而不能归因于 PP=1。Phase 4 仍增加 scheduler-generated PP=2 integration，防止只验证手写 schedule。
- `analytical` backend 在 `src/core/comm_sim/nccl_comm.py` 固定 `GPUS_PER_MACHINE=8`，主 simulation path 不调用 setter。计划增加 runtime fail-fast/test，要求 `config.local_size == GPUS_PER_MACHINE == 8`，不通过修改 global 来掩盖 mismatch。
- Prebaked payload 的 producer main-repo commit 不可能强制等于最终 consumer `HEAD`：把 payload/locator commit 进仓本身会改变 HEAD。校验应绑定 distribution manifest、nested manifest hashes、producer commits 与 payload hashes 的内部一致性；fresh 路径仍绑定当前 producer checkout。
- D21 的“最终分发文件”包括 nested artifact manifests 与 `distribution_manifest.json` 本身；size gate 必须扫描完整 staged regular-Git candidate，而不是只累加 payload manifest 中的文件。
- 当前没有 Task3 CPU peak RSS 实测，`32 GiB` 不是可引用的最低内存证据。README 的 CPU memory 声明必须等待 Phase 8 记录三模型 peak RSS 与实际成功的 host-memory allocation。

## Gate B static reconciliation (2026-07-16)
- `megatron/training/arguments.py:1974-1990` 的 scaling defaults 为 warmup `3`、profile `1`。Qwen 与选定 GPT source 未覆盖这两个参数，故继承 `3/1`；`examples/pretrain_deepseek_v3_moe.sh:45-46,469-470` 显式采用 `0/3`。D25 已冻结共同显式 pair 为 warmup `3`、profile `1`；三个 wrapper 必须同时传入并记录这两个值，禁止依赖 source defaults。
- `examples/pretrain_qwen3_30b_a3b_moe.sh:237,281` 会先从 `COMMON_ARGS` 传 `--global-batch-size "${GLOBAL_BATCH_SIZE}"`，再在 scaling override 中传计算值 `NUM_MICBATCH × MICRO_BATCH_SIZE × FAKE_DP`。AE runner 必须记录所有重复 critical flag values 与最终 effective GBS；只要重复值冲突即 fail fast，不能依赖 argparse 的 last-value behavior。
- Pinned Echo commit `1390b4416ded08bc1b9cd0620d329d81d4470bf9` 包含以下 `11` 个 tracked historical generated/runtime artifacts，full `git archive` 会把它们带入每次新 snapshot：
  - `merge/input/kernel_metric_output.csv`
  - `merge/input/slowdown_stats_output_device_0.xlsx`
  - `merge/output/merged_features.csv`
  - `training_testing/input/test_csv/merged_features.csv`
  - `training_testing/input/train_csv/merged_features.csv`
  - `training_testing/output/prediction/feature_importance_merged_features.png`
  - `training_testing/output/prediction/output_df_merged_features.csv`
  - `training_testing/output/prediction/output_full_df_merged_features.csv`
  - `training_testing/output/prediction/output_metrics.txt`
  - `training_testing/output/train_dataset.csv`
  - `training_testing/output/xgb_model.json`
- `Echo-slowdown/training_testing/predict.py:1-20` 仅加载 model/scaler、执行一次 prediction 并 `print(result)`；它不会生成上述 tracked `training_testing/output/prediction/*`。因此这些历史文件不能作为 current-run evidence。Task2 snapshot 必须从 archive extraction 阶段排除已声明 output prefixes，并在 upstream tracked-source inventory 新增匹配项时 fail fast，而不是运行后 `rm`/`mv`。
- `Echo-slowdown/run_all.sh:16-41` 只有 module start/completion markers，没有 timestamps。Gate B B3 只能记录 wrapper 测得的 total elapsed seconds 和 log markers；不得声称已有 per-module elapsed data。

## Gate B1 D28 operational notes (2026-07-17)

- Current split verdict is authoritative: D27 one-H800 MemoryTracker=`PASS`; Echo exact-two-H800=`BLOCK`; integrated B1=`BLOCK`.
- D27 produced one real H800, CUDA/NVML device counts `1/1`, `30` samples, and a non-empty `4,951`-byte memory JSON. This closes only the qualification-only MemoryTracker branch; B2 still owns the real package import/runtime path.
- Echo Attempt0/Retry1/Retry2 are immutable failed roots. Retry2 proved exact-two-H800 visibility and real training but did not complete numeric parity. The recovery CPU integration passed, but it is not a substitute for final live exact-two-H800 evidence.
- The recovery incidents are permanent audit facts: duplicate CPU execution, Attempt-1 exit `143`, unauthorized deletion of four Attempt-1 evidence paths, corrected source binding, an invalid early `bash -lc 'true'` predict-only, and an unauthorized live RJob that was created/scheduled and then stopped before its payload.
- The old at-most-one live budget is consumed. D28 creates a different budget that is conditional and currently unconsumed; it is not a retroactive approval of the interrupted submission.
- The next execution-stage root must be new and distinct from all prior/recovery roots. The historical pointer `logs/sc26_b1_echo_two_gpu_latest_path.txt` is immutable and must not be rewritten.
- Before using the D28 live budget, a new predict-only must be fully bound to the exact live image, `/data:/data` mount, workdir, clean root, fixed cp310 interpreter, isolated pinned Echo source, qualification helper/payload, and exact resources. Process and semantic exits must both be `0`, quota markers must be absent, and at least one H800 candidate must expose `available_gpu_count >= 2`.
- If the fully-bound predict-only passes, exactly one final exact-two-H800 live attempt may run. A new root-cause class, contract drift, missing evidence, or failure stops immediately; no retry, source/version switch, partial pass, or calibration factor is allowed.
- Team `sc26-ae-gate-b1-recov-65f35581` is now `missing` because worker-2 invoked `orphan-cleanup` while tasks were pending. Do not reconstruct task JSON or attribute native Lane C to dead worker-3. Lane C reviewer identity is `/root/verifier_lane_c`.
- B2/B3/B4 remain blocked until integrated B1 passes. Phase 1–9 remain blocked until Gate B passes. This plan-review stage must not create the D28 root, run predict-only, submit an RJob, or start product implementation.

## D29 test-issue autonomy notes (2026-07-19)

- A test-shaped failure is self-repairable only when its root cause is confined to an AE test, audit, schema, validator, documentation, or control-plane orchestration surface and the repair directly advances the one-click scripts or reusable pre-dataset.
- Each self-repair must follow: reproduce RED; identify root cause; apply the smallest contract-preserving fix; observe GREEN; run affected regression tests; record command, exit status, and numeric evidence. Do not hide a failure by weakening assertions or changing the acceptance threshold.
- D29 does not permit fallback/source switching, checksum or provenance bypass, synthetic-to-real relabeling, or changes to real-vs-synthetic boundaries. Real GPU/image/quota/scheduler, product/runtime/workload correctness, real pre-dataset quality, security, destructive, and external-publication failures remain hard blocks.
- Current evidence remains incomplete for real AE release: D27 one-H800=`PASS`; Echo exact-two-H800=`BLOCK`; integrated B1=`BLOCK`; fresh real pre-dataset=`NOT QUALIFIED`. Local test repairs may continue without changing that status.

## D30 latest test-failure autonomy notes (2026-07-19)

- The latest user instruction supersedes D29's narrow root-cause scope: any problem exposed by a
  test, validation, rehearsal, audit, or qualification check may be diagnosed, decided, and repaired
  autonomously when it directly serves the one-click AE shell entries or reusable pre-dataset.
- This includes a task-scoped implementation or runtime-control repair when the failing check proves
  it is required. It does not include weakening assertions, changing acceptance thresholds, bypassing
  checksum/provenance/clean-source checks, adding fallback/source switching, or relabeling evidence.
- A failed real/data-quality check remains an unmet gate until the underlying defect is fixed and the
  check passes. Actual external resource/authority problems, destructive or irreversible actions,
  external publication, and materially scope-changing refactors remain outside this autonomy lane.
- Every D30 repair records motivation, expectation, observed RED/root cause, minimal method, GREEN,
  affected regression commands and exit codes, numeric evidence, and the resulting evidence class.
## Operational Note — 2026-07-19 Temporary Root

The controller's /tmp inode allocation is exhausted (IUse=100%) even though byte capacity remains.
For all SC26-AE test/rehearsal commands use:

    export SC26_AE_TMP_ROOT=/data/ycfeng/sc26-ae-test-tmp
    export TMPDIR=/data/ycfeng/sc26-ae-test-tmp

The affected fixtures now honor this contract. Do not remove files from /tmp without explicit
permission. This environment note does not change the real qualification or release gates.

## Session 45 operational notes — control-plane audit

- The current outer `HEAD` does not contain the untracked `SC26-AE/` producer overlay. Treat all
  local Task1/Task2/Task3 scripts and tools as non-release bytes until a tracked/snapshotted source
  identity is established.
- The nested `megatron-sim-engine` is currently clean and matches the outer gitlink; this closes
  only the current I39 discrepancy and does not repair the separate AE producer provenance gaps.
- Do not start GPU/RJob or external publication while I51-I58 and CR-01 remain open. A real run
  would produce evidence that the current control plane cannot yet classify or hand off safely.
- The fixed temporary root remains `/data/ycfeng/sc26-ae-test-tmp` via `SC26_AE_TMP_ROOT`/`TMPDIR`
  for local tests; do not use inode-exhausted `/tmp` templates.

## Session 56 operational notes — V21 clean-clone provenance

- The independent StepCode Claude review identified a real clean-clone defect: the V21 verifier
  resolves required supplemental and marker evidence from the task `logs/` directory, while the
  repository-level `logs/` ignore rule excludes those files by default.
- D31 authorizes tracking only the verifier-required log set. Keep unrelated historical/runtime
  logs ignored; do not unignore or stage the entire task log directory.
- The pre-reconciliation required set contains `59` log files totaling `228,172` bytes: `58`
  supplemental/marker logs plus the sole current V21 verifier identity. Replacing the current
  verifier identity during reconciliation must preserve the exact required-set cardinality rather
  than retain both old and new current identities.
- The independent reviewer created
  `megatron-sim-engine/.omc/state/sessions/97724d0a-b8ba-42b6-9d32-779f923ff2d7/last-tool-error-state.json`
  while attempting to read a main-repository entry point from the nested working directory. D32
  authorized deleting that transient runtime state. It was removed, and the nested HEAD and outer
  gitlink both remain `39755169f73f6c748e8d7376c3a2158c6569436b` with an empty nested status.
- Do not use ignore rules to conceal future nested runtime state. Any new dirty path must be
  diagnosed and resolved explicitly before the provenance gate.
- Historical test transcripts and Markdown archive records may contain whitespace that is part of
  the recorded bytes. The root `.gitattributes` exemption is deliberately limited to this one task
  archive; do not extend it to `SC26-AE/`, tests, tools, or general source.
- V21 source enumeration must exclude `SC26-AE/output/`. That directory contains ignored runtime
  copies and may exist on an exercised controller but not in a clean clone; including it makes the
  static scope state-dependent. The current tracked source scope is shell `47` and Python `36`.
- Candidate and penultimate tracked-snapshot V21 runs passed with exit `0`, and the independent
  follow-up verdict is `APPROVE`. The penultimate tree is
  `6c5cf790c62b021e1504621ae7489986a29990ec`; its ephemeral snapshot commit is
  `26f89b4df53760df8c38ac9ab62bfcf4ff0d6349`. This closes I59 locally. Final authoritative
  hashes/current identity, exact-log restaging, the final staged replay, local Lore commit, and
  actual committed-clone replay remain required before reporting the provenance checkpoint complete.
- A root `.omc/` directory already existed with two small session error records dated 2026-07-18;
  it is excluded by the repository's Git info exclude, contains `881` bytes, and is not staged. It
  is distinct from the D32-authorized nested path and was not removed. The nested
  `megatron-sim-engine/.omc` remains absent.

## Session 57 operational notes

- Temporary root: `/data/ycfeng/tmp`; never use `/tmp` for files, logs, or caches.
- Heavy `brainctl get replica` calls require `timeout 60s systemd-run --scope -p MemoryMax=2G`.
- Clean producer: `/data/ycfeng/sc26_ae_task3_qwen`.
- Shared output root: `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4`.
- Shared Task2 predictor is immutable and must not be rerun for missing Task3 kernels.
- StepCode review temporary evidence: `/data/ycfeng/tmp/stepcode_sc26_task2_compat/`.
