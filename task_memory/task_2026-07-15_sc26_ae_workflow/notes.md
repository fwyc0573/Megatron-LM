# Notes — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes            |
|------------|-------------------------------|
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
- AE image: `hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef`；镜像补齐入口 `tools/ae/setup_grouped_gemm_v1.sh`（幂等，manifest at `$STATE_DIR/manifest.env`；装 grouped_gemm v1.0 + absl-py==2.3.1；期望 Python 3.9.18 / torch 2.1.2 / CUDA 12.1，失配即 fail-fast）。
- GPU worker: 按 `/data/ycfeng/stepfun-env-handbook/guidence.md`，rlaunch 验证组合 `--charged-group=codesign --private-machine=group --positive-tags=h800`，`--backoff-limit>0`，大额申请前 `--predict-only`。
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
- `simulator_wall_clock_s` 候选定义为 `load_time + execution_time`，同时保留 `load_time_s` 与 `execution_time_s` 诊断字段；report generation/visualization 不计入，确保值可复现。该 paper-facing schema/语义需 grilling 确认。
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
