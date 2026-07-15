# Issues — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes        |
|------------|---------------------------|
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
task2 依赖 ncu ≥2024.3 / nsys ≥2024.4.2；AE 镜像内版本未验证。缺失时 setup 体系需补装（CUDA toolkit 自带版本可能过旧）。

### I8. submodule pinned commit 公网可达性
**当前 pinned commits 已验证可达。** Echo-slowdown `1390b441...`、sim-engine `2044cccc...` 可从各自公开 URL 直接 fetch；nested collective-sim `6e06e3f...` 可从公开 `ft-cc` branch history fetch。剩余 gate 仅为：sim-engine 后续 `sc26-ae` commit push 后，必须在 clean temporary path 实测 `git clone --recursive`，并核对主仓 gitlink 指向公开可取的 commit。

### I9. rank0 report 的计量语义与 JSON schema
**Resolved by D19+D20.** 主值从内存 timeline 计算 rank0 span；fwd/bwd/optimizer 是 exact op-name scheduled duration sums；wall-clock=load+execution。`comp+comm` 仅输出为 `rank0_comp_plus_comm_diagnostic_ms`，不能替代主值；timeline/span 缺失或非法必须 fail fast。完整 JSON schema、non-negative/finite invariants、empty timeline/error branches 和 unit/e2e assertions将在增强 plan 中固定，不再需要产品决策。

### I10. prebaked artifact 的分发介质与 size gate
**Resolved by D21, runtime measurement pending.** 正式 dry-run 必须产出逐文件 size/SHA256 manifest。最终文件全部 <50 MiB 且 bundle <=500 MiB 时使用 regular Git；否则使用 GitHub Release assets，repo 只保存 manifest/checksum/下载入口。该规则是确定性 release gate；实际介质仍需在生成真实 artifacts 后按规则判定并记录，不允许静默切换。

### I12. D22 automatic fallback 与 No Fallbacks / Fail Fast 冲突
**Resolved by D23 (supersedes D22).** 不实现 automatic fallback 或三态 auto-selection。每次 Task3 必须显式传 `ARTIFACT_SOURCE=fresh|prebaked`；所选来源缺失、partial、checksum/provenance 不符均 fail fast。

### I13. slowdown-assets manifest path portability
`megatron-sim-engine/tools/data_prep/slowdown/build_ddp_slowdown_assets.py` 当前会把调用时的 model/scaler/trace/SQLite/NCU 路径原样写入内部 `manifest.json`；若这些是生成机绝对路径，搬运后的 prebaked bundle 不能依赖它们定位 runtime files，provenance 也不够完整。增强计划必须冻结以下最低契约：(a) Task3 显式传 bundle 内 `--slowdown-model-path` 与 `--slowdown-scaler-path`；(b) AE 外层 manifest 使用 portable relative paths，并记录每个文件 SHA256、source commits、topology、profile 与 artifact source；(c) builder 内部 source fields 仅作原始 provenance，不能被 Task3 当 runtime locator。是否还要修改 builder 让其内部 manifest 使用 relative paths，等待独立 Claude 计划审查；在结论形成前不得加入临时路径 rewrite 或 fallback。

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

## Resolved

### R-I16. Task3 `LOCAL_SIZE` 与 Task1 fake-node-size 契约
**Resolution (independent StepCode Claude WATCH adjudication):** 不修改两个 MoE Task1 source scripts。Tracer 的 `fake_gpus_per_node` 只影响内存中的 `RankZoo.local_rank/server_id`，这些字段未写入 Task1 trace；sim-engine 使用自己的 `--local-size` 重建 node/local-rank mapping。Manifest 分别记录实际 `capture_runtime.fake_gpus_per_node` 与固定的 `simulation_topology.local_size=8`，Gate B 验证跨值消费路径。

### R-I22. Scheduler `--model-size` 值域
**Resolution (static code fact):** `mg_test.py` 接受任意 string，`mg_scheduling_plan.py` 只将其拼入 legacy 输出目录名；`run*.sh` 中的 numeric/Mixtral 分支不是 Task3 直接调用路径。AE 直接传 `gpt175b|qwen3_a30b|dsv3`，Task 4.1 做 exact-label tests，不新增 mapping 或隐式 architecture lookup。

### R-I11. Task2 会污染 pinned Echo-slowdown checkout
**Resolution (D17):** 不在 submodule 原地运行。每个实际 collection run 从 pinned commit 通过 `git archive` 建立 `SC26-AE/output/_work/` 隔离快照，在快照内运行 `update_configs.py`/`run_all.sh`，再归档 canonical outputs 和 provenance manifest。验收必须同时证明 submodule `git status --short` 为空、manifest source commit 等于主仓 gitlink、三模型 marker 均指向同一 verified predictor bundle。
