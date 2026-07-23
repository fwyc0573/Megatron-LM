# Requirements — SC'26 AE Workflow (task_2026-07-15_sc26_ae_workflow)

## Modification History

| Date       | Summary of Changes                                       |
|------------|----------------------------------------------------------|
| 2026-07-23 | Captured the single canonical branch and final artifact/script inventory request |
| 2026-07-23 | Captured the reduced two-model functional AE scope, Task2 non-rerun rule, missing-kernel slowdown policy, and `/data/ycfeng/tmp` runtime rule |
| 2026-07-20 | Captured D31 approval to track exactly the V21-required evidence logs and D32 approval to delete the reviewer-generated nested `.omc/` runtime state |
| 2026-07-19 | Added D30: latest user instruction makes all test/validation/rehearsal failures autonomous when they serve the AE scripts and reusable pre-dataset, without relaxing release gates |
| 2026-07-19 | Clarified the test-autonomy gate across all AE-serving test/control-plane surfaces and retained real qualification hard blocks |
| 2026-07-18 | Captured D29 authorization for autonomous test-issue repair in service of the AE workflow |
| 2026-07-17 | Captured D28 authorization for one clean B1 retry after the interrupted unauthorized submission |
| 2026-07-17 | Captured D27 MemoryTracker qualification probe-only remediation selection |
| 2026-07-16 | Captured D26 current-container environment remediation and execution-continuation directive |
| 2026-07-16 | Captured D25 common scaling warmup/profile policy |
| 2026-07-16 | Captured D24 new-image remediation and current-container installation authorization |
| 2026-07-16 | Recorded explicit Gate A approval and authorization to complete the reviewed plan with team/subagent acceleration |
| 2026-07-15 | Captured D23, superseding D22 with explicit source selection |
| 2026-07-15 | Captured enhanced-review grilling decision D22 and rule conflict |
| 2026-07-15 | Captured enhanced-review grilling decision D21           |
| 2026-07-15 | Captured enhanced-review grilling decision D20           |
| 2026-07-15 | Captured enhanced-review grilling decision D19           |
| 2026-07-15 | Captured enhanced-review grilling decision D18           |
| 2026-07-15 | Captured enhanced-review grilling decision D17           |
| 2026-07-15 | Captured enhanced-review grilling decision D16           |
| 2026-07-15 | Initial capture of raw user intent from grill-me kickoff |

> Scope note: this file records RAW USER INTENT and interactive Q&A follow-ups only.
> No implementation strategy or execution plan lives here (see plan.md).

---

## R1. Overall AE delivery model
[Original Request] 筹备 SC'26 的 AE 流程：提供镜像 + GitHub repo，AE 人员 clone 仓库 + pull 镜像复现。流程：(1) 将 clone 的仓库 mount 进启动的镜像，运行提前准备好的 AE shell scripts 补充镜像缺失的环境配置；(2) 执行具体 task 的 shell scripts（共 3 个 task，依据 `2026-SC-first-submission/sc25-ad-ae/for-paper-authors/sc26-ad.tex` 的规划）；(3) 将输出产物集中到一个目录，方便 AE 人员查看。

## R2. Output artifacts to collect (centralized directory)
[Original Request] 集中输出目录需包含：
- workload tracer 捕获的 execution graphs 文件；
- slowdown module 收集的数据和训练模型的权重；
- e2e simulation engine 完成的 report——默认报告 rank0，需包含：模拟得到的 one step time、one step time 的 forward / backward / optimizer 分别耗时、模拟器的 wall-clock time。

## R3. Current phase
[Original Request] 当前为 plan 和讨论阶段，充分探讨和分析、确认处理细节，先将 docs 落地。

## R4. Paper understanding scope
[Original Request] paper 位于 `2026-SC-first-submission`，需充分理解，特别是 workload tracer、timeline composer、kernel slowdown module；communication module 在本次 AE 中为弱验证（设备与 benchmark 限制），用其他 comm backend 服务 comm op 的预测，论文 comm 方法论暂不验证。

## R5. Naming / terminology conventions
[Original Request] system name 有两个：echo 与 moye，指同一对象。ep 一般指 embedding parallel 而非 expert parallel；exp 代表 expert parallel（具体依 codebase 上下文判断，可能存在混淆）。

## R6. AE draft as design reference
[Original Request] `sc26-ad.tex` 已给出 AE 流程初稿，需参考它设计和组织测试 shell scripts；按最 practical、最方便 AE 人员操作/调试的方式组织自动化 scripts，支持一键运行和查阅结果，减少 AE 人员额外工程修改与测试。

## R7. Tracer 基础设施（已有）
[Original Request] `Megatron-LM/megatron` 已包含魔改机制：单卡 sequential 运行各 rank 的 train step，记录 comp 信息、comm 元数据和 overlap 相关数据（txt），供 `megatron-sim-engine`（timeline composer）构建 timeline；workflow 参考 AGENTS.md。

## R8. Slowdown 模块（已有）
[Original Request] kernel slowdown dataset 与 predictor 用 `Megatron-LM/Echo-slowdown` 收集和训练；训练完成后在 `megatron-sim-engine` 中被使用（ddp overlap）。

## R9. 分支管理
[Original Request] 为 Megatron-LM 及引入的 submodule 创建 new branch `sc26-ae` 并 checkout；在开始 SC'26-AE workflow scripts 设计前，先充分运行 3 个关键 task，理解它们如何协调与串联。

## R10. mg_scheduling 与 scaling-mode 产物的区分
[Original Request] simulator e2e 模拟前可能需用 `mg_scheduling` 生成 pp stage 的 execution plan 文件。区分：mg_scheduling 产物规定不同 pp stage 的执行流程（几次 forward/backward 等 high-level 调度 plan）；examples/ 下 scaling-mode shell 产物规定并细化每个 rank 的细节（comp、comm 等）。需结合 codebase 充分理解。

## R11. 任务关注点与优先级
[Original Request] 关注 workflow 的通畅性、完整性、合理性和可用性（AE 人员按 doc 和 task shell 逐一完成 3 个 task）。不关注模拟结果与 groundtruth 的接近度，数值正常级别、符合逻辑即可。最高优先级：用 scripts 组织串联不同模块——task1: 指定 model、GPU 规模的 workload tracing；task2: 指定 model 的 kernel slowdown dataset collection 和 predictor 训练；task3: 在 sim engine 中用 task1 workload files + task2 predictor 做 e2e 模拟。暂不考虑模块重构（除非确认重大错误/bugs），只做 workflow 串联与文件调用。允许规范 output files（tracer workload files、predictor）的位置（如统一存入 SC 专用目录）。

## R12. 模型与规模矩阵（3 model × 3 task = 9 scripts）
[Original Request] task shell scripts 需覆盖 3 个 model 的完整测量与验证，每个 model 每个 task 一个 script（共 3×3）：
- MoE：qwen3-a30b、deepseek3-variant；256 卡（tp=8, pp=4, dp=xx, exp=xx——dp 与 expert parallel 度由 agent 决定）；
- Dense：gpt175b；1024 卡（pp=8, tp=8, dp=16）；
- 统一 bf16、mock-data 模式、开启 ddp overlap。

## R13. 最小代价原则
[Original Request] 用最小代价完成任务，减少不必要的代码模块修改和开销；第一优先级是提供给 AE 人员的 shell scripts 可用、易用。允许将 3 个 task 的 scripts 拆分成多个（如 dense 与 moe 分开避免混淆），但要明确各自目的，并在 doc 中清晰给出使用说明。

## R14. 镜像与环境补齐
[Original Request] AE 默认镜像：`hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef`；缺失的库包和环境补充到 `Megatron-LM/tools/ae/setup_grouped_gemm_v1.sh`，一键配置安装。

## R15. 禁触分支
[Original Request] worktree branch `task/ddp-overlap-comprehensive-review-20260713` 正在被其他 agents 优化 slowdown overlap 模块，不要修改该 branch。

---

## Q&A Follow-ups (grilling session, 2026-07-15)

### D1. Repo 发布形态
[Original Request] 整仓推到 `github.com/fwyc0573/sc26-reproduce`：当前 Megatron-LM 仓库的 sc26-ae 分支整体 push 为该 repo 的默认分支，`SC26-AE/` 脚本目录建在 repo root，与 sc26-ad.tex 承诺的 URL 一致。

### D2. Echo-slowdown submodule 策略
[Original Request] 不动上游 `NetX-lab/Echo-slowdown.git`：保持 pinned commit + url 不变，本地建 sc26-ae 分支但不依赖 push；task2 的 AE wrapper（config 注入、输出重定向）全部放主仓 SC26-AE/ 下，运行时写入 config。若执行阶段发现必须改其内部代码，再升级为 fork。megatron-sim-engine（fwyc0573 名下）直接建 sc26-ae 分支并 push。

### D3. 分支基点
[Original Request] 先在 overlap-tracing 上把 AE 相关修改（examples 两个脚本、tests/e2e、tools/ae、docs/ae、tests/integration+unit）commit，再从该点建 sc26-ae。排除 `.omc/` 与另一 agent 的 `task_memory/task_2026-07-13_ddp_overlap_comprehensive_review/`。

### D4. 脚本入口形态
[Original Request] 仅 9 个独立脚本（task{1,2,3} × {gpt175b, qwen3_a30b, dsv3}），不做 dispatcher；后续通过 tex 修改建议清单把 tex 入口描述改为 per-model 命名。

### D5. Task2 语义与硬件
[Original Request] 三个 task2_<model>.sh 共享同一套 Echo-slowdown collection+training 核心（自带 micro-benchmark），仅输出目录/命名按 model 区分，幂等可复用；产出的 predictor 通用、被三个 model 的 task3 共用。经 fact-check（`slowdown_collection/input/train_script.py:24` 的 `torch.cuda.set_device(rank)` + NCCL 不允许双 rank 共卡），Echo-slowdown 原生不支持单卡采集，task2 按 ws=2（2 GPU）最小要求进行。

### D6. MoE 并行配置
[Original Request] 两个 MoE model：256 卡，tp=8、pp=4、dp=8（唯一解）、exp=8（=dp，铺满）。

### D7. DeepSeek-V3 variant profile
[Original Request] 采用缩配 variant 对齐 paper evaluation（32L / 32 experts / top-2，保留 MLA），在现有脚本中加一套 AE 用缩配 profile 分支。

### D8. MoE task1 rank 范围
[Original Request] 默认全量 256 ranks + 提供 QUICK=1 冒烟模式（子集 rank + 少量 iters）；若 dry-run 实证子集 trace 可被 sim-engine 完整消费，再把默认切换为子集。

### D9. Task3 comm backend
[Original Request] 默认 `analytical`（零外部数据、无嵌套 submodule 依赖、CPU 可跑，匹配 comm 弱验证定位）；collective-sim 作为可选项在 README 提及（若 dry-run 验证其依赖可公开拉取）。

### D10. Rank0 report 实现层
[Original Request] 在 megatron-sim-engine（sc26-ae 分支）内置 reporter：输出 rank0 的 one step time、forward/backward/optimizer 分解、simulator wall-clock time。

### D11. Prebaked 产物
[Original Request] dry-run 产出的 3 model traces + 一份训好的 predictor 作为 prebaked 产物 git 提交进仓（SC26-AE/prebaked/）；task3 自动优先用新鲜产物、缺失时回退 prebaked。

### D12. 输出目录结构
[Original Request] 按 model 分组：`SC26-AE/output/<model>/{task1,task2,task3}/`；task2 共享产物用 marker 文件指向，避免重复存储。

### D13. tex 同步
[Original Request] 本任务只产出 tex 修改建议清单（逐条旧文→新文）放 task_memory，由用户自行修改 sc26-ad.tex。

### D14. Dry-run 环境
[Original Request] 按 §14 手册用 rlaunch（codesign/h800 验证组合）拉 GPU worker，直接用 AE 镜像 `hub.i.basemind.com/mg-echo/megatron-h800:v1.1-image-11c794ef` 启动，mount 仓库先跑 setup_grouped_gemm_v1.sh 再跑 3 个 task，全链路与 AE 人员体验一致；task2 申请 2 GPU。

### D15. 低争议默认方案（8 条，已整体确认）
[Original Request] ①task3 内嵌自动生成 mg_scheduling plan（canonical 生成器 dry-run 确认，初步倾向 sim-engine 内置副本）；②gpt175b task1 沿用每 pp stage 一个代表 rank（8 ranks）；③统一 mock-data + bf16（gpt 脚本 fp16 由 wrapper 覆盖）+ --overlap-grad-reduce；④SC26-AE/README.md 英文；⑤硬件声明 task1/3 单卡、task2 双卡、task3(prebaked) CPU-only；⑥SC26-AE/setup.sh 单入口包装 tools/ae/setup_grouped_gemm_v1.sh，新依赖追加进该体系；⑦reporter 落 sim-engine sc26-ae 分支，report.json+report.md 输出到 output/<model>/task3/；⑧硬编码路径全部 wrapper 注入（Echo-slowdown 用自带 update_configs.py），.gitignore 为 prebaked 做白名单。

### D16. Fresh slowdown provenance 与耗时 gate
[Original Request] 首先尝试方案 1：Task1 通过同一次 Nsight Systems capture 包住选定 fake-rank loop，Task3 使用同源 traces、SQLite 以及 Task2 的 NCU/model/scaler 构建并消费 slowdown assets；方案 2（只让 Task3 使用成套 prebaked traces+SQLite+assets，fresh Task1/Task2 分别展示模块能力）作为备选。如果方案 1 导致 workload tracing 耗时过长，例如 256 卡 MoE model 的 tracing 需要 2 小时以上，则采用方案 2。应先通过单 rank 的 tracing 耗时估计 256 ranks 的总耗时。

### D17. Task2 隔离运行目录
[Original Request] 采用 `isolated_git_archive_snapshot`：每个实际 Task2 collection run 从 pinned Echo-slowdown commit 通过 `git archive` 在 `SC26-AE/output/_work/` 下建立独立源码快照，在快照内注入 config 并生成结果，再把 canonical outputs 与 provenance manifest 归档到 Task2 输出目录；不得在 pinned submodule checkout 中原地运行和改写 tracked 文件。

### D18. Task3 canonical mg_scheduling generator
[Original Request] 采用 `sim_engine_builtin`：Task3 只调用 `megatron-sim-engine/src/scheduler/mg_scheduling/` 作为唯一 canonical generator；Megatron-LM 主仓顶层 `mg_scheduling/` 不作为 AE runtime 入口。

### D19. rank0 report 主口径与备选口径
[Original Request] rank0 report 以方案 1 为主：`rank0_step_time_ms` 采用 rank0 final timeline 的 `max(finish_time) - min(join_time)`，forward/backward/optimizer 分别采用 exact-name operation 的 `(finish_time - join_time)` 总和，另报告 simulator load/execution/wall-clock，其中 wall-clock 为 load + execution，且三个 operation-duration sums 不要求等于 step span。方案 3（现有 visualization 的 `comp_time + comm_time`）作为备选。方案 3 的启用条件与输出标记方式仍需后续 grilling 明确。

### D20. `comp+comm` diagnostic 边界
[Original Request] 采用 `diagnostic_only_fail_fast`：允许报告 `rank0_comp_plus_comm_diagnostic_ms` 作为诊断值；若 rank0 timeline span 无法计算，则不得生成或填充 `rank0_step_time_ms`，必须明确报错并 fail fast。`comp+comm` 不得替代 overlap-aware one-step 主口径。

### D21. Prebaked artifact 分发策略
[Original Request] 采用 `size_gate_then_release`：正式 dry-run 生成逐文件 size/checksum manifest 后，只有当最终分发文件全部小于 50 MiB 且三模型 prebaked bundle 总量不超过 500 MiB 时，才把 bundle 放入 regular Git；否则使用 GitHub Release assets，repo 内只保存 manifest、SHA256 和下载/校验入口，并在 README 中明确唯一 canonical 分发路径。

### D22. Task3 artifact source 自动选择偏好
[Original Request] 在 fresh 与 prebaked 两套输入之间，用户选择 `automatic_fallback`：fresh 存在时使用 fresh，否则自动改用 prebaked。该偏好与仓库强制的 No Fallbacks / Fail Fast 规则存在冲突；对“fresh 不存在”与“fresh 已尝试但不完整/损坏”的区分尚需后续 grilling 明确。

### D23. Task3 artifact source 最终选择规则（supersedes D22）
[Original Request] 改为 `explicit_source_only`：每次运行 Task3 都必须显式指定 `ARTIFACT_SOURCE=fresh|prebaked`；不得自动从 fresh 切换到 prebaked。所选 bundle 缺失、部分生成、checksum/provenance 不匹配时必须 fail fast。

### D24. AE image 依赖补齐与当前验证授权
[Original Request] 采用 `new_pinned_ae_image`：将容器环境中任何缺失的必要库包和工具依赖单独整理到一个 doc，由用户后续在其他机器上补齐这些依赖并推送新的 image。当前任务执行阶段，允许 agent 在容器中额外下载和安装已确认缺失的依赖，以避免当前 AE 验证任务因旧 image 缺包而阻塞。

### D25. 三模型统一 scaling warmup/profile
[Original Request] 采用 `warmup3_profile1`：GPT-175B、Qwen3-A30B 和 DeepSeek-V3 三个 Task1 wrapper 都显式传入 `--scaling-min-warmup-iters=3` 与 `--scaling-profile-iters=1`；不得继承各 source script 不一致的默认值。

### D26. 当前容器环境修复与继续执行
[Original Request] 用户当前无法推送新的容器镜像，并明确该步骤不是当前任务继续推进的阻塞原因。Agent 必须重新检查当前容器中的全部 conda env，重点查找名称类似 `myenv_yc` 且已经提供 Megatron-LM 必要 runtime 的可用环境；对于其余缺失依赖或工具，例如 `nsys`，以及 slowdown module 可能需要的更高版本 Python 或独立 conda env，允许并要求在当前容器中完成配置。Agent 应解决这些基础环境和库包依赖问题并继续任务执行；仅当存在无法通过容器或仓库事实确定的关键信息或决策时，才使用单题 `grill-me` 向用户确认。

### D27. MemoryTracker qualification probe-only remediation
[Original Request] 用户选择方案 1：仅修复 qualification probe。允许使用 isolated loader 或等价的 probe-only import 方式，在不修改 Megatron/Echo 产品源码、不跳过 MemoryTracker、且仍要求生成非空 memory JSON 的前提下，使用新的 artifact root 重新执行 H800 B1 qualification。CPU controller 的 isolated-loader import 成功只能作为可行性证据，不能代替 H800 NVML/CUDA/non-empty JSON qualification；B2 仍需验证真实产品 import/runtime 路径。

### D28. Gate B1 interrupted-submission recovery authorization
[Original Request] 对 2026-07-17 14:37:44 +08:00 未经授权提交、随后在 Echo qualification payload 执行前停止的 exact-two-H800 RJob，用户选择 `authorize_one_clean_retry`：完整保留并披露该违规事件；在独立审计通过并完成与实际 live contract 完整绑定的 fresh predict-only 后，只允许再提交一次 exact-two-H800 live qualification。

### D29. Test 问题自主修复授权与 AE 核心目标

[Original Request] 该 test 自主修复授权覆盖与 test 直接相关的 audit、schema、validator、documentation 和 control-plane 问题；不得通过弱化 assertion/acceptance criteria、跳过 checksum/provenance、加入 fallback，或把 local/synthetic PASS 冒充真实 qualification 来规避问题。真实 GPU、quota、image、scheduler、实际 product/runtime/workload、真实 pre-dataset data-quality 或 qualification 失败仍按 fail-fast 处理。
[Original Request] 修改 gate：任何 test 类型的错误和问题，允许 agent 自行修复和决策；前提是所有处理都服务于当前 task 的 AE 流程核心目标，即编写可一键运行的 shell scripts，并收集符合要求、可供 AE 人员直接复用运行的 pre dataset。

### D30. Latest test-failure autonomy gate (supersedes the narrow D29 interpretation)

[Original Request] 任何由 test、validation 或 rehearsal 暴露的错误和问题，只要直接服务当前 task 的 AE 核心目标（编写可一键运行的 shell scripts，并收集符合要求、可供 AE 人员直接复用运行的 pre dataset），agent 可以自行诊断、决策和修复；不需要因为这类 test 问题再次等待用户批准。测试的验收标准、数据质量、checksum/provenance、real-vs-synthetic 证据边界和 no-fallback 规则不能被降低或绕过；测试未通过前不得宣称对应 gate 已通过。

### D31. V21 clean-clone evidence-log tracking approval

[Original Request] 批准“精确追踪 59 个 V21 必需 logs”。

### D32. Reviewer-generated nested runtime-state cleanup approval

[Original Request] 批准删除 `megatron-sim-engine/.omc`。

## A1. Gate A 批准与执行授权
[Original Request] 批准 Gate A，完成已审查的 plan；允许启用并行 team 模式和 subagents，以尽可能加速当前任务的执行速度。

## D60. Reduced functional AE target
[Original Request] 当前只要求 fake-level Task1/2/3 功能链可运行，不要求在真实分布式多节点多卡环境验证精度；AE 目标是获得开源与功能可运行徽章，不要求复现论文数值。

## D61. Representative model scope
[Original Request] 当前正式代表模型为 `gpt175b` 与 `qwen3_a30b`；DeepSeek-V3 问题记录到 docs 后暂存，不在本轮修复范围内。

## D62. Task1 rank scope
[Original Request] GPT dense Task1 只 trace 8 个 PP representative ranks：`0,128,256,384,512,640,768,896`；Qwen3 MoE Task1 trace 32 个 PP×EP representatives：`0,8,16,...,248`；Task1 NCU 只采集 global rank 0。

## D63. Qwen3 topology
[Original Request] Qwen3-A3B workload tracing 使用 `world_size=256, pp=8, tp=8, ep=4, dp=4`。

## D64. Task2 hardware and reuse
[Original Request] Task2 必须使用两个真实 GPU。Task3 缺少 kernel feature 时不得因此重新运行 Task2；已有已校验 two-GPU dataset/predictor 应直接复用。

## D65. Missing-kernel slowdown behavior
[Original Request] Task3 对缺失 kernel feature 的处理顺序为 exact feature、唯一明确的类似 kernel alias、否则跳过 slowdown；跳过时 `slowdown_factor=0` 且 `predicted_duration=baseline_duration`。当前关注 workflow 正常和输出一般逻辑合理，不关注 fidelity。

## D66. Temporary-storage and heavy-command safety
[Original Request] 禁止向 `/tmp` 写 temporary files、logs 或 caches；统一使用 `/data/ycfeng/tmp`。`brainctl get replica` 等重命令必须使用 `timeout 60s` 和 `systemd-run --scope -p MemoryMax=2G`。

## D67. Canonical branch and artifact consolidation
[Original Request] 进入 scripts、docs、workload tracing 文件、slowdown dataset、slowdown predictor 权重和关键测试记录的清晰规范整理：确认一个唯一最终 branch，不按 model 分散交付；把 `/data/ycfeng/tmp` 中最终 PASS 的关键记录合理归档到该 branch；并提供一份可以快速核查、清点和复用脚本/文件的清单。
