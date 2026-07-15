# Progress — SC'26 AE Workflow

## Modification History

| Date       | Summary of Changes                                  |
|------------|------------------------------------------------------|
| 2026-07-16 | Session 5: recorded Gate A approval and started Phase 0 safety-baseline execution |
| 2026-07-15 | Session 4: recorded post-review Gate A fresh validation and explicit approval stop |
| 2026-07-15 | Session 3: integrated independent Claude WATCH review and canonical scheduler/simulator reconciliation |
| 2026-07-15 | Session 2: completed author self-review corrections; independent review pending |
| 2026-07-15 | Session 2: recorded LOCAL_SIZE/fake-node-size contract gap |
| 2026-07-15 | Session 2: recorded setup source and overlap-mode fail-fast gates |
| 2026-07-15 | Session 2: recorded manifest portability risk and plan-rewrite checkpoint |
| 2026-07-15 | Session 2: captured D23 and closed no-fallback conflict |
| 2026-07-15 | Session 2: captured D22 and opened no-fallback reconciliation |
| 2026-07-15 | Session 2: captured grilling decision D21            |
| 2026-07-15 | Session 2: captured grilling decision D20            |
| 2026-07-15 | Session 2: captured grilling decision D19 and identified fallback ambiguity |
| 2026-07-15 | Session 2: captured grilling decision D18            |
| 2026-07-15 | Session 2: captured grilling decision D17            |
| 2026-07-15 | Session 2: captured grilling decision D16            |
| 2026-07-15 | Session 2: completed reachability and artifact-size fact checks |
| 2026-07-15 | Session 2: enhanced plan review started; implementation explicitly gated |
| 2026-07-15 | Session 1: exploration + grilling + docs landed      |

## Status board

| Phase | Item                                        | Status      |
|-------|---------------------------------------------|-------------|
| —     | Codebase/paper exploration (7-reader sweep) | completed   |
| —     | Grilling session (D1–D23 resolved)          | completed   |
| —     | Initial docs landed (requirements/plan/notes/issues/progress) | completed |
| —     | Enhanced plan-doc review                    | completed: Gate A approved 2026-07-16 |
| P0    | Git preparation (commit + sc26-ae branches) | completed   |
| P1    | Gate B dry-run 3 tasks (rlaunch + AE image) | in progress |
| P2    | SC26-AE script suite (9 scripts)            | pending     |
| P3    | sim-engine rank0 reporter                   | pending     |
| P4    | README + tex change suggestions             | pending     |
| P5    | Fresh-clone rehearsal + test report + review | pending    |

## Log

### 2026-07-15 Session 1 (planning)
- **Motivation**: user kicked off SC'26 AE preparation via /grill-me; plan-and-discuss phase, docs-first.
- **Done**:
  - Parallel exploration workflow (7 readers): tracer scripts, Echo-slowdown, sim-engine, mg_scheduling, AE assets, git structure, paper core. Key facts recorded in notes.md.
  - Fact-check: Echo-slowdown 单卡采集原生不支持（train_script.py:24 set_device(rank) + NCCL 双 rank 禁共卡）→ 用户确认 task2 按 ws=2 执行。
  - Grilling interview: 10 questions asked, all decisions resolved → requirements.md D1–D15.
  - Landed: requirements.md, plan.md, notes.md, issues.md (I1–I8), progress.md.
- **Decisions of note**: 整仓推 fwyc0573/sc26-reproduce；Echo-slowdown 不动上游；9 独立脚本无 dispatcher；dsv3 用缩配 variant；MoE 全量 trace + QUICK；analytical comm backend；reporter 内置 sim-engine；prebaked 产物入仓；输出按 model 分组；tex 只出建议清单；dry-run 走 rlaunch+AE 镜像全链路。
- **Next**: P0 git preparation（需用户明确放行后执行 commit/branch 操作）。

### 2026-07-15 Session 2 (enhanced plan review)
- **Motivation**: 用户要求审查并增强 `task_2026-07-15_sc26_ae_workflow` 的计划文档；对模糊或缺失的关键决策使用 grilling；本阶段禁止进入实现。
- **Expectation**: 形成可执行、可验证、无关键歧义的计划文档，并把独立审查意见、开放问题和后续 gate 全部留档；不修改产品代码、脚本或 submodule。
- **Method**: 先恢复 requirements/plan/notes/progress/issues 上下文，再依据仓库、paper、AE draft、现有 scripts/tests 和 submodule 接口逐项 fact-check；只有无法由代码事实回答的决策才逐题询问用户。
- **Result (current)**: 已确认工作区存在预先已有的未提交改动；本轮写入范围锁定为本任务目录内的计划类文档。初步发现当前 P1 混合事实调查、设计决策和实现动作，需在审查后重新分阶段；`review.md`、`summary.md`、`lessons.md` 尚未建立。
- **Evidence added**: 主仓/工作树/submodule 状态已核对；AE draft 对 T1 memory report 的承诺未被当前 acceptance criteria 覆盖；slowdown assets 构建器已在 sim-engine 中定位；DeepSeek-V3-Proxy 的已验证 MHA 基线与计划中的 MLA variant 存在能力边界，需在计划中显式设为待实现/待验证项。
- **Plan assumptions corrected**: DeepSeek-V3 32L/32E/top2/MLA profile 与 GPT-175B 结构配置均已存在，计划不得重复开发；后续审查聚焦 canonical source-script 选择、bf16/DDP overlap/memory/output 契约和 runtime 证据。
- **Task1 source review**: 两个 wall-clock scan scripts 的固定拓扑不符合 AE matrix，且缺少 DDP overlap/memory 契约；后续计划将其降级为测试参考，并以模型主脚本/`update_pretrain_gpt.sh` 为最小适配 baseline。
- **Critical pipeline gap found**: slowdown assets builder 需要 fresh trace + trace-compatible nsys SQLite + NCU metrics + model/scaler；当前计划未产生 SQLite，也未定义 fresh/prebaked compatibility，故 T1+T2→T3 尚未闭环。另确认 Echo-slowdown 原生运行会覆盖 tracked config/output，Task2 需要隔离工作区策略。
- **Strictness confirmed**: blueprint 是 trace-set-specific 且缺失即 fail-fast；现有 e2e 通过单次 nsys capture 包住 selected-rank loop 保证原子性。两份 mg_scheduling 核心实现已漂移，sim-engine 内置版本因现有 e2e 覆盖更适合作 canonical 候选。
- **Reporter semantics review**: 现有 comp+comm stdout 不能表示 overlap-aware step span；拟在 grilling 中确认从 rank0 in-memory timeline 计算 step span、按 op name 聚合 fwd/bwd/optimizer、wall-clock=load+execution 的 schema。
- **Public reachability evidence**: 通过公开 URL 的 `git ls-remote` 与独立 `/tmp` shallow fetch，确认 Echo-slowdown `1390b441...`、sim-engine `2044cccc...`、nested collective-sim `6e06e3f...` 均可获取；collective-sim pinned commit 是公开 `ft-cc` history 的祖先。未修改主仓或 submodule。
- **Artifact-distribution evidence**: 当前没有三模型 fresh SQLite/assets 可供量化；现有 34 个 trace txt 合计 0.165 MiB、现有 predictor 0.592 MiB，不足以外推全量 AE bundle。确认 GitHub regular Git 对 >50 MiB 单文件警告、>100 MiB 单文件阻止，故最终计划必须在 dry-run 后用 size manifest 决定 regular Git / Git LFS / Release assets，而不能无条件承诺 regular Git。
- **Grilling D16 — fresh provenance**: 用户选择先尝试 atomic fresh bundle；以单 rank tracing 时长估算 256-rank MoE 总时长，预计超过 2 小时时，显式采用仅 prebaked bundle 闭环。该选择必须是 dry-run 后记录证据的 plan gate，不得实现为掩盖错误的静默 fallback。
- **Grilling D17 — Task2 isolation**: 用户选择 `isolated_git_archive_snapshot`。每次实际 collection run 在 `SC26-AE/output/_work/` 中使用 pinned commit 的独立 tracked-source snapshot，所有 config/output 改写发生在快照内；submodule clean status、source commit 和归档 checksum 将成为验收证据。
- **Grilling D18 — schedule generator**: 用户选择 sim-engine 内置 `src/scheduler/mg_scheduling/` 为 Task3 唯一 canonical generator；主仓顶层副本仅作为 legacy/reference。计划必须对三模型输出结构做契约测试，而不是继续尝试维护两份实现等价。
- **Grilling D19 — report semantics**: 用户选择 timeline span + exact-op duration sums 为主口径，`comp+comm` 为备选。由于 `comp+comm` 在 overlap 下重复计时，尚须继续 grilling 冻结其启用条件；在此之前不得将它静默映射到 `rank0_step_time_ms`。
- **Grilling D20 — diagnostic boundary**: 用户选择 `diagnostic_only_fail_fast`。`rank0_comp_plus_comm_diagnostic_ms` 可作为附加诊断值；主 timeline span 缺失/非法必须报错，不能用 diagnostic 替换 `rank0_step_time_ms`。
- **Grilling D21 — artifact delivery**: 用户选择 `size_gate_then_release`。最终文件全部 <50 MiB 且 bundle <=500 MiB 时使用 regular Git；否则使用 GitHub Release assets，repo 保存 manifest/SHA256/下载入口。最终介质只能由 dry-run 的实际 size manifest 决定。
- **Grilling D22 — source preference**: 用户选择 automatic fresh→prebaked fallback。该选择与强制 No Fallbacks / Fail Fast 规则冲突，尚不能进入 plan；下一轮将确认安全 auto-selection：clean/no-attempt→prebaked，complete+verified fresh→fresh，partial/invalid fresh→fail-fast。
- **Grilling D23 — source rule (supersedes D22)**: 用户最终选择 `explicit_source_only`。Task3 每次必须显式指定 fresh/prebaked；不存在自动换源，任何所选 bundle 的缺失或校验错误均 fail fast。I12 已关闭。
- **Manifest portability risk (I13)**: slowdown-assets builder 当前会把生成机输入路径原样写入内部 manifest；prebaked 搬运后不能依赖这些路径。计划将要求外层 portable provenance manifest + SHA256，并让 Task3 始终显式传 bundle 内 model/scaler 路径；是否同步最小增强 builder 由独立 Claude 计划审查裁决，不做临时路径 fallback。
- **Setup source strictness (I14)**: 静态核对确认 `tools/ae/setup_grouped_gemm_v1.sh` 当前自动执行 VCS→archive recovery，与 No Fallbacks 冲突。增强计划将改为显式 `GROUPED_GEMM_SOURCE=vcs|archive` 契约，并在 implementation 前要求 RED tests 覆盖未指定、非法值和 selected-source failure；本轮只记录计划，不修改 installer。
- **DDP-overlap execution gate (I15)**: 静态核对确认 `simu_main.py` 默认 `--overlap-mode auto`。增强计划将要求三个 Task3 wrapper 始终显式传 `--overlap-mode on`，并让 metadata 缺失走 fail-fast negative test；本轮不修改 sim-engine。
- **Current checkpoint**: 代码事实核对与 D16–D23 grilling 已完成；现进入 `plan.md` 全量重写，随后创建 `review.md`、执行独立 Claude review 和 Markdown/路径/traceability 验证。implementation gate 保持关闭。
- **I16 motivation**: 新版计划在 Task3 命令中引用了 `${LOCAL_SIZE}`，但固定模型矩阵和 artifact topology 未定义其值；这会让模拟 node boundary 与 analytical backend 的 8-GPU-per-node 假设失去可审计的一致性。
- **I16 expectation**: 在不进入 implementation 的前提下，把 Task3 的 `LOCAL_SIZE=8` 冻结为计划契约，同时明确两个 MoE Task1 源脚本的 fake-node-size 差异是否需要后续代码修改。
- **I16 method**: 静态核对 `simu_main.py`、sim-engine group/rank manager、`nccl_comm.py`、scheduler presets，以及 GPT/Qwen/DeepSeek 三个 canonical Task1 源脚本的 `--fake-gpus-per-node` 参数。
- **I16 result**: Task3 侧有充分事实依据固定为 8；Task1 侧只确认 GPT=8、MoE=world-size 的差异及其会改变 tracer `server_id/local_rank`，尚不足以证明对 trace consumer 的实际影响。已写入 notes/issues，计划将把 MoE 侧交给独立 Claude review 作为 WATCH，当前未修改任何 implementation 文件。

### 2026-07-15 Session 2 — author self-review correction pass
- **Motivation**: 全量阅读增强后的 `plan.md` 后，发现若直接执行会出现跨运行 stale artifact 混入、Task3 database 来源未定义、prebaked producer commit 自引用、Release 下载缺口、遗漏 metadata size、缺失 operation 静默记零、外部发布未经 exact-target gate，以及 CPU RAM 数值无证据等问题。
- **Expectation**: 在不进入 implementation 的前提下，把这些问题转成明确的路径、schema、fail-fast、测试和审批 contract，使独立 reviewer 能针对可执行计划而不是模糊意图给出 verdict。
- **Method**: 静态核对 Qwen/DeepSeek rank selector、tracer/memory/replay 的 CWD-relative 输出、`arguments.py` 的 overlap 自动启用、现有 slowdown e2e 的 trace/database 参数、`RankManager.fake_gpus_per_node` 消费面、slowdown builder manifest、grouped-gemm installer/doc 与 sim-engine CLI；随后只修改本 task 目录内 Markdown。
- **Result**: `plan.md` 已加入 Task1/Task3 immutable versioned runs 与 verified markers、`DATABASE_DIR="${TRACE_DIR}"`、`capture_runtime`/`simulation_topology` 分离、fresh/prebaked commit 校验差异、完整 staged-byte D21 gate、显式 Release fetch、external publication approval、missing-operation fail-fast 和 measured CPU RSS gate；`notes.md` 修正 Qwen=`FAKE_RANK_ORDER`、DeepSeek=`SCALING_FAKE_RANK_ORDER` 并记录源码事实；`issues.md` 新增 I17–I19。未修改 implementation、submodule、branch、commit 或 GPU 状态。
- **Next**: 建立 `review.md` 作者记录，运行 StepCode Claude 独立计划审查，根据 APPROVE/WATCH/BLOCK 处理，再执行 Gate A 文档验证。
- **Validation attempt 1 failure**: 预审的 `functions.exec` JavaScript 使用 template literal，计划 literal `${TRACE_DIR}` 在 shell 启动前被误作 JavaScript 插值，工具返回 `ReferenceError: TRACE_DIR is not defined`；没有任何 nested command 执行。
- **Root-cause resolution**: 改用不发生 JavaScript template interpolation 的 command 字符串（或显式转义 `${`），保持校验内容不变后重跑；该失败不归因于 repository、environment 或 plan contract。

### 2026-07-15 Session 3 — independent WATCH integration
- **Motivation**: StepCode Claude 对增强计划给出 `WATCH`，要求明确 scheduler `--model-size` 值域、验证 rank0 `optimizer_step`、标注 §7.4 为 target CLI，并防止 analytical backend 的 8-GPU node-size coupling 漂移。
- **Expectation**: 在不进入 Gate B/Phase 0/implementation 的前提下，将每个 WATCH 转成代码事实支持的明确 contract 和后续 verification gate；独立审查中的非-canonical 引用不能未经核对直接写入计划。
- **Method**: 读取 `.omx/artifacts/claude-review-task-memory-task-2026-07-15-sc26-ae-workflow-plan-md--2026-07-15T09-25-53-479Z.md`；静态追踪 `mg_test.py -> mg_scheduling_plan.py` 的 `model_size` 消费面、`simu_main.py -> src/core/simu_engine.py` 的 canonical import/direct mapping、现有 PP=1 slowdown smoke schedule，以及 `nccl_comm.GPUS_PER_MACHINE` setter 的调用面；随后只修改本 task 目录 Markdown。
- **Result**: I16 已按独立审查裁决为“不改 MoE source scripts”；`--model-size` 冻结为 opaque AE label；Gate B 缺失 `optimizer_step` 明确定义为 blocker；Task 4.3 增加固定为 `WORLD_SIZE=8/LOCAL_SIZE=8/PP=2/TP=1/EXP=1/DP=4/MBS=1/GBS=8` 的 scheduler-generated integration，以及 analytical local-size fail-fast test；§7.4 标为 Task 4.1 target CLI。未修改 implementation、submodule、branch、commit、worktree 或 GPU 状态。
- **Validation command/environment**: 在 repo root 使用 Python heredoc 对 6 份 Markdown 执行 existence/history/fence/placeholder/R-D traceability/entry uniqueness/contract/review-artifact assertions，并运行 `git diff --name-only`、`git submodule status --recursive`、SHA256/byte inventory；环境为 `CONDA_ENV=none`, Python `3.12.3`。
- **Validation criteria**: 6/6 必需文档存在；15/15 requirements 和 23/23 decisions 有 `[Original Request]`；9/9 public entries 唯一；所有 fence 平衡；4 个 Claude WATCH correction/gate 与 1 个 advisor artifact 可检索；exit code 0；tracked diff/submodule inventory 不扩大。
- **Validation result**: PASS，exit code `0`；`docs=6`, `requirements=15`, `decisions=23`, `public_entries=9`, `fence_pairs=38`, `watch_gates=4`, `review_artifacts=1`；task docs 共 `159512` bytes。Tracked diff 仍为 `3` 个既有文件：`examples/realistic_run_gpt.sh`、`examples/update_pretrain_gpt.sh`、`tests/e2e/run_ddp_slowdown_compare.py`；recursive submodule status 仍为 `3` 条，Echo/sim-engine gitlink 分别为 `1390b441...`/`2044cccc...`，nested collective-sim 仍未初始化。`requirements.md` SHA256 仍为 `df95fd918d593f5f85ce845d26bda24ba1618c5ac98653f39c29652f91378c5b`，证明最终 D23 capture 未被改写。
- **Next**: 执行一次 post-record fresh verification，随后停止在 explicit user approval gate；不得进入 Phase 0、Gate B 或 implementation。

### 2026-07-15 Session 4 — post-review Gate A verification
- **Motivation**: Session 3 的 PASS 发生在最后两次文档记录补丁之前，不能作为当前文件内容的 fresh completion evidence；同时必须确认 D23 的 `explicit_source_only` capture 和独立 Claude `WATCH` 修订没有在收尾过程中漂移。
- **Expectation**: 当前 6 份文档重新通过完整结构、traceability、contract 和 review-artifact 校验；tracked diff 与 recursive submodule inventory 不扩大；随后严格停止在 explicit user approval gate。
- **Method**: 在 repo root 使用 Python `3.12.3`、`CONDA_ENV=none` 执行 Task A4 assertions，并增加逐文件 Markdown fence、`[Original Request]` 总数和 4 个 WATCH gate 的断言；另运行 `git diff --name-only`、`git submodule status --recursive`、SHA256 和 byte inventory。首次附加断言错误地要求不存在的固定短语 `This is the target interface...`，而 §7.4 实际等价措辞为 `specifies the target CLI after Task 4.1...`；根因是 validator 文本匹配过窄，不是计划缺项。修正断言为文档实际 contract 后重跑。
- **Result**: 修正后的 pre-record validation PASS，exit code `0`：`docs=6`, `requirements=15`, `decisions=23`, `original_request_tags=38`, `public_entries=9`, `fence_pairs=38`, `watch_gates=4`, `review_artifacts=1`。Post-record task docs 共 `163275` bytes；tracked diff 仍仅为 3 个既有文件，recursive submodule status 仍为 3 条；`requirements.md` SHA256 仍为 `df95fd918d593f5f85ce845d26bda24ba1618c5ac98653f39c29652f91378c5b`。最终 post-record validation 在此记录后执行，且不再修改文档。
- **Next**: Gate A 的内部增强、独立审查和文档验证已完成；Gate A 本身仍为 `IN PROGRESS`，只等待用户明确审批。不得自动进入 Phase 0、Gate B、implementation、GPU、commit、branch/worktree、push 或 Release。

### 2026-07-16 Session 5 — Gate A approval and Phase 0 start
- **Motivation**: 用户明确批准 Gate A，并要求完成整份已审查 plan，同时授权使用并行 team/subagents 加速执行。
- **Expectation**: 严格按 plan 的依赖 gate 推进；先建立可审计的 D3 baseline 和隔离 worktree，再运行 Gate B，之后才允许按 TDD 修改 feature code。并行执行不得绕过 shared-file ownership、测试或 review gate。
- **Method**: 重新读取当前 Git/submodule/worktree 状态、Phase 0/Gate B 合同及 `executing-plans`、`using-git-worktrees`、`test-driven-development`、`team` skill；将本次原始授权写入 `requirements.md`，更新 Gate A/Phase 0 状态，并准备只 stage D3 明列路径。
- **Result (current)**: Gate A 状态更新为 `APPROVED 2026-07-16`，Phase 0 更新为 `IN PROGRESS`；Gate B 和 Phase 1–9 仍保持 blocked。当前仍在 `overlap-tracing`，尚未 stage、commit、创建 branch/worktree、启动 GPU 或修改 feature implementation。
- **Next**: 完成 Task 0.1 ownership/safety inventory，运行 Task 0.4 baseline tests，再按 Task 0.2 精确 stage/commit D3 baseline 并创建 `sc26-ae` isolated worktree。

#### Task 0.1 ownership inventory and pre-commit baseline verification
- **Motivation**: 大规模执行前必须证明现有 dirty workspace 的所有权与 Gate A inventory 一致，并建立可区分后续回归的测试基线。
- **Expectation**: 只允许 3 个既有 tracked edits 和 D3 明列 AE assets 进入 baseline；`.omc/`、旧 overlap-review task、空文件 `=10.1` 及任何未授权路径不得 stage。四个 baseline test commands 必须全部 exit `0`。
- **Method**: 核对 branch/HEAD、full status/diff、recursive submodules 和两个 worktree；单独检查 `=10.1` 为 `0` byte 的既有空文件；运行 grouped-gemm shell unit、GPT mock integration 以及 sim-engine 两个 pytest suites。
- **Result**: 当前 branch=`overlap-tracing`、HEAD=`a5bcd3d9a6053b992da45bcc4ade180c4dd94c73`、worktree count=`2`、recursive submodule status count=`3`。受保护/排除路径均未 stage。Baseline tests PASS：grouped-gemm `30/30`，GPT mock integration `22/22`，sim-engine pytest `16/16`；三个命令 exit code 均为 `0`。未启动 GPU、未修改 feature code。
- **Next**: 仅按 Task 0.2 D3 path list stage；验证 staged path set 精确相等后创建 Lore baseline commit。

#### Task 0.2–0.4 baseline commit, isolation, and second baseline gate
- **Motivation**: feature execution 必须从可审计的用户批准 baseline 开始，并与 dirty `overlap-tracing` checkout 及受保护 review worktree 物理隔离。
- **Expectation**: D3 commit 只能包含批准的 15 个路径；新 worktree/branches 必须从同一 commit 和 pinned submodule commits 创建；隔离 worktree 中同一 baseline tests 必须再次全部通过。
- **Method**: 精确 `git add -- <D3 paths>` 后用 sorted path diff 和 `git diff --cached --check` 验证；创建 Lore commit；在外部路径 `/data/ycfeng/Megatron-LM-sc26-ae` 创建主仓 `sc26-ae` branch，初始化 recursive submodules，并分别创建 Echo/sim-engine 本地 `sc26-ae` branch。等待同一个 submodule update process 完整退出后验证 commits，再重跑三组 baseline commands。
- **Result**: baseline commit=`0ad3cb4eda2248f4e09908a80e5693cffa6e0c1e`，staged path count=`15`，protected staged count=`0`。新 worktree branch=`sc26-ae`；Echo=`1390b4416ded08bc1b9cd0620d329d81d4470bf9`，sim-engine=`2044cccc8fff222172b7f91571a617886841001f`，nested collective-sim=`6e06e3f5140cd4e2e7c12a35586ebcdc0f410df0`。隔离 baseline 再次 PASS：`30/30`、`22/22`、`16/16`，全部 exit `0`。
- **Issue and resolution**: 首次 submodule status 读取发生在 unified exec session 尚未退出时，因而看到半初始化 checkout。根因是 leader 过早读取异步进程，而非 repository/submodule 损坏；等待原 session `32236` 正常完成后 commits 与 recursive status 全部正确，没有执行清理、fallback 或重复 clone。
- **Next**: Phase 0 完成；读取 GPU/env handbooks，执行 Gate B 的 1-GPU/2-GPU `rlaunch --predict-only` 与现有三任务 runtime reconnaissance。任何接口 blocker 先回写 notes/issues 并停止 feature implementation。
