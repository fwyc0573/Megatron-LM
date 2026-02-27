## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-27 | Added stage-2 round11 scaling replay-write-phase / scheduler-increment semantic-alignment experiments and new fidelity evidence |
| 2026-02-27 | Added stage-2 optimizer microphase protocolfix8 phase-aware fidelity findings (run1/2/3 + median-of-runs) |
| 2026-02-27 | Implemented stage-2 optimizer microphase trace-only path (default-off) and added unit-validation notes |
| 2026-02-27 | Added stage-2 protocolfix8 fidelity notes and optimizer semantic-touching proposal (design-only, pending user confirmation) |
| 2026-02-27 | Added stage-2 round5 replay-cache iteration-alignment design notes and latest fidelity status |
| 2026-02-24 | Added architecture comparison, gap analysis, stage-1 simplifications, and stage-2 backlog |
| 2026-02-27 | Added stage-2 (DeepSeek-V3 architecture standard) detailed spec mapping to upstream YAML, plus environment constraints (TE=1.3.0) and implementation decisions |
| 2026-02-27 | Added stage-2 execution notes: shared-expert gate support, short-run scheduler guard, and distributed PP2 bf16 NaN observations |
| 2026-02-27 | Added stage-2 round2 notes: PP2 bf16 NaN root cause isolation and MLA-only p2p dtype-alignment fix with router finite-normalization hardening |
| 2026-02-27 | Added stage-2 round3 fidelity notes: CMD sync default rollback, scaling optimizer timing-boundary alignment, and residual backward/optimizer gap characterization |
| 2026-02-27 | Added stage-2 round4 fidelity notes: scaling optimizer pre-CMD side-effect parity (`numel` pre-scan), refreshed pair runs, and updated residual-gap status |

# Architecture Notes (Mixtral vs Qwen3 vs DeepSeek-V3-Proxy)

## 1) Mixtral vs Qwen3 vs DeepSeek-V3-Proxy

- **Mixtral (current baseline)**
  - All transformer layers are MoE in current usage.
  - Router path: Mixtral-style top-k + aux_loss (existing code path).
  - Standard MHA/GQA + RMSNorm + SwiGLU.
- **Qwen3-30B-A3B (stage-1 target)**
  - MoE model requiring explicit `moe_ffn_hidden_size` and `rotary_base` support.
  - Router can be kept in Mixtral-compatible top-k + aux_loss in stage-1.
  - All layers can be treated as MoE (`moe_layer_freq=1`) for stage-1.
- **DeepSeek-V3-Proxy (stage-1 target)**
  - Mixed dense/MoE layer schedule (proxy: 3 dense + 11 MoE).
  - Full upstream stack includes MLA, seq_aux_loss routing variants, shared experts, and optional MTP.
  - Stage-1 keeps MHA simplification and Mixtral-style router semantics for tracing correctness.

## 2) Missing modules / params in legacy codebase

### Missing or not yet ported modules (from latest reference)

- `multi_latent_attention.py` (MLA)
- `moe/shared_experts.py`
- `moe/router_replay.py`
- `moe/fused_a2a.py`
- `multi_token_prediction.py`
- DeepSeek-specific router variants (`seq_aux_loss`, score/group/bias semantics)
- `flex` dispatcher + `deepep` integration

### Missing first-class config/CLI support before stage-1 patch

- `--moe-layer-freq`
- `--moe-ffn-hidden-size`
- `--rotary-base`
- `TransformerConfig.moe_layer_freq`
- `TransformerConfig.moe_ffn_hidden_size`
- `TransformerConfig.rotary_base`

### Baseline layer-spec gap

- Old `gpt_layer_specs.py` only produced a single layer spec repeated by block.
- No native dense/MoE mixed pattern generation by layer index.

## 3) Stage-1 simplifications and impact

- **DeepSeek attention simplification**: MLA -> MHA
  - Impact: architecture fidelity reduced for DeepSeek-specific attention path.
  - Benefit: keeps tracing/scaling integration stable and runnable in current environment.
- **Router semantics simplification**: keep Mixtral-style top-k + aux_loss
  - Impact: no DeepSeek-specific router behavior (seq_aux_loss/group-topk/bias).
  - Benefit: distributed/scaling route-control path remains deterministic and debuggable.
- **Checkpoint path omitted**: random/mock initialization only
  - Impact: no direct training continuation from official checkpoints.
  - Benefit: avoids blocker dependencies and allows fast tracing verification.
- **Smoke profiles in scripts**
  - Impact: reduced model size defaults for local validation throughput.
  - Benefit: practical stage-1 verification with mock data and short iterations.

## 4) Stage-2 backlog

1. Add full DeepSeek MLA path (`multi_latent_attention` stack).
2. Add DeepSeek router semantics (`seq_aux_loss`, sigmoid/group-topk/expert-bias).
3. Add shared experts support.
4. Add optional MTP branch.
5. Add `flex`/`deepep` dispatcher integration.
6. Implement checkpoint conversion/loading (HF -> Megatron).

## Environment constraints identified

- `deepep` is not installed in current environment.
- `transformer-engine` / `triton` versions are older than latest DeepSeek advanced feature expectations.

# Stage-2 Notes: DeepSeek-V3（架构标准）对齐与关键实现决策

## 1) Upstream YAML 对齐清单（只取架构标准所需）

对齐来源：`latest-megatron/Megatron-MoE-ModelZoo/model_configs/benchmarking/DeepSeek-V3.yaml`

架构标准必须覆盖的 flags（本 repo 需新增/扩展）：

- MLA:
  - `--multi-latent-attention`
  - `--q-lora-rank`, `--kv-lora-rank`
  - `--qk-head-dim`, `--qk-pos-emb-head-dim`, `--v-head-dim`
- YaRN RoPE:
  - `--rotary-scaling-factor`
  - `--mscale`, `--mscale-all-dim`
  - `--original-max-position-embeddings`（如本 repo 不提供则用 `--max-position-embeddings` 作为默认来源，但需要显式规则并写入 config 验证）
- MoE + router semantics:
  - `--moe-router-load-balancing-type seq_aux_loss`
  - `--moe-router-num-groups`, `--moe-router-group-topk`
  - `--moe-router-score-function sigmoid`
  - `--moe-router-topk-scaling-factor`
  - `--moe-router-enable-expert-bias`, `--moe-router-bias-update-rate`
  - `--moe-router-dtype fp32`
- shared experts:
  - `--moe-shared-expert-intermediate-size`

明确 out-of-scope（Stage-2 不实现，避免误导/隐藏问题）：
- `--moe-router-fusion`, `--moe-permute-fusion`（依赖 TE 版本与 fused kernels）
- DeepEP/flex dispatcher
- checkpoint load/save/convert
- MTP

## 2) 环境约束与实现策略（关键决策已锁定）

当前环境事实：
- Transformer Engine: 1.3.0（不具备 upstream DeepSeek-V3 常用的 TE>=2.6 fused MLA/router 支撑）

因此 Stage-2 策略：
- MLA attention core：使用 PyTorch `scaled_dot_product_attention`（SDPA）
- YaRN RoPE：移植 upstream 数学实现（不走 fused apply）
- router：实现 seq_aux_loss/group-limited/sigmoid/scaling_factor/expert_bias（全部走 torch 实现）

原则：
- **Fail fast**：遇到不支持组合直接 `raise`，不做 silent fallback（例如不允许在传了 `--moe-router-fusion` 的情况下悄悄退化到 unfused）

## 3) 关键语义点（必须在实现中显式处理）

### 3.1 Attention mask 语义（SDPA vs Megatron）

本 repo dataloader/训练 utils 的 `attention_mask`：
- bool mask：`True = masked`，`False = allowed`

PyTorch SDPA 的 bool `attn_mask` 语义：
- `True = allowed`，`False = masked`

因此 MLA 的 SDPA 调用必须做一次显式取反：
- `sdpa_mask = ~attention_mask`

### 3.2 MLA head dims（DeepSeek-V3）

- Q/K dot-product 维度：`q_head_dim = qk_head_dim + qk_pos_emb_head_dim`
- V 维度：`v_head_dim`
- SDPA 支持 Q/K/V 最后一维不同（已验证 PyTorch 2.1.2 可行）

### 3.3 Router fixed routing（本 fork 特有）

当前 `pretrain_llama.py` 在 EP>1 时会注入 `config.pre_fixed_routing_results`，`moe_layer.py` 会使用固定 indices 重新从 logits 计算 scores 来保持图连通。

Stage-2 必须保证该 fixed-routing 分支与新 router 语义一致：
- 当 score_function=sigmoid、存在 topk_scaling_factor、启用 seq_aux_loss 等时不能“只 softmax top_logits”完事
- 否则会在 distributed/scaling 路径上产生语义分叉

建议实现策略：
- 将“从 logits + fixed indices 计算 probs/scores”的逻辑收敛到 router helper（单一真源），MoELayer 调用它。

### 3.4 expert bias 的更新时机

upstream 里 expert bias 的更新通常发生在 global batch 粒度，需要 allreduce tokens_per_expert。
本 repo 对齐点：
- 在 `finalize_model_grads.py` 的训练收尾路径里做 bias 更新
- scaling mode world size=1 时 allreduce 应为 no-op，但代码路径不能 crash

## 4) 需要在文档中持续声明的限制（避免 paper 误导）

- 本阶段不以 comp<=5% 为硬门禁（现有 compare 波动风险已记录在 issues.md）
- 只保证“架构标准跑通 + trace 落盘 + 可复现命令”

## 5) Stage-2 执行期补充结论（2026-02-27）

1. **Shared experts 语义补齐**
   - 增加了 `moe_shared_expert_gate`（CLI + `TransformerConfig` + `SharedExpertMLP`）：
     - shared expert 输出可按 DeepSeek 语义走 `sigmoid(linear(hidden_states))` gate。
   - 默认保持关闭，不影响已有 model 路径。

2. **脚本层 fail-fast 与可复现实验增强**
   - `examples/pretrain_deepseek_v3_moe.sh` 增加短跑保护：
     - 自动保证 `LR_WARMUP_ITERS < TRAIN_ITERS`，避免 `OptimizerParamScheduler` 断言失败。
   - 增加诊断开关（仅新脚本生效，不改已有脚本）：
     - `MOE_SHARED_EXPERT_GATE`
     - `MOE_ROUTER_TOPK_SCALING_FACTOR`
     - `USE_BF16`
     - `MOE_GROUPED_GEMM`

3. **分布式 NaN 现象的最新定位边界**
   - `PP=2, EP=2, bf16`（架构标准 smoke）在 distributed 下稳定触发：
     - last pipeline stage ranks (`4..7`) 出现 forward loss NaN。
   - `PP=2, EP=2` 时，即使关闭 `MOE_SHARED_EXPERT_GATE=0`，NaN 仍在。
   - `PP=1, EP=1` distributed/scaling 双模式可稳定完成并落盘 traces。
   - 推断：当前 blocker 更接近 **PP>1 + bf16 路径的数值/执行一致性问题**，而非 shared-expert gate 单点问题。

4. **PP2 bf16 NaN 的 round2 根因与修复（已验证）**
   - 根因定位证据（以 `PP=2, EP=1` 诊断配置先收敛问题空间）：
     - `PP=1` 下 distributed PASS；`PP=2` 下 bf16 FAIL，说明问题与 pipeline forward p2p 路径相关。
     - 打开诊断后首个 non-finite 出现在 last PP stage 的 decoder input（非 loss 端二次传播症状）。
     - fp32（`USE_BF16=0`）下同配置可过，说明属于 bf16 数值/传输一致性问题。
   - 代码修复（最小范围）：
     - `megatron/core/pipeline_parallel/p2p_communication.py`：
       - 新增 `_align_forward_tensor_dtype(...)`，在 `send_forward*` 路径对 forward activation 做 `pipeline_dtype` 对齐。
       - 该路径只在 `config.multi_latent_attention=True` 时启用，避免影响已有非-MLA模型路径。
     - `megatron/core/transformer/moe/moe_utils.py` + `router.py`：
       - 对 sigmoid routing 的归一化改为 fp32 安全归一化并 `clamp` 分母，避免极端 underflow 下 `0/0` 风险。
   - 修复后结果：
     - target distributed smoke（`PP=2,EP=2,bf16`）PASS 且无 NaN assertion。
     - target scaling smoke 仍 PASS，rank `0..7` trace 覆盖完整。

## 6) Stage-2 fidelity round3（2026-02-27）

1. **测量口径回归稳定默认**
   - `examples/pretrain_deepseek_v3_moe.sh` 的 `TRACE_CMD_SYNC_MODE` 默认值从 `event` 回退为 `global`。
   - 原因：event 模式在真实跑测中出现明显离群（单次 forward/backward 可出现百毫秒级尖峰），不适合作为默认 fidelity 采样模式。

2. **optimizer_step timing boundary 对齐（scaling vs distributed）**
   - `megatron/training/training.py`：
     - 新增 `_prepare_scaling_optimizer_step(...)`，将 scaling 路径中的 optimizer prefetch（`get_parameters` / `get_main_grads_for_grad_norm`）移到 traced `optimizer_step` CMD 之外。
   - 目的：与 distributed `train_step` 的 timing 边界一致，避免 scaling 侧多计入 prefetch 耗时。

3. **最新证据与结论**
   - 在 `trace_start=4, train_iters=6, scaling_warmup=3, scaling_profile=3` 下重跑 distributed + scaling（两次 pass）后：
     - 最佳配对之一（`pair=20260227141611`）：
       - `forward_step` median `3.83%`（PASS）
       - `backward_step` median `11.09%`（FAIL）
       - `optimizer_step` median `7.84%`（FAIL）
     - 另一配对（`pair=20260227141950`）：
       - `forward_step` median `7.58%`（FAIL）
       - `backward_step` median `9.86%`（FAIL）
       - `optimizer_step` median `10.54%`（FAIL）
   - 结论：
     - forward 已可在部分稳定 pair 达到阈值；
     - backward 仍对 distributed comm subtraction 高敏感（阶段/运行间波动大）；
     - optimizer 仍存在 scaling 系统性偏高（约 +8%~+12% 中位数）残差。

## 7) Stage-2 fidelity round4（2026-02-27）

1. **scaling optimizer pre-CMD side effects 继续对齐**
   - `megatron/training/training.py`：
     - `_prepare_scaling_optimizer_step(...)` 在 existing prefetch 之外，新增与 distributed `train_step` 一致的 `numel` pre-scan：
       - `sum(param.numel() for param in params)`
       - `sum(grad.numel() for grad in grads_for_norm)`
   - 目标：复制 distributed 侧 optimizer 进入 CMD 之前的完整准备副作用，进一步压缩 optimizer 差距。

2. **定向证据（rank0 optimizer）**
   - paired to distributed `rank0@20260227141950`：
     - before patch（`scaling rank0@20260227142506`）：`optimizer_step diff = 12.76%`
     - after patch（`scaling rank0@20260227144128`）：`optimizer_step diff = 6.02%`
   - 结论：该对齐改动有效降低 optimizer 偏差，但仍略高于 5% 门限。

3. **全量配对新结论**
   - 使用 interleaved two-pass scaling（`cache_tag=stage2_fidelityfix5_interleave`）+ distributed rerun（`ts=20260227145522`）：
     - `forward_step` rank median `4.02%`（PASS）
     - `backward_step` rank median `5.11%`（FAIL，接近门限）
     - `optimizer_step` rank median `7.68%`（FAIL，较 round3 有改善）
   - 备注：
     - 同期出现一次 `MASTER_PORT` 占用冲突（`Address already in use`）；通过切换高位端口区间（`MASTER_PORT=7400/7600/7700/7800`）后复现稳定。

4. **当前 root-cause 状态（更新）**
   - `optimizer_step` 残差并非单纯 timing boundary 问题，仍存在运行态相关的系统性偏高（当前约 `+6%`~`+8%` 中位区间）。
   - `backward_step` 已逼近 5%，但对 subtraction 口径与 run-state 仍敏感。

## 8) Stage-2 fidelity round5（2026-02-27）

1. **新定位的 temporal fidelity 根因（pipeline replay 维度）**
   - scaling replay cache 之前只按 `dst_rank` 存一个文件：
     - `activation_to_rank{dst}.pt`
     - `grad_to_rank{dst}.pt`
   - 在 `trace_start=4, train_iters=6, warmup=3, profile=3` 场景下，这会导致消费侧 rank 在 profile 期间反复加载“最后一次覆盖写入”的同一 replay tensor，而不是逐 iteration 对应的 tensor。
   - 对 MoE 路径而言，这会引入额外的时序失配（同一目的 rank 的多次 profile step 输入/梯度被压缩为单样本 replay）。

2. **round5 修复方案（已落地）**
   - `megatron/training/training.py`
     - activation/grad replay cache 的写入路径改为 iteration-indexed：
       - `activation_to_rank{dst}_iter{current_iter}.pt`
       - `grad_to_rank{dst}_iter{current_iter}.pt`
     - replay grad 读取优先按 `current_iter` 命中迭代文件；旧格式 `grad_to_rank{dst}.pt` 保留兼容读取。
   - `megatron/profiler/utils.py`
     - 新增 `resolve_scaling_replay_path(cache_dir, rank_id, current_iter)`。
     - `sim_forward_step` replay 加载改为优先迭代文件，旧格式作为兼容回退。
   - 单测：
     - 新增 `tests/unit_tests/profiler/test_scaling_replay_cache_paths.py`（iter优先/legacy回退/缺失返回None/cache_dir为空）。

3. **round5 实测结论（最新）**
   - 功能正确性：
     - scaling rank3→rank7 probe PASS，cache 目录出现 `*_iter1..5.pt` 文件。
   - fidelity（pair `20260227145502`）仍未过线：
     - subtract-comm：
       - `forward_step` median `4.23%`（PASS）
       - `backward_step` median `14.18%`（FAIL）
       - `optimizer_step` median `7.57%`（FAIL）
     - no-subtract：
       - `forward_step` median `14.80%`（FAIL）
       - `backward_step` median `17.53%`（FAIL）
       - `optimizer_step` median `7.57%`（FAIL）

4. **当前状态更新**
   - round5 已修复一个明确的 replay 时序对齐缺陷；
   - 但 stage-2 `<=5%` 目标仍未满足，主要残余集中在 backward/optimizer，且 distributed run-to-run 波动对结论敏感。

## 9) Stage-2 protocolfix8 + optimizer proposal（2026-02-27, design-only）

1. **non-semantic protocol 执行结果（本轮）**
   - 已执行固定端口段 + 固定 rank-order + repeated pairing：
     - 端口段：`9400/9500/9600` 家族；
     - rank-order：`0,4,1,5,2,6,3,7`（对照 `0..7`）。
   - 关键结论：
     - 最优单次：`forward=3.06%`, `backward=7.51%`, `optimizer=5.97%`；
     - backward 最近门限值：`5.66%`（仍略高于 5%）；
     - 说明 protocol 对齐已显著缩小误差，但仍未稳定跨过 backward/optimizer 门限。

2. **可能触及 scaling 执行语义的 optimizer 优化提案（仅方案，未改代码）**
   - 目标：
     - 继续压缩 `optimizer_step` residual，同时保持 distributed/scaling 语义可对齐，并保留 rank-level comp 可观测性。
   - 提案名称：**Rank-local optimizer microphase decomposition（trace-only segmentation）**
   - 核心思路：
     1. 在 scaling `optimizer_step` 内部增加细粒度 microphase trace（仅记录，不改变计算）：
        - `optimizer_main_update`（主参数更新）
        - `optimizer_state_update`（state tensor update）
        - `optimizer_post_update`（post hooks / grad clear）
     2. distributed 侧在相同逻辑点增加同名 microphase trace（保持同构），并在 compare 里做 phase-aware 对齐。
     3. top-level `optimizer_step` 保持原定义，microphase 只用于差异归因和后续等价对齐，不改学习率/梯度/参数更新语义。
   - 为什么“可能触及语义”：
     - 尽管目标是 trace-only，但需要在 optimizer hot path 增加额外同步点或 trace 边界，可能改变 kernel launch 排布与 overlap，进而轻微影响 wall-time。
   - 语义对齐保证（设计约束）：
     - 不改 optimizer 算法、不改参数更新次序、不改 grad lifecycle；
     - distributed 与 scaling 同步加点，同名 phase 对齐，避免只改单边；
     - 默认关闭（通过新 flag 启用），避免影响已有模型默认路径性能。
   - rank-level comp 表达：
     - 每个 rank 输出 `optimizer_step` + microphase duration；
     - compare 输出 phase-level diff 与总和 diff，支持定位“哪个 phase 造成 residual”。
   - 验证计划（代码变更前先约定）：
     - 单测：trace 结构完整性（phase 名称、顺序、总和一致性）；
     - 集成：同一 run_config 下 distributed/scaling 对比；
     - 门限：先看 `optimizer_step` op-rank-median，再看 phase 归因是否稳定。

## 10) Stage-2 optimizer microphase 实施结果（2026-02-27, round9）

1. **实现范围（已落地）**
   - 新增 flag（默认关闭）：
     - `--trace-optimizer-microphases`
   - 在 distributed/scaling 两侧增加同构 microphase trace 点：
     - `optimizer_main_update`
     - `optimizer_state_update`
     - `optimizer_post_update`
   - 保留原有 top-level `optimizer_step` CMD，不改变其名称和存在性。

2. **语义约束执行情况**
   - 未修改 optimizer 算法、参数更新顺序、学习率调度逻辑条件。
   - 默认路径（未开启新 flag）不新增 microphase CMD，避免影响既有模型默认性能与 trace 口径。
   - microphase 仅作为附加诊断视图；开启后只增加 trace 边界与记录，不引入 fallback 分支。

3. **代码触点**
   - `megatron/training/arguments.py`
     - 新增 `--trace-optimizer-microphases`。
   - `megatron/training/training.py`
     - 新增 `_optimizer_microphase_cmd(...)` 等 helper；
     - distributed `train_step` 和 scaling profiling 路径均接入同名 phase；
     - 扩展 `simu_micro_batch_ids` 字典，保证 phase batch_id 可独立递增记录。

4. **验证结论（单测）**
   - 新增 `tests/unit_tests/test_training_optimizer_microphase.py`：
     - parser 默认值/开启值校验；
     - microphase key 注入校验；
     - phase 顺序/存在性校验；
     - invalid phase fail-fast 校验。
   - 当前结果：
     - `pytest -q tests/unit_tests/test_training_optimizer_microphase.py` → `6 passed`。

## 11) Stage-2 optimizer microphase fidelity 结果（2026-02-27, round10）

1. **执行协议**
   - distributed/scaling 均启用：
     - `TRACE_START=4`, `TRAIN_ITERS=6`
     - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
     - `TRACE_OPTIMIZER_MICROPHASES=1`
   - scaling 固定执行协议：
     - rank-order：`0,4,1,5,2,6,3,7`
     - run1/run2/run3（端口 `9630/9631/9632`），同一 distributed baseline (`ts=20260227174546`)。

2. **phase-aware 单次证据（op-rank-median）**
   - run1:
     - `forward=8.34%`, `backward=6.81%`, `optimizer_step=11.19%`
     - `optimizer_main_update=10.86%`
     - `optimizer_state_update=33.33%`
     - `optimizer_post_update=0.00%`
   - run2:
     - `forward=10.53%`, `backward=13.39%`, `optimizer_step=10.43%`
     - `optimizer_main_update=9.75%`
     - `optimizer_state_update=27.78%`
     - `optimizer_post_update=16.67%`
   - run3:
     - `forward=7.79%`, `backward=8.17%`, `optimizer_step=13.22%`
     - `optimizer_main_update=12.52%`
     - `optimizer_state_update=30.00%`
     - `optimizer_post_update=12.50%`

3. **median-of-runs（3 runs, op-rank-median）**
   - `forward_step=8.34%`
   - `backward_step=8.17%`
   - `optimizer_step=11.19%`
   - `optimizer_main_update=10.86%`
   - `optimizer_state_update=30.00%`
   - `optimizer_post_update=12.50%`

4. **结论（当前阶段）**
   - microphase trace 已提供归因能力，但 fidelity 未达 `<=5%`：
     - optimizer 主残差并非集中在 `state_update/post_update`；
     - `optimizer_main_update` 本身仍在 ~10% 量级，说明 residual 主要仍来自 optimizer 主更新阶段的跨模式执行差异。
   - `optimizer_state_update` / `optimizer_post_update` 由于绝对时长很短（约 `0.01~0.05ms`），相对误差易放大；应结合绝对时长解读，不宜单独作为主 gate。

## 12) Stage-2 fidelity round11（2026-02-27）：语义触及优化试验（scaling only, default-off）

1. **根因聚焦（round10 后）**
   - phase-aware 结果显示 `optimizer_main_update` 残差仍主导，且主要集中在 `PP stage1` ranks。
   - 推断 scaling replay I/O（尤其 grad replay 写回）在 profiling iteration 内的时序会对后续测量产生扰动。

2. **试验 A：replay grad 写回时序可选对齐（已落地）**
   - 新增 fake/scaling 参数：
     - `--scaling-replay-write-phase {pre_optimizer,post_optimizer}`（default=`pre_optimizer`）
   - `training.py` 行为：
     - `post_optimizer` 下将 `grad_to_rank*.pt` 写回推迟到 `optimizer_step` 之后；
     - 保持 replay 语义可用（同 iteration 文件仍写出），默认路径不变。
   - 脚本接线：
     - `examples/pretrain_deepseek_v3_moe.sh` 新增 `SCALING_REPLAY_WRITE_PHASE`（校验 + 透传）。

3. **试验 B：scheduler increment 对齐开关（已落地，默认关闭）**
   - 新增 fake/scaling 参数：
     - `--scaling-align-scheduler-increment`（default-off）
   - 作用：
     - scaling scheduler increment 由 `fake_dp` 切换为真实 `data_parallel_size` 口径（仅启用时生效）。
   - 目的：
     - 验证 optimizer dynamics 对残差的贡献是否明显。

4. **round11 证据（同一 distributed baseline: `20260227182456`）**
   - `post_optimizer`（A）：
     - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepost_run1.log`
     - op-rank-median:
       - `forward=10.74%`
       - `backward=12.71%`
       - `optimizer_step=6.56%`
       - `optimizer_main_update=6.21%`
   - `pre_optimizer`（对照）：
     - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepre_run1.log`
     - op-rank-median:
       - `forward=14.09%`
       - `backward=19.52%`
       - `optimizer_step=6.47%`
       - `optimizer_main_update=6.04%`
   - `post_optimizer + align_scheduler_increment`（A+B）：
     - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepost_aligninc_run1.log`
     - op-rank-median:
       - `forward=8.10%`
       - `backward=10.92%`
       - `optimizer_step=7.16%`
       - `optimizer_main_update=6.50%`

5. **round11 结论**
   - A（post write）对 `forward/backward` 有稳定改善迹象；对 `optimizer_main_update` 仅小幅波动（~6% 区间），尚未跨过 `<=5%`。
   - B（scheduler increment 对齐）在本轮未带来 optimizer 主指标收益，且 `optimizer_step` 有回退风险；建议继续保持 default-off，仅作实验开关。
   - 当前最稳健结论：
     - **主要残差仍在 `optimizer_main_update` 的 stage1 ranks；distributed baseline run-to-run 漂移仍显著影响 gate 结论。**
