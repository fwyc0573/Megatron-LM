## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Created stage-1 implementation plan for Qwen3-MoE + DeepSeek-V3-Proxy scaling-mode port |
| 2026-02-27 | Added stage-2 implementation plan for DeepSeek-V3 (architecture standard) distributed+scaling bring-up (MLA/YaRN RoPE/router semantics/shared experts) |
| 2026-02-27 | Added stage-2 fidelity round4 execution note: scaling optimizer pre-CMD side-effect parity and refreshed pairing protocol status |

# Stage-1 Plan: Qwen3-MoE + DeepSeek-V3-Proxy Port

## Scope

- Port Qwen3-30B-A3B (stage-1 subset) and DeepSeek-V3-Proxy (MHA simplified) into current tracing/scaling Megatron-LM.
- Keep existing tracing/scaling framework behavior unchanged as first priority.
- Deliver dual-mode scripts:
  - Distributed tracing mode
  - Scaling mode (single GPU, sequential per-rank)

## Implementation Steps

1. Extend public CLI/config surface
   - Add `--moe-layer-freq`, `--moe-ffn-hidden-size`, `--rotary-base`
   - Thread fields into `TransformerConfig` and validation logic
2. Activate MoE layer pattern + expert FFN size
   - Support mixed dense/MoE decoder layer patterns by `moe_layer_freq`
   - Use `moe_ffn_hidden_size` on expert MLP paths only
3. Integrate model entrypoint changes
   - Reuse `pretrain_llama.py`
   - Select block spec generation for MoE models
   - Keep routing pre-generation active in distributed + scaling when EP>1
4. Add runnable scripts
   - `examples/pretrain_qwen3_30b_a3b_moe.sh`
   - `examples/pretrain_deepseek_v3_proxy_moe.sh`
5. Add tests and execute validation
   - Unit tests for parser/config/layer pattern/expert MLP
   - Integration smoke runs: 2 models × 2 modes
6. Document simplifications and known limitations

## Acceptance Criteria

- New args parse correctly and are visible in `TransformerConfig`.
- DeepSeek proxy layer pattern (`[0]*3 + [1]*11`) builds dense+MoE mixed block specs.
- Expert MLPs honor `moe_ffn_hidden_size`; dense MLP remains on `ffn_hidden_size`.
- Qwen3 and DeepSeek scripts can run in distributed and scaling modes with trace enabled.
- Stage-1 limitations are explicitly documented.

# Stage-2 Plan: DeepSeek-V3（架构标准）Distributed + Scaling 跑通

## Goal

在当前 tracing/scaling fork 上，实现“标准 DeepSeek-V3（架构标准）”的 **distributed + scaling** 双模式可运行与可 tracing：

- distributed：真实多 GPU（Realistic Mode tracing）可跑通并产出 `realistic_trace/<run_config>/...`
- scaling：单 GPU 顺序 fake ranks 0..7 可跑通并产出 `profiler_log/<run_config>/...`

> 本阶段只要求“架构标准跑通 + trace 落盘 + 可复现验证”；DeepEP / ckpt / MTP 等不在 scope。

## Definition: “标准 DeepSeek-V3（架构标准）”包含哪些特性

对齐 `latest-megatron/Megatron-MoE-ModelZoo/model_configs/benchmarking/DeepSeek-V3.yaml` 中与架构相关的关键特性：

1. Attention：启用 **MLA (multi-latent attention)**（不再使用 proxy 的 MHA 简化）
2. RoPE：使用 **YaRN RoPE**（`rotary_scaling_factor/mscale/mscale_all_dim` 等）
3. MoE router：支持 DeepSeek 的 router 语义组合：
   - `--moe-router-load-balancing-type seq_aux_loss`
   - group-limited routing（`moe_router_num_groups/moe_router_group_topk`）
   - `--moe-router-score-function sigmoid`
   - `--moe-router-topk-scaling-factor`
   - `--moe-router-enable-expert-bias` + `--moe-router-bias-update-rate`
   - `--moe-router-dtype fp32`（至少 fp32 routing）
4. shared experts：支持 `--moe-shared-expert-intermediate-size`

## Non-Goals（明确不做）

- DeepEP / flex dispatcher / fused_a2a
- checkpoint 转换与加载（HF -> Megatron）
- MTP（multi-token prediction）
- TE fused MLA / fused router / permute fusion（当前环境 TE=1.3.0，不具备 upstream 依赖的 TE>=2.6 能力）
- paper-level comp <=5% 硬门禁（本 repo 现存 forward/bwd comp 波动问题仍未完全收敛）

## Upstream References（实现对照来源）

- Config baseline:
  - `latest-megatron/Megatron-MoE-ModelZoo/model_configs/benchmarking/DeepSeek-V3.yaml`
- Core modules:
  - `latest-megatron/Megatron-LM/megatron/core/transformer/multi_latent_attention.py`
  - `latest-megatron/Megatron-LM/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py`
  - `latest-megatron/Megatron-LM/megatron/core/transformer/moe/router.py`
  - `latest-megatron/Megatron-LM/megatron/core/transformer/moe/moe_utils.py`
  - `latest-megatron/Megatron-LM/megatron/core/transformer/moe/shared_experts.py`
  - `latest-megatron/Megatron-LM/megatron/core/models/gpt/gpt_layer_specs.py`

## Public Interface Changes（必须精确落地）

### A) CLI 参数（`megatron/training/arguments.py`）

新增/扩展（DeepSeek-V3.yaml 所需）：

- `--multi-latent-attention` (bool)
- `--q-lora-rank` (int)
- `--kv-lora-rank` (int)
- `--qk-head-dim` (int)
- `--qk-pos-emb-head-dim` (int)
- `--v-head-dim` (int)
- `--rope-type` (choices: `rope|yarn`, optional; 默认由 config 决定)
- `--rotary-scaling-factor` (float)
- `--original-max-position-embeddings` (int)
- `--beta-fast` / `--beta-slow` (float)
- `--mscale` / `--mscale-all-dim` (float)

MoE router 相关：
- `--moe-router-load-balancing-type` 扩展 choices：增加 `seq_aux_loss`
- `--moe-router-num-groups` (int, optional)
- `--moe-router-group-topk` (int, optional)
- `--moe-router-score-function` (choices: `softmax|sigmoid`)
- `--moe-router-topk-scaling-factor` (float, optional)
- `--moe-router-enable-expert-bias` (bool)
- `--moe-router-bias-update-rate` (float)
- `--moe-router-dtype` (choices: `fp32|fp64|none`)

shared experts：
- `--moe-shared-expert-intermediate-size` (int, optional)

### B) TransformerConfig（`megatron/core/transformer/transformer_config.py`）

新增字段（默认不破坏其他模型）：
- `multi_latent_attention: bool = False`
- `q_lora_rank, kv_lora_rank, qk_head_dim, qk_pos_emb_head_dim, v_head_dim`
- `rope_type, rotary_scaling_factor, original_max_position_embeddings, beta_fast, beta_slow, mscale, mscale_all_dim`
- `moe_router_score_function, moe_router_num_groups, moe_router_group_topk, moe_router_topk_scaling_factor`
- `moe_router_enable_expert_bias, moe_router_bias_update_rate, moe_router_dtype`
- `moe_shared_expert_intermediate_size`

并在 `__post_init__` 中做 fail-fast 验证：
- MLA 开启时，head dim、rope_type 与必要参数必须完整且合法
- group-limited routing 时，num_groups/group_topk/topk 的整除与边界必须合法

## Implementation Steps（按依赖顺序执行，必须 TDD）

> 说明：每个 Task 都需要 unit test 覆盖新增/修改的代码路径；集成 smoke 要覆盖 distributed+scaling 两个模式。

### Task 1: 参数面（argparse）补齐 + 单测
- Modify: `megatron/training/arguments.py`
- Test (new): `tests/unit_tests/test_deepseek_v3_args.py`
- Run:
  - `pytest tests/unit_tests/test_deepseek_v3_args.py -v`

### Task 2: Config 面（TransformerConfig）补齐 + 验证规则 + 单测
- Modify: `megatron/core/transformer/transformer_config.py`
- Modify (兼容 YAML 默认值): `megatron/training/yaml_arguments.py`
- Test (new): `tests/unit_tests/transformer/test_deepseek_v3_config_validation.py`
- Run:
  - `pytest tests/unit_tests/transformer/test_deepseek_v3_config_validation.py -v`

### Task 3: YaRN RoPE 最小实现 + 单测
- Create: `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py`
- Modify: `megatron/core/models/common/embeddings/__init__.py`
- Test (new): `tests/unit_tests/transformer/test_yarn_rotary_embedding.py`
- Run:
  - `pytest tests/unit_tests/transformer/test_yarn_rotary_embedding.py -v`

### Task 4: MLA（基于 PyTorch SDPA 的 core，非 TE fused）+ 单测
- Create: `megatron/core/transformer/multi_latent_attention.py`
- Modify: `megatron/core/models/gpt/gpt_layer_specs.py`（根据 `config.multi_latent_attention` 选择 MLA）
- Modify: `megatron/core/models/gpt/gpt_model.py`（MLA 模式禁止构建/传入普通 RoPE）
- Test (new): `tests/unit_tests/transformer/test_multi_latent_attention.py`
- Run:
  - `pytest tests/unit_tests/transformer/test_multi_latent_attention.py -v`

### Task 5: DeepSeek router 语义（seq_aux_loss + group-limited + sigmoid + scaling_factor + router_dtype）+ 单测
- Modify: `megatron/core/transformer/moe/router.py`
- Modify: `megatron/core/transformer/moe/moe_utils.py`
- Modify: `megatron/core/transformer/moe/moe_layer.py`（fixed routing 分支需与新 router 语义一致）
- Modify (tests): `tests/unit_tests/transformer/moe/test_routers.py`
- Run:
  - `pytest tests/unit_tests/transformer/moe/test_routers.py -v`

### Task 6: shared experts（最小可用，不做 overlap）+ 单测
- Create: `megatron/core/transformer/moe/shared_experts.py`
- Modify: `megatron/core/transformer/moe/moe_layer.py`（MoE 输出 + shared experts 输出融合）
- Test (new): `tests/unit_tests/transformer/moe/test_shared_experts.py`
- Run:
  - `pytest tests/unit_tests/transformer/moe/test_shared_experts.py -v`

### Task 7: expert bias 更新（训练阶段集成）+ 单测
- Modify: `megatron/core/distributed/finalize_model_grads.py`
- Test (new): `tests/unit_tests/transformer/moe/test_expert_bias_update.py`
- Run:
  - `pytest tests/unit_tests/transformer/moe/test_expert_bias_update.py -v`

### Task 8: 新增标准 DeepSeek-V3 训练脚本（distributed + scaling）
- Create: `examples/pretrain_deepseek_v3_moe.sh`
- 需求：
  - `MODE=distributed|scaling`
  - `MODEL_PROFILE=smoke|full`（默认 smoke；smoke 也必须开启 MLA/YaRN/router/shared experts）
  - scaling 模式顺序 fake ranks 0..7，trace 正常落盘

### Task 9: 集成 smoke 验证（必须落盘 trace）
- Scaling smoke:
  - `MODE=scaling MODEL_PROFILE=smoke TRACE_START=1 TRAIN_ITERS=3 bash examples/pretrain_deepseek_v3_moe.sh`
  - 验收：`profiler_log/<run_config>/...rank0..rank7...txt` 文件齐全
- Distributed smoke:
  - `MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=1 TRAIN_ITERS=3 bash examples/pretrain_deepseek_v3_moe.sh`
  - 验收：`realistic_trace/<run_config>/...rank0..rank7...txt` 文件齐全

### Task 10: compare（非硬门禁，但必须可复现并落盘报告）
- Run:
  - `python tests/performance/compare_qwen_trace_comp.py --distributed-dir realistic_trace/<run_config> --scaling-dir profiler_log/<run_config> --ranks 0,1,2,3,4,5,6,7 --ops forward_step,backward_step,optimizer_step --threshold-pct 5`
- 产物：将 compare 输出保存到本 task 的 `logs/` 或 test report 中

## Acceptance Criteria（验收阈值）

### Gate A（Stage-2 必须达成）
1. 单测全部通过（至少包含本计划列出的新增/修改测试）
2. `MODE=scaling MODEL_PROFILE=smoke` 能完成 fake ranks 0..7，trace 文件齐全
3. `MODE=distributed MODEL_PROFILE=smoke` 能完成 ranks 0..7，trace 文件齐全
4. 关键架构路径确实启用：
   - MLA 已启用（非 proxy 的 MHA）
   - YaRN RoPE 已启用
   - router 路径包含 seq_aux_loss + group-limited + sigmoid + shared experts

### Gate B（本阶段不做硬门禁，但必须产出）
- compare 脚本可跑通并输出可读报告（PASS/FAIL 均可，但必须可复现并记录）

## Test Report Storage（必须）
- 在同一 task 目录落盘：`task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-27_deepseek_v3_stage2_docs.md`
- 内容必须包含：环境、命令、验收点、结果证据
