# Scaling Mode Rank-Skipping Exploratory Analysis (Dense & MoE)

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Initial analysis for dense and MoE models |
| 2026-03-02 | Added engine MoE simulate/profile debug notes and linked artifacts |
| 2026-03-03 | Consolidated dense/MoE rank-skipping conclusions, 8-GPU case deep dive, DP/EP communication semantics, and simulator guidance |

---

## 0) 结论先行（可直接用于后续测量策略）

### Dense 模型
1. **可以跳过 DP/TP 重复测量**，主因是 Dense 计算分片只由 `PP/TP` 结构决定，DP 仅影响数据与梯度同步语义，不改变核心算子形状。  
2. **最小保守测量集**：每个 PP stage 测 1 个代表 rank（`tp_rank=0, dp_rank=0`），即 `PP` 个 rank。  
3. 若确认中间 PP stage 的 layer pattern 完全一致，可进一步压缩为 `min(PP, 3)`（首段/中段代表/尾段）。

### MoE 模型
1. **EP 维度通常不能跳过**：不同 `exp_rank` 对应不同 local experts 和 token 分配，`GroupedGEMM` 耗时取决于 `tokens_per_expert`。  
2. **当前 fork 的 Scaling 语义下**，同一个 `(pp_rank, exp_rank)` 的不同 DP rank 往往被“预固定 routing 结果”压平成近似重复。  
3. **推荐最小测量集（当前 fork）**：`PP × EP`（每个 PP stage 保留所有 EP rank，TP/DP 选代表）。  
4. 若目标是逼近真实 distributed 动态 routing，建议加 **DP spot-check**；若波动超过阈值，不应把 DP 全部当 duplicate。

### 8-GPU case（`PP=2, TP=1, EP=2, DP=4, world=8`）
- **主测量集合**：`{0, 1, 4, 5}`（覆盖 2 个 PP stage × 2 个 EP rank）。  
- **建议 spot-check**：`{2, 3, 6, 7}`，用于验证同 `(PP,EP)` 下不同 DP 的时延一致性。  

---

## 1) 关键代码证据索引（Dense / MoE / 通信）

1. **Scaling 驱动与 fake rank 注入**  
   - `megatron/training/training.py:371-385`（把 `pp_rank/dp_rank/tp_rank/exp_rank` 注入 args）  
   - `megatron/training/training.py:492-523`（Scaling fake-rank 主流程）

2. **PP 切层（决定 stage 计算差异）**  
   - `megatron/core/transformer/transformer_block.py:31-40`（`num_layers // fake_pp`）  
   - `megatron/core/models/gpt/gpt_layer_specs.py:191-204`（按 `pp_rank` slice local layers）

3. **Dense TP 分片一致性**  
   - `megatron/core/tensor_parallel/layers.py:745-750`（`ColumnParallelLinear` 用 `fake_tp`）  
   - `megatron/core/transformer/attention.py:78-86`（head/query_group 按 `fake_tp` 切分）  
   - `megatron/core/transformer/custom_layers/transformer_engine.py:142-147`、`262-269`、`530-534`（TE 层 tp_size 取 config）

4. **MoE expert 分配与 token 依赖计算**  
   - `megatron/core/transformer/moe/moe_layer.py:42-54`（`exp_rank -> local_expert_indices`）  
   - `megatron/core/transformer/moe/experts.py:162-168`（`GroupedGEMM` 依赖 `tokens_per_expert`）

5. **Scaling 下 MoE routing/dispatch 预固定机制**  
   - `pretrain_llama.py:93-107`（`set_pre_distribution_moe(config)`）  
   - `megatron/profiler/moe/sim_routing.py:61-63`（`seed + ep_rank` 生成每个 EP routing）  
   - `megatron/profiler/moe/sim_dispatching.py:34-43`（每个 EP 的 token 计数）  
   - `megatron/core/transformer/moe/moe_layer.py:167-175`（按 `exp_rank` 取预固定 indices）  
   - `megatron/core/transformer/moe/token_dispatcher.py:345-348`（按 `exp_rank` 取 `num_local_tokens_per_expert`）

6. **通信发生位置与通信组**  
   - `megatron/core/pipeline_parallel/schedules.py:1897-1927`（训练 finalize 调用 `finalize_model_grads`）  
   - `megatron/core/distributed/finalize_model_grads.py:177-179`（`model_chunk.finish_grad_sync()`）  
   - `megatron/training/training.py:1077-1078`（DDP 绑定 dense DP group + expert DP group）  
   - `megatron/core/distributed/distributed_data_parallel.py:109-113`（按 `param.allreduce` 划分 dense/expert 参数）  
   - `megatron/core/distributed/distributed_data_parallel.py:169-173`（expert 参数用 `expert_data_parallel_group`）  
   - `megatron/core/parallel_state.py:664-692`（`tp-ep`、`ep`、`dp_modulo_ep` 组初始化）  
   - `megatron/profiler/sim_parallel_state.py:291-357`（sim groups 生成逻辑）

7. **Scaling 通信拦截（只记录不执行）**  
   - `megatron/profiler/comm_utils/interception_comm.py:25-27`、`47-48`、`58-61`  
   - `megatron/core/tensor_parallel/mappings.py:468-495`（`all_to_all` scaling branch 仅 shape/data 仿真）

8. **当前 simulator rank 选择策略（用于对齐后续流程）**  
   - `megatron-sim-engine/src/core/simu_engine.py:4292-4303`（Dense 选每个 PP 的代表 rank）  
   - `megatron-sim-engine/src/core/simu_engine.py:4276-4284`（MoE 当前保留 all ranks）

---

## 2) Dense 模型详细分析

### 2.1 为什么 DP/TP 可视作重复（主结论）

1. **PP 决定“哪几层在本 rank 上执行”**，DP 不参与切层。  
   - `get_num_layers_to_build` 仅使用 `num_layers/fake_pp`：`transformer_block.py:31-40`  
   - `get_gpt_decoder_block_spec` 按 `pp_rank` 切 `layer_specs`：`gpt_layer_specs.py:191-204`

2. **TP 切分在 Dense 主干上是均匀的**（线性层、attention heads/group）。  
   - `ColumnParallelLinear`：`output_size_per_partition = divide(output_size, fake_tp)`：`layers.py:745-750`  
   - attention head/group 分区：`attention.py:85-86`

3. **Scaling 模式下通信被拦截，TP/DP collectives 不执行真实 NCCL**。  
   - `interception_comm.py:25-27`、`47-48`、`58-61`

### 2.2 Dense 需要保留 PP 的原因

1. **不同 PP stage 的计算图不同**（特别是首尾 stage）。  
   - 首段有 embedding / 输入准备；尾段有 output projection / loss path。  
2. 即使层数均匀，**首尾 stage 与中间 stage 的 kernel 组合仍可能不同**。  

### 2.3 Dense 边界与例外（必须记录）

1. `RowParallelLinear` 构造里仍直接读真实 TP world size：`layers.py:1011-1012`。  
   - 在 TE 路径下通常由 TE wrapper 的 `tp_size` 机制兜住，但这仍是潜在不一致点。

2. `get_batch_on_this_tp_rank` 在 `tp_rank!=0` 时构造随机 token/label，且 `broadcast` 在 scaling 是 no-op：  
   - `utils.py:330-360` + `interception_comm.py:47-48`。  
   - 这会让不同 TP rank 的数据值不同（虽然 shape 一样），可能带来微小 timing 噪声。

3. backward replay fallback seed 是 rank-aware：`training.py:433-442`。  
   - 会引入 rank-specific 数值路径，通常不改变宏观 compute 拓扑，但可造成细微波动。

### 2.4 Dense 推荐测量集合

1. **保守推荐**：`{(pp, tp=0, dp=0) | pp in [0..PP-1]}`。  
2. **激进推荐（需先验证）**：`{pp=0, pp=1(中段代表), pp=PP-1}`，仅当中间 stage 完全同构。  

---

## 3) MoE 模型详细分析

### 3.1 EP 为什么不等价于 DP duplicate

1. 每个 `exp_rank` 绑定不同 local expert ID 区间：`moe_layer.py:42-54`。  
2. MoE compute 核心 `GroupedGEMM` 直接依赖 `tokens_per_expert`：`experts.py:162-168`。  
3. 因此 **不同 EP rank 的 token 负载差异会直接改变 compute 时延**。

### 3.2 当前 fork 下“同 EP 不同 DP”为何常近似重复

1. 模型构建时会预生成并写入 routing/dispatch 结果：`pretrain_llama.py:93-107`。  
2. MoE forward 使用 `pre_fixed_routing_results[exp_rank]`：`moe_layer.py:167-175`。  
3. dispatcher 在 scaling 也按 `exp_rank` 读取 `num_local_tokens_per_expert`：`token_dispatcher.py:345-348`。  

这意味着：
- 同 `(pp_rank, exp_rank)` 的不同 DP rank 在当前实现里经常共享同一组 routing 统计，
- 计算耗时被“人为压平”为 duplicate（这对 profiling 稳定性友好，但不等价于真实动态 routing）。

### 3.3 通信语义：EP 通信 vs DP 通信（用户疑问核心）

#### A) EP 通信何时发生、目的是什么？
1. 在 MoE token dispatch/unpermutation 时发生 `all_to_all`：  
   - `token_dispatcher.py:477-483`（dispatch）  
   - `token_dispatcher.py:533-539`（回传）
2. 目的：**token 交换**，把 token 发到目标专家所在 rank，再回收结果。  
3. 这不是参数梯度同步。

#### B) DP 通信涉及哪些 rank、什么时候发生？
1. 在训练 finalize（或 overlap 流程）执行梯度同步：  
   - 调用入口：`schedules.py:1897-1927`  
   - 实际调用：`finalize_model_grads.py:177-179`

2. DDP 内部有两类参数缓冲：  
   - **dense params** 走 `data_parallel_group`  
   - **expert params** 走 `data_modulo_expert_parallel_group`（通过 `expert_data_parallel_group` 传入）  
   证据：`training.py:1077-1078`、`distributed_data_parallel.py:109-113`、`169-173`。

3. 如果 `overlap_grad_reduce=False`，allreduce 在 finalize 同步触发；若 `True`，可能在 backward 过程中 bucket ready 即发起，finalize 主要做 wait：  
   - `param_and_grad_buffer.py:165-167`、`175-190`。

### 3.4 MoE 推荐测量集合

1. **当前 fork（含 pre-fixed routing）主推荐**：`PP × EP`。  
2. **为防止 DP routing 波动漏检，建议加 DP spot-check**：每个 `(PP,EP)` 至少再抽 1 个同组 DP duplicate。  
3. 若 spot-check 超阈值，应升级到更细粒度（例如 `PP × EP × DP_mod_EP` 甚至全量）。

---

## 4) 8-GPU Case 深入复盘

### 4.1 配置
- `world_size=8, PP=2, TP=1, EP=2, DP=4, num_experts=64`
- 默认并行顺序（训练初始化）为 `tp-cp-ep-dp-pp`：`initialize.py:271`

### 4.2 组结构（sim 生成逻辑）
由 `sim_parallel_state.py:291-357` 生成：

1. `dp_groups`: `[0,1,2,3]`, `[4,5,6,7]`  
2. `pp_groups`: `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`  
3. `exp_groups`（EP 组）: `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`  
4. `dp_modulo_exp_groups`: `[0,2]`, `[1,3]`, `[4,6]`, `[5,7]`

### 4.3 rank 映射（关键澄清）

Stage 0 (`pp=0`):
- rank0: `dp=0, exp=0`
- rank1: `dp=1, exp=1`
- rank2: `dp=2, exp=0`
- rank3: `dp=3, exp=1`

Stage 1 (`pp=1`) 同理：
- rank4: `dp=0, exp=0`
- rank5: `dp=1, exp=1`
- rank6: `dp=2, exp=0`
- rank7: `dp=3, exp=1`

**因此不是**“dp0/dp1 同 exp0，dp2/dp3 同 exp1”。

### 4.4 这个 case 的推荐测量 rank

1. **主测量**：`{0,1,4,5}`  
   - 覆盖 `PP(2) × EP(2)` 的最小组合。

2. **spot-check**：`{2,3,6,7}`  
   - 对比 `(0 vs 2)`, `(1 vs 3)`, `(4 vs 6)`, `(5 vs 7)`，检验同 `(PP,EP)` 下 DP duplicate 假设。

### 4.5 用户“rank0 与 rank2 routing 不同会不会破坏 timeline”的回答

1. **在真实动态 routing 语义下**，你的担心成立：不同 batch 可导致 token->expert 分布不同，耗时可能不同。  
2. **在当前 fork 的 Scaling 路径下**，同 `exp_rank` routing/dispatch 结果被预固定并复用，所以 rank0 与 rank2 更接近 duplicate。  
3. 若后续要让 sim-engine 对真实系统更稳健，必须引入 **DP 波动校验门槛**，而不是盲目把 duplicate 视为完全一致。

---

## 5) 面向 megatron-sim-engine 的落地指导

### 5.1 与当前 engine 行为对齐

1. Dense：engine 已按 PP 代表 rank 选取（`simu_engine.py:4292-4303`）。  
2. MoE：engine 当前保守为 all ranks（`simu_engine.py:4276-4284`）。

### 5.2 建议的两阶段策略（探索期）

1. **Stage A: 主测量集**  
   - Dense: `PP` 代表 rank  
   - MoE: `PP × EP` 代表 rank

2. **Stage B: 一致性校验集（spot-check）**  
   - 每个等价类额外抽样 1 个 duplicate rank（优先同 PP、同 EP、不同 DP）

3. **判定阈值（建议）**  
   - 对 `forward_step/backward_step/optimizer_step`：  
     - median 相对差 < 2%  
     - p95 相对差 < 5%  
     - max 相对差 < 8%

4. **回退策略**  
   - 任一指标超阈值：该类不再共享代表值，升级为更细粒度测量（最差回退全量）。

### 5.3 对 timeline 构建的影响

1. 若 duplicate 假设成立，timeline compute 节点可安全复用代表值，显著降低 profiling 成本。  
2. 若 duplicate 假设不成立却强行复用，会导致：
   - 某些 EP/DP 路径的 critical path 被低估或高估，
   - 集合通信与 compute overlap 预测偏移，
   - 端到端 E2E 误差扩大（尤其 MoE tail latency）。

---

## 6) Caveats / 限制说明

1. **Scaling 与真实 distributed 的语义边界**：当前实现对 routing 有预固定机制，结论主要面向本 fork 的 profiling 语义。  
2. **`ep_allreduce` 命名歧义**：`finalize_model_grads.py:213` 的 `ep_allreduce` 实际是 embedding grads 同步，不是 expert grads 同步。  
3. **Scaling 通信本质是 metadata + shape 仿真**，非真实链路时延。  
4. **TP 数据路径在 scaling 下存在 rank-specific 随机 batch 构造**（`utils.py:330-360`），会引入细微噪声。

---

## 7) 最终推荐（供后续执行）

1. Dense：先按 `PP` 代表 rank 测；仅在发现显著差异时再扩展。  
2. MoE：先按 `PP × EP` 测 + DP spot-check；通过阈值后再将 DP 合并为 duplicate。  
3. 对于 `PP=2,TP=1,EP=2,DP=4`：  
   - 主测 `{0,1,4,5}`  
   - 校验 `{2,3,6,7}`
4. 在 sim-engine 中保留 fail-fast：一旦通信组不完整或 duplicate 超阈值，自动回退更保守策略。

