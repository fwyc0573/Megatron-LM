# Mixtral 8×22B Configuration Guide for Megatron-LM

## 模型架构参数设置理由

### 1. 核心架构参数

| 参数 | Mixtral 8×7B | Mixtral 8×22B | 设置理由 |
|------|--------------|---------------|----------|
| `HIDDEN_SIZE` | 4096 | 6144 | 基于官方架构规格 |
| `NUM_HEAD` | 32 | 48 | 注意力头数，影响并行计算效率 |
| `NUM_LAYERS` | 32 | 56 | 更深的网络提供更强的表达能力 |
| `FFN_HIDDEN_SIZE` | 14336 | 16384 | 前馈网络维度，通常为 hidden_size 的 2.67 倍 |
| `NUM_KEY_VALUE_HEADS` | 8 | 8 | Group Query Attention，减少 KV cache 内存 |
| `VOCAB_SIZE` | 32000 | 32768 | 更大词汇表支持更多语言 |

### 2. MoE 特定配置

```bash
NUM_EXPERTS=8                    # 8 个专家，符合 Mixtral 架构
--moe-router-topk 2             # Top-2 routing，平衡性能和质量
--moe-aux-loss-coeff 0.01       # 辅助损失系数，促进负载均衡
--moe-z-loss-coeff 0.001        # Z-loss，防止路由器输出过大
--moe-token-dispatcher-type alltoall  # 高效的 token 分发机制
--moe-grouped-gemm              # 批量 GEMM 操作，提高效率
```

### 3. 并行策略优化

#### Mixtral 8×22B (141B 参数) 并行配置：

```bash
PP=4  # Pipeline Parallelism
TP=1  # Tensor Parallelism  
EP=2  # Expert Parallelism
DP=2  # Data Parallelism
```

**设置理由：**

1. **Pipeline Parallelism (PP=4)**：
   - 56 层需要更多 pipeline stages
   - 每个 stage 处理 14 层 (56/4)
   - 减少单个 GPU 的内存压力

2. **Tensor Parallelism (TP=1)**：
   - MoE 模型的最佳实践
   - 避免 expert 参数的复杂切分
   - 简化通信模式

3. **Expert Parallelism (EP=2)**：
   - 8 experts / 2 = 4 experts per rank
   - 平衡计算负载和通信开销
   - 支持 MoE Data Parallelism

4. **Data Parallelism (DP=2)**：
   - 8 GPUs / (4 PP × 1 TP) = 2
   - 提供基础的数据并行能力

### 4. 内存和性能优化

#### Batch Size 配置：
```bash
MICRO_BATCH_SIZE=1              # 大模型使用小 batch size
GLOBAL_BATCH_SIZE=2             # DP=2 时的全局 batch size
```

#### 序列长度配置：
```bash
MAX_SEQ_LEN=4096                # 训练时使用 4K context
MAX_POSITION_EMBEDDINGS=65536   # 支持 64K context length
```

### 5. Mixtral 特定架构特性

#### Group Query Attention (GQA)：
```bash
--group-query-attention
--num-query-groups 8            # 8 个 key-value heads
```
- 减少 KV cache 内存使用
- 保持注意力质量

#### RoPE 位置编码：
```bash
--no-position-embedding
--position-embedding-type rope
--rotary-interleaved
```
- 支持长序列外推
- 更好的位置感知能力

#### SwiGLU 激活函数：
```bash
--swiglu
--disable-bias-linear
```
- 更好的非线性表达能力
- 与 Mixtral 原始架构一致

#### RMSNorm 归一化：
```bash
--normalization RMSNorm
--norm-epsilon 1e-5
```
- 更稳定的训练
- 更好的数值精度

### 6. 训练超参数

#### 学习率配置：
```bash
--lr 3e-4                       # 适合大模型的学习率
--min-lr 3e-5                   # 最小学习率
--lr-warmup-fraction 0.01       # 1% warmup
--lr-decay-style cosine         # 余弦衰减
```

#### 优化器配置：
```bash
--adam-beta1 0.9
--adam-beta2 0.95               # 适合大模型的 beta2
--weight-decay 0.1              # 权重衰减
--clip-grad 1.0                 # 梯度裁剪
```

### 7. 通信优化

#### Expert Parallelism 通信：
- AlltoAll 用于 token 分发和收集
- 在 expert parallel group 内进行

#### MoE Data Parallelism 通信：
- AllReduce 用于 expert 参数梯度同步
- 在 data modulo expert parallel group 内进行

#### 通信组大小：
- Expert Parallel Group: 4 ranks (world_size / EP)
- MoE Data Parallel Group: 2 ranks (DP / EP)

### 8. 内存估算

#### Mixtral 8×22B 内存需求 (BF16)：
- 模型参数：141B × 2 bytes = 282 GB
- 梯度：141B × 2 bytes = 282 GB  
- 优化器状态：141B × 8 bytes = 1128 GB
- 总计：~1.7 TB

#### 8 GPU 分布 (PP=4, EP=2)：
- 每个 GPU：~210 GB
- 需要 A100 80GB 或 H100 80GB

### 9. 性能调优建议

1. **内存优化**：
   - 启用 gradient checkpointing
   - 使用 ZeRO optimizer
   - 考虑 CPU offloading

2. **通信优化**：
   - 启用 overlap_grad_reduce
   - 使用高速 InfiniBand 网络
   - 优化 NCCL 配置

3. **计算优化**：
   - 使用 Flash Attention
   - 启用 Grouped GEMM
   - 优化 expert load balancing

### 10. 验证和监控

#### 关键指标：
- Expert utilization balance
- Router entropy
- Auxiliary loss value
- Memory usage per GPU
- Communication overhead

#### 调试建议：
- 监控 expert 负载分布
- 检查梯度同步正确性
- 验证数值稳定性
- 测试收敛性能

这个配置为 Mixtral 8×22B 在 Megatron-LM 框架下的训练提供了优化的起点，可根据具体硬件和数据情况进行调整。
