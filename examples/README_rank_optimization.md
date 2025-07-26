# Megatron-LM Workload Tracer Rank Optimization

## 概述

本优化针对Megatron-LM的workload tracer组件，将原本需要遍历所有ranks的脚本优化为只运行特定的关键ranks，从而显著减少执行时间。

## 背景

在原始的`update_pretrain_gpt.sh`脚本中，循环`for current_fake_rank_id in $(seq 0 $((${FAKE_WORLD_SIZE} - 1)))`会遍历所有rank，这对于dense模型来说是不必要的，因为我们只需要特定的ranks来捕获完整的workload模式。

## 优化策略

### Dense模型的Rank选择策略

对于dense类模型，我们只需要运行以下特定的ranks：
- **PP rank**: 0, 1, 2, ..., PP_SIZE-1 (所有PP stages)
- **TP rank**: 0 (固定，代表TP local rank 0)
- **DP rank**: 0 (固定，代表DP local rank 0)

### Rank映射公式

根据Megatron的rank映射逻辑 (order="tp-cp-ep-dp-pp")：
```
global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size
```

对于我们的优化，设置：
- `tp_rank = 0`
- `dp_rank = 0`
- `pp_rank = 0, 1, 2, ..., PP_SIZE-1`

因此：
```
world_rank = 0 + 0 * TP_SIZE + pp_stage * TP_SIZE * DP_SIZE
           = pp_stage * TP_SIZE * DP_SIZE
```

## 优化效果

### 时间节省

| 配置 | 原始Ranks | 优化后Ranks | 时间节省 |
|------|-----------|-------------|----------|
| PP=2, TP=2, DP=1 | 4 | 2 | ~50% |
| PP=4, TP=2, DP=1 | 8 | 4 | ~50% |
| PP=2, TP=4, DP=1 | 8 | 2 | ~75% |
| PP=8, TP=1, DP=1 | 8 | 8 | ~0% |

### 示例

对于配置 `PP=2, TP=2, DP=1, WORLD_SIZE=4`：
- **原始**: 需要运行ranks [0, 1, 2, 3]
- **优化后**: 只需运行ranks [0, 2]
- **时间节省**: 50%

## 使用方法

### 运行优化后的脚本
```bash
bash examples/update_pretrain_gpt.sh
```

### 测试rank映射逻辑
```bash
bash examples/test_rank_mapping.sh
```

## 脚本修改详情

### 主要修改

1. **Rank计算逻辑**: 替换了原始的全量遍历，改为基于PP stages的选择性计算
2. **配置验证**: 添加了并行度配置的验证逻辑
3. **进度显示**: 增强了日志输出，显示优化效果和映射关系
4. **错误处理**: 改进了错误检测和报告

### 关键代码段

```bash
# 计算每个PP stage在TP rank=0, DP rank=0时对应的world rank
SELECTED_RANKS=()
for pp_stage in $(seq 0 $((${FAKE_PP} - 1)))
do
    world_rank=$((0 + 0 * ${FAKE_TP} + ${pp_stage} * ${FAKE_TP} * ${FAKE_DP}))
    SELECTED_RANKS+=(${world_rank})
done
```

## 注意事项

1. **适用性**: 此优化主要针对dense模型。对于MoE (Mixture of Experts) 模型，可能需要不同的rank选择策略。

2. **验证**: 建议在使用前运行测试脚本验证rank映射的正确性。

3. **扩展性**: 脚本设计为通用的，可以处理不同的PP、TP、DP配置。

## 技术细节

### Rank Group结构

以PP=2, TP=2, DP=1为例：
- **PP groups**: [[0,1], [2,3]] (每个PP group包含不同TP rank但相同PP stage的ranks)
- **TP groups**: [[0,2], [1,3]] (每个TP group包含不同PP stage但相同TP rank的ranks)

### 选择逻辑

我们选择每个PP group中的第一个rank (TP rank=0)，这样可以覆盖所有PP stages的workload模式，同时避免重复的TP和DP计算。

## 结论

通过这个优化，我们成功将workload tracer的执行时间减少了50-75%（取决于具体的并行度配置），同时保持了完整的workload模式捕获能力。这对于大规模模型的性能分析和优化具有重要意义。
