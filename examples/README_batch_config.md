# Megatron-LM Workload Tracer - 批量配置支持

## 概述

本文档介绍了Megatron-LM workload tracer的批量配置支持功能。该功能允许在单次运行中测试多组并行度配置，同时保持原有的rank优化逻辑，显著提高大规模配置测试的效率。

## 功能特性

### 🚀 **批量配置支持**
- 支持在单次运行中测试多组并行度配置
- 每组配置包含：WORLD_SIZE, PP_SIZE, TP_SIZE
- 自动计算DP_SIZE = WORLD_SIZE / (PP_SIZE * TP_SIZE)

### 🎯 **保持Rank优化**
- 继承原有的rank优化逻辑
- 只运行PP stages 0到PP_SIZE-1，TP=0，DP=0
- 每组配置都享受相同的时间节省效果

### 📊 **独立日志管理**
- 为每组配置创建独立的日志目录
- 清晰的目录结构便于结果分析
- 详细的优化统计信息

### 🔄 **向后兼容**
- 如果未定义批量配置，自动使用单一配置模式
- 保持所有现有参数和功能不变

## 使用方法

### 1. 配置批量设置

在脚本中修改`BATCH_CONFIGS`数组：

```bash
BATCH_CONFIGS=(
    "8192 16 8"   # world_size=8192, pp=16, tp=8
    "8192 32 8"   # world_size=8192, pp=32, tp=8  
    "8192 32 4"   # world_size=8192, pp=32, tp=4
    "4096 16 4"   # world_size=4096, pp=16, tp=4
)
```

### 2. 运行批量配置

```bash
bash examples/update_pretrain_gpt.sh
```

### 3. 单一配置模式（向后兼容）

如果要使用单一配置，将`BATCH_CONFIGS`数组设为空：

```bash
BATCH_CONFIGS=()
```

脚本将自动使用默认配置：
- `DEFAULT_FAKE_WORLD_SIZE=8192`
- `DEFAULT_FAKE_PP=2`
- `DEFAULT_FAKE_TP=2`

## 配置格式

每个配置项的格式为：`"WORLD_SIZE PP_SIZE TP_SIZE"`

### 示例配置

```bash
BATCH_CONFIGS=(
    "8192 16 8"    # 8192 GPUs, 16 PP stages, 8 TP groups -> DP=64
    "8192 32 8"    # 8192 GPUs, 32 PP stages, 8 TP groups -> DP=32
    "8192 32 4"    # 8192 GPUs, 32 PP stages, 4 TP groups -> DP=64
    "4096 16 4"    # 4096 GPUs, 16 PP stages, 4 TP groups -> DP=64
    "2048 8 4"     # 2048 GPUs, 8 PP stages, 4 TP groups -> DP=64
)
```

### 配置验证

脚本会自动验证每个配置：
- 检查`WORLD_SIZE`是否能被`PP_SIZE * TP_SIZE`整除
- 确保`DP_SIZE > 0`
- 跳过无效配置并继续处理其他配置

## 输出结构

### 日志目录结构

```
logs/
├── SIM_GPT_485_Config1_WS8192_PP16_TP8_DP64_nMICROB1_MICROB_SIZE1_GLOBAL_BATCH64/
│   ├── fake_rank_0.log
│   ├── fake_rank_512.log
│   ├── fake_rank_1024.log
│   └── ...
├── SIM_GPT_485_Config2_WS8192_PP32_TP8_DP32_nMICROB1_MICROB_SIZE1_GLOBAL_BATCH32/
│   ├── fake_rank_0.log
│   ├── fake_rank_256.log
│   └── ...
└── ...
```

### 执行输出示例

```
============================================================
Megatron-LM Workload Tracer - Batch Configuration Support
============================================================
Running in BATCH CONFIGURATION mode
Total configurations to process: 3

Configurations to process:
  Config 1: WORLD_SIZE=8192, PP=16, TP=8, DP=64
  Config 2: WORLD_SIZE=8192, PP=32, TP=8, DP=32
  Config 3: WORLD_SIZE=8192, PP=32, TP=4, DP=64

Processing configuration 1/3...
============================================================
CONFIGURATION 1: WORLD_SIZE=8192, PP=16, TP=8, DP=64
============================================================
...
```

## 优化效果

### 时间节省统计

| 配置 | 原始Ranks | 优化后Ranks | 时间节省 |
|------|-----------|-------------|----------|
| WS=8192, PP=16, TP=8 | 8192 | 16 | ~99.8% |
| WS=8192, PP=32, TP=8 | 8192 | 32 | ~99.6% |
| WS=8192, PP=32, TP=4 | 8192 | 32 | ~99.6% |
| WS=4096, PP=16, TP=4 | 4096 | 16 | ~99.6% |

### 批量处理优势

- **一次设置，多次测试**：无需手动修改配置参数
- **并行度探索**：快速测试不同的PP/TP组合
- **结果对比**：统一的输出格式便于性能对比
- **时间效率**：相比逐个手动测试，节省大量时间

## 测试和验证

### 运行测试脚本

```bash
# 测试批量配置逻辑
bash examples/test_batch_config.sh

# 测试rank映射逻辑
bash examples/test_rank_mapping.sh
```

### 验证配置有效性

脚本提供了内置的配置验证功能：
- 自动检查并行度配置的数学正确性
- 显示详细的错误信息
- 跳过无效配置，继续处理其他配置

## 技术实现

### 核心函数

1. **`validate_config()`** - 验证配置有效性
2. **`execute_single_config()`** - 执行单个配置的模拟
3. **批量执行逻辑** - 遍历配置数组并统计结果

### 关键特性

- **动态参数构建**：每个配置动态生成GPT_ARGS和SIM_ARGS
- **独立日志管理**：每个配置有独立的日志目录
- **统计信息收集**：使用关联数组收集每个配置的统计信息
- **错误处理**：单个配置失败不影响其他配置的执行

## 注意事项

1. **内存要求**：确保系统有足够内存处理大规模配置
2. **存储空间**：每个配置会生成独立的日志文件
3. **执行时间**：虽然已优化，但大量配置仍需要相当的执行时间
4. **配置合理性**：确保配置符合实际硬件约束

## 总结

批量配置支持功能显著提升了Megatron-LM workload tracer的实用性，使得大规模并行度配置的测试和优化变得更加高效和系统化。结合原有的rank优化逻辑，该功能为深度学习模型的性能分析提供了强大的工具支持。
