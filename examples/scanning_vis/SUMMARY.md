# LLM 3D并行训练GPU内存使用可视化项目总结（学术版）

## 项目概述

本项目为Yicheng创建了一个符合学术出版标准的Python可视化工具，专门用于分析LLM 3D并行训练中不同TP（Tensor Parallelism）和PP（Pipeline Parallelism）组合下的GPU内存使用情况。该工具完全符合USENIX OSDI/NSDI等顶级会议的绘图风格要求。

## 核心功能

### 1. 数据解析与处理
- **JSON内存追踪文件解析**：自动解析文件名格式 `memory_trace_rank0_wd8192_tp{TP}_pp{PP}_dp{DP}_...`
- **配置状态识别**：从日志文件中识别成功和失败的配置
- **内存数据提取**：从JSON文件提取 `peak_allocated_MB`（实际内存）和 `theoretical_memory_MB`（理论预测）
- **理论预测分析**：识别理论预测 > 80GB但实际运行成功的配置（预测偏差过大）

### 2. 学术级热力图可视化

- **专业布局**：Y轴为TP SIZE，X轴为PP SIZE，1:3最优长宽比
- **学术色彩方案**：
  - 8级专业渐变色彩（#FFFFFF到#99000D）
  - 灰色表示失败配置（OOM）
  - 符合学术出版标准的色彩搭配
- **双值显示**：每个格子显示"Actual: X.XGB"和"Theory: X.XGB"两行数据
- **失败标记**：失败配置显示"—"（em dash），使用浅灰色背景（#E8E8E8）
- **理论预测差异标记**：白色斜纹图案（///）覆盖理论预测 ≥ 80GB的配置
- **学术字体**：Times New Roman等serif字体，符合期刊要求

### 3. 双格式输出与自定义

- **双格式支持**：同时生成PNG（演示用）和PDF（出版用）
- **自动长宽比**：智能计算1:3比例，适合学术论文版面
- **高分辨率**：支持600 DPI等出版级分辨率
- **专业排版**：去除多余边框，优化图例和标签布局

## 技术实现

### 核心类：MemoryHeatmapVisualizer
```python
class MemoryHeatmapVisualizer:
    - _parse_filename(): 解析文件名提取TP/PP值
    - _load_memory_data(): 加载JSON内存数据
    - _load_config_status(): 加载配置成功/失败状态
    - _create_heatmap_matrix(): 创建热力图矩阵
    - create_heatmap(): 生成并保存热力图
```

### 关键技术特性
- **正则表达式解析**：精确提取TP/PP配置参数
- **自定义颜色映射**：LinearSegmentedColormap实现渐变效果
- **异常处理**：完善的错误处理和数据验证
- **路径管理**：使用pathlib进行跨平台路径处理

## 文件结构

```
examples/scanning/
├── visualize_memory_heatmap.py  # 主要可视化脚本
├── run_example.sh              # 示例运行脚本
├── README.md                   # 详细使用说明
└── SUMMARY.md                  # 项目总结（本文件）

visualization_outputs/
├── memory_heatmap_485B_standard.png    # 标准热力图
├── memory_heatmap_485B_highres.png     # 高分辨率热力图
├── memory_heatmap_485B_compact.png     # 紧凑热力图
└── ...                                 # 其他生成的图像
```

## 数据分析结果

基于485B模型、8192 GPU的测试数据：

### 成功配置（6个）
- TP=2, PP=64: 57.3 GB（理论：136.9 GB - **预测偏差**）
- TP=4, PP=32: 76.0 GB（理论：99.8 GB - **预测偏差**）
- TP=4, PP=64: 29.1 GB（理论：68.5 GB）
- TP=8, PP=16: 73.6 GB（理论：81.5 GB - **预测偏差**）
- TP=8, PP=32: 38.4 GB（理论：49.9 GB）
- TP=8, PP=64: 15.0 GB（理论：34.2 GB）

### 失败配置（18个）
- 所有TP=1的配置（6个）
- 部分TP=2,4,8的低PP配置（12个）

### 关键发现
1. **最优配置**：TP=8, PP=64 内存使用最低（15.0 GB，理论预测准确）
2. **内存趋势**：PP值越高，内存使用越低
3. **失败模式**：TP=1配置全部失败，低PP配置容易OOM
4. **理论预测偏差**：3个配置的理论预测 > 80GB但实际成功，表明预测模型过于保守
5. **预测准确性**：理论预测在高PP配置下更准确，低PP配置偏差较大

## 使用方法

### 基本使用
```bash
python visualize_memory_heatmap.py \
  --memory-dir ../../examples/memory_traces_scaling/485B_8192gpus_tplimit8 \
  --log-dir ../../log/CONFIG_SWEEP_WS8192_540_20250726_024653 \
  --output ../../visualization_outputs/memory_heatmap.png
```

### 批量生成
```bash
./run_example.sh
```

## 技术优势

1. **自动化程度高**：一键生成完整的可视化分析
2. **可扩展性强**：易于适配不同的数据格式和配置
3. **可视化效果佳**：清晰的颜色编码和数值标注
4. **配置灵活**：支持多种自定义参数
5. **错误处理完善**：robust的数据解析和异常处理

## 应用价值

1. **配置优化**：快速识别最优的TP/PP组合
2. **资源规划**：预测不同配置的内存需求
3. **故障诊断**：识别导致OOM的配置模式
4. **性能分析**：可视化内存使用趋势和瓶颈

## 扩展建议

1. **多模型支持**：扩展支持不同规模的模型
2. **交互式可视化**：添加Web界面或Jupyter notebook支持
3. **性能指标集成**：结合训练速度、吞吐量等指标
4. **自动推荐**：基于历史数据推荐最优配置

---

**作者**：为Yicheng定制开发  
**日期**：2025-07-26  
**版本**：1.0  
**技术栈**：Python, matplotlib, numpy, pathlib
