# GPU Memory Usage Heatmap Visualization

This directory contains tools for visualizing GPU memory usage across different TP (Tensor Parallelism) and PP (Pipeline Parallelism) configurations in LLM 3D parallel training.

## Overview

The `visualize_memory_heatmap.py` script creates a heatmap visualization showing GPU memory usage patterns across different parallelization strategies. It helps identify optimal configurations and visualize memory bottlenecks in large-scale LLM training.

## Features

- **Academic Publication Style**: USENIX/OSDI/NSDI compatible typography and layout
- **Dual Format Output**: Generates both PNG (presentations) and PDF (publications) formats
- **Optimal Aspect Ratio**: Auto-calculated 1:3 (height:width) ratio for professional appearance
- **Memory Usage Visualization**: Shows peak GPU memory usage with GB units for each TP/PP combination
- **Configuration Status**: Distinguishes between successful and failed (OOM) configurations
- **Theory-Reality Analysis**: Highlights configurations where theoretical prediction > 80GB but actual execution succeeded
- **Professional Color Scheme**: Academic-grade color mapping with subtle gradients
- **Customizable Output**: Configurable figure size, DPI, and color scaling

## Requirements

- Python 3.7+
- matplotlib
- numpy
- pathlib (built-in)

## Usage

### Basic Usage

```bash
python visualize_memory_heatmap.py
```

### Academic Style with Auto 1:3 Ratio

```bash
python visualize_memory_heatmap.py \
  --memory-dir ../../examples/memory_traces_scaling/485B_8192gpus_tplimit8 \
  --log-dir ../../log/CONFIG_SWEEP_WS8192_540_20250726_024653 \
  --output ../../visualization_outputs/memory_heatmap_485B_academic
```

### Publication Ready (High Resolution)

```bash
python visualize_memory_heatmap.py \
  --memory-dir ../../examples/memory_traces_scaling/485B_8192gpus_tplimit8 \
  --log-dir ../../log/CONFIG_SWEEP_WS8192_540_20250726_024653 \
  --output ../../visualization_outputs/memory_heatmap_485B_publication \
  --dpi 600
```

### Custom Size and Options

```bash
python visualize_memory_heatmap.py \
  --memory-dir ../../examples/memory_traces_scaling/485B_8192gpus_tplimit8 \
  --log-dir ../../log/CONFIG_SWEEP_WS8192_540_20250726_024653 \
  --output ../../visualization_outputs/memory_heatmap_485B_custom \
  --figsize 12.0 4.0 \
  --max-memory 80.0 \
  --dpi 300
```

## Command Line Arguments

- `--memory-dir`: Directory containing memory trace JSON files
- `--log-dir`: Directory containing configuration log files
- `--output`: Base output path (without extension, generates .png and .pdf)
- `--figsize`: Figure size (width height) in inches (default: auto 1:3 ratio)
- `--max-memory`: Maximum GPU memory in GB for color scaling (default: 80.0)
- `--dpi`: DPI for PNG output (default: 300)

## Input Data Format

### Memory Trace Files

JSON files with naming pattern:
```
memory_trace_rank0_wd8192_tp{TP}_pp{PP}_dp{DP}_exp1_expNumNone_l96_mbs1_nmbs{NMBS}_gbs{GBS}_ts{TS}.json
```

Example:
```
memory_trace_rank0_wd8192_tp2_pp64_dp64_exp1_expNumNone_l96_mbs1_nmbs384_gbs24576_ts60530.json
```

JSON structure:
```json
{
    "0": {
        "samples": [...],
        "peak_allocated_MB": 58716.72,
        "theoretical_memory_MB": 140204.06
    }
}
```

**Key Fields:**

- `peak_allocated_MB`: Actual peak memory usage during execution
- `theoretical_memory_MB`: Theoretical memory prediction before execution

### Configuration Log Files

- `successful_configs.log`: Lists successful configurations
- `failed_configs.log`: Lists failed configurations (typically OOM)

Format:
```
Config X: TP=Y, PP=Z, DP=W, NUM_MICBATCH=N, GLOBAL_BATCH_SIZE=G - SUCCESS/FAILED
```

## Output

The script generates academic-style heatmaps in both PNG and PDF formats with:

- **X-axis**: Pipeline Parallelism (PP) size
- **Y-axis**: Tensor Parallelism (TP) size
- **Cell Colors**: Professional color gradient (white = low, deep red = high memory usage)
- **Cell Values**: Dual display showing both "Actual: X.XGB" and "Theory: X.XGB" for successful configurations
- **Light Gray Cells**: Failed configurations (OOM) marked with "—" (em dash) in lighter gray
- **White Diagonal Hatching**: Configs where theory ≥ 80GB marked with white /// pattern overlay
- **Colorbar**: Memory usage scale in GB with clean typography
- **Legend**: Professional legend with clear color coding including theory prediction indicator
- **Typography**: Academic publication style (serif fonts, proper spacing)
- **Aspect Ratio**: Optimized 1:3 (height:width) ratio for publication layout

## Interpretation

- **White/Light colors**: Low memory usage, good efficiency
- **Red/Dark colors**: High memory usage, approaching limits
- **Gray cells with "--"**: Failed configurations due to OOM
- **Missing cells**: No data available for that configuration

## Example Output

The heatmap helps identify:

1. Memory-efficient configurations (light colors)
2. Memory bottlenecks (dark colors)
3. Failed configurations (light gray)
4. Theory-reality mismatches (white diagonal hatching)
5. Actual vs theoretical memory comparison (dual values in each cell)
6. Optimal TP/PP trade-offs

## Technical Details

- Memory values are extracted from `peak_allocated_MB` field in JSON files
- Color intensity is calculated as `peak_memory_GB / max_memory_GB`
- Maximum color intensity corresponds to RGB(214, 39, 40)
- Failed configurations are identified from log files and marked separately

## Author

Generated for Yicheng - LLM Training Memory Analysis Tool
Date: 2025-07-26
