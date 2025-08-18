#!/bin/bash

# GPU Memory Usage Heatmap Visualization - Example Script
# Author: Generated for Yicheng
# Date: 2025-07-26

echo "=== GPU Memory Usage Heatmap Visualization ==="
echo "Generating heatmap for 485B model with 8192 GPUs..."
echo

# Set paths relative to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMORY_DIR="../../examples/memory_traces_scaling/485B_8192gpus_tplimit8"
LOG_DIR="../../log/CONFIG_SWEEP_WS8192_485_20250726_012803"
OUTPUT_DIR="../../visualization_outputs"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate academic-style heatmap with auto 1:3 ratio
echo "1. Generating academic-style heatmap (auto 1:3 ratio)..."
python "$SCRIPT_DIR/visualize_memory_heatmap.py" \
  --memory-dir "$MEMORY_DIR" \
  --log-dir "$LOG_DIR" \
  --output "$OUTPUT_DIR/memory_heatmap_485B_academic" \
    --figsize 8 3.1\
  --dpi 300

echo "   ✓ Academic heatmap saved to: $OUTPUT_DIR/memory_heatmap_485B_academic.png/.pdf"
echo

# Generate high-resolution version for publications
echo "2. Generating high-resolution academic heatmap..."
python "$SCRIPT_DIR/visualize_memory_heatmap.py" \
  --memory-dir "$MEMORY_DIR" \
  --log-dir "$LOG_DIR" \
  --output "$OUTPUT_DIR/memory_heatmap_485B_publication" \
  --figsize 9 3.5\
  --dpi 600

echo "   ✓ Publication heatmap saved to: $OUTPUT_DIR/memory_heatmap_485B_publication.png/.pdf"
echo

# Generate custom size version
echo "3. Generating custom-sized heatmap..."
python "$SCRIPT_DIR/visualize_memory_heatmap.py" \
  --memory-dir "$MEMORY_DIR" \
  --log-dir "$LOG_DIR" \
  --output "$OUTPUT_DIR/memory_heatmap_485B_custom" \
  --figsize 8 3 \
  --dpi 300

echo "   ✓ Custom heatmap saved to: $OUTPUT_DIR/memory_heatmap_485B_custom.png/.pdf"
echo

echo "=== All academic-style heatmaps generated successfully! ==="
echo
echo "Output files (both PNG and PDF formats):"
echo "  - Academic style:     $OUTPUT_DIR/memory_heatmap_485B_academic.{png,pdf}"
echo "  - Publication ready:  $OUTPUT_DIR/memory_heatmap_485B_publication.{png,pdf}"
echo "  - Custom size:        $OUTPUT_DIR/memory_heatmap_485B_custom.{png,pdf}"
echo
echo "Features:"
echo "  ✓ Academic publication style (USENIX/OSDI/NSDI compatible)"
echo "  ✓ 1:3 aspect ratio (height:width) for optimal layout"
echo "  ✓ Memory values with GB units"
echo "  ✓ Both PNG (presentations) and PDF (publications) formats"
echo "  ✓ Professional typography and color scheme"
echo
echo "You can use these images in academic papers, presentations, and reports"
echo "to analyze GPU memory usage patterns across TP/PP configurations."
