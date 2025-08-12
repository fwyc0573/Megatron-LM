#!/bin/bash

# Memory Trace Visualization Script
# This script calls megatron/profiler/trace_memory.py to visualize memory usage comparison
# between real runs and simulation runs.

set -e  # Exit on any error

# Default values
REAL_TRACES_DIR=""
SIM_TRACES_DIR=""
OUTPUT_DIR="memory_plots"
SHOW_THEORETICAL=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRACE_MEMORY_SCRIPT="$SCRIPT_DIR/standalone_visualizer.py"

# Function to display usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

This script visualizes memory trace comparisons between real runs and simulation runs.

OPTIONS:
    -r, --real-dir DIR      Directory containing real run memory traces (required)
    -s, --sim-dir DIR       Directory containing simulation memory traces (required)
    -o, --output-dir DIR    Output directory for generated plots (default: memory_plots)
    -t, --show-theoretical  Display theoretical memory lines in plots (default: disabled)
    -h, --help              Show this help message

EXAMPLES:
    # Basic usage
    $0 -r memory_traces -s memory_traces_scaling

    # With custom output directory
    $0 -r memory_traces -s memory_traces_scaling -o my_plots

    # With theoretical memory lines enabled
    $0 -r memory_traces -s memory_traces_scaling -t

    # Using absolute paths with theoretical memory
    $0 -r /path/to/real/traces -s /path/to/sim/traces -o /path/to/output -t

DESCRIPTION:
    This script calls the trace_memory.py visualization function to generate
    comparison plots between real training runs and simulation runs. It expects
    JSON trace files with names like 'memory_trace_rank*.json' in both directories.

    The generated plots will show:
    - Allocated memory over time for the last iteration
    - Comparison between real and simulation runs
    - Static memory baseline (always shown when available)
    - Theoretical memory usage (only when --show-theoretical is enabled)

REQUIREMENTS:
    - Python with matplotlib installed
    - Memory trace JSON files in the specified directories
    - Files should follow the naming pattern: memory_trace_rank*.json

EOF
}

# Function to check if directory exists and contains trace files
check_trace_dir() {
    local dir="$1"
    local type="$2"
    
    if [[ ! -d "$dir" ]]; then
        echo "Error: $type directory '$dir' does not exist."
        return 1
    fi
    
    local trace_files=$(find "$dir" -name "memory_trace_rank*.json" 2>/dev/null | wc -l)
    if [[ $trace_files -eq 0 ]]; then
        echo "Warning: No memory trace files found in $type directory '$dir'."
        echo "Expected files with pattern: memory_trace_rank*.json"
        return 1
    fi
    
    echo "Found $trace_files trace file(s) in $type directory: $dir"
    return 0
}

# Function to check Python dependencies
check_dependencies() {
    echo "Checking Python dependencies..."
    
    if ! python3 -c "import matplotlib" 2>/dev/null; then
        echo "Error: matplotlib is not installed."
        echo "Please install it with: pip install matplotlib"
        return 1
    fi
    
    if ! python3 -c "import json" 2>/dev/null; then
        echo "Error: json module is not available."
        return 1
    fi
    
    echo "Dependencies check passed."
    return 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--real-dir)
            REAL_TRACES_DIR="$2"
            shift 2
            ;;
        -s|--sim-dir)
            SIM_TRACES_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--show-theoretical)
            SHOW_THEORETICAL=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$REAL_TRACES_DIR" ]]; then
    echo "Error: Real traces directory is required."
    echo "Use -r or --real-dir to specify the directory."
    echo ""
    show_usage
    exit 1
fi

if [[ -z "$SIM_TRACES_DIR" ]]; then
    echo "Error: Simulation traces directory is required."
    echo "Use -s or --sim-dir to specify the directory."
    echo ""
    show_usage
    exit 1
fi

# Check if trace_memory.py exists
if [[ ! -f "$TRACE_MEMORY_SCRIPT" ]]; then
    echo "Error: trace_memory.py not found at: $TRACE_MEMORY_SCRIPT"
    exit 1
fi

echo "=== Memory Trace Visualization ==="
echo "Real traces directory: $REAL_TRACES_DIR"
echo "Simulation traces directory: $SIM_TRACES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Show theoretical memory: $SHOW_THEORETICAL"
echo "Trace memory script: $TRACE_MEMORY_SCRIPT"
echo ""

# Check dependencies
if ! check_dependencies; then
    exit 1
fi

# Check trace directories
echo "Checking trace directories..."
if ! check_trace_dir "$REAL_TRACES_DIR" "real traces"; then
    echo "Continuing anyway - some plots may be incomplete."
fi

if ! check_trace_dir "$SIM_TRACES_DIR" "simulation traces"; then
    echo "Continuing anyway - some plots may be incomplete."
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting visualization..."

# Build command with optional theoretical memory flag
if [[ "$SHOW_THEORETICAL" == true ]]; then
    echo "Command: python3 $TRACE_MEMORY_SCRIPT $REAL_TRACES_DIR $SIM_TRACES_DIR $OUTPUT_DIR --show-theoretical"
    VISUALIZATION_CMD=(python3 "$TRACE_MEMORY_SCRIPT" "$REAL_TRACES_DIR" "$SIM_TRACES_DIR" "$OUTPUT_DIR" --show-theoretical)
else
    echo "Command: python3 $TRACE_MEMORY_SCRIPT $REAL_TRACES_DIR $SIM_TRACES_DIR $OUTPUT_DIR"
    VISUALIZATION_CMD=(python3 "$TRACE_MEMORY_SCRIPT" "$REAL_TRACES_DIR" "$SIM_TRACES_DIR" "$OUTPUT_DIR")
fi
echo ""

# Run the visualization
if "${VISUALIZATION_CMD[@]}"; then
    echo ""
    echo "=== Visualization completed successfully! ==="
    echo "Output plots saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*.png" -type f | sort
else
    echo ""
    echo "=== Visualization failed! ==="
    echo "Please check the error messages above."
    exit 1
fi
