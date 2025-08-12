#!/bin/bash

# Quick Memory Visualization Script
# A simplified wrapper for common memory trace visualization tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default directories based on common patterns
DEFAULT_REAL_DIR="examples/memory_traces"
DEFAULT_SIM_DIR="examples/memory_traces_scaling"
DEFAULT_OUTPUT_DIR="memory_plots"

echo "=== Quick Memory Trace Visualization ==="
echo ""

# Function to check if directory exists and has trace files
check_dir() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        local count=$(find "$dir" -name "memory_trace_rank*.json" 2>/dev/null | wc -l)
        if [[ $count -gt 0 ]]; then
            echo "✓ Found $count trace files in: $dir"
            return 0
        else
            echo "✗ No trace files found in: $dir"
            return 1
        fi
    else
        echo "✗ Directory not found: $dir"
        return 1
    fi
}

# Check for default directories
echo "Checking for default trace directories..."
REAL_OK=false
SIM_OK=false

if check_dir "$DEFAULT_REAL_DIR"; then
    REAL_OK=true
fi

if check_dir "$DEFAULT_SIM_DIR"; then
    SIM_OK=true
fi

echo ""

# If both default directories exist, run visualization
if [[ "$REAL_OK" == true && "$SIM_OK" == true ]]; then
    echo "🚀 Running visualization with default directories..."
    echo "Real traces: $DEFAULT_REAL_DIR"
    echo "Sim traces: $DEFAULT_SIM_DIR"
    echo "Output: $DEFAULT_OUTPUT_DIR"
    echo ""
    
    "$SCRIPT_DIR/visualize_memory_traces.sh" \
        -r "$DEFAULT_REAL_DIR" \
        -s "$DEFAULT_SIM_DIR" \
        -o "$DEFAULT_OUTPUT_DIR"
    
    echo ""
    echo "✅ Visualization completed!"
    echo "Check the generated plots in: $DEFAULT_OUTPUT_DIR/"
    
elif [[ "$REAL_OK" == true || "$SIM_OK" == true ]]; then
    echo "⚠️  Only found one trace directory."
    echo "Need both real and simulation traces for comparison."
    echo ""
    echo "Available directories:"
    [[ "$REAL_OK" == true ]] && echo "  - $DEFAULT_REAL_DIR (real traces)"
    [[ "$SIM_OK" == true ]] && echo "  - $DEFAULT_SIM_DIR (simulation traces)"
    echo ""
    echo "To run visualization manually:"
    echo "$SCRIPT_DIR/visualize_memory_traces.sh -r <real_dir> -s <sim_dir>"
    
else
    echo "❌ No default trace directories found."
    echo ""
    echo "Expected directories:"
    echo "  - $DEFAULT_REAL_DIR (for real traces)"
    echo "  - $DEFAULT_SIM_DIR (for simulation traces)"
    echo ""
    echo "To run visualization with custom directories:"
    echo "$SCRIPT_DIR/visualize_memory_traces.sh -r <real_dir> -s <sim_dir>"
    echo ""
    echo "For help:"
    echo "$SCRIPT_DIR/visualize_memory_traces.sh --help"
fi
