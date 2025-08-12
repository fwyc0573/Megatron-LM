#!/bin/bash

# Example usage script for memory trace visualization
# This script demonstrates how to use the visualize_memory_traces.sh script

set -e

echo "=== Memory Trace Visualization Examples ==="
echo ""

# Example 1: Basic usage with default output directory
echo "Example 1: Basic usage"
echo "Command: ./examples/visualize_memory_traces.sh -r memory_traces -s memory_traces_scaling"
echo ""

# Example 2: With custom output directory
echo "Example 2: With custom output directory"
echo "Command: ./examples/visualize_memory_traces.sh -r memory_traces -s memory_traces_scaling -o my_memory_plots"
echo ""

# Example 3: Using absolute paths
echo "Example 3: Using absolute paths"
echo "Command: ./examples/visualize_memory_traces.sh -r /path/to/real/traces -s /path/to/sim/traces -o /path/to/output"
echo ""

# Example 4: Show help
echo "Example 4: Show help information"
echo "Command: ./examples/visualize_memory_traces.sh --help"
echo ""

# Check if we have sample directories to demonstrate with
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Checking for sample trace directories ==="

# Look for potential trace directories
POTENTIAL_DIRS=(
    "memory_traces"
    "memory_traces_scaling" 
    "examples/memory_traces"
    "examples/memory_traces_scaling"
)

FOUND_DIRS=()
for dir in "${POTENTIAL_DIRS[@]}"; do
    if [[ -d "$PROJECT_ROOT/$dir" ]]; then
        trace_count=$(find "$PROJECT_ROOT/$dir" -name "memory_trace_rank*.json" 2>/dev/null | wc -l)
        if [[ $trace_count -gt 0 ]]; then
            FOUND_DIRS+=("$dir")
            echo "Found trace directory: $dir ($trace_count files)"
        fi
    fi
done

if [[ ${#FOUND_DIRS[@]} -ge 2 ]]; then
    echo ""
    echo "=== Running demonstration with found directories ==="
    real_dir="${FOUND_DIRS[0]}"
    sim_dir="${FOUND_DIRS[1]}"
    
    echo "Using:"
    echo "  Real traces: $real_dir"
    echo "  Sim traces: $sim_dir"
    echo ""
    
    # Run the actual visualization
    "$SCRIPT_DIR/visualize_memory_traces.sh" -r "$real_dir" -s "$sim_dir" -o "demo_memory_plots"
    
elif [[ ${#FOUND_DIRS[@]} -eq 1 ]]; then
    echo ""
    echo "Only found one trace directory: ${FOUND_DIRS[0]}"
    echo "Need at least two directories (real and simulation) for comparison."
    echo ""
    echo "You can still run the visualization script manually:"
    echo "./examples/visualize_memory_traces.sh -r ${FOUND_DIRS[0]} -s another_trace_dir"
    
else
    echo ""
    echo "No trace directories found with memory_trace_rank*.json files."
    echo ""
    echo "To use the visualization script, you need:"
    echo "1. A directory with real run traces (memory_trace_rank*.json files)"
    echo "2. A directory with simulation traces (memory_trace_rank*.json files)"
    echo ""
    echo "Then run:"
    echo "./examples/visualize_memory_traces.sh -r real_traces_dir -s sim_traces_dir"
fi

echo ""
echo "=== For more information ==="
echo "Run: ./examples/visualize_memory_traces.sh --help"
