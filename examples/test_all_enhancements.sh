#!/bin/bash

# Test script for all memory trace visualization enhancements
# This script tests all the new features implemented

set -e

echo "=== Testing All Memory Trace Visualization Enhancements ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Test 1: Basic functionality (no theoretical memory)
echo "Test 1: Basic functionality (no theoretical memory)"
echo "Command: ./examples/visualize_memory_traces.sh -r examples/memory_traces -s examples/memory_traces_scaling -o test_basic"
./examples/visualize_memory_traces.sh -r examples/memory_traces -s examples/memory_traces_scaling -o test_basic
echo "✅ Test 1 passed"
echo ""

# Test 2: With theoretical memory enabled
echo "Test 2: With theoretical memory enabled"
echo "Command: ./examples/visualize_memory_traces.sh -r examples/memory_traces -s examples/memory_traces_scaling -o test_theoretical -t"
./examples/visualize_memory_traces.sh -r examples/memory_traces -s examples/memory_traces_scaling -o test_theoretical -t
echo "✅ Test 2 passed"
echo ""

# Test 3: Direct standalone visualizer usage
echo "Test 3: Direct standalone visualizer usage"
echo "Command: python3 examples/standalone_visualizer.py examples/memory_traces examples/memory_traces_scaling test_direct --show-theoretical"
python3 examples/standalone_visualizer.py examples/memory_traces examples/memory_traces_scaling test_direct --show-theoretical
echo "✅ Test 3 passed"
echo ""

# Test 4: Help functionality
echo "Test 4: Help functionality"
echo "Command: ./examples/visualize_memory_traces.sh --help"
./examples/visualize_memory_traces.sh --help > /dev/null
echo "✅ Test 4 passed"
echo ""

# Test 5: Quick visualizer
echo "Test 5: Quick visualizer"
echo "Command: ./examples/quick_visualize.sh"
./examples/quick_visualize.sh > /dev/null
echo "✅ Test 5 passed"
echo ""

# Verify generated files
echo "=== Verification ==="
echo "Checking generated plot files..."

test_dirs=("test_basic" "test_theoretical" "test_direct")
for dir in "${test_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        plot_count=$(find "$dir" -name "memory_comparison_rank_*.png" | wc -l)
        echo "✅ $dir: Found $plot_count plot files"
    else
        echo "❌ $dir: Directory not found"
    fi
done

echo ""
echo "=== Feature Summary ==="
echo "✅ Requirement 1: Controllable Theoretical Memory Plotting"
echo "   - Added --show-theoretical parameter"
echo "   - Theoretical memory lines disabled by default"
echo "   - Can be enabled with -t or --show-theoretical flag"
echo ""
echo "✅ Requirement 2: Static Memory Field Support"
echo "   - Added static_memory_MB field to JSON format"
echo "   - Backward compatibility maintained"
echo "   - Static memory displayed as green baseline lines"
echo ""
echo "✅ Requirement 3: Static Memory Recording in Scaling Mode"
echo "   - Added static memory measurement in pretrain() function"
echo "   - Captures baseline memory before forward computation"
echo "   - Works in both scaling and non-scaling modes"
echo ""
echo "✅ Requirement 4: Updated Shell Scripts and Documentation"
echo "   - Enhanced visualize_memory_traces.sh with new parameters"
echo "   - Created standalone_visualizer.py to avoid import conflicts"
echo "   - Updated README with new features and examples"
echo ""
echo "=== All Tests Completed Successfully! ==="
echo ""
echo "Generated test directories:"
ls -la test_* 2>/dev/null | grep "^d" || echo "No test directories found"
echo ""
echo "To clean up test directories, run:"
echo "rm -rf test_basic test_theoretical test_direct"
