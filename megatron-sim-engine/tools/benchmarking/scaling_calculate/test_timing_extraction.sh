#!/bin/bash

# Test script to validate timing extraction functionality

echo "============================================================"
echo "Testing Timing Extraction from simu_main.py Output"
echo "============================================================"

# Create a sample output file that mimics simu_main.py output
SAMPLE_OUTPUT="/tmp/sample_simu_output.txt"

cat > "$SAMPLE_OUTPUT" << 'EOF'
INIT | tmp stages dataset and timeline manager have been set.
sim load time:5.234567
Starting simulation...
SIMULATE, rank0 comp_time: 45.67 ms / 42.34
SIMULATE, rank0 comm_time: 78.90 ms
SIMULATE, rank0 sum_time: 124.57 ms
SIMULATE, rank1 comp_time: 46.12 ms / 43.01
SIMULATE, rank1 comm_time: 79.23 ms
SIMULATE, rank1 sum_time: 125.35 ms
world_size: 8192, sim load time:5.234567, sim execution time:12.345678
Timeline visualization saved to: ./log/visualization_outputs/timeline_SIMULATE_Dense_PP16_TP8_DP64_ranks0-15_20250821_041500.png
EOF

echo "Sample output file created: $SAMPLE_OUTPUT"
echo ""
echo "Testing timing extraction function..."

# Define the timing extraction function (copied from run_performance_tests.sh)
extract_timing_info() {
    local output_file=$1
    local setting_name=$2

    # Extract sim load time and sim execution time from the output
    local load_time=$(grep "sim load time:" "$output_file" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)
    local execution_time=$(grep "sim execution time:" "$output_file" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)

    # Also look for the combined output line
    local combined_line=$(grep "world_size:.*sim load time:.*sim execution time:" "$output_file" | tail -1)

    if [ -n "$combined_line" ]; then
        # Extract from combined line if available
        load_time=$(echo "$combined_line" | grep -oE 'sim load time:[0-9]+\.?[0-9]*' | grep -oE '[0-9]+\.?[0-9]*')
        execution_time=$(echo "$combined_line" | grep -oE 'sim execution time:[0-9]+\.?[0-9]*' | grep -oE '[0-9]+\.?[0-9]*')
    fi

    # Extract simulated LLM training step time from rank0 sum_time output
    # Look for pattern: "SIMULATE, rank0 sum_time: XXX.XX ms"
    local simulated_step_time=$(grep "rank0 sum_time:" "$output_file" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)

    # Convert from milliseconds to seconds for consistency
    if [ -n "$simulated_step_time" ]; then
        simulated_step_time=$(echo "scale=6; $simulated_step_time / 1000" | bc)
        # Ensure proper formatting (add leading zero if needed)
        if [[ "$simulated_step_time" =~ ^\. ]]; then
            simulated_step_time="0$simulated_step_time"
        fi
    fi

    if [ -z "$load_time" ] || [ -z "$execution_time" ] || [ -z "$simulated_step_time" ]; then
        echo "WARNING: Could not extract complete timing info for $setting_name" >&2
        echo "  load_time: $load_time, execution_time: $execution_time, simulated_step_time: $simulated_step_time" >&2
        echo "N/A N/A N/A"
    else
        echo "$load_time $execution_time $simulated_step_time"
    fi
}

# Test the extraction
echo "Calling extract_timing_info function..."
timing_result=$(extract_timing_info "$SAMPLE_OUTPUT" "test_setting")
echo "Raw result: '$timing_result'"

# Parse the result
read -r load_time execution_time simulated_step_time <<< "$timing_result"

echo ""
echo "Extracted timing values:"
echo "  Load time: '$load_time' seconds"
echo "  Execution time: '$execution_time' seconds"
echo "  Simulated step time: '$simulated_step_time' seconds"

echo ""
echo "Expected values:"
echo "  Load time: 5.234567 seconds"
echo "  Execution time: 12.345678 seconds"
echo "  Simulated step time: 0.12457 seconds (124.57 ms / 1000)"

echo ""
echo "Validation:"
if [ "$load_time" = "5.234567" ]; then
    echo "  ✓ Load time extraction correct"
else
    echo "  ✗ Load time extraction failed: expected 5.234567, got '$load_time'"
fi

if [ "$execution_time" = "12.345678" ]; then
    echo "  ✓ Execution time extraction correct"
else
    echo "  ✗ Execution time extraction failed: expected 12.345678, got '$execution_time'"
fi

# Check simulated step time (allowing for floating point precision)
expected_step_time="0.124570"
if [ "$simulated_step_time" = "$expected_step_time" ]; then
    echo "  ✓ Simulated step time extraction correct"
else
    echo "  ✗ Simulated step time extraction failed: expected $expected_step_time, got '$simulated_step_time'"
fi

# Clean up
rm -f "$SAMPLE_OUTPUT"

echo ""
echo "============================================================"
echo "Timing extraction test completed"
echo "============================================================"
