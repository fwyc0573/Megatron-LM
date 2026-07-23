#!/bin/bash

# Test script for single MoE configuration scheduling plan generation
# Usage: ./test_single_config.sh [config_name]

# Default test configuration
DEFAULT_CONFIG="pp2_tp1_exp2_expn8_dp8_nl8_hs4096_sl1024"

# Get configuration name from argument or use default
CONFIG_NAME="${1:-$DEFAULT_CONFIG}"

echo "Testing scheduling plan generation for: $CONFIG_NAME"
echo "========================================================"

# Check if the configuration directory exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TARGET_BASE_DIR="${REPO_ROOT}/simulation_inputs/megatron_operation_log/new_moe"
CONFIG_DIR="$TARGET_BASE_DIR/$CONFIG_NAME"

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Error: Configuration directory does not exist: $CONFIG_DIR"
    echo ""
    echo "Available configurations:"
    ls -1 "$TARGET_BASE_DIR" | grep -E "^pp[0-9]+" | head -10
    exit 1
fi

echo "Configuration directory: $CONFIG_DIR"
echo "Schedule directory: $CONFIG_DIR/schedule"

# Check current state of schedule directory
echo ""
echo "Current schedule directory contents:"
ls -la "$CONFIG_DIR/schedule/"

echo ""
echo "Running scheduling plan generation..."
echo "========================================"

# Run the scheduling plan generation for single config
cd "$SCRIPT_DIR"
./run_moe.sh "$CONFIG_NAME"

# Check results
echo ""
echo "Results:"
echo "========================================"
echo "Generated files in schedule directory:"
ls -la "$CONFIG_DIR/schedule/"

echo ""
echo "File count: $(find "$CONFIG_DIR/schedule/" -name "*.txt" | wc -l)"

# Show sample content if files were generated
SAMPLE_FILE=$(find "$CONFIG_DIR/schedule/" -name "*.txt" | head -1)
if [[ -n "$SAMPLE_FILE" ]]; then
    echo ""
    echo "Sample content from $(basename "$SAMPLE_FILE"):"
    echo "----------------------------------------"
    head -10 "$SAMPLE_FILE"
    echo "----------------------------------------"
fi

echo ""
echo "Test completed for: $CONFIG_NAME"
