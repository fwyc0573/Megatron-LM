#!/bin/bash

# Test script to validate the performance testing and metrics calculation scripts

echo "============================================================"
echo "Testing Performance and Metrics Calculation Scripts"
echo "============================================================"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIME_LOGS_DIR="$SCRIPT_DIR/time_logs"

echo "Script Directory: $SCRIPT_DIR"
echo "Time Logs Directory: $TIME_LOGS_DIR"
echo ""

# Test 1: Check if scripts exist and are executable
echo "Test 1: Checking script files..."
echo "------------------------------------------------------------"

if [ -x "$SCRIPT_DIR/run_performance_tests.sh" ]; then
    echo "✓ run_performance_tests.sh exists and is executable"
else
    echo "✗ run_performance_tests.sh is missing or not executable"
fi

if [ -x "$SCRIPT_DIR/calculate_metrics.sh" ]; then
    echo "✓ calculate_metrics.sh exists and is executable"
else
    echo "✗ calculate_metrics.sh is missing or not executable"
fi

echo ""

# Test 2: Check dependencies
echo "Test 2: Checking dependencies..."
echo "------------------------------------------------------------"

if command -v bc &> /dev/null; then
    echo "✓ bc calculator is available"
else
    echo "✗ bc calculator is not installed"
    echo "  Install with: sudo apt-get install bc"
fi

if command -v python &> /dev/null; then
    echo "✓ Python is available"
else
    echo "✗ Python is not available"
fi

echo ""

# Test 3: Create sample time log for testing metrics calculation
echo "Test 3: Creating sample time logs for testing..."
echo "------------------------------------------------------------"

mkdir -p "$TIME_LOGS_DIR"

# Create sample time records with new format
cat > "$TIME_LOGS_DIR/test_setting1_time.txt" << EOF
test_pp16_tp8_setting: load_time=12.34 execution_time=45.67 simulated_step_time=0.123
EOF

cat > "$TIME_LOGS_DIR/test_setting2_time.txt" << EOF
test_pp32_tp4_setting: load_time=8.91 execution_time=38.92 simulated_step_time=0.098
EOF

echo "✓ Created sample time log files"
echo ""

# Test 4: Test metrics calculation with sample data
echo "Test 4: Testing metrics calculation with sample data..."
echo "------------------------------------------------------------"

if [ -x "$SCRIPT_DIR/calculate_metrics.sh" ]; then
    echo "Running calculate_metrics.sh with sample data..."
    cd "$SCRIPT_DIR"
    ./calculate_metrics.sh
    echo ""
    echo "✓ Metrics calculation test completed"
else
    echo "✗ Cannot test metrics calculation - script not executable"
fi

echo ""

# Test 5: Cleanup test files
echo "Test 5: Cleaning up test files..."
echo "------------------------------------------------------------"

rm -f "$TIME_LOGS_DIR/test_setting1_time.txt"
rm -f "$TIME_LOGS_DIR/test_setting2_time.txt"

echo "✓ Test files cleaned up"
echo ""

echo "============================================================"
echo "Script testing completed"
echo "============================================================"
