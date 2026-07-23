#!/bin/bash

# Cost Calculation Script for Training Metrics and Cost Analysis
# This script calculates training metrics and costs based on performance test results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Training Metrics and Cost Analysis Calculator"
echo "============================================================"

# Configuration variables (easily configurable)
GLOBAL_BATCH_SIZE=6144
SEQ_LENGTH=2048
MICRO_BATCH_SIZE=1
WORLD_SIZE=8192
DATASET_SIZE_TOKENS=500000000000  # 500B tokens
CARDS_PER_MACHINE=8
MACHINE_COST_RMB_PER_DAY=1344     # 8-card machine cost per day in RMB
EXCHANGE_RATE_USD_PER_RMB=0.14    # USD per RMB (configurable)
SECONDS_PER_DAY=86400

# Derived variables
MACHINE_COST_USD_PER_DAY=$(echo "scale=4; $MACHINE_COST_RMB_PER_DAY * $EXCHANGE_RATE_USD_PER_RMB" | bc)
TOTAL_MACHINES=$(echo "scale=0; $WORLD_SIZE / $CARDS_PER_MACHINE" | bc)

# Directories
TIME_LOGS_DIR="${SCRIPT_DIR}/time_logs"
OUTPUT_DIR="${SCRIPT_DIR}"
RESULTS_FILE="$OUTPUT_DIR/training_metrics_$(date +%Y%m%d_%H%M%S).txt"
CSV_FILE="$OUTPUT_DIR/training_metrics_$(date +%Y%m%d_%H%M%S).csv"

echo "Configuration Parameters:"
echo "  Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Micro Batch Size: $MICRO_BATCH_SIZE"
echo "  World Size: $WORLD_SIZE cards"
echo "  Dataset Size: $DATASET_SIZE_TOKENS tokens"
echo "  Cards per Machine: $CARDS_PER_MACHINE"
echo "  Total Machines: $TOTAL_MACHINES"
echo "  Machine Cost (RMB/day): $MACHINE_COST_RMB_PER_DAY"
echo "  Exchange Rate (USD/RMB): $EXCHANGE_RATE_USD_PER_RMB"
echo "  Machine Cost (USD/day): $MACHINE_COST_USD_PER_DAY"
echo ""

# Check if bc is available for calculations
if ! command -v bc &> /dev/null; then
    echo "ERROR: bc calculator is required but not installed"
    echo "Please install bc: sudo apt-get install bc"
    exit 1
fi

# Check if time logs directory exists
if [ ! -d "$TIME_LOGS_DIR" ]; then
    echo "ERROR: Time logs directory does not exist: $TIME_LOGS_DIR"
    echo "Please run the performance testing script first"
    exit 1
fi

# Function to parse timing information from new format
parse_timing_info() {
    local timing_line=$1

    # Parse format: "setting_name: load_time=X.XX execution_time=Y.YY simulated_step_time=Z.ZZ"
    local load_time=$(echo "$timing_line" | grep -oE 'load_time=[0-9]+\.?[0-9]*' | grep -oE '[0-9]+\.?[0-9]*')
    local execution_time=$(echo "$timing_line" | grep -oE 'execution_time=[0-9]+\.?[0-9]*' | grep -oE '[0-9]+\.?[0-9]*')
    local simulated_step_time=$(echo "$timing_line" | grep -oE 'simulated_step_time=[0-9]+\.?[0-9]*' | grep -oE '[0-9]+\.?[0-9]*')

    echo "$load_time $execution_time $simulated_step_time"
}

# Function to calculate metrics for a single setting
calculate_setting_metrics() {
    local setting_name=$1
    local load_time=$2
    local execution_time=$3
    local simulated_step_time=$4

    # Skip if simulated step time is not a valid number
    if ! [[ "$simulated_step_time" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "SKIP: Invalid simulated step time value for $setting_name: $simulated_step_time"
        return 1
    fi

    # Use simulated step time as the primary metric for throughput calculation
    local total_time=$simulated_step_time

    # Calculate throughput (tokens/second)
    local tokens_per_iteration=$(echo "scale=0; $GLOBAL_BATCH_SIZE * $SEQ_LENGTH" | bc)
    local throughput=$(echo "scale=4; $tokens_per_iteration / $total_time" | bc)

    # Calculate training time in days
    local training_time_days=$(echo "scale=6; $DATASET_SIZE_TOKENS / ($throughput * $SECONDS_PER_DAY)" | bc)

    # Calculate total cost in USD
    local total_cost_usd=$(echo "scale=2; $TOTAL_MACHINES * $training_time_days * $MACHINE_COST_USD_PER_DAY" | bc)

    # Store results in associative arrays (simulated with variables)
    eval "${setting_name}_throughput=$throughput"
    eval "${setting_name}_training_days=$training_time_days"
    eval "${setting_name}_total_cost=$total_cost_usd"
    eval "${setting_name}_load_time=$load_time"
    eval "${setting_name}_execution_time=$execution_time"
    eval "${setting_name}_simulated_step_time=$simulated_step_time"
    eval "${setting_name}_total_time=$total_time"

    echo "✓ Calculated metrics for $setting_name"
    return 0
}

# Function to format numbers for display
format_number() {
    local number=$1
    local decimals=${2:-2}
    printf "%.${decimals}f" "$number"
}

# Function to format large numbers with commas
format_large_number() {
    local number=$1
    # Use printf to format with commas
    printf "%'d" "$(echo "scale=0; $number/1" | bc)"
}

# Initialize results file
echo "Training Metrics and Cost Analysis Report" > "$RESULTS_FILE"
echo "Generated on: $(date)" >> "$RESULTS_FILE"
echo "============================================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "Configuration Parameters:" >> "$RESULTS_FILE"
echo "  Global Batch Size: $GLOBAL_BATCH_SIZE" >> "$RESULTS_FILE"
echo "  Sequence Length: $SEQ_LENGTH" >> "$RESULTS_FILE"
echo "  World Size: $WORLD_SIZE cards" >> "$RESULTS_FILE"
echo "  Dataset Size: $(format_large_number $DATASET_SIZE_TOKENS) tokens" >> "$RESULTS_FILE"
echo "  Total Machines: $TOTAL_MACHINES" >> "$RESULTS_FILE"
echo "  Machine Cost: \$$(format_number $MACHINE_COST_USD_PER_DAY)/day" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Initialize CSV file
echo "Setting,Load_Time_Seconds,Execution_Time_Seconds,Simulated_Step_Time_Seconds,Throughput_Tokens_Per_Second,Training_Days,Total_Cost_USD" > "$CSV_FILE"

echo "Processing time log files..."
echo ""

# Arrays to store setting names and their data
declare -a setting_names=()
declare -a valid_settings=()

# Read all time log files and calculate metrics
for time_file in "$TIME_LOGS_DIR"/*_time.txt; do
    if [ -f "$time_file" ]; then
        # Extract setting name and timing info from file
        while IFS=': ' read -r setting_name timing_info; do
            if [ -n "$setting_name" ] && [ -n "$timing_info" ]; then
                setting_names+=("$setting_name")

                echo "Processing: $setting_name"
                echo "  Timing info: $timing_info"

                # Parse timing information
                parsed_times=$(parse_timing_info "$timing_info")
                read -r load_time execution_time simulated_step_time <<< "$parsed_times"

                echo "  Load time: $load_time seconds"
                echo "  Execution time: $execution_time seconds"
                echo "  Simulated step time: $simulated_step_time seconds"

                if calculate_setting_metrics "$setting_name" "$load_time" "$execution_time" "$simulated_step_time"; then
                    valid_settings+=("$setting_name")
                fi
            fi
        done < "$time_file"
    fi
done

echo ""
echo "Metrics calculation completed for ${#valid_settings[@]} settings"
echo ""

# Generate formatted output
echo "============================================================"
echo "Training Metrics and Cost Analysis Results"
echo "============================================================"

# Table header
printf "%-40s %10s %10s %10s %15s %12s %15s\n" "Setting" "Load (s)" "Exec (s)" "Step (s)" "Throughput" "Days" "Cost (USD)"
printf "%-40s %10s %10s %10s %15s %12s %15s\n" "----------------------------------------" "----------" "----------" "----------" "---------------" "------------" "---------------"

# Write table header to results file
echo "Results Table:" >> "$RESULTS_FILE"
printf "%-40s %10s %10s %10s %15s %12s %15s\n" "Setting" "Load (s)" "Exec (s)" "Step (s)" "Throughput" "Days" "Cost (USD)" >> "$RESULTS_FILE"
printf "%-40s %10s %10s %10s %15s %12s %15s\n" "----------------------------------------" "----------" "----------" "----------" "---------------" "------------" "---------------" >> "$RESULTS_FILE"

# Process each valid setting and display results
for setting_name in "${valid_settings[@]}"; do
    # Retrieve calculated values
    eval "load_time=\$${setting_name}_load_time"
    eval "execution_time=\$${setting_name}_execution_time"
    eval "simulated_step_time=\$${setting_name}_simulated_step_time"
    eval "throughput=\$${setting_name}_throughput"
    eval "training_days=\$${setting_name}_training_days"
    eval "total_cost=\$${setting_name}_total_cost"

    # Format values for display
    formatted_load_time=$(format_number "$load_time" 2)
    formatted_execution_time=$(format_number "$execution_time" 2)
    formatted_simulated_step_time=$(format_number "$simulated_step_time" 3)
    formatted_throughput=$(format_number "$throughput" 0)
    formatted_days=$(format_number "$training_days" 1)
    formatted_cost=$(format_number "$total_cost" 0)

    # Display table row
    printf "%-40s %10s %10s %10s %15s %12s %15s\n" "$setting_name" "$formatted_load_time" "$formatted_execution_time" "$formatted_simulated_step_time" "$formatted_throughput" "$formatted_days" "\$$formatted_cost"

    # Write to results file
    printf "%-40s %10s %10s %10s %15s %12s %15s\n" "$setting_name" "$formatted_load_time" "$formatted_execution_time" "$formatted_simulated_step_time" "$formatted_throughput" "$formatted_days" "\$$formatted_cost" >> "$RESULTS_FILE"

    # Write to CSV file
    echo "$setting_name,$load_time,$execution_time,$simulated_step_time,$throughput,$training_days,$total_cost" >> "$CSV_FILE"
done

echo ""
echo "============================================================"

# Find best performing configuration (highest throughput)
if [ ${#valid_settings[@]} -gt 0 ]; then
    echo ""
    echo "Performance Analysis:"
    echo "------------------------------------------------------------"

    best_setting=""
    best_throughput=0
    lowest_cost_setting=""
    lowest_cost=999999999
    fastest_setting=""
    fastest_days=999999999

    for setting_name in "${valid_settings[@]}"; do
        eval "throughput=\$${setting_name}_throughput"
        eval "total_cost=\$${setting_name}_total_cost"
        eval "training_days=\$${setting_name}_training_days"

        # Check for best throughput
        if (( $(echo "$throughput > $best_throughput" | bc -l) )); then
            best_throughput=$throughput
            best_setting=$setting_name
        fi

        # Check for lowest cost
        if (( $(echo "$total_cost < $lowest_cost" | bc -l) )); then
            lowest_cost=$total_cost
            lowest_cost_setting=$setting_name
        fi

        # Check for fastest training
        if (( $(echo "$training_days < $fastest_days" | bc -l) )); then
            fastest_days=$training_days
            fastest_setting=$setting_name
        fi
    done

    echo "Best Throughput: $best_setting ($(format_number $best_throughput 0) tokens/s)"
    echo "Lowest Cost: $lowest_cost_setting (\$$(format_number $lowest_cost 0))"
    echo "Fastest Training: $fastest_setting ($(format_number $fastest_days 1) days)"

    # Write analysis to results file
    echo "" >> "$RESULTS_FILE"
    echo "Performance Analysis:" >> "$RESULTS_FILE"
    echo "------------------------------------------------------------" >> "$RESULTS_FILE"
    echo "Best Throughput: $best_setting ($(format_number $best_throughput 0) tokens/s)" >> "$RESULTS_FILE"
    echo "Lowest Cost: $lowest_cost_setting (\$$(format_number $lowest_cost 0))" >> "$RESULTS_FILE"
    echo "Fastest Training: $fastest_setting ($(format_number $fastest_days 1) days)" >> "$RESULTS_FILE"
fi

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Total settings processed: ${#setting_names[@]}"
echo "Valid calculations: ${#valid_settings[@]}"
echo ""
echo "Output files generated:"
echo "  Detailed report: $RESULTS_FILE"
echo "  CSV data: $CSV_FILE"
echo ""

# Display file locations
echo "Files saved to:"
echo "  $RESULTS_FILE"
echo "  $CSV_FILE"
echo ""

echo "Cost calculation script completed at: $(date)"
echo "============================================================"
