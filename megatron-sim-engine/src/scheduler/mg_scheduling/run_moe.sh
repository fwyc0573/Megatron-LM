#!/bin/bash

# MoE Scheduling Plan Generator for All Configurations
# This script automatically generates scheduling plans for all MoE configurations
# listed in simulation_inputs/megatron_operation_log/new_moe/reorganization_summary.txt

# Configuration
train_iters=3
trace_iter_num=1 # trace_iter_num的范围<=train_iters-2
trace_start=$(($train_iters-$trace_iter_num+1)) # [start, train_iters]

# 检查trace_iter_num是否在合理的范围内
if [ $trace_iter_num -gt $((train_iters - 2)) ]; then
  echo "Error: trace_iter_num must be less than or equal to train_iters - 2"
  exit 1
fi

# Paths configuration - determine base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCHED_LOG_DIR="${REPO_ROOT}/log/mg_scheduling"
SCHED_PLAN_LOG_DIR="${REPO_ROOT}/simulation_inputs/scheduling_plans/mg_scheduling_plan_log"
mkdir -p "${SCHED_LOG_DIR}" "${SCHED_PLAN_LOG_DIR}"

# Check if we're in the megatron-sim-engine directory structure
if [[ -d "${REPO_ROOT}/simulation_inputs/megatron_operation_log/new_moe" ]]; then
    SUMMARY_FILE="${REPO_ROOT}/simulation_inputs/megatron_operation_log/new_moe/reorganization_summary.txt"
    TARGET_BASE_DIR="${REPO_ROOT}/simulation_inputs/megatron_operation_log/new_moe"
elif [[ -d "/research/d1/gds/ytyang/yichengfeng/megatron-sim-engine/simulation_inputs/megatron_operation_log/new_moe" ]]; then
    SUMMARY_FILE="/research/d1/gds/ytyang/yichengfeng/megatron-sim-engine/simulation_inputs/megatron_operation_log/new_moe/reorganization_summary.txt"
    TARGET_BASE_DIR="/research/d1/gds/ytyang/yichengfeng/megatron-sim-engine/simulation_inputs/megatron_operation_log/new_moe"
else
    echo "Error: Cannot find simulation_inputs/megatron_operation_log/new_moe directory"
    exit 1
fi

LOG_FILE="${SCHED_LOG_DIR}/moe_scheduling_batch_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

# Function to parse configuration parameters from config name
parse_config_params() {
    local config_name=$1

    # Extract parameters using regex
    if [[ $config_name =~ pp([0-9]+)_tp([0-9]+)_exp([0-9]+)_expn([0-9]+)_dp([0-9]+)_nl([0-9]+)_hs([0-9]+)_sl([0-9]+) ]]; then
        PP=${BASH_REMATCH[1]}
        TP=${BASH_REMATCH[2]}
        EP=${BASH_REMATCH[3]}
        NUM_EXPERTS=${BASH_REMATCH[4]}
        DP=${BASH_REMATCH[5]}
        NUM_LAYERS=${BASH_REMATCH[6]}
        HIDDEN_SIZE=${BASH_REMATCH[7]}
        SEQ_LENGTH=${BASH_REMATCH[8]}

        # Calculate world_size
        WORLD_SIZE=$((PP * TP * DP))

        return 0
    else
        log_message "ERROR" "Failed to parse configuration: $config_name"
        return 1
    fi
}

# Function to determine model configuration based on parameters
determine_model_config() {
    local hidden_size=$1
    local num_layers=$2
    local num_experts=$3

    # Set NUM_HEAD based on HIDDEN_SIZE (common pattern)
    case $hidden_size in
        128)   NUM_HEAD=8 ;;
        4096)  NUM_HEAD=32 ;;
        5120)  NUM_HEAD=32 ;;
        6144)  NUM_HEAD=56 ;;
        7680)  NUM_HEAD=48 ;;
        8192)  NUM_HEAD=64 ;;
        9216)  NUM_HEAD=72 ;;
        12288) NUM_HEAD=96 ;;
        *)     NUM_HEAD=32 ;; # Default
    esac

    # Set FFN_HIDDEN_SIZE (typically 3.5x hidden_size for MoE models)
    FFN_HIDDEN_SIZE=$((hidden_size * 14336 / 4096))  # Scale based on 4096->14336 ratio

    # Determine MODEL_SIZE based on parameters
    if [[ $hidden_size -eq 4096 && $num_layers -eq 8 ]]; then
        MODEL_SIZE="Mixtral_${num_experts}x1.75B"
    elif [[ $hidden_size -eq 4096 && $num_layers -eq 32 ]]; then
        MODEL_SIZE="Mixtral_${num_experts}x7B"
    elif [[ $hidden_size -eq 6144 && $num_layers -eq 56 ]]; then
        MODEL_SIZE="Mixtral_${num_experts}x22B"
    else
        MODEL_SIZE="Custom_${num_experts}x_hs${hidden_size}_nl${num_layers}"
    fi

    log_message "INFO" "Model config: SIZE=$MODEL_SIZE, HIDDEN=$hidden_size, LAYERS=$num_layers, HEADS=$NUM_HEAD, FFN=$FFN_HIDDEN_SIZE"
}


# Function to generate scheduling plan for a single configuration
generate_scheduling_plan() {
    local config_name=$1
    local target_dir="$TARGET_BASE_DIR/$config_name/schedule"

    log_message "INFO" "Processing configuration: $config_name"

    # Parse configuration parameters
    if ! parse_config_params "$config_name"; then
        log_message "ERROR" "Failed to parse parameters for $config_name"
        return 1
    fi

    # Determine model configuration
    determine_model_config "$HIDDEN_SIZE" "$NUM_LAYERS" "$NUM_EXPERTS"

    # Calculate batch parameters
    local num_micbatch=$((4 * PP))
    local micro_batch_size=1
    local global_batch_size=$((num_micbatch * micro_batch_size * DP))

    # Validate target directory
    if [[ ! -d "$target_dir" ]]; then
        log_message "ERROR" "Target directory does not exist: $target_dir"
        return 1
    fi

    # Check if scheduling plans already exist
    local existing_files=$(find "$target_dir" -name "*.txt" 2>/dev/null | wc -l)
    if [[ $existing_files -gt 0 ]]; then
        log_message "WARN" "Scheduling plans already exist in $target_dir ($existing_files files). Skipping..."
        return 0
    fi

    log_message "INFO" "Generating scheduling plan with parameters:"
    log_message "INFO" "  PP=$PP, TP=$TP, DP=$DP, EP=$EP, NUM_EXPERTS=$NUM_EXPERTS"
    log_message "INFO" "  WORLD_SIZE=$WORLD_SIZE, HIDDEN_SIZE=$HIDDEN_SIZE, SEQ_LENGTH=$SEQ_LENGTH"
    log_message "INFO" "  MICRO_BATCH_SIZE=$micro_batch_size, GLOBAL_BATCH_SIZE=$global_batch_size"

    # Run the python script with the calculated variables
    local temp_log_file="${SCHED_LOG_DIR}/temp_${config_name}_$(date +%s).log"

    # Change to script directory to run mg_test.py
    cd "$SCRIPT_DIR"

    if python mg_test.py \
        --local-size 8 \
        --world-size $WORLD_SIZE \
        --micro-batch-size $micro_batch_size \
        --global-batch-size $global_batch_size \
        --seq-length $SEQ_LENGTH \
        --hidden-size $HIDDEN_SIZE \
        --train-iters $train_iters \
        --model-size "$MODEL_SIZE" \
        -pp $PP \
        -tp $TP \
        -exp $EP \
        --num-experts $NUM_EXPERTS \
        --untie-embeddings-and-output-weights \
        --trace-start $trace_start > "$temp_log_file" 2>&1; then

        # Move generated scheduling plan files to target directory
        # Find the most recent scheduling plan files
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local generated_files=$(find "$SCHED_PLAN_LOG_DIR" -name "*.txt" -newer "$temp_log_file" 2>/dev/null | sort)
        local file_count=0

        # If no files found with -newer, try to find recent files by timestamp pattern
        if [[ -z "$generated_files" ]]; then
            local current_time=$(date +%Y%m%d_%H%M)
            generated_files=$(find "$SCHED_PLAN_LOG_DIR" -name "*${current_time}*.txt" 2>/dev/null | sort)
        fi

        # If still no files, try to find the most recent files
        if [[ -z "$generated_files" ]]; then
            generated_files=$(find "$SCHED_PLAN_LOG_DIR" -name "*.txt" -mmin -2 2>/dev/null | sort)
        fi

        # Extract unique stage files (only keep the latest version of each stage)
        declare -A latest_stage_files

        for file in $generated_files; do
            if [[ -f "$file" ]]; then
                local filename=$(basename "$file")
                # Extract stage number from filename (e.g., stage0, stage1, etc.)
                if [[ $filename =~ ^stage([0-9]+)_ ]]; then
                    local stage_num=${BASH_REMATCH[1]}
                    # Keep only the latest file for each stage
                    latest_stage_files[$stage_num]="$file"
                fi
            fi
        done

        # Copy only the required number of stage files (equal to PP value)
        for ((stage=0; stage<PP; stage++)); do
            if [[ -n "${latest_stage_files[$stage]}" ]]; then
                local source_file="${latest_stage_files[$stage]}"
                local target_filename="stage${stage}_${timestamp}_scheduling_plan.txt"
                cp "$source_file" "$target_dir/$target_filename"
                ((file_count++))
                log_message "INFO" "Moved stage$stage -> $target_filename"
            else
                log_message "WARN" "Missing stage$stage file for $config_name"
            fi
        done

        if [[ $file_count -gt 0 ]]; then
            log_message "SUCCESS" "Generated $file_count scheduling plan files for $config_name"
            rm -f "$temp_log_file"
            return 0
        else
            log_message "ERROR" "No scheduling plan files were generated for $config_name"
            log_message "ERROR" "Check log: $temp_log_file"
            return 1
        fi
    else
        log_message "ERROR" "Failed to generate scheduling plan for $config_name"
        log_message "ERROR" "Check log: $temp_log_file"
        return 1
    fi
}

# Function to extract configurations from summary file
extract_configurations() {
    if [[ ! -f "$SUMMARY_FILE" ]]; then
        log_message "ERROR" "Summary file not found: $SUMMARY_FILE"
        return 1
    fi

    # Extract configuration names from Setting Mappings section
    local configs=()

    # Use grep and sed to extract configuration names more reliably
    while IFS= read -r line; do
        if [[ "$line" =~ ^setting[0-9]+:\ (.+)$ ]]; then
            local config_name="${BASH_REMATCH[1]}"
            configs+=("$config_name")
        fi
    done < <(grep "^setting[0-9]*:" "$SUMMARY_FILE")

    if [[ ${#configs[@]} -eq 0 ]]; then
        log_message "ERROR" "No configurations found in summary file"
        return 1
    fi

    log_message "INFO" "Found ${#configs[@]} configurations to process"
    printf '%s\n' "${configs[@]}"
    return 0
}

# Main execution function
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}MoE Scheduling Plan Batch Generator${NC}"
    echo -e "${BLUE}========================================${NC}"

    log_message "INFO" "Starting batch scheduling plan generation"
    log_message "INFO" "Summary file: $SUMMARY_FILE"
    log_message "INFO" "Target base directory: $TARGET_BASE_DIR"
    log_message "INFO" "Log file: $LOG_FILE"

    # Check if summary file exists
    if [[ ! -f "$SUMMARY_FILE" ]]; then
        log_message "ERROR" "Summary file not found: $SUMMARY_FILE"
        echo -e "${RED}Error: Summary file not found!${NC}"
        exit 1
    fi

    # Extract configurations
    local configs
    if ! configs=($(extract_configurations)); then
        echo -e "${RED}Error: Failed to extract configurations!${NC}"
        exit 1
    fi

    local total_configs=${#configs[@]}
    local success_count=0
    local failure_count=0
    local skip_count=0

    echo -e "${GREEN}Found $total_configs configurations to process${NC}"
    echo ""

    # Process each configuration
    for i in "${!configs[@]}"; do
        local config_name="${configs[$i]}"
        local current=$((i + 1))

        echo -e "${YELLOW}[$current/$total_configs] Processing: $config_name${NC}"

        if generate_scheduling_plan "$config_name"; then
            ((success_count++))
            echo -e "${GREEN}✓ Success${NC}"
        else
            ((failure_count++))
            echo -e "${RED}✗ Failed${NC}"
        fi

        echo ""
    done

    # Final summary
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Batch Processing Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Total configurations: $total_configs"
    echo -e "${GREEN}Successful: $success_count${NC}"
    echo -e "${RED}Failed: $failure_count${NC}"
    echo -e "Success rate: $(( success_count * 100 / total_configs ))%"
    echo ""

    log_message "INFO" "Batch processing completed"
    log_message "INFO" "Total: $total_configs, Success: $success_count, Failed: $failure_count"

    if [[ $failure_count -gt 0 ]]; then
        echo -e "${YELLOW}Check log file for details: $LOG_FILE${NC}"
        exit 1
    else
        echo -e "${GREEN}All configurations processed successfully!${NC}"
        exit 0
    fi
}

# Check if running in single config mode or batch mode
if [[ $# -eq 1 ]]; then
    # Single configuration mode
    config_name="$1"
    echo "Processing single configuration: $config_name"
    if generate_scheduling_plan "$config_name"; then
        echo "✓ Successfully generated scheduling plan for $config_name"
        exit 0
    else
        echo "✗ Failed to generate scheduling plan for $config_name"
        exit 1
    fi
else
    # Batch mode - process all configurations
    main
fi
