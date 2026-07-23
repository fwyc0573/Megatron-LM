#!/bin/bash

# Check Scheduling Results Script
# This script checks the results of the batch scheduling plan generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TARGET_BASE_DIR="${REPO_ROOT}/simulation_inputs/megatron_operation_log/new_moe"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MoE Scheduling Plan Generation Results${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ ! -d "$TARGET_BASE_DIR" ]]; then
    echo -e "${RED}Error: Target directory not found: $TARGET_BASE_DIR${NC}"
    exit 1
fi

# Count total configurations
total_configs=0
configs_with_schedules=0
total_schedule_files=0

echo -e "\n${YELLOW}Configuration Analysis:${NC}"
echo "----------------------------------------"

for config_dir in "$TARGET_BASE_DIR"/pp*; do
    if [[ -d "$config_dir" ]]; then
        config_name=$(basename "$config_dir")
        schedule_dir="$config_dir/schedule"
        
        ((total_configs++))
        
        if [[ -d "$schedule_dir" ]]; then
            file_count=$(find "$schedule_dir" -name "*.txt" | wc -l)
            total_schedule_files=$((total_schedule_files + file_count))
            
            if [[ $file_count -gt 0 ]]; then
                ((configs_with_schedules++))
                echo -e "${GREEN}✓${NC} $config_name: $file_count scheduling plan files"
            else
                echo -e "${RED}✗${NC} $config_name: No scheduling plan files"
            fi
        else
            echo -e "${RED}✗${NC} $config_name: Schedule directory missing"
        fi
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary Statistics${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Total configurations found: $total_configs"
echo -e "Configurations with scheduling plans: ${GREEN}$configs_with_schedules${NC}"
echo -e "Configurations without scheduling plans: ${RED}$((total_configs - configs_with_schedules))${NC}"
echo "Total scheduling plan files generated: $total_schedule_files"

if [[ $configs_with_schedules -gt 0 ]]; then
    success_rate=$((configs_with_schedules * 100 / total_configs))
    echo -e "Success rate: ${GREEN}${success_rate}%${NC}"
else
    echo -e "Success rate: ${RED}0%${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}File Distribution by Configuration${NC}"
echo -e "${BLUE}========================================${NC}"

# Show detailed file distribution
for config_dir in "$TARGET_BASE_DIR"/pp*; do
    if [[ -d "$config_dir" ]]; then
        config_name=$(basename "$config_dir")
        schedule_dir="$config_dir/schedule"
        
        if [[ -d "$schedule_dir" ]]; then
            file_count=$(find "$schedule_dir" -name "*.txt" | wc -l)
            
            if [[ $file_count -gt 0 ]]; then
                # Extract PP value to understand expected file count
                if [[ $config_name =~ pp([0-9]+) ]]; then
                    pp_value=${BASH_REMATCH[1]}
                    expected_files=$pp_value  # Should have exactly PP files (one per stage)

                    if [[ $file_count -eq $expected_files ]]; then
                        status="${GREEN}✓${NC}"
                    elif [[ $file_count -gt $expected_files ]]; then
                        status="${YELLOW}⚠${NC} (too many)"
                    else
                        status="${RED}✗${NC} (missing)"
                    fi

                    echo -e "$status $config_name: $file_count files (PP=$pp_value, expected=$expected_files)"
                else
                    echo -e "${GREEN}✓${NC} $config_name: $file_count files"
                fi
            fi
        fi
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Sample File Content${NC}"
echo -e "${BLUE}========================================${NC}"

# Show sample content from one of the generated files
sample_file=$(find "$TARGET_BASE_DIR" -name "stage0_*.txt" | head -1)
if [[ -n "$sample_file" ]]; then
    config_name=$(echo "$sample_file" | sed 's|.*/\([^/]*\)/schedule/.*|\1|')
    echo -e "Sample from: ${YELLOW}$config_name${NC}"
    echo -e "File: ${YELLOW}$(basename "$sample_file")${NC}"
    echo "----------------------------------------"
    head -5 "$sample_file"
    echo "..."
    echo "----------------------------------------"
    echo "Total lines: $(wc -l < "$sample_file")"
else
    echo -e "${RED}No scheduling plan files found${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Directory Structure Verification${NC}"
echo -e "${BLUE}========================================${NC}"

# Verify directory structure
structure_ok=true
for config_dir in "$TARGET_BASE_DIR"/pp*; do
    if [[ -d "$config_dir" ]]; then
        config_name=$(basename "$config_dir")
        
        # Check required subdirectories
        for subdir in "database_profile" "global_ranks_profile" "schedule"; do
            if [[ ! -d "$config_dir/$subdir" ]]; then
                echo -e "${RED}✗${NC} $config_name: Missing $subdir directory"
                structure_ok=false
            fi
        done
    fi
done

if [[ $structure_ok == true ]]; then
    echo -e "${GREEN}✓ All configurations have proper directory structure${NC}"
fi

echo ""
if [[ $configs_with_schedules -eq $total_configs ]]; then
    echo -e "${GREEN}🎉 All configurations have been successfully processed!${NC}"
elif [[ $configs_with_schedules -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Partial success: $configs_with_schedules/$total_configs configurations processed${NC}"
    echo -e "${YELLOW}Consider re-running the batch script for failed configurations${NC}"
else
    echo -e "${RED}❌ No configurations were successfully processed${NC}"
    echo -e "${RED}Check the log files for error details${NC}"
fi

echo ""
