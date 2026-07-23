#!/bin/bash

# Clean Schedule Files Script
# This script cleans all existing scheduling plan files from schedule directories

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
echo -e "${BLUE}Cleaning Schedule Files${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ ! -d "$TARGET_BASE_DIR" ]]; then
    echo -e "${RED}Error: Target directory not found: $TARGET_BASE_DIR${NC}"
    exit 1
fi

total_files_removed=0
configs_cleaned=0

echo -e "\n${YELLOW}Cleaning schedule directories:${NC}"
echo "----------------------------------------"

for config_dir in "$TARGET_BASE_DIR"/pp*; do
    if [[ -d "$config_dir" ]]; then
        config_name=$(basename "$config_dir")
        schedule_dir="$config_dir/schedule"
        
        if [[ -d "$schedule_dir" ]]; then
            # Count existing files
            file_count=$(find "$schedule_dir" -name "*.txt" | wc -l)
            
            if [[ $file_count -gt 0 ]]; then
                # Remove all .txt files
                find "$schedule_dir" -name "*.txt" -delete
                total_files_removed=$((total_files_removed + file_count))
                ((configs_cleaned++))
                echo -e "${GREEN}✓${NC} $config_name: Removed $file_count files"
            else
                echo -e "${YELLOW}-${NC} $config_name: No files to remove"
            fi
        else
            echo -e "${RED}✗${NC} $config_name: Schedule directory missing"
        fi
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Cleanup Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Configurations cleaned: $configs_cleaned"
echo "Total files removed: $total_files_removed"

if [[ $total_files_removed -gt 0 ]]; then
    echo -e "${GREEN}✅ Cleanup completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ No files were found to clean${NC}"
fi

echo ""
