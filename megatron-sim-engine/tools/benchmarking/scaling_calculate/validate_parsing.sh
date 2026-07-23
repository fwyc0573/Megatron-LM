#!/bin/bash

# Validation script to test setting name parsing functionality

echo "============================================================"
echo "Setting Name Parsing Validation"
echo "============================================================"

# Function to parse setting name and extract PP, TP, DP values
parse_setting_name() {
    local setting_name=$1

    # Parse format: pp{X}_tp{Y}_ep1_expnNone_dp{Z}_nl96_hs20480_sl2048
    local pp_size=$(echo "$setting_name" | grep -oE 'pp[0-9]+' | grep -oE '[0-9]+')
    local tp_size=$(echo "$setting_name" | grep -oE 'tp[0-9]+' | grep -oE '[0-9]+')
    local dp_size=$(echo "$setting_name" | grep -oE 'dp[0-9]+' | grep -oE '[0-9]+')

    echo "$pp_size $tp_size $dp_size"
}

# Test cases based on actual setting directory names
TEST_SETTINGS=(
    "pp16_tp8_ep1_expnNone_dp64_nl96_hs20480_sl2048"
    "pp32_tp8_ep1_expnNone_dp32_nl96_hs20480_sl2048"
    "pp32_tp4_ep1_expnNone_dp64_nl96_hs20480_sl2048"
    "pp64_tp8_ep1_expnNone_dp16_nl96_hs20480_sl2048"
    "pp64_tp4_ep1_expnNone_dp32_nl96_hs20480_sl2048"
    "pp64_tp2_ep1_expnNone_dp64_nl96_hs20480_sl2048"
)

echo "Testing setting name parsing..."
echo ""

for setting in "${TEST_SETTINGS[@]}"; do
    echo "Setting: $setting"

    parsed_values=$(parse_setting_name "$setting")
    read -r pp_size tp_size dp_size <<< "$parsed_values"

    echo "  PP: $pp_size, TP: $tp_size, DP: $dp_size"

    # Calculate world_size and visualization parameter
    world_size=8192
    calculated_world_size=$((pp_size * tp_size * dp_size))
    vis_end_param=$((8 * tp_size))
    if [ $vis_end_param -gt $world_size ]; then
        vis_end_param=$world_size
    fi

    echo "  Calculated World Size: $calculated_world_size"
    echo "  Expected World Size: $world_size"
    echo "  Visualization End Parameter: $vis_end_param"

    if [ $calculated_world_size -eq $world_size ]; then
        echo "  ✓ World size calculation matches"
    else
        echo "  ✗ World size calculation mismatch"
    fi

    echo ""
done

echo "============================================================"
echo "Parsing validation completed"
echo "============================================================"