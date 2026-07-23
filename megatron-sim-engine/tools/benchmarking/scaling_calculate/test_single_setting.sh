#!/bin/bash

# Run one performance test setting through CLI arguments (no source mutation).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SETTINGS_BASE_DIR="${REPO_ROOT}/simulation_inputs/megatron_operation_log/8192_sim"
TEST_SETTING="pp64_tp2_ep1_expnNone_dp64_nl96_hs20480_sl2048"
SIMU_MAIN_SCRIPT="${REPO_ROOT}/simu_main.py"
TIME_LOGS_DIR="${SCRIPT_DIR}/time_logs"

mkdir -p "${TIME_LOGS_DIR}"

echo "============================================================"
echo "Testing Single Setting Performance Test"
echo "============================================================"
echo "Test setting: ${TEST_SETTING}"
echo "Settings directory: ${SETTINGS_BASE_DIR}/${TEST_SETTING}"

if [ ! -d "${SETTINGS_BASE_DIR}/${TEST_SETTING}/schedule" ]; then
    echo "ERROR: schedule directory not found"
    exit 1
fi

if [ ! -d "${SETTINGS_BASE_DIR}/${TEST_SETTING}/database_profile" ]; then
    echo "ERROR: database_profile directory not found"
    exit 1
fi

echo "✓ Setting directory structure validated"

parse_setting_name() {
    local setting_name=$1
    local pp_size
    local tp_size
    local dp_size

    pp_size=$(echo "${setting_name}" | grep -oE pp[0-9]+ | grep -oE [0-9]+ || true)
    tp_size=$(echo "${setting_name}" | grep -oE tp[0-9]+ | grep -oE [0-9]+ || true)
    dp_size=$(echo "${setting_name}" | grep -oE dp[0-9]+ | grep -oE [0-9]+ || true)

    echo "${pp_size} ${tp_size} ${dp_size}"
}

read -r pp_size tp_size dp_size <<< "$(parse_setting_name "${TEST_SETTING}")"
echo "Parsed configuration: PP=${pp_size}, TP=${tp_size}, DP=${dp_size}"

world_size=8192
calculated_world_size=$((pp_size * tp_size * dp_size))
if [ "${calculated_world_size}" -ne "${world_size}" ]; then
    echo "ERROR: World size mismatch. Expected: ${world_size}, Calculated: ${calculated_world_size}"
    exit 1
fi

echo "✓ World size calculation validated"

vis_end_param=$((8 * tp_size))
if [ "${vis_end_param}" -gt "${world_size}" ]; then
    vis_end_param=${world_size}
fi

echo "Running performance test (timeout: 10 minutes)..."
temp_output="${TIME_LOGS_DIR}/${TEST_SETTING}_single_test_output.txt"

start_time=$(date)
echo "Test started at: ${start_time}"

(
    cd "${REPO_ROOT}" || exit 1
    timeout 600 python "${SIMU_MAIN_SCRIPT}" \
        --framework megatron-lm \
        --mode simulate \
        --schedule-dir "simulation_inputs/megatron_operation_log/8192_sim/${TEST_SETTING}/schedule" \
        --database-dir "simulation_inputs/megatron_operation_log/8192_sim/${TEST_SETTING}/database_profile" \
        --world-size "${world_size}" \
        --pp-size "${pp_size}" \
        --tp-size "${tp_size}" \
        --exp-size 1 \
        --local-size 8 \
        --visualize-rank-start 0 \
        --visualize-rank-end "${vis_end_param}" \
        --no-visualize
) > "${temp_output}" 2>&1
exit_code=$?

end_time=$(date)
echo "Test completed at: ${end_time}"

if [ ${exit_code} -eq 124 ]; then
    echo "WARNING: Test timed out after 10 minutes"
elif [ ${exit_code} -ne 0 ]; then
    echo "ERROR: Test failed with exit code ${exit_code}"
    echo "Last 20 lines of output:"
    tail -20 "${temp_output}"
else
    echo "✓ Test completed successfully"
    echo ""
    echo "Extracting timing information..."

    load_time=$(grep "sim load time:" "${temp_output}" | grep -oE [0-9]+.?[0-9]* | tail -1)
    execution_time=$(grep "sim execution time:" "${temp_output}" | grep -oE [0-9]+.?[0-9]* | tail -1)
    simulated_step_time=$(grep "rank0 sum_time:" "${temp_output}" | grep -oE [0-9]+.?[0-9]* | tail -1)

    if [ -n "${simulated_step_time}" ]; then
        simulated_step_time=$(echo "scale=6; ${simulated_step_time} / 1000" | bc)
        if [[ "${simulated_step_time}" =~ ^\. ]]; then
            simulated_step_time="0${simulated_step_time}"
        fi
    fi

    echo "Timing Results:"
    echo "  Load time: ${load_time:-N/A} seconds"
    echo "  Execution time: ${execution_time:-N/A} seconds"
    echo "  Simulated step time: ${simulated_step_time:-N/A} seconds"

    if [ -n "${load_time}" ] && [ -n "${execution_time}" ] && [ -n "${simulated_step_time}" ]; then
        echo "${TEST_SETTING}: load_time=${load_time} execution_time=${execution_time} simulated_step_time=${simulated_step_time}" > "${TIME_LOGS_DIR}/${TEST_SETTING}_single_test_time.txt"
        echo "✓ Timing record saved"
    else
        echo "WARNING: Could not extract complete timing information"
    fi
fi

echo ""
echo "============================================================"
echo "Single setting test completed"
echo "============================================================"
