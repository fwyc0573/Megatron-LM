#!/bin/bash

# Performance Testing Script for 8192 setting configurations.
# This version invokes simu_main.py via CLI arguments and never edits source files.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_DIR="${REPO_ROOT}/simulation_inputs/megatron_operation_log/8192_sim"
TIME_LOGS_DIR="${SCRIPT_DIR}/time_logs"
SIMU_MAIN_SCRIPT="${REPO_ROOT}/simu_main.py"
LOG_FILE="${TIME_LOGS_DIR}/performance_test_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${TIME_LOGS_DIR}"

echo "Performance Testing Started: $(date)" > "${LOG_FILE}"
echo "Base Directory: ${BASE_DIR}" >> "${LOG_FILE}"
echo "Time Logs Directory: ${TIME_LOGS_DIR}" >> "${LOG_FILE}"
echo "Simulation Script: ${SIMU_MAIN_SCRIPT}" >> "${LOG_FILE}"

echo "============================================================"
echo "Performance Testing Script for 8192 Setting Configurations"
echo "============================================================"
echo "Base Directory: ${BASE_DIR}"
echo "Time Logs Directory: ${TIME_LOGS_DIR}"
echo "Simulation Script: ${SIMU_MAIN_SCRIPT}"
echo "Log File: ${LOG_FILE}"
echo ""

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

extract_timing_info() {
    local output_file=$1

    local load_time
    local execution_time
    local simulated_step_time

    load_time=$(grep "sim load time:" "${output_file}" | grep -oE [0-9]+.?[0-9]* | tail -1)
    execution_time=$(grep "sim execution time:" "${output_file}" | grep -oE [0-9]+.?[0-9]* | tail -1)
    simulated_step_time=$(grep "rank0 sum_time:" "${output_file}" | grep -oE [0-9]+.?[0-9]* | tail -1)

    if [ -n "${simulated_step_time}" ]; then
        simulated_step_time=$(echo "scale=6; ${simulated_step_time} / 1000" | bc)
        if [[ "${simulated_step_time}" =~ ^\. ]]; then
            simulated_step_time="0${simulated_step_time}"
        fi
    fi

    if [ -z "${load_time}" ] || [ -z "${execution_time}" ] || [ -z "${simulated_step_time}" ]; then
        echo "N/A N/A N/A"
    else
        echo "${load_time} ${execution_time} ${simulated_step_time}"
    fi
}

run_performance_test() {
    local setting_dir=$1
    local setting_name
    setting_name=$(basename "${setting_dir}")

    echo "------------------------------------------------------------"
    echo "Testing Setting: ${setting_name}"
    echo "------------------------------------------------------------"

    if [ ! -d "${setting_dir}/schedule" ] || [ ! -d "${setting_dir}/database_profile" ]; then
        echo "ERROR: required subdirectories missing for ${setting_name}" | tee -a "${LOG_FILE}"
        return 1
    fi

    local parsed_values
    parsed_values=$(parse_setting_name "${setting_name}")
    local pp_size tp_size dp_size
    read -r pp_size tp_size dp_size <<< "${parsed_values}"

    if [ -z "${pp_size}" ] || [ -z "${tp_size}" ] || [ -z "${dp_size}" ]; then
        echo "ERROR: Could not parse setting name: ${setting_name}" | tee -a "${LOG_FILE}"
        return 1
    fi

    local world_size=8192
    local local_size=8
    local vis_end_param=$((8 * tp_size))
    if [ "${vis_end_param}" -gt "${world_size}" ]; then
        vis_end_param=${world_size}
    fi

    local temp_output="${TIME_LOGS_DIR}/${setting_name}_temp_output.txt"
    local time_record_file="${TIME_LOGS_DIR}/${setting_name}_time.txt"

    echo "Running simulation: PP=${pp_size}, TP=${tp_size}, DP=${dp_size}" | tee -a "${LOG_FILE}"

    (
        cd "${REPO_ROOT}" || exit 1
        timeout 1800 python "${SIMU_MAIN_SCRIPT}" \
            --framework megatron-lm \
            --mode simulate \
            --schedule-dir "simulation_inputs/megatron_operation_log/8192_sim/${setting_name}/schedule" \
            --database-dir "simulation_inputs/megatron_operation_log/8192_sim/${setting_name}/database_profile" \
            --world-size "${world_size}" \
            --pp-size "${pp_size}" \
            --tp-size "${tp_size}" \
            --exp-size 1 \
            --local-size "${local_size}" \
            --visualize-rank-start 0 \
            --visualize-rank-end "${vis_end_param}" \
            --no-visualize
    ) > "${temp_output}" 2>&1
    local exit_code=$?

    if [ ${exit_code} -eq 124 ]; then
        echo "WARNING: Test timed out for ${setting_name}" | tee -a "${LOG_FILE}"
        echo "${setting_name}: TIMEOUT TIMEOUT TIMEOUT" > "${time_record_file}"
    elif [ ${exit_code} -ne 0 ]; then
        echo "ERROR: Test failed with exit code ${exit_code} for ${setting_name}" | tee -a "${LOG_FILE}"
        echo "${setting_name}: ERROR ERROR ERROR" > "${time_record_file}"
    else
        local timing_info
        timing_info=$(extract_timing_info "${temp_output}")
        local load_time execution_time simulated_step_time
        read -r load_time execution_time simulated_step_time <<< "${timing_info}"

        echo "${setting_name}: load_time=${load_time} execution_time=${execution_time} simulated_step_time=${simulated_step_time}" > "${time_record_file}"
        echo "✓ Load time: ${load_time} seconds" | tee -a "${LOG_FILE}"
        echo "✓ Execution time: ${execution_time} seconds" | tee -a "${LOG_FILE}"
        echo "✓ Simulated step time: ${simulated_step_time} seconds" | tee -a "${LOG_FILE}"
    fi

    {
        echo "=== Output for ${setting_name} ==="
        cat "${temp_output}"
        echo "=== End Output for ${setting_name} ==="
        echo ""
    } >> "${LOG_FILE}"

    rm -f "${temp_output}"

    echo "✓ Performance test completed for ${setting_name}"
    echo ""
    return 0
}

if [ ! -d "${BASE_DIR}" ]; then
    echo "ERROR: Base directory does not exist: ${BASE_DIR}"
    exit 1
fi

all_setting_dirs=("${BASE_DIR}"/*/)
if [ ${#all_setting_dirs[@]} -eq 0 ]; then
    echo "No setting directories found under ${BASE_DIR}"
    exit 1
fi

total_settings=0
for setting_dir in "${BASE_DIR}"/*/; do
    [ -d "${setting_dir}" ] && total_settings=$((total_settings + 1))
done

successful_tests=0
failed_tests=0
current_test=0

for setting_dir in "${BASE_DIR}"/*/; do
    if [ -d "${setting_dir}" ]; then
        current_test=$((current_test + 1))
        echo "Progress: ${current_test}/${total_settings}"
        if run_performance_test "${setting_dir}"; then
            successful_tests=$((successful_tests + 1))
        else
            failed_tests=$((failed_tests + 1))
        fi
        echo "============================================================"
    fi
done

echo ""
echo "============================================================"
echo "Performance Testing Summary"
echo "============================================================"
echo "Total settings tested: ${total_settings}"
echo "Successful tests: ${successful_tests}"
echo "Failed tests: ${failed_tests}"
echo "Time records saved in: ${TIME_LOGS_DIR}"
echo "Detailed log saved in: ${LOG_FILE}"

echo ""
echo "Time Records Summary:"
echo "------------------------------------------------------------"
for time_file in "${TIME_LOGS_DIR}"/*_time.txt; do
    [ -f "${time_file}" ] && cat "${time_file}"
done

echo ""
echo "Performance testing script completed at: $(date)"
echo "============================================================"
