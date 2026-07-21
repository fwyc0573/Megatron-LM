#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "${REPO_ROOT}/SC26-AE/lib/common.sh"
source "${REPO_ROOT}/SC26-AE/lib/task1_trace.sh"

TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/sc26-ae-task1-ncu.XXXXXX")
FAKE_NCU="${TEST_ROOT}/ncu"
FAKE_TORCHRUN="${TEST_ROOT}/torchrun"
SOURCE_SCRIPT="${TEST_ROOT}/source.sh"
CAPTURE_ROOT="${TEST_ROOT}/capture"
WORK_ROOT="${TEST_ROOT}/work"

cat >"${FAKE_TORCHRUN}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
exit 0
SH
chmod +x "${FAKE_TORCHRUN}"

cat >"${SOURCE_SCRIPT}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
"${AE_TASK1_REAL_TORCHRUN}" --global-batch-size 768 --fake-current-rank-id 0
SH
chmod +x "${SOURCE_SCRIPT}"

cat >"${FAKE_NCU}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
report_base=""
log_file=""
for ((index = 1; index <= $#; index++)); do
    argument=${!index}
    next_index=$((index + 1))
    case "${argument}" in
        -o)
            report_base=${!next_index}
            ;;
        --log-file)
            log_file=${!next_index}
            ;;
    esac
done
if [[ " $* " == *" bash "* ]]; then
    loop=${!#}
    bash "${loop}"
    printf 'fake ncu report\n' >"${report_base}.ncu-rep"
elif [[ " $* " == *" --page details "* ]]; then
    cat >"${log_file}" <<'CSV'
ID,Kernel Name,Section Name,Metric Name,Metric Value
1,kernel_a,GPU Speed Of Light Throughput,Compute (SM) Throughput,10.0
1,kernel_a,GPU Speed Of Light Throughput,Memory Throughput,20.0
1,kernel_a,GPU Speed Of Light Throughput,DRAM Throughput,30.0
1,kernel_a,Occupancy,Achieved Occupancy,40.0
1,kernel_a,Occupancy,Theoretical Occupancy,50.0
1,kernel_a,Memory Workload Analysis,L1/TEX Hit Rate,60.0
1,kernel_a,Memory Workload Analysis,L2 Hit Rate,70.0
CSV
else
    cat <<'CSV'
ID,Kernel Name
1,kernel_a
CSV
fi
SH
chmod +x "${FAKE_NCU}"

ae_task1_load_config gpt175b "${REPO_ROOT}"
AE_TASK1_REAL_TORCHRUN="${FAKE_TORCHRUN}"
AE_T1_SOURCE_SCRIPT="${SOURCE_SCRIPT}"
export AE_TASK1_REAL_TORCHRUN AE_T1_SOURCE_SCRIPT

ae_task1_execute_ncu_capture \
    "$(command -v python3)" "${FAKE_NCU}" gpt175b ncu-test-capture 0 \
    "${CAPTURE_ROOT}" "${WORK_ROOT}"

[[ -s "${CAPTURE_ROOT}/ncu/rank0.ncu-rep" ]]
[[ -s "${CAPTURE_ROOT}/ncu/rank0_details.csv" ]]
[[ -s "${CAPTURE_ROOT}/ncu/rank0_raw.csv" ]]
[[ -s "${CAPTURE_ROOT}/ncu/kernel_metric_output.csv" ]]
grep -Fq 'Kernel Name,Compute throughput,Memory throughput,DRAM throughput,Achieved occupancy,Maximum occupancy,L1 hit rate,L2 hit rate' \
    "${CAPTURE_ROOT}/ncu/kernel_metric_output.csv"

METADATA="${TEST_ROOT}/metadata.json"
printf '{"schema_version":"sc26-ae-artifact-manifest-v1"}\n' >"${METADATA}"
ae_task1_attach_ncu_metadata \
    "$(command -v python3)" "${METADATA}" "${CAPTURE_ROOT}" \
    "${CAPTURE_ROOT}/ncu/kernel_metric_output.csv" \
    "${CAPTURE_ROOT}/ncu/rank0_details.csv" \
    "${CAPTURE_ROOT}/ncu/rank0_raw.csv"
python3 - "${METADATA}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["ncu_feature_provenance"]["rank_scope"] == "global_rank_0"
assert payload["ncu_feature_provenance"]["rank_ids"] == [0]
assert payload["ncu_feature_provenance"]["physical_gpu_count"] == 1
assert payload["ncu_feature_provenance"]["required_kernel_count"] == 1
PY

printf 'PASS: Task1 rank-0 NCU capture and provenance normalization\n'
