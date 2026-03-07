## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Added overlap decomposition report for new-v1/new-v2 assumptions |

# Test Report: Overlap Decomposition for new-v1/new-v2 (WS256)

**Date**: 2026-03-05

## 1. Test Script Information
- Source CSV:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_new_v1_new_v2_e2e_comp_comm_bubble.csv`
- Output CSV:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_new_v1_new_v2_pure_comp_pure_comm_bubble_overlap.csv`
- Output JSON:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_new_v1_new_v2_pure_comp_pure_comm_bubble_overlap_summary.json`

### Reproducible Command
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
python - <<'PY'
import csv
from pathlib import Path

csv_path = Path("task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_new_v1_new_v2_pure_comp_pure_comm_bubble_overlap.csv")
max_diff = 0.0
with csv_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        diff = abs(float(row["recompose_minus_e2e_ms"]))
        max_diff = max(max_diff, diff)
        if diff > 1e-9:
            raise SystemExit(f"FAIL: {row['variant']} diff={diff}")
print("PASS")
print(f"max_abs_recompose_diff_ms={max_diff:.12f}")
PY
```

### Environment
- Python: `3.9.18`

## 2. Validation Criteria
- `groundtruth overlap = 5.57% * groundtruth_e2e`
- `ours overlap error vs groundtruth = +1.05%`
- `new-v1 overlap error vs groundtruth = +21.05%`
- `new-v2 overlap error vs groundtruth = -6.75%`
- `pure_comp = comp_execute - overlap/2`
- `pure_comm_execute = comm_execute - overlap/2`
- `e2e == pure_comp + pure_comm_execute + overlap + bubble`

## 3. Decomposition Results (ms)

| variant | pure_comp_ms | pure_comm_execute_ms | bubble_ms | overlap_ms | overlap/e2e |
|---------|-------------:|---------------------:|----------:|-----------:|------------:|
| groundtruth | 854.240820 | 2164.232004 | 766.158816 | 223.238359 | 5.570000% |
| ours | 835.678819 | 1861.940893 | 664.787926 | 225.582362 | 6.287151% |
| new-v1 | 1693.451135 | 1882.108438 | 1341.006545 | 270.230034 | 5.209961% |
| new-v2 | 800.539591 | 1635.803345 | 620.371770 | 208.169770 | 6.376023% |

## 4. Component Errors vs Groundtruth (%)

| variant | pure_comp_err | pure_comm_execute_err | bubble_err | overlap_err |
|---------|--------------:|----------------------:|-----------:|------------:|
| ours | -2.172924% | -13.967593% | -13.231054% | 1.050000% |
| new-v1 | 98.240484% | -13.035736% | 75.029839% | 21.050000% |
| new-v2 | -6.286427% | -24.416452% | -19.028306% | -6.750000% |

## 5. Evidence (E2E Recomposition Check)

| variant | e2e_ms | recomposed_ms | diff_ms |
|---------|-------:|--------------:|--------:|
| groundtruth | 4007.870000 | 4007.870000 | 0.000000000000 |
| ours | 3587.990000 | 3587.990000 | 0.000000000000 |
| new-v1 | 5186.796152 | 5186.796152 | 0.000000000000 |
| new-v2 | 3264.884476 | 3264.884476 | 0.000000000000 |

All variants pass the decomposition conservation check.

### Verification Log
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/overlap_decomp_new_v1_new_v2_verification.log`
- Key output:
  - `PASS`
  - `max_abs_recompose_diff_ms=0.000000000000`
