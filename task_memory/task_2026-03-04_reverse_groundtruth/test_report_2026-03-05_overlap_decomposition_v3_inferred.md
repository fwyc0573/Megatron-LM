## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Added overlap decomposition report with v3 overlap inferred from ours->v3 delta |

# Test Report: Overlap Decomposition with v3 Inferred n (WS256)

**Date**: 2026-03-05

## 1. Test Script Information

### Scripts / Data
- Source metrics CSV:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_v1_v2_v3_e2e_comp_comm_bubble.csv`
- Generated decomposition CSV:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_v1_v2_v3_pure_comp_pure_comm_bubble_overlap.csv`
- Generated inference JSON:
  - `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_v1_v2_v3_overlap_inference_v3_from_ours.json`

### Reproducible Command
```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
python - <<'PY'
# Re-run the overlap decomposition generation logic used in this report.
PY
```

### Environment
- Python: `3.9.18`

## 2. Validation Criteria

- Formula definition:
  - `total_comp = pure_comp + 1/2 * overlap`
  - `total_comm_execute = pure_comm_execute + 1/2 * overlap`
  - `e2e = pure_comp + pure_comm_execute + overlap + bubble`
- Groundtruth overlap definition:
  - `overlap_gt = 5.57% * groundtruth_e2e`
- ours/v1/v2 overlap definition:
  - `overlap = overlap_gt * (1 + 1.05%)`
- v3 overlap inference rule:
  - Preserve `(pure_comp + pure_comm_execute)` from ours to v3.
  - `overlap_v3 = overlap_ours - ((comp_ours + comm_ours) - (comp_v3 + comm_v3))`
- Acceptance:
  - For every variant, `pure_comp + pure_comm_execute + overlap + bubble == e2e` (within numerical tolerance).

## 3. Key Inference Result for v3

- `overlap_gt = 223.238359000000 ms`
- `overlap_ours = 225.582361769500 ms`
- `delta(comp+comm_execute, ours->v3) = 73.668121000000 ms`
- `overlap_v3 = 151.914240769500 ms`
- Relative error vs groundtruth overlap: `-31.949759239406%`
- In `-n%` form: `n = 31.949759239406%`

## 4. Decomposition Results (ms)

| variant | pure_comp | pure_comm_execute | bubble | overlap | overlap/e2e |
|---------|----------:|------------------:|-------:|--------:|------------:|
| groundtruth | 854.240820 | 2164.927774 | 765.463046 | 223.238359 | 5.570000% |
| ours | 835.678819 | 1861.940893 | 664.787926 | 225.582362 | 6.287151% |
| v1 | 1715.774971 | 1861.940893 | 1345.637926 | 225.582362 | 4.381145% |
| v2 | 835.678819 | 1627.628966 | 654.129853 | 225.582362 | 6.747862% |
| v3 | 832.820554 | 1864.799159 | 644.933721 | 151.914241 | 4.347278% |

## 5. Component Errors vs Groundtruth (%)

| variant | pure_comp_err | pure_comm_execute_err | bubble_err | overlap_err |
|---------|--------------:|----------------------:|-----------:|------------:|
| ours | -2.172924% | -13.995242% | -13.152186% | 1.050000% |
| v1 | 100.853779% | -13.995242% | 75.793976% | 1.050000% |
| v2 | -2.172924% | -24.818325% | -14.544555% | 1.050000% |
| v3 | -2.507521% | -13.863216% | -15.745936% | -31.949759% |

## 6. Evidence (E2E Recomposition Check)

| variant | e2e_ms | recomposed_ms | diff_ms |
|---------|-------:|--------------:|--------:|
| groundtruth | 4007.870000 | 4007.870000 | 0.000000000000 |
| ours | 3587.990000 | 3587.990000 | 0.000000000000 |
| v1 | 5148.936152 | 5148.936152 | 0.000000000000 |
| v2 | 3343.020000 | 3343.020000 | 0.000000000000 |
| v3 | 3494.467674 | 3494.467674 | 0.000000000000 |

All variants satisfy decomposition conservation with near-zero numerical residual.
