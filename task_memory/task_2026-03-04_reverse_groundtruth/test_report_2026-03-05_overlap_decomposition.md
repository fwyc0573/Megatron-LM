## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Initial overlap decomposition report |

# Overlap Decomposition Report (WS256 Dense H800)

**Date**: 2026-03-05

## 1. Decomposition Model

The existing 3-component split:
- `comp_execute = pure_comp + 0.5 * overlap`
- `comm_execute = pure_comm + 0.5 * overlap`
- `e2e = comp_execute + comm_execute + bubble`

Is refined to a 4-component split:
- `e2e = pure_comp + pure_comm + overlap + bubble`

Where `overlap` represents the time during which computation and communication execute concurrently.

## 2. Overlap Inputs

- **Groundtruth**: overlap = 5.57% of e2e = 223.238359 ms
- **Ours, v1, v2**: overlap error vs groundtruth = +1.05% → overlap = 225.582362 ms
- **v3**: overlap error vs groundtruth = -31.949759% (solved) → overlap = 151.914241 ms

### v3 Overlap Derivation

v3 shares the same `pure_comp` and `pure_comm` as ours. Only `overlap` changes.

Using the sum constraint:
```
(comp_execute + comm_execute) = pure_comp + pure_comm + overlap
(comp+comm)_ours = 948.470000 + 1974.732074 = 2923.202074
(comp+comm)_v3   = 908.777674 + 1940.756279 = 2849.533953
delta            = -73.668121
overlap_v3       = overlap_ours + delta = 225.582362 + (-73.668121) = 151.914241 ms
```

**v3 overlap error n = 31.949759%**

## 3. 4-Component Decomposition (Critical Rank, ms)

| variant | e2e_ms | pure_comp_ms | pure_comm_ms | overlap_ms | bubble_ms |
|---------|--------|-------------|-------------|-----------|----------|
| groundtruth | 4007.870000 | 854.240821 | 2164.927774 | 223.238359 | 765.463046 |
| ours | 3587.990000 | 835.678819 | 1861.940893 | 225.582362 | 664.787926 |
| v1 | 5148.936152 | 1715.774971 | 1861.940893 | 225.582362 | 1345.637926 |
| v2 | 3343.020000 | 835.678819 | 1627.628966 | 225.582362 | 654.129853 |
| v3 | 3494.467674 | 832.820554 | 1864.799159 | 151.914241 | 644.933721 |

## 4. Per-Component Error vs Groundtruth (%)

| variant | e2e_err% | pure_comp_err% | pure_comm_err% | overlap_err% | bubble_err% |
|---------|---------|---------------|---------------|------------|------------|
| groundtruth | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ours | -10.476388 | -2.172924 | -13.995242 | 1.050000 | -13.152186 |
| v1 | 28.470638 | 100.853779 | -13.995242 | 1.050000 | 75.793976 |
| v2 | -16.588612 | -2.172924 | -24.818325 | 1.050000 | -14.544555 |
| v3 | -12.809855 | -2.507521 | -13.863216 | -31.949759 | -15.745936 |

## 5. Component as Percentage of E2E

| variant | pure_comp% | pure_comm% | overlap% | bubble% | sum% |
|---------|-----------|-----------|---------|--------|------|
| groundtruth | 21.3141 | 54.0169 | 5.5700 | 19.0990 | 100.0000 |
| ours | 23.2910 | 51.8937 | 6.2872 | 18.5281 | 100.0000 |
| v1 | 33.3229 | 36.1617 | 4.3811 | 26.1343 | 100.0000 |
| v2 | 24.9977 | 48.6874 | 6.7479 | 19.5670 | 100.0000 |
| v3 | 23.8325 | 53.3643 | 4.3473 | 18.4559 | 100.0000 |

## 6. Relationship to Original 3-Component Split

| variant | comp_execute_ms | = pure_comp + 0.5*overlap | comm_execute_ms | = pure_comm + 0.5*overlap |
|---------|----------------|--------------------------|----------------|--------------------------|
| groundtruth | 965.860000 | 965.860000 | 2276.546954 | 2276.546954 |
| ours | 948.470000 | 948.470000 | 1974.732074 | 1974.732074 |
| v1 | 1828.566152 | 1828.566152 | 1974.732074 | 1974.732074 |
| v2 | 948.470000 | 948.470000 | 1740.420147 | 1740.420147 |
| v3 | 908.777674 | 908.777674 | 1940.756279 | 1940.756279 |

## 7. Artifacts

- Computation script: `tests/performance/compute_overlap_decomposition.py`
- 4-component CSV: `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_overlap_decomposition.csv`
- JSON summary: `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_overlap_decomposition.json`
- This report: `task_memory/task_2026-03-04_reverse_groundtruth/test_report_2026-03-05_overlap_decomposition.md`

## 8. Reproducible Command

```bash
cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
python tests/performance/compute_overlap_decomposition.py \
  --gt-overlap-pct 5.57 \
  --ours-overlap-error-pct 1.05
```
