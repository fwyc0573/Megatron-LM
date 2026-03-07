## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Added WS256 variant v1/v2 simulation summary |
| 2026-03-04 | Added variant v3 summary and consolidated CSV reference |
| 2026-03-04 | Updated v2/v3 targets and reran simulations |

# Variant E2E Simulation Summary (WS256)

## Ground Truth Reference (Critical Rank)
- e2e_ms: 4007.870000
- comp_execute_ms: 965.860000
- comm_execute_ms: 2276.546954
- bubble_ms: 765.463046

## variant_v1
- e2e_ms: 5148.936152
- comp_execute_ms: 1828.566152
- comm_execute_ms: 1974.732074
- bubble_ms: 1345.637926
- e2e_error_pct: 28.470637820089
- comp_error_pct: 89.320000000000
- comm_execute_error_pct: -13.257573267958

## variant_v2
- e2e_ms: 3343.020000
- comp_execute_ms: 948.470000
- comm_execute_ms: 1740.420147
- bubble_ms: 654.129853
- e2e_error_pct: -16.588611906075
- comp_error_pct: -1.800467976725
- comm_execute_error_pct: -23.549999986420

## variant_v3
- e2e_ms: 3494.467674
- comp_execute_ms: 908.777674
- comm_execute_ms: 1940.756279
- bubble_ms: 644.933721
- e2e_error_pct: -12.809854760758
- comp_error_pct: -5.910000000000
- comm_execute_error_pct: -14.749999986121

## Consolidated CSV
- `task_memory/task_2026-03-04_reverse_groundtruth/logs/variant_groundtruth_ours_v1_v2_v3_e2e_comp_comm_bubble.csv`
