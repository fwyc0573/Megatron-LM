## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-05 | Initialize issue tracking for 6-case simulation task |
| 2026-03-05 | Added infeasible-bound analysis for 4 blocked cases |
| 2026-03-05 | Updated after expanded bounds rerun: 5/6 feasible, 1 blocked |
| 2026-03-05 | Resolved final blocked case with relaxed lower bounds |
| 2026-03-06 | Reframed issues under strict comp-scale and non-negative comm-factor constraints |
| 2026-03-06 | Added diversified-error and DP-overlap ordering validation status |

# Issues

## Active
1. Simulator emits large logs by default.
- Mitigation: redirect script stdout/stderr to `task_memory/task_2026-03-05_6case_e2e_sim_csv/logs/` for smoke/full runs.

2. Some DeepSeek cases may require aggressive communication correction factors.
- Mitigation: record parameter selection rationale in `notes.md` and diagnostics JSON.

3. `deepseek_v3_variant_case1` is infeasible under current strict constraints.
- Strict constraints:
  - `comp_scale=[0.965,0.988]` (fixed)
  - `overlap=[0.01,0.12]`
  - `intra/cross=[0.0,10.0]` (non-negative)
  - `abs_error_pct in [0.2, 9.0]` (non-zero policy)
- Current fail-fast evidence:
  - `Best abs_error_pct=18.309371`
  - `comp_scale=0.965000, intra=0.000000, cross=0.000000, overlap=0.010000`
  - log: `logs/rerun_deepseek_case1_nonzero.log`

## Resolved
- 5 feasible cases rerun and exported with non-zero errors (`~0.2%`) and non-negative comm factors:
  - `qwen3_case1`
  - `qwen3_case2`
  - `qwen3_case3`
  - `deepseek_v3_variant_case2`
  - `deepseek_v3_variant_case3`
- Diversified-error requirement satisfied on feasible set:
  - all 5 `error_pct` values are unique and within `[-8, 8]`.
- DP-overlap ordering requirement satisfied on feasible set:
  - `min(overlap_ms@dp8)=112.261360` > `max(overlap_ms@dp4)=108.176819` > `overlap_ms@dp2=61.106245`.
