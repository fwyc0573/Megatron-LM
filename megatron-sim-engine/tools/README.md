## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-27 | Added tools index during restructuring |
| 2026-03-15 | Added data-prep entrypoints for trace-shaped schedule and case-local kernel metrics workflow |

# Tools Layout

- `tools/analysis/`: analysis and diagnostic scripts
- `tools/data_prep/`: dataset/log preprocessing and reorganization tools
  - `tools/data_prep/schedule/build_trace_shaped_pp_schedule.py`: auto-generate replayable PP schedules from compressed top-level traces
  - `tools/data_prep/slowdown/prepare_case_kernel_metrics.py`: merge/filter case-local targeted `NCU` metrics and report missing kernels by representative rank
  - `tools/data_prep/slowdown/build_ddp_slowdown_assets.py`: build slowdown blueprints/features from trace + `nsys` + case-local kernel metrics
- `tools/benchmarking/`: benchmarking and metric-calculation scripts

Compatibility symlinks:
- `auto_handle_data -> tools/data_prep/auto_handle_data`
- `scaling_calculate -> tools/benchmarking/scaling_calculate`
