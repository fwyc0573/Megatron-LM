## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initial wall-clock scaling measurements for megatron-sim-engine with analytical(comm_sim) backend |

# Megatron-Sim-Engine Wall-clock Scaling Results

- Reference config source: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-02-28_simai_wallclock_scaling/parallel_configs.md`
- Parallel policy: new-plan table from reference task
- Model scope: Dense only (`exp_size=1`)
- Rank construction: per PP stage representative ranks (`tp=0, dp=0`)
- CC backend: `analytical` (calls `comm_sim` directly)

| world_size | tp | pp | dp | selected_ranks_count | sim_load_time_s | sim_execution_time_s | outer_wallclock_s |
|------------|----|----|----|----------------------|-----------------|----------------------|-------------------|
| 8 | 8 | 1 | 1 | 1 | 0.001418 | 0.000002 | 0.149821 |
| 16 | 8 | 2 | 1 | 2 | 0.002669 | 0.000698 | 0.146023 |
| 32 | 8 | 4 | 1 | 4 | 0.004330 | 0.001264 | 0.156379 |
