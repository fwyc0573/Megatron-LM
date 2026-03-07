## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Initial wall-clock scaling measurements for megatron-sim-engine with analytical(comm_sim) backend |

# Megatron-Sim-Engine Wall-clock Scaling Results

- Reference config source: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/task_memory/task_2026-02-28_simai_wallclock_scaling/parallel_configs.md`
- Parallel policy: new-plan table from reference task
- Model scope: Dense only (`exp_size=1`)
- Rank construction: per PP stage representative ranks (`tp=0, dp=0`)
- CC backend: `collective-sim`

| world_size | tp | pp | dp | selected_ranks_count | sim_load_time_s | sim_execution_time_s | outer_wallclock_s |
|------------|----|----|----|----------------------|-----------------|----------------------|-------------------|
| 8 | 8 | 1 | 1 | 1 | 0.001444 | 0.000002 | 0.153552 |
| 16 | 8 | 2 | 1 | 2 | 0.002462 | 0.000635 | 0.146319 |
| 32 | 8 | 4 | 1 | 4 | 0.004330 | 0.001462 | 0.164680 |
| 64 | 8 | 8 | 1 | 8 | 0.006562 | 0.002479 | 0.150854 |
| 256 | 8 | 8 | 4 | 8 | 0.007341 | 0.003170 | 0.179395 |
| 512 | 8 | 8 | 8 | 8 | 0.007109 | 0.002756 | 0.198486 |
| 1024 | 8 | 16 | 8 | 16 | 0.013535 | 0.006455 | 0.353226 |
| 4096 | 8 | 16 | 32 | 16 | 0.023341 | 0.010385 | 2.115906 |
| 8192 | 8 | 16 | 64 | 16 | 0.017656 | 10370.790508 | 10378.513177 |
