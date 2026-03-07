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
| 8 | 8 | 1 | 1 | 1 | 0.001381 | 0.000002 | 0.144351 |
| 16 | 8 | 2 | 1 | 2 | 0.002721 | 0.000712 | 0.146191 |
| 32 | 8 | 4 | 1 | 4 | 0.004285 | 0.001267 | 0.153547 |
| 64 | 8 | 8 | 1 | 8 | 0.007310 | 0.002396 | 0.155715 |
| 256 | 8 | 8 | 4 | 8 | 0.007287 | 0.003207 | 0.167944 |
| 512 | 8 | 8 | 8 | 8 | 0.007329 | 0.003247 | 0.199272 |
| 1024 | 8 | 16 | 8 | 16 | 0.011968 | 0.005833 | 0.284925 |
| 4096 | 8 | 16 | 32 | 16 | 0.013830 | 0.006704 | 2.111493 |
| 8192 | 8 | 16 | 64 | 16 | 0.012798 | 0.006114 | 6.474381 |
