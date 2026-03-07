## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Added constraints and implementation notes for wall-clock scaling task |

# Notes

## Config Source
- Primary source: `task_memory/task_2026-02-28_simai_wallclock_scaling/parallel_configs.md`
- Use section: `new plan：Formal Scaling Configurations (11 Points)`
- This run filters only required scales: `8,16,32,64,256,512,1024,4096,8192`

## Fixed Workload Parameters
- `TP=8`
- `micro_batch_size=1`
- `seq_length=2048`
- `GA=1` => `global_batch_size=dp`
- `exp_size=1` (dense model only)

## Engine Constraints
- `mg_test.py` currently fails for `PP=1`; schedule for scale 8 is manually generated in same textual schema.
- Simulator schedule file count must be `pp_size` or `world_size`.
- Dense rank optimization in engine selects representative ranks by `pp * tp * dp`.

## Backend Rule
- Must use `--cc-backend analytical` only.
- `analytical` backend maps to `src/core/comm_sim/nccl_comm.py`.
