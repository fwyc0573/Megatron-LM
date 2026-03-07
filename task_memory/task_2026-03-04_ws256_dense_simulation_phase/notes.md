## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-04 | Added run-time notes and fixed decisions for WS256 dense simulation |

# Notes: WS256 Dense (H800)

## Fixed Decisions
- Experiment directory name: `h800_256gpus_gpt175b_tp8_pp16_dp2`.
- Data organization strategy: `copy` (no move/delete).
- Measured comparison strategy: `N/A (skipped by task decision)`.

## Parameters (from profiling script)
- `world_size=256`
- `pp=16`
- `tp=8`
- `dp=2`
- `exp=1`
- `num_layers=96`
- `hidden_size=12288`
- `seq_length=2048`
- `micro_batch_size=1`
- `global_batch_size=128`
- `train_iters=10`
- `trace_start=10`
- `fp16=true`
- `model_size=175`

## Runtime Notes
- Baseline `collective-sim` run is very sensitive to verbose debug output and repeated htsim invocations.
- For stable completion in this environment, the executed simulation used:
  - `--cc-backend collective-sim`
  - `--cc-backend-options-json '{"collective-sim":{"placement_mode":"group_size"}}'`
- This keeps backend type as `collective-sim` while reducing scenario dimensionality for better runtime practicality.
