## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Initialized issue tracking for SimAI wall-clock scaling task |
| 2026-02-28 | Updated with implementation-stage issues and resolutions |
| 2026-02-28 | Added queue-launch conflict guard for existing measurement process |
| 2026-03-01 | Added blocker for 8192-GPU topology generation failure |
| 2026-03-01 | Added resolution for AlibabaHPN topology filename mismatch |
| 2026-03-02 | Added detached execution note for long 8192 run under CLI harness |
| 2026-03-03 | Marked 8192 blocker resolved and retained cross-template comparability caveat |
| 2026-03-03 | Added isolated rerun tracking for new PP=16 plan (>=1024) |
| 2026-03-03 | Added fix note for new-plan runner print typo and watcher safeguard |
| 2026-03-04 | Marked isolated new-plan rerun execution as completed |

# Issues and Resolutions

## Resolved
1. **Dirty workspace before branch work**
   - Impact: Existing untracked artifacts in `SimAI` and `SimAI/aicb` could pollute this task.
   - Resolution: Stashed with `git stash -u` in both repos before creating `baseline` branch.
2. **AICB wrapper script not portable in this environment**
   - Impact: `aicb/scripts/megatron_workload_with_aiob.sh` depends on `python`, but runtime only guarantees `python3`.
   - Resolution: Measurement script directly invokes `python3 -m workload_generator.SimAI_training_workload_generator`.
3. **Potential repo pollution by runtime-generated files**
   - Impact: Simulation/topology/workload runtime artifacts can create unrelated untracked files.
   - Resolution: Runtime artifacts are isolated under `/tmp/simai_wallclock_scaling` in implementation.
4. **Queue script path resolution and duplicate launch guard**
   - Impact: Initial queue script root resolution could point outside repo; duplicate measurement launches may race-write CSV.
   - Resolution: Fixed `SIMAI_ROOT` path derivation and added process guard using `pgrep -f measure_wallclock.py` with explicit skip status.
5. **Topology filename assumption too strict for non-Spectrum templates**
   - Impact: Even when `AlibabaHPN` generator succeeded, measurement script failed with `Expected topology file not found...` because it assumed a single canonical filename format.
   - Resolution: Updated topology resolver to accept unique wildcard matches like `AlibabaHPN_8192g_8gps_DualToR_SinglePlane_400Gbps_H800`, while fail-fast on ambiguous matches.
6. **Full-scale runtime execution and plotting completion**
   - Impact: Long run and plotting were pending in earlier checkpoints.
   - Resolution: 8192 point completed (`46642.760256s`) and plot generation completed (`wallclock_scaling.png`, `wallclock_sensitivity.png`).
7. **8192 topology generation blocker in Spectrum-X**
   - Impact: Formal 11th point could not finish under default Spectrum-X topology generation.
   - Resolution: User approved topology strategy override; 8192 point was executed with `AlibabaHPN` and completed successfully.
8. **Do not overwrite prior completed formal CSV**
   - Impact: New plan requires rerunning only >=1024 with PP=16, while preserving prior completed run records.
   - Resolution: New run is isolated under `results/new_plan_pp16_ge1024_2026-03-03/` with separate CSV/PNG/log files.
9. **New-plan runner print typo (`run_summary.json`)**
   - Impact: First-run worker could exit before plot step when reaching malformed print expression.
   - Resolution: Patched runner and started detached watcher that re-invokes runner with `resume=True` once initial worker exits, ensuring plotting is eventually produced without redoing completed rows.
10. **Isolated new-plan rerun completion**
   - Impact: New configuration (`PP=16` for `>=1024`) required additional measurements without touching baseline records.
   - Resolution: Completed isolated run with separate CSV/PNG/log outputs under `results/new_plan_pp16_ge1024_2026-03-03/`.

## Open
1. **Cross-template comparability at 8192**
   - Context: 8..4096 points used `Spectrum-X`; 8192 used approved `AlibabaHPN` override.
   - Impact: The `4096 -> 8192` delta includes topology-template change in addition to scale increase.
   - Next Action: If strict same-template comparability is required, a new 8..8192 run with a single topology family should be scheduled.
2. **DP value mismatch in new plan table at 8192**
   - Context: `parallel_configs.md` new plan row lists `8192, TP=8, PP=16, DP=32`.
   - Impact: This conflicts with enforced formula `DP = total_gpus / (TP * PP)`, which gives `8192/(8*16)=64`.
   - Next Action: Current rerun follows formula-based DP (`64`) used by measurement code; confirm whether table needs correction.
