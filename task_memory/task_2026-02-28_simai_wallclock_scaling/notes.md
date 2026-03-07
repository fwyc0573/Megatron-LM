## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Initialized notes and locked decisions for SimAI wall-clock scaling task |
| 2026-02-28 | Added implementation notes for measurement pipeline, strict schema, and tests |

# Notes: SimAI Wall-clock Scaling

## Locked Decisions
- Measurement target is simulator wall-clock, not simulated E2E runtime.
- Simulation backend is NS3 (`SimAI_simulator`) only.
- No Analytical control-group measurement in this task.
- PP range is `[1, 12]`.
- For this scale set (`total_gpus` as powers of 2), feasible PP values are:
  - 8 -> [1]
  - 16 -> [1, 2]
  - 32 -> [1, 2, 4]
  - >=64 -> [1, 2, 4, 8]
- Small-scale sensitivity uses 8/16/32 only.
- Similarity threshold is 20% (`max/min <= 1.20`).
- Formal stage chooses one PP per scale:
  - similar: compromise PP = floor(max_feasible_pp / 2), lower-bounded at 1
  - severe: max feasible PP
- Model/workload constants:
  - model size 22B
  - `micro_batch=1`
  - `seq_length=2048`
  - `global_batch = dp * micro_batch` (GA=1)
- Run count per configuration is 1.
- No timeout in official measurement flow.

## Implementation Notes
- Measurement script: `SimAI/tests/performance/simai_wallclock_scaling/measure_wallclock.py`
  - strict fail-fast validation for PP/DP integrality and CSV schema.
  - uses `python3 -m workload_generator.SimAI_training_workload_generator` directly.
  - does not rely on `aicb/scripts/megatron_workload_with_aiob.sh`.
  - isolates runtime artifacts under `/tmp/simai_wallclock_scaling`.
  - only wraps `SimAI_simulator` process time for wall-clock measurement.
- Plot script: `SimAI/tests/performance/simai_wallclock_scaling/plot_wallclock.py`
  - strict CSV schema check.
  - scaling curve annotates `(PP,DP)`.
  - sensitivity curve groups by small scale and annotates `DP`.
- Unit tests: `SimAI/tests/unit/test_simai_wallclock_scaling.py`
  - coverage includes policy selection, PP feasibility, fail-fast paths, schema checks, and plotting output generation.

## Environment/Code Facts
- NS3 simulation main loop uses `Simulator::Run()`.
- Collective implementation defaults to `NcclFlowModel`.
- Flow models are generated dynamically for communication events.
- AICB helper script uses `python` and is environment fragile here.

## Risks
- Long runtime is expected for large scales and seq_length=2048.
- Full 11-point execution is intentionally left as a dedicated long run, not part of lightweight verification commands.
