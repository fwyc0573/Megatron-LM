## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-11 | Added final commit map and exclusion decisions |
| 2026-07-11 | Recorded initial branch inventory and classification constraints |

# Notes: Branch Change Organization

## Initial Inventory

- Branch: `overlap-tracing`
- Primary related tasks: `task_2026-03-08_ddp_overlap_tracing`, `task_2026-03-11_ddp_overlap_review`, and `task_2026-03-12_sim_engine_slowdown_support`
- Dirty submodules: `Echo-slowdown` and `megatron-sim-engine`
- Explicit exclusions: `docs/`, files larger than 50 MB, logs, caches, profiling databases, trace outputs, and generated e2e artifacts

## Commit Map

- `Echo-slowdown`: reusable collection, merge, training, and prediction source only.
- `megatron-sim-engine`: kernel-aware DDP slowdown runtime, asset/schedule tools, examples, and tests.
- Main repository CLI/examples: overlap auto-enable, strict boolean parsing, and wrapper scripts.
- Main repository trace identity: add `cmd_uid` to kernel-ground-truth NVTX labels.
- Main repository scaling DDP: select bucketing policy from the fake PP rank.
- Main repository integration: slowdown comparison/e2e source, submodule pointers, and related task records.

## Exclusions

- All `docs/` paths.
- All `tests/e2e/artifacts/` and `simulation_inputs/megatron_operation_log/` generated content.
- All files larger than 50 MB.
- Logs, profiler databases, NCU/NSYS reports, caches, model outputs, trained scalers, CSV/XLSX/PNG outputs, and machine-specific input configs.
- Unrelated CSV/TEX drafts and task cursor files.
