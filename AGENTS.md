## Modification History

| Date       | Summary of Changes                                      |
|------------|----------------------------------------------------------|
| 2026-02-26 | Initial version: document modes, traces, and active tasks |

# AGENTS.md (for AI Coding Agents)

This repository is a **modified / instrumented fork of Megatron-LM**. It adds two execution modes
and a trace format that supports **trace-driven scaling simulation**:

- **Realistic Mode**: run native distributed Megatron-LM on real multi-GPU hardware and record
  wall-clock timings + metadata (ground truth).
- **Scaling Mode**: run on a **single physical GPU** while simulating a large parallel topology by
  executing one *fake rank* at a time; communication is **recorded and skipped**, while compute is
  **actually executed** and timed.

This file is intended to help AI agents quickly locate the relevant code paths and complete the
active tasks described below.

---

# 1) Project Overview

This fork implements a workload tracing layer (via `megatron/profiler/cmd.py`) and a scaling-mode
execution path (via `megatron/training/training.py`) so that:

1. You can collect per-rank compute timings on one GPU (Scaling Mode), without requiring a full
   cluster for every configuration you want to study.
2. You can collect ground-truth distributed timings on a real multi-GPU run (Realistic Mode).
3. You can compare the two, and feed Scaling Mode outputs into an end-to-end timeline simulation
   (compute reused as-measured; communication reconstructed from metadata + trigger points).

This repo already contains a **stage-1** port for:
- Qwen3-30B-A3B (MoE, stage-1 subset)
- DeepSeek-V3-Proxy (MoE, **MHA simplification**, stage-1 subset)

See `README.md` for the stage-1 status table and run examples.

---

# 2) Codebase Structure (Key Paths)

## Entry points (training scripts)

- `pretrain_llama.py`
  - Despite the name, this is the main **instrumented decoder-only** pretrain entry in this fork.
  - Implements `forward_step()` wrapped by `CMD` tracing and injects scaling-mode config fields.

## Mode selection / training loop

- `megatron/training/arguments.py`
  - Defines the mode flags:
    - `--is-scaling-mode`
    - `--fake-world-size`, `--fake-pp`, `--fake-dp`, `--fake-tp`, `--fake-exp`, `--fake-num-experts`
    - `--fake-current-rank-id`
  - Defines tracing flags:
    - `--do-trace`, `--trace-start`
    - `--trace-subop-sync-mode` (`global` or `event`)
    - `--trace-kernel-ground-truth` and `--trace-kernel-ground-truth-prefix`

- `megatron/training/training.py`
  - **Scaling Mode driver**: builds a fake-rank topology, runs warmup, then profiles:
    - `forward_step`
    - `backward_step`
    - `optimizer_step`
  - **Realistic Mode tracing**: runs normal Megatron training loop and writes trace logs to disk.

## Tracing implementation and artifacts

- `megatron/profiler/cmd.py`
  - `CMD` context manager measures op durations and records:
    - `rank_id`, `stage_id`, `mg_state`
    - `duration` (ms) and a `timestamp` (ms)
    - `sub_operations` (decorator-recorded sub-ops with metadata)
  - `CMD.get_trace_decorator(...)` wraps sub-ops (especially comm ops) and records trigger metadata.
  - `write_list_to_file(...)` writes trace files to:
    - `realistic_trace/<run_config>/...` (Realistic Mode; `file_path=None`)
    - `profiler_log/<run_config>/...` (Scaling Mode; `file_path="profiler_log"`)

## Scaling Mode fake topology and replay

- `megatron/profiler/parallel_group_manager.py`, `megatron/profiler/rank_manager.py`
  - Build a fake PP/TP/DP/EP topology and map each fake world-rank to:
    - `pp_rank`, `tp_rank`, `dp_rank`, `exp_rank`
    - neighbor ranks for pipeline (`pp_prev_rank`, `pp_next_rank`)

- `megatron/profiler/utils.py`
  - Implements scaling-mode helpers:
    - `sim_forward_step(...)`
    - `sim_backward_step(...)`
  - Uses **activation/grad replay cache** on disk to emulate pipeline dependencies.

## Scaling-mode-aware model/parallel code (compute partitioning)

Scaling Mode relies heavily on config fields (`is_scaling_mode`, `fake_tp`, `pp_rank`, etc.) so that
compute partitioning matches the target topology even though the runtime world-size is 1.

Key files:
- `megatron/core/transformer/transformer_config.py`
  - Adds `is_scaling_mode` and `fake_tp`; uses `fake_tp` in divisibility checks.
- `megatron/core/model_parallel_config.py`
  - Uses `fake_tp` to validate sequence parallel constraints in scaling mode.
- `megatron/core/models/gpt/gpt_layer_specs.py`
  - Uses `config.pp_rank` in scaling mode to build the correct local layer slice.
- `megatron/core/models/common/language_module/language_module.py`
  - Uses scaling-mode config fields for embedding-group checks (cannot rely on real `parallel_state`).
- `megatron/core/models/common/embeddings/rotary_pos_embedding.py`
  - Special-case: RoPE seq-len logic avoids over-expanding positions in scaling mode.
- `megatron/core/transformer/custom_layers/transformer_engine.py`
  - Forces Transformer-Engine layers (Linear, LayerNormLinear, DotProductAttention) to use
    `tp_size=config.tensor_model_parallel_size` even when `tp_group=None` in scaling mode.

## Communication tracing / interception

Communication ops are recorded as CMD sub-operations (with `comm_func=...`, tensor shape/dtype, etc).
Scaling Mode either bypasses the real comm call or simulates only shape/buffer creation.

Key files:
- `megatron/profiler/comm_utils/interception_comm.py`
  - `allreduce_wrapper`, `broadcast_wrapper`, `reduce_wrapper`
  - In scaling mode: returns early (no real `torch.distributed` collective executed).
- `megatron/core/tensor_parallel/mappings.py`
  - Decorated allgather / reduce_scatter / all-to-all paths.
  - `_profiled_all_to_all_single(...)` has an explicit scaling-mode branch that avoids real comm and
    only returns correctly shaped tensors for downstream compute.
- `megatron/core/tensor_parallel/layers.py`
  - Uses `allreduce_wrapper` and checks `args.fake_tp` in scaling mode.

## Realistic Mode instrumentation in pipeline and grad sync

- `megatron/core/pipeline_parallel/schedules.py`
  - Wraps pipeline send/recv (`recv_forward`, `send_forward`, etc.) and backward steps with `CMD`.
- `megatron/core/distributed/finalize_model_grads.py`
  - Wraps DP grad sync (`dp_allreduce`) and embedding grad sync (`ep_allreduce`) with `CMD`.

## Accuracy comparison scripts (paper-facing metrics)

- `tests/performance/compare_qwen_trace_comp.py`
  - Compares Scaling vs Realistic traces for `forward_step`, `backward_step`, `optimizer_step`.
  - Supports robust aggregation (median / trimmed mean) and repeated-run summaries (JSONL).
  - Computes a “comp” estimate by optionally subtracting comm sub-op durations from distributed.

- `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`
  - Reads Nsight Systems `.sqlite` and CMD NVTX labels (from `--trace-kernel-ground-truth`).
  - Outputs JSON rows with per-window compute-only kernel time.

- `tests/performance/compare_qwen_nsys_compute_only.py`
  - Compares compute-only kernel-time JSON between Scaling and Realistic.

---

# 3) Execution Modes

## 3.1 Scaling Mode (single GPU, sequential fake ranks)

### Intent

Scaling Mode simulates large PP/TP/DP/EP configurations on one GPU:

- **Compute** is executed normally (forward/backward/optimizer) on the single device.
- **Communication** is **recorded but not executed**:
  - Collectives are intercepted (or bypassed due to real world-size=1).
  - All-to-all is shape-simulated where needed so downstream compute still runs.
  - Communication metadata is recorded as `CMD` sub-ops (tensor shape/dtype, group kind, etc).

### How it is invoked

Scaling Mode is enabled by passing:

- `--is-scaling-mode`
- a fake topology (`--fake-world-size`, `--fake-pp`, `--fake-dp`, `--fake-tp`, `--fake-exp`)
- the current simulated rank (`--fake-current-rank-id`)

In practice, **you run one fake rank per process** and loop over rank IDs (0..fake_world_size-1).
The stage-1 scripts do this for you:

- `examples/pretrain_qwen3_30b_a3b_moe.sh` (Qwen3 stage-1)
- `examples/pretrain_deepseek_v3_proxy_moe.sh` (DeepSeek stage-1 proxy)

### Output artifacts

- Scaling traces (per fake rank): `profiler_log/<run_config>/*_rank<id>_<YYYYMMDDHHMMSS>.txt`
- Replay cache (pipeline activation/grad handoff): `profiler_log/scaling_replay_cache/<cache_tag>/*.pt`
- Optional memory traces (JSON): `memory_traces_scaling/*.json` (enabled via `--trace-memory`)

### Notes / constraints

- Current scaling-mode simulation helpers (`sim_forward_step`) assume the model is a Megatron-Core
  `GPTModel` (decoder-only). Adding new model families may require expanding the sim helpers.
- MoE in scaling mode expects **GroupedMLP** (`--moe-grouped-gemm`), because `SequentialMLP` is
  explicitly rejected in scaling mode (`NotImplementedError`).

## 3.2 Realistic Mode (native distributed ground truth)

### Intent

Realistic Mode runs the unmodified distributed training algorithm (DP/TP/PP/EP) on actual multiple
GPUs, while the fork’s instrumentation records:

- wall-clock op durations for:
  - pipeline recv/send
  - forward/backward steps
  - dp/embedding grad sync
  - optimizer step
- metadata for comm ops via sub-op decorators

### How it is invoked

Use standard Megatron-LM `torchrun` on real GPUs (for the stage-1 scripts):

```bash
MODE=distributed bash examples/pretrain_qwen3_30b_a3b_moe.sh
MODE=distributed bash examples/pretrain_deepseek_v3_proxy_moe.sh
```

### Output artifacts

- Distributed traces (per real rank): `realistic_trace/<run_config>/*_rank<id>_<YYYYMMDDHHMMSS>.txt`
- Optional memory traces (JSON): `memory_traces/*.json` (enabled via `--trace-memory`)

---

# 4) Trace Format and Output Layout

Each trace file is line-based text. Each line corresponds to a top-level operation (a `CMD`) and
includes a `sub_operations=[...]` list for decorated sub-ops.

Example (abridged):

```text
rank:0:forward_step(stage_id=0,batch_id=0,mg_state=warmup,duration=146.54,...,timestamp=...,sub_operations=[...])
```

Important fields:

- `rank:<id>`: fake rank (Scaling Mode) or real distributed rank (Realistic Mode)
- `op`: e.g., `get_batch`, `forward_step`, `backward_step`, `dp_allreduce`, `optimizer_step`
- `mg_state`: pipeline scheduling state, e.g. `warmup`, `steady`, `cooldown`, `finalize`
- `duration`: milliseconds measured by CUDA Events (or set to 0 in metadata-only cases)
- `timestamp`: milliseconds from `time.perf_counter()`; used for ordering/trigger inference
- `sub_operations`: strings containing `comm_func=...`, `group=...`, and tensor shape/dtype metadata

Directory layout differences:

- Scaling Mode: `profiler_log/pp{pp}_tp{tp}_ep{ep}_expn{num_experts}_dp{dp}_nl{nl}_hs{hs}_sl{sl}/...`
- Realistic Mode: `realistic_trace/pp{pp}_tp{tp}_exp{ep}_expn{num_experts}_dp{dp}_nl{nl}_hs{hs}_sl{sl}/...`

Note the historical naming mismatch: Scaling uses `ep{...}` while Realistic uses `exp{...}`.

---

# 5) Simulation Pipeline (Scaling traces → end-to-end timeline simulation)

At a high level:

1. **Collect Scaling Mode traces**:
   - Run 1 fake rank per process (single GPU), sequentially for all fake ranks.
   - Keep compute real; record comm metadata without executing multi-GPU comm.
2. **Collect Realistic Mode traces**:
   - Run the same workload on real multi-GPU and record ground truth.
3. **Compute-only accuracy evaluation**:
   - Use `tests/performance/compare_qwen_trace_comp.py` (trace-based) and/or
     `tests/performance/compare_qwen_nsys_compute_only.py` (kernel-ground-truth based).
4. **Timeline simulation**:
   - Scaling Mode provides per-rank compute durations (reuse as measured).
   - Communication events are reconstructed from metadata (tensor shape/dtype, group kind) and the
     inferred trigger points (where the ops were recorded/submitted).
   - The end-to-end simulator composes global timelines using dependencies (e.g., PP 1F1B rules).

For background on the intended architecture, also read `docs/echo_paper/echo_paper.md`.

---

# 6) Currently Supported Models (Scaling Mode)

Scaling Mode currently supports **Megatron-Core decoder-only GPTModel-based architectures**, with:

- **Dense GPT/LLaMA/Qwen-style configs** (RoPE, RMSNorm, SwiGLU, GQA, etc.)
- **MoE variants** based on Megatron-Core `MoELayer` and token dispatchers

Stage-1 scripts that are known to work in both modes:

- `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - Qwen3-30B-A3B (stage-1 subset), MoE-enabled.
- `examples/pretrain_deepseek_v3_proxy_moe.sh`
  - DeepSeek-V3-Proxy (stage-1 subset), MoE-enabled, **MHA simplification** (no MLA yet).

Known limitations (important when extending model support):

- Scaling-mode forward helper asserts `GPTModel` (`megatron/profiler/utils.py`).
- MoE scaling-mode requires `--moe-grouped-gemm` (no `SequentialMLP` in scaling mode).
- Stage-1 DeepSeek proxy does not implement MLA or DeepSeek-specific router semantics.

---

# 7) Active Tasks for Agents

## Task 1: Add Scaling Mode Support for New Models

Add Scaling Mode simulation support for:

1. **Qwen3.5-MoE**
2. **DeepSeek-V3** (full, not proxy)

Requirements:

- Must run in Scaling Mode on the current **8-GPU testbed** by using **one physical GPU** to
  simulate **8 fake ranks** (`fake_world_size=8`).
- Compute timing accuracy must be within **5% relative error** vs Realistic Mode ground truth for
  the **compute (comp)** portion, for:
  - `forward_step`
  - `backward_step`
  - `optimizer_step`

Suggested starting points (read first):

- Mode + trace driver:
  - `megatron/training/training.py`
  - `megatron/training/arguments.py`
  - `megatron/profiler/cmd.py`
- Scaling-mode model partitioning:
  - `megatron/core/transformer/transformer_config.py`
  - `megatron/core/models/gpt/gpt_layer_specs.py`
  - `megatron/core/transformer/custom_layers/transformer_engine.py`
- MoE routing/dispatching in scaling mode:
  - `megatron/profiler/moe/sim_pre_moe.py`
  - `megatron/core/transformer/moe/*`

Acceptance checklist:

- New model has an example script under `examples/` that can run:
  - `MODE=distributed` (Realistic Mode)
  - `MODE=scaling` (Scaling Mode sequential ranks 0..7)
- Traces are produced under:
  - `realistic_trace/<run_config>/...`
  - `profiler_log/<run_config>/...`
- `tests/performance/compare_qwen_trace_comp.py` (or kernel-ground-truth scripts) reports comp
  relative error <= 5% for the target ops across all ranks (or a documented, defensible subset).

## Task 2: Select Optimal Statistical Aggregation Method

Goal:

- Select and justify the statistical aggregation method(s) used to report scaling-vs-realistic
  simulation accuracy in a research-paper setting.

Constraints:

- Must be academically defensible (robust, interpretable).
- Should be numerically favorable while still faithful (avoid misleading cherry-picking).

Where to implement/validate:

- Use and (if needed) extend:
  - `tests/performance/compare_qwen_trace_comp.py`
  - `tests/performance/compare_qwen_nsys_compute_only.py`

Recommended workflow:

1. Define the per-sample error unit (per-rank, per-op, per-state bucket).
2. Compare candidate aggregations:
   - median across ranks (robust)
   - trimmed mean across ranks (robust + smoother)
   - p75/p90 summaries (tail risk)
3. Report both a primary metric and a secondary distribution view (e.g., median + p75).

Deliverable:

- A short writeup under `task_memory/` documenting the chosen metric(s) and rationale, plus the
  command(s) used to produce the reported numbers.

## Task 3: Identify Optimal Workload and Configuration Settings

Goal:

- Recommend workload parameters and runtime settings that yield the best simulation accuracy
  metrics suitable for paper-level reporting, while staying realistic and representative.

Parameters to explore (examples):

- batch sizing: `MICRO_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`
- sequence length: `SEQ_LEN`
- parallelism degrees: `TP`, `PP`, `EP` (and derived `DP`)
- implementation knobs: `TRANSFORMER_IMPL`, `TRACE_SUBOP_SYNC_MODE`
- model profiles: `MODEL_PROFILE=smoke|full|proxy`

Constraints:

- Avoid misleading cherry-picking; settings should be representative.
- Document why the chosen configuration is reasonable.

Deliverable:

- A recommended “paper config set” with:
  - at least one “smoke” config (fast) and one “full/proxy” config (realistic)
  - trace + comparison commands and their outputs

---

# 8) How to Run & Validate

## 8.1 Generate traces (stage-1 Qwen3 / DeepSeek proxy)

These scripts already support both modes:

```bash
# Realistic Mode (multi-GPU)
MODE=distributed bash examples/pretrain_qwen3_30b_a3b_moe.sh
MODE=distributed bash examples/pretrain_deepseek_v3_proxy_moe.sh

# Scaling Mode (single GPU, sequential fake ranks 0..7)
MODE=scaling bash examples/pretrain_qwen3_30b_a3b_moe.sh
MODE=scaling bash examples/pretrain_deepseek_v3_proxy_moe.sh
```

Tips:

- Use `MODEL_PROFILE=smoke` for quick validation; use `MODEL_PROFILE=full` (Qwen) or
  `MODEL_PROFILE=proxy` (DeepSeek) for larger stage-1 runs.
- Set `TRACE_START` and `TRAIN_ITERS` consistently across Scaling and Realistic runs so the traced
  iteration(s) are comparable.

## 8.2 Compare Scaling vs Realistic (trace-based comp metric)

Pick the matching `run_config` directories:

- Realistic: `realistic_trace/<run_config>/`
- Scaling: `profiler_log/<run_config>/`

Then run:

```bash
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/<run_config> \
  --scaling-dir profiler_log/<run_config> \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp <YYYYMMDDHHMMSS> \
  --report-path task_memory/<task_dir>/logs/compare_trace.log
```

Notes:

- `--pair-timestamp` caps pairing to “latest trace file per rank with timestamp <= cap”. Use it to
  align a distributed run with the closest scaling run.
- The script can optionally subtract distributed comm sub-op duration to estimate “distributed comp”.

## 8.3 Kernel ground truth via Nsight Systems (optional, higher fidelity)

1. Run training with CMD NVTX kernel-ground-truth labels enabled:
   - `--trace-kernel-ground-truth`
   - `--trace-kernel-ground-truth-prefix cmd_trace` (default)
2. Capture an Nsight Systems `.sqlite` trace.
3. Convert sqlite → JSON:

```bash
python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite <nsys_trace.sqlite> \
  --label-prefix cmd_trace \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/<task_dir>/logs/nsys_kernel_breakdown.json \
  --report-path task_memory/<task_dir>/logs/nsys_kernel_breakdown.md
```

4. Compare distributed vs scaling compute-only:

```bash
python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/<task_dir>/logs/nsys_dist.json \
  --scaling-json task_memory/<task_dir>/logs/nsys_scale.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5
```

---

# 9) Agent Guidelines (How to Work on These Tasks)

1. Start from the stage-1 working baseline:
   - Re-run `examples/pretrain_qwen3_30b_a3b_moe.sh` and/or `examples/pretrain_deepseek_v3_proxy_moe.sh`
     in both modes and confirm traces are produced.
2. Treat Scaling Mode as “compute must be real”:
   - If you change partitioning logic, ensure scaling-mode compute still matches the fake topology
     (e.g., attention head partitioning should use `fake_tp`).
3. Treat comm in Scaling Mode as “metadata + shape, no execution”:
   - If you add new collectives, ensure they are either intercepted or safely bypassed while still
     recording metadata needed by the simulator.
4. Keep results reproducible:
   - Store experiment logs and test reports under `task_memory/<task_dir>/`.
   - Prefer adding/using scripts under `tests/performance/` for comparisons (not ad-hoc root scripts).
5. Validate with real runs:
   - Any claim of accuracy must be backed by a concrete comparison run and saved report output.

