## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-09 | Added read-only analysis of megatron-sim-engine DDP overlap communication prediction path |

# Analysis: DDP Overlap Communication Prediction Path

## Scope
- Inspect the `megatron-sim-engine` simulate-mode path for `ddp_grad_comm` overlay replay.
- Focus on how DDP overlap communication is converted into CC backend requests.
- Explain likely causes for under-predicting actual comm latency from roughly `0.28-0.50 ms` down to roughly `0.02-0.09 ms`.
- No code changes.

## End-to-End Function Path

### 1. Trace emission from Megatron runtime
- `megatron/core/distributed/param_and_grad_buffer.py:128` `Bucket._init_pending_trace()` initializes DDP overlap metadata:
  - `comm_func`
  - `bucket_numel`
  - `bucket_numel_unpadded`
  - `grad_dtype`
  - `data_parallel_world_size`
  - `trigger_cmd_uid`
  - `wait_cmd_uid`
- `megatron/core/distributed/param_and_grad_buffer.py:171` `Bucket._emit_pending_trace()` serializes the event as `ddp_grad_comm(...)`.
- `megatron/core/distributed/param_and_grad_buffer.py:231` `Bucket.start_grad_sync()` marks metadata-only vs actual timing and chooses `allreduce` or `reduce_scatter`.
- `megatron/core/distributed/param_and_grad_buffer.py:327` `Bucket.finish_grad_sync()` fills wait-side timestamps.

### 2. Overlay ingestion in simulator
- `megatron-sim-engine/src/core/simu_engine.py:4893` `SimulatorEngine._build_trace_overlay_state()` indexes `ddp_grad_comm` records by `trigger_cmd_uid`.
- `megatron-sim-engine/src/core/simu_engine.py:4922` `SimulatorEngine._consume_matching_trace_top_level_operation()` matches normal top-level ops.
- `megatron-sim-engine/src/core/simu_engine.py:4934` `SimulatorEngine._apply_trace_overlay_metadata_to_schedule_operation()` copies `cmd_uid`, `op_semantics`, and trace metadata onto schedule ops.
- `megatron-sim-engine/src/core/simu_engine.py:4952` `SimulatorEngine._collect_ddp_overlap_overlay_operations()` attaches bucket comm records after the matched `backward_step`.
- `megatron-sim-engine/src/core/simu_engine.py:4856`–`4884` is the actual schedule/trace merge point.

### 3. Parsing `ddp_grad_comm` into simulator operations
- `megatron-sim-engine/src/core/simu_engine.py:3221` parses each textual trace line.
- `megatron-sim-engine/src/core/simu_engine.py:3225`–`3232` maps:
  - `ddp_grad_comm + comm_func=allreduce -> operation_name=dp_allreduce`
  - `ddp_grad_comm + comm_func=reduce_scatter -> operation_name=dp_reducescatter`
- `megatron-sim-engine/src/core/simu_engine.py:3242`–`3247` forces `ddp_grad_comm` to `op_kind='comm'`.
- `megatron-sim-engine/src/core/simu_engine.py:3427`–`3434` converts payload metadata into simulator tensor metadata:
  - `tensor_shape = [bucket_numel_unpadded]` (fallback to `bucket_numel` only if unpadded is missing)
  - `tensor_dtype = grad_dtype`

### 4. Converting the overlay op into a CC backend request
- `megatron-sim-engine/src/core/simu_engine.py:975` routes simulate-mode `ddp_grad_comm` into `TimelinesManager._add_trace_driven_async_ddp_overlap_comm()`.
- `megatron-sim-engine/src/core/simu_engine.py:652` `TimelinesManager._predict_profile_overlap_comm_duration()` delegates to `_calculate_comm_duration([operation])`.
- `megatron-sim-engine/src/core/simu_engine.py:1216` `_get_comm_group_for_operation()` reconstructs the full communication group from simulator topology, not from trace payload.
- `megatron-sim-engine/src/core/simu_engine.py:1458` `_get_comm_data_size()` turns `tensor_shape + tensor_dtype` into bytes.
- `megatron-sim-engine/src/core/simu_engine.py:1501` `_build_comm_request_metadata()` only adds extra metadata for `p2p`; DDP collectives only carry `mg_state`.
- `megatron-sim-engine/src/core/simu_engine.py:1574` `_calculate_comm_duration()` builds `CommunicationPredictionRequest.from_raw(...)` with:
  - `comm_group`
  - `op_name`
  - `data_size_bytes`
  - `group_kind`
  - `domain_dims`
  - `tensor_shape`
  - `tensor_dtype`
  - `mpu_info`
  - `metadata`

### 5. CC backend consumption
- `megatron-sim-engine/src/core/cc_backend/types.py:31` defines `CommunicationPredictionRequest`.
- `megatron-sim-engine/src/core/cc_backend/op_mapping.py:29` `infer_collective_kind()` maps:
  - `dp_allreduce -> allreduce`
  - `dp_reducescatter -> reducescatter`
- `megatron-sim-engine/src/core/cc_backend/collective_sim_backend.py:563` `_build_global_placement_context()` derives:
  - `servers`
  - `gpus_per_server`
  - `participant_ranks`
  - full parallelism dimensions
- `megatron-sim-engine/src/core/cc_backend/collective_sim_backend.py:418` `_build_collective_options()` emits collective-sim payload fields:
  - `kind`
  - `tensor_bytes`
  - `domain_dims`
  - `placement_order`
  - `exclude_intra_server`
  - `participant_ranks`
- `megatron-sim-engine/src/core/cc_backend/collective_sim_backend.py:660` `CollectiveSimCCBackend.predict()` forwards the normalized request into collective-sim.

## Exact Request Semantics Observed

For a DDP overlap bucket, the simulator-side request is effectively:

1. **Collective type**
   - Derived from `comm_func` during parsing.
   - `allreduce` becomes `dp_allreduce`.
   - `reduce_scatter` becomes `dp_reducescatter`.

2. **Tensor bytes**
   - Derived from `bucket_numel_unpadded * sizeof(grad_dtype)`.
   - Implemented through `tensor_shape=[bucket_numel_unpadded]` and `_get_comm_data_size()`.

3. **Tensor dtype**
   - Taken from `grad_dtype` in the trace event.

4. **Group size / participants**
   - Reconstructed from simulator topology via `_get_comm_group_for_operation()`.
   - Not taken from `data_parallel_world_size` in trace.

5. **Placement info**
   - Reconstructed from `mpu_info` and backend defaults.
   - Default H800 config uses `placement_mode=auto`, `exclude_intra_server=True`, `gpus_per_server=8`.

## Suspicious Mismatches

### Mismatch 1: DDP trace carries `data_parallel_world_size`, but the simulator ignores it
- Trace emission records `data_parallel_world_size` at `megatron/core/distributed/param_and_grad_buffer.py:163`.
- Request build never validates reconstructed `comm_group` against that trace field.
- Consequence:
  - If simulator topology or rank mapping is wrong, request group size can silently become too small or too local.
  - That would directly push predictions downward.

### Mismatch 2: `bucket_numel_unpadded` is preferred over actual communicated buffer size
- `megatron-sim-engine/src/core/simu_engine.py:3428` prefers `bucket_numel_unpadded` over `bucket_numel`.
- Actual collective uses `self.grad_data`, i.e. the padded bucket buffer.
- This is a one-sided bias toward smaller payloads.
- Likely impact is small-to-moderate, but it always pushes prediction lower, never higher.

### Mismatch 3: `calculate_comm_message_size()` ignores collective semantics completely
- `megatron-sim-engine/src/utils/message_size_calculator.py:86` returns raw tensor bytes and ignores both `comm_op` and `world_size`.
- This is suspicious for collectives whose semantic payload differs by operation type.
- Most importantly, collective-sim documents at `megatron-sim-engine/src/core/cc_backend/collective-sim/htsim_runner.py:64`–`71` that for `reducescatter`, `tensor_bytes` is **per-rank output**.
- But the DDP trace feeds full bucket bytes into the simulator for `reduce_scatter`.
- This mismatch can distort `dp_reducescatter` predictions.

### Mismatch 4: Intra-server collectives are forced through an idealized model and then combined with `max(...)`
- Default backend options come from `megatron-sim-engine/src/core/simulator_config.py:109` and `:59`:
  - `exclude_intra_server=True`
  - H800 NVLink analytic intra-server model
  - 400 Gbps network defaults
- collective-sim combines `network_ms` and `intra_server_ms` by `max(...)` in `megatron-sim-engine/src/core/cc_backend/collective-sim/python/collective_sim_core/predictor.py:61`–`69`.
- That means intra-server and network-side costs are not accumulated by default for non-hierarchical collectives.

### Mismatch 5: Dummy 1-byte intra-server flows create a flat network floor
- `megatron-sim-engine/src/core/cc_backend/collective-sim/htsim_runner.py:480` sets `_INTRA_SERVER_BYTES = 1`.
- `megatron-sim-engine/src/core/cc_backend/collective-sim/htsim_runner.py:500`–`503` substitutes `1` byte flows for excluded intra-server edges instead of removing them.
- On a fully intra-server DDP group, this yields a nearly constant non-zero `network_ms` floor.
- Because the predictor then uses `max(network_ms, intra_ms)`, that floor clamps small/medium payload predictions to an almost constant value.

## Direct Read-Only Evidence

### Empirical backend sweep
Using the current repository code and default `create_h800_sxm_ib_config()` settings, direct backend queries produced:

#### Group `(0,1,2,3)` with `dp_allreduce`
- `0.25 MB -> 0.053784 ms`
- `0.5 MB -> 0.053784 ms`
- `1 MB -> 0.053784 ms`
- `2 MB -> 0.053784 ms`
- `4 MB -> 0.053784 ms`
- `8 MB -> 0.082243 ms`

This is a strong red flag: payload grows by `32x`, but prediction stays flat until `8 MB`.

#### Breakdown for `(0,1,2,3)`, `1 MB`, `allreduce`
- `network_ms = 0.0537843`
- `intra_server_ms = 0.0134304`
- `combined_ms = 0.0537843`
- combine rule = `max`

#### Breakdown for `(0,1,2,3)`, `4 MB`, `allreduce`
- `network_ms = 0.0537843`
- `intra_server_ms = 0.0429216`
- `combined_ms = 0.0537843`
- combine rule = `max`

This shows the constant network floor dominating the final answer.

#### Group `(0,8,16,24)` with `dp_allreduce`
- `0.25 MB -> 0.027542 ms`
- `0.5 MB -> 0.032606 ms`
- `1 MB -> 0.049669 ms`
- `2 MB -> 0.085742 ms`
- `4 MB -> 0.147498 ms`
- `8 MB -> 0.272035 ms`

This is better than the single-node flatline, but still very optimistic versus real NCCL timings if actual measurements are already in the `0.28-0.50 ms` range for smaller buckets.

## Most Likely Root Causes

### Primary root cause
The dominant issue is **not** the trace-to-request field mapping itself. The mapping is mostly coherent for `allreduce`.

The dominant issue is the **collective-sim backend model and combine rule** used after the request is built:
- idealized H800 NVLink / network defaults,
- default `ring_steps` collective model,
- `exclude_intra_server=True`,
- dummy `1`-byte intra-server network flows,
- final `max(network_ms, intra_server_ms)` combination.

This stack naturally produces `~0.02-0.09 ms` predictions for small DDP buckets, which matches the under-predicted range you reported.

### Secondary root cause candidates
1. **Wrong or overly local comm group reconstruction**
   - because `data_parallel_world_size` is ignored and no validation checks exist.
2. **Payload undercount from using `bucket_numel_unpadded`**
   - always biases down.
3. **Wrong `reducescatter` byte semantics when distributed optimizer is enabled**
   - the backend expects per-rank output bytes, but the simulator uses full bucket bytes.
   - This is a semantic mismatch even if it is not the best explanation for the specific under-prediction you observed.

## Bottom-Line Assessment

- The **conversion path** from DDP overlap event to CC backend request is:
  `Bucket._init_pending_trace()`
  → `ddp_grad_comm(...)`
  → `process_mg_files()` parser/mapping
  → `_add_trace_driven_async_ddp_overlap_comm()`
  → `_calculate_comm_duration()`
  → `CommunicationPredictionRequest.from_raw()`
  → `CollectiveSimCCBackend.predict()`.

- The **most suspicious code-level mismatch for your symptom** is the backend-side modeling stack, especially:
  - `collective-sim/python/collective_sim_core/predictor.py:61`
  - `collective-sim/htsim_runner.py:480`
  - `collective-sim/htsim_runner.py:500`
  - `src/core/simulator_config.py:109`

- If your DDP groups are single-node on the 8-GPU testbed, these lines are the most likely direct explanation for seeing actual `~0.28-0.50 ms` versus predicted `~0.02-0.09 ms`.

