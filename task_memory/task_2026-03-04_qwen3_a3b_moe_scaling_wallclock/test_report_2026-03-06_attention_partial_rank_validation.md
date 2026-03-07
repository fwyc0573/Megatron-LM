## Test Report: Attention Contiguity and Partial-Rank MoE Validation

**Date**: 2026-03-06
**Environment**: `CUDA_DEVICE_MAX_CONNECTIONS=1` for `torchrun` repro, Python `python`, GPU `A800-SXM4-80GB`

### Test Script Information
- Files:
  - `tests/unit_tests/transformer/test_attention_qk_layernorm_contiguous.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_moe_rank_selection.py`
  - `megatron-sim-engine/tests/unit/test_simu_main_moe_rank_selection.py`
  - `megatron-sim-engine/tests/integration/test_moe_simulate_all_ranks_tp_barrier.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_ep_exp_semantics.py`
  - `megatron-sim-engine/tests/performance/wallclock_scaling/test_collective_sim_backend_cache.py`
- Commands:
  ```bash
  pytest -q megatron-sim-engine/tests/unit/test_simu_engine_moe_rank_selection.py \
    megatron-sim-engine/tests/unit/test_simu_main_moe_rank_selection.py \
    tests/unit_tests/transformer/test_attention_qk_layernorm_contiguous.py
  pytest -q megatron-sim-engine/tests/integration/test_moe_simulate_all_ranks_tp_barrier.py \
    megatron-sim-engine/tests/unit/test_simu_engine_ep_exp_semantics.py
  pytest -q megatron-sim-engine/tests/performance/wallclock_scaling/test_collective_sim_backend_cache.py
  python -m py_compile megatron/core/transformer/attention.py \
    megatron-sim-engine/simu_main.py \
    megatron-sim-engine/src/core/simu_engine.py \
    megatron-sim-engine/tests/unit/test_simu_engine_moe_rank_selection.py \
    megatron-sim-engine/tests/unit/test_simu_main_moe_rank_selection.py \
    tests/unit_tests/transformer/test_attention_qk_layernorm_contiguous.py
  CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0 torchrun ... --fake-current-rank-id 1400
  ```

### Validation Criteria
- Attention q/k layernorm path passes contiguous tensors into TE RMSNorm.
- `simu_main.py` accepts `--moe-rank-selection pp-ep` and forwards it into `SimulatorEngine`.
- MoE optimized selection uses `PP * EP` representative ranks in `pp-ep` mode.
- Existing MoE TP/EP semantics and collective-sim cache tests remain passing.
- Real repro rank `1400` completes warmup + forward + backward + optimizer without TE RMSNorm crash.

### Test Results

| Validation | Result | Evidence |
|------------|--------|----------|
| Unit tests (selection + CLI + attention) | PASS | `6 passed` |
| Integration tests (MoE TP barrier + EP/EXP semantics) | PASS | `8 passed` |
| collective-sim cache test | PASS | `1 passed` |
| Syntax check | PASS | Exit code `0` |
| Real repro rank1400 | PASS | `finish warm up`, `finish FWD`, `finish BWD`, `finish optimizer.step` in log |

### Evidence
- Rank1400 repro log: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/logs/repro_rank1400_20260306.log`
- Partial-rank simulation batch report: `task_memory/task_2026-03-04_qwen3_a3b_moe_scaling_wallclock/test_report_2026-03-06_megatron_sim_engine_partial_ranks.md`
