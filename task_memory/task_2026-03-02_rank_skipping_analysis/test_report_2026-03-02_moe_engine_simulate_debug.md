## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Added MoE simulate engine debug verification report: unit/integration tests + 4-case Mixtral replay and gap comparison |

## Test Report: MoE Engine Simulating Debug (All Ranks + TP Barrier + EXP/EP Semantics)

**Date**: 2026-03-02  
**Environment**:
- Conda: `myenv_yc`
- Python: `3.9.18` (`/opt/anaconda/envs/myenv_yc/bin/python`)
- PYTHONPATH: repo root + `megatron-sim-engine/`

### 1) Test Script Information

- Modified core file:
  - `megatron-sim-engine/src/core/simu_engine.py`
- Added tests:
  - `megatron-sim-engine/tests/unit/test_simu_engine_moe_detection.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_moe_rank_selection.py`
  - `megatron-sim-engine/tests/unit/test_simu_engine_ep_exp_semantics.py`
  - `megatron-sim-engine/tests/integration/test_moe_simulate_all_ranks_tp_barrier.py`

#### Reproducible commands

```bash
# Unit + integration (new tests)
pytest megatron-sim-engine/tests/unit/test_simu_engine_moe_detection.py -v
pytest megatron-sim-engine/tests/unit/test_simu_engine_moe_rank_selection.py -v
pytest megatron-sim-engine/tests/unit/test_simu_engine_ep_exp_semantics.py -v
pytest megatron-sim-engine/tests/integration/test_moe_simulate_all_ranks_tp_barrier.py -v

# Combined regression for this patch set
pytest \
  megatron-sim-engine/tests/unit/test_simu_engine_moe_detection.py \
  megatron-sim-engine/tests/unit/test_simu_engine_moe_rank_selection.py \
  megatron-sim-engine/tests/unit/test_simu_engine_ep_exp_semantics.py \
  megatron-sim-engine/tests/integration/test_moe_simulate_all_ranks_tp_barrier.py -v

# Existing semantic regression check
pytest megatron-sim-engine/tests/unit/test_simu_engine_cc_semantics.py -v

# 4-case replay via template script
MIXTRAL_INPUT_DIR=simulation_inputs/megatron_operation_log/moe_mixtral8_1.75b_2pp_1tp_4dp_2ep_4096seq \
WORLD_SIZE=8 PP_SIZE=2 TP_SIZE=1 EXP_SIZE=2 LOCAL_SIZE=8 \
bash megatron-sim-engine/examples/03_mixtral_16gpu_2node_template.sh

MIXTRAL_INPUT_DIR=simulation_inputs/megatron_operation_log/moe_mixtral8_1.75b_2pp_1tp_4dp_4ep_4096seq \
WORLD_SIZE=8 PP_SIZE=2 TP_SIZE=1 EXP_SIZE=4 LOCAL_SIZE=8 \
bash megatron-sim-engine/examples/03_mixtral_16gpu_2node_template.sh

MIXTRAL_INPUT_DIR=simulation_inputs/megatron_operation_log/h800_moe_mixtral8_1.75b_2pp_1tp_4dp_2ep_4096seq \
WORLD_SIZE=8 PP_SIZE=2 TP_SIZE=1 EXP_SIZE=2 LOCAL_SIZE=8 \
bash megatron-sim-engine/examples/03_mixtral_16gpu_2node_template.sh

MIXTRAL_INPUT_DIR=simulation_inputs/megatron_operation_log/h800_moe_mixtral8_1.75b_2pp_1tp_4dp_4ep_4096seq \
WORLD_SIZE=8 PP_SIZE=2 TP_SIZE=1 EXP_SIZE=4 LOCAL_SIZE=8 \
bash megatron-sim-engine/examples/03_mixtral_16gpu_2node_template.sh
```

### 2) Validation Criteria

1. MoE detection in simulate mode is robust without trace-only dependency.
2. MoE simulate path selects all ranks.
3. TP barrier is preserved in MoE simulate (peer matching is not cleared).
4. EP/EXP semantic mapping is explicit and fail-fast on mismatch.
5. 4 Mixtral cases show reduced simulate/profile E2E gap compared with baseline.

### 3) Test Results and Evidence

#### 3.1 Pytest status

| Suite | Result | Evidence |
|------|--------|----------|
| `test_simu_engine_moe_detection.py` | PASS | 5/5 passed |
| `test_simu_engine_moe_rank_selection.py` | PASS | 3/3 passed |
| `test_simu_engine_ep_exp_semantics.py` | PASS | 7/7 passed |
| `test_moe_simulate_all_ranks_tp_barrier.py` | PASS | 1/1 passed |
| Combined new tests | PASS | 16/16 passed |
| Existing `test_simu_engine_cc_semantics.py` | PASS | 4/4 passed |

#### 3.2 Replay evidence (script logs)

- Log files:
  - `/tmp/mixtral_a800_2ep_after_fix.log`
  - `/tmp/mixtral_a800_4ep_after_fix.log`
  - `/tmp/mixtral_h800_2ep_after_fix.log`
  - `/tmp/mixtral_h800_4ep_after_fix.log`
- Key grep evidence:
  - All 4 logs show: `优化策略: MOE 模型，选择了 8 个ranks: [0, 1, 2, 3, 4, 5, 6, 7]`
  - No log shows legacy pattern `Dense 模型，选择了 2 个ranks` for MoE case.

#### 3.3 E2E gap comparison (baseline vs current)

E2E extracted from timeline max-finish-time (`max(op.finish_time)` across all ranks).

| Case | Baseline Sim/Profile (ms) | Baseline Gap | Current Sim/Profile (ms) | Current Gap | Improvement (pct-point) |
|------|----------------------------|--------------|---------------------------|-------------|--------------------------|
| a800_2ep | 783.80 / 939.36 | -16.56% | 806.62 / 939.36 | -14.13% | +2.43 |
| a800_4ep | 597.19 / 981.14 | -39.13% | 836.63 / 981.14 | -14.73% | +24.40 |
| h800_2ep | 351.30 / 431.77 | -18.64% | 358.99 / 431.77 | -16.86% | +1.78 |
| h800_4ep | 253.80 / 357.74 | -29.05% | 334.51 / 357.74 | -6.49% | +22.56 |

### 4) Failure Handling

- No code/test failures remained unresolved in this patch cycle.
- One transient command issue occurred during an ad-hoc analysis script (syntax typo), fixed immediately and rerun successfully.

### 5) Residual Risk Notes

- Remaining gap in 2ep cases is no longer dominated by rank-skipping; it is mainly from communication realism mismatch (notably `recv_backward`, `exp_all_to_all`, and in some cases `dp_allreduce`) between profile-measured latency and current CC/backend estimation + dependency replay.
- This residual is outside the strict engine-rank-selection bug fixed in this task, and should be addressed in a follow-up calibration task for communication prediction and scheduling wait modeling.
