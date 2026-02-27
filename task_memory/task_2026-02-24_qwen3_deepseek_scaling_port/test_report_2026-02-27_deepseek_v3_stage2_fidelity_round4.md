## Test Report: DeepSeek-V3 Stage-2 Fidelity Round4

**Date**: 2026-02-27  
**Environment**: `conda activate myenv_yc` (Python 3.9), GPU testbed (8 GPUs), repo `Megatron-LM`

### 1) Test Script Information

- Modified code path:
  - `megatron/training/training.py` (`_prepare_scaling_optimizer_step`)
- Repro commands (core):

```bash
# syntax check
python -m py_compile megatron/training/training.py

# scaling rank0 probe (after patch)
MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
SCALING_PROFILE_ITERS=3 SCALING_MIN_WARMUP_ITERS=0 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global DO_TRACE=True \
SCALE_GPU=5 SCALING_FAKE_RANK_ORDER=0 \
SCALING_REPLAY_CACHE_TAG=stage2_fidelityfix5_probe_rank0 \
bash examples/pretrain_deepseek_v3_moe.sh

# scaling interleaved two-pass
MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
SCALING_PROFILE_ITERS=3 SCALING_MIN_WARMUP_ITERS=0 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global DO_TRACE=True \
SCALE_GPU=5 MASTER_PORT=7600 \
SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALING_REPLAY_CACHE_TAG=stage2_fidelityfix5_interleave \
bash examples/pretrain_deepseek_v3_moe.sh

MODE=scaling MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
SCALING_PROFILE_ITERS=3 SCALING_MIN_WARMUP_ITERS=0 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global DO_TRACE=True \
SCALE_GPU=5 MASTER_PORT=7700 \
SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 \
SCALING_REPLAY_CACHE_TAG=stage2_fidelityfix5_interleave \
bash examples/pretrain_deepseek_v3_moe.sh

# distributed refresh run
MODE=distributed MODEL_PROFILE=smoke TRACE_START=4 TRAIN_ITERS=6 \
TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global DO_TRACE=True \
MASTER_PORT=7800 \
bash examples/pretrain_deepseek_v3_moe.sh

# compare reports
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --pair-timestamp 20260227145522
```

### 2) Validation Criteria

- Code-level criterion:
  - Scaling path optimizer pre-CMD prep should mirror distributed pre-CMD side effects.
- Runtime criterion:
  - Scaling and distributed runs finish with exit code `0` and produce rank-complete traces.
- Fidelity criterion (paper-facing auxiliary):
  - `forward_step/backward_step/optimizer_step` op-rank-median diff target `<= 5%`.

### 3) Test Results and Evidence

#### 3.1 Code verification

- `python -m py_compile megatron/training/training.py` -> **PASS** (exit code `0`).

#### 3.2 Runtime execution

- Scaling rank0 probe (`stage2_fidelityfix5_probe_rank0`) -> **PASS** (exit code `0`).
- Scaling interleaved two-pass (`stage2_fidelityfix5_interleave`) -> **PASS** (both passes exit code `0`).
- Distributed refresh run (`ts=20260227145522`) -> **PASS** (exit code `0`, rank0..7 traces generated).

#### 3.3 Fidelity evidence

- Rank0 optimizer targeted evidence:
  - before patch pairing (`scaling ts=20260227142506`): `optimizer_step diff = 12.76%`
    - report: `logs/deepseek_v3_stage2_compare_optimizer_rank0_before_patch.log`
  - after patch pairing (`scaling ts=20260227144128`): `optimizer_step diff = 6.02%`
    - report: `logs/deepseek_v3_stage2_compare_optimizer_rank0_after_patch.log`

- Latest full-pair evidence (subtract-comm, distributed `ts=20260227145522`, scaling `ts<=20260227145409`):
  - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_profile3_fidelityfix5_interleave_vs_dist145522_subtract.log`
  - `op_rank_median_aux_summary`:
    - `forward_step`: **4.02%** (PASS)
    - `backward_step`: **5.11%** (FAIL, near threshold)
    - `optimizer_step`: **7.68%** (FAIL)

### 4) Failures and Resolutions

- Failure encountered:
  - scaling pass2 during one run hit TCPStore bind failure:
    - `Address already in use` on `MASTER_PORT + rank`.
- Resolution:
  - rerun with high sparse port ranges (`MASTER_PORT=7400/7600/7700/7800`) and split rank batches when needed.
- Post-resolution status:
  - affected runs completed successfully; reports regenerated.

### 5) Conclusion

- Round4 patch **improves optimizer fidelity** (clear reduction on targeted probe and overall median trend), but **full 3-op <=5% gate is still not met**.
- Remaining dominant residuals are `optimizer_step` and near-threshold `backward_step` under current protocol.
