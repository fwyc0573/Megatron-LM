## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Implemented debug-only attention backward segment NVTX (including MLA path), completed clean x1 validation and formal seq8192 repeat-x5 run, and produced segment-level residual diagnosis summary |

## Test Report: Seq8192 Attention Segment Debug + Repeat-x5

**Date**: 2026-03-01  
**Environment**: `conda activate myenv_yc` (Python 3.9.18, CUDA 12.1, Nsight Systems 2023.1.2.43)

### 1) Scope

1. Implement debug-only attention backward micro-segments (3-4 logical segments) without changing training semantics.
2. Reuse existing phase-pure NSYS analyze/compare flow for clean x1:
   - distributed / scaling_on / scaling_off
   - `SEQ_LEN=8192`, `TRAIN_ITERS=3`, phase tracing enabled.
3. Verify segment labels are actually emitted in workload-relevant path (DeepSeek MLA).
4. If x1 diagnosis is stable, execute formal repeat-x5 with the same protocol and summarize robustness.

### 2) Code Changes

- Trace/CLI:
  - `megatron/training/arguments.py`
    - new arg: `--trace-attention-backward-segments`.
  - `megatron/core/transformer/transformer_config.py`
    - new config field: `trace_attention_backward_segments: bool = False`.
- NVTX label support:
  - `megatron/profiler/cmd.py`
    - phase NVTX now accepts `extra_tags` and writes them into CMD label.
- Attention module instrumentation:
  - `megatron/core/transformer/attention.py`
    - register full backward pre/post hooks for `attn_qkv_bwd`, `attn_qk_layernorm_bwd`, `attn_core_bwd`, `attn_proj_bwd`.
  - `megatron/core/transformer/multi_latent_attention.py`
    - added `_MLASDPACoreAttention` submodule for SDPA core path;
    - added the same backward segment hooks in MLA path (critical for DeepSeek workload).
- Script passthrough:
  - `examples/pretrain_deepseek_v3_moe.sh`
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - new env passthrough: `TRACE_ATTENTION_BACKWARD_SEGMENTS`.
- Segment diagnosis script:
  - `tests/performance/analyze_nsys_attention_family_delta.py`
    - support segment filtering (`--segment-key/--segment-values`);
    - add small-kernel adjacency and stream diagnostics.

### 3) Unit / Regression Validation

#### 3.1 Commands

```bash
python -m py_compile \
  megatron/profiler/cmd.py \
  megatron/core/transformer/attention.py \
  megatron/core/transformer/multi_latent_attention.py \
  megatron/training/arguments.py \
  megatron/core/transformer/transformer_config.py \
  tests/performance/analyze_nsys_attention_family_delta.py

LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0 \
python -m pytest \
  tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py \
  tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py \
  tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py \
  tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py \
  tests/unit_tests/performance/test_check_nsys_nvtx_structural_health.py \
  tests/unit_tests/tensor_parallel/test_mappings_moe_api.py \
  tests/unit_tests/transformer/test_attention.py \
  -k "segment_hooks or cmd_kernel_ground_truth or analyze_nsys_attention_family_delta or analyze_nsys_cmd_kernel_breakdown or compare_qwen_nsys_compute_only or check_nsys_nvtx_structural_health" -q

LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0 \
python -m pytest tests/unit_tests/test_training.py \
  -k "trace_kernel_ground_truth_phase_defaults or trace_kernel_ground_truth_phase_args or core_transformer_config_injects_new_fields" -q

LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0 \
python -m pytest tests/unit_tests/transformer/test_multi_latent_attention.py -k "backward_segment_hooks" -q
```

#### 3.2 Results

- `py_compile`: PASS
- pytest set-1: `35 passed, 14 deselected`
- pytest set-2: `3 passed, 21 deselected`
- pytest set-3: `2 passed, 2 deselected`

### 4) Clean x1 (patched + attention segment enabled)

Artifacts directory:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_seg_x1_v2/`

#### 4.1 Core evidence

1. Segment labels are present in sqlite (MLA path confirmed):
   - `segment_label_count=768`
   - sample labels contain `attn_bwd_segment=attn_proj_bwd/attn_core_bwd`.
2. NVTX structural gate: all PASS (`open_forward=0`, `open_backward=0`, overlap=0).
3. phase-pure compare (shared primary stream, pure_primary_union):
   - scaling_on backward op-rank median: `18.09%`
   - scaling_off backward op-rank median: `16.30%`
4. Segment diagnosis (`stage1/steady/rank4..7`) indicates stable dominance:
   - `attn_core_bwd` is the only segment with material gap;
   - `fmha_gap_share_pct` ~`98%` (both on/off);
   - other segments (`attn_proj_bwd`, `attn_qk_layernorm_bwd`, `attn_qkv_bwd`) are near-zero gap.

#### 4.2 Stream / small-kernel checks

For `attn_core_bwd` (x1):
- `primary_stream_id_mismatch_pairs=0`
- `fmha_stream_set_mismatch_pairs=0`
- small kernels adjacent **before** fmha are consistently higher on distributed side (`dist 32` vs `scale 23` in x1 core slice), with zero post-fmha adjacency on both sides.

Interpretation:
- No stream-id routing mismatch evidence.
- Residual is concentrated in SDPA core segment (`fmha_cutlassB`) and accompanied by distributed-side pre-fmha small-kernel adjacency inflation.

### 5) Formal Repeat-x5 (same protocol)

Artifacts directory:
- `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_seg_repeat5/`

Summary files:
- `deepseek_phase_sl8192_attnseg_repeat5_summary.md`
- `deepseek_phase_sl8192_attnseg_repeat5_summary.json`

#### 5.1 Repeat-x5 summary (median-of-runs)

- NVTX gate:
  - dist/scaling_on/scaling_off all PASS in all runs.
- op-rank median (pure_primary_union):
  - scaling_on: `forward=15.61%`, `backward=11.20%`, `optimizer=2.29%`
  - scaling_off: `forward=15.74%`, `backward=8.99%`, `optimizer=2.76%`
- `attn_core_bwd` (rank4..7, stage1, steady):
  - scaling_on: `gap_ms median=22.814`, `fmha_gap_ms median=22.336`, `fmha_share median=98.01%`
  - scaling_off: `gap_ms median=20.199`, `fmha_gap_ms median=19.783`, `fmha_share median=97.96%`
  - top1 kernel is `fmha_cutlassB` in **all 5/5 runs** (on/off).

#### 5.2 Stream/small-kernel robustness

`attn_core_bwd` repeat-x5 medians:
- `dist_pre_count=35`, `scale_pre_count=22`
- `dist_pre_ms=1.751`, `scale_pre_ms~1.10`
- stream mismatch counters remain zero across runs.

Interpretation:
- residual source is robustly concentrated in attention core (`fmha_cutlassB`) rather than segment routing mismatch;
- distributed-side pre-fmha small-kernel adjacency overhead is persistent and likely contributes to contention context.

### 6) Verdict

1. x1 localization is stable and reproducible under repeat-x5.
2. Root-cause priority should stay on attention core path (`attn_core_bwd`) and its immediate pre-fmha neighborhood behavior.
3. comm-adjacent emulation is not the main lever for this residual.
4. backward gate still not frozen to <=5% under this protocol.

### 7) Key Output Paths

- x1 v2 logs:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_seg_x1_v2/`
- repeat-x5 logs:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_seg_repeat5/`
- repeat-x5 summary:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_seg_repeat5/deepseek_phase_sl8192_attnseg_repeat5_summary.md`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_phase_attn_seg_repeat5/deepseek_phase_sl8192_attnseg_repeat5_summary.json`
