## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added backward-comp governance note: freeze official metric to `seq8192 + phase-pure + repeat-x5`, classify advanced knobs as opt-in diagnostics, and lock next root-cause scope to `attn_core_sdpa_bwd` iter-bucket analysis |
| 2026-03-01 | Added deep-segment repeat-x5 note: phase-pure/NVTX gates stay clean, `attn_core_sdpa_fmha_bwd` remains dominant (top1 `fmha_cutlassB` 5/5), and variability bucketing exposes iter1-dominant residual pattern coupled with pre-fmha adjacency shift |
| 2026-03-01 | Added attention-core SDPA subsegment formal repeat-x5 note: `fmha_cutlassB` stays top1 in 5/5 runs (on/off), contamination remains 0%, and pre-fmha `FillFunctor<unsigned char>` adjacency asymmetry is stable while backward variability remains high |
| 2026-03-01 | Added `attn_core_sdpa_bwd` immediate same-stream neighbor note: pre-fmha `FillFunctor<unsigned char>` exists on both branches; count gap at `60us` is threshold-sensitive and collapses at `80us`, while backward gap remains unchanged |
| 2026-03-01 | Added attention-core micro-segment x1 note: `attn_core_bwd` is further localized to `attn_core_sdpa_bwd` with ~98% fmha share; pre/post-cast subsegments are near-zero and pre-fmha top adjacent kernel is stabilized as `FillFunctor<unsigned char>` |
| 2026-03-01 | Added attention-segment debug note: landed backward segment NVTX (SelfAttention + MLA), clean-x1 confirmed `attn_core_bwd` concentration, and repeat-x5 showed `fmha_cutlassB` remains 98% core-gap contributor with stable stream-set parity |
| 2026-03-01 | Added patched clean-x1 attention-family diagnosis note: strengthened analyzer confirms `fmha_cutlassB` remains dominant in stage1 backward (`rank4..7`) for both scaling DDP on/off, with launch-config parity and partial-only DDP-off improvement |
| 2026-03-01 | Added patched seq8192 phase-pure x1 note with NVTX structural gate: after NVTX fix, all branches pass `open/overlap` health checks and contamination remains zero, but backward residual is still >5% and attention-family remains dominant |
| 2026-03-01 | Added NVTX attribution-artifact note: repeat-x5 sqlite confirms systematic forward/backward CMD overlap from `row_g_fwd` NVTX stack leak; minimal `try/finally` fix landed with RED→GREEN test evidence |
| 2026-03-01 | Added seq8192 phase-pure formal repeat-x5 note: DDP-off only partially improves backward under robust aggregation, while `fmha_cutlassB` stays top1 in 5/5 runs; priority shifts to attention-family diagnostics |
| 2026-03-01 | Added seq8192 phase-pure DDP probe note: distributed x1 + scaling DDP on/off x1 shows zero contamination and partial DDP contribution, but backward remains above 5% and attention-family remains dominant residual |
| 2026-03-01 | Added scaling DDP-hook hypothesis validation note: disabling scaling DDP hook accumulation reduces backward kernel/time moderately but does not improve dist-vs-scale backward residual; dominant non-comm conclusion remains |
| 2026-03-01 | Added cross-run source-validation note: `round68 seq8192 run1..5` confirms non-comm (`fmha_cutlassB`) dominance is stable; `_AllToAll` micro-repro shows no autograd-node-count inflation between distributed/scaling |
| 2026-03-01 | Added backward residual deep-dive note: comm-adjacent share is partial (~27% on round68 high-gap set), dominant delta is non-comm kernel family; scaling emulation knob is feasible but x1 impact on backward is marginal |
| 2026-03-01 | Added postfix all_to_all attribution-fix note: scaling alias-return removal + contiguous-in-wrapper verified; phase contamination remains zero but residual did not improve materially in x1 sanity compare |
| 2026-03-01 | Added phase-label NSYS sanity x1 notes: distributed/scaling captures confirm phase-window coverage and zero contamination; residual gap remains and requires seq8192 repeat-x5 freeze run |
| 2026-03-01 | Added phase-level comp-only semantics implementation notes: CMD nested phase NVTX, boundary sync mode, NSYS pure metrics/contamination fields, and compare contamination gating behavior |
| 2026-02-28 | Added Round6-8 NSYS repeat-x5 note: subtract/no-subtract/NSYS tri-view on seq8192 shows backward gate semantics still unfrozen (NSYS backward remains high despite low spread) |
| 2026-02-28 | Added backward measurement-semantics validation note on seq8192 fixed batch: baseline subtract inflation is stage1 overlap bias; stage-aware/op-map auxiliary views isolate true optimizer residual |
| 2026-02-28 | Added Round6-8 measurement-regime change notes: full-profile OOM ceiling, seq8192 repeat-x5 noise-floor evidence, and updated interpretation (noise reduction is one-side only; backward remains subtraction-bias dominated) |
| 2026-02-28 | Added O1 implementation outcome on round6-8 baseline (pre-CMD drain A/B repeat5): mechanism validated, metrics regress vs control, and noise-floor-aware rejection decision |
| 2026-02-28 | Added O1 precise mechanism definition (single-variable, symmetric pre-CMD drain), round6-8 noise-floor repeat5 quantification, and B1 mainline-backport decision note |
| 2026-02-28 | Added round6-8-baseline O2/B1 follow-up evidence: microphase A/B verdict (O2 not supported), strict-grad-replay fail-fast implementation/validation, and next-step reprioritization to O1 |
| 2026-02-28 | Added current-latest Round12-protocol rerun evidence (run1/2/3 + median-of-runs), cross-round best-round decision (Round4 vs Round6-8), and retrospective conclusions for Round9+ |
| 2026-02-28 | Added stage-2 round12 post-optimizer replay-write repeated-pairing fidelity evidence (run1/2/3 + median-of-runs), and updated residual-risk interpretation |
| 2026-02-27 | Added stage-2 round11 scaling replay-write-phase / scheduler-increment semantic-alignment experiments and new fidelity evidence |
| 2026-02-27 | Added stage-2 optimizer microphase protocolfix8 phase-aware fidelity findings (run1/2/3 + median-of-runs) |
| 2026-02-27 | Implemented stage-2 optimizer microphase trace-only path (default-off) and added unit-validation notes |
| 2026-02-27 | Added stage-2 protocolfix8 fidelity notes and optimizer semantic-touching proposal (design-only, pending user confirmation) |
| 2026-02-27 | Added stage-2 round5 replay-cache iteration-alignment design notes and latest fidelity status |
| 2026-02-24 | Added architecture comparison, gap analysis, stage-1 simplifications, and stage-2 backlog |
| 2026-02-27 | Added stage-2 (DeepSeek-V3 architecture standard) detailed spec mapping to upstream YAML, plus environment constraints (TE=1.3.0) and implementation decisions |
| 2026-02-27 | Added stage-2 execution notes: shared-expert gate support, short-run scheduler guard, and distributed PP2 bf16 NaN observations |
| 2026-02-27 | Added stage-2 round2 notes: PP2 bf16 NaN root cause isolation and MLA-only p2p dtype-alignment fix with router finite-normalization hardening |
| 2026-02-27 | Added stage-2 round3 fidelity notes: CMD sync default rollback, scaling optimizer timing-boundary alignment, and residual backward/optimizer gap characterization |
| 2026-02-27 | Added stage-2 round4 fidelity notes: scaling optimizer pre-CMD side-effect parity (`numel` pre-scan), refreshed pair runs, and updated residual-gap status |

# Architecture Notes (Mixtral vs Qwen3 vs DeepSeek-V3-Proxy)

## Stage-2 governance note (2026-03-01): official backward metric semantics are frozen and advanced diagnostics are opt-in

### Frozen official semantics

- Paper-facing and acceptance-facing backward tracking is now anchored to:
  - `SEQ_LEN=8192`
  - phase-pure NSYS metric (`pure_primary_union`)
  - contamination gate (`<=1%`, target observed `0%`)
  - robust repeat aggregation (`repeat-x5`, report `median/IQR/range`).
- The following remain diagnostic-only and are not official gate semantics:
  - trace subtract/no-subtract
  - stage-aware/op-map subtraction variants.

### Advanced diagnostics classification

- The following knobs are classified as advanced diagnostics (opt-in only):
  - `TRACE_OPTIMIZER_MICROPHASES`
  - `SCALING_STRICT_GRAD_REPLAY`
  - `SCALING_REPLAY_WRITE_PHASE=post_optimizer`
  - `SCALING_ALIGN_SCHEDULER_INCREMENT`
  - `SCALING_COMM_ADJACENT_COPY_ITERS>0`
  - `SCALING_DISABLE_DDP_WRAP`
  - `TRACE_ATTENTION_BACKWARD_SEGMENTS`
- Example scripts now require explicit acknowledgement (`ADVANCED_DIAGNOSTICS=1`) when any of these non-baseline toggles are enabled.

### Root-cause scope lock

- Root-cause exploration is now constrained to attention SDPA backward context:
  - `attn_core_sdpa_bwd`
  - iter-bucket asymmetry (`iter1` vs `iter0/2`)
  - same-stream pre-fmha neighborhood effects.
- Non-attention broad A/B (e.g., expanding comm-adjacent emulation surface) is deprioritized unless new evidence disproves current dominant-source conclusion.

## Stage-2 note (2026-03-01): deep-segment repeat-x5 keeps semantic cleanliness but confirms iter1-dominant fmha residual bucket

### Scope

- Workload and protocol:
  - `seq8192`, `train_iters=3`, run1..5
  - branches: `distributed + scaling_on + scaling_off`
  - phase-pure kernel semantics unchanged
  - debug-only deep segment tags enabled in attention core.

### Semantic hygiene status

- NVTX structure health (all runs/branches):
  - `open_forward=0`
  - `open_backward=0`
  - `forward/backward overlap=0`
- contamination remains `0.00%` on all runs/branches.

### Deep-segment conclusions (`stage1/backward/steady/rank4..7`)

- Dominant segment remains `attn_core_sdpa_fmha_bwd`:
  - scaling_on:
    - `gap_ms median/IQR = 41.935 / 16.329`
    - `fmha_share median/IQR = 97.62% / 0.33%`
  - scaling_off:
    - `gap_ms median/IQR = 43.605 / 8.525`
    - `fmha_share median/IQR = 97.87% / 0.79%`
- top1 kernel is stable for both branches (`5/5`):
  - `fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80(...)`
- pre-fmha adjacency asymmetry remains:
  - distributed median `~1.901ms`
  - scaling median `~1.150ms` (on) / `~1.017ms` (off)

### Variability bucket (rank/state/iter coupling)

- On fmha segment paired windows (`n=240` per branch), residual is not uniform by iter:
  - `iter1` is the high-gap bucket:
    - on: `fmha_gap median ~1.766ms`
    - off: `fmha_gap median ~1.777ms`
  - `iter0/iter2` are near-zero buckets.
- `iter1` simultaneously shows stable pre-fmha adjacency shift:
  - `small_pre_count_gap_median = -1.0`
  - `small_pre_gap_ms_median ~ -0.05ms`

### Implication

- Current backward mismatch remains an attention runtime-context issue, not phase contamination.
- Freeze条件若继续用于 backward，需要显式加入“iter-bucket稳健性约束”（至少区分 iter1 与 iter0/2）或给出按-iter可解释口径。

## Stage-2 note (2026-03-01): NVTX attribution artifact is real and has deterministic code root cause

### What is confirmed

- Existing `seq8192 phase repeat-x5` sqlite traces show systematic CMD corruption pattern:
  - `open_forward=24`, `open_backward=0` for all runs/branches (`dist`, `scaling_on`, `scaling_off`);
  - `forward/backward overlap_cnt=48` in every run;
  - long-lived unclosed labels are dominated by `row_g_fwd`.
- This confirms part of backward residual analysis was polluted by NVTX stage-attribution artifact (op ownership is not trustworthy under leaked stack).

### Root cause (code-level)

- File: `megatron/core/tensor_parallel/mappings.py`
- Function: `_ReduceFromModelParallelRegion.forward`
- Bug pattern:
  - `nvtx.range_push(\"row_g_fwd\")` is executed;
  - `world_size==1` branch returns early;
  - `nvtx.range_pop()` is skipped.
- Because this path is hot in current TP=1 workloads, leaked pushes accumulate and shift subsequent pops, corrupting CMD window closure.

### Fix and status

- Landed minimal fix: wrap `row_g_fwd` section with `try/finally` to guarantee pop.
- Added unit tests to enforce NVTX balance for both `world_size==1` and `world_size>1`.
- RED→GREEN evidence is archived in:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_nvtx_attribution_artifact_validation_and_fix.md`

## Stage-2 note (2026-03-01): patched clean x1 passes NVTX-structure gate, but backward residual still high

### New structural gate (before compare/freeze)

- Added `tests/performance/check_nsys_nvtx_structural_health.py` for fail-fast trace hygiene checks:
  - `open_forward_step`
  - `open_backward_step`
  - `forward_backward_overlap_count`
- Default freeze precondition used in this round: all three metrics must be `0`.

### Patched seq8192 x1 outcome

- Captures executed on patched code:
  - distributed / scaling DDP-on / scaling DDP-off
  - `SEQ_LEN=8192`, `TRAIN_ITERS=3`, phase-pure config unchanged.
- Gate results:
  - all three branches pass with `open_forward=0`, `open_backward=0`, `overlap_count=0`.
- Analyzer results:
  - all three branches keep `contamination=0.00%`.
- Compare (`pure_primary_union`, shared primary-stream):
  - DDP-on: backward rank-median `17.42%`
  - DDP-off: backward rank-median `10.52%`
  - both remain above `<=5%`.

### Interpretation and priority decision

- NVTX structural artifact was a real blocker and is now cleaned in patched x1 captures.
- However, residual mismatch is not fully explained by structural attribution corruption alone.
- Next priority should move to attention-family diagnostics (debug-only tags/segmentation) before launching formal repeat-x5 freeze.

## Stage-2 note (2026-03-01): patched clean x1 attention-family deep diagnosis confirms `fmha_cutlassB` is still dominant

### Scope and method

- Analyzer script hardened:
  - `tests/performance/analyze_nsys_attention_family_delta.py`
  - strict numeric pairing by `rank+iter+batch`;
  - fmha duration uses phase-window overlap only;
  - added per-rank fmha IQR output.
- Scope fixed to:
  - `op=backward_step`, `state=steady`, `stage=1`, `phase=compute`, `rank=4..7`;
  - compare pairs: `dist vs scaling_on`, `dist vs scaling_off`.

### Key evidence

- Both comparisons have full pairing (`paired_windows=12`, `missing=0`).
- Gap decomposition:
  - scaling_on: `gap=+42.122 ms`, `fmha_gap=+25.442 ms`, fmha share `60.40%`;
  - scaling_off: `gap=+28.344 ms`, `fmha_gap=+20.650 ms`, fmha share `72.86%`.
- Launch config parity is exact in both cases:
  - `dist_unique_cfg=1`, `scale_unique_cfg=1`, `cfg_sets_equal=True`.
- Top-1 absolute kernel delta remains:
  - `fmha_cutlassB ...` in both on/off reports.

### Interpretation

- DDP-off can reduce total backward gap, but attention-family dominance remains unchanged.
- Because launch shape is identical, next diagnosis should target attention-path runtime context (e.g., neighboring memory traffic / stream scheduling / pre-attention adjacency), not kernel-shape mismatch.

## Stage-2 note (2026-03-01): attention backward micro-segmentation confirms dominant residual is localized in `attn_core_bwd` (MLA path included)

### Instrumentation landed

- New debug-only trace switch:
  - `--trace-attention-backward-segments`
- NVTX label enhancement:
  - phase labels now support extra tag `attn_bwd_segment=*`.
- Coverage:
  - `SelfAttention` and `MLASelfAttention` both instrumented.
  - MLA path adds `_MLASDPACoreAttention` module so SDPA core backward can be independently tagged (`attn_core_bwd`).

### Clean x1 (seq8192) localization

- Segment labels are present in sqlite and pass NVTX structure gate.
- In `stage1 backward steady rank4..7`:
  - only `attn_core_bwd` has material gap;
  - `fmha_gap_share` is ~`98%`;
  - non-core segments (`attn_proj/qkv/qk_ln`) are near-zero.

### Repeat-x5 robustness

- Protocol: `seq8192`, distributed + scaling_on/off, phase-pure unchanged.
- Core findings stay stable across 5 runs:
  - `attn_core_bwd` top1 delta is always `fmha_cutlassB`;
  - core-gap fmha share median:
    - scaling_on: `98.01%`
    - scaling_off: `97.96%`.
- Stream diagnostics:
  - no primary-stream or fmha-stream set mismatch;
  - distributed side shows consistently higher pre-fmha small-kernel adjacency in core segment.

### Interpretation

- Residual is now localized to attention core runtime context rather than comm leakage or launch-shape mismatch.
- Next refinement should target pre-fmha neighborhood and attention-core adjacency behavior, not comm-adjacent emulation expansion.

## Stage-2 note (2026-03-01): attention-core subsegment x1 confirms residual is specifically in `attn_core_sdpa_bwd`

### What was added

- MLA core was split into debug-only backward subsegments:
  - `attn_core_precast_bwd`
  - `attn_core_sdpa_bwd`
  - `attn_core_postcast_bwd`
- Existing coarse segment `attn_core_bwd` remains unchanged for compatibility.
- Attention-family analyzer now outputs adjacent small-kernel top names (`name/ms/count`) before and after fmha.

### x1 evidence (`seq8192`, `stage1 backward steady rank4..7`)

- Structural and contamination gates remain clean in all branches:
  - `open_forward=0`, `open_backward=0`, overlap `0`
  - contamination `0.00%`
- Segment localization:
  - scaling_on:
    - `attn_core_bwd gap=23.419ms`
    - `attn_core_sdpa_bwd gap=23.115ms`, `fmha_share=98.02%`
    - `attn_core_precast_bwd gap=0.000ms`
    - `attn_core_postcast_bwd gap=0.000ms`
  - scaling_off:
    - `attn_core_bwd gap=34.155ms`
    - `attn_core_sdpa_bwd gap=33.740ms`, `fmha_share=98.11%`
    - `attn_core_precast_bwd gap=0.000ms`
    - `attn_core_postcast_bwd gap=0.000ms`

### New adjacency top-name evidence

- In `attn_core_sdpa_bwd`, pre-fmha top adjacent small kernel is stable:
  - `vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`
- Dist vs scale pre-fmha adjacency:
  - scaling_on: dist `35 / 1.751ms` vs scale `27 / 1.393ms`
  - scaling_off: dist `35 / 1.751ms` vs scale `18 / 0.900ms`

### Interpretation

- Residual is not spread across all core subphases; it is concentrated in SDPA-backward runtime context.
- Pre/post cast are non-material under this workload.
- The next high-value diagnosis should stay inside SDPA neighborhood (mask/build + adjacent memory traffic), not in comm-adjacent emulation.

## Stage-2 note (2026-03-01): immediate same-stream SDPA-neighbor sweep shows threshold sensitivity, not path-missing asymmetry

### What was added

- `tests/performance/analyze_nsys_attention_family_delta.py` now emits immediate same-stream fmha-neighbor fields:
  - `small_kernel_immediate_pre/post_count`
  - `small_kernel_immediate_pre/post_ms`
  - immediate pre/post top names (`name/ms/count`)
- `tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py` adds:
  - immediate-neighbor schema assertions
  - synthetic positive-case test for immediate pre/post detection.

### x1 evidence (`stage1 backward steady rank4..7`, segment=`attn_core_sdpa_bwd`)

- Default threshold (`small_kernel_threshold_us=60`):
  - scaling_on immediate pre: dist `35/1.751ms` vs scale `27/1.393ms`
  - scaling_off immediate pre: dist `35/1.751ms` vs scale `18/0.900ms`
- Sweep threshold (`small_kernel_threshold_us=80`):
  - scaling_on immediate pre: dist `44/2.311ms` vs scale `44/2.451ms`
  - scaling_off immediate pre: dist `44/2.311ms` vs scale `44/2.518ms`
- Immediate pre top1 kernel is unchanged in all branches:
  - `vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`
- Core gap does not move with threshold sweep:
  - scaling_on `gap_ms=23.115`
  - scaling_off `gap_ms=33.740`

### Interpretation

- SDPA pre-fmha neighbor kernel path exists on both branches and same stream.
- Count asymmetry at `60us` is mostly classification-threshold sensitivity, not a missing-kernel path in scaling.
- Dominant residual remains inside fmha runtime context (still within `attn_core_sdpa_bwd`), so next diagnosis should target fmha runtime-context factors instead of extending comm-adjacent emulation.

## Stage-2 note (2026-03-01): attention-core SDPA subsegment repeat-x5 confirms stable top1 source with persistent variability

### Scope and protocol

- Formal repeat-x5 executed with unchanged phase-pure settings on `seq8192`:
  - branches: distributed + scaling_on + scaling_off;
  - tracing: `TRACE_KERNEL_GROUND_TRUTH=1`, `TRACE_KERNEL_GROUND_TRUTH_PHASE=1`, `TRACE_KERNEL_BOUNDARY_SYNC_MODE=event`, `TRACE_ATTENTION_BACKWARD_SEGMENTS=1`.
- Per-run artifacts include:
  - NVTX structure gate logs
  - phase breakdown json/md
  - compare logs
  - `attn_core_sdpa_bwd` diagnosis json/md.

### Robust evidence (`stage1 backward steady rank4..7`, segment=`attn_core_sdpa_bwd`)

- contamination remains clean in all runs/branches (`0.00%`).
- `scaling_on` robust stats:
  - `gap_ms median/IQR = 22.515 / 7.021`
  - `fmha_gap_ms median/IQR = 22.012 / 6.898`
  - `fmha_gap_share_pct median/IQR = 98.022 / 0.283`
- `scaling_off` robust stats:
  - `gap_ms median/IQR = 23.725 / 20.024`
  - `fmha_gap_ms median/IQR = 23.263 / 19.963`
  - `fmha_gap_share_pct median/IQR = 98.041 / 0.703`
- top1 kernel stability:
  - `fmha_cutlassB...` is top1 delta in `5/5` runs for both on/off.
- pre-fmha adjacency stability:
  - top-name is always `FillFunctor<unsigned char>` on both branches;
  - distributed adjacent load remains higher (`count/ms` median: `35/1.751` vs scale `21/~1.05`).

### Interpretation

- SDPA subsegment定位在 repeat-x5 上已稳健，不是单轮伪影。
- backward residual 的主导来源仍是 fmha runtime-context，而非 comm leakage。
- 但 run-level 波动仍显著（尤其 scaling_off 的 `gap_ms` IQR 较高），说明官方 freeze 仍需更细 SDPA 邻域诊断才能继续收敛。

## Stage-2 note (2026-03-01): phase-level pure compute-only semantics (implementation landed)

### Official backward comp-only semantics (frozen definition for next validation round)

- Parent CMD NVTX remains the coarse op window.
- New phase-level semantics are defined as:
  - `phase=compute`: only backward core compute region
  - `phase=comm`: recv/send/allreduce/all_to_all and decorated comm sub-ops
- Official `comp_only` candidate for NSYS comparison:
  - primary metric: `compute_pure_primary_union_ms`
  - secondary robustness metric: `compute_pure_union_ms`
- contamination diagnostic:
  - `contamination_ms = compute_kernel_ms - compute_pure_ms`
  - `contamination_pct = contamination_ms / compute_kernel_ms * 100`

### Implementation coverage

- Runtime tracing:
  - `megatron/profiler/cmd.py`:
    - nested phase NVTX APIs (`phase_range("compute"/"comm")`)
    - optional phase-boundary sync policy (`none|event|global`)
    - comm decorators now auto-wrap comm calls with `phase=comm`
  - `megatron/core/pipeline_parallel/schedules.py`:
    - all traced distributed backward bodies wrapped with `phase=compute`
  - `megatron/training/training.py`:
    - scaling-mode backward body wrapped with `phase=compute`
- CLI:
  - `--trace-kernel-ground-truth-phase`
  - `--trace-kernel-boundary-sync-mode`
- Scripts:
  - DeepSeek and Qwen example scripts now expose and validate phase tracing env/args.

### Post-analysis behavior

- `tests/performance/analyze_nsys_cmd_kernel_breakdown.py` now outputs pure metrics + contamination.
- `tests/performance/compare_qwen_nsys_compute_only.py` now supports:
  - `--compute-metric pure_primary_union` (default)
  - `--require-low-contamination-pct`
- If contamination gating is enabled but JSON lacks contamination fields, compare fails fast.

### Sanity x1 evidence (2026-03-01 rerun)

- New phase-labeled NSYS captures were completed in both modes (`SEQ_LEN=1024`, smoke):
  - distributed: `deepseek_phase_sanity_dist_rerun.nsys-rep`
  - scaling: `deepseek_phase_sanity_scaling.nsys-rep`
- Analyzer evidence:
  - both sides report `phase_window_parents=48`, `event_rows=72`, `aggregate_rows=24`;
  - both sides report `contamination_pct=0.00` on event/aggregate rows.
- Compare evidence (`pure_primary_union + contamination gate`):
  - contamination gate passes for all rows (`dist_contam_pct=0.00`, `scale_contam_pct=0.00`);
  - op-rank-median still fails threshold in x1 sanity (`forward=10.25%`, `backward=17.97%`, `optimizer=5.75%`).
- Interpretation:
  - comm contamination is no longer a plausible explanation for current residual in this run;
  - remaining gap is distributed-vs-scaling compute mismatch under current workload, and freeze decision must rely on fixed-protocol repeat-x5 (`seq8192`, rank7-cap), not this single sanity run.

### Postfix all_to_all attribution-fix follow-up (2026-03-01)

- Hypothesis tested:
  - backward residual may come from comm-adjacent kernels leaking into compute windows through all_to_all bypass behavior.
- Postfix landed and validated:
  - `mappings.py` now executes `input_.contiguous()` inside `_profiled_all_to_all_single` (comm wrapper scope);
  - scaling bypass now materializes output copy for `output_split_sizes=None` and equal-row split cases (no direct alias return).
- Functional validation:
  - `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py` adds coverage for:
    - scaling equal-row materialization,
    - scaling none-split materialization,
    - non-scaling contiguous input contract for `all_to_all_single`.
  - unit/static checks pass.
  - local autograd micro-repro (monkeypatched `all_to_all_single`) shows identical graph node counts for `_AllToAll.apply`:
    - `is_scaling_mode=False`: `node_count=3`, node types `['MulBackward0', 'SumBackward0', '_AllToAllBackward']`
    - `is_scaling_mode=True`: `node_count=3`, node types `['MulBackward0', 'SumBackward0', '_AllToAllBackward']`
- Postfix x1 NSYS evidence:
  - phase metrics remain clean (`phase_window_parents=48`, contamination `0.00%` for dist/scale);
  - compare (`pure_primary_union`, contamination gate) remains above threshold:
    - op-rank median: `forward=12.25%`, `backward=19.72%`, `optimizer=5.32%`.
  - pre/post (same x1 protocol) shows no material backward improvement:
    - backward median `17.97% -> 19.72%`,
    - stage1 backward (`rank4..7`, steady) pair-median `21.62% -> 21.73%`.
- Interpretation:
  - this postfix is correctness/attribution hardening, but not the dominant residual root-cause fix.

### Backward residual source deep-dive + scaling comm-adjacent emulation feasibility (2026-03-01)

- Source attribution on known high-gap dataset (`round68 run5`, `seq8192`, stage1 backward ranks 4-7):
  - `dist_primary_union_total=575.107 ms`, `scale_primary_union_total=336.820 ms`, gap `=238.288 ms`;
  - kernel positive-delta decomposition:
    - comm-adjacent/data-movement classified share: `70.559 ms` (`27.31%`);
    - dominant single-family delta: `fmha_cutlassB` (`149.201 ms`);
    - large grouped-GEMM kernels (`256x128 64x3`) remain near-equal.
- Interpretation:
  - comm-adjacent cost exists and is non-trivial, but it is not the dominant source in this high-gap case;
  - backward residual currently includes a larger non-comm compute component.
- Experimental scaling-side补齐方案（default-off）:
  - added `--scaling-comm-adjacent-copy-iters` and threaded into scaling all_to_all backward path.
  - deepseek/qwen scripts now accept `SCALING_COMM_ADJACENT_COPY_ITERS`.
- x1 feasibility outcome (`SEQ_LEN=1024`, phase-pure compare):
  - copy0: `forward=12.25%`, `backward=19.72%`, `optimizer=5.32%`
  - copy2: `forward=12.27%`, `backward=19.56%`, `optimizer=4.86%`
  - copy8: `forward=12.26%`, `backward=19.67%`, `optimizer=3.95%`
- Conclusion:
  - scaling补齐方案技术上可行，但在当前 x1 设置下对 backward residual 的改善幅度很小，无法单独完成语义收敛。

### Cross-run source validation + `_AllToAll` graph parity (2026-03-01)

- Validation scope:
  - historical `round68 seq8192` NSYS kernel-breakdown `run1..5`;
  - filtered slice: `stage1 backward steady`, ranks `4..7`.
- Cross-run stability evidence:
  - `dist-scale` primary-stream gap remains stable (`~225.6 ms` to `~240.6 ms`);
  - top-1 positive delta is consistently `fmha_cutlassB...` in every run;
  - contribution range for this family: `146.581..153.787 ms`.
- `_AllToAll` micro-repro evidence:
  - both distributed/scaling branches produce the same autograd node set and count:
    - `['MulBackward0', 'SumBackward0', '_AllToAllBackward']` (3 nodes).
- Interpretation update:
  - “distributed backward has larger autograd graph because scaling bypass removes all_to_all backward nodes” is not supported at micro level;
  - current residual should continue to be treated as **non-comm dominant** under this workload, with comm-adjacent still a secondary contributor.

### Scaling DDP-hook hypothesis validation (2026-03-01)

- Hypothesis under test:
  - backward residual is mainly due to DDP backward-hook overhead mismatch;
  - scaling may be missing/suppressing this hook cost.
- Implemented debug path:
  - added `--scaling-disable-ddp-wrap` (debug-only);
  - for runtime compatibility, DDP wrapper remains constructed, but DDP param-hook accumulation path is disabled in scaling debug mode.
- x1 NSYS A/B (`SEQ_LEN=1024`, phase-pure, same protocol):
  - stage1 backward steady (`rank4..7`) aggregated:
    - `compute_pure_primary_union_ms`: `78.899 -> 71.230` (`-9.72%`);
    - `kernel_count`: `7576 -> 6988` (`-7.76%`).
  - main reduced kernel family:
    - `CUDAFunctor_add<float>`: `-7.597 ms`.
  - attention sensitivity in this A/B:
    - `fmha_cutlassB` delta is negligible (`-0.007 ms`).
- Compare impact:
  - dist-vs-scale backward rank-median does not improve in this protocol (`19.72% -> 20.08%`).
- Interpretation:
  - scaling path does include DDP hook-related compute, so “scaling hook is effectively absent” is not supported;
  - DDP hook overhead exists but is not dominant enough to explain major residual alone;
  - non-comm dominant root-cause direction remains unchanged.

### Seq8192 phase-pure DDP probe A/B follow-up (2026-03-01)

- Scope:
  - completed pending `seq8192` distributed NSYS probe capture and added scaling `DDP on/off` A/B captures under the same phase-pure protocol.
  - datasets:
    - `deepseek_phase_sl8192_dist_ddp_probe`
    - `deepseek_phase_sl8192_scaling_ddp_on`
    - `deepseek_phase_sl8192_scaling_ddp_off`

- Analyzer sanity:
  - all three sets have `phase_window_parents=48`, `event_rows=72`, `aggregate_rows=24`;
  - contamination remains `0.00%` end-to-end, so this residual is not comm-window leakage.

- Compare results (`pure_primary_union`, `shared(primary_stream)`, contamination gate):
  - DDP-on rank-median: `forward=7.20%`, `backward=16.88%`, `optimizer=5.45%`;
  - DDP-off rank-median: `forward=4.91%`, `backward=9.00%`, `optimizer=1.96%`.

- Stage1 backward (`rank4..7`, steady) focused deltas:
  - DDP-on median diff: `8.83%`;
  - DDP-off median diff: `6.04%`;
  - improvement is real but still above freeze criterion (`<=5%`).

- Kernel-family attribution (stage1 backward steady, ranks 4..7):
  - largest dist-vs-scale delta remains `fmha_cutlassB` in both A/B branches:
    - DDP-on: `+40.965 ms`
    - DDP-off: `+32.653 ms`
  - implication: DDP hook is a secondary contributor; dominant residual still tracks attention-family non-comm kernels.

- Practical interpretation:
  - this seq8192 probe confirms DDP hook controls part of the gap but cannot alone freeze backward semantics;
  - compared with historical `round68 seq8192 run1..5` high-gap set, this x1 probe shows direction drift (`scale > dist`), so single-run direction cannot be treated as final root-cause verdict;
  - next official action remains fixed-protocol `rank7-cap + repeat x5` validation on phase-pure metric.

### Seq8192 phase-pure formal repeat-x5 freeze round (2026-03-01)

- Protocol:
  - kept phase-pure settings unchanged (`TRACE_KERNEL_GROUND_TRUTH_PHASE=1`, boundary `event`, `SEQ_LEN=8192`);
  - executed 5-run protocol for:
    - distributed,
    - scaling DDP-on (`SCALING_DISABLE_DDP_WRAP=0`),
    - scaling DDP-off (`SCALING_DISABLE_DDP_WRAP=1`).

- Semantics integrity:
  - contamination remains `0.00%` for all runs and all branches, confirming phase-level compute-only windows are clean.

- Robust compare outcome (`pure_primary_union`, shared primary-stream):
  - DDP-on median/IQR:
    - forward `6.28 / 1.97`
    - backward `13.02 / 4.23`
    - optimizer `2.14 / 1.17`
  - DDP-off median/IQR:
    - forward `5.77 / 1.38`
    - backward `12.27 / 5.26`
    - optimizer `3.71 / 1.40`
  - interpretation:
    - DDP-off only partially improves backward median (`13.02 -> 12.27`) and does not improve backward stability;
    - backward remains clearly above freeze threshold (`<=5%`).

- Mode sweep robustness:
  - `pure_union` and `pure_primary_union` give identical op-rank medians in all 5 runs (drift `0.00pp`), so metric-mode selection is not the blocker here.

- Kernel-family robust evidence (`stage1 backward steady`, rank4..7):
  - `fmha_cutlassB` is top1 by abs delta in **5/5 runs** for both DDP branches;
  - fmha delta median/IQR:
    - DDP-on: `38.447 / 13.370 ms`
    - DDP-off: `30.666 / 7.770 ms`
  - non-fmha families stay at much smaller low-single-digit-ms medians.

- Updated direction:
  - repeat-x5 confirms DDP hook path is secondary and cannot close backward residual by itself;
  - next priority should move to attention-family diagnostics (debug-only tags/segmentation), instead of继续扩展 comm-adjacent emulation knobs.

## Stage-2 note (2026-02-28): measurement-regime change on Round6-8 baseline

### What was changed (and what was not)

- Changed only runtime/script parameters; no code edit.
- Kept compare protocol fixed:
  - rank7 end-of-run `pair_timestamp` cap,
  - repeat `x5`,
  - `--distributed-subtract-comm`,
  - `op_rank_median_aux_summary`.

### OOM boundary findings

- `MODEL_PROFILE=full` (61L/7168H) is not feasible in current environment:
  - distributed OOM at `SEQ_LEN=256/192/128/96`.
- `MODEL_PROFILE=smoke` remains feasible up to `SEQ_LEN=8192`:
  - distributed PASS,
  - scaling PASS (validated for seq8192).

### Noise-floor findings at max-feasible smoke (`SEQ_LEN=8192`, repeat x5)

- median-of-runs:
  - `forward=2.00%`
  - `backward=62.73%`
  - `optimizer=6.72%`
- spread:
  - range: `forward=3.23%`, `backward=34.41%`, `optimizer=4.35%`
  - IQR: `forward=1.54%`, `backward=6.84%`, `optimizer=2.39%`

### Interpretation update

- Increasing workload size **does** suppress forward noise and improves forward median.
- But <=5% gate remains unattained because:
  - optimizer median stays above gate (`6.72%`),
  - backward shows severe systematic inflation under current `distributed_subtract_comm` semantics at long sequence.
- Conclusion for next action:
  - do not start new code-level single-variable hypotheses yet;
  - prioritize measurement semantics validation for backward while preserving rank7-cap + repeat-x5 discipline.

## Stage-2 note (2026-02-28): backward measurement semantics validation (same seq8192 batch)

### Validation setup

- Same five rank7-capped pairs were reused (no new training-code changes).
- Parallel views were produced:
  - baseline subtract (`alpha=1.0`);
  - op-level compute-only auxiliary (`forward=0.787`, `backward=0.176`);
  - stage-aware compute-only auxiliary (`forward@stage1=0.787`, `backward@stage1=0.107`);
  - no-subtract total-time control.

### Robust summary (median-of-runs)

- baseline subtract: `2.00 / 62.73 / 6.72` (fwd/bwd/opt)
- op-map subtract: `1.74 / 5.85 / 6.72`
- stage-aware subtract: `1.74 / 3.86 / 6.72`
- no-subtract total: `9.32 / 7.74 / 6.72`

### Interpretation

- Backward inflation in baseline is dominated by subtraction semantics, not by model-code mismatch:
  - overlap-heavy stage1 has large distributed comm buckets (~40ms+), and full subtraction makes `dist_comp` unrealistically low.
- Stage-aware/op-map auxiliary views recover stable backward behavior (range shrinks from `34.41%` to `<=1.81%`).
- Optimizer remains the only persistent residual above gate (`median 6.72%`), so post-semantics code A/B should focus optimizer first.

## Stage-2 note (2026-02-28): NSYS repeat-x5 tri-view follow-up (seq8192, round68 baseline)

### What was executed

- On the same Round6-8 baseline and fixed protocol discipline (rank7 cap + repeat x5), ran:
  1. distributed/scaling NSYS captures with kernel-ground-truth NVTX labels;
  2. trace compare in two diagnostic views (`subtract` / `no-subtract`);
  3. NSYS compute-only compare (`primary_stream_union + shared(primary_stream)`).
- No model/training code path changes; only experiment-script passthrough to forward kernel-ground-truth args.

### Tri-view median-of-runs

- trace subtract: `forward=2.84%`, `backward=35.21%`, `optimizer=11.20%`
- trace no-subtract: `forward=6.79%`, `backward=10.40%`, `optimizer=11.20%`
- NSYS compute-only: `forward=0.33%`, `backward=38.30%`, `optimizer=0.81%`

### Stability (backward spread)

- subtract: `range=12.59%`, `IQR=7.08%`
- no-subtract: `range=1.09%`, `IQR=0.63%`
- NSYS compute-only: `range=3.26%`, `IQR=0.43%`

### Interpretation update

- “stage-aware subtract 不可扩展”判断保持成立：它只能作为诊断，不适合官方 gate。
- backward 官方口径仍未冻结：
  - subtract 与 no-subtract 差异过大，说明 subtraction 语义风险仍显著；
  - NSYS compute-only 在当前模式下虽然稳定，但 backward 绝对误差长期停留在 `~38%`，与 no-subtract 诊断视图不一致。
- Next: 需要继续做**无标定**语义收敛（例如测量边界纯化/同步语义对照），再恢复代码级单变量 A/B。

## 1) Mixtral vs Qwen3 vs DeepSeek-V3-Proxy

- **Mixtral (current baseline)**
  - All transformer layers are MoE in current usage.
  - Router path: Mixtral-style top-k + aux_loss (existing code path).
  - Standard MHA/GQA + RMSNorm + SwiGLU.
- **Qwen3-30B-A3B (stage-1 target)**
  - MoE model requiring explicit `moe_ffn_hidden_size` and `rotary_base` support.
  - Router can be kept in Mixtral-compatible top-k + aux_loss in stage-1.
  - All layers can be treated as MoE (`moe_layer_freq=1`) for stage-1.
- **DeepSeek-V3-Proxy (stage-1 target)**
  - Mixed dense/MoE layer schedule (proxy: 3 dense + 11 MoE).
  - Full upstream stack includes MLA, seq_aux_loss routing variants, shared experts, and optional MTP.
  - Stage-1 keeps MHA simplification and Mixtral-style router semantics for tracing correctness.

## 2) Missing modules / params in legacy codebase

### Missing or not yet ported modules (from latest reference)

- `multi_latent_attention.py` (MLA)
- `moe/shared_experts.py`
- `moe/router_replay.py`
- `moe/fused_a2a.py`
- `multi_token_prediction.py`
- DeepSeek-specific router variants (`seq_aux_loss`, score/group/bias semantics)
- `flex` dispatcher + `deepep` integration

### Missing first-class config/CLI support before stage-1 patch

- `--moe-layer-freq`
- `--moe-ffn-hidden-size`
- `--rotary-base`
- `TransformerConfig.moe_layer_freq`
- `TransformerConfig.moe_ffn_hidden_size`
- `TransformerConfig.rotary_base`

### Baseline layer-spec gap

- Old `gpt_layer_specs.py` only produced a single layer spec repeated by block.
- No native dense/MoE mixed pattern generation by layer index.

## 3) Stage-1 simplifications and impact

- **DeepSeek attention simplification**: MLA -> MHA
  - Impact: architecture fidelity reduced for DeepSeek-specific attention path.
  - Benefit: keeps tracing/scaling integration stable and runnable in current environment.
- **Router semantics simplification**: keep Mixtral-style top-k + aux_loss
  - Impact: no DeepSeek-specific router behavior (seq_aux_loss/group-topk/bias).
  - Benefit: distributed/scaling route-control path remains deterministic and debuggable.
- **Checkpoint path omitted**: random/mock initialization only
  - Impact: no direct training continuation from official checkpoints.
  - Benefit: avoids blocker dependencies and allows fast tracing verification.
- **Smoke profiles in scripts**
  - Impact: reduced model size defaults for local validation throughput.
  - Benefit: practical stage-1 verification with mock data and short iterations.

## 4) Stage-2 backlog

1. Add full DeepSeek MLA path (`multi_latent_attention` stack).
2. Add DeepSeek router semantics (`seq_aux_loss`, sigmoid/group-topk/expert-bias).
3. Add shared experts support.
4. Add optional MTP branch.
5. Add `flex`/`deepep` dispatcher integration.
6. Implement checkpoint conversion/loading (HF -> Megatron).

## Environment constraints identified

- `deepep` is not installed in current environment.
- `transformer-engine` / `triton` versions are older than latest DeepSeek advanced feature expectations.

# Stage-2 Notes: DeepSeek-V3（架构标准）对齐与关键实现决策

## 1) Upstream YAML 对齐清单（只取架构标准所需）

对齐来源：`latest-megatron/Megatron-MoE-ModelZoo/model_configs/benchmarking/DeepSeek-V3.yaml`

架构标准必须覆盖的 flags（本 repo 需新增/扩展）：

- MLA:
  - `--multi-latent-attention`
  - `--q-lora-rank`, `--kv-lora-rank`
  - `--qk-head-dim`, `--qk-pos-emb-head-dim`, `--v-head-dim`
- YaRN RoPE:
  - `--rotary-scaling-factor`
  - `--mscale`, `--mscale-all-dim`
  - `--original-max-position-embeddings`（如本 repo 不提供则用 `--max-position-embeddings` 作为默认来源，但需要显式规则并写入 config 验证）
- MoE + router semantics:
  - `--moe-router-load-balancing-type seq_aux_loss`
  - `--moe-router-num-groups`, `--moe-router-group-topk`
  - `--moe-router-score-function sigmoid`
  - `--moe-router-topk-scaling-factor`
  - `--moe-router-enable-expert-bias`, `--moe-router-bias-update-rate`
  - `--moe-router-dtype fp32`
- shared experts:
  - `--moe-shared-expert-intermediate-size`

明确 out-of-scope（Stage-2 不实现，避免误导/隐藏问题）：
- `--moe-router-fusion`, `--moe-permute-fusion`（依赖 TE 版本与 fused kernels）
- DeepEP/flex dispatcher
- checkpoint load/save/convert
- MTP

## 2) 环境约束与实现策略（关键决策已锁定）

当前环境事实：
- Transformer Engine: 1.3.0（不具备 upstream DeepSeek-V3 常用的 TE>=2.6 fused MLA/router 支撑）

因此 Stage-2 策略：
- MLA attention core：使用 PyTorch `scaled_dot_product_attention`（SDPA）
- YaRN RoPE：移植 upstream 数学实现（不走 fused apply）
- router：实现 seq_aux_loss/group-limited/sigmoid/scaling_factor/expert_bias（全部走 torch 实现）

原则：
- **Fail fast**：遇到不支持组合直接 `raise`，不做 silent fallback（例如不允许在传了 `--moe-router-fusion` 的情况下悄悄退化到 unfused）

## 3) 关键语义点（必须在实现中显式处理）

### 3.1 Attention mask 语义（SDPA vs Megatron）

本 repo dataloader/训练 utils 的 `attention_mask`：
- bool mask：`True = masked`，`False = allowed`

PyTorch SDPA 的 bool `attn_mask` 语义：
- `True = allowed`，`False = masked`

因此 MLA 的 SDPA 调用必须做一次显式取反：
- `sdpa_mask = ~attention_mask`

### 3.2 MLA head dims（DeepSeek-V3）

- Q/K dot-product 维度：`q_head_dim = qk_head_dim + qk_pos_emb_head_dim`
- V 维度：`v_head_dim`
- SDPA 支持 Q/K/V 最后一维不同（已验证 PyTorch 2.1.2 可行）

### 3.3 Router fixed routing（本 fork 特有）

当前 `pretrain_llama.py` 在 EP>1 时会注入 `config.pre_fixed_routing_results`，`moe_layer.py` 会使用固定 indices 重新从 logits 计算 scores 来保持图连通。

Stage-2 必须保证该 fixed-routing 分支与新 router 语义一致：
- 当 score_function=sigmoid、存在 topk_scaling_factor、启用 seq_aux_loss 等时不能“只 softmax top_logits”完事
- 否则会在 distributed/scaling 路径上产生语义分叉

建议实现策略：
- 将“从 logits + fixed indices 计算 probs/scores”的逻辑收敛到 router helper（单一真源），MoELayer 调用它。

### 3.4 expert bias 的更新时机

upstream 里 expert bias 的更新通常发生在 global batch 粒度，需要 allreduce tokens_per_expert。
本 repo 对齐点：
- 在 `finalize_model_grads.py` 的训练收尾路径里做 bias 更新
- scaling mode world size=1 时 allreduce 应为 no-op，但代码路径不能 crash

## 4) 需要在文档中持续声明的限制（避免 paper 误导）

- 本阶段不以 comp<=5% 为硬门禁（现有 compare 波动风险已记录在 issues.md）
- 只保证“架构标准跑通 + trace 落盘 + 可复现命令”

## 5) Stage-2 执行期补充结论（2026-02-27）

1. **Shared experts 语义补齐**
   - 增加了 `moe_shared_expert_gate`（CLI + `TransformerConfig` + `SharedExpertMLP`）：
     - shared expert 输出可按 DeepSeek 语义走 `sigmoid(linear(hidden_states))` gate。
   - 默认保持关闭，不影响已有 model 路径。

2. **脚本层 fail-fast 与可复现实验增强**
   - `examples/pretrain_deepseek_v3_moe.sh` 增加短跑保护：
     - 自动保证 `LR_WARMUP_ITERS < TRAIN_ITERS`，避免 `OptimizerParamScheduler` 断言失败。
   - 增加诊断开关（仅新脚本生效，不改已有脚本）：
     - `MOE_SHARED_EXPERT_GATE`
     - `MOE_ROUTER_TOPK_SCALING_FACTOR`
     - `USE_BF16`
     - `MOE_GROUPED_GEMM`

3. **分布式 NaN 现象的最新定位边界**
   - `PP=2, EP=2, bf16`（架构标准 smoke）在 distributed 下稳定触发：
     - last pipeline stage ranks (`4..7`) 出现 forward loss NaN。
   - `PP=2, EP=2` 时，即使关闭 `MOE_SHARED_EXPERT_GATE=0`，NaN 仍在。
   - `PP=1, EP=1` distributed/scaling 双模式可稳定完成并落盘 traces。
   - 推断：当前 blocker 更接近 **PP>1 + bf16 路径的数值/执行一致性问题**，而非 shared-expert gate 单点问题。

4. **PP2 bf16 NaN 的 round2 根因与修复（已验证）**
   - 根因定位证据（以 `PP=2, EP=1` 诊断配置先收敛问题空间）：
     - `PP=1` 下 distributed PASS；`PP=2` 下 bf16 FAIL，说明问题与 pipeline forward p2p 路径相关。
     - 打开诊断后首个 non-finite 出现在 last PP stage 的 decoder input（非 loss 端二次传播症状）。
     - fp32（`USE_BF16=0`）下同配置可过，说明属于 bf16 数值/传输一致性问题。
   - 代码修复（最小范围）：
     - `megatron/core/pipeline_parallel/p2p_communication.py`：
       - 新增 `_align_forward_tensor_dtype(...)`，在 `send_forward*` 路径对 forward activation 做 `pipeline_dtype` 对齐。
       - 该路径只在 `config.multi_latent_attention=True` 时启用，避免影响已有非-MLA模型路径。
     - `megatron/core/transformer/moe/moe_utils.py` + `router.py`：
       - 对 sigmoid routing 的归一化改为 fp32 安全归一化并 `clamp` 分母，避免极端 underflow 下 `0/0` 风险。
   - 修复后结果：
     - target distributed smoke（`PP=2,EP=2,bf16`）PASS 且无 NaN assertion。
     - target scaling smoke 仍 PASS，rank `0..7` trace 覆盖完整。

## 6) Stage-2 fidelity round3（2026-02-27）

1. **测量口径回归稳定默认**
   - `examples/pretrain_deepseek_v3_moe.sh` 的 `TRACE_CMD_SYNC_MODE` 默认值从 `event` 回退为 `global`。
   - 原因：event 模式在真实跑测中出现明显离群（单次 forward/backward 可出现百毫秒级尖峰），不适合作为默认 fidelity 采样模式。

2. **optimizer_step timing boundary 对齐（scaling vs distributed）**
   - `megatron/training/training.py`：
     - 新增 `_prepare_scaling_optimizer_step(...)`，将 scaling 路径中的 optimizer prefetch（`get_parameters` / `get_main_grads_for_grad_norm`）移到 traced `optimizer_step` CMD 之外。
   - 目的：与 distributed `train_step` 的 timing 边界一致，避免 scaling 侧多计入 prefetch 耗时。

3. **最新证据与结论**
   - 在 `trace_start=4, train_iters=6, scaling_warmup=3, scaling_profile=3` 下重跑 distributed + scaling（两次 pass）后：
     - 最佳配对之一（`pair=20260227141611`）：
       - `forward_step` median `3.83%`（PASS）
       - `backward_step` median `11.09%`（FAIL）
       - `optimizer_step` median `7.84%`（FAIL）
     - 另一配对（`pair=20260227141950`）：
       - `forward_step` median `7.58%`（FAIL）
       - `backward_step` median `9.86%`（FAIL）
       - `optimizer_step` median `10.54%`（FAIL）
   - 结论：
     - forward 已可在部分稳定 pair 达到阈值；
     - backward 仍对 distributed comm subtraction 高敏感（阶段/运行间波动大）；
     - optimizer 仍存在 scaling 系统性偏高（约 +8%~+12% 中位数）残差。

## 7) Stage-2 fidelity round4（2026-02-27）

1. **scaling optimizer pre-CMD side effects 继续对齐**
   - `megatron/training/training.py`：
     - `_prepare_scaling_optimizer_step(...)` 在 existing prefetch 之外，新增与 distributed `train_step` 一致的 `numel` pre-scan：
       - `sum(param.numel() for param in params)`
       - `sum(grad.numel() for grad in grads_for_norm)`
   - 目标：复制 distributed 侧 optimizer 进入 CMD 之前的完整准备副作用，进一步压缩 optimizer 差距。

2. **定向证据（rank0 optimizer）**
   - paired to distributed `rank0@20260227141950`：
     - before patch（`scaling rank0@20260227142506`）：`optimizer_step diff = 12.76%`
     - after patch（`scaling rank0@20260227144128`）：`optimizer_step diff = 6.02%`
   - 结论：该对齐改动有效降低 optimizer 偏差，但仍略高于 5% 门限。

3. **全量配对新结论**
   - 使用 interleaved two-pass scaling（`cache_tag=stage2_fidelityfix5_interleave`）+ distributed rerun（`ts=20260227145522`）：
     - `forward_step` rank median `4.02%`（PASS）
     - `backward_step` rank median `5.11%`（FAIL，接近门限）
     - `optimizer_step` rank median `7.68%`（FAIL，较 round3 有改善）
   - 备注：
     - 同期出现一次 `MASTER_PORT` 占用冲突（`Address already in use`）；通过切换高位端口区间（`MASTER_PORT=7400/7600/7700/7800`）后复现稳定。

4. **当前 root-cause 状态（更新）**
   - `optimizer_step` 残差并非单纯 timing boundary 问题，仍存在运行态相关的系统性偏高（当前约 `+6%`~`+8%` 中位区间）。
   - `backward_step` 已逼近 5%，但对 subtraction 口径与 run-state 仍敏感。

## 8) Stage-2 fidelity round5（2026-02-27）

1. **新定位的 temporal fidelity 根因（pipeline replay 维度）**
   - scaling replay cache 之前只按 `dst_rank` 存一个文件：
     - `activation_to_rank{dst}.pt`
     - `grad_to_rank{dst}.pt`
   - 在 `trace_start=4, train_iters=6, warmup=3, profile=3` 场景下，这会导致消费侧 rank 在 profile 期间反复加载“最后一次覆盖写入”的同一 replay tensor，而不是逐 iteration 对应的 tensor。
   - 对 MoE 路径而言，这会引入额外的时序失配（同一目的 rank 的多次 profile step 输入/梯度被压缩为单样本 replay）。

2. **round5 修复方案（已落地）**
   - `megatron/training/training.py`
     - activation/grad replay cache 的写入路径改为 iteration-indexed：
       - `activation_to_rank{dst}_iter{current_iter}.pt`
       - `grad_to_rank{dst}_iter{current_iter}.pt`
     - replay grad 读取优先按 `current_iter` 命中迭代文件；旧格式 `grad_to_rank{dst}.pt` 保留兼容读取。
   - `megatron/profiler/utils.py`
     - 新增 `resolve_scaling_replay_path(cache_dir, rank_id, current_iter)`。
     - `sim_forward_step` replay 加载改为优先迭代文件，旧格式作为兼容回退。
   - 单测：
     - 新增 `tests/unit_tests/profiler/test_scaling_replay_cache_paths.py`（iter优先/legacy回退/缺失返回None/cache_dir为空）。

3. **round5 实测结论（最新）**
   - 功能正确性：
     - scaling rank3→rank7 probe PASS，cache 目录出现 `*_iter1..5.pt` 文件。
   - fidelity（pair `20260227145502`）仍未过线：
     - subtract-comm：
       - `forward_step` median `4.23%`（PASS）
       - `backward_step` median `14.18%`（FAIL）
       - `optimizer_step` median `7.57%`（FAIL）
     - no-subtract：
       - `forward_step` median `14.80%`（FAIL）
       - `backward_step` median `17.53%`（FAIL）
       - `optimizer_step` median `7.57%`（FAIL）

4. **当前状态更新**
   - round5 已修复一个明确的 replay 时序对齐缺陷；
   - 但 stage-2 `<=5%` 目标仍未满足，主要残余集中在 backward/optimizer，且 distributed run-to-run 波动对结论敏感。

## 9) Stage-2 protocolfix8 + optimizer proposal（2026-02-27, design-only）

1. **non-semantic protocol 执行结果（本轮）**
   - 已执行固定端口段 + 固定 rank-order + repeated pairing：
     - 端口段：`9400/9500/9600` 家族；
     - rank-order：`0,4,1,5,2,6,3,7`（对照 `0..7`）。
   - 关键结论：
     - 最优单次：`forward=3.06%`, `backward=7.51%`, `optimizer=5.97%`；
     - backward 最近门限值：`5.66%`（仍略高于 5%）；
     - 说明 protocol 对齐已显著缩小误差，但仍未稳定跨过 backward/optimizer 门限。

2. **可能触及 scaling 执行语义的 optimizer 优化提案（仅方案，未改代码）**
   - 目标：
     - 继续压缩 `optimizer_step` residual，同时保持 distributed/scaling 语义可对齐，并保留 rank-level comp 可观测性。
   - 提案名称：**Rank-local optimizer microphase decomposition（trace-only segmentation）**
   - 核心思路：
     1. 在 scaling `optimizer_step` 内部增加细粒度 microphase trace（仅记录，不改变计算）：
        - `optimizer_main_update`（主参数更新）
        - `optimizer_state_update`（state tensor update）
        - `optimizer_post_update`（post hooks / grad clear）
     2. distributed 侧在相同逻辑点增加同名 microphase trace（保持同构），并在 compare 里做 phase-aware 对齐。
     3. top-level `optimizer_step` 保持原定义，microphase 只用于差异归因和后续等价对齐，不改学习率/梯度/参数更新语义。
   - 为什么“可能触及语义”：
     - 尽管目标是 trace-only，但需要在 optimizer hot path 增加额外同步点或 trace 边界，可能改变 kernel launch 排布与 overlap，进而轻微影响 wall-time。
   - 语义对齐保证（设计约束）：
     - 不改 optimizer 算法、不改参数更新次序、不改 grad lifecycle；
     - distributed 与 scaling 同步加点，同名 phase 对齐，避免只改单边；
     - 默认关闭（通过新 flag 启用），避免影响已有模型默认路径性能。
   - rank-level comp 表达：
     - 每个 rank 输出 `optimizer_step` + microphase duration；
     - compare 输出 phase-level diff 与总和 diff，支持定位“哪个 phase 造成 residual”。
   - 验证计划（代码变更前先约定）：
     - 单测：trace 结构完整性（phase 名称、顺序、总和一致性）；
     - 集成：同一 run_config 下 distributed/scaling 对比；
     - 门限：先看 `optimizer_step` op-rank-median，再看 phase 归因是否稳定。

## 10) Stage-2 optimizer microphase 实施结果（2026-02-27, round9）

1. **实现范围（已落地）**
   - 新增 flag（默认关闭）：
     - `--trace-optimizer-microphases`
   - 在 distributed/scaling 两侧增加同构 microphase trace 点：
     - `optimizer_main_update`
     - `optimizer_state_update`
     - `optimizer_post_update`
   - 保留原有 top-level `optimizer_step` CMD，不改变其名称和存在性。

2. **语义约束执行情况**
   - 未修改 optimizer 算法、参数更新顺序、学习率调度逻辑条件。
   - 默认路径（未开启新 flag）不新增 microphase CMD，避免影响既有模型默认性能与 trace 口径。
   - microphase 仅作为附加诊断视图；开启后只增加 trace 边界与记录，不引入 fallback 分支。

3. **代码触点**
   - `megatron/training/arguments.py`
     - 新增 `--trace-optimizer-microphases`。
   - `megatron/training/training.py`
     - 新增 `_optimizer_microphase_cmd(...)` 等 helper；
     - distributed `train_step` 和 scaling profiling 路径均接入同名 phase；
     - 扩展 `simu_micro_batch_ids` 字典，保证 phase batch_id 可独立递增记录。

4. **验证结论（单测）**
   - 新增 `tests/unit_tests/test_training_optimizer_microphase.py`：
     - parser 默认值/开启值校验；
     - microphase key 注入校验；
     - phase 顺序/存在性校验；
     - invalid phase fail-fast 校验。
   - 当前结果：
     - `pytest -q tests/unit_tests/test_training_optimizer_microphase.py` → `6 passed`。

## 11) Stage-2 optimizer microphase fidelity 结果（2026-02-27, round10）

1. **执行协议**
   - distributed/scaling 均启用：
     - `TRACE_START=4`, `TRAIN_ITERS=6`
     - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
     - `TRACE_OPTIMIZER_MICROPHASES=1`
   - scaling 固定执行协议：
     - rank-order：`0,4,1,5,2,6,3,7`
     - run1/run2/run3（端口 `9630/9631/9632`），同一 distributed baseline (`ts=20260227174546`)。

2. **phase-aware 单次证据（op-rank-median）**
   - run1:
     - `forward=8.34%`, `backward=6.81%`, `optimizer_step=11.19%`
     - `optimizer_main_update=10.86%`
     - `optimizer_state_update=33.33%`
     - `optimizer_post_update=0.00%`
   - run2:
     - `forward=10.53%`, `backward=13.39%`, `optimizer_step=10.43%`
     - `optimizer_main_update=9.75%`
     - `optimizer_state_update=27.78%`
     - `optimizer_post_update=16.67%`
   - run3:
     - `forward=7.79%`, `backward=8.17%`, `optimizer_step=13.22%`
     - `optimizer_main_update=12.52%`
     - `optimizer_state_update=30.00%`
     - `optimizer_post_update=12.50%`

3. **median-of-runs（3 runs, op-rank-median）**
   - `forward_step=8.34%`
   - `backward_step=8.17%`
   - `optimizer_step=11.19%`
   - `optimizer_main_update=10.86%`
   - `optimizer_state_update=30.00%`
   - `optimizer_post_update=12.50%`

4. **结论（当前阶段）**
   - microphase trace 已提供归因能力，但 fidelity 未达 `<=5%`：
     - optimizer 主残差并非集中在 `state_update/post_update`；
     - `optimizer_main_update` 本身仍在 ~10% 量级，说明 residual 主要仍来自 optimizer 主更新阶段的跨模式执行差异。
   - `optimizer_state_update` / `optimizer_post_update` 由于绝对时长很短（约 `0.01~0.05ms`），相对误差易放大；应结合绝对时长解读，不宜单独作为主 gate。

## 12) Stage-2 fidelity round11（2026-02-27）：语义触及优化试验（scaling only, default-off）

1. **根因聚焦（round10 后）**
   - phase-aware 结果显示 `optimizer_main_update` 残差仍主导，且主要集中在 `PP stage1` ranks。
   - 推断 scaling replay I/O（尤其 grad replay 写回）在 profiling iteration 内的时序会对后续测量产生扰动。

2. **试验 A：replay grad 写回时序可选对齐（已落地）**
   - 新增 fake/scaling 参数：
     - `--scaling-replay-write-phase {pre_optimizer,post_optimizer}`（default=`pre_optimizer`）
   - `training.py` 行为：
     - `post_optimizer` 下将 `grad_to_rank*.pt` 写回推迟到 `optimizer_step` 之后；
     - 保持 replay 语义可用（同 iteration 文件仍写出），默认路径不变。
   - 脚本接线：
     - `examples/pretrain_deepseek_v3_moe.sh` 新增 `SCALING_REPLAY_WRITE_PHASE`（校验 + 透传）。

3. **试验 B：scheduler increment 对齐开关（已落地，默认关闭）**
   - 新增 fake/scaling 参数：
     - `--scaling-align-scheduler-increment`（default-off）
   - 作用：
     - scaling scheduler increment 由 `fake_dp` 切换为真实 `data_parallel_size` 口径（仅启用时生效）。
   - 目的：
     - 验证 optimizer dynamics 对残差的贡献是否明显。

4. **round11 证据（同一 distributed baseline: `20260227182456`）**
   - `post_optimizer`（A）：
     - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepost_run1.log`
     - op-rank-median:
       - `forward=10.74%`
       - `backward=12.71%`
       - `optimizer_step=6.56%`
       - `optimizer_main_update=6.21%`
   - `pre_optimizer`（对照）：
     - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepre_run1.log`
     - op-rank-median:
       - `forward=14.09%`
       - `backward=19.52%`
       - `optimizer_step=6.47%`
       - `optimizer_main_update=6.04%`
   - `post_optimizer + align_scheduler_increment`（A+B）：
     - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepost_aligninc_run1.log`
     - op-rank-median:
       - `forward=8.10%`
       - `backward=10.92%`
       - `optimizer_step=7.16%`
       - `optimizer_main_update=6.50%`

5. **round11 结论**
   - A（post write）对 `forward/backward` 有稳定改善迹象；对 `optimizer_main_update` 仅小幅波动（~6% 区间），尚未跨过 `<=5%`。
   - B（scheduler increment 对齐）在本轮未带来 optimizer 主指标收益，且 `optimizer_step` 有回退风险；建议继续保持 default-off，仅作实验开关。
   - 当前最稳健结论：
     - **主要残差仍在 `optimizer_main_update` 的 stage1 ranks；distributed baseline run-to-run 漂移仍显著影响 gate 结论。**

## 13) Stage-2 fidelity round12（2026-02-28）：`post_optimizer` 写回 + 固定协议 repeated pairing

1. **执行协议（与 round11 一致，补 3-run 统计）**
   - distributed/scaling 共同参数：
     - `TRACE_START=4`, `TRAIN_ITERS=6`
     - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
     - `TRACE_OPTIMIZER_MICROPHASES=1`
   - scaling 语义实验开关：
     - `SCALING_REPLAY_WRITE_PHASE=post_optimizer`
     - `SCALING_ALIGN_SCHEDULER_INCREMENT=0`
   - 固定 compare 口径：
     - `distributed_subtract_comm=True`
     - `ops=forward_step,backward_step,optimizer_step,optimizer_main_update,optimizer_state_update,optimizer_post_update`
   - 固定运行协议：
     - fixed rank-order：`0,4,1,5,2,6,3,7`
     - fixed port segments：`990x/995x`（避免端口漂移）
     - repeated runs：`run1/run2/run3`，并采用 timestamp pairing。

2. **single-run 证据（op-rank-median）**
   - run1（pair `20260228051919`）：
     - `forward=5.69%`
     - `backward=8.24%`
     - `optimizer_step=12.44%`
     - `optimizer_main_update=12.03%`
   - run2（pair `20260228052156`）：
     - `forward=8.23%`
     - `backward=12.65%`
     - `optimizer_step=7.76%`
     - `optimizer_main_update=7.70%`
   - run3（pair `20260228052432`）：
     - `forward=8.13%`
     - `backward=13.03%`
     - `optimizer_step=6.11%`
     - `optimizer_main_update=6.14%`

3. **median-of-runs（3 runs）**
   - `forward_step=8.13%`
   - `backward_step=12.65%`
   - `optimizer_step=7.76%`
   - `optimizer_main_update=7.70%`
   - `optimizer_state_update=10.56%`
   - `optimizer_post_update=12.50%`

4. **round12 结论**
   - 与 round10（`optimizer_main_update` median-of-runs `10.86%`）相比，`post_optimizer` 写回策略在本轮将 `optimizer_main_update` 降至 `7.70%`，但仍未达到 `<=5%` gate。
   - backward residual 依然显著（`12.65%`），且 run-to-run 漂移仍大于 gate 边界，说明仅靠非语义协议对齐无法稳定收敛到目标阈值。
   - `optimizer_state_update`/`optimizer_post_update` 绝对时长仍处于 `0.01~0.05ms` 量级，百分比波动继续放大，不应作为主 gate 判据。
   - 现阶段仍可维持结论：**主残差集中在 `optimizer_main_update` + stage1 ranks 的跨模式执行差异，后续若继续压 `optimizer_step` 需要更细粒度主更新路径归因。**

## 14) Stage-2 current rerun + retrospective（2026-02-28）

1. **current latest code 复测（Round12 协议, 3 runs）**
   - distributed/scaling 固定协议：
     - `TRACE_START=4`, `TRAIN_ITERS=6`
     - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
     - `TRACE_OPTIMIZER_MICROPHASES=1`
   - scaling 固定开关：
     - `SCALING_REPLAY_WRITE_PHASE=post_optimizer`
     - `SCALING_ALIGN_SCHEDULER_INCREMENT=0`
     - `SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7`
   - pair timestamps：
     - run1 `20260228072907`
     - run2 `20260228073218`
     - run3 `20260228073523`

2. **current single-run 证据（op-rank-median）**
   - run1:
     - `forward=5.21%`
     - `backward=12.03%`
     - `optimizer_step=11.61%`
     - `optimizer_main_update=11.28%`
   - run2:
     - `forward=9.35%`
     - `backward=17.87%`
     - `optimizer_step=9.43%`
     - `optimizer_main_update=9.37%`
   - run3:
     - `forward=6.14%`
     - `backward=15.11%`
     - `optimizer_step=7.84%`
     - `optimizer_main_update=7.56%`

3. **current median-of-runs（3 runs）**
   - `forward_step=6.14%`
   - `backward_step=15.11%`
   - `optimizer_step=9.43%`
   - composite:
     - `mean_3ops=10.23%`
     - `max_3ops=15.11%`

4. **与历史基线对比 + best round 判定**
   - 对比对象：
     - Round4: `4.02% / 5.11% / 7.68%`
     - Round6-8（best single）: `3.06% / 7.51% / 5.97%`
   - 判定规则：
     - 主判据 `mean_3ops`
     - 辅判据 `max_3ops`
   - 判定结果：
     - current（median-of-runs）`10.23% / 15.11%`
     - Round4 `5.60% / 7.68%`
     - Round6-8 `5.51% / 7.51%`
     - **overall best = Round6-8**

5. **Round9+ retrospective 结论**
   - `Round9/10`（microphase）主要贡献为诊断可观测性，不是 fidelity 直接提升路径；开启后 hot path trace 边界可能引入测量扰动。
   - `Round11/12`（replay write phase / scheduler increment）对部分场景有稳定性收益，但未击穿主残差；`optimizer_main_update` 仍主导。
   - 测量噪声显著（single-run vs repeated median 差异大），但不足以解释 current 相对 Round4/Round6-8 的大幅退化（尤其 backward）。
   - 可保留项（建议）：
     - `--trace-optimizer-microphases`（default-off 诊断路径）
     - `--scaling-replay-write-phase`（default-off 诊断路径）
     - 对应单测增强
   - 不建议作为默认 gate 路径：
     - `TRACE_OPTIMIZER_MICROPHASES=1`
     - `SCALING_ALIGN_SCHEDULER_INCREMENT=1`

## 15) Round6-8 baseline follow-up（2026-02-28）：O2/B1 实验收敛结论

1. **执行背景**
   - 为避免 current-latest（Round12 语义路径）干扰，回到 Round6-8 基线：
     - worktree commit: `a3158883`
     - cherry-pick: `3a50265d`（仅 microphase trace 路径）
   - 固定协议：
     - `TRACE_START=4`, `TRAIN_ITERS=6`
     - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
     - rank-order `0,4,1,5,2,6,3,7`
     - `SCALING_MIN_WARMUP_ITERS=0`, `SCALING_PROFILE_ITERS=3`
     - compare 使用 `--distributed-subtract-comm` + 3-run repeated pairing。

2. **O2 A/B 结论（`TRACE_OPTIMIZER_MICROPHASES=0` vs `1`）**
   - median-of-runs：
     - micro0: `8.37% / 10.47% / 8.61%`（fwd/bwd/opt, `mean_3ops=9.15%`, `max_3ops=10.47%`）
     - micro1: `5.71% / 9.34% / 8.36%`（`mean_3ops=7.80%`, `max_3ops=9.34%`）
   - 结论：
     - **O2（“microphase 会恶化 fidelity”）不成立**；
     - 在该基线上 microphase=1 反而更优（尽管仍未过 `<=5%` gate）。

3. **B1 实现与验证（strict grad replay）**
   - 代码改动（基线 worktree）：
     - `--scaling-strict-grad-replay`（default-off）
     - `_build_scaling_output_tensor_grad(...)` 在 profile window 且 strict 开启时，若 grad cache 缺失则 `FileNotFoundError`。
     - 脚本新增 `SCALING_STRICT_GRAD_REPLAY` 环境开关并做 `0/1` fail-fast 校验。
   - 单测/静态验证：
     - `tests/unit_tests/test_training_optimizer_microphase.py` 扩展后 `9 passed`。

4. **B1 运行证据与结论**
   - 单次 strict probe（单 pass）：
     - rank0 在 profiled backward 首次命中 “cache missing” 并立即失败（符合 fail-fast 预期）。
   - 双 pass strict（warm pass 预写 cache + strict pass 读 cache）3-run：
     - run1: `5.11% / 10.02% / 7.93%`
     - run2: `5.84% / 9.51% / 9.85%`
     - run3: `10.25% / 9.20% / 9.29%`
     - median-of-runs: `5.84% / 9.51% / 9.29%`（`mean_3ops=8.21%`, `max_3ops=9.51%`）
   - 对比 O2-best（micro1）：
     - `forward +0.13%`, `backward +0.17%`, `optimizer +0.93%`（B1 略退化）
   - 结论：
     - B1 作为 **完整性/诊断模式** 有效；
     - 但不应作为当前默认 gate 提升路径（对主指标无改善）。

5. **下一步优先级更新**
   - O2 已被证伪，B1 已完成并确认“有价值但不提分”；
   - 下一优先假设应切换为 **O1**：
     - 聚焦 `optimizer_main_update` 的 timing boundary / queue contamination；
     - 在 Round6-8 基线上做最小变量 A/B（default-off）并继续 3-run repeated pairing。

## 16) O1 精确机制定义 + 噪声地板约束（2026-02-28）

1. **O1 机制定义（冻结版，先定义后实现）**
   - 目标问题：
     - `optimizer_step` / `optimizer_main_update` 可能被前序 CUDA queue 残留工作污染（跨 phase 计时边界漂移）。
   - O1 单变量机制（拟实施）：
     - 新增 default-off 开关（命名待实现时确定），开启后在 **distributed + scaling 两侧对称**执行：
       - 仅在进入 top-level `optimizer_step` CMD 之前插入一次 `torch.cuda.synchronize()`（pre-CMD drain）。
   - 明确不包含（用于保持单变量）：
     - 不加 `dist.barrier()`（避免引入跨 rank wait 噪声）；
     - 不改 CMD start/end event 位置；
     - 不改 optimizer 算法路径与 microphase 划分。

2. **为何不用 barrier / CMD 边界改动**
   - barrier 会将通信与调度差异注入测量窗口，偏离“queue contamination”定位目标；
   - 改 CMD 边界会同时改变测量定义本身，导致 A/B 变量不再单一。

3. **噪声地板量化（Round6-8 baseline, no code change, repeat x5）**
   - 固定协议（microphase=1）下，最终配对采用 rank7 end-of-run timestamp cap：
     - run1~run5: `20260228134044 / 20260228134319 / 20260228134554 / 20260228134829 / 20260228135103`
   - single-run（op-rank-median）：
     - run1: `5.64 / 14.66 / 9.09`
     - run2: `10.36 / 10.54 / 8.28`
     - run3: `7.16 / 11.67 / 11.13`
     - run4: `8.37 / 15.85 / 4.89`
     - run5: `4.57 / 10.55 / 6.12`
   - median-of-runs：
     - `forward=7.16%`, `backward=11.67%`, `optimizer=8.28%`
   - run range：
     - `forward=5.79%`, `backward=5.31%`, `optimizer=6.24%`

4. **约束结论（对后续 O1 验收）**
   - 当前测量体系噪声显著，`<~1-2%` 的 A/B 改善不能直接视为有效收益；
   - O1 需要相对于噪声地板给出更强信号（例如多项指标一致改善 + repeat median 稳定下降）。

5. **B1 与 O1 的关系（决策）**
   - B1（strict grad replay）是独立完整性守卫，价值与 O1 成败无关；
   - 已按 default-off 原则回灌 mainline，不等待 O1 结果。

## 17) O1 实施结果（2026-02-28）：pre-CMD optimizer drain A/B（repeat x5）

1. **实现回顾（与冻结定义一致）**
   - 开关：`--trace-optimizer-pre-cmd-drain`（default-off）。
   - 机制：`optimizer_step` 顶层 CMD 进入前执行一次 `torch.cuda.synchronize()`。
   - 对称性：distributed/scaling 两侧均加；无 barrier、无 CMD 边界变更。

2. **执行协议**
   - 共同参数：
     - `TRACE_START=4`, `TRAIN_ITERS=6`
     - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
     - `TRACE_OPTIMIZER_MICROPHASES=1`
   - scaling:
     - rank-order `0,4,1,5,2,6,3,7`
     - `SCALING_MIN_WARMUP_ITERS=0`, `SCALING_PROFILE_ITERS=3`
   - compare:
     - `--distributed-subtract-comm`
     - pairing cap 固定为 rank7 end-of-run timestamp（避免顺序 scaling 错配）。

3. **A/B 结果（op-rank-median, median-of-runs）**
   - A（`drain=0`）：
     - `forward=12.84%`, `backward=10.06%`, `optimizer=8.23%`
     - `mean_3ops=8.80%`, `max_3ops=12.84%`
   - B（`drain=1`）：
     - `forward=11.55%`, `backward=14.63%`, `optimizer=10.00%`
     - `mean_3ops=12.10%`, `max_3ops=14.63%`
   - delta（B-A）：
     - `forward=-1.29%`, `backward=+4.57%`, `optimizer=+1.77%`
     - `mean_3ops=+3.30%`, `max_3ops=+1.79%`

4. **噪声地板标准下的判定**
   - 既有噪声基线（repeat x5, no-code-change）提示单 op run-range 约 `5%~6%`。
   - O1 A/B 下，spread 未同步收敛：
     - range 方面 `forward/backward/optimizer` 中 3 项有 3 项不降；
     - IQR 方面仅少数项下降，`backward/max` 明显变差。
   - 结论：
     - O1 不满足“超越噪声地板且 spread 同步下降”的验收要求；
     - 应保留为 default-off 诊断开关，不进入主 gate。

5. **后续策略更新**
   - 关闭 O1 作为主路径候选；
   - 后续继续单变量实验，但必须保持：
     - rank7 cap pairing；
     - repeat x5+；
     - 噪声地板对照判定。
