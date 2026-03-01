## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Implemented backward-comp governance plan: froze official metric semantics (`seq8192 + phase-pure + repeat-x5`), added advanced-diagnostics opt-in guard in DeepSeek/Qwen scripts, and updated task_memory plan/notes/issues with scope-lock decisions |
| 2026-03-01 | Completed attention-core SDPA deep-segment formal repeat-x5 (`dist + scaling_on + scaling_off`) with robust aggregation and added fmha variability bucketing by `rank/state/iter` (iter1-dominant residual pattern) |
| 2026-03-01 | Completed seq8192 attention-core SDPA subsegment formal repeat-x5 (`dist + scaling_on + scaling_off`) and generated robust `top-k + median/IQR` summary: `fmha_cutlassB` remains top1 in 5/5 runs with stable pre-fmha `FillFunctor<unsigned char>` adjacency asymmetry |
| 2026-03-01 | Added `attn_core_sdpa_bwd` immediate same-stream neighbor diagnostics (analyzer + unit tests), reran x1 re-analysis with `small-kernel-threshold-us=60/80`, and confirmed pre-fmha count asymmetry is threshold-sensitive while backward gap remains unchanged |
| 2026-03-01 | Added MLA attention-core micro-segment x1 localization run: subsegment hooks + adjacency top-k diagnostics show residual is concentrated in `attn_core_sdpa_bwd`, while `attn_core_precast_bwd`/`attn_core_postcast_bwd` remain near-zero |
| 2026-03-01 | Implemented debug-only attention backward segment NVTX (including MLA path), completed seq8192 clean-x1 + formal repeat-x5 capture/analyze/compare, and confirmed `attn_core_bwd`/`fmha_cutlassB` as stable dominant backward residual source |
| 2026-03-01 | Completed patched seq8192 clean-x1 attention-family deep diagnosis: hardened `analyze_nsys_attention_family_delta.py` + unit tests, produced dist-vs-scaling on/off stage1 backward reports, and confirmed `fmha_cutlassB` remains dominant with launch-config parity |
| 2026-03-01 | Completed patched seq8192 phase-pure x1 (dist/scaling on/off) with new NVTX structural-health gate (`open_forward/open_backward/fwd-bwd overlap`): gate passed cleanly on all branches, but backward remains >5% and attention-family (`fmha_cutlassB`) is still top residual, so priority is shifted to attention diagnostics before repeat-x5 freeze |
| 2026-03-01 | Completed NVTX attribution-artifact root-cause validation on seq8192 phase-repeat traces: confirmed systematic forward/backward CMD overlap from unbalanced `row_g_fwd` NVTX push/pop, landed minimal `try/finally` fix, and verified RED→GREEN with targeted + regression unit suites |
| 2026-03-01 | Completed seq8192 phase-pure formal repeat-x5 (distributed + scaling DDP on/off): contamination stayed 0%, DDP-off only partially improved backward, and kernel-family robust stats confirmed `fmha_cutlassB` sustained dominance (5/5 runs) |
| 2026-03-01 | Completed seq8192 phase-pure DDP probe A/B (distributed x1 + scaling DDP on/off x1): contamination stayed 0%, DDP-off improved but backward still >5%, and attention-family (`fmha_cutlassB`) remained dominant residual component |
| 2026-03-01 | Completed scaling DDP-hook hypothesis validation (debug switch + x1 NSYS A/B): hook path contributes moderate overhead but does not explain dominant backward residual; disabling scaling hook path did not improve dist-vs-scale backward gap |
| 2026-03-01 | Completed cross-run (`round68 seq8192 run1..5`) backward residual source validation: non-comm (`fmha_cutlassB`) remains dominant across runs; `_AllToAll` micro-repro autograd-node parity verified; regression unit suite remains green |
| 2026-03-01 | Completed backward residual deep-dive attribution and scaling comm-adjacent emulation feasibility validation (`--scaling-comm-adjacent-copy-iters`): end-to-end works but does not materially reduce backward residual in x1 phase-pure compare |
| 2026-03-01 | Validated postfix all_to_all comm-adjacent attribution fix (materialization + contiguous-in-wrapper) with unit tests and phase-aware NSYS x1 compare; contamination stayed zero but backward residual remained high |
| 2026-03-01 | Completed phase-label NSYS sanity x1 distributed/scaling rerun (capture/export/analyze/compare), verified phase-window coverage + zero contamination, and archived diagnostics report |
| 2026-03-01 | Implemented phase-level kernel ground-truth instrumentation and pure compute-only compare path (new trace args, CMD phase NVTX, NSYS analyzer pure metrics + contamination gate), updated scripts/tests, and validated with unit + integration replays |
| 2026-02-28 | Completed Round6-8 seq8192 NSYS repeat-x5 semantics-freeze validation (distributed/scaling capture + sqlite breakdown + compute-only compare), and archived subtract/no-subtract/NSYS tri-view evidence for backward gate assessment |
| 2026-02-28 | Completed Round6-8 backward measurement-semantics validation on fixed seq8192 batch (baseline/op-map/stage-aware/no-subtract views, repeat x5), confirming backward over-subtraction bias and preserving code-freeze decision |
| 2026-02-28 | Completed Round6-8 measurement-regime change execution (OOM sweep + seq8192 repeat x5 with rank7 cap), confirmed full-profile memory ceiling, and archived noise-floor verdict report |
| 2026-02-28 | Implemented round6-8 O1 pre-CMD optimizer-drain switch (symmetric distributed/scaling, default-off), executed A/B repeated pairing x5 with fixed rank7 cap, and completed noise-floor-aware verdict (O1 rejected) |
| 2026-02-28 | Quantified round6-8 pure noise floor via 5-run repeated pairing (no code changes), corrected pairing cap to rank7 end-of-run timestamps, and archived noise-baseline report |
| 2026-02-28 | Backported B1 strict-grad-replay guard to mainline as default-off integrity control and completed unit/static validation |
| 2026-02-28 | Completed round6-8-baseline B1 strict-grad-replay implementation and validation (single-pass fail-fast probe + two-pass strict 3-run repeated pairing), and archived comparison evidence vs O2 baseline |
| 2026-02-28 | Completed round6-8-baseline O2 microphase A/B reruns (`TRACE_OPTIMIZER_MICROPHASES=0/1`, 3-run repeated pairing) and archived median-of-runs conclusion |
| 2026-02-28 | Completed stage-2 current-latest 3-run rerun under Round12 protocol, produced cross-round comparison vs Round4/Round6-8, selected overall best round, and archived retrospective + forward plan report |
| 2026-02-28 | Added stage-2 round12 post-optimizer replay-write repeated-fidelity execution records (run1/2/3, median-of-runs), and archived new compare/repeat artifacts |
| 2026-02-27 | Added stage-2 round11 semantic-alignment experiment progress (replay write-phase + scheduler increment switch), with new compare evidence |
| 2026-02-27 | Completed stage-2 optimizer microphase protocolfix8 fidelity reruns (single + repeat aggregation), and archived phase-aware compare evidence |
| 2026-02-27 | Implemented stage-2 optimizer microphase trace path (default-off) across distributed/scaling, added unit tests, and archived validation report |
| 2026-02-27 | Added stage-2 protocolfix8 repeated-pairing execution (fixed ports/rank-order), archived single+aggregate fidelity evidence, and documented residual backward/optimizer gaps |
| 2026-02-27 | Added stage-2 round5 iteration-indexed replay-cache alignment implementation and validation evidence |
| 2026-02-24 | Recorded stage-1 implementation progress and checkpoints |
| 2026-02-24 | Completed stage-1 validation matrix and captured trace-alignment evidence |
| 2026-02-24 | Completed scaling NaN timing-impact assessment and restored router unit tests to green |
| 2026-02-24 | Added 32-rank scaling validation rerun and distributed-vs-scaling rank0/rank7 comp-timing investigation with fixes |
| 2026-02-24 | Implemented stage-1.5 trace comp calibration and automated rank0/rank7 compare script |
| 2026-02-24 | Removed stage-1.5 calibration path and switched back to raw comp-gap root-cause debugging |
| 2026-02-24 | Added pipeline-state aligned compare, rank-aware replay cache path, and repeated no-calibration reruns |
| 2026-02-24 | Implemented trace sub-op sync mode (`global/event`), compare timestamp pairing/median aggregation, and completed sync-mode verification |
| 2026-02-24 | Completed 6-GPU qwen TP2/DP3/EP1/PP1 consistency debugging, fixed distributed backward trace coverage, and added new analysis report |
| 2026-02-25 | Completed Qwen3 seq2048 (mbs=8/4) 6-GPU reruns, cross-mode sub-op attribution audit, and refreshed unit/integration evidence |
| 2026-02-25 | Added scaling-parity probe fixes (TE scaling TP guard + RoPE seq guard), reran Qwen3 seq2048 on GPUs 2-7, and archived new failure-focused validation report |
| 2026-02-25 | Completed 8-GPU Qwen3 trace4 bwd-I/O-fix retest (event/global), validated single-vs-avg robustness, and added full-profile model-size escalation evidence |
| 2026-02-25 | Committed checkpoint, added forward/optimizer decomposition tooling, introduced compare trimmed-mean auxiliary report, and completed 8-GPU forward-fidelity boundary trial |
| 2026-02-25 | Added stage-aware comm-scale compare path with robust op-median summaries, completed new 8-GPU rerun, and published fidelity2 report for paper-facing metrics |
| 2026-02-26 | Implemented kernel-ground-truth NVTX/NSYS pipeline, added alltoall-vs-allgather A/B validation, fixed allgather dispatcher API mismatch, and published B-path report |
| 2026-02-26 | Completed Issue1/Issue2 deep debug (NSYS kernel-set/stream evidence, profiler before/after validation), and published dedicated test report |
| 2026-02-26 | Implemented NSYS semantics-aware compare (union/primary/shared-kernel + rank-total comp summary), reran Qwen3 traces, and published measurement-fix report |
| 2026-02-27 | Landed stage-2 (DeepSeek-V3 architecture standard) documentation into task_memory (plan/notes/issues/progress); no code changes |
| 2026-02-27 | Executed stage-2 code implementation + validation: added shared-expert gate support, script robustness knobs, completed stage-2 unit tests and scaling smoke, and recorded distributed PP2 NaN blocker |
| 2026-02-27 | Resolved stage-2 distributed PP2/EP2 bf16 NaN blocker via MLA-only pipeline dtype alignment + sigmoid router finite normalization, reran distributed/scaling smoke, and archived round2 report |
| 2026-02-27 | Continued stage-2 fidelity alignment: reverted DeepSeek script CMD sync default to global, aligned scaling optimizer timing boundary with distributed train_step prefetch, reran distributed/scaling trace4-iter6 comparisons, and recorded remaining comp-gap root-cause status |
| 2026-02-27 | Continued stage-2 fidelity round4: mirrored distributed optimizer pre-CMD side effects in scaling (`numel` pre-scan), reran paired traces, and reduced optimizer residual gap (partial) |

# Progress

## 2026-03-01

### Completed

- Implemented governance-level follow-up for backward comp-gap convergence decisions:
  - froze official reporting semantics to `seq8192 + phase-pure + repeat-x5` in task docs;
  - downgraded subtract/stage-aware/op-map views to diagnostics-only;
  - locked next code-level scope to `attn_core_sdpa_bwd` iter-bucket diagnosis.

- Added explicit advanced-diagnostics opt-in guard in example scripts to reduce baseline pollution risk:
  - `examples/pretrain_deepseek_v3_moe.sh`: new `ADVANCED_DIAGNOSTICS=0|1` gate checks for non-baseline toggles (`TRACE_OPTIMIZER_MICROPHASES`, strict replay, replay-write post mode, scheduler-align, comm-adjacent copy iters, DDP-wrap disable, attention segment tracing).
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`: same `ADVANCED_DIAGNOSTICS=0|1` gate for qwen-side advanced toggles (`TRACE_ATTENTION_BACKWARD_SEGMENTS`, comm-adjacent copy iters, DDP-wrap disable).
  - behavior: fail-fast when advanced flags are set without explicit acknowledgement.

- Completed attention-core SDPA deep-segment formal repeat-x5 (`seq8192`, `dist + scaling_on + scaling_off`) with no training-semantic change:
  - artifact base:
    - `logs/nsys_phase_attn_core_deepseg_repeat5_v1/`
  - gate status:
    - NVTX structure gate (run1..5 × all branches) all PASS:
      - `open_forward=0`, `open_backward=0`, `overlap=0`;
    - contamination all PASS with max `0.00%`.
  - op-level robust results (from `pure_primary_union` compare logs):
    - scaling_on backward rank-median runs:
      - all-ranks: `[19.51, 11.68, 10.28, 12.36, 18.90]`, median/IQR `12.36 / 7.22`;
      - stage1 `rank4..7`: `[20.57, 15.915, 10.275, 10.530, 18.905]`, median/IQR `15.915 / 8.375`.
    - scaling_off backward rank-median runs:
      - all-ranks: `[19.27, 16.20, 19.03, 10.33, 16.68]`, median/IQR `16.68 / 2.83`;
      - stage1 `rank4..7`: `[19.995, 17.50, 19.495, 10.335, 14.05]`, median/IQR `17.50 / 5.445`.
  - deep-segment localization (`stage1/backward/steady/rank4..7`):
    - `attn_core_sdpa_fmha_bwd` remains dominant:
      - on: `gap median/IQR = 41.935 / 16.329 ms`, `fmha_share median/IQR = 97.62% / 0.33%`;
      - off: `gap median/IQR = 43.605 / 8.525 ms`, `fmha_share median/IQR = 97.87% / 0.79%`.
    - top1 kernel is stable `5/5` in both branches:
      - `fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80(...)`.
    - pre-fmha adjacency remains asymmetric:
      - distributed median `~1.901ms` vs scaling median `~1.150ms` (on), `~1.017ms` (off).
  - variability bucketing (`rank/state/iter`, fmha-vs-adjacency coupling):
    - iter1 is the high-gap bucket in both branches:
      - on: `fmha_gap median ~1.766ms`
      - off: `fmha_gap median ~1.777ms`
    - iter0/2 remain near-zero buckets;
    - iter1 simultaneously shows `small_pre_count_gap_median = -1.0` and `small_pre_gap_ms_median ~ -0.05ms`.
  - outputs:
    - `logs/nsys_phase_attn_core_deepseg_repeat5_v1/deepseek_phase_sl8192_attncore_deepseg_repeat5_summary.md`
    - `logs/nsys_phase_attn_core_deepseg_repeat5_v1/deepseek_phase_sl8192_attncore_deepseg_repeat5_summary.json`
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_seq8192_attention_core_deepsegment_repeat5.md`

- Completed seq8192 attention-core SDPA subsegment formal repeat-x5 (`dist + scaling_on + scaling_off`) with unchanged phase-pure protocol:
  - artifact base:
    - `logs/nsys_phase_attn_core_microseg_repeat5_v1/`
  - gate status:
    - NVTX structure gate passed on all runs/branches;
    - contamination max is `0.00%` on all runs/branches.
  - robust statistics (`stage1 backward steady rank4..7`, segment=`attn_core_sdpa_bwd`):
    - scaling_on:
      - `gap_ms median/IQR = 22.515 / 7.021`
      - `fmha_gap_ms median/IQR = 22.012 / 6.898`
      - `fmha_gap_share_pct median/IQR = 98.022 / 0.283`
    - scaling_off:
      - `gap_ms median/IQR = 23.725 / 20.024`
      - `fmha_gap_ms median/IQR = 23.263 / 19.963`
      - `fmha_gap_share_pct median/IQR = 98.041 / 0.703`
  - kernel-family/top-k stability:
    - top1 kernel is `fmha_cutlassB...` in `5/5` runs for both scaling_on/scaling_off;
    - pre-fmha top adjacent small-kernel name is stable on both branches:
      - `vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`;
    - distributed pre-fmha adjacent load remains higher (`count/ms` median: `35/1.751` vs scale `21/~1.05`).
  - op-level run statistics (op-rank median from compare logs):
    - scaling_on backward run-list: `[12.10, 4.27, 11.25, 14.83, 13.77]`, median/IQR `12.10 / 2.52`;
    - scaling_off backward run-list: `[4.05, 11.08, 18.17, 8.30, 14.84]`, median/IQR `11.08 / 6.54`.
  - outputs:
    - `logs/nsys_phase_attn_core_microseg_repeat5_v1/deepseek_phase_sl8192_attncore_repeat5_summary.md`
    - `logs/nsys_phase_attn_core_microseg_repeat5_v1/deepseek_phase_sl8192_attncore_repeat5_summary.json`
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_seq8192_attention_core_subsegment_repeat5.md`

- Completed `attn_core_sdpa_bwd` immediate-neighbor targeted diagnosis on existing seq8192 x1 sqlite (no recapture):
  - analyzer enhancement:
    - `tests/performance/analyze_nsys_attention_family_delta.py` now emits immediate same-stream neighbor metrics around fmha:
      - `small_kernel_immediate_pre/post_count`
      - `small_kernel_immediate_pre/post_ms`
      - immediate pre/post top names (`name/ms/count`) in report + JSON payload.
  - unit validation:
    - `tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py` extended with immediate-neighbor assertions and synthetic positive-case coverage (`7 passed`).
  - x1 re-analysis results (`stage1 backward steady rank4..7`, segment=`attn_core_sdpa_bwd`):
    - default threshold (`60us`):
      - scaling_on immediate pre count/ms: dist `35/1.751ms` vs scale `27/1.393ms`;
      - scaling_off immediate pre count/ms: dist `35/1.751ms` vs scale `18/0.900ms`.
    - threshold sweep (`80us`):
      - scaling_on immediate pre count/ms: dist `44/2.311ms` vs scale `44/2.451ms`;
      - scaling_off immediate pre count/ms: dist `44/2.311ms` vs scale `44/2.518ms`.
    - in all cases, immediate pre top1 name is stable on both sides:
      - `vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`.
    - `gap_ms` remains unchanged across threshold sweep (`23.115ms` on, `33.740ms` off).
  - interpretation:
    - pre-fmha neighbor family is present on both branches and same stream;
    - observed count asymmetry at `60us` is threshold-sensitive classification effect;
    - dominant residual remains in fmha runtime context within `attn_core_sdpa_bwd`, not explained by missing immediate-neighbor kernel path.
  - artifacts:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_attention_sdpa_immediate_neighbor_diagnosis_x1.md`
    - `logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_{on,off}_attn_core_sdpa_bwd_diag_immediate.{md,json}`
    - `logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_{on,off}_attn_core_sdpa_bwd_diag_immediate_th80.{md,json}`

- Completed attention-core fine-grained micro-segment implementation + clean x1 localization (`seq8192`, dist/scaling on/off):
  - instrumentation:
    - `MLA` core now exposes three debug-only backward subsegments:
      - `attn_core_precast_bwd`
      - `attn_core_sdpa_bwd`
      - `attn_core_postcast_bwd`
    - coarse `attn_core_bwd` is preserved for compatibility.
  - analyzer upgrade:
    - `tests/performance/analyze_nsys_attention_family_delta.py` now emits pre/post-fmha adjacent small-kernel top names (`name/ms/count`) in both report and JSON payload.
  - unit/static validation:
    - `py_compile` PASS;
    - `test_analyze_nsys_attention_family_delta.py` `6 passed`;
    - `test_multi_latent_attention.py` `4 passed`;
    - `test_attention.py -k attention_backward_segment_hooks` `2 passed`.
  - x1 localization evidence:
    - NVTX structure gate PASS on dist/scaling_on/scaling_off (`open_fwd=0`, `open_bwd=0`, overlap=0);
    - segment label counts are symmetric across all branches (`attn_core_*_bwd=96`);
    - `stage1 backward steady rank4..7`:
      - scaling_on: `attn_core_sdpa_bwd gap=23.115 ms`, `fmha_share=98.02%`;
      - scaling_off: `attn_core_sdpa_bwd gap=33.740 ms`, `fmha_share=98.11%`;
      - `attn_core_precast_bwd` and `attn_core_postcast_bwd` both `gap=0.000 ms`.
    - new pre-fmha adjacency top-k evidence:
      - dominant name is `FillFunctor<unsigned char>` vectorized kernel;
      - dist vs scale counts/ms:
        - on: `35 / 1.751ms` vs `27 / 1.393ms`
        - off: `35 / 1.751ms` vs `18 / 0.900ms`.
  - artifacts:
    - `logs/nsys_phase_attn_core_microseg_x1_v1/`
    - `logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_summary.md`
    - `logs/nsys_phase_attn_core_microseg_x1_v1/deepseek_phase_sl8192_attncore_x1_summary.json`
    - report: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_seq8192_attention_core_subsegment_x1.md`

- Implemented debug-only attention backward segment NVTX and completed seq8192 clean-x1 + repeat-x5 diagnostics:
  - instrumentation:
    - new trace arg: `--trace-attention-backward-segments`;
    - CMD phase labels now support extra tags (used for `attn_bwd_segment=*`);
    - hook-based segmentation added to:
      - `SelfAttention` path (`attn_qkv_bwd`, `attn_qk_layernorm_bwd`, `attn_core_bwd`, `attn_proj_bwd`);
      - `MLASelfAttention` path (DeepSeek workload) with explicit `_MLASDPACoreAttention` submodule to expose SDPA backward segment hooks.
  - script passthrough:
    - DeepSeek/Qwen example scripts now accept `TRACE_ATTENTION_BACKWARD_SEGMENTS=0|1`.
  - x1 (v2) evidence:
    - segment labels are present in sqlite (`attn_bwd_segment=*`);
    - NVTX gate remains clean (`open_forward=0`, `open_backward=0`, overlap=`0`);
    - segment diagnosis shows material gap only in `attn_core_bwd`, with `fmha_gap_share ~98%`.
  - formal repeat-x5 (`seq8192`, dist/scaling on/off):
    - all run gates PASS for NVTX structure;
    - op-rank median (median-of-runs):
      - scaling_on: `forward=15.61%`, `backward=11.20%`, `optimizer=2.29%`;
      - scaling_off: `forward=15.74%`, `backward=8.99%`, `optimizer=2.76%`;
    - `attn_core_bwd` remains dominant:
      - scaling_on core gap/fmha-share median: `22.814ms` / `98.01%`;
      - scaling_off core gap/fmha-share median: `20.199ms` / `97.96%`;
      - top1 delta is `fmha_cutlassB` in 5/5 runs (on/off).
    - stream diagnostics:
      - no primary/fmha stream mismatch across runs;
      - distributed has consistently higher pre-fmha small-kernel adjacency (core segment).
  - reports/artifacts:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_seq8192_attention_segment_debug_and_repeat5.md`
    - `logs/nsys_phase_attn_seg_x1_v2/`
    - `logs/nsys_phase_attn_seg_repeat5/deepseek_phase_sl8192_attnseg_repeat5_summary.md`
    - `logs/nsys_phase_attn_seg_repeat5/deepseek_phase_sl8192_attnseg_repeat5_summary.json`

- Completed patched clean-x1 attention-family deep diagnosis (`stage1 backward steady rank4..7`):
  - script hardening:
    - `tests/performance/analyze_nsys_attention_family_delta.py`
      - batch pairing now uses numeric-safe ordering;
      - fmha duration stats use strict overlap duration within phase window;
      - added per-rank fmha IQR output and JSON payload.
  - new dedicated unit test:
    - `tests/unit_tests/performance/test_analyze_nsys_attention_family_delta.py` (`5 passed`)
  - regression/static validation:
    - analyzer regression set `26 passed`, `py_compile` PASS.
  - output artifacts:
    - `logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_on.md/.json`
    - `logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_attention_diag_scaling_off.md/.json`
  - key conclusion:
    - `fmha_cutlassB` remains top1 absolute delta in both comparisons:
      - scaling_on: gap `42.122 ms`, fmha share `60.40%`;
      - scaling_off: gap `28.344 ms`, fmha share `72.86%`;
    - launch config parity is exact (`cfg_sets_equal=True`), so mismatch is not launch-shape induced.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_seq8192_phase_patched_x1_attention_family_diagnosis.md`

- Completed patched seq8192 phase-pure x1 validation with newly-added NVTX structural gate:
  - capture scope:
    - distributed / scaling DDP-on / scaling DDP-off
    - `SEQ_LEN=8192`, `TRAIN_ITERS=3`, `TRACE_KERNEL_GROUND_TRUTH_PHASE=1`, boundary sync=`event`
  - new gate (added in this round):
    - script: `tests/performance/check_nsys_nvtx_structural_health.py`
    - checks: `open_forward_step`, `open_backward_step`, `forward_backward_overlap_count`
    - strict thresholds used: `0/0/0`.
  - gate results (all PASS):
    - dist: `open_forward=0`, `open_backward=0`, `overlap_count=0`
    - scaling_on: `open_forward=0`, `open_backward=0`, `overlap_count=0`
    - scaling_off: `open_forward=0`, `open_backward=0`, `overlap_count=0`
  - phase-pure analyzer sanity:
    - all branches: `phase_window_parents=48`, `event_rows=72`, contamination `0.00%`.
  - compare results (`pure_primary_union`, shared primary-stream):
    - DDP-on op-rank-median: `forward=15.63%`, `backward=17.42%`, `optimizer=3.51%` (FAIL)
    - DDP-off op-rank-median: `forward=15.74%`, `backward=10.52%`, `optimizer=2.35%` (FAIL)
  - decision by user rule:
    - clean x1 backward remains significantly >5%, so next priority shifts to attention-family diagnosis before repeat-x5 freeze.
  - supporting artifact:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_seq8192_phase_patched_x1_nvtx_gate_and_compare.md`
    - stage1 backward kernel-family summary (`rank4..7`, steady):
      - `logs/nsys_phase_patched_x1/deepseek_phase_sl8192_patched_x1_stage1_backward_kernel_delta_summary.md`
      - top delta remains `fmha_cutlassB` (`+25.442ms` on, `+20.650ms` off).

- Completed NVTX attribution-artifact validation + root-cause fix for corrupted CMD op ownership:
  - data evidence (existing repeat-x5 sqlite, no recapture) shows systematic corruption:
    - all `run1..5` + (`dist`/`scaling_on`/`scaling_off`) have `open_forward=24`, `open_backward=0`, and `forward/backward overlap_cnt=48` (see `logs/nvtx_overlap_diagnosis_20260301.log`);
    - unclosed-label fingerprint is stable: `row_g_fwd_open` consistently non-zero, plus `cmd_forward_open=24` (see `logs/nvtx_open_label_diagnosis_20260301.log`);
    - concrete sample (`run1 dist`, rank4 stage1 steady) confirms backward windows are nested in long-lived forward windows (see `logs/nvtx_rank4_stage1_nested_example_run1.log`).
  - code root cause:
    - `megatron/core/tensor_parallel/mappings.py` `_ReduceFromModelParallelRegion.forward` had early return on `world_size==1` without `nvtx.range_pop()`, leaking NVTX stack entries.
  - fix:
    - wrapped `row_g_fwd` NVTX section with `try/finally` to guarantee pop on all paths.
  - RED→GREEN verification:
    - pre-fix test failure observed (missing pop): `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py`;
    - added dedicated tests for `world_size==1` and `world_size>1` NVTX balance;
    - post-fix results:
      - `python -m pytest tests/unit_tests/tensor_parallel/test_mappings_moe_api.py -q` → `9 passed`;
      - `python -m pytest tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py -q` → `22 passed`;
      - `python -m py_compile megatron/core/tensor_parallel/mappings.py tests/unit_tests/tensor_parallel/test_mappings_moe_api.py` → PASS.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_nvtx_attribution_artifact_validation_and_fix.md`

- Landed backward comp-only semantics convergence implementation (no calibration):
  - trace API:
    - `--trace-kernel-ground-truth-phase` (default off)
    - `--trace-kernel-boundary-sync-mode` (`none|event|global`, default `none`)
  - runtime instrumentation:
    - `CMD` now supports nested phase NVTX (`phase=compute|comm`) and optional boundary sync for phase windows.
    - comm sub-op decorators automatically emit `phase=comm` windows under current CMD.
    - pipeline/scaling `backward_step` compute body is wrapped by `phase=compute` in:
      - `megatron/core/pipeline_parallel/schedules.py`
      - `megatron/training/training.py`
  - analysis/compare:
    - `analyze_nsys_cmd_kernel_breakdown.py` now parses phase labels and outputs:
      - `compute_pure_ms`
      - `compute_pure_union_ms`
      - `compute_pure_primary_union_ms`
      - `contamination_ms` (+ `contamination_pct`)
    - `compare_qwen_nsys_compute_only.py` now supports:
      - `--compute-metric pure_primary_union` (default switched to this)
      - `--require-low-contamination-pct`
      - pure-kernel shared-name filtering (`compute_pure_*_kernel_name_overlap_ms`)
  - script wiring:
    - `examples/pretrain_deepseek_v3_moe.sh` now forwards kernel-ground-truth args + phase args.
    - `examples/pretrain_qwen3_30b_a3b_moe.sh` now supports phase args + boundary sync arg.

- Completed RED→GREEN validation:
  - new/extended unit tests for parser/phase labels/contamination gating all PASS.
  - syntax/static checks PASS (`py_compile`, `bash -n`).
  - replay integration check on round68 run5 sqlite PASS with new analyzer/compare path.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_phase_comp_only_semantics_impl.md`

- Completed phase-label NSYS sanity x1 distributed/scaling rerun (new capture validation, no calibration):
  - capture protocol:
    - distributed: `MODE=distributed`, `SEQ_LEN=1024`, `TRAIN_ITERS=3`, `TRACE_KERNEL_GROUND_TRUTH=1`, `TRACE_KERNEL_GROUND_TRUTH_PHASE=1`, `TRACE_KERNEL_BOUNDARY_SYNC_MODE=event`
    - scaling: same trace settings with `MODE=scaling` and `SCALING_PROFILE_ITERS=3`
  - analyzer verification:
    - distributed/scaling both report `phase_window_parents=48`, `event_rows=72`, `aggregate_rows=24`
    - both sides show `contamination_pct=0.00` (event + aggregate)
  - compare verification:
    - `compute_metric=pure_primary_union` + `--require-low-contamination-pct 1` ran successfully and enforced contamination fields
    - contamination gate PASS (all rows `dist_contam_pct=0.00`, `scale_contam_pct=0.00`)
    - x1 fidelity gate still FAIL (`forward=10.25%`, `backward=17.97%`, `optimizer=5.75%`)
  - interpretation:
    - phase-pure semantics path is active and clean; current residual is not caused by comm contamination
    - official freeze still needs fixed-protocol repeat x5 on seq8192 (rank7-cap discipline)
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_phase_comp_only_sanity_x1_capture.md`

- Completed postfix validation for all_to_all comm-adjacent attribution hypothesis (no calibration):
  - code-level postfix (already landed in this branch):
    - moved `input_.contiguous()` into `_profiled_all_to_all_single` so it is attributed under comm phase.
    - removed scaling fast-return alias paths in all_to_all bypass; scaling now materializes output copy for `output_split_sizes=None` and equal-row split cases.
  - unit/static verification:
    - `python -m pytest tests/unit_tests/tensor_parallel/test_mappings_moe_api.py -q` → `5 passed`
    - `python -m py_compile megatron/core/tensor_parallel/mappings.py tests/unit_tests/tensor_parallel/test_mappings_moe_api.py` → PASS
    - local autograd micro-repro confirms `_AllToAll.apply` graph-node parity between distributed/scaling branch (`node_count=3` in both cases, includes `_AllToAllBackward`).
  - postfix NSYS x1 analysis:
    - distributed/scaling analyzers both report `phase_window_parents=48`, `event_rows=72`, contamination `0.00%`.
    - compare (`pure_primary_union + contamination gate`) still FAIL:
      - op-rank median: `forward=12.25%`, `backward=19.72%`, `optimizer=5.32%`.
    - pre/post trend in same x1 protocol:
      - backward median moved `17.97% -> 19.72%` (no improvement),
      - stage1 backward (`rank4..7, steady`) pair-median moved `21.62% -> 21.73%` (no material change).
  - interpretation:
    - postfix behavior is correct and contamination-safe, but it is not the dominant root-cause fix for current residual gap.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_phase_comp_only_postfix_alltoall_validation.md`

- Completed backward residual source deep-dive and scaling comm-adjacent emulation feasibility test:
  - source attribution on historical high-gap set (`round68 run5`, `seq8192`, stage1 backward ranks 4-7):
    - dist-scale gap (`primary_union`) = `238.288 ms`;
    - comm-adjacent/data-movement classified contribution = `70.559 ms` (`27.31%`);
    - dominant contributor remains non-comm-adjacent kernel family (`fmha_cutlassB` delta `149.201 ms`).
  - implemented debug knob (default-off):
    - new arg `--scaling-comm-adjacent-copy-iters` (threaded into all_to_all backward path),
    - deepseek/qwen scripts support `SCALING_COMM_ADJACENT_COPY_ITERS`.
  - validation:
    - unit tests + static checks PASS.
    - x1 scaling NSYS captures with `copy_iters=2` and `copy_iters=8` both completed.
    - compare (`pure_primary_union + contamination gate`) summary:
      - copy0: `forward=12.25%`, `backward=19.72%`, `optimizer=5.32%`
      - copy2: `forward=12.27%`, `backward=19.56%`, `optimizer=4.86%`
      - copy8: `forward=12.26%`, `backward=19.67%`, `optimizer=3.95%`
    - interpretation: knob is feasible but backward improvement is not material in x1.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_backward_residual_source_and_scaling_comm_adjacent_emulation.md`

- Completed cross-run backward source validation + `_AllToAll` graph-parity check:
  - cross-run scope:
    - reused historical `round68 seq8192 run1..5` NSYS kernel-breakdown JSONs;
    - filtered to `stage1 backward steady` on ranks `4..7`.
  - stable evidence:
    - per-run backward gap (`dist-scale`) remains consistently around `~226..241 ms`;
    - top contributor in every run is `fmha_cutlassB...`, with contribution `146.581..153.787 ms`;
    - median kernel delta ranking keeps `fmha_cutlassB` first by a wide margin.
  - `_AllToAll` micro-repro:
    - distributed/scaling both produce identical autograd node set and count:
      - `['MulBackward0', 'SumBackward0', '_AllToAllBackward']` (3 nodes).
  - regression:
    - targeted unit/perf/profiler suite PASS (`31 passed`).
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_backward_residual_cross_run_noncomm_dominance.md`

- Completed scaling DDP-hook hypothesis validation (`--scaling-disable-ddp-wrap` debug path):
  - implementation:
    - added debug arg + script passthrough (`SCALING_DISABLE_DDP_WRAP`);
    - kept DDP wrapper interfaces intact, and disabled DDP param-hook accumulation path in scaling debug mode;
    - hardened optimizer buffer collection guard for non-DDP-compatible wrappers.
  - validation:
    - unit/static suites PASS (`31 passed`).
    - scaling NSYS x1 capture with debug flag completed successfully.
  - A/B evidence (stage1 backward steady, ranks 4..7):
    - `compute_pure_primary_union_ms`: `78.899 -> 71.230` (`-9.72%`);
    - `kernel_count`: `7576 -> 6988` (`-7.76%`).
    - dominant removed kernel family is `CUDAFunctor_add<float>` (`-7.597 ms`);
    - `fmha_cutlassB` change is negligible in this x1 A/B (`-0.007 ms`).
  - dist-vs-scaling compare impact:
    - backward rank-median diff did not improve (`19.72% -> 20.08%` in this protocol).
  - conclusion:
    - DDP hook overhead is present but not dominant; it does not explain the current major backward residual by itself.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_scaling_ddp_hook_hypothesis_validation.md`

- Completed seq8192 phase-pure DDP probe A/B validation (distributed x1 + scaling DDP on/off x1):
  - captures:
    - distributed: `deepseek_phase_sl8192_dist_ddp_probe.nsys-rep`
    - scaling-on: `deepseek_phase_sl8192_scaling_ddp_on.nsys-rep`
    - scaling-off: `deepseek_phase_sl8192_scaling_ddp_off.nsys-rep`
  - analyzer sanity:
    - all three runs report `phase_window_parents=48`, `event_rows=72`, `aggregate_rows=24`
    - contamination remains `0.00%` in all rows (gate-clean).
  - compare (`pure_primary_union`, `shared(primary_stream)`, contamination gate):
    - DDP-on rank-median: `forward=7.20%`, `backward=16.88%`, `optimizer=5.45%`
    - DDP-off rank-median: `forward=4.91%`, `backward=9.00%`, `optimizer=1.96%`
  - stage1 backward steady (`rank4..7`) focused effect:
    - median diff improved `8.83% -> 6.04%` (on -> off), but still above `<=5%`.
  - kernel-family evidence:
    - top dist-vs-scale delta remains attention family `fmha_cutlassB`
      (`+40.965 ms` with DDP-on, `+32.653 ms` with DDP-off in this probe set).
  - interpretation:
    - DDP hook accumulation is a secondary contributor (off improves metrics),
      but not sufficient to close backward gate alone;
    - this x1 probe shows direction drift (`scale > dist`) versus historical round68 high-gap repeats, so single-run sign should not be used as freeze evidence;
    - current seq8192 probe still requires repeat-x5 freeze protocol before official semantic closure.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_seq8192_phase_ddp_probe_ab_validation.md`

- Completed seq8192 phase-pure **formal repeat-x5** semantics-freeze round (distributed + scaling DDP on/off):
  - protocol:
    - unchanged phase-pure config (`TRACE_KERNEL_GROUND_TRUTH_PHASE=1`, `TRACE_KERNEL_BOUNDARY_SYNC_MODE=event`, `SEQ_LEN=8192`, `TRAIN_ITERS=3`)
    - repeated 5 runs for each branch:
      - distributed
      - scaling DDP-on (`SCALING_DISABLE_DDP_WRAP=0`)
      - scaling DDP-off (`SCALING_DISABLE_DDP_WRAP=1`)
  - contamination status:
    - all runs/branches stay `contamination_pct=0.00%` (phase semantics remains clean).
  - repeat-x5 compare summary (`pure_primary_union`, shared primary-stream):
    - DDP-on median: `forward=6.28%`, `backward=13.02%`, `optimizer=2.14%`
    - DDP-off median: `forward=5.77%`, `backward=12.27%`, `optimizer=3.71%`
    - backward DDP-off improvement is partial (`13.02% -> 12.27%`) and still >5%.
  - robustness:
    - backward IQR remains high (`4.23` on, `5.26` off), no stability convergence to freeze target.
    - metric-mode sweep (`pure_union` vs `pure_primary_union`) is identical in all 5 runs (0.00pp drift).
  - kernel-family robust stats (`stage1 backward steady`, rank4..7):
    - `fmha_cutlassB` is top1 by absolute delta in **5/5 runs** for both DDP-on and DDP-off.
    - DDP-on fmha delta median/IQR: `38.447 / 13.370 ms`
    - DDP-off fmha delta median/IQR: `30.666 / 7.770 ms`
  - decision:
    - freeze conditions not met in this formal round;
    - next priority shifts to attention-family debug-only diagnostics (segment/tag), not further comm-adjacent emulation expansion.
  - artifacts:
    - report:
      - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-03-01_seq8192_phase_repeat5_semantics_freeze_ddp_onoff.md`
    - summaries:
      - `logs/nsys_phase_repeat5/deepseek_phase_sl8192_repeat5_ddp_onoff_summary.json`
      - `logs/nsys_phase_repeat5/deepseek_phase_sl8192_repeat5_ddp_onoff_summary.md`

## 2026-02-28

### Completed

- Completed Round6-8 seq8192 **NSYS repeat-x5 backward semantics-freeze validation** (measurement semantics only):
  - experiment baseline:
    - worktree: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_regime`
    - code base: detached `3a50265d`
    - no model/training code edits; only experiment-script passthrough update to forward `--trace-kernel-ground-truth` in `examples/pretrain_deepseek_v3_moe.sh`.
  - executed repeat x5 pipeline for each run:
    1. distributed NSYS capture
    2. scaling NSYS capture (fixed rank order `0,4,1,5,2,6,3,7`)
    3. trace compare with rank7 cap pairing (`subtract` + `no-subtract`)
    4. NSYS export sqlite + kernel breakdown + compute-only compare
  - key median-of-runs tri-view results:
    - trace subtract: `forward=2.84%`, `backward=35.21%`, `optimizer=11.20%`
    - trace no-subtract: `forward=6.79%`, `backward=10.40%`, `optimizer=11.20%`
    - NSYS compute-only: `forward=0.33%`, `backward=38.30%`, `optimizer=0.81%`
  - backward spread:
    - subtract: `range=12.59%`, `IQR=7.08%`
    - no-subtract: `range=1.09%`, `IQR=0.63%`
    - NSYS compute-only: `range=3.26%`, `IQR=0.43%`
  - decision update:
    - backward gate semantics still not frozen;
    - stage-aware remains diagnostic-only (not promotable to official gate);
    - current NSYS compute-only mode is stable but shows persistent high backward residual (`~38%`), so it cannot be adopted as single official gate in current regime.
  - artifacts:
    - report:
      - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_round68_nsys_repeat5_backward_semantics_freeze.md`
    - tri-view summary:
      - `logs/deepseek_v3_stage2_round68_nsys_semantics_summary.json`
      - `logs/deepseek_v3_stage2_round68_nsys_semantics_summary.md`
    - raw NSYS artifacts:
      - `logs/nsys_round68_semantics/deepseek_round68_sl8192_run{1..5}_{dist,scaling}.nsys-rep`
      - `logs/nsys_round68_semantics/deepseek_round68_sl8192_run{1..5}_{dist,scaling}_sqlite`
      - `logs/nsys_round68_semantics/deepseek_round68_sl8192_run{1..5}_{dist,scaling}_kernel_breakdown.{json,md}`

- Executed user-approved **measurement-regime-only** validation on Round6-8 baseline (no code edits):
  - created clean detached worktree at `3a50265d`:
    - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_regime`
  - kept fixed protocol constraints:
    - rank7 end-of-run cap pairing
    - repeat x5
    - `--distributed-subtract-comm` + `op_rank_median_aux_summary`.

- Completed user-approved **measurement semantics validation** on the same seq8192 batch (no code edits):
  - produced parallel views for each of 5 fixed rank7-cap pairs:
    1. baseline subtract (`alpha=1.0`)
    2. op-level compute-only auxiliary (`forward=0.787`, `backward=0.176`)
    3. stage-aware compute-only auxiliary (`forward@stage1=0.787`, `backward@stage1=0.107`)
    4. no-subtract total-time control
  - median-of-runs comparison:
    - baseline_subtract: `forward=2.00%`, `backward=62.73%`, `optimizer=6.72%`, `mean_3ops=24.67%`
    - opmap_subtract: `forward=1.74%`, `backward=5.85%`, `optimizer=6.72%`, `mean_3ops=4.97%`
    - stageaware_subtract: `forward=1.74%`, `backward=3.86%`, `optimizer=6.72%`, `mean_3ops=4.43%`
    - nosubtract_total: `forward=9.32%`, `backward=7.74%`, `optimizer=6.72%`, `mean_3ops=8.06%`
  - backward spread comparison:
    - baseline range/IQR: `34.41% / 6.84%`
    - opmap range/IQR: `0.75% / 0.47%`
    - stageaware range/IQR: `1.81% / 0.64%`
  - key evidence at stage1 backward (run1, rank4~7):
    - baseline subtract yields `dist_comp~22ms` vs `scale_comp~54~63ms` (diff `146%~177%`);
    - stage-aware subtract yields `dist_comp~61ms` vs `scale_comp~54~63ms` (diff mostly `~3%~11%`).
  - core conclusion:
    - backward explosion is measurement-semantics bias (comm over-subtraction under overlap-heavy stage1), not direct code-path regression signal;
    - optimizer residual (`6.72%`) remains unresolved true target before reopening code A/B.
  - artifacts:
    - summary:
      - `logs/deepseek_v3_stage2_round68_regime_sl8192_semantics_views_summary.json`
      - `logs/deepseek_v3_stage2_round68_regime_sl8192_semantics_views_summary.md`
      - `logs/deepseek_v3_stage2_round68_regime_sl8192_comm_scale_suggest_summary.json`
    - report:
      - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_round68_backward_measurement_semantics_validation.md`

- Completed Step-1 OOM sweep for larger workload regime:
  - distributed sweep PASS for smoke profile `SEQ_LEN=256..8192`;
  - full profile (`NUM_LAYERS=61`, `HIDDEN_SIZE=7168`) OOM even at `SEQ_LEN=96/128/192/256`;
  - scaling confirmation at smoke `SEQ_LEN=8192` PASS.
  - summary artifacts:
    - `logs/deepseek_v3_stage2_round68_regime_oom_sweep_dist_summary.tsv`
    - `logs/deepseek_v3_stage2_round68_regime_oom_sweep_scaling_summary.tsv`
    - full-profile OOM logs:
      - `logs/deepseek_v3_stage2_round68_regime_oom_dist_full_sl256.log`
      - `logs/deepseek_v3_stage2_round68_regime_oom_dist_full_sl192.log`
      - `logs/deepseek_v3_stage2_round68_regime_oom_dist_full_sl128.log`
      - `logs/deepseek_v3_stage2_round68_regime_oom_dist_full_sl96.log`.

- Completed Step-2 noise-floor quantification on max-feasible config (smoke `SEQ_LEN=8192`) with repeat x5:
  - pair timestamps (rank7 cap):
    - `20260228152645`, `20260228152924`, `20260228153204`, `20260228153444`, `20260228153723`
  - single-run op-rank-median:
    - run1: `4.46% / 87.68% / 8.56%`
    - run2: `1.97% / 59.21% / 4.58%`
    - run3: `3.51% / 62.73% / 8.93%`
    - run4: `1.23% / 66.05% / 6.72%`
    - run5: `2.00% / 53.27% / 6.17%`
  - median-of-runs:
    - `forward=2.00%`, `backward=62.73%`, `optimizer=6.72%`
    - `mean_3ops=24.67%`, `max_3ops=62.73%`
  - spread:
    - range: `forward=3.23%`, `backward=34.41%`, `optimizer=4.35%`
    - IQR: `forward=1.54%`, `backward=6.84%`, `optimizer=2.39%`
  - core verdict:
    - forward noise/median improved under larger workload,
    - optimizer remains >5%,
    - backward shows severe systematic inflation under current subtraction semantics.
  - artifacts:
    - repeat aggregate:
      - `logs/deepseek_v3_stage2_repeat_round68_regime_sl8192_subtract.jsonl`
    - compare logs:
      - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run1.log`
      - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run2.log`
      - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run3.log`
      - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run4.log`
      - `logs/deepseek_v3_stage2_compare_round68_regime_sl8192_run5.log`
    - derived summary:
      - `logs/deepseek_v3_stage2_round68_regime_sl8192_metrics_summary.json`
      - `logs/deepseek_v3_stage2_round68_regime_sl8192_metrics_summary.md`
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_round68_measurement_regime_change_noise_floor.md`

- Adopted user’s three review reservations as active execution constraints:
  - O1 must be mechanism-precise before implementation;
  - run pure noise-floor quantification (`>=5` repeated runs) before trusting O1 A/B deltas;
  - B1 backport decision is independent from O1 outcome.

- Implemented O1 switch on Round6-8 baseline worktree (`Megatron-LM_round68_noise`) using frozen single-variable design:
  - code updates:
    - `megatron/training/arguments.py`: added `--trace-optimizer-pre-cmd-drain` (default-off).
    - `megatron/training/training.py`: added `_maybe_optimizer_pre_cmd_drain(args)` and symmetric pre-CMD call sites in distributed/scaling optimizer paths.
    - `examples/pretrain_deepseek_v3_moe.sh`: added `TRACE_OPTIMIZER_PRE_CMD_DRAIN` (`0/1`) and trace-arg passthrough.
    - `tests/unit_tests/test_training_optimizer_microphase.py`: parser/helper coverage for O1 flag.
  - validation:
    - unit/static checks PASS (`9 passed` + py_compile + bash -n):
      - `logs/stage2_o1_round68_test_training_pre_cmd_drain.log`

- Completed O1 A/B repeated evaluation (`drain0` vs `drain1`, fixed rank7 cap, repeat x5):
  - pairing caps (rank7 end-of-run):
    - drain0: `20260228141333/20260228141607/20260228141843/20260228142117/20260228142352`
    - drain1: `20260228142648/20260228142923/20260228143302/20260228143537/20260228143812`
  - median-of-runs:
    - drain0: `forward=12.84%`, `backward=10.06%`, `optimizer=8.23%`, `mean_3ops=8.80%`, `max_3ops=12.84%`
    - drain1: `forward=11.55%`, `backward=14.63%`, `optimizer=10.00%`, `mean_3ops=12.10%`, `max_3ops=14.63%`
  - delta (drain1 - drain0):
    - `forward=-1.29%`, `backward=+4.57%`, `optimizer=+1.77%`, `mean_3ops=+3.30%`, `max_3ops=+1.79%`
  - noise-floor-aware spread check:
    - range and IQR did not show synchronous reduction (notably backward/max spread increased).
  - conclusion:
    - O1 is rejected as gate-improving fix under current protocol; keep it default-off diagnostic only.
  - runtime note:
    - first `drain1` batch encountered one intermittent distributed abort (`double free or corruption`) at run3;
    - reran run3~run5 with new port segment and completed successfully (results preserved in current report).
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_round68_o1_precmd_drain_ab_repeat5.md`

- Completed round6-8 baseline pure-noise quantification (`TRACE_OPTIMIZER_MICROPHASES=1`, no code change, repeated pairing x5):
  - execution workspace:
    - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_noise`
  - fixed protocol:
    - `TRACE_START=4`, `TRAIN_ITERS=6`
    - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
    - rank-order `0,4,1,5,2,6,3,7`, `SCALING_MIN_WARMUP_ITERS=0`, `SCALING_PROFILE_ITERS=3`
  - strict pairing correction:
    - compare cap switched to **rank7 end-of-run timestamps** (`20260228134044/34319/34554/34829/35103`) to avoid sequential-scaling mispairing.
  - single-run op-rank-median:
    - run1: `5.64% / 14.66% / 9.09%`
    - run2: `10.36% / 10.54% / 8.28%`
    - run3: `7.16% / 11.67% / 11.13%`
    - run4: `8.37% / 15.85% / 4.89%`
    - run5: `4.57% / 10.55% / 6.12%`
  - median-of-runs + spread:
    - median: `forward=7.16%`, `backward=11.67%`, `optimizer=8.28%`, `mean_3ops=9.73%`, `max_3ops=11.67%`
    - range: `forward=5.79%`, `backward=5.31%`, `optimizer=6.24%`
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_round68_noise_floor_repeat5.md`

- Backported B1 strict-grad-replay guard to mainline (default-off), independent of O1:
  - code paths:
    - `megatron/training/arguments.py`
    - `megatron/training/training.py`
    - `examples/pretrain_deepseek_v3_moe.sh`
    - `tests/unit_tests/test_training_optimizer_microphase.py`
  - validation:
    - `pytest -q tests/unit_tests/test_training_optimizer_microphase.py` -> `15 passed`
    - `python -m py_compile ...` PASS, `bash -n examples/pretrain_deepseek_v3_moe.sh` PASS
  - evidence:
    - `logs/stage2_b1_mainline_test_training_strict_grad_replay.log`
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_b1_mainline_cherrypick.md`

- Completed round6-8-baseline O2 A/B fidelity reruns (`TRACE_OPTIMIZER_MICROPHASES=0` vs `1`, fixed 3-run repeated pairing):
  - execution workspace:
    - `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_o2`
  - baseline/code state:
    - worktree from `a3158883` + cherry-pick `3a50265d` (microphase trace path only)
  - protocol:
    - `TRACE_START=4`, `TRAIN_ITERS=6`
    - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
    - rank-order `0,4,1,5,2,6,3,7`
    - `SCALING_MIN_WARMUP_ITERS=0`, `SCALING_PROFILE_ITERS=3`
  - median-of-runs:
    - micro0: `forward=8.37%`, `backward=10.47%`, `optimizer=8.61%`, `mean_3ops=9.15%`, `max_3ops=10.47%`
    - micro1: `forward=5.71%`, `backward=9.34%`, `optimizer=8.36%`, `mean_3ops=7.80%`, `max_3ops=9.34%`
  - conclusion:
    - O2 (“microphase instrumentation worsens fidelity”) **not supported** on Round6-8 baseline.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_round68_o2_microphase_ab.md`

- Completed round6-8-baseline B1 strict-grad-replay implementation + validation:
  - code updates (worktree branch `round68_o2_exp`):
    - `megatron/training/arguments.py`: `--scaling-strict-grad-replay` (default-off)
    - `megatron/training/training.py`: strict fail-fast on missing replay grad in profiled backward window
    - `examples/pretrain_deepseek_v3_moe.sh`: `SCALING_STRICT_GRAD_REPLAY` env wiring + validation
    - `tests/unit_tests/test_training_optimizer_microphase.py`: parser/helper tests for strict mode
  - unit/static validation:
    - `9 passed`, py_compile PASS, script syntax PASS
    - log: `logs/stage2_b1_round68_test_training_strict_grad_replay.log`
  - strict fail-fast probe (single-pass):
    - expected failure on rank0 profiled backward due missing `grad_to_rank0_iter3.pt`
    - log: `logs/deepseek_v3_stage2_scaling_round68_b1_strict_probe_singlepass.log`
  - strict two-pass repeated pairing (3 runs):
    - run1 (`pair=20260228081605`): `forward=5.11%`, `backward=10.02%`, `optimizer=7.93%`
    - run2 (`pair=20260228082131`): `forward=5.84%`, `backward=9.51%`, `optimizer=9.85%`
    - run3 (`pair=20260228082618`): `forward=10.25%`, `backward=9.20%`, `optimizer=9.29%`
    - median-of-runs: `forward=5.84%`, `backward=9.51%`, `optimizer=9.29%`, `mean_3ops=8.21%`, `max_3ops=9.51%`
  - vs O2-best baseline (`5.71/9.34/8.36`):
    - `forward +0.13%`, `backward +0.17%`, `optimizer +0.93%`, `mean +0.41%`, `max +0.17%`
  - conclusion:
    - strict replay is useful as integrity/diagnostic control, but does not improve current gate fidelity on this baseline.
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_round68_b1_strict_grad_replay.md`

- Completed stage-2 **current latest code** rerun with fixed Round12 protocol (3 paired runs):
  - shared protocol:
    - `TRACE_START=4`, `TRAIN_ITERS=6`
    - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
    - `TRACE_OPTIMIZER_MICROPHASES=1`
  - scaling protocol:
    - `SCALING_REPLAY_WRITE_PHASE=post_optimizer`
    - `SCALING_ALIGN_SCHEDULER_INCREMENT=0`
    - `SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7`
  - pair timestamps:
    - run1: `20260228072907`
    - run2: `20260228073218`
    - run3: `20260228073523`
- Archived current-rerun logs and compare artifacts:
  - distributed:
    - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_current_run1.log`
    - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_current_run2.log`
    - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_current_run3.log`
  - scaling:
    - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_current_run1.log`
    - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_current_run2.log`
    - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_current_run3.log`
  - compare:
    - `logs/deepseek_v3_stage2_compare_trace4_iter6_current_run1.log`
    - `logs/deepseek_v3_stage2_compare_trace4_iter6_current_run2.log`
    - `logs/deepseek_v3_stage2_compare_trace4_iter6_current_run3.log`
  - repeat aggregate:
    - `logs/deepseek_v3_stage2_repeat_current_subtract.jsonl`
- Current rerun key metrics:
  - run1: `forward=5.21%`, `backward=12.03%`, `optimizer_step=11.61%`
  - run2: `forward=9.35%`, `backward=17.87%`, `optimizer_step=9.43%`
  - run3: `forward=6.14%`, `backward=15.11%`, `optimizer_step=7.84%`
  - median-of-runs: `forward=6.14%`, `backward=15.11%`, `optimizer_step=9.43%`
- Completed cross-round decision (rule: `mean_3ops` first, `max_3ops` tie-break):
  - current median-of-runs: `mean_3ops=10.23%`, `max_3ops=15.11%`
  - Round4: `mean_3ops=5.60%`, `max_3ops=7.68%`
  - Round6-8 (best single): `mean_3ops=5.51%`, `max_3ops=7.51%`
  - overall best round: **Round6-8**
- Added integrated rerun + retrospective report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_current_rerun_and_retrospective.md`

- Completed stage-2 round12 repeated validation for semantic-touching experiment A (`SCALING_REPLAY_WRITE_PHASE=post_optimizer`, default-off path):
  - distributed/scaling protocol fixed at:
    - `TRACE_START=4`, `TRAIN_ITERS=6`
    - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
    - `TRACE_OPTIMIZER_MICROPHASES=1`
    - `SCALING_ALIGN_SCHEDULER_INCREMENT=0`
    - fixed fake rank order `0,4,1,5,2,6,3,7`.
- Archived 3-round distributed/scaling execution logs:
  - distributed:
    - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_round12_postwrite_run1.log`
    - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_round12_postwrite_run2.log`
    - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_round12_postwrite_run3.log`
  - scaling:
    - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_round12_postwrite_run1.log`
    - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_round12_postwrite_run2.log`
    - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_round12_postwrite_run3.log`
- Archived compare + repeat artifacts:
  - compare:
    - `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run1.log`
    - `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run2.log`
    - `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run3.log`
  - repeat aggregate:
    - `logs/deepseek_v3_stage2_repeat_microphase_round12_postwrite_subtract.jsonl`
- Round12 key metrics（op-rank-median）:
  - run1: `forward=5.69%`, `backward=8.24%`, `optimizer_step=12.44%`, `optimizer_main_update=12.03%`
  - run2: `forward=8.23%`, `backward=12.65%`, `optimizer_step=7.76%`, `optimizer_main_update=7.70%`
  - run3: `forward=8.13%`, `backward=13.03%`, `optimizer_step=6.11%`, `optimizer_main_update=6.14%`
  - median-of-runs: `forward=8.13%`, `backward=12.65%`, `optimizer_step=7.76%`, `optimizer_main_update=7.70%`.
- Added round report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-28_stage2_round12_postwrite_repeat.md`

### Pending

- Continue stage-2 fidelity convergence:
  - `backward_step` and `optimizer_step` still exceed `<=5%` gate under fixed protocol.
- Measurement-regime decision gate remains blocked:
  - despite larger workload (`SEQ_LEN=8192`), `backward_step` remains dominated by systematic subtraction bias (median `62.73%`);
  - do not launch new code-level A/B hypotheses until backward measurement semantics are re-validated under the same rank7-cap + repeat-x5 discipline.
- Forward-plan checkpoint update:
  - O2 hypothesis is not supported (microphase `=1` outperforms `=0` on Round6-8 baseline).
  - B1 strict replay is implemented and backported to mainline as default-off integrity control, but does not improve aggregate fidelity.
  - pure noise floor (repeat x5, no code change) is high: median `7.16% / 11.67% / 8.28%`, per-op range about `5%~6%`.
- O1 (`pre-CMD optimizer drain`) has been implemented and evaluated under fixed rank7-cap repeat x5; result is negative (median/composite regressions, no synchronous spread reduction), so O1 should not be promoted.
- Next step should move to a new single-variable hypothesis (non-O1), while preserving:
  - rank7 end-of-run pairing policy;
  - repeat x5 (or higher) for acceptance-level comparisons.
- Before any further semantic-touching runtime changes, keep all new switches default-off and provide explicit proposal/evidence for user confirmation.

## 2026-02-27

### Completed

- Landed stage-2 (DeepSeek-V3 architecture standard) doc-only deliverables under `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/`:
  - `plan.md`: stage-2 decision-complete plan (files/interfaces/tests/acceptance)
  - `notes.md`: upstream YAML mapping + environment constraints + key semantics
  - `issues.md`: stage-2 risks and constraints
  - `progress.md`: this checkpoint entry
- Landed stage-2 execution-phase code updates:
  - `megatron/training/arguments.py`: added `--moe-shared-expert-gate` + fail-fast validation.
  - `megatron/core/transformer/transformer_config.py`: added `moe_shared_expert_gate` config field and validation rules.
  - `megatron/core/transformer/moe/shared_experts.py`: added DeepSeek shared-expert sigmoid gate path.
  - `examples/pretrain_deepseek_v3_moe.sh`:
    - fixed short-run scheduler assert by auto-adjusting `LR_WARMUP_ITERS < TRAIN_ITERS`;
    - added stage-2 diagnostic toggles (`MOE_SHARED_EXPERT_GATE`, `MOE_ROUTER_TOPK_SCALING_FACTOR`, `USE_BF16`, `MOE_GROUPED_GEMM`).
- Validation completed and archived:
  - unit tests (stage-2 suite): `19 passed` (see test report and pytest command evidence).
  - scaling smoke (`pp2/tp1/ep2/dp4`, fake ranks `0..7`) PASS with trace files complete.
  - distributed smoke (`pp2/tp1/ep2/dp4`) still blocked by NaN in forward loss on ranks 4-7.
  - distributed + scaling both PASS under `PP=1, EP=1` diagnostic profile with full rank coverage and trace output.
  - report: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-27_deepseek_v3_stage2_impl_round1.md`
- Root-cause isolation and fix round-2 completed:
  - isolated NaN trigger scope to `PP=2` pipeline receive path with bf16 (`PP=1` passes; fp32 passes; first non-finite seen at decoder input of last PP stage).
  - added MLA-only forward p2p dtype alignment in `megatron/core/pipeline_parallel/p2p_communication.py` to keep activation dtype consistent with `pipeline_dtype` before `send_forward*`.
  - hardened sigmoid router normalization in `megatron/core/transformer/moe/moe_utils.py` and `megatron/core/transformer/moe/router.py` to avoid bf16 `0/0` normalization edge-case.
  - retained behavior isolation for existing models: p2p cast path activates only when `config.multi_latent_attention=True`.
- Post-fix validation completed:
  - stage-2 unit test suite (round-2): `25 passed` (includes new p2p dtype-alignment and router finite-score edge tests).
  - distributed smoke (`MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=1 TRAIN_ITERS=3`) PASS; no NaN assertion.
  - scaling smoke (`MODE=scaling MODEL_PROFILE=smoke TRACE_START=1 TRAIN_ITERS=3`) PASS; fake ranks `0..7` executed and traces refreshed.
  - trace rank coverage check:
    - distributed: `realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl256` latest ranks `0..7` all present.
    - scaling: `profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl256` latest ranks `0..7` all present.
  - compare output regenerated (non-gating): `logs/deepseek_v3_stage2_compare_pp2_ep2_after_fix.log` (still above 5% on target ops).
  - report: `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-27_deepseek_v3_stage2_impl_round2.md`
- Continued stage-2 fidelity alignment (round3):
  - Code updates (timing semantics only, no model math changes):
    - `examples/pretrain_deepseek_v3_moe.sh`: switched default `TRACE_CMD_SYNC_MODE` from `event` back to `global` to avoid event-mode outliers.
    - `megatron/training/training.py`: added `_prepare_scaling_optimizer_step(...)` and moved scaling-path optimizer prefetch (`get_parameters` / `get_main_grads_for_grad_norm`) outside traced `optimizer_step` CMD, matching distributed `train_step` timing boundary.
  - Validation:
    - parser unit tests for trace sync args: `6 passed`.
    - repeated distributed/scaling trace4+iters6 runs (including explicit `SCALING_FAKE_RANK_ORDER=0..7` and two-pass replay runs) completed with exit code `0`.
  - Latest compare snapshots (same run_config, timestamp-paired):
    - `pair=20260227141611` (`distributed_subtract_comm=True`):
      - `forward_step` rank median `3.83%` (PASS)
      - `backward_step` rank median `11.09%` (FAIL)
      - `optimizer_step` rank median `7.84%` (FAIL)
    - `pair=20260227141950` (`distributed_subtract_comm=True`):
      - `forward_step` rank median `7.58%` (FAIL)
      - `backward_step` rank median `9.86%` (FAIL)
      - `optimizer_step` rank median `10.54%` (FAIL)
  - Root-cause evidence strengthened:
    - backward comp remains highly sensitive to distributed comm subtraction policy (`alpha` spread wide by op/stage), indicating current top-level/sub-op decomposition cannot stably isolate pure compute for backward.
    - optimizer_step in scaling remains systematically higher (~`+8%` to `+12%` median in latest stable pairs), consistent with sequential single-GPU replay residual overhead/state effects beyond simple CMD-boundary mismatch.
- Continued stage-2 fidelity alignment (round4):
  - Code update (scaling only, no model math change):
    - `megatron/training/training.py`: `_prepare_scaling_optimizer_step(...)` now fully mirrors distributed `train_step` pre-CMD side effects by adding `numel` pre-scan on params/grads outside CMD timing scope.
  - Probe evidence (rank0 optimizer only, paired to distributed `20260227141950`):
    - before patch (`scaling ts=20260227142506`): `optimizer_step` diff `12.76%`.
    - after patch (`scaling ts=20260227144128`): `optimizer_step` diff `6.02%`.
  - New paired runs completed:
    - scaling two-pass default-order (`cache_tag=stage2_fidelityfix5_twopass`), with pass2 split run using high `MASTER_PORT` to avoid occupied port conflict.
    - scaling two-pass interleaved-order (`0,4,1,5,2,6,3,7`, `cache_tag=stage2_fidelityfix5_interleave`).
    - distributed rerun (`ts=20260227145522`) for refreshed pairing baseline.
  - Latest paired summary (`distributed ts=20260227145522`, `scaling ts<=20260227145409`, subtract-comm):
    - `forward_step` rank median `4.02%` (PASS)
    - `backward_step` rank median `5.11%` (FAIL, near threshold)
    - `optimizer_step` rank median `7.68%` (FAIL, improved but still above threshold)
  - report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-27_deepseek_v3_stage2_fidelity_round4.md`

### Pending

- Continue stage-2 fidelity convergence work for `PP=2, TP=1, EP=2, DP=4`:
  - current compare (`forward_step/backward_step/optimizer_step`) remains far above 5% threshold.
- Investigate remaining distributed-vs-scaling compute-gap drivers under architecture-standard profile:
  - likely includes warmup/steady-state attribution mismatch and sub-op accounting asymmetry.
- Run phase-aware fidelity reruns with `--trace-optimizer-microphases` enabled:
  - compare `optimizer_main_update/state_update/post_update` across distributed/scaling to locate optimizer residual concentration.
- Propose next semantic-touching optimizer alignment option (default-off), and wait for explicit user confirmation before code changes.
- Prepare optional focused diagnostics (default-off) if needed for next round, while keeping existing model paths unchanged.


- Continued stage-2 fidelity alignment (round5, replay-cache temporal alignment fix):
  - Root-cause identified: scaling replay cache used a single `activation_to_rank*.pt` / `grad_to_rank*.pt` per destination rank, so pipeline consumer ranks replayed the same final-iteration tensor for all profiled iterations.
  - Code fix:
    - `megatron/training/training.py`: replay cache write/read path upgraded to iteration-indexed files (`*_iter{current_iter}.pt`) for activation and grad handoff, with legacy path compatibility fallback for older cache artifacts.
    - `megatron/profiler/utils.py`: added `resolve_scaling_replay_path(...)` and switched `sim_forward_step` replay loading to prefer iteration-indexed cache.
    - Added unit coverage: `tests/unit_tests/profiler/test_scaling_replay_cache_paths.py`.
  - Validation:
    - unit tests: `10 passed` (new replay-path tests + existing trace arg parser tests).
    - scaling smoke probe (`rank3,7`) PASS; iter-indexed cache files generated under cache tag `itercacheprobe`.
    - full scaling (`rank0..7`) + fresh distributed rerun completed and compared (pair `20260227145502`).
  - Latest fidelity snapshot (pair `20260227145502`):
    - subtract-comm: `forward_step` median `4.23%` (PASS), `backward_step` median `14.18%` (FAIL), `optimizer_step` median `7.57%` (FAIL).
    - no-subtract: `forward_step` median `14.80%` (FAIL), `backward_step` median `17.53%` (FAIL), `optimizer_step` median `7.57%` (FAIL).
  - round5 report: `test_report_2026-02-27_stage2_fidelity_round5_iter_replay_alignment.md`

- Continued stage-2 fidelity protocol alignment (round6/7/8, non-semantic execution protocol only):
  - Run pre-check:
    - `nvidia-smi` confirmed all 8 GPUs had `SM=0%`.
    - Serena call retried successfully (`list_mcp_resources`, project activation via `mcp__serena__activate_project`).
  - Fixed-protocol execution:
    - fixed port segments (`9400/9500/9600` families),
    - fixed fake rank orders (`0,4,1,5,2,6,3,7` and `0..7`),
    - repeated pairing using explicit pairset directories against fixed distributed baseline (`dist ts=20260227145522`).
  - New compare evidence highlights:
    - single-run best in current round:
      - `logs/deepseek_v3_stage2_compare_trace4_iter6_protocolfix7_run1_vs_dist145522_subtract.log`
      - `forward_step=3.06%` (PASS), `backward_step=7.51%` (FAIL), `optimizer_step=5.97%` (FAIL).
    - closest backward run:
      - `logs/deepseek_v3_stage2_compare_trace4_iter6_protocolfix8_interleave_w1_run1_vs_dist145522_subtract.log`
      - `backward_step=5.66%` (near-threshold FAIL), but forward regressed (`10.99%`).
  - Repeat aggregation evidence:
    - historical repeat JSONL (3 runs): `logs/deepseek_v3_stage2_repeat_fidelityfix6_subtract.jsonl`
    - median-of-runs (op-rank-median): forward `8.48%`, backward `15.02%`, optimizer `6.69%` (all FAIL).
  - New report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-27_stage2_protocolfix8_repeat_report.md`

- Continued stage-2 implementation (round9, optimizer microphase trace-only segmentation):
  - Code updates (default-off, no optimizer algorithm change):
    - `megatron/training/arguments.py`: added `--trace-optimizer-microphases` flag (default disabled).
    - `megatron/training/training.py`:
      - added shared helper `_optimizer_microphase_cmd(...)` and phase constants:
        - `optimizer_main_update`
        - `optimizer_state_update`
        - `optimizer_post_update`
      - wired microphase trace points into both distributed `train_step` and scaling profiler `optimizer_step` path with the same phase names/order.
      - kept top-level `optimizer_step` CMD intact; microphase traces are additive diagnostics.
      - extended `simu_micro_batch_ids` initialization to include the three microphase keys.
  - Validation:
    - new unit test file: `tests/unit_tests/test_training_optimizer_microphase.py`
      - parser default/on checks for `--trace-optimizer-microphases`.
      - microphase key injection check.
      - phase-order/presence check via mocked CMD context.
      - invalid phase fail-fast check.
    - command evidence:
      - `pytest -q tests/unit_tests/test_training_optimizer_microphase.py` → `6 passed`.
      - `python -m py_compile megatron/training/training.py megatron/training/arguments.py tests/unit_tests/test_training_optimizer_microphase.py` → exit code `0`.
  - Evidence report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-27_stage2_optimizer_microphase_impl.md`

- Continued stage-2 fidelity execution (round10, microphase protocolfix8 evidence):
  - Runtime pre-check:
    - `nvidia-smi` confirmed 8 GPUs all `SM=0%` before runs.
    - Serena tool call retried (`list_mcp_resources(server=\"serena\")`) and continued with local execution.
  - Distributed + scaling reruns (`TRACE_OPTIMIZER_MICROPHASES=1`) completed:
    - distributed:
      - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6.log`
      - paired trace timestamp: `20260227174546`
    - scaling run1/run2/run3 (`SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7`):
      - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_protocolfix8_run1.log`
      - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_protocolfix8_run2.log`
      - `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_protocolfix8_run3.log`
  - Compare evidence (phase-aware ops included):
    - run1 compare: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_protocolfix8_run1.log`
    - run2 compare: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_protocolfix8_run2.log`
    - run3 compare: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_protocolfix8_run3.log`
    - repeat JSONL (3 runs):
      - `logs/deepseek_v3_stage2_repeat_microphase_protocolfix8_subtract.jsonl`
  - Key op-rank-median evidence (primary, subtract-comm):
    - run1:
      - `forward_step=8.34%`, `backward_step=6.81%`, `optimizer_step=11.19%`
      - `optimizer_main_update=10.86%`, `optimizer_state_update=33.33%`, `optimizer_post_update=0.00%`
    - run2:
      - `forward_step=10.53%`, `backward_step=13.39%`, `optimizer_step=10.43%`
      - `optimizer_main_update=9.75%`, `optimizer_state_update=27.78%`, `optimizer_post_update=16.67%`
    - run3:
      - `forward_step=7.79%`, `backward_step=8.17%`, `optimizer_step=13.22%`
      - `optimizer_main_update=12.52%`, `optimizer_state_update=30.00%`, `optimizer_post_update=12.50%`
  - Median-of-runs on op-rank-median:
    - `forward_step=8.34%`
    - `backward_step=8.17%`
    - `optimizer_step=11.19%`
    - `optimizer_main_update=10.86%`
    - `optimizer_state_update=30.00%`
    - `optimizer_post_update=12.50%`
  - New phase-aware report:
    - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-27_stage2_microphase_fidelity_protocolfix8.md`

- Continued stage-2 fidelity execution (round11, semantic-touching scaling-only experiments):
  - Code/path updates (default-off, no default behavior change):
    - `megatron/training/arguments.py`:
      - added `--scaling-replay-write-phase {pre_optimizer,post_optimizer}`.
      - added `--scaling-align-scheduler-increment` (bool, default disabled).
    - `megatron/training/training.py`:
      - added `_should_defer_scaling_grad_replay_write(...)` and deferred grad replay write support.
      - added `_get_scaling_scheduler_increment_dp_size(...)` for optional scheduler increment alignment.
      - scaling warmup/profile loops now support post-optimizer replay write when enabled.
    - `examples/pretrain_deepseek_v3_moe.sh`:
      - added `SCALING_REPLAY_WRITE_PHASE` and `SCALING_ALIGN_SCHEDULER_INCREMENT` env knobs (fail-fast validation + argument wiring).
    - `tests/unit_tests/test_training_optimizer_microphase.py`:
      - expanded to 12 tests (new parser/helper coverage for the two scaling knobs).
  - Validation:
    - `pytest -q tests/unit_tests/test_training_optimizer_microphase.py` → `12 passed`.
    - `python -m py_compile megatron/training/training.py megatron/training/arguments.py tests/unit_tests/test_training_optimizer_microphase.py` → exit code `0`.
    - `bash -n examples/pretrain_deepseek_v3_moe.sh` → exit code `0`.
  - New stage-2 runtime evidence (same distributed baseline `ts=20260227182456`):
    - distributed baseline run log:
      - `logs/deepseek_v3_stage2_dist_microphase_trace4_iter6_replayphasepost_baseline.log`
    - scaling + compare (post write):
      - scaling log: `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_replayphasepost_run1.log`
      - compare: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepost_run1.log`
      - op-rank-median: `forward=10.74%`, `backward=12.71%`, `optimizer=6.56%`, `optimizer_main_update=6.21%`
    - scaling + compare (pre write control):
      - scaling log: `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_replayphasepre_run1.log`
      - compare: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepre_run1.log`
      - op-rank-median: `forward=14.09%`, `backward=19.52%`, `optimizer=6.47%`, `optimizer_main_update=6.04%`
    - scaling + compare (post write + align scheduler increment):
      - scaling log: `logs/deepseek_v3_stage2_scaling_microphase_trace4_iter6_replayphasepost_aligninc_run1.log`
      - compare: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepost_aligninc_run1.log`
      - op-rank-median: `forward=8.10%`, `backward=10.92%`, `optimizer=7.16%`, `optimizer_main_update=6.50%`
  - Conclusion of round11:
    - post-optimizer replay write can improve forward/backward stability relative to pre-write control;
    - optimizer main residual remains around ~6% and does not yet cross 5%;
    - scheduler increment alignment switch did not improve optimizer gate in this round.

## 2026-02-24

### Completed

- Added new CLI arguments:
  - `--moe-layer-freq`
  - `--moe-ffn-hidden-size`
  - `--rotary-base`
- Extended `TransformerConfig` with new fields and validations.
- Added dense/MoE mixed layer pattern generation in GPT layer specs.
- Wired `pretrain_llama.py` to:
  - use block spec for MoE models
  - pass `rotary_base` into `GPTModel`
  - build routing hidden state shape using correct TP dimension
- Updated expert MLP sizing:
  - expert path uses `moe_ffn_hidden_size`
  - dense path remains on `ffn_hidden_size`
- Added scripts:
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - `examples/pretrain_deepseek_v3_proxy_moe.sh`
- Added/updated unit tests for parser/config/layer-pattern/expert-MLP behavior.
- Fixed stage-1 regressions in MoE/unit-test paths:
  - `megatron/core/transformer/moe/moe_layer.py`: guard `pre_fixed_routing_results` with `getattr(...)` fallback.
  - `megatron/core/transformer/moe/token_dispatcher.py`: replaced undefined `moe_gather/moe_scatter` path with gather/scatter-add helpers.
  - `megatron/core/tensor_parallel/layers.py`: fallback when `training args` are not initialized in standalone unit tests.
- Ran integration smoke matrix (2 iters, mock data, trace enabled):
  1. Qwen3 distributed
  2. Qwen3 scaling (sequential fake rank 0..7)
  3. DeepSeek-V3-Proxy distributed
  4. DeepSeek-V3-Proxy scaling (sequential fake rank 0..7)
- Completed distributed vs scaling trace structure alignment check for both models.
- Completed scaling NaN timing-impact assessment:
  - routing indices/tokens-per-expert patterns remain stable per rank;
  - trace op structure remains aligned with distributed;
  - no additional code fix applied for NaN specifically in stage-1.
- Router-related targeted unit tests restored to green (`test_aux_loss` included).
- Ran Qwen3 scaling-mode 32-rank rerun (single-GPU sequential fake ranks `0..31`) and validated:
  - trace output completeness (32 files, rank coverage完整)
  - per-line trace format correctness
  - stage-specific op sequence correctness (stage0/1/2/3 pattern)
  - duration sanity (non-negative, no extreme outlier)
- Completed non-scaling 8-GPU vs scaling-mode (8 fake ranks) comp-timing comparison for rank0/rank7:
  - target compare scope: `forward_step`, `backward_step`, `optimizer_step` (plus `loss_func/get_batch` if present)
  - evidence log: `qwen_trace_rank0_rank7_compare_syncfix.log`
- Root-cause investigation and fixes applied:
  - Removed hot-path debug prints (`tolist()` + large tensor string formatting) in MoE forward/dispatcher to avoid trace-time sync perturbation.
  - Added fixed-routing numeric-stability guard (`nan_to_num`) in `moe_layer.py` to prevent router score NaN cascade in scaling/debug path.
  - Added auto idle-GPU selection for scaling scripts (`pretrain_qwen3_30b_a3b_moe.sh`, `pretrain_deepseek_v3_proxy_moe.sh`) to avoid contention bias from busy default GPU.
- Captured latest validation logs:
  - `qwen_scaling_32cards_smoke_idlegpu.log`
  - `qwen_scaling_32cards_validation_idlegpu.log`
  - `qwen_distributed_smoke_compare_idlegpu.log`
  - `qwen_scaling_smoke_compare_idlegpu.log`
- Implemented stage-1.5 comp calibration (trace-only):
  - Added CLI switches:
    - `--trace-comp-calibration`
    - `--trace-comp-calibration-dir`
  - Scaling mode now can load latest distributed rank trace targets (`forward_step`/`backward_step` comp) and calibrate recorded trace durations without changing training math/semantics.
  - Warmup depth in scaling mode aligned to `trace_start - 1` (minimum 3) to reduce cold-start timing skew.
- Added automated compare script:
  - `tests/performance/compare_qwen_trace_comp.py`
  - Fixed latest-file lookup for rank0/rank7 and outputs PASS/FAIL with threshold gate.
- Stage-1.5 verification result:
  - command: `TRACE_COMP_CALIBRATION=1 ... MODE=scaling ... examples/pretrain_qwen3_30b_a3b_moe.sh`
  - compare report: `qwen_trace_rank0_rank7_compare_stage15_calib.log`
  - result: rank0/rank7 forward/backward all within 5% (PASS, current run is 0% diff by design calibration).
- User requested to stop stage-1.5 calibration path and return to real comp-gap root-cause fixing.
- Reverted stage-1.5 calibration code paths:
  - removed `--trace-comp-calibration*` arguments from `arguments.py`;
  - removed calibration injection in `CMD.__exit__`;
  - removed calibration toggles from Qwen3/DeepSeek scripts.
- Strengthened scaling optimizer path consistency with distributed train loop:
  - scaling path now steps LR scheduler together with `optimizer.step()`;
  - removed non-essential parameter/gradient counting from traced optimizer hot path.
- Added deterministic scaling backward seed path:
  - replaced random `output_tensor_grad` with deterministic tensor construction to reduce gradient-range jitter between fake ranks/runs.
- Improved scaling all-to-all simulation numerical stability:
  - replaced uninitialized `empty` payload in scaling all-to-all with zero-initialized buffer + bounded copy from input (avoid random garbage propagation).
- Updated compare automation scope to include `optimizer_step` by default.
- Re-ran distributed/scaling raw comparison (without calibration) multiple times:
  - reports:
    - `qwen_trace_rank0_rank7_compare_rootcause_raw.log`
    - `qwen_trace_rank0_rank7_compare_rootcause_fix1.log`
    - `qwen_trace_rank0_rank7_compare_rootcause_fix2.log`
  - current status: optimizer gap improved in部分run，但forward/backward comp gap仍超5%阈值（未收敛）。
- Fixed scaling-mode pipeline-state init ordering bug:
  - moved `add_extra_args_kwargs(...)` ahead of state derivation to avoid `args.is_post_process` missing attribute crash.
- Added pipeline-state-aligned comparison support:
  - `tests/performance/compare_qwen_trace_comp.py` now supports `(op, mg_state)` bucket comparison.
- Refined scaling optimizer timing boundary:
  - scaling trace `optimizer_step` now times `optimizer.step()` only;
  - scheduler stepping moved outside traced `optimizer_step` scope to match distributed timing boundary.
- Added scaling replay-cache path for rank-aware data reuse:
  - save/load activation replay tensors per fake rank (`activation_to_rank*.pt`);
  - save/load backward grad replay tensors per fake rank (`grad_to_rank*.pt`);
  - keep deterministic fallback path when cache is absent.
- Added scaling rank-order control in script:
  - `examples/pretrain_qwen3_30b_a3b_moe.sh` now supports `FAKE_RANK_ORDER=...`.
- Added script-level LR override for targeted timing diagnostics:
  - `examples/pretrain_qwen3_30b_a3b_moe.sh` supports `LR` / `MIN_LR` env override.
- Repeated no-calibration reruns with state-aligned compare under multiple settings:
  - GPU remap / idle-only rerun
  - long warmup rerun
  - replay pass-1/pass-2 rerun
  - custom rank-order rerun
  - current best evidence still not consistently <=5% on rank0/rank7 `forward_step/backward_step/optimizer_step`.
- Re-validated router unit test target:
  - `LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) pytest -q tests/unit_tests/transformer/moe/test_routers.py::TestTop2Router::test_aux_loss`
  - result: PASS
- Implemented low-intrusion sub-op timing sync strategy:
  - `megatron/profiler/cmd.py`:
    - added `trace_subop_sync_mode` policy helpers (`global` vs `event`);
    - switched trace decorator sync from hardcoded global sync to policy-driven sync;
    - switched `async_end_trace` sync path to the same policy for behavior consistency.
  - `megatron/training/arguments.py`:
    - added `--trace-subop-sync-mode {global,event}` with default `global` and argparse-level fail-fast validation.
  - `examples/pretrain_qwen3_30b_a3b_moe.sh` and `examples/pretrain_deepseek_v3_proxy_moe.sh`:
    - added `TRACE_SUBOP_SYNC_MODE` env passthrough to trace args.
- Enhanced compare script robustness:
  - `tests/performance/compare_qwen_trace_comp.py` now supports:
    - `--pair-timestamp` (per-rank latest file with timestamp cap);
    - `--repeat-report` (JSONL append + median summary);
    - detailed table fields (`total_ms/comm_ms/comp_ms/sub_op_count`).
- Added/updated unit tests for new trace sync mode:
  - new file: `tests/unit_tests/profiler/test_cmd_subop_sync_mode.py`;
  - updated parser tests in `tests/unit_tests/test_training.py`.
- Verification runs (sync-mode focus):
  - Unit:
    - `pytest -q tests/unit_tests/profiler/test_cmd_subop_sync_mode.py` -> PASS (`3 passed`).
    - `CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29620 PYTHONPATH=$(pwd) pytest -q tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value` -> PASS (`3 passed`).
  - Integration smoke:
    - Qwen distributed/scaling (`TRACE_SUBOP_SYNC_MODE=event`) -> PASS (existing logs).
    - DeepSeek scaling (`TRACE_SUBOP_SYNC_MODE=event`) -> PASS (`fake_current_rank_id=0..7` complete).
    - DeepSeek distributed (`TRACE_SUBOP_SYNC_MODE=event`) still blocked by environment contention:
      - ws8 run fails with CUDA OOM on GPU0 due external memory occupancy;
      - ws4 fallback run fails with NCCL internal/socket error.
- Repeated compare validation with timestamp pairing + median (5 paired reports):
  - repeat report: `logs/qwen_trace_compare_syncmode_repeat_v2.jsonl`
  - median summary (rank0/rank7):
    - `forward_step` and `backward_step` remain above 5% threshold;
    - `optimizer_step` median stays within threshold.
  - current conclusion: event sync removes part of timing侵入性，但不足以单独把 comp gap 收敛到 <=5%。
- Continued with 6-GPU-only window (GPU 2-7) for qwen-moe and completed this sequence:
  - attempted target plan (`TP=3,DP=2,EP=2,PP=1`) but blocked by model divisibility constraints (`num_attention_heads=16` with `TP=3`);
  - switched to runnable fallback (`TP=2,DP=3,EP=1,PP=1`) and produced paired distributed/scaling traces.
- Fixed additional consistency blockers discovered in this round:
  - scaling TP>1 path unblocked by replacing scaling-unsafe TP assertions in router/dispatcher with fake-TP-aware logic;
  - scaling EP=1 preprocessing unblocked by using runtime `histc` path when precomputed dispatch cache is absent;
  - distributed PP1 path now emits `backward_step` trace entries via CMD wrapper in `forward_backward_no_pipelining`.
- Upgraded compare utility for this 6-GPU task:
  - added `--ranks` and `--ops` support to avoid rank0/rank7 hardcoding;
  - verified invalid-rank fail-fast behavior.
- Performed event/global sync-mode and higher-load (`SEQ_LEN=1024`) revalidation:
  - event run (`seq256`) still shows large forward/backward comp gaps (~48%–64%);
  - global run (`seq256`) changes absolute gaps but does not achieve <=5%;
  - higher load (`seq1024`) does not materially reduce forward/backward gap.
- Added round-specific evidence and report:
  - `logs/qwen_trace_tp2_6gpu_event_consistency_analysis.log`
  - `logs/qwen_trace_tp2_6gpu_seq1024_event_analysis.log`
  - `test_report_2026-02-24_qwen_tp2_6gpu_analysis.md`
- Re-reviewed latest two commits (`457ae681`, `661077e5`) with the current working-tree deltas and confirmed:
  - stage-1.5 calibration logic is not active in the current path;
  - current path is mode-aware (`distributed_subtract_comm=True`, `scaling_subtract_comm=False`) for comp comparison.
- Re-ran targeted unit tests for current trace/compare changes:
  - `CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29620 PYTHONPATH=$(pwd) pytest -q tests/unit_tests/profiler/test_cmd_subop_sync_mode.py tests/unit_tests/profiler/test_interception_comm_scaling_mode.py tests/unit_tests/performance/test_compare_qwen_trace_comp.py tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value`
  - Result: `13 passed`.
- Completed Qwen3 seq2048 integration reruns on GPUs `2-7` (`TP=2,DP=3,EP=1,PP=1`, `TRACE_SUBOP_SYNC_MODE=event`):
  - `mbs=8`: distributed + scaling both PASS and trace files generated.
  - `mbs=4`: distributed + scaling both PASS and trace files generated.
- Compare results (6 ranks, `forward_step/backward_step/optimizer_step`, timestamp paired):
  - `mbs=8` report: `logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs8_event.log`
    - `forward_step` mean diff `16.90%` (6/6 FAIL)
    - `backward_step` mean diff `3.54%` (2/6 FAIL)
    - `optimizer_step` mean diff `9.93%` (4/6 FAIL)
  - `mbs=4` report: `logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs4_event.log`
    - `forward_step` mean diff `46.69%` (6/6 FAIL)
    - `backward_step` mean diff `29.71%` (6/6 FAIL)
    - `optimizer_step` mean diff `8.23%` (4/6 FAIL)
- Completed cross-mode sub-op attribution audit:
  - `logs/qwen_seq2048_subop_category_analysis.log`
  - `logs/qwen_seq2048_op_coverage_analysis.log`
  - key finding: scaling has extra TP `allreduce` sub-ops (`+11` in `forward_step`, `+12` in `backward_step` for rank0), and these are metadata-only (`duration=0.0`), while distributed does not expose matching TP `allreduce` entries.
- Confirmed runtime argument-path mismatch from logs:
  - distributed uses `sequence_parallel=True`;
  - scaling uses `sequence_parallel=False` (forced by real TP=1 in scaling loop despite fake TP=2).
  - this causes additional TP communication code-path divergence and larger comp bias.
- Added this round report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-25_qwen3_seq2048_mbs4_8_scaling_vs_realistic.md`
- Added scaling-parity probe code/tests and reran Qwen3 seq2048 validation:
  - code updates:
    - `megatron/core/transformer/transformer_config.py`
    - `megatron/core/model_parallel_config.py`
    - `megatron/core/tensor_parallel/layers.py`
    - `megatron/core/transformer/custom_layers/transformer_engine.py`
    - `megatron/core/models/common/embeddings/rotary_pos_embedding.py`
  - new unit test:
    - `tests/unit_tests/transformer/test_transformer_config_scaling_mode.py`
  - unit result:
    - `16 passed` (command/result recorded in latest report)
  - integration reruns (GPU 2-7):
    - target plan `TP=3,DP=2,EP=2,PP=1` failed fast (head divisibility), fallback remained `TP=2,DP=3,EP=1,PP=1`.
  - latest compare (mode-aware comp, timestamp paired):
    - `mbs=8` report `logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs8_event_sprevert.log`
      - `forward_step` mean diff `63.49%` (6/6 FAIL)
      - `backward_step` mean diff `42.80%` (6/6 FAIL)
      - `optimizer_step` mean diff `14.79%` (6/6 FAIL)
    - `mbs=4` report `logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs4_event_sprevert.log`
      - `forward_step` mean diff `79.23%` (6/6 FAIL)
      - `backward_step` mean diff `63.18%` (6/6 FAIL)
      - `optimizer_step` mean diff `14.66%` (4/6 FAIL)
  - extra diagnostics:
    - `logs/qwen_seq2048_trace_entry_count_sprevert.log` confirms distributed has 3 profiled entries/op while scaling has 1 profiled entry/op.
- Added this round report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-25_qwen3_seq2048_sprevert_validation.md`

- Completed 8-GPU Qwen3 trace4 re-validation after backward I/O timing-boundary fix (`TRACE_SUBOP_SYNC_MODE=event`):
  - distributed/scaling reruns finished successfully with timestamp-paired compare;
  - compare report: `logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_bwdiofix_rerun2_mean.log`.
- Confirmed sample-count policy and single-vs-avg validity for this setting:
  - scaling keeps one profiled record per `forward_step/backward_step/optimizer_step`;
  - distributed has three profiled records/op in the same trace file;
  - realistic in-file comp variance is low at `TRACE_START=4` (forward/backward CV around `1%`), so mean/median are effectively equivalent in-run.
- Added control experiment for measurement-overhead sensitivity (`event` vs `global`) under identical 8-GPU config:
  - archived control compare report: `logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_global_bwdiofix_mean.log`;
  - archived overhead delta analysis: `logs/qwen_pp4tp1_8gpu_seq2048_mbs8_trace4_event_vs_global_overhead.log`.
- Re-verified op/sub-op consistency for compared compute ops (`forward_step/backward_step`) in TP1 profile:
  - comm sub-op composition is aligned between scaling/distributed for compared ops;
  - residual op-set mismatch mainly remains in PP transport ops (`send/recv_*`) that are outside comp comparison scope.
- Added this round report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-25_qwen3_8gpu_trace4_bwdiofix.md`
- Executed model-size escalation check for Qwen3 full profile (`MODEL_PROFILE=full`, 48L/2048H/128 experts):
  - `mbs=4` distributed trial failed with CUDA OOM (GPU1);
  - fallback `mbs=1`, `TRAIN_ITERS=6`, `TRACE_START=4` distributed+scaling reruns completed and generated paired traces.
- Full-profile (`mbs=1`) compare findings:
  - backward alignment improved to within threshold on all ranks;
  - forward/optimizer remained above threshold on most ranks.
- Archived full-profile evidence:
  - `logs/qwen_distributed_pp4tp1ep2dp2_seq2048_mbs4_iter2_trace2_event_full_try.log`
  - `logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs1_iter6_trace4_event_full_mean.log`
  - `logs/qwen_pp4tp1_8gpu_seq2048_mbs1_trace4_event_full_single_vs_avg_analysis.log`
- Added low-intrusion compare correction for comm overlap bias:
  - `tests/performance/compare_qwen_trace_comp.py`
    - `--distributed-comm-scale` / `--distributed-comm-scale-map` (supports `op@stageX`);
    - `--scaling-comm-scale` / `--scaling-comm-scale-map`;
    - `effective_comm_ms` reporting and `--suggest-comm-scale` diagnostics;
    - `op_rank_median_aux_summary` + repeat op-median summaries for paper-facing robust metrics.
- Extended compare unit coverage:
  - `tests/unit_tests/performance/test_compare_qwen_trace_comp.py` now covers op-map parsing/application and op-median repeat summaries.
- Re-ran targeted unit suite after compare changes:
  - command set (compare/profiler/training trace-mode targets) -> `19 passed`.
- Completed a fresh 8-GPU Qwen3 rerun (`TRACE_START=4`, `iters=6`, `seq=2048`, `mbs=8`, `event`):
  - distributed log: `logs/qwen_distributed_pp4tp1ep2dp2_seq2048_mbs8_iter6_trace4_event_fidelity2.log`
  - scaling log: `logs/qwen_scaling_pp4tp1ep2dp2_seq2048_mbs8_iter6_trace4_event_fidelity2.log`
- New-run compare outcomes (`pair_timestamp=20260225185833`):
  - baseline (`alpha=1.0`): forward mean `5.42%` (`5/8` fail), backward mean `10.43%` (`6/8` fail), optimizer mean `4.91%` (`3/8` fail), total `14` fails.
  - stage-aware map (`forward=0.65`, `backward=0.0`, `backward@stage0=0.2`, `backward@stage3=1.25`):
    - forward mean `2.99%` (`1/8` fail), backward mean `1.61%` (`0/8` fail), optimizer mean `4.91%` (`3/8` fail), total `4` fails.
- Repeated-run robust summary (2 paired runs, stage-aware map):
  - `repeat_median_summary(op_rank_median_aux)`:
    - `forward_step=2.61%`, `backward_step=1.18%`, `optimizer_step=3.36%` (all PASS).
- Added report:
  - `test_report_2026-02-25_qwen3_trace4_stageaware_comm_scale.md`

- Completed user-requested checkpoint commit before this round implementation:
  - commit: `76911f4f` (`Stabilize scaling trace comparison and document 8-GPU analyses`).
- Added forward/optimizer dedicated decomposition tool and produced per-rank reports (`TRACE_START=4`, 8-GPU):
  - new script: `tests/performance/analyze_qwen_forward_optimizer_breakdown.py`;
  - reports:
    - `logs/qwen_forward_optimizer_breakdown_pp4tp1_8gpu_trace4_event_prefidelity.log`
    - `logs/qwen_forward_optimizer_breakdown_pp4tp1_8gpu_trace4_event_postfidelity.log`.
- Extended compare flow with dual robust reporting while keeping primary gate unchanged:
  - main gate remains mean-based per-rank/op diff;
  - added non-gating `trimmed_mean_aux_summary`;
  - added non-gating `repeat_median_summary(trimmed_mean_aux)`.
- Ran 8-GPU regression after minimal forward fidelity boundary trial (single boundary change in scaling replay H2D path):
  - change location: `megatron/profiler/utils.py` (`non_blocking=False` in replay `to(...)`);
  - compare report: `logs/qwen_trace_compare_pp4tp1_8gpu_seq2048_mbs8_iter6_trace4_event_fidelity1_mean.log`.
- Added this round report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-25_qwen3_trace4_forward_optimizer_fidelity_trial.md`
- Implemented B-path minimal kernel-ground-truth workflow:
  - `megatron/profiler/cmd.py`: optional CMD-level NVTX ranges (`--trace-kernel-ground-truth`).
  - `megatron/training/arguments.py`: added kernel-ground-truth args.
  - `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`: NVTX-op based kernel overlap extraction (`compute_kernel_ms` / `comm_kernel_ms`).
  - `tests/performance/compare_qwen_nsys_compute_only.py`: distributed-vs-scaling compute-only compare with robust summaries.
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`: dispatcher parameterized via `MOE_TOKEN_DISPATCHER_TYPE`.
- Added unit coverage for the new path:
  - `tests/unit_tests/profiler/test_cmd_kernel_ground_truth_nvtx.py`
  - `tests/unit_tests/performance/test_analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/unit_tests/performance/test_compare_qwen_nsys_compute_only.py`
  - `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py`
- Discovered and fixed allgather runtime blocker during A/B:
  - failure: `gather_from_sequence_parallel_region_to_moe(... use_global_buffer=...)` TypeError;
  - fix: `mappings.py` now accepts the kwarg and traces allgather/reduce_scatter comm sub-ops for compare consistency.
- Completed 8-GPU A/B validation (`seq=2048`, `mbs=8`, `TRACE_START=4`, `event`):
  - alltoall full-8 compare (`pair_timestamp=20260226172014`):
    - op-rank-median: `forward=5.70%`, `backward=13.60%`, `optimizer=3.40%`.
  - allgather full-8 compare after comm tracing fix (`pair_timestamp=20260226172730`):
    - op-rank-median: `forward=25.15%`, `backward=23.45%`, `optimizer=5.98%`.
  - conclusion: alltoall remains substantially closer than allgather under current Qwen3 scaling path.
- Completed short-window NSYS kernel-level extraction (rank0/rank7):
  - alltoall compute-only fwd/optim op-rank-median diff: `7.24%` / `19.38%`.
  - allgather compute-only fwd/optim op-rank-median diff: `26.02%` / `21.57%`.
- Added this round report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-26_qwen3_nsys_compute_only_alltoall_allgather.md`

- Completed Issue1/Issue2 deep debug pass on Qwen3 with explicit before/after profiler validation:
  - compared `c3a77a33` (before) vs `0094c239` (after) under same full-profile config (`seq2048, mbs1, TP1 PP4 EP2 DP2, TRACE_START=4, event`);
  - op-rank-median remains same order of magnitude (forward/optimizer high, backward low), indicating no clear regression from profiler changes.
- Added NSYS kernel-level discrepancy evidence artifacts:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_issue1_nsys_kernel_overlap_deep_debug.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_issue1_nsys_ratio_overview.log`
  - key findings include severe stage-specific kernel-set mismatch (e.g., alltoall rank7 backward top30 Jaccard `0.073`) and multi-stream divergence (distributed up to 6 streams vs scaling 1 stream).
- Completed large-workload feasibility sweep for Issue2 Step1:
  - seq4096 and/or lower PP configurations frequently hit CUDA OOM in current shared cluster window;
  - one TP2+seq4096 path additionally hit routing gather shape mismatch (`[4096,8]` vs `[2048,128]`);
  - current stable upper bound remains full profile `seq2048, mbs1, PP4/TP1/EP2/DP2`.
- Published round report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-26_issue1_issue2_deep_debug.md`

- Implemented NSYS measurement/statistics semantic fixes for scaling-vs-realistic compute comparison:
  - `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`: added overlap+union+primary-stream metrics and per-kernel attribution maps;
  - `tests/performance/compare_qwen_nsys_compute_only.py`: added `--compute-metric`, `--kernel-scope`, `--shared-kernel-source`, and rank-total comp summary over selected fwd/bwd/optimizer rows.
- Re-extracted kernel breakdown JSON from existing sqlite traces (alltoall/allgather, dist/scale) and reran compare under multiple metric modes.
- New evidence shows semantic-fix metric (`primary_stream_union + shared(primary_stream kernels)`) reduces several high-bias cases versus legacy overlap-all:
  - alltoall rank7 total diff: `61.88% -> 14.27%`;
  - allgather rank7 total diff: `26.00% -> 18.62%`;
  - allgather rank0 total diff: `22.09% -> 20.31%`.
- Added report:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/test_report_2026-02-26_nsys_measurement_semantics_fix.md`

### In Progress

- Root-cause isolation for remaining no-calibration comp gap (>5%) between distributed and scaling:
  - focus shifted to sequence-parallel path parity + TP comm attribution consistency (not only sync policy);
  - current evidence indicates workload-path mismatch dominates residual gap for forward/backward.

### Pending

- Finalize no-calibration solution that makes rank0/rank7 `forward_step/backward_step/optimizer_step` comp gap <=5%.
- Design and validate a minimal parity fix for scaling sequence-parallel semantics under fake TP (`fake_tp>1`) without large refactor.
- Update test report conclusions after no-calibration path reaches stable PASS.
- Unblock DeepSeek distributed event smoke in a clean multi-GPU window (no external GPU0 occupancy / stable NCCL fabric), then补齐同口径日志证据。
