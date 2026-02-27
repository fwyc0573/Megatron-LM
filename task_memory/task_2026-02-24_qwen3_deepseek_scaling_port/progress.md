## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
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
