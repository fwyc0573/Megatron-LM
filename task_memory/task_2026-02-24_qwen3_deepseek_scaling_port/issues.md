## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-27 | Added Issue 35 for stage-2 residual fidelity risks after iter-replay alignment fix |
| 2026-02-24 | Added stage-1 blockers/risks and mitigation notes |
| 2026-02-24 | Added validation-time findings (scaling NaN router scores and unit-test harness constraints) |
| 2026-02-24 | Updated with NaN timing-impact conclusion and router test-path fixes |
| 2026-02-24 | Added 32-rank scaling validation findings and rank0/rank7 comp-timing gap root-cause notes |
| 2026-02-24 | Added stage-1.5 calibration dependency notes and compare-script automation |
| 2026-02-24 | Replaced calibration risk notes with raw comp-gap root-cause findings |
| 2026-02-24 | Added pipeline-state compare and rank-aware replay rerun findings (no-calibration still unstable) |
| 2026-02-24 | Added trace sync-mode rollout findings, compare pairing/median results, and DeepSeek distributed event-mode blockers |
| 2026-02-24 | Added 6-GPU TP2 consistency findings: sub-op attribution mismatch, no-pipeline backward trace gap fix, and higher-load revalidation |
| 2026-02-25 | Added Qwen3 seq2048 mbs4/8 6-GPU findings: sequence-parallel path mismatch and TP allreduce attribution drift |
| 2026-02-25 | Added scaling-parity probe findings (RoPE mismatch under forced SP), trace-entry count mismatch, and latest high-variance compare evidence |
| 2026-02-25 | Added 8-GPU trace4 rerun findings after backward I/O timing fix, plus full-profile model-size escalation OOM/alignment evidence |
| 2026-02-25 | Added forward/optimizer decomposition + compare trimmed-mean auxiliary report findings and minimal forward-fidelity trial outcome |
| 2026-02-25 | Added stage-aware comm-overlap correction findings, new 8-GPU rerun evidence, and robust op-median metric recommendation |
| 2026-02-26 | Added kernel-ground-truth NSYS findings, alltoall-vs-allgather A/B results, and allgather comm-path consistency fix status |
| 2026-02-26 | Added Issue1/Issue2 deep-debug findings: NSYS kernel-set/stream mismatch evidence and before/after profiler verdict |
| 2026-02-26 | Added NSYS semantics-fix findings: shared-primary metric improvement and residual mismatch status |
| 2026-02-27 | Added stage-2 (DeepSeek-V3 architecture standard) risks: TE version mismatch, SDPA mask semantics, fixed-routing parity, expert-bias update integration, MCP serena unavailable |
| 2026-02-27 | Added stage-2 execution findings: distributed PP2 bf16 NaN blocker, PP1 diagnostic pass, and grouped-gemm/fp32 diagnostic behavior |
| 2026-02-27 | Updated stage-2 risk status: PP2 bf16 NaN blocker resolved via MLA-only p2p dtype alignment + sigmoid finite normalization; added post-fix fidelity gap risk |
| 2026-02-27 | Added stage-2 fidelity round3 residual risk after timing-boundary alignment (`trace_cmd_sync_mode` default rollback + scaling optimizer prefetch boundary parity) |
| 2026-02-27 | Added stage-2 fidelity round4 residual risk after scaling optimizer pre-CMD side-effect parity (`numel` pre-scan) and refreshed distributed/scaling pair runs |

# Issues and Risks

## Open Risks

1. **DeepSeek full feature gap (expected for stage-1)**
   - MLA/shared-expert/seq_aux_loss/flex-deepep are not implemented in stage-1.
   - Mitigation: explicit documentation + stage-2 backlog.

2. **Environment dependency gap for stage-2**
   - `deepep` missing; TE/Triton generation gap with latest upstream advanced path.
   - Mitigation: lock stage-1 to compatible feature subset and mock-data validation.

3. **Scaling-mode router `scores` NaN in debug prints**
   - In Qwen3/DeepSeek scaling smoke logs, later fake ranks show `scores (first 10): [nan, ...]`.
   - Assessment result for stage-1 trace timing fidelity:
     - fixed-routing indices stay constant per rank;
     - `tokens_per_expert` pattern stays constant per rank;
     - distributed/scaling trace op structures stay aligned.
   - Conclusion: **does not block timing-trace accuracy target in stage-1** (structure and workload shape are preserved).
   - Mitigation: keep as numeric-stability follow-up item for stage-2.

4. **Unit test harness relies on distributed env vars / visible GPU count**
   - Some tests require `LOCAL_RANK` and can hang when `torch.cuda.device_count()` > launched world size.
   - Mitigation: run with `CUDA_VISIBLE_DEVICES=0`, `LOCAL_RANK=0`, `RANK=0`, `WORLD_SIZE=1`, `MASTER_ADDR`, `MASTER_PORT`.

5. **Raw comp comparison is still unstable and >5% for forward/backward**
   - In no-calibration compare reruns, `rank0/rank7` forward/backward comp gaps remain above threshold.
   - Observation:
     - `optimizer_step` gap improved after scaling optimizer-path cleanup, but forward/backward remains unstable.
     - distributed run-to-run comp decomposition fluctuates strongly (single-iteration sample sensitivity).
   - Current root-cause hypotheses:
     - pipeline state mismatch (warmup/steady/cooldown) for rank-level op slicing;
     - communication sub-op attribution vs compute boundary mismatch in per-op decomposition.
   - Mitigation in progress: continue systematic no-calibration debugging and keep automated compare script as hard gate.

6. **Per-op timing sensitivity to run context is high (single-iteration sample)**
   - `rank0/rank7` comp diffs vary significantly across reruns even under same smoke config.
   - Evidence: latest reports show forward/backward gaps moving between ~2% and >30% depending run context.
   - Mitigation:
     - keep `(op, mg_state)` aligned compare;
     - record exact command + runtime context in report;
     - avoid using a single run as final acceptance evidence.

7. **Scaling replay fidelity remains limited for backward causality**
   - Activation replay for non-first pipeline stages can be sourced from cached upstream outputs.
   - Backward replay for first pipeline stages still relies on delayed/cached downstream grads and is order-sensitive.
   - Impact: `rank0 backward_step` remains the most unstable mismatch source.
   - Mitigation:
     - use second-pass replay reruns;
     - test custom `FAKE_RANK_ORDER` to refresh target rank caches before measurement.

8. **Scaling loop `MASTER_PORT + rank` can hit occupied ports**
   - When base port collides with existing services, later fake ranks fail with TCPStore bind errors.
   - Mitigation: use high, sparse `MASTER_PORT` ranges for sequential scaling runs and document exact ports in report.

9. **`event` sync mode alone is insufficient for <=5% comp alignment**
   - After replacing sub-op global sync with event sync, repeated compare with timestamp pairing + median still shows high residual gap:
     - rank0 `forward_step/backward_step` median around `45.20%/40.13%`;
     - rank7 `forward_step/backward_step` median around `31.42%/30.25%`;
     - only `optimizer_step` stays within threshold.
   - Impact: current target (`median diff <= 5%`) not met yet.
   - Mitigation:
     - keep event mode as low-intrusion default option for alignment experiments;
     - continue root-cause isolation on workload fidelity/path parity (not only timing sync policy).

10. **DeepSeek distributed smoke under event mode is blocked by runtime environment contention**
   - ws8 run (`GPUS_PER_NODE=8`) fails with CUDA OOM on GPU0:
     - log shows external process occupancy plus current run memory pressure.
   - ws4 fallback run (`GPUS_PER_NODE=4, PP=2, EP=1`) fails with NCCL internal/socket recv error.
   - Impact:
     - cannot provide a clean DeepSeek distributed PASS artifact in current shared cluster window.
   - Mitigation:
     - reserve a clean GPU window (especially GPU0) and rerun ws8 baseline;
     - if needed, isolate to dedicated nodes or enforce stronger NCCL/network isolation.

11. **Distributed vs scaling sub-op attribution is still structurally inconsistent (TP path)**
   - In 6-GPU TP2 runs, scaling `forward_step/backward_step` contain extra `allreduce x12` sub-ops, while distributed traces do not expose matching TP allreduce sub-ops.
   - Impact:
     - `comm_ms` subtraction basis differs between modes;
     - `comp_ms` loses apples-to-apples meaning even when using identical compare logic.
   - Mitigation:
     - align comm instrumentation coverage across modes (especially TE TP path), or
     - define compare on a shared comm-subset basis until full instrumentation parity is available.

12. **Higher-load retry (SEQ_LEN=1024) does not reduce forward/backward comp gap**
   - Evidence (`rank0/rank5`, TP2/DP3/EP1/PP1, event mode):
     - forward diff remains ~48%–51%;
     - backward diff remains ~49%–66%.
   - Impact: simply increasing sequence length is insufficient to absorb residual bias.
   - Mitigation:
     - prioritize path/attribution parity fixes before further scale-up sweeps.

13. **Alternative local transformer path is currently blocked for qwen config**
   - `TRANSFORMER_IMPL=local` fails with `RMSNorm is not supported in FusedLayerNorm`.
   - Impact: cannot use local impl as immediate A/B control for TE-path attribution gap.
   - Mitigation:
     - add a dedicated local-compatible qwen smoke profile (normalization/spec choices), then re-run control experiment.

14. **Scaling vs distributed uses different `sequence_parallel` behavior under TP2 setup**
   - Qwen3 seq2048 reruns show:
     - distributed logs: `sequence_parallel=True`;
     - scaling logs: `sequence_parallel=False` (real TP is forced to 1 in scaling loop).
   - Impact:
     - TP linear communication path diverges between modes;
     - comp timing is no longer pure apples-to-apples even after comm subtraction.
   - Mitigation:
     - introduce a minimal scaling parity path for sequence-parallel semantics when `fake_tp > 1`, or
     - run distributed control with SP disabled for diagnosis-only baseline.

15. **Qwen3 seq2048 validation still fails <=5% gate for core ops**
   - `mbs=8` (6 ranks):
     - `forward_step` mean diff `16.90%` (6/6 FAIL)
     - `backward_step` mean diff `3.54%` (2/6 FAIL)
     - `optimizer_step` mean diff `9.93%` (4/6 FAIL)
   - `mbs=4` (6 ranks):
     - `forward_step` mean diff `46.69%` (6/6 FAIL)
     - `backward_step` mean diff `29.71%` (6/6 FAIL)
     - `optimizer_step` mean diff `8.23%` (4/6 FAIL)
   - Impact:
     - current no-calibration solution still cannot satisfy acceptance threshold reliably.
   - Mitigation:
     - prioritize path-parity fix (risk #14) before further scaling-size sweeps.

16. **Forced sequence-parallel parity experiment in scaling mode introduces additional instability**
   - When probing SP parity (`sequence_parallel=True` under fake TP) scaling hit RoPE length mismatch:
     - `t.shape=[256,...], freqs.shape=[512,...]` due scaling config using fake TP for `tensor_model_parallel_size`.
   - A guard was added in RoPE seq-length calculation for scaling mode, but comp alignment still did not improve.
   - Impact:
     - parity probe path is not a direct fix for <=5% target in current architecture.
   - Mitigation:
     - keep scaling default behavior unchanged for acceptance comparisons;
     - treat SP-forced path as diagnostic-only until TP comm attribution is fully aligned.

17. **Distributed/scaling profiled sample count mismatch remains a major bias source**
   - Latest Qwen3 seq2048 evidence (`rank0`):
     - distributed trace: `forward/backward/optimizer` each appears 3 times;
     - scaling trace: each appears 1 time.
   - Impact:
     - compare currently averages heterogeneous sample counts;
     - distributed outliers dominate mean comp and inflate gap.
   - Mitigation:
     - add compare mode for last-step/median-per-file or explicitly normalize sample-count policy.

18. **Distributed per-step comp variance is very large even inside one trace file**
   - Example (`rank0`, mbs=4/8 latest reruns): forward/backward comp values show high spread with extreme outliers.
   - Impact:
     - a single paired run is not reliable for acceptance;
     - current mean-based compare over a few samples is fragile.
   - Mitigation:
     - enforce repeated paired runs + robust statistic (median or trimmed mean);
     - annotate outlier runs and avoid mixing unstable windows in acceptance decisions.

19. **Timestamp-cap pairing can still mispair sequential scaling runs when ranks finish at different times**
   - Scaling rank-by-rank execution creates staggered timestamps; if cap is chosen too early, later ranks may bind to an older batch.
   - Impact:
     - false FAIL/PASS due cross-batch file selection.
   - Mitigation:
     - add filename-tag filter (e.g., `bs4/bs8`) or explicit run-id pairing in compare script.

20. **After backward I/O timing fix, residual mismatch is still systematic (not mainly from mean-vs-median choice)**
   - 8-GPU `TRACE_START=4` rerun shows realistic in-file variance is already low:
     - forward/backward comp CV ~`1%`;
     - mean-vs-median comparator difference is small.
   - Yet mid ranks still show persistent forward/backward gap (~8%-16%).
   - Impact:
     - replacing mean with median alone will not close to <=5% acceptance gate.
   - Mitigation:
     - keep “scaling single sample vs realistic avg” as per-run comparator;
     - use repeated paired runs + median-of-runs for acceptance;
     - continue targeting residual path-fidelity differences instead of statistic-only tuning.

21. **Full-profile model-size escalation is memory-limited for target batch and does not uniformly improve comp alignment**
   - Qwen3 full profile (`MODEL_PROFILE=full`) with `seq=2048`:
     - `mbs=4` distributed run OOM on GPU1;
     - fallback `mbs=1` is runnable.
   - Under runnable `mbs=1` + `TRACE_START=4`:
     - `backward_step` diff improves to ~`3%`;
     - `forward_step`/`optimizer_step` still stay high (roughly `10%`-`20%`).
   - Impact:
     - increasing model size alone cannot close the <=5% target for all ops.
   - Mitigation:
     - treat model-size scaling as secondary lever;
     - prioritize residual forward/optimizer path-fidelity investigation.

22. **Single-boundary forward-fidelity tweak yields only marginal aggregate improvement and mixed rank impact**
   - Trial change: scaling replay H2D activation copy switched to blocking before `forward_step` timing window.
   - 8-GPU regression result:
     - aggregate forward/backward/optimizer diff slightly improved on average,
     - but per-rank PASS count did not consistently improve.
   - Impact:
     - this boundary alone is insufficient to reach <=5% acceptance gate.
   - Mitigation:
     - keep this tweak as diagnostic evidence, not final fix;
     - continue isolating rank2-5 forward path-fidelity differences.

23. **`comp = total - comm` full subtraction can over-correct in overlap-heavy stages**
   - New 8-GPU rerun (`pair_timestamp=20260225185833`) confirms baseline subtraction still inflates backward mismatch:
     - forward mean diff `5.42%`, backward mean diff `10.43%`.
   - Compare-side stage-aware comm-scale correction significantly reduces this:
     - forward mean diff `2.99%`, backward mean diff `1.61%`.
   - Impact:
     - full comm subtraction is no longer a reliable universal estimator for pure comp in all stages.
   - Mitigation:
     - use optional comm-scale map (`op`/`op@stage`) in compare for overlap compensation;
     - keep default behavior unchanged for backward compatibility.

24. **Forward stage0 remains sensitive (rank0 outlier persists under fixed map)**
   - Under stage-aware map, one forward row (`rank0`) still exceeds threshold in latest run.
   - Impact:
     - strict per-rank hard gate remains brittle to stage0 local jitter.
   - Mitigation:
   - use repeated paired runs + robust op-level metric (`median_of_run_rank_median_diff_pct`) as paper-facing primary indicator;
   - keep per-rank table as supplementary diagnostic evidence.

25. **Allgather dispatcher remains high-bias under current scaling path even after comm tracing fix**
   - 8-GPU rerun (`pair_timestamp=20260226172730`) with allgather comm sub-op tracing enabled shows:
     - forward op-rank-median diff `25.15%`;
     - backward op-rank-median diff `23.45%`;
     - optimizer op-rank-median diff `5.98%`.
   - Impact:
     - allgather currently does not satisfy paper-facing alignment target;
     - using allgather as primary presentation setting would be misleading.
   - Mitigation:
     - keep alltoall as main MoE dispatcher for current scaling-vs-realistic evaluation;
     - treat allgather as controlled negative case until path-fidelity gaps are further reduced.

26. **Kernel-level B口径与trace口径 currently diverge on some ops/stages**
   - NSYS compute-only compare (rank0/rank7) indicates:
     - alltoall forward can be close on rank0, but optimizer/backward gaps remain high;
     - allgather forward/optimizer remain >20%.
   - Impact:
     - B口径暂不适合作为唯一 acceptance gate；
     - still useful for解释 overlap/stream-level timing physics.
   - Mitigation:
     - use B口径 as “ground-truth diagnostic view”;
     - use repeated trace robust metric (`op_rank_median + median_of_runs`) as main reporting indicator.

27. **Current NSYS compute-only extractor can amplify distributed-vs-scaling gap under stream overlap**
   - Evidence (`qwen_issue1_nsys_ratio_overview.log`): distributed backward windows can have `total_kernel_ms / wall_ms > 1.0` (e.g., alltoall rank0 backward `1.036`, rank7 backward `1.012`).
   - Mechanism: `analyze_nsys_cmd_kernel_breakdown.py` sums overlapped durations per kernel; under multi-stream overlap this is not timeline-union time.
   - Impact:
     - NSYS compute-only result can look worse than trace-level robust results;
     - this path is good for physics diagnosis, but not a standalone acceptance metric.
   - Mitigation:
     - keep NSYS as auxiliary diagnostic view;
     - add a non-gating timeline-union/de-dup summary for robustness.

28. **Issue2 verdict: profiler code changes are not the dominant source, but large-op (>300ms) validation is still infra-blocked**
   - Before/after evidence (`c3a77a33` vs `0094c239`): op-rank-median stays in same range (forward/optimizer high, backward low), no clear regression signature.
   - Large-workload blockers:
     - seq4096 attempts frequently OOM on shared GPUs;
     - one TP2+seq4096 path hit routing gather shape mismatch (`[4096,8]` vs `[2048,128]`).
   - Impact:
     - cannot reliably run the requested >=300ms single-op validation in current shared window.
   - Mitigation:
     - continue with stable full-profile config (`seq2048, mbs1, pp4/tp1/ep2/dp2`) for iterative diagnosis;
     - schedule dedicated non-contention GPU window before claiming final large-op conclusions.

29. **NSYS semantic-fix metric improves bias but does not fully close all gaps**
   - New metric mode (`compute-metric=primary_stream_union`, `kernel-scope=shared`, `shared-kernel-source=primary_stream`) reduces major outliers:
     - alltoall rank7 total diff: `61.88% -> 14.27%`;
     - allgather rank7 total diff: `26.00% -> 18.62%`.
   - Residual gaps remain (notably alltoall rank0 total ~`35%`, allgather forward still ~`26%`).
   - Impact:
     - pure statistics fix is necessary but insufficient;
     - residual workload-path mismatch still dominates some op/stage combinations.
   - Mitigation:
     - keep semantic-fix metric as NSYS primary view;
     - continue targeted fidelity alignment on remaining hotspots.

30. **Stage-2 DeepSeek-V3 architecture standard bring-up risks**
   - Transformer Engine version is 1.3.0 in this environment:
     - Cannot use upstream TE>=2.6 fused MLA/router paths.
     - Must implement MLA core via PyTorch SDPA and router semantics via torch ops.
   - SDPA attention mask boolean semantics mismatch risk:
     - Megatron mask uses `True=masked`, SDPA uses `True=allowed`; must invert explicitly.
   - Fixed-routing parity risk (fork-specific):
     - `config.pre_fixed_routing_results` path must share the exact same router semantics
       (sigmoid/group-limited/topk scaling/seq_aux_loss) to avoid distributed vs scaling divergence.
   - expert bias update integration risk:
     - Needs a clear update point (e.g., `finalize_model_grads`) with correct allreduce group;
       scaling mode must remain safe with world_size=1.
   - MCP `serena` retrieval is unavailable in this environment (resources list is empty):
     - Repo exploration relies on local `rg`/file inspection only.

31. **Stage-2 distributed smoke blocker (`PP=2, EP=2, bf16`) — forward loss NaN on ranks 4..7**
   - Evidence:
     - `MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=1 TRAIN_ITERS=3 bash examples/pretrain_deepseek_v3_moe.sh`
     - error: `AssertionError: Rank 7/6/5/4: found NaN in local forward loss calculation`.
     - logs:
       - `logs/deepseek_v3_stage2_dist_smoke_iter3.log`
       - `logs/deepseek_v3_stage2_dist_smoke_iter3_gate.log`
       - `logs/deepseek_v3_stage2_dist_smoke_iter3_gate0.log`
   - Additional isolation:
     - `PP=1, EP=1` runs distributed+scaling successfully (trace rank0..7 complete):
       - `logs/deepseek_v3_stage2_dist_smoke_pp1_ep1_iter3_gate.log`
       - `logs/deepseek_v3_stage2_scaling_smoke_pp1_ep1_iter3_gate.log`
     - `MOE_SHARED_EXPERT_GATE=0` does **not** remove NaN under `PP=2`.
     - `MOE_GROUPED_GEMM=0, USE_BF16=0` removes NaN but run exits with `SIGABRT`/`double free` after training.
   - Historical impact (before round2 fix):
     - Stage-2 Gate A for architecture-standard distributed smoke under `PP=2` was blocked.
     - compare/accuracy reporting for the target config could not proceed.
   - Historical mitigation path:
     - keep architecture-standard path unchanged by default;
     - use `PP=1,EP=1` only as diagnostic baseline;
     - perform focused PP+bfloat16 root-cause debugging.

   - **Status update (2026-02-27, round2): RESOLVED**
     - Root cause narrowed to MLA bf16 pipeline forward p2p path (last PP stage receives non-finite activation).
     - Fix:
       - `megatron/core/pipeline_parallel/p2p_communication.py`:
         - align forward send tensor dtype to `pipeline_dtype` in `send_forward*` paths.
         - guarded by `config.multi_latent_attention=True` to avoid changing existing non-MLA model behavior.
       - `megatron/core/transformer/moe/moe_utils.py` + `router.py`:
         - fp32-safe sigmoid score normalization with denominator clamp to remove `0/0` edge-case risk.
     - Post-fix evidence:
       - target distributed smoke PASS:
         - `MODE=distributed MODEL_PROFILE=smoke GPUS_PER_NODE=8 TRACE_START=1 TRAIN_ITERS=3`
         - log: `logs/deepseek_v3_stage2_dist_smoke_iter3_after_fix.log`
       - target scaling smoke PASS:
         - `MODE=scaling MODEL_PROFILE=smoke TRACE_START=1 TRAIN_ITERS=3`
         - log: `logs/deepseek_v3_stage2_scaling_smoke_iter3_after_fix.log`
       - trace rank coverage complete (`0..7`) in both run_config dirs.

32. **Stage-2 fidelity gap remains high after smoke-stability fix**
   - Evidence:
     - compare run (same target run_config pair with timestamp pairing):
       - `python tests/performance/compare_qwen_trace_comp.py ... --pair-timestamp 20260227111506`
       - report: `logs/deepseek_v3_stage2_compare_pp2_ep2_after_fix.log`
     - op-rank median diff remains high:
       - `forward_step ~94.85%`
       - `backward_step ~92.98%`
       - `optimizer_step ~38.75%`
   - Impact:
     - Stage-2 “架构标准双模式跑通 + trace落盘”已满足；
     - 但 paper-facing `<=5%` fidelity objective still not met for this config.
   - Mitigation in progress:
     - continue non-gating fidelity diagnosis under stable run baseline;
     - prioritize state/phase alignment and sub-op accounting parity checks.

33. **Stage-2 fidelity residual persists after round3 timing-boundary alignment**
   - Round3 changes already applied:
     - `examples/pretrain_deepseek_v3_moe.sh`: default `TRACE_CMD_SYNC_MODE` reverted to `global` (event outlier mitigation).
     - `megatron/training/training.py`: scaling-path optimizer prefetch (`get_parameters` / `get_main_grads_for_grad_norm`) moved outside traced `optimizer_step` CMD to mirror distributed `train_step` boundary.
   - Fresh paired evidence (`trace4/iters6`, rank `0..7`):
     - `pair=20260227141611`, `distributed_subtract_comm=True`:
       - `forward_step` median `3.83%` (PASS), `backward_step` median `11.09%` (FAIL), `optimizer_step` median `7.84%` (FAIL).
     - `pair=20260227141950`, `distributed_subtract_comm=True`:
       - `forward_step` median `7.58%` (FAIL), `backward_step` median `9.86%` (FAIL), `optimizer_step` median `10.54%` (FAIL).
   - Root-cause status:
     - backward comp still depends strongly on distributed comm subtraction policy (`alpha` spread wide; stage/run sensitive), indicating top-level/sub-op decomposition is not yet stable for backward pure-compute isolation.
     - optimizer_step remains systematically higher in scaling (typically `+8%` to `+12%` median on stable pairs), consistent with residual sequential single-GPU replay/state effects beyond a single CMD-boundary mismatch.
   - Impact:
     - stage-2 runability + trace artifact gates remain satisfied;
     - paper-facing `<=5%` fidelity is still not reached for backward/optimizer in target config.
   - Mitigation options (pending explicit decision):
     - evaluation-side: introduce optional stage-aware backward subtraction mode in compare script (default unchanged).
     - runtime-side: deeper scaling executor alignment experiments that may alter scaling execution semantics (must obtain user confirmation before applying).

34. **Stage-2 fidelity residual persists after round4 optimizer pre-work parity**
   - Round4 change:
     - scaling `_prepare_scaling_optimizer_step(...)` now mirrors distributed pre-CMD side effects (`numel` pre-scan on params/grads).
   - Improved evidence:
     - rank0 optimizer paired diff reduced from `12.76%` to `6.02%` (same distributed baseline family).
   - Latest full-pair evidence (`distributed ts=20260227145522`, interleaved two-pass scaling):
     - `forward_step` rank median `4.02%` (PASS)
     - `backward_step` rank median `5.11%` (FAIL, near threshold)
     - `optimizer_step` rank median `7.68%` (FAIL)
   - Additional execution risk observed:
     - scaling sequential rank loop can hit `Address already in use` when base `MASTER_PORT` overlaps existing jobs.
   - Impact:
     - stage-2 runability/trace gates remain green, but paper-facing `<=5%` target is still not fully met.
   - Mitigation:
     - continue measurement-stability protocol (explicit high `MASTER_PORT` ranges, two-pass cache reuse, fixed rank-order pairing);
     - if remaining optimizer/backward residual cannot be removed without execution-semantic changes, obtain user confirmation before applying such changes.


35. **Stage-2 residual fidelity risk after iter-indexed replay alignment fix**
   - Fix applied (round5):
     - replay cache upgraded from single-file-per-dst to iteration-indexed (`*_iter{current_iter}.pt`) for activation/grad handoff in scaling pipeline replay.
     - relevant files:
       - `megatron/training/training.py`
       - `megatron/profiler/utils.py`
   - Verified effect scope:
     - temporal replay alignment defect is removed (consumer ranks can load per-iteration replay tensors).
     - new unit tests added and passed (`tests/unit_tests/profiler/test_scaling_replay_cache_paths.py`).
   - Remaining evidence (`pair=20260227145502`, target config `PP2/TP1/EP2/DP4`):
     - subtract-comm op-rank-median:
       - `forward_step=4.23%` (PASS)
       - `backward_step=14.18%` (FAIL)
       - `optimizer_step=7.57%` (FAIL)
     - no-subtract op-rank-median:
       - `forward_step=14.80%` (FAIL)
       - `backward_step=17.53%` (FAIL)
       - `optimizer_step=7.57%` (FAIL)
   - Risk assessment:
     - one confirmed fidelity bug is fixed, but dominant residual error remains;
     - distributed baseline run-to-run drift is now large enough to materially affect pass/fail conclusions.
   - Proposed next mitigation:
     - stabilize comparison protocol with repeated paired runs + robust aggregation as gating input;
     - isolate optimizer residual via focused per-rank repeated profiling (fixed GPU, fixed rank order, controlled port window) before considering runtime-semantic changes.

## Resolved During Stage-1

- **Scaling backward CMD timing-region pollution by grad-cache I/O** resolved:
  - moved `torch.save(grad_to_rank*)` outside scaling `backward_step` CMD scope;
  - 8-GPU evidence: backward diff for ranks 2-7 dropped from ~125.56% (pre-fix) to ~11.18% (post-fix) under the same compare protocol.

- **MoE pattern injection gap** resolved by introducing block-level mixed dense/MoE specs.
- **Expert FFN sizing gap** resolved by separating expert vs dense FFN hidden-size usage.
- **Routing control duplication** cleaned by removing redundant router call in MoE forward path.
- **Non-scaling MoE forward regression** fixed by guarding `pre_fixed_routing_results` attribute access with `getattr` fallback.
- **Router aux-loss unit path broken (`moe_gather/moe_scatter`)** fixed by replacing undefined path with gather/scatter-add implementations in token dispatcher.
- **Router aux-loss unit path arg dependency (`args is not initialized`)** fixed with fallback branch in tensor-parallel all-reduce helper for standalone unit-test runs.
- **Scaling mode default GPU contention bias** mitigated by auto idle-GPU selection in Qwen3/DeepSeek scaling scripts.
- **MoE trace-time debug perturbation** mitigated by removing hot-path debug prints from `moe_layer.py` and `token_dispatcher.py`.
- **Fixed-routing NaN cascade in scaling/debug path** mitigated by finite-guard (`nan_to_num`) before score softmax in `moe_layer.py`.
- **Stage-1.5 calibration path** intentionally removed from active validation flow after user review:
  - no longer used as acceptance evidence for comp-accuracy.
- **Sub-op timing global pipeline drain risk** mitigated with policy-based sync:
  - added `trace_subop_sync_mode` (`global/event`);
  - trace decorator + async path now use the same policy;
  - default remains `global` for backward compatibility.
- **Compare latest-file mismatch risk** mitigated:
  - compare script now supports timestamp-cap pairing and repeated-run median summary.
- **Compare overlap-bias diagnosis/mitigation path** added:
  - compare now supports optional `comm_scale` and `comm_scale_map` (including `op@stageX`);
  - reports `effective_comm_ms` and `comm_scale_suggestion` (op + stage-aware) to avoid hidden calibration.
- **Paper-facing robust metric view** added:
  - compare now emits `op_rank_median_aux_summary` and repeat op-median summary;
  - latest two-run evidence shows `forward/backward/optimizer` op-median all within `<=5%`.
- **Scaling TP>1 hard-block in router/dispatcher path** resolved for current tracing workflow:
  - replaced scaling-unsafe TP assertions with fake-TP-aware handling.
- **Scaling EP=1 preprocessing crash** resolved:
  - when precomputed dispatch cache is absent, dispatcher now uses runtime token histogram path.
- **Distributed PP1 backward trace coverage gap** resolved:
  - no-pipeline path now wraps backward with CMD and emits `backward_step` trace records.
- **6-GPU compare rank hardcoding** resolved:
  - compare script now supports `--ranks` and `--ops` for non-8GPU runs.
- **Allgather dispatcher API mismatch (`use_global_buffer` kwarg) in tensor-parallel mapping** resolved:
  - `gather_from_sequence_parallel_region_to_moe` now accepts `use_global_buffer`;
  - added regression unit test: `tests/unit_tests/tensor_parallel/test_mappings_moe_api.py`.
- **Kernel-ground-truth extraction capability gap** resolved:
  - added CMD NVTX switch (`--trace-kernel-ground-truth`) and NSYS post-analysis scripts:
    - `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`
    - `tests/performance/compare_qwen_nsys_compute_only.py`.
