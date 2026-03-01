## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-01 | Added Issues 69-71 for governance implementation: official backward semantics freeze (`seq8192 + phase-pure + repeat-x5`), advanced-diagnostics opt-in policy in example scripts, and scope lock to attention SDPA iter-bucket diagnosis |
| 2026-03-01 | Added Issue 68 for attention-core deep-segment repeat-x5 variability bucketing: residual is iter1-dominant with stable pre-fmha adjacency shift, so backward freeze needs explicit iter-bucket robustness constraints |
| 2026-03-01 | Added Issue 67 for seq8192 SDPA-subsegment formal repeat-x5: `fmha_cutlassB` remains top1 in 5/5 runs with zero contamination, but run-level variability (especially scaling_off) remains high and blocks backward freeze |
| 2026-03-01 | Added Issue 66 for `attn_core_sdpa_bwd` immediate-neighbor threshold sweep: pre-fmha `FillFunctor` path is present on both branches; count asymmetry at `60us` is threshold-sensitive and does not change residual gap |
| 2026-03-01 | Added Issue 65 for attention-core micro-segment x1: residual is further localized to `attn_core_sdpa_bwd` (~98% fmha share), while `attn_core_precast_bwd`/`attn_core_postcast_bwd` are near-zero; pre-fmha top adjacent kernel is stabilized as `FillFunctor<unsigned char>` |
| 2026-03-01 | Added Issues 63-64 for attention-segment debug + repeat-x5: residual is stably concentrated in `attn_core_bwd/fmha_cutlassB` with no stream-set mismatch, but backward gate remains >5% even after segmentation-based localization |
| 2026-03-01 | Added Issue 62 for patched clean-x1 attention-family deep diagnosis: `fmha_cutlassB` remains top residual in stage1 backward (`rank4..7`) for both scaling DDP on/off, with exact launch-config parity, so root-cause focus shifts to attention runtime-context rather than launch-shape mismatch |
| 2026-03-01 | Added Issue 61 for post-fix clean x1 result: NVTX structural gate now passes on dist/scaling on/off (`open/overlap=0`), but backward residual remains above threshold (DDP-on 17.42%, DDP-off 10.52%), so root cause focus must shift to attention-family path |
| 2026-03-01 | Added Issue 60 for NVTX attribution corruption: repeat-x5 traces show systematic forward/backward CMD overlap and unclosed `row_g_fwd` labels caused by missing NVTX pop on TP world_size==1 fast path; minimal fix landed with RED→GREEN unit evidence |
| 2026-03-01 | Added Issue 59 for seq8192 phase-pure formal repeat-x5 freeze round: DDP-off only partially improves backward and does not improve stability; `fmha_cutlassB` remains dominant (5/5 top1), so attention-family diagnostics become next priority |
| 2026-03-01 | Added Issue 58 for seq8192 x1 probe drift: residual direction/sign differs from historical round68 high-gap set (scale>dist in this probe), indicating strong protocol/run-context sensitivity and the need for repeat-x5 freeze evidence |
| 2026-03-01 | Added Issue 57 for seq8192 phase-pure DDP probe A/B: contamination remains zero and DDP-off improves metrics, but backward still exceeds 5% and attention-family residual remains dominant |
| 2026-03-01 | Added Issue 56 for scaling DDP-hook hypothesis validation: scaling DDP hook overhead exists and is measurable, but disabling it does not improve backward residual and does not support “scaling hook missing” as primary cause |
| 2026-03-01 | Added Issue 55 for cross-run validation: non-comm (`fmha_cutlassB`) dominance in backward residual is stable across `round68 seq8192 run1..5`, and `_AllToAll` micro-repro does not support autograd-node-count inflation as primary root cause |
| 2026-03-01 | Added Issue 54 for backward residual deep-dive + scaling comm-adjacent emulation: emulation knob is feasible but x1 backward improvement is marginal, and dominant residual contribution appears non-comm (`fmha_cutlassB` family) |
| 2026-03-01 | Added Issue 53 for postfix all_to_all comm-adjacent attribution fix validation: contamination remains zero and behavior is correct, but backward residual does not materially improve in x1 sanity compare |
| 2026-03-01 | Added Issue 52 for phase-label NSYS sanity x1 rerun: contamination is zero in both distributed/scaling pure windows, but fidelity residual remains high, so freeze must proceed with seq8192 repeat-x5 protocol |
| 2026-03-01 | Added Issue 51 for newly landed phase-level pure-compute semantics path: implementation complete, but official freeze still depends on fresh NSYS captures with phase labels and repeat-x5 acceptance checks |
| 2026-02-28 | Added Issues 49-50 for Round6-8 NSYS repeat-x5 tri-view findings: backward remains high under current NSYS compute-only semantics, and official backward gate semantics is still unfrozen without introducing calibration-dependent stage-aware fitting |
| 2026-02-28 | Added Issue 48 for seq8192 backward semantics matrix: baseline full subtraction causes stage1 over-subtraction inflation, while stage-aware/op-map auxiliary views restore low backward spread and isolate optimizer as remaining residual |
| 2026-02-28 | Added Issue 47 for measurement-regime change result at seq8192: forward noise improved but backward subtraction bias exploded, so <=5% gate is still blocked by semantics rather than only smoke-scale noise |
| 2026-02-28 | Added Issue 46 for Round6-8 workload-scaling OOM boundary: full profile (61L/7168H) remains infeasible even at short sequence, limiting model-size escalation path in current environment |
| 2026-02-28 | Added Issue 45 for intermittent distributed abort (`double free or corruption`) observed during long repeat runs; rerun succeeded but indicates environment/runtime instability risk |
| 2026-02-28 | Added Issue 44 for O1 pre-CMD optimizer-drain A/B repeat5 outcome: backward/optimizer and composite regressions, spread non-convergence, and rejection under noise-floor criterion |
| 2026-02-28 | Added Issue 43 for round6-8 noise-floor repeat5 and sequential-scaling pairing-cap pitfall; clarified O1 acceptance must account for measured noise floor |
| 2026-02-28 | Added Issue 42 for round6-8-baseline O2/B1 follow-up: O2 not supported, strict-grad-replay works as integrity guard but does not improve gate metrics, and O1 becomes next priority |
| 2026-02-28 | Added Issue 41 for current-latest Round12-protocol rerun: metrics regress vs Round4/Round6-8, best round remains Round6-8, and Round9+ retrospective indicates diagnostic-path/value-path decoupling is required |
| 2026-02-28 | Added Issue 40 for round12 repeated post-optimizer replay-write evidence: backward/optimizer residual remains above 5% despite fixed protocol and median-of-runs aggregation |
| 2026-02-27 | Added Issue 39 for round11 semantic-touching experiments: replay-write timing helps stability but optimizer main residual remains >5%, scheduler increment switch not beneficial |
| 2026-02-27 | Added Issue 38 for optimizer microphase phase-aware evidence: main-update residual remains dominant after protocolfix8 run1/2/3 repeats |
| 2026-02-27 | Added Issue 37 for newly landed optimizer microphase trace path: phase-level fidelity evidence still pending (runtime verification next step) |
| 2026-02-27 | Added Issue 36 for protocolfix8 residual fidelity drift after fixed-port/fixed-order repeated pairing; updated Serena availability note to intermittent |
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
  - MCP `serena` availability is intermittent in this environment:
    - some sessions return handshake timeout / empty resources;
    - later retries can succeed for project activation and symbolic tools.

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

36. **Protocolfix8 residual drift after fixed-port/fixed-order repeated pairing (non-semantic protocol already saturated)**
   - Newly executed protocol family (`round6/7/8`) enforced:
     - fixed high port ranges,
     - fixed fake rank order,
     - explicit pairset construction against fixed distributed baseline (`20260227145522`),
     - repeated run evidence archived.
   - Best current single-run in this family:
     - `logs/deepseek_v3_stage2_compare_trace4_iter6_protocolfix7_run1_vs_dist145522_subtract.log`
     - `forward=3.06%` (PASS), `backward=7.51%` (FAIL), `optimizer=5.97%` (FAIL).
   - Closest backward run:
     - `logs/deepseek_v3_stage2_compare_trace4_iter6_protocolfix8_interleave_w1_run1_vs_dist145522_subtract.log`
     - `backward=5.66%` (near-threshold FAIL), but `forward=10.99%` regressed.
   - Repeat aggregation evidence (3-run protocolfix6 series):
     - median-of-runs: `forward=8.48%`, `backward=15.02%`, `optimizer=6.69%` (all FAIL).
   - Risk assessment:
     - non-semantic protocol alignment knobs (ports/order/pairing/repeat) improve interpretability but cannot stably push `backward_step` below 5%;
     - residual error is concentrated on steady-state stage1 ranks (notably rank4/rank6), with run-to-run drift still material.
   - Mitigation:
     - keep current protocol as baseline evidence path;
     - before any runtime-semantics change, provide design proposal and obtain explicit user approval.

37. **Optimizer microphase path已落地，但 phase-level fidelity 证据尚未补齐**
   - 当前状态：
     - 代码已支持 `--trace-optimizer-microphases`（default-off），distributed/scaling 均输出同名三段：
       - `optimizer_main_update`
       - `optimizer_state_update`
       - `optimizer_post_update`
     - 单元测试已通过（结构/顺序/参数解析）。
   - 未完成项：
     - 尚未完成基于同一 pairset 的 phase-level 对比报告（尤其 `optimizer_state_update` vs `optimizer_main_update` 的误差贡献分解）。
   - 影响：
     - 目前仍无法用 phase-level 证据证明 optimizer residual 的主因归属。
   - 下一步缓解：
     - 在固定 protocol（端口段 + rank-order + repeated pairing）下启用 microphase flag 进行 distributed/scaling rerun；
     - 使用 compare 脚本 `--ops` 扩展到三段 microphase，沉淀 phase-aware 证据后再决定是否申请语义触及优化。

38. **Microphase phase-aware 证据显示 optimizer 主残差仍由 `optimizer_main_update` 主导**
   - 新证据（protocolfix8, run1/run2/run3, fixed distributed baseline）：
     - `optimizer_main_update` op-rank-median:
       - run1 `10.86%`
       - run2 `9.75%`
       - run3 `12.52%`
       - median-of-runs `10.86%`（FAIL）
     - `optimizer_step` op-rank-median:
       - run1 `11.19%`
       - run2 `10.43%`
       - run3 `13.22%`
       - median-of-runs `11.19%`（FAIL）
   - 解释：
     - microphase 已验证记录链路正确，但 residual 并未主要来自 scheduler/post hooks；
     - 主更新阶段（`optimizer_main_update`）本身跨模式差异仍高，说明后续若继续压误差，可能需要触及 scaling 执行语义或测量边界设计。
   - 补充观察：
     - `optimizer_state_update`/`optimizer_post_update` 绝对时长极短（约 `0.01~0.05ms`），相对误差百分比易被放大，不宜单独作为主门限结论。
   - 缓解建议：
   - 保持当前 microphase 路径 default-off；
   - 下一步先提交“可能触及语义”的方案并等待用户确认，再做代码修改。

39. **Round11 semantic-touching试验后，optimizer主残差仍未过线（`~6%`）**
   - 新增试验改动（均 default-off）：
     - `--scaling-replay-write-phase {pre_optimizer,post_optimizer}`：
       - post 模式把 grad replay 写回延后到 optimizer 之后。
     - `--scaling-align-scheduler-increment`：
       - scaling scheduler increment 改用 real `data_parallel_size` 口径（实验开关）。
   - 对照证据（同 distributed baseline `20260227182456`）：
     - post-write:
       - `forward=10.74%`, `backward=12.71%`, `optimizer=6.56%`, `main=6.21%`
       - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepost_run1.log`
     - pre-write:
       - `forward=14.09%`, `backward=19.52%`, `optimizer=6.47%`, `main=6.04%`
       - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepre_run1.log`
     - post-write + align-increment:
       - `forward=8.10%`, `backward=10.92%`, `optimizer=7.16%`, `main=6.50%`
       - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_replayphasepost_aligninc_run1.log`
   - 结论：
     - replay-write 时序调整对 forward/backward 漂移有正向效果，但 optimizer main/update 仍在约 `6%` 区间，尚未满足 `<=5%`。
     - scheduler increment 对齐开关在本轮未改善 optimizer gate，且有回退风险。
   - 后续缓解建议：
     - 继续保持这两项为 default-off 实验开关；
     - 下一轮优先针对 `stage1 ranks` 做更细粒度 optimizer 主更新路径归因（参数桶/主梯度集合一致性）并配合 repeated median 报告。

40. **Round12 重复验证后，`post_optimizer` 写回方案仍无法将 backward/optimizer 压到 `<=5%`**
   - 固定协议（已执行 3 runs）：
     - fixed rank-order: `0,4,1,5,2,6,3,7`
     - fixed high-port 段：`990x/995x`
     - `TRACE_START=4`, `TRAIN_ITERS=6`, `TRACE_OPTIMIZER_MICROPHASES=1`
     - `SCALING_REPLAY_WRITE_PHASE=post_optimizer`, `SCALING_ALIGN_SCHEDULER_INCREMENT=0`
   - 证据（single-run, op-rank-median）：
     - run1（pair `20260228051919`）：
       - `forward=5.69%`, `backward=8.24%`, `optimizer=12.44%`, `main=12.03%`
       - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run1.log`
     - run2（pair `20260228052156`）：
       - `forward=8.23%`, `backward=12.65%`, `optimizer=7.76%`, `main=7.70%`
       - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run2.log`
     - run3（pair `20260228052432`）：
       - `forward=8.13%`, `backward=13.03%`, `optimizer=6.11%`, `main=6.14%`
       - report: `logs/deepseek_v3_stage2_compare_trace4_iter6_microphase_round12_postwrite_run3.log`
   - 聚合证据（median-of-runs）：
     - `forward=8.13%`, `backward=12.65%`, `optimizer=7.76%`, `optimizer_main_update=7.70%`
     - repeat artifact: `logs/deepseek_v3_stage2_repeat_microphase_round12_postwrite_subtract.jsonl`
   - 风险评估：
     - 非语义协议对齐 + repeat aggregation 已基本饱和，仍无法稳定达到 `<=5%`；
     - backward/optimizer 的 run-to-run 漂移继续影响 gate 判定稳健性；
     - optimizer 主残差虽较 round10 降低，但仍显著高于目标阈值。
   - 缓解建议：
     - 继续保持当前语义触及开关 default-off（不影响默认路径）；
     - 下一步先做更细粒度 `optimizer_main_update` 诊断（参数桶/主梯度集合/phase 切分），再决定是否申请新的执行语义调整。

41. **Current latest rerun（Round12 协议）相对 Round4 / Round6-8 明显回退，best round 仍为 Round6-8**
   - 当前复测（3 runs，固定协议）：
     - shared protocol:
       - `TRACE_START=4`, `TRAIN_ITERS=6`
       - `TRACE_SUBOP_SYNC_MODE=global`, `TRACE_CMD_SYNC_MODE=global`
       - `TRACE_OPTIMIZER_MICROPHASES=1`
     - scaling protocol:
       - `SCALING_REPLAY_WRITE_PHASE=post_optimizer`
       - `SCALING_ALIGN_SCHEDULER_INCREMENT=0`
       - `SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7`
   - current single-run（op-rank-median）：
     - run1 (`pair=20260228072907`): `forward=5.21%`, `backward=12.03%`, `optimizer=11.61%`
     - run2 (`pair=20260228073218`): `forward=9.35%`, `backward=17.87%`, `optimizer=9.43%`
     - run3 (`pair=20260228073523`): `forward=6.14%`, `backward=15.11%`, `optimizer=7.84%`
   - current median-of-runs：
     - `forward=6.14%`, `backward=15.11%`, `optimizer=9.43%`
     - `mean_3ops=10.23%`, `max_3ops=15.11%`
   - 历史对比：
     - Round4: `4.02% / 5.11% / 7.68%`（`mean_3ops=5.60%`, `max_3ops=7.68%`）
     - Round6-8 best-single: `3.06% / 7.51% / 5.97%`（`mean_3ops=5.51%`, `max_3ops=7.51%`）
   - 判定（主判据 mean，辅判据 max）：
     - **best round 仍为 Round6-8**；current 明显劣于 Round4/Round6-8。
   - retrospective 风险结论：
     - Round9+ 中 microphase/replay-write 路径对诊断价值明确，但未转化为主 gate 的稳定收益；
     - single-run 与 repeated median 差异继续说明噪声显著，必须坚持 repeated pairing；
     - 后续需将“诊断路径”和“主 gate 路径”解耦（例如 default gate 不强制 microphase）。

42. **Round6-8 基线 follow-up：O2 被证伪，B1 仅具完整性价值但不提升主指标**
   - O2 A/B（3-run repeated pairing）证据：
     - micro0 (`TRACE_OPTIMIZER_MICROPHASES=0`) median-of-runs:
       - `forward=8.37%`, `backward=10.47%`, `optimizer=8.61%`
       - `mean_3ops=9.15%`, `max_3ops=10.47%`
     - micro1 (`TRACE_OPTIMIZER_MICROPHASES=1`) median-of-runs:
       - `forward=5.71%`, `backward=9.34%`, `optimizer=8.36%`
       - `mean_3ops=7.80%`, `max_3ops=9.34%`
   - O2 结论：
     - “microphase instrumentation 恶化 fidelity”在该基线/协议下不成立，反向证据更强（micro1 全面优于 micro0）。
   - B1（strict grad replay）证据：
     - strict single-pass probe 能在缺 cache 时 fail-fast（rank0 backward profile 首次即报错），说明完整性守卫生效；
     - strict two-pass（warm+strict）3-run median-of-runs：
       - `forward=5.84%`, `backward=9.51%`, `optimizer=9.29%`
       - `mean_3ops=8.21%`, `max_3ops=9.51%`
   - 对比 O2-best（micro1）：
     - `forward +0.13%`, `backward +0.17%`, `optimizer +0.93%`
     - 说明 B1 不改善 gate fidelity（尤其 optimizer 有明显回退）。
   - 风险与下一步：
     - B1 应保留为 default-off 的完整性/诊断开关，而非主 gate 运行口径；
     - 下一优先级转向 O1：`optimizer_main_update` timing boundary / queue contamination 的最小变量 A/B。

43. **Round6-8 纯噪声基线（repeat x5）显示测量地板较高，且存在顺序 scaling 的 timestamp-cap 配对陷阱**
   - 纯噪声量化（不改代码，microphase=1，fixed protocol）：
     - single-run:
       - run1: `5.64% / 14.66% / 9.09%`
       - run2: `10.36% / 10.54% / 8.28%`
       - run3: `7.16% / 11.67% / 11.13%`
       - run4: `8.37% / 15.85% / 4.89%`
       - run5: `4.57% / 10.55% / 6.12%`
     - median-of-runs:
       - `forward=7.16%`, `backward=11.67%`, `optimizer=8.28%`
     - run range:
       - `forward=5.79%`, `backward=5.31%`, `optimizer=6.24%`
   - 配对陷阱证据：
     - 顺序 scaling 下若用 rank0 timestamp 作为 cap，可能导致后续 rank 尚未落盘而触发错配/异常（run1 复现过 `IsADirectoryError`）。
     - 需改为 end-of-run cap（实操采用 rank7 timestamp）以保证同批次覆盖。
   - 影响：
     - 在该测量体系下，`<~1-2%` 级别 A/B 改善可信度不足；
     - O1 后续验收不能只看单次结果，必须结合 repeat median 与 spread 一致下降。
   - 缓解建议：
     - 固化 pairing policy：使用每轮 end-of-run timestamp cap；
     - O1 开发保持单变量，且至少 repeat x5 才做结论判断。

44. **O1（pre-CMD optimizer drain）在 Round6-8 基线 A/B repeat5 下未通过噪声地板判定**
   - 实施机制（单变量）：
     - `--trace-optimizer-pre-cmd-drain` 开启后，仅在进入 top-level `optimizer_step` CMD 前调用一次 `torch.cuda.synchronize()`；
     - distributed/scaling 对称启用；不改 barrier 与 CMD 边界。
   - A/B median-of-runs（op-rank-median）：
     - drain0: `forward=12.84%`, `backward=10.06%`, `optimizer=8.23%`, `mean_3ops=8.80%`, `max_3ops=12.84%`
     - drain1: `forward=11.55%`, `backward=14.63%`, `optimizer=10.00%`, `mean_3ops=12.10%`, `max_3ops=14.63%`
   - 关键回退（drain1 - drain0）：
     - `backward +4.57%`
     - `optimizer +1.77%`
     - `mean_3ops +3.30%`
     - `max_3ops +1.79%`
   - spread 观察（range/IQR）：
     - 多数核心项未下降（尤其 backward 与 max 指标），未满足“精度提升 + 波动收敛”的联合条件。
   - 结论：
     - O1 在当前协议下应判定为 **不通过**；
     - 保留该开关 default-off，仅作诊断用途，不纳入主 gate。

45. **长时间 repeated distributed 运行中出现一次间歇性 abort（`double free or corruption`）**
   - 现象：
     - O1 `drain1` 第一次批量执行在 run3 distributed 结束后出现：
       - `double free or corruption (!prev)`，`SIGABRT`，rank5 退出（`exitcode -6`）。
   - 影响：
     - 该次批量执行中断，需要 rerun run3~run5。
   - 后续处理：
     - 采用新的端口段重新执行 run3~run5 后全部完成，compare/repeat 结果可复现。
   - 风险判断：
   - 当前更像环境/运行时偶发不稳定，而非 O1 逻辑确定性错误；
   - 但对长批次重复实验会增加失败重试成本，应持续监控。

46. **Round6-8 workload scaling 的模型尺寸上限受内存硬约束（full profile 不可行）**
   - 现象（script-parameter-only OOM sweep）：
     - `MODEL_PROFILE=full`（61L/7168H）在 distributed 下即便 `SEQ_LEN=96` 也 OOM；
     - `SEQ_LEN=128/192/256` 同样 OOM。
   - 影响：
     - 当前环境无法通过“同时放大 `NUM_LAYERS/HIDDEN_SIZE`”来推进测量体制；
     - 后续 workload 扩大只能沿 smoke 线加大 `SEQ_LEN`（或更换资源/并行配置）。
   - 缓解建议：
     - 若必须验证 full 维度路径，需资源侧变更（更大单卡显存或不同并行切分）；
     - 在现有资源下，优先利用 smoke `SEQ_LEN` 扩展做测量分析，但需防范 subtraction 语义偏置。

47. **测量体制放大到 `SEQ_LEN=8192` 后，backward 指标由系统偏置主导而非纯噪声**
   - 证据（Round6-8 baseline, rank7-cap repeat x5）：
     - median-of-runs:
       - `forward=2.00%`（显著改善）
       - `backward=62.73%`（显著恶化）
       - `optimizer=6.72%`（仍 >5%）
     - range:
       - `forward=3.23%`, `backward=34.41%`, `optimizer=4.35%`
   - 对比旧 smoke 基线（seq256）：
     - `forward` 从 `7.16%` 降到 `2.00%`，
     - `backward` 从 `11.67%` 升到 `62.73%`。
   - 结论：
     - “smoke 太小导致全部指标噪声吞没”并非完整解释；
     - 在长序列下，`distributed_subtract_comm` 对 backward 的系统偏置成为主导因素。
   - 风险：
     - 若继续在该语义下推进代码级 A/B，容易把测量偏置误判为代码改进/回退。
   - 缓解建议：
     - 暂停新代码假设实验，先验证 backward measurement semantics；
     - 同时保留 rank7-cap + repeat-x5 作为固定统计纪律。

48. **同批次语义矩阵验证确认：backward 失真主因是 stage1 overlap 下的 comm 过度扣减**
   - 方法（同一批次/同一 pair 集）：
     - baseline subtract（alpha=1.0）；
     - op-map subtract（`forward=0.787, backward=0.176`）；
     - stage-aware subtract（`forward@stage1=0.787, backward@stage1=0.107`）；
     - no-subtract total control。
   - repeat x5 median 结果：
     - baseline: `forward=2.00%`, `backward=62.73%`, `optimizer=6.72%`
     - op-map: `forward=1.74%`, `backward=5.85%`, `optimizer=6.72%`
     - stage-aware: `forward=1.74%`, `backward=3.86%`, `optimizer=6.72%`
     - no-subtract: `forward=9.32%`, `backward=7.74%`, `optimizer=6.72%`
   - backward spread 对比：
     - baseline range/IQR: `34.41% / 6.84%`
     - op-map range/IQR: `0.75% / 0.47%`
     - stage-aware range/IQR: `1.81% / 0.64%`
   - 行级证据（run1 stage1 rank4~7）：
     - baseline 将 `dist_comm~42~44ms` 全扣，`dist_comp` 压到 `~22ms`，对比 `scale_comp~54~63ms`，导致 `146%~177%` diff；
     - stage-aware 仅有效扣减 `~4.5~4.7ms`，`dist_comp~61ms`，与 scaling 接近，diff 大幅回落。
   - 结论：
     - backward 的主要问题是测量语义（over-subtraction），而不是训练逻辑回退；
     - 当前真正剩余 gate residual 是 optimizer（`median 6.72%`）。
   - 缓解建议：
   - 在 backward 语义未冻结前，不启动新的代码级 A/B；
   - 语义层固定后再恢复单变量实验，并把优化重点放在 optimizer phase。

49. **Round6-8 seq8192 NSYS repeat-x5 显示：当前 compute-only 口径下 backward 仍稳定高残差（~38%）**
   - 条件：
     - baseline round68、rank7-cap repeat x5；
     - NSYS compare 口径：`compute-metric=primary_stream_union` + `kernel-scope=shared` + `shared-kernel-source=primary_stream`。
   - 结果（median-of-runs）：
     - `forward=0.33%`
     - `backward=38.30%`
     - `optimizer=0.81%`
   - spread：
     - backward `range=3.26%`, `IQR=0.43%`（低方差但高偏差）。
   - 风险结论：
     - 当前 NSYS compute-only 方案在该 workload 下不适合作为唯一 backward gate（稳定但系统偏高）。

50. **backward 三视图长期分歧（subtract vs no-subtract vs NSYS）表明官方语义仍未冻结**
   - 同一批次 repeat x5 对照：
     - trace subtract backward median: `35.21%`
     - trace no-subtract backward median: `10.40%`
     - NSYS compute-only backward median: `38.30%`
   - 解读：
     - subtract 与 no-subtract 差距依然大，说明 subtraction 口径仍有系统偏置风险；
     - stage-aware 虽能压低 backward，但依赖标定参数，不可作为正式可扩展方案。
   - 缓解建议：
     - 继续执行“无标定”语义收敛实验（测量边界纯化 + 同步语义对照）；
     - 在 backward 官方语义冻结之前，暂停基于该指标的代码级收益宣称。

51. **phase-level pure-compute 语义路径已落地，但官方 freeze 仍依赖“新采集”验证而非旧 traces 回放**
   - 已完成（实现侧）：
     - 新增 trace 参数：`--trace-kernel-ground-truth-phase`、`--trace-kernel-boundary-sync-mode`；
     - backward compute 区间已在 distributed/scaling 路径标注 `phase=compute`；
     - comm 区间已在 comm decorator/top-level comm CMD 标注 `phase=comm`；
     - NSYS analyzer 已输出 `compute_pure_*` 与 `contamination_*`；
     - compare 已支持 `pure_primary_union` 与 contamination gate。
   - 当前风险：
     - 现有 round68 历史 NSYS trace 不含新 phase labels（`phase_window_parents=0`），仅能做兼容回放验证，不能直接证明“纯化后 backward residual”。
   - 影响：
     - backward 官方 gate 语义不能仅凭历史 sqlite 重跑冻结，必须基于新一轮带 phase labels 的 repeat-x5 结果。
   - 缓解建议：
   - 在固定协议（rank7-cap + repeat-x5 + seq8192）下重采 distributed/scaling NSYS；
   - 使用 `compute_metric=pure_primary_union` + contamination gate 复核 acceptance 条件后再冻结口径。

52. **phase-label NSYS x1 新采集已验证“无污染”，但 fidelity 仍未达标（非 comm-contamination 主因）**
   - 新采集条件（smoke `SEQ_LEN=1024`，distributed/scaling 对齐）：
     - `TRACE_KERNEL_GROUND_TRUTH=1`
     - `TRACE_KERNEL_GROUND_TRUTH_PHASE=1`
     - `TRACE_KERNEL_BOUNDARY_SYNC_MODE=event`
   - 证据：
     - distributed/scaling 分析均为 `phase_window_parents=48`, `event_rows=72`;
     - contamination 指标在 event/aggregate 均为 `0.00%`;
     - compare 在 `--compute-metric pure_primary_union --require-low-contamination-pct 1` 下 contamination gate 全量 PASS。
   - 结果：
     - op-rank-median 仍 FAIL：`forward=10.25%`, `backward=17.97%`, `optimizer=5.75%`。
   - 结论：
     - 当前 residual 不能归因于“comm 混入 compute-only”；
     - 单次 x1 不具冻结代表性，必须继续执行固定协议 `seq8192 + rank7-cap + repeat-x5`。
   - 缓解建议：
   - 以新 phase 语义跑完整 repeat-x5；
   - 仅当 acceptance 四条件同时满足时再冻结官方 backward comp-only 口径。

53. **all_to_all comm-adjacent attribution postfix 已验证正确性，但并未显著降低 backward residual（x1）**
   - postfix 内容（已落地）：
     - `_profiled_all_to_all_single` 内执行 `input_.contiguous()`（归属 comm phase）；
     - scaling bypass 取消 alias fast-return，`output_split_sizes=None` 与 equal-rows 场景均强制 materialize copy。
   - 验证结果：
     - unit/static 均 PASS（`test_mappings_moe_api.py` 5/5）；
     - 局部 autograd 微复现实验中，`_AllToAll.apply` 在 `is_scaling_mode=False/True` 下图节点计数一致（均为 3，且都包含 `_AllToAllBackward`）；
     - postfix NSYS analyzer（dist/scale）均 `phase_window_parents=48`, contamination `0.00%`；
     - compare（`pure_primary_union + contamination gate`）仍 FAIL：
       - op-rank median: `forward=12.25%`, `backward=19.72%`, `optimizer=5.32%`。
   - pre/post 对照（同一 x1 协议）：
     - backward median `17.97% -> 19.72%`（无改善）；
     - stage1 backward (`rank4..7, steady`) pair-median `21.62% -> 21.73%`（近似不变）。
   - 结论：
     - 该 postfix 是语义硬化/归因修正，不是当前 residual 的主因修复；
     - 仍需按 `seq8192 + rank7-cap + repeat-x5` 做 phase 语义冻结验证。

54. **scaling comm-adjacent 补齐方案可行但对 x1 backward 收敛效果有限，且主残差并非 comm-adjacent 主导**
   - 深入归因证据（`round68 run5`, `seq8192`, stage1 backward ranks 4-7）：
     - `dist-scale` gap（primary_union）=`238.288 ms`；
     - comm-adjacent/data-movement 分类贡献 `70.559 ms`（`27.31%`）；
     - 主要增量来自非 comm kernel 家族：`fmha_cutlassB` delta `149.201 ms`。
   - 实验改动（default-off）：
     - 新增 `--scaling-comm-adjacent-copy-iters`，在 scaling all_to_all backward 路径注入可控 copy 代价。
   - x1 验证（`SEQ_LEN=1024`, phase-pure compare）：
     - copy0: `backward=19.72%`
     - copy2: `backward=19.56%`
     - copy8: `backward=19.67%`
   - 结论：
   - 该方案技术可行，但当前观测下对 backward residual 改善不具决定性；
   - 需要把下一步重点转向非 comm 主导项（尤其 attention backward 家族）与 stage1 语义对齐。

55. **cross-run 证据确认：backward residual 的主导项稳定为 non-comm（`fmha_cutlassB`），`_AllToAll` 图节点膨胀假设在微复现层面不成立**
   - 数据范围（历史 artifacts 复核）：
     - `round68 seq8192 run1..5`
     - 过滤片段：`stage1 backward steady`, `rank=4..7`
   - 稳定性证据：
     - 每轮 `dist-scale` gap 均在 `~225.6..240.6 ms`；
     - 每轮 top-1 增量 kernel 都是 `fmha_cutlassB...`，贡献 `146.581..153.787 ms`；
     - 说明“non-comm 主导”不是单次偶发现象。
   - `_AllToAll` 微复现证据：
     - distributed/scaling 两分支的 autograd 节点集合和计数一致：
       - `['MulBackward0', 'SumBackward0', '_AllToAllBackward']`，均为 3 节点。
   - 结论：
     - “distributed backward 主要因为 scaling bypass 丢失 `_AllToAll` backward 节点而导致图更大”缺乏直接证据；
     - 根因定位应继续聚焦 attention-family / non-comm 路径差异，而非仅 comm-adjacent 补齐。
   - 缓解建议：
     - 在现有 phase-pure 语义下增加 attention-family 诊断标签与归因报表（rank/stage/state 分桶）；
     - 保留 `--scaling-comm-adjacent-copy-iters` 作为 debug-only 开关，不纳入官方 gate 语义。

56. **DDP-hook 假设验证结论：scaling 中 hook 成本并非缺失，且不是当前 backward 残差主导项**
   - 实验方式：
     - 新增 debug 开关 `--scaling-disable-ddp-wrap`（在 scaling 下禁用 DDP param-hook accumulation path，保持 DDP wrapper 接口）；
     - 运行 x1 NSYS 协议（`SEQ_LEN=1024`, phase-pure）做 scaling A/B。
   - 关键证据（stage1 backward steady, rank4..7）：
     - `compute_pure_primary_union_ms`: `78.899 -> 71.230`（`-9.72%`）；
     - `kernel_count`: `7576 -> 6988`（`-7.76%`）；
     - 主要减少 kernel 是 `CUDAFunctor_add<float>`（`-7.597 ms`）；
     - `fmha_cutlassB` 在该 A/B 中变化极小（`-0.007 ms`）。
   - 对齐影响：
     - dist-vs-scale compare 下 backward rank-median 未改善（`19.72% -> 20.08%`）。
   - 结论：
     - “scaling backward 基本没有 DDP hook 行为”不成立；
     - DDP hook overhead 存在但不足以解释主残差，当前主导项仍需继续聚焦 non-comm 路径（attention family）。
   - 备注（执行中发现并已处理）：
     - 直接跳过 DDP wrapper 会触发训练接口不兼容（如 `zero_grad_buffer`/`expert_parallel_buffers` 依赖）；
   - 已转为“保留 wrapper、禁用 hook accumulation”的 debug 路径，并补充优化器 buffer-collection 健壮性判断。

57. **seq8192 phase-pure DDP probe A/B 显示：DDP-off 可降低残差但不足以冻结 backward，主残差仍由 attention-family 主导**
   - 新采集范围（x1 probe）：
     - distributed: `deepseek_phase_sl8192_dist_ddp_probe`
     - scaling on/off: `deepseek_phase_sl8192_scaling_ddp_on|off`
   - 语义洁净性：
     - 三组数据均为 `phase_window_parents=48`, `event_rows=72`, `aggregate_rows=24`;
     - contamination 全量 `0.00%`，排除 comm-window 泄漏作为主因。
   - compare（`pure_primary_union + shared(primary_stream)`）：
     - DDP-on rank-median: `forward=7.20%`, `backward=16.88%`, `optimizer=5.45%`
     - DDP-off rank-median: `forward=4.91%`, `backward=9.00%`, `optimizer=1.96%`
   - stage1 backward steady (`rank4..7`)：
     - DDP-on median diff `8.83%`；
     - DDP-off median diff `6.04%`（改善但仍 >5%）。
   - kernel-family 证据：
     - dist-vs-scale top delta 在 on/off 两组都仍是 `fmha_cutlassB`：
       - on: `+40.965 ms`
       - off: `+32.653 ms`
   - 风险结论：
     - DDP hook accumulation 是“可观但次级”贡献项，不足以单独解释并修复 backward residual；
     - backward 官方 freeze 仍需 `seq8192 + rank7-cap + repeat x5` 的 phase-pure 协议验证，并继续聚焦 attention-family 归因。

58. **seq8192 x1 probe 与历史 round68 高残差集出现“方向翻转”（本轮 scale>dist），说明单轮结论对协议/运行上下文高度敏感**
   - 现象：
     - 本轮 seq8192 probe（phase-pure）中，stage1 backward steady（rank4..7）呈现 `scale > dist`：
       - DDP-on median diff `+8.83%`
       - DDP-off median diff `+6.04%`
     - 但历史 `round68 seq8192 run1..5` 高残差集主要表现为 `dist > scale`（且 `fmha_cutlassB` 主导）。
   - 影响：
     - 若只基于单轮 x1 probe，容易得到与历史 repeat 数据方向不一致的结论；
     - backward freeze 结论的可信度需要更严格的重复采样和稳健聚合。
   - 缓解建议：
     - 坚持官方冻结协议：`seq8192 + rank7-cap + repeat x5`；
     - 除 median 外同时看 IQR/P75 与 rank-level稳定性（尤其 rank1/6/7 异常敏感点）；
     - 在 freeze 轮中保留 DDP on/off 与 attention-family 归因快照，避免“方向翻转”误判。

59. **seq8192 phase-pure 正式 repeat-x5 结果显示：DDP-off 仅部分改善 backward，且稳定性未收敛；attention-family 仍为主导残差**
   - 正式轮结果（`pure_primary_union`, shared primary-stream）：
     - DDP-on backward median/IQR：`13.02% / 4.23`
     - DDP-off backward median/IQR：`12.27% / 5.26`
     - 结论：DDP-off 仅小幅改善 backward median（`-0.75pp`），但 IQR 变大，未达到冻结条件。
   - 语义完整性：
     - distributed/scaling on/off 全部 run 均 `contamination_pct=0.00%`，可排除 comm-window 泄漏。
   - kernel-family 稳健证据（stage1 backward steady, rank4..7）：
     - `fmha_cutlassB` 在 on/off 两分支均为 `5/5` run 的 top1 absolute delta；
     - fmha delta median：
       - on: `38.447 ms`
       - off: `30.666 ms`
   - 风险结论：
     - backward residual 不能通过 DDP-hook 路径单独收敛；
     - 继续增加 comm-adjacent emulation 的收益预期低。
   - 优先级调整：
     - 下一步优先 attention-family 诊断（debug-only NVTX tags/segmentation），并保留现有 comm-adjacent knobs 为 debug-only。

60. **NVTX 归属污染已被确认：`row_g_fwd` push/pop 不平衡导致 forward CMD 长窗泄漏并系统性覆盖 backward**
   - 证据（基于已采集 `seq8192 phase repeat-x5` sqlite，非新采集）：
     - 所有 run/branch（dist/scaling_on/scaling_off）均出现：
       - `open_forward=24`, `open_backward=0`;
       - `forward/backward overlap_cnt=48`；
     - unclosed label 指纹稳定：
       - `row_g_fwd_open` 持续非零（dist=48，scaling=96）；
       - `cmd_forward_open=24`。
   - 根因定位：
     - `megatron/core/tensor_parallel/mappings.py` 中 `_ReduceFromModelParallelRegion.forward` 在 `world_size==1` 分支提前 `return`，未执行 `nvtx.range_pop()`。
     - 该路径在当前 TP=1 workload 为高频路径，导致 NVTX 栈持续泄漏，后续 CMD pop 目标错位。
   - 影响：
     - op-window 归属语义（尤其 forward/backward 边界）可被系统性污染；
     - 受污染 traces 上的 per-op kernel 计数/时长结论可信度下降。
   - 修复状态：
     - 已落地最小修复：`try/finally` 保证 `row_g_fwd` 总能 pop；
     - 已完成 RED→GREEN 单测验证（新增 `world_size==1/ >1` NVTX 平衡用例）。
   - 后续风险与要求：
     - 在 patched 代码上必须先做一轮 seq8192 x1 重新采集，确认 `open_forward==0` 且不再出现系统性 fwd/bwd overlap；
     - 通过后再进行正式 repeat-x5 freeze 复测。

61. **post-fix clean x1 已通过 NVTX 结构 gate，但 backward residual 仍显著超阈值，说明主矛盾已转向 attention-family 路径**
   - 新证据（patched `seq8192 phase-pure x1`）：
     - dist/scaling_on/scaling_off 三组都满足：
       - `open_forward_step=0`
       - `open_backward_step=0`
       - `forward_backward_overlap_count=0`
     - 同时三组 contamination 仍为 `0.00%`。
   - 对比结果（`pure_primary_union`, shared primary-stream）：
     - DDP-on：`forward=15.63%`, `backward=17.42%`, `optimizer=3.51%`
     - DDP-off：`forward=15.74%`, `backward=10.52%`, `optimizer=2.35%`
   - 结论：
     - NVTX 结构污染已不是当前 x1 残差的主要解释；
     - backward 在 clean traces 上仍高于门限，且 DDP-off 仅部分改善。
   - 新优先级：
     - 转向 attention-family 根因诊断（debug-only tags/segmentation）；
     - 在 attention 诊断得到可执行修复方向前，不进入 repeat-x5 freeze 正式轮。

62. **attention-family 深诊断确认：`fmha_cutlassB` 仍为 patched clean-x1 backward 残差主导，且不是 launch-config mismatch**
   - 诊断范围：
     - 数据源：patched `seq8192 phase-pure x1`；
     - 过滤：`op=backward_step`, `state=steady`, `stage=1`, `phase=compute`, `rank=4..7`；
     - 比对：`dist vs scaling_on`、`dist vs scaling_off`。
   - 关键证据：
     - pairing 完整：两组均 `paired_windows=12`, `missing=0`；
     - 残差分解：
       - on：`gap=+42.122 ms`, `fmha_gap=+25.442 ms`, share=`60.40%`；
       - off：`gap=+28.344 ms`, `fmha_gap=+20.650 ms`, share=`72.86%`；
     - launch parity：
       - 两组均 `dist_unique_cfg=1`, `scale_unique_cfg=1`, `cfg_sets_equal=True`；
       - 说明 kernel launch shape/config 并未分叉。
     - top-k：
       - 两组 top1 absolute delta 均是 `fmha_cutlassB...`。
   - 风险结论：
     - DDP-off 只能部分减小 gap，但不改变主导项；
     - 根因优先级应转向 attention-path runtime context（邻接 memory traffic / stream scheduling / micro-phase 归因），而非继续扩展 comm-adjacent emulation 或 kernel-shape 假设。
   - 缓解建议：
     - 增加 debug-only attention 微分段 tags（qkv / softmax-bwd / dropout-bwd / proj-bwd）并复用 phase-pure compare；
     - 在进入下一轮 repeat-x5 freeze 前，先完成一轮 clean x1 attention 微分段验证。

63. **attention backward 微分段定位已稳定：主残差集中在 `attn_core_bwd`，且 top1 恒为 `fmha_cutlassB`（repeat-x5）**
   - 实施：
     - 新增 debug-only 开关 `--trace-attention-backward-segments`；
     - 分段标签 `attn_bwd_segment=*` 覆盖 SelfAttention 与 MLA 路径（DeepSeek workload）。
   - clean x1 + repeat-x5 证据（`stage1 backward steady rank4..7`）：
     - `attn_core_bwd` 是唯一 material-gap segment；
     - core segment 的 `fmha_gap_share` 中位数：
       - scaling_on: `98.01%`
       - scaling_off: `97.96%`
     - top1 delta kernel 在 on/off 的 `5/5` run 都是 `fmha_cutlassB`。
   - 结论：
     - 分段定位已收敛，可判定 residual 主源为 attention core runtime-context。

64. **stream-set 不匹配证据不足，但 distributed 在 fmha 前邻接小 kernel 成本持续更高（core segment）**
   - repeat-x5 core-segment 诊断：
     - `primary_stream_id_mismatch_pairs=0`、`fmha_stream_set_mismatch_pairs=0`（各 run 均无异常）；
     - 但 pre-fmha small-kernel 邻接统计稳定偏高于 scaling：
       - count median: dist `35` vs scale `22`
       - ms median: dist `1.751` vs scale `~1.10`
   - 风险结论：
   - 当前更像同流内邻接负载/时序上下文差异，而非 stream route mismatch；
   - 仍需更细 attention 邻接诊断（core 前后 micro-tag）来识别可消减项。
   - gate 影响：
     - 即使定位收敛，op-level backward 在 repeat-x5 仍 >5%（on `11.20%`, off `8.99%`），官方 backward freeze 仍未达成。

65. **attention-core 进一步微分段后确认：残差几乎全部集中在 `attn_core_sdpa_bwd`，pre/post-cast 子段可忽略**
   - 新增证据（`seq8192` clean x1，dist/scaling on/off）：
     - 新增子段标签：
       - `attn_core_precast_bwd`
       - `attn_core_sdpa_bwd`
       - `attn_core_postcast_bwd`
     - 三分支标签计数一致（`attn_core_*_bwd` 各 `96`），排除标签覆盖偏差。
   - 定位结果（`stage1 backward steady rank4..7`）：
     - scaling_on：
       - `attn_core_bwd gap=23.419ms`
       - `attn_core_sdpa_bwd gap=23.115ms`, `fmha_share=98.02%`
       - `attn_core_precast_bwd gap=0.000ms`
       - `attn_core_postcast_bwd gap=0.000ms`
     - scaling_off：
       - `attn_core_bwd gap=34.155ms`
       - `attn_core_sdpa_bwd gap=33.740ms`, `fmha_share=98.11%`
       - `attn_core_precast_bwd gap=0.000ms`
       - `attn_core_postcast_bwd gap=0.000ms`
   - 新邻接证据（top-name）：
     - `attn_core_sdpa_bwd` pre-fmha 邻接主导 kernel 稳定为
       - `vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`
     - dist vs scale pre-fmha 统计：
       - scaling_on：`35 / 1.751ms` vs `27 / 1.393ms`
       - scaling_off：`35 / 1.751ms` vs `18 / 0.900ms`
   - 风险结论：
     - 当前 residual 不是“core 内多段均匀扩散”，而是 SDPA-backward 邻域上下文主导；
     - 后续应优先做 SDPA 邻域专项诊断（mask/build 与 memory-traffic 邻接），而非继续扩大 comm-adjacent 仿真范围。

66. **`attn_core_sdpa_bwd` immediate-neighbor 诊断显示：pre-fmha 邻接差异对阈值敏感，不能作为主残差根因**
   - 变更与方法：
     - 在 `analyze_nsys_attention_family_delta.py` 新增 immediate same-stream 邻接统计：
       - `small_kernel_immediate_pre/post_count`
       - `small_kernel_immediate_pre/post_ms`
       - immediate pre/post top names（`name/ms/count`）。
     - 使用现有 `seq8192` clean x1 sqlite（`dist/scaling_on/scaling_off`）重分析 `attn_core_sdpa_bwd`（`stage1 backward steady rank4..7`）。
   - 证据（默认阈值 `small_kernel_threshold_us=60`）：
     - scaling_on immediate pre：dist `35 / 1.751ms` vs scale `27 / 1.393ms`
     - scaling_off immediate pre：dist `35 / 1.751ms` vs scale `18 / 0.900ms`
     - top1 immediate pre kernel 在两侧都相同：
       - `vectorized_elementwise_kernel<... FillFunctor<unsigned char> ...>`
   - 阈值扫（`small_kernel_threshold_us=80`）：
     - scaling_on immediate pre：dist `44 / 2.311ms` vs scale `44 / 2.451ms`
     - scaling_off immediate pre：dist `44 / 2.311ms` vs scale `44 / 2.518ms`
     - `gap_ms` 不变（on `23.115ms`, off `33.740ms`）。
   - 结论：
     - pre-fmha immediate-neighbor 路径在 distributed/scaling 均存在，不支持“scaling 缺失该邻接路径”为主因；
     - `60us` 口径下的 count 差异主要由 small-kernel 分类阈值敏感性导致；
     - backward 主残差仍应聚焦 fmha runtime-context 本体（`attn_core_sdpa_bwd`），而非邻接 path 缺失假设。

67. **seq8192 attention-core SDPA-subsegment 正式 repeat-x5 显示：主导源稳定但 run-level 波动仍高，backward freeze 仍未满足**
   - 协议：
     - `dist + scaling_on + scaling_off`，`SEQ_LEN=8192`，`TRAIN_ITERS=3`；
     - phase-pure 与 attention segment tracing 配置保持不变（无训练语义改动）。
   - gate 与语义洁净性：
     - NVTX structure gate 在 run1..5 三分支全部通过；
     - contamination gate 在 run1..5 三分支全部为 `0.00%`。
   - 稳健定位（`stage1 backward steady rank4..7`, `segment=attn_core_sdpa_bwd`）：
     - top1 kernel 在 on/off 均为 `5/5` 的 `fmha_cutlassB...`；
     - `fmha_gap_share_pct` median：
       - scaling_on: `98.022%`
       - scaling_off: `98.041%`
     - pre-fmha top-name 在 on/off 全部 run 稳定为 `FillFunctor<unsigned char>`；
     - distributed pre-fmha 邻接负载中位数仍高于 scaling（count/ms: `35/1.751` vs `21/~1.05`）。
   - 波动风险：
     - `gap_ms` IQR：
       - scaling_on: `7.021`
       - scaling_off: `20.024`（显著更高）
     - backward op-rank-median（compare logs）仍存在较大 run 间漂移，未达到 freeze 收敛预期。
   - 结论与影响：
   - `attn_core_sdpa_bwd` 主导来源已可视为稳健事实（非单轮伪影）；
   - backward freeze 仍被 run-level variability 阻塞，下一步应继续 SDPA 邻域 runtime-context 的 targeted 诊断，而不是扩展 comm-adjacent emulation。

68. **attention-core deep-segment repeat-x5 波动分桶显示：residual 主要由 iter1 子桶驱动，freeze 需补充按-iter 稳健性约束**
   - 协议与语义状态：
     - `seq8192`, run1..5, `dist + scaling_on + scaling_off`；
     - phase-pure 与 NVTX structure gate 全通过（`open/overlap=0`），contamination 全为 `0.00%`。
   - 主导源稳定性（复核）：
     - `attn_core_sdpa_fmha_bwd` 仍是主导段；
     - top1 kernel `fmha_cutlassB...` 在 on/off 都是 `5/5`。
   - 分桶证据（`stage1/backward/steady/rank4..7`, fmha segment paired windows）：
     - on/off 各 `n=240`；
     - `iter1` 是一致的高-gap 子桶：
       - on: `fmha_gap median ~1.766ms`
       - off: `fmha_gap median ~1.777ms`
     - `iter0` 与 `iter2` 接近低-gap子桶（中位数接近 0）；
     - `iter1` 同时出现稳定 pre-fmha 邻接偏移：
       - `small_pre_count_gap_median = -1.0`
       - `small_pre_gap_ms_median ~ -0.05ms`
   - 风险结论：
     - backward residual 不是纯随机噪声，也不是 phase 泄漏，而是有稳定的 iter-context 结构性来源；
     - 当前 freeze 口径若不区分 iter 子桶，会把结构性上下文波动混成单一 IQR，导致 gate 判定不稳定。
   - 缓解建议：
     - freeze 评估增加按-iter 子桶报告（至少独立报告 iter1 与 iter0/2）；
     - 在深分段语义下优先诊断 iter1 对应的 runtime-context（同流 pre-fmha 邻接、batch/调度相位）后再决定是否收紧/放宽 backward 稳健性约束。

69. **官方 backward 口径已冻结为 `seq8192 + phase-pure + repeat-x5`，stage-aware/op-map/no-subtract 降级为诊断视图**
   - 决策依据：
     - round68 以后三视图长期分歧已被确认，stage-aware 虽可降数值但属于不可扩展标定口径；
     - phase-pure 路径已提供 contamination=0 的可验证语义。
   - 实施状态：
     - 在 task plan/notes 中落盘“主口径冻结”条款；
     - 后续汇报必须优先给出 `median/IQR/range`，再附诊断口径对照。
   - 风险控制：
     - 避免以口径切换替代 root-cause 修复；
     - 避免使用 stage-aware 数值作为官方对外结论。

70. **示例脚本已实施 advanced-diagnostics 显式确认策略，降低默认路径被实验开关污染的风险**
   - 变更摘要：
     - `examples/pretrain_deepseek_v3_moe.sh`、`examples/pretrain_qwen3_30b_a3b_moe.sh` 新增 `ADVANCED_DIAGNOSTICS=0|1`；
     - 当检测到高级诊断开关被设置且未显式确认时，脚本 fail-fast 并输出活动开关列表。
   - 影响：
     - baseline 运行语义保持不变；
     - 实验性开关不再“静默混入”默认流程。
   - 后续要求：
     - 所有涉及高级诊断开关的报告需显式记录 `ADVANCED_DIAGNOSTICS=1`。

71. **后续代码级修复范围已锁定：仅允许 `attn_core_sdpa_bwd` iter-bucket（尤其 iter1）定向诊断，不再扩展非主线 A/B**
   - 决策依据：
     - `fmha_cutlassB` 在多轮 repeat 中稳定 top1；
     - comm-adjacent 与 DDP-hook 路径已验证为次级项。
   - 允许方向：
     - `attn_core_sdpa_bwd` 前后微段与同流邻接 runtime-context 诊断；
     - iter1 vs iter0/2 分桶一致性验证。
   - 禁止方向（除非新证据）：
     - 扩大 comm-adjacent emulation 口径；
     - 继续以 subtraction-policy 调参追求数值对齐。

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
