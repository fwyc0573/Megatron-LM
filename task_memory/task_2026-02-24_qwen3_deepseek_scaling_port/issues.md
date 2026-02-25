## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
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
- **Scaling TP>1 hard-block in router/dispatcher path** resolved for current tracing workflow:
  - replaced scaling-unsafe TP assertions with fake-TP-aware handling.
- **Scaling EP=1 preprocessing crash** resolved:
  - when precomputed dispatch cache is absent, dispatcher now uses runtime token histogram path.
- **Distributed PP1 backward trace coverage gap** resolved:
  - no-pipeline path now wraps backward with CMD and emits `backward_step` trace records.
- **6-GPU compare rank hardcoding** resolved:
  - compare script now supports `--ranks` and `--ops` for non-8GPU runs.
