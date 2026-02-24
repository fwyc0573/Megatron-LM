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

## Resolved During Stage-1

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
