## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-24 | Added stage-1 blockers/risks and mitigation notes |
| 2026-02-24 | Added validation-time findings (scaling NaN router scores and unit-test harness constraints) |
| 2026-02-24 | Updated with NaN timing-impact conclusion and router test-path fixes |
| 2026-02-24 | Added 32-rank scaling validation findings and rank0/rank7 comp-timing gap root-cause notes |

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

5. **rank0/rank7 comp timing gap (>5%) between distributed and scaling remains partially unresolved**
   - Observation (latest compare log): `forward_step/backward_step` still has >5% deviation on部分rank.
   - Confirmed non-functional blockers removed:
     - trace文件完整、格式正确、stage-op序列正确；
     - scaling脚本已避免默认忙卡（自动选择idle GPU）；
     - MoE hot-path debug sync噪声已移除。
   - Current hypothesis:
     - scaling-mode中的simulation dispatch/permute路径仍引入额外开销，且未被完全归类为comm子操作；
     - distributed路径与scaling路径在调度执行语义上仍存在结构性差异（stage-1设计限制）。
   - Mitigation: keep stage-1可用性结论（功能/trace完整）并将“timing calibration for comp parity”列入stage-1.5/2。

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
