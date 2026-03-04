## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-03 | Initialized issues tracker |
| 2026-03-03 | Recorded old PP4 TP8 OOM blocker and matrix update |
| 2026-03-03 | Added TP32 collective-sim runtime bottleneck analysis and mitigation |
| 2026-03-03 | Added bug1/bug3 re-audit conclusions and follow-up actions |
| 2026-03-03 | Closed bug3 parser legacy-estimator issue after minimal fix and regression validation |

# Issues

## Open
1. Bug1 follow-up pending:
   - TP partition correctness已验证，但 `sub_comp` 时延对 TP 不敏感的问题仍在。
   - 需按建模问题设计校准方法（例如分段效率模型或基于 kernel-family 的 correction）。

## Resolved
1. Step0 static backend capability confirmed.
2. Step0 dynamic probe under world_size=1024 confirmed `g_in_group == TP` and monotonic latency for TP=8/16/32/64.
3. Strict GPT-175B TP64 infeasibility confirmed by code-level divisibility check (`96 % 64 != 0`).
4. Old blocker (PP4 TP8 optimizer-state OOM) is historical only after PP=8 matrix update.
5. TP32 simulate runtime bottleneck resolved for execution practicality:
   - Symptom: very long repeated `htsim_ndp -rail 32 ... -nodes 1024` calls under collective-sim global placement.
   - Mitigation: runtime-only wrapper `tests/performance/run_simu_with_collective_cache.py` with canonical participant cache key for topology-equivalent groups.
   - Scope: no public API changes; used only in simulation invocation path.
6. fake_tp TP partition activation concern resolved:
   - `pretrain_llama.py` scaling-mode path explicitly sets `config.tensor_model_parallel_size = args.fake_tp`.
   - TE weight-shape validation confirms TP partition follows this overridden value.
7. Bug1 classification clarified:
   - Symptom valid (compute time non-linear vs TP), but evidence does not support TP-partition misconfiguration in current path.
   - Treated as modeling/performance characteristic issue pending calibration design.
8. Bug3 parser legacy-estimator path fixed:
   - Removed parser-time `get_comm_op_exc_time` dependency for `dp_allreduce/ep_allreduce/exp_dp_allreduce` in `process_mg_profile_files`.
   - Added unit tests to ensure parser does not call legacy estimator in this path.
   - Regression comparison confirms after-patch parser logs no longer emit legacy `Comm time calculation`.

## Evidence
- `logs/step0_static_evidence.md`
- `logs/step0_quick_probe.md`
- `logs/tp64_blocking_evidence.md`
- `logs/step4_simulate.log`
- `logs/simulate_tp32_pp8_dp4.log`
- `logs/tp_weight_partition_validation.log`
- `logs/bug1_bug3_audit_2026-03-03.md`
- `logs/bug3_parser_after_patch.log`
- `logs/bug3_regression_compare_2026-03-03.md`
