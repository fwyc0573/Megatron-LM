## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-06 | Added routing-skew experiment summary and paper-integration decisions |

# Results

## Paper-facing decisions

- Use the existing latest-model MoE results in `docs/isca-rebuttal/rebuttal/evaluation.tex` as the main answer to the "Mixtral-only / outdated MoE" concern.
- Use the existing `all-to-all` model and validation to answer the explicit MoE communication-modeling question.
- Add one small routing-skew mini-table for the DeepSeek-V3 variant to answer the routing-imbalance / straggler concern.
- Keep the capacity-factor discussion honest: current paired validation covers the no-token-dropping path only.

## Routing-skew experiment summary

### Selected paper table (`SEQ_LEN=256`, `MICRO_BATCH_SIZE=1`)

| Skew level | Hottest/median expert load | GT straggler ratio | Moye straggler ratio | Critical-path proxy error |
|---|---:|---:|---:|---:|
| Balanced | 1.00 | 1.04 | 1.24 | 7.88% |
| Moderate | 1.54 | 1.02 | 1.08 | 4.40% |
| Strong | 3.00 | 1.08 | 1.22 | 6.64% |

### Larger-workload recheck (`SEQ_LEN=512`, `MICRO_BATCH_SIZE=1`)

| Skew level | Hottest/median expert load | GT straggler ratio | Moye straggler ratio | Critical-path proxy error |
|---|---:|---:|---:|---:|
| Balanced | 1.00 | 1.04 | 1.11 | 11.18% |
| Moderate | 1.50 | 1.04 | 1.08 | 0.95% |
| Strong | 3.00 | 1.05 | 1.08 | 4.88% |

## Interpretation

- The strongest skew regime is the clearest reviewer-facing evidence: it raises the GT straggler ratio above the near-balanced baseline range and remains within single-digit critical-path proxy error.
- The moderate-skew regime stays close to balanced in both `SEQ_LEN=256` and `SEQ_LEN=512`, indicating that moderate imbalance is partly masked by system noise in this small smoke configuration.
- Therefore the paper text should avoid claiming strict monotonicity across all three skew levels, and instead claim that `Moye` captures the stressed strong-skew regime beyond near-balanced routing.

## Rebuttal integration

- `docs/isca-rebuttal/rebuttal/evaluation.tex`
  - Reframe Qwen3-A3B + DeepSeek-V3 variant as modern MoE generalization rather than extra models.
  - Add a compact routing-skew mini-table and a cautious interpretation paragraph.
  - Tie the `all-to-all` evaluation explicitly to MoE expert dispatch/combine.
- `docs/isca-rebuttal/rebuttal/comm_pred.tex`
  - State that `all-to-all` is the MoE dispatch/combine primitive and is modeled explicitly as chunked point-to-point communication.
- `docs/isca-rebuttal/rebuttal/workload_const.tex`
  - Explain why MoE dependencies cannot be relaxed under routing imbalance.
- `docs/isca-rebuttal/rebuttal/discussion.tex`
  - State that token-dropping / capacity-aware admission control remains future work.
## Additional long-sequence validation

- See `task_memory/task_2026-03-06_isca_rebuttal_moe/long_seq_routing_skew_validation.md` for the `SEQ_LEN=1024/2048/3072` paired results and the `SEQ_LEN=4096` OOM boundary.
- Conclusion: the longer-sequence runs are useful as scope validation, but they do not replace the stronger `SEQ_LEN=256` reviewer-facing skew table.

