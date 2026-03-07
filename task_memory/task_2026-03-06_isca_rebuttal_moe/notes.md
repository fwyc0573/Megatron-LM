## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-06 | Initialized working notes for ISCA rebuttal MoE task |
| 2026-03-06 | Added final experiment and paper-integration notes |
| 2026-03-07 | Added GPU memory simulation rebuttal notes and wording constraints |

# Notes

## Reviewer-driven requirements
- Address latest-model MoE generalization concerns.
- Explicitly tie MoE communication to `all-to-all` modeling and overlap behavior.
- Add direct evidence for routing imbalance / straggler sensitivity.
- Avoid over-claiming capacity-factor support because token-dropping is unsupported in the current stack.

## Current evidence already in paper
- Latest MoE section covers Qwen3-A3B and DeepSeek-V3 variant.
- `all-to-all` model and validation already exist.
- MoE dependency preservation is mentioned in timeline composing.

## Implementation constraints
- Keep changes surgical and isolated from unrelated repo modifications.
- Prefer one small experiment path and one mini-table instead of adding a large new figure.

## Final decisions
- Keep a single DeepSeek-V3-variant routing-skew mini-table in the paper.
- Use the `SEQ_LEN=256` smoke table for the paper because the strong-skew regime is clearest there.
- Treat the `SEQ_LEN=512` rerun as a robustness check stored under this task directory.
- Phrase the paper claim conservatively around the clearly stressed strong-skew regime rather than claiming strict monotonicity across all three skew levels.

## GPU memory simulation rebuttal notes
- Scaling mode launches one fake rank per process via single-process `torchrun`; ranks are not stacked in one long-lived CUDA process.
- This process isolation is the key answer to the reviewer's concern about residual allocator state polluting subsequent rank measurements.
- The practical single-GPU memory requirement tracks the largest per-rank shard plus step-local states under the target parallel configuration, not the aggregate cluster-wide memory footprint.
- Runtime memory tracing records `allocated_memory_MB`, `reserved_memory_MB`, `peak_allocated_MB`, and `theoretical_memory_MB`.
- Do not claim faithful replay of NCCL communication-time memory regions: current rebuttal evaluation explicitly scopes memory validation to per-rank training-step behavior excluding communication-time execution.
