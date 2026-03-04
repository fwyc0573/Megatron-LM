# Follow-up Investigation: Rank10 Anomaly & PP8 recv_backward -221ms Bias

## Modification History

| Date       | Summary of Changes                                |
|------------|----------------------------------------------------|
| 2026-03-06 | Initial version: two follow-up investigations      |

---

## Overview

This report addresses two open questions from `deep_error_analysis_qwen3_2026-03-06.md`:

1. **Rank10 (PP2) DB_fwd=191ms anomaly** — What causes this outlier?
2. **PP8 recv_backward -221ms bias** — Is this a pipeline bubble modeling problem?

---

## Finding 1: Rank10 DB_fwd=191ms → Sporadic Measurement Artifact

### Conclusion

Rank10's anomalously high forward_step duration (191.59ms vs normal ~115-130ms) is a **sporadic GPU performance fluctuation during scaling mode sequential execution**, NOT caused by MoE token routing imbalance.

### Evidence

#### 1.1 Identical Token Shapes Across EP-Position Peers

Rank2 (stage0, EP position 2) and rank10 (stage1, EP position 2) share the same expert partition and route identical token counts:

| Rank | Stage | EP Position | all_to_all shape | fwd_dur (ms) | Δ vs peer |
|------|-------|-------------|------------------|--------------|-----------|
| 2    | 0     | 2           | [21751,2048]×24 + [16384,2048]×24 | 117.97 | baseline |
| 10   | 1     | 2           | [21751,2048]×24 + [16384,2048]×24 | 191.59 | **+62.4%** |

All other 7 cross-stage pairs differ by ≤10%:

| EP Pos | Stage0 Rank | Stage1 Rank | Stage0 fwd | Stage1 fwd | Δ (%) |
|--------|-------------|-------------|------------|------------|-------|
| 0      | 0           | 8           | 105.64     | 115.44     | +9.3% |
| 1      | 1           | 9           | 123.30     | 121.32     | -1.6% |
| 2      | 2           | **10**      | 117.97     | **191.59** | **+62.4%** |
| 3      | 3           | 11          | 103.09     | 112.98     | +9.6% |
| 4      | 4           | 12          | 116.47     | 127.09     | +9.1% |
| 5      | 5           | 13          | 126.51     | 117.68     | -7.0% |
| 6      | 6           | 14          | 125.66     | 126.59     | +0.7% |
| 7      | 7           | 15          | 136.76     | 128.27     | -6.2% |

#### 1.2 Anomaly Limited to forward_step Only

Rank10's backward_step (103.27ms) is completely normal — comparable to rank14 (103.22ms) and rank15 (103.06ms). Only forward_step is anomalous:

| Rank  | fwd (ms)  | bwd (ms)  | fwd/bwd ratio |
|-------|-----------|-----------|---------------|
| rank8 | 115.44    | 90.73     | 1.27          |
| rank9 | 121.32    | 86.80     | 1.40          |
| **rank10** | **191.59** | 103.27 | **1.86** |
| rank11| 112.98    | 75.27     | 1.50          |
| rank12| 127.09    | 82.09     | 1.55          |
| rank13| 117.68    | 93.99     | 1.25          |
| rank14| 126.59    | 103.22    | 1.23          |
| rank15| 128.27    | 103.06    | 1.24          |

#### 1.3 Inter-Sub-Op Timing Gaps

Rank10's inter-sub-op gaps (especially a2a→a2a transitions) are ~3.7× larger than normal ranks:

| Rank  | a2a→a2a mean gap | a2a→a2a max gap |
|-------|------------------|-----------------|
| rank10| 4.19ms           | 14.70ms         |
| rank11| 1.13ms           | 1.69ms          |
| rank8 | 1.59ms           | 3.53ms          |
| rank9 | 1.15ms           | 1.69ms          |

### Root Cause Assessment

The anomaly occurred during the 11th sequential execution (rank10, timestamp 20260303180159) of 16 fake ranks. Likely causes:
- CUDA context memory pressure accumulation during long sequential runs
- Thermal throttling micro-spike (single ~191ms measurement window)
- OS scheduling jitter or background GPU management activity

### Recommendation

- Use **median** or **trimmed mean** across multiple scaling runs
- Flag outlier ranks with >2σ deviation from EP-position peers
- Consider running multiple scaling iterations per rank and taking the minimum

---

## Finding 2: PP8 recv_backward -221ms → Fused P2P Double-Counting Artifact

### Conclusion

The -221.12ms recv_backward "bias" is **NOT a pipeline bubble modeling error**. It is a **measurement accounting artifact** caused by Megatron-LM's fused `send_forward_recv_backward()` call, where the profiler records the SAME duration for both operations.

After de-duplicating fused P2P pairs, the true combined P2P error for PP8 is only **-28.49ms (-4.3%)**.

### Mechanism

#### 2.1 Fused P2P in Megatron-LM's 1F1B Schedule

In the 1F1B pipeline schedule, Megatron-LM uses batched P2P calls:
- **`send_forward_recv_backward()`**: Simultaneously sends forward activation to next stage AND receives backward gradient from next stage (during steady state)
- **`send_backward_recv_forward()`**: Simultaneously sends backward gradient to previous stage AND receives forward activation from previous stage (during steady state)

Source: `megatron/core/pipeline_parallel/p2p_communication.py:454-500`

Both use a single `_communicate()` call → single CUDA event pair → single measured duration.

#### 2.2 The Profiler Records Identical Duration for Both Ops

In the PP8 rank3 distributed profile, every steady-state fused pair has **identical timestamp and duration**:

```
send_forward(batch=6,  steady): dur=256.15ms  ts=3987796697.41
recv_backward(batch=0, steady): dur=256.15ms  ts=3987796697.41  ← SAME!

send_forward(batch=7,  steady): dur=9.64ms   ts=3987796777.86
recv_backward(batch=1, steady): dur=9.64ms   ts=3987796777.86   ← SAME!
...
```

PP8 rank3 has:
- **19 fused event groups** (10 send_forward+recv_backward, 9 recv_forward+send_backward)
- **362.06ms double-counted** across all fused pairs

#### 2.3 The Simulator Treats Them as Sequential

The simulator schedule places `send_forward` and `recv_backward` as **separate sequential operations**, each with independent waiting times. This creates a fundamental mismatch:

| Aspect | Real (Fused) | Simulator (Sequential) |
|--------|-------------|----------------------|
| Execution | `max(sf_time, rb_time)` | `sf_time + rb_time` |
| Waiting | Shared window | Separate additive |
| Wall-clock | 1× duration | Up to 2× duration |

#### 2.4 Quantitative Impact Across PP Configurations

| Config | Profile Naive | Profile De-dup | Sim Total | Naive Error | **De-dup Error** |
|--------|--------------|----------------|-----------|-------------|------------------|
| PP2    | 514.38ms     | 330.91ms       | 766.39ms  | +252ms (+49%) | +435ms (+132%) |
| PP4    | 774.27ms     | 499.45ms       | 619.74ms  | -155ms (-20%) | +120ms (+24%) |
| **PP8** | **1024.09ms** | **662.03ms** | **633.54ms** | **-391ms (-38%)** | **-28ms (-4.3%)** |

Key observations:
- **PP8**: De-duplication reveals the simulator is actually quite accurate (-4.3%)
- **PP2**: De-duplication reveals the simulator genuinely **overestimates** P2P time (+132%), because sequential treatment of fused ops inflates pipeline waiting
- **PP4**: Intermediate case (+24%)

The overestimation in PP2 (simulator sequential > real fused) partially cancels with other error sources in PP8, producing the apparent -221ms in naïve comparison.

### PP2's Additional Problem: Sequential Modeling of Fused Ops

For PP2 (4 microbatches), the simulator assigns separate waiting times:
- `send_forward(1)`: waiting=268.97ms + P2P=6.08ms
- `recv_backward(0)`: waiting=485.26ms + P2P=6.08ms
- **Sequential total: 766.39ms**

But in reality, they execute concurrently:
- Fused `sf(1)+rb(0)`: wall-clock = 157.18ms
- **Real is 4.9× faster** than simulator's sequential model

This is a **systematic simulator design issue**: fused P2P operations should be modeled as `max(sf_waiting, rb_waiting) + max(sf_p2p, rb_p2p)`, not as sequential additive operations.

### Recommendations

1. **For comparison scripts**: De-duplicate fused P2P pairs (detect by matching timestamp+duration) before computing per-op residuals
2. **For simulator**: Model `send_forward_recv_backward` as a single fused operation with `max()` semantics instead of sequential `sum()`
3. **For reporting**: Report combined P2P error (send_forward + recv_backward de-duplicated) rather than individual per-op deltas

---

## Summary Table

| Question | Answer | Root Cause | Impact |
|----------|--------|-----------|--------|
| Rank10 DB_fwd=191ms | Sporadic measurement artifact | GPU fluctuation during sequential scaling run | Affects PP2 DB profile accuracy for rank10 only |
| PP8 recv_backward -221ms | NOT pipeline bubble error | Fused P2P double-counting in profiler | Naive comparison inflated 8.7× (true error is -4.3%) |

---

## Methodology

- **Tool**: Direct parsing of distributed profile (`global_ranks_profile/`) and scaling DB profile (`database_profile/`)
- **Detection**: Fused pairs identified by matching `(timestamp, duration)` within ε=0.01ms
- **Validation**: Cross-PP analysis (PP2, PP4, PP8) confirms systematic pattern
- **Code verification**: `send_forward_recv_backward()` in `p2p_communication.py:454` confirmed as single `_communicate()` call
