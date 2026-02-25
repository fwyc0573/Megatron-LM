## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-25 | Added Qwen3 seq2048 (mbs=8/4) distributed-vs-scaling validation, sub-op attribution audit, and commit-level review conclusions |

## Test Report: Qwen3 seq2048 (6-GPU) Scaling vs Realistic Comp Alignment

**Date**: 2026-02-25  
**Environment**: `conda activate myenv_yc` (Python 3.9.18)  
**Project Root**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM`

### 1) Test Script Information

#### 1.1 Reviewed Code Scope

- Current commit: `457ae681` (`Refine no-calibration scaling trace diagnostics`)
- Previous commit: `661077e5` (`Add stage-1.5 trace timing calibration and compare tool`)
- Current working-tree target files (this round focus):
  - `megatron/profiler/cmd.py`
  - `megatron/profiler/comm_utils/interception_comm.py`
  - `megatron/core/tensor_parallel/mappings.py`
  - `tests/performance/compare_qwen_trace_comp.py`
  - `examples/pretrain_qwen3_30b_a3b_moe.sh`
  - `megatron/core/pipeline_parallel/schedules.py`
  - `megatron/core/distributed/finalize_model_grads.py`

#### 1.2 Reproducible Commands

```bash
# 1) Unit validations (trace sync mode + compare + scaling comm interception)
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29620 PYTHONPATH=$(pwd) \
pytest -q \
  tests/unit_tests/profiler/test_cmd_subop_sync_mode.py \
  tests/unit_tests/profiler/test_interception_comm_scaling_mode.py \
  tests/unit_tests/performance/test_compare_qwen_trace_comp.py \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_default_global \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_event \
  tests/unit_tests/test_training.py::TestTraining::test_trace_subop_sync_mode_invalid_value

# 2) Attempt requested primary parallel strategy (expected to fail-fast on model divisibility)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke \
TRAIN_ITERS=1 TRACE_START=1 SEQ_LEN=2048 MICRO_BATCH_SIZE=8 MASTER_PORT=7301 GPUS_PER_NODE=6 \
PP=1 TP=3 EP=2 FAKE_WORLD_SIZE=6 FAKE_TP=3 FAKE_PP=1 FAKE_EXP=2 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

# 3) Runnable fallback: TP=2, DP=3, EP=1, PP=1, mbs=8, seq=2048
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke \
TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=2048 MICRO_BATCH_SIZE=8 MASTER_PORT=7310 GPUS_PER_NODE=6 \
PP=1 TP=2 EP=1 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_EXP=1 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=8 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_EXP=1 \
SCALE_GPU=2 MASTER_PORT=7410 GPUS_PER_NODE=6 PP=1 TP=2 EP=1 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225091619 \
  --threshold-pct 5 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs8_event.log

# 4) Same setup with mbs=4
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 TRACE_SUBOP_SYNC_MODE=event MODE=distributed MODEL_PROFILE=smoke \
TRAIN_ITERS=2 TRACE_START=2 SEQ_LEN=2048 MICRO_BATCH_SIZE=4 MASTER_PORT=7320 GPUS_PER_NODE=6 \
PP=1 TP=2 EP=1 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_EXP=1 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

TRACE_SUBOP_SYNC_MODE=event MODE=scaling MODEL_PROFILE=smoke TRAIN_ITERS=2 TRACE_START=2 \
SEQ_LEN=2048 MICRO_BATCH_SIZE=4 FAKE_WORLD_SIZE=6 FAKE_PP=1 FAKE_TP=2 FAKE_EXP=1 \
SCALE_GPU=2 MASTER_PORT=7420 GPUS_PER_NODE=6 PP=1 TP=2 EP=1 \
bash examples/pretrain_qwen3_30b_a3b_moe.sh

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl2048 \
  --scaling-dir profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl2048 \
  --ranks 0,1,2,3,4,5 \
  --ops forward_step,backward_step,optimizer_step \
  --pair-timestamp 20260225154912 \
  --threshold-pct 5 \
  --no-align-by-state \
  --report-path task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs4_event.log

# 5) Sub-op category/coverage audit
python - <<'PY' | tee task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_seq2048_subop_category_analysis.log
import ast,re
from pathlib import Path

pairs=[
("dist_bs8",Path("realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl2048/wd6_tp2_pp1_exp1_expNum32_l12_bs8_rank0_20260225091433.txt")),
("scale_bs8",Path("profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl2048/wd6_tp2_pp1_exp1_expNum32_numl12_bs8_rank0_20260225091457.txt")),
("dist_bs4",Path("realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl2048/wd6_tp2_pp1_exp1_expNum32_l12_bs4_rank0_20260225091707.txt")),
("scale_bs4",Path("profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl2048/wd6_tp2_pp1_exp1_expNum32_numl12_bs4_rank0_20260225154749.txt")),
]
D=re.compile(r"duration=([0-9.]+)")
S=re.compile(r"sub_operations=(\\[.*\\])")
OP=re.compile(r"^rank:\\d+:(\\w+)\\(")
for tag,path in pairs:
    data=path.read_text().splitlines()
    print(f"==== {tag} {path.name}")
    for target in ("forward_step","backward_step"):
        comm_count={}
        comm_dur={}
        name_count={}
        total_sub=0
        total_comm=0.0
        n=0
        for line in data:
            m=OP.match(line)
            if not m or m.group(1)!=target:
                continue
            n+=1
            sm=S.search(line)
            if not sm:
                continue
            subs=ast.literal_eval(sm.group(1))
            total_sub+=len(subs)
            for sub in subs:
                if "comm_func=" not in sub:
                    continue
                dur=float(D.search(sub).group(1)) if D.search(sub) else 0.0
                total_comm += dur
                cm=sub.split("comm_func=",1)[1].split(",",1)[0]
                nm=sub.split("trace_src_func=",1)[1].split(",",1)[0]
                comm_count[cm]=comm_count.get(cm,0)+1
                comm_dur[cm]=comm_dur.get(cm,0.0)+dur
                name_count[nm]=name_count.get(nm,0)+1
        comm_dur_round={k: round(v,2) for k,v in comm_dur.items()}
        print(f"{target}: records={n}, avg_subops={total_sub/n if n else 0:.2f}, total_comm={total_comm:.2f}")
        print(f"  comm_count={comm_count}")
        print(f"  comm_dur={comm_dur_round}")
        print(f"  trace_src_func_count={name_count}")
    print()
PY

python - <<'PY' | tee task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_seq2048_op_coverage_analysis.log
from pathlib import Path
import re
cases={
"dist_bs8":"realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl2048/wd6_tp2_pp1_exp1_expNum32_l12_bs8_rank0_20260225091433.txt",
"scale_bs8":"profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl2048/wd6_tp2_pp1_exp1_expNum32_numl12_bs8_rank0_20260225091457.txt",
"dist_bs4":"realistic_trace/pp1_tp2_exp1_expn32_dp3_nl12_hs1024_sl2048/wd6_tp2_pp1_exp1_expNum32_l12_bs4_rank0_20260225091707.txt",
"scale_bs4":"profiler_log/pp1_tp2_ep1_expn32_dp3_nl12_hs1024_sl2048/wd6_tp2_pp1_exp1_expNum32_numl12_bs4_rank0_20260225154749.txt",
}
for tag,path in cases.items():
    counts={}
    for line in Path(path).read_text().splitlines():
        m=re.match(r"^rank:\\d+:(\\w+)\\(",line)
        if m:
            op=m.group(1)
            counts[op]=counts.get(op,0)+1
    print(tag,counts)
PY
```

#### 1.3 Evidence Files

- Distributed logs:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_distributed_tp2dp3ep1pp1_seq2048_mbs8_event.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_distributed_tp2dp3ep1pp1_seq2048_mbs4_event.log`
- Scaling logs:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_scaling_tp2dp3ep1pp1_seq2048_mbs8_event.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_scaling_tp2dp3ep1pp1_seq2048_mbs4_event.log`
- Compare reports:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs8_event.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_trace_compare_tp2_6gpu_seq2048_mbs4_event.log`
- Attribution/coverage audit:
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_seq2048_subop_category_analysis.log`
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/qwen_seq2048_op_coverage_analysis.log`

### 2) Validation Criteria

1. **6-GPU run requirement**: use GPUs `2-7` for realistic run.
2. **Parallel strategy fallback rule**: try `TP=3,DP=2,EP=2,PP=1`, fallback only when blocked.
3. **Comp definition check**:
   - verify `total_ms` contains comm sub-op durations in realistic run;
   - compare with `comp_ms = total_ms - comm_ms` for realistic;
   - keep scaling `comp_ms = total_ms` (comm metadata-only, no real collective).
4. **Coverage consistency check**:
   - compare op set and sub-op composition between modes.
5. **Overhead/path consistency check**:
   - inspect whether measurement or execution path adds mode-specific overhead.
6. **Model/load sensitivity check**:
   - compare mbs=8 vs mbs=4 at seq=2048.

### 3) Test Results and Evidence

#### 3.1 Unit Tests

| Suite | Result | Evidence |
|------|--------|---------|
| trace + compare + interception unit tests | PASS | `13 passed, 3 warnings in 8.31s` |

#### 3.2 Parallel Strategy Outcome

| Strategy | Result | Root Cause |
|----------|--------|------------|
| `TP=3,DP=2,EP=2,PP=1` | FAIL (fail-fast) | `num_attention_heads (16) must be a multiple of tensor_model_parallel_size (3)` |
| `TP=2,DP=3,EP=1,PP=1` | PASS | Used as runnable fallback for all integration tests |

#### 3.3 Core Comp Comparison (6 ranks)

| Config | forward_step | backward_step | optimizer_step | Overall |
|--------|--------------|---------------|----------------|---------|
| `seq2048, mbs8` | mean diff `16.90%` (6/6 FAIL) | mean diff `3.54%` (2/6 FAIL) | mean diff `9.93%` (4/6 FAIL) | FAIL |
| `seq2048, mbs4` | mean diff `46.69%` (6/6 FAIL) | mean diff `29.71%` (6/6 FAIL) | mean diff `8.23%` (4/6 FAIL) | FAIL |

Acceptance gate (`<=5%`) is not met.

#### 3.4 Forward/Backward Includes Comm in Realistic Trace (Confirmed)

Representative rank0 (`mbs8`):

- `forward_step`: `total_ms=156.70`, `comm_ms=35.72`, `comp_ms=120.98`
- `backward_step`: `total_ms=201.83`, `comm_ms=28.13`, `comp_ms=173.70`

This confirms realistic `total_ms` includes comm contributions; subtracting comm is required for comp comparison.

#### 3.5 Cross-Mode Coverage and Attribution Findings

1. **Op-level mismatch still exists**
   - realistic rank trace: `get_batch/forward_step/backward_step/dp_allreduce/optimizer_step`
   - scaling rank trace: adds standalone `loss_func`

2. **Sub-op composition mismatch (key blocker)**
   - rank0 `mbs8`:
     - realistic `forward_step`: `allreduce=4`, `all_to_all=48`, `allgather=12`, `reduce_scatter=12`
     - scaling `forward_step`: `allreduce=15`, `all_to_all=48`, `allgather=12`, `reduce_scatter=12`
   - scaling has extra TP allreduce trace sources:
     - `trace_src_func=allreduce` appears 12 times in forward and 12 times in backward (`func_name=linear_fwd` / `normlinear_bwd_allreduce`)

3. **Runtime path divergence (critical evidence)**
   - realistic logs: `sequence_parallel=True`
   - scaling logs: `sequence_parallel=False`
   - this changes TP linear communication path and directly leads to attribution divergence.

### 4) Failure -> Diagnosis -> Re-run Chain

1. **Primary target strategy unavailable**
   - Failure: TP=3 violates model head divisibility.
   - Action: switched to TP=2 fallback and re-ran all required tests.

2. **Comp gate still failing after event-mode and metadata-only comm path**
   - Diagnosis: mismatch is dominated by execution-path parity (sequence-parallel + TP allreduce attribution), not by sync policy alone.
   - Evidence: large forward gaps persist; `mbs=4` worsens significantly.

### 5) Final Conclusion (This Round)

- Current code changes correctly enforce:
  - scaling comm sub-ops are metadata-only (`duration=0.0`), no real collective;
  - realistic comp uses `total - comm` in compare script;
  - timestamp pairing and mode-aware compare are working.
- However, Qwen3 seq2048 still fails acceptance because scaling and realistic are not yet on identical TP/SP communication paths.
- The dominant blocker is path/attribution inconsistency (especially `sequence_parallel` mismatch and extra TP allreduce instrumentation in scaling), not merely sub-op sync overhead.
