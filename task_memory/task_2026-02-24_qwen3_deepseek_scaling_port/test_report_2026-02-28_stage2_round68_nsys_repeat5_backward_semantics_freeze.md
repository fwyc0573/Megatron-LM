## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-02-28 | Added Round6-8 seq8192 NSYS repeat-x5 validation with fixed rank7-cap pairing, compared trace subtract/no-subtract and NSYS compute-only views, and assessed backward gate semantics stability |

## Test Report: Stage-2 Round6-8 Backward Semantics Freeze (NSYS repeat x5)

**Date**: 2026-02-28  
**Environment**: `conda activate /opt/anaconda/envs/myenv_yc` (Python 3.9.18), CUDA 12.1, Nsight Systems 2023.1.2.43  
**Execution workspace**: `/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM_round68_regime`  
**Code state**: detached `3a50265d` baseline + experiment-only script passthrough update in `examples/pretrain_deepseek_v3_moe.sh` (added `TRACE_KERNEL_GROUND_TRUTH` argument forwarding only; no model/training code edit)

---

### 1) Goal

在保持 Round6-8 基线代码逻辑不变的前提下，完成 backward 评估语义复核：

1. 继续执行固定协议（rank7 end-of-run cap + repeat x5）；
2. 并行输出三种视图：
   - trace `distributed_subtract_comm`（诊断）；
   - trace `no-subtract`（诊断对照）；
   - NSYS compute-only（候选主口径）；
3. 评估 backward 指标是否具备可作为正式 gate 的稳定性与可解释性。

---

### 2) Test Script Information

- 关键脚本/工具：
  - `examples/pretrain_deepseek_v3_moe.sh`
  - `tests/performance/compare_qwen_trace_comp.py`
  - `tests/performance/analyze_nsys_cmd_kernel_breakdown.py`
  - `tests/performance/compare_qwen_nsys_compute_only.py`
- 运行参数（固定）：
  - `MODEL_PROFILE=smoke`
  - `SEQ_LEN=8192`
  - `TRACE_START=4`
  - `TRAIN_ITERS=6`
  - `TRACE_SUBOP_SYNC_MODE=global`
  - `TRACE_CMD_SYNC_MODE=global`
  - `TRACE_OPTIMIZER_MICROPHASES=1`
  - scaling rank order: `0,4,1,5,2,6,3,7`
- NSYS 采集开关：
  - `TRACE_KERNEL_GROUND_TRUTH=1`
  - `TRACE_KERNEL_GROUND_TRUTH_PREFIX=cmd_trace`

---

### 3) Repro Commands (pattern)

```bash
# 1) Distributed NSYS capture (run k)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nsys profile -w true -t cuda,nvtx,osrt --sample=none \
  --force-overwrite=true --trace-fork-before-exec=true \
  -o task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run<k>_dist \
  bash -lc "source /opt/anaconda/etc/profile.d/conda.sh && conda activate myenv_yc && \
    MODE=distributed MODEL_PROFILE=smoke TP=1 PP=2 EP=2 GPUS_PER_NODE=8 \
    MICRO_BATCH_SIZE=1 SEQ_LEN=8192 TRACE_START=4 TRAIN_ITERS=6 \
    TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
    TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PREFIX=cmd_trace DO_TRACE=True \
    MASTER_PORT=<dist_port> bash examples/pretrain_deepseek_v3_moe.sh"

# 2) Scaling NSYS capture (run k)
CUDA_VISIBLE_DEVICES=0 \
nsys profile -w true -t cuda,nvtx,osrt --sample=none \
  --force-overwrite=true --trace-fork-before-exec=true \
  -o task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run<k>_scaling \
  bash -lc "source /opt/anaconda/etc/profile.d/conda.sh && conda activate myenv_yc && \
    MODE=scaling MODEL_PROFILE=smoke TP=1 PP=2 EP=2 FAKE_WORLD_SIZE=8 FAKE_PP=2 FAKE_TP=1 FAKE_EXP=2 \
    MICRO_BATCH_SIZE=1 SEQ_LEN=8192 TRACE_START=4 TRAIN_ITERS=6 \
    TRACE_SUBOP_SYNC_MODE=global TRACE_CMD_SYNC_MODE=global TRACE_OPTIMIZER_MICROPHASES=1 \
    TRACE_KERNEL_GROUND_TRUTH=1 TRACE_KERNEL_GROUND_TRUTH_PREFIX=cmd_trace DO_TRACE=True \
    SCALING_MIN_WARMUP_ITERS=0 SCALING_PROFILE_ITERS=3 \
    SCALING_FAKE_RANK_ORDER=0,4,1,5,2,6,3,7 SCALE_GPU=0 MASTER_PORT=<scale_port> \
    bash examples/pretrain_deepseek_v3_moe.sh"

# 3) Trace compare (subtract / no-subtract)
python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl8192 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl8192 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 --pair-timestamp <rank7_ts> --distributed-subtract-comm

python tests/performance/compare_qwen_trace_comp.py \
  --distributed-dir realistic_trace/pp2_tp1_exp2_expn16_dp4_nl8_hs1024_sl8192 \
  --scaling-dir profiler_log/pp2_tp1_ep2_expn16_dp4_nl8_hs1024_sl8192 \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 --pair-timestamp <rank7_ts> --no-distributed-subtract-comm

# 4) NSYS export + breakdown + compare
nsys export --type sqlite --force-overwrite=true \
  --output task_memory/.../deepseek_round68_sl8192_run<k>_dist_sqlite \
  task_memory/.../deepseek_round68_sl8192_run<k>_dist.nsys-rep

nsys export --type sqlite --force-overwrite=true \
  --output task_memory/.../deepseek_round68_sl8192_run<k>_scaling_sqlite \
  task_memory/.../deepseek_round68_sl8192_run<k>_scaling.nsys-rep

python tests/performance/analyze_nsys_cmd_kernel_breakdown.py \
  --sqlite task_memory/.../deepseek_round68_sl8192_run<k>_dist_sqlite \
  --label-prefix cmd_trace --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --json-path task_memory/.../deepseek_round68_sl8192_run<k>_dist_kernel_breakdown.json \
  --report-path task_memory/.../deepseek_round68_sl8192_run<k>_dist_kernel_breakdown.md

python tests/performance/compare_qwen_nsys_compute_only.py \
  --distributed-json task_memory/.../deepseek_round68_sl8192_run<k>_dist_kernel_breakdown.json \
  --scaling-json task_memory/.../deepseek_round68_sl8192_run<k>_scaling_kernel_breakdown.json \
  --ranks 0,1,2,3,4,5,6,7 \
  --ops forward_step,backward_step,optimizer_step \
  --threshold-pct 5 \
  --compute-metric primary_stream_union \
  --kernel-scope shared --shared-kernel-source primary_stream \
  --dist-reducer trimmed_mean --scale-reducer median --trim-ratio 0.2
```

---

### 4) Validation Criteria

1. repeat x5 的每轮 distributed/scaling NSYS 采集、sqlite 导出、kernel breakdown 与 compare 全部可复现；
2. 同批次必须同时给出 subtract / no-subtract / NSYS compute-only 三视图；
3. backward 口径判定必须同时看 median 与 spread（range/IQR）。

---

### 5) Results (repeat x5, median-of-runs)

#### 5.1 Trace subtract view

- `forward=2.84%`
- `backward=35.21%`
- `optimizer=11.20%`
- `mean_3ops=16.92%`
- backward spread: `range=12.59%`, `IQR=7.08%`

#### 5.2 Trace no-subtract view

- `forward=6.79%`
- `backward=10.40%`
- `optimizer=11.20%`
- `mean_3ops=9.37%`
- backward spread: `range=1.09%`, `IQR=0.63%`

#### 5.3 NSYS compute-only view (`primary_stream_union + shared(primary_stream)`)

- `forward=0.33%`
- `backward=38.30%`
- `optimizer=0.81%`
- `mean_3ops=13.10%`
- backward spread: `range=3.26%`, `IQR=0.43%`

---

### 6) Key Evidence and Interpretation

1. backward 结果在三视图间出现结构性分歧：
   - no-subtract: `~10%`
   - subtract: `~35%`
   - NSYS compute-only: `~38%`
2. subtract 与 no-subtract 的巨大差值再次说明 global-subop subtraction 口径存在显著系统偏置风险。
3. NSYS compute-only 在当前 workload 下虽然**稳定**（低 IQR），但 backward 绝对误差长期停留在 `~38%`，与 no-subtract 诊断口径明显不一致；现阶段不宜直接作为唯一 backward gate。
4. 在不引入 alpha 标定（stage-aware 不可扩展）的前提下，backward gate 语义仍未冻结，需要继续做“无标定”的测量语义收敛实验。
5. 对 run5 做 NSYS metric-mode sweep（`overlap_sum/union/primary_stream_union` × `kernel-scope all/shared`）后，backward 仍处在 `36.91%~38.30%` 区间，说明当前高残差并非单一 metric mode 选择导致。

---

### 7) Artifacts

- NSYS采集与解析：
  - `task_memory/task_2026-02-24_qwen3_deepseek_scaling_port/logs/nsys_round68_semantics/deepseek_round68_sl8192_run{1..5}_{dist,scaling}.nsys-rep`
  - `.../deepseek_round68_sl8192_run{1..5}_{dist,scaling}_sqlite`
  - `.../deepseek_round68_sl8192_run{1..5}_{dist,scaling}_kernel_breakdown.{json,md}`
- 运行日志：
  - `.../logs/deepseek_v3_stage2_dist_round68_nsys_run{1..5}.log`
  - `.../logs/deepseek_v3_stage2_scaling_round68_nsys_run{1..5}.log`
- trace compare：
  - `.../logs/deepseek_v3_stage2_compare_round68_nsys_run{1..5}_subtract.log`
  - `.../logs/deepseek_v3_stage2_compare_round68_nsys_run{1..5}_nosubtract.log`
- NSYS compare：
  - `.../logs/deepseek_v3_stage2_compare_round68_nsys_run{1..5}_compute_only.log`
- 聚合汇总：
  - `.../logs/deepseek_v3_stage2_round68_nsys_semantics_summary.json`
  - `.../logs/deepseek_v3_stage2_round68_nsys_semantics_summary.md`
  - `.../logs/deepseek_v3_stage2_round68_nsys_metric_mode_run5.tsv`
