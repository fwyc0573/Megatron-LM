## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-02 | Add task notes: constraints, semantics, baseline, key paths, and run artifacts |

# Notes: MoE Engine Simulating Debug

## Hard Constraints (from request)
- 当前只做 engine 的 MoE simulating debug/验证
- MoE simulate 过程必须 all ranks 构建 timeline
- TP 在该流程中保持 barrier
- 语义区分：
  - engine: `exp` = expert parallel, `ep` = embedding parallel
  - `database_profile` 文件 token：`expX` expert parallel, `epX` embedding parallel
- 语义不一致采用 fail-fast（不做 silent fallback）

## Baseline Gap (before fix)
- a800_2ep: `783.80 / 939.36 ms` (`-16.56%`)
- a800_4ep: `597.19 / 981.14 ms` (`-39.13%`)
- h800_2ep: `351.30 / 431.77 ms` (`-18.64%`)
- h800_4ep: `253.80 / 357.74 ms` (`-29.05%`)

## Key Code Paths
- MoE detection / rank selection / init:
  - `megatron-sim-engine/src/core/simu_engine.py:4167`
  - `megatron-sim-engine/src/core/simu_engine.py:4261`
  - `megatron-sim-engine/src/core/simu_engine.py:4309`
- TP barrier + comm semantics:
  - `megatron-sim-engine/src/core/simu_engine.py:938`
  - `megatron-sim-engine/src/core/simu_engine.py:1304`
  - `megatron-sim-engine/src/core/simu_engine.py:1388`
- schedule allreduce group-size 显式估算:
  - `megatron-sim-engine/src/core/simu_engine.py:2959`
- filename semantic validation:
  - `megatron-sim-engine/src/core/simu_engine.py:4196`
  - `megatron-sim-engine/src/core/simu_engine.py:4245`
  - `megatron-sim-engine/src/core/simu_engine.py:4439`
- 文案去歧义（EXP/EP）:
  - `megatron-sim-engine/src/core/simu_engine.py:1688`
  - `megatron-sim-engine/src/core/simu_engine.py:1860`

## Run Artifacts
- Script replay logs:
  - `/tmp/mixtral_a800_2ep_after_fix.log`
  - `/tmp/mixtral_a800_4ep_after_fix.log`
  - `/tmp/mixtral_h800_2ep_after_fix.log`
  - `/tmp/mixtral_h800_4ep_after_fix.log`
- Evidence pattern:
  - `优化策略: MOE 模型，选择了 8 个ranks: [0, 1, 2, 3, 4, 5, 6, 7]`
- Detailed report:
  - `task_memory/task_2026-03-02_rank_skipping_analysis/test_report_2026-03-02_moe_engine_simulate_debug.md`
