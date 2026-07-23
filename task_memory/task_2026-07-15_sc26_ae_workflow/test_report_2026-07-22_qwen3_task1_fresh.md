# Test Report: Fresh Qwen3-A3B Task1 Workload Tracing

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-22 | Recorded independent validation of the producer-bound Fresh Qwen3-A3B Task1 capture |

**Date:** 2026-07-22  
**Result:** PASS  
**Scope:** Fake-level workload tracing only; no claim is made about real distributed multi-node accuracy.

## 1. Test Script Information

- Worker script: `task_memory/task_2026-07-15_sc26_ae_workflow/logs/qwen3_i72_recap_task1_worker.sh`
- Canonical output root: `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4`
- Canonical run: `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/qwen3_a30b/task1/runs/qwen3_a30b-20260722T112049Z`
- Marker: `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/qwen3_a30b/task1/capture_marker.json`
- Manifest: `/data/ycfeng/SC26-AE/output_gpu_20260722T1935_qwen3_i72_fix_r4/qwen3_a30b/task1/runs/qwen3_a30b-20260722T112049Z/artifact_manifest.json`
- RJob: `sc26-ae-qwen3-i72-fix-20260722e`
- Replica: `sc26-ae-qwen3-i72-fix-20260722e-cfcd2084`
- Worker result: `exit_code=0`, `TASK1_STATUS=verified`, `MANIFEST_STATUS=verified`, `MANIFEST_FILE_COUNT=243`
- Image: `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4`
- Megatron environment: `/opt/conda/envs/megatron_env/bin/python` (Python 3.9.x, CUDA 12.1 worker image)
- Temporary storage: `/data/ycfeng/tmp`; no new validation output was written to `/tmp`.

## 2. Validation Criteria

The capture is accepted only when all of the following hold:

1. Qwen3 fake topology is `fake_world_size=256`, `pp=8`, `tp=8`, `ep=4`, `dp=4`.
2. The representative PP×EP rank vector contains exactly 32 ranks:
   `0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248`.
3. Every selected rank has a profiler trace containing `forward_step`, `backward_step`, and `optimizer_step`.
4. Every selected rank has a non-empty scaling memory JSON.
5. Nsight Systems exports both `.nsys-rep` and `.sqlite`.
6. NCU is scoped to global rank 0, has no missing required kernel features, and records all 49 required kernels.
7. Marker and manifest are verified, and their recorded manifest digest agrees.
8. Fresh capture timing remains below the 7200-second gate.

## 3. Test Results and Evidence

| Check | Expected | Actual | Result |
|------|----------|--------|--------|
| Worker exit code | `0` | `0` | PASS |
| Manifest verification | `verified` | `verified` | PASS |
| Manifest file count | non-zero | `243` | PASS |
| Trace files | `32` | `32` | PASS |
| Memory JSON files | `32` | `32` | PASS |
| Rank vector | exact 32-rank PP×EP vector | exact match | PASS |
| Trace operation coverage | all 3 required operations per rank | present | PASS |
| NCU required kernels | `49` | `49` | PASS |
| NCU missing kernels | `0` | `0` | PASS |
| NCU feature scope | `global_rank_0` | `global_rank_0` | PASS |
| NCU feature rank IDs | `[0]` | `[0]` | PASS |
| Physical GPU count | `1` for Task1 | `1` | PASS |
| Maximum peak allocated memory | recorded and finite | `3165.50 MB` | PASS |
| Estimated full fake run | `<7200 s` | `4278.25383168 s` | PASS |
| Fresh capture elapsed | recorded | `706 s` | PASS |
| Source provenance bypass | `false` | `false` | PASS |

## 4. Artifact Sizes and Checksums

| Artifact | Size / digest |
|----------|---------------|
| `nsys/qwen3_a30b.nsys-rep` | `40,734,246` bytes |
| `nsys/qwen3_a30b.sqlite` | `144,748,544` bytes |
| `ncu/rank0.ncu-rep` | `1,947,793,310` bytes |
| `ncu/rank0_details.csv` | `84,709,178` bytes |
| `ncu/rank0_raw.csv` | `40,570,120` bytes |
| `ncu/kernel_metric_output.csv` | `371,055` bytes; SHA256 `138c15936f17e3b50d0a74830db7e6f2da25705eafd0913a23ea0c28291d1ab3` |
| `artifact_manifest.json` | SHA256 `a29941939c7b9b19b5d7cc2508ac5be94fafc926934f171d1d7a2602dd6fd123` |
| `capture_marker.json` | SHA256 `5b74bee94819c883b83c68678f7ce7f5af449ecd6a6e23a97f8407b19db8a52d` |

Source commits recorded by the manifest:

```text
megatron_lm:         df940b09c25537add927441594664c71ce01d473
megatron_sim_engine: 2b18afc9ad3b860de2f46b9fc4b364313a21647a
echo_slowdown:       1390b4416ded08bc1b9cd0620d329d81d4470bf9
```

## 5. Operational Warning

The worker source log contains Python multiprocessing cleanup tracebacks with
`OSError: [Errno 16] Device or resource busy: '.nfs...'`. They occurred while removing
temporary NFS files after the rank loop. They did not affect the worker exit code, rank loop,
memory writes, Nsight exports, NCU exports, manifest verification, or marker verification.
No CUDA, NCCL, segmentation-fault, NaN, or model-execution failure was observed.

## 6. Conclusion

Fresh Qwen3-A3B Task1 is independently verified at the fake level. The producer identity,
representative rank vector, rank-0 NCU provenance, memory vector, Nsight artifacts, manifest,
and marker are ready to feed the new exactly-two-GPU Fresh Task2. This report does not qualify
Task2, Fresh Task3, the functional bundle, or clean-clone replay.
