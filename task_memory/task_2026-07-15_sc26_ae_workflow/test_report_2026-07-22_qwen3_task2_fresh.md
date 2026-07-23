# Test Report: Fresh Qwen3-A3B Task2 Slowdown Predictor

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-22 | Recorded the new producer-bound exactly-two-H800 Fresh Qwen3-A3B Task2 run |

**Date:** 2026-07-22  
**Result:** PASS for the requested fake-level AE workflow  
**Scope:** Fresh slowdown dataset/predictor generation on exactly two H800 GPUs; this is not a
claim of multi-node distributed accuracy or an external release qualification.

## 1. Test Script Information

- Worker script: `task_memory/task_2026-07-15_sc26_ae_workflow/logs/qwen3_fresh_df940_task2_worker.sh`
- Predict-only command log: `task_memory/task_2026-07-15_sc26_ae_workflow/logs/qwen3_fresh_df940_task2_predict_20260722.log`
- RJob command log: `task_memory/task_2026-07-15_sc26_ae_workflow/logs/qwen3_fresh_df940_task2_rlaunch_20260722.log`
- Canonical output root: `/data/ycfeng/SC26-AE/output_gpu_20260722T2224_qwen3_df940_task2`
- Predictor run: `/data/ycfeng/SC26-AE/output_gpu_20260722T2224_qwen3_df940_task2/_shared/task2/runs/task2-20260722T142810Z-192-11368`
- Model marker: `/data/ycfeng/SC26-AE/output_gpu_20260722T2224_qwen3_df940_task2/qwen3_a30b/task2/predictor_marker.json`
- Shared marker: `/data/ycfeng/SC26-AE/output_gpu_20260722T2224_qwen3_df940_task2/_shared/task2/predictor_marker.json`
- Manifest: `/data/ycfeng/SC26-AE/output_gpu_20260722T2224_qwen3_df940_task2/_shared/task2/runs/task2-20260722T142810Z-192-11368/artifact_manifest.json`
- RJob: `sc26-ae-qwen3-df940-t2-20260722`
- Replica: `sc26-ae-qwen3-df940-t2-20260722-cfcd2084`
- Image: `hub.i.basemind.com/mg-echo/megatron-h800@sha256:b7072775e8a4dd7bd21ef5efe73b398875923968274ef001fa807985d9e101e4`
- Echo environment: `/opt/conda/envs/echo_slowdown/bin/python` (Python 3.10.20)
- Temporary storage: worker `TMPDIR=/data/ycfeng/tmp/qwen3-df940-task2`; no new output was written to `/tmp`.

## 2. Validation Criteria

1. The 2-GPU predict-only gate exits `0` using the handbook `codesign`/H800 recipe.
2. The worker sees two H800 devices and runs the pinned real-mode command with
   `CUDA_VISIBLE_DEVICES=0,1`.
3. Outer, simulator, and Echo commits match the Fresh Task1 producer identity.
4. Echo produces a non-empty dataset, XGBoost predictor, scaler, metrics, and checksum-verified
   manifest/markers.
5. The predictor reload check has zero maximum absolute prediction delta.
6. No automatic fallback or synthetic execution evidence is used.

## 3. Test Results and Numeric Evidence

| Check | Expected | Actual | Result |
|------|----------|--------|--------|
| 2-GPU predict-only exit | `0` | `0` | PASS |
| Worker H800 count | `2` | `2` (`nvidia-smi -L`) | PASS |
| Provenance `cuda_visible_devices` | `0,1` | `0,1` | PASS |
| Worker exit | `0` | `0` | PASS |
| Manifest status | `verified` | `verified` | PASS |
| Manifest file count | non-zero | `18` | PASS |
| Dataset rows | non-zero | `727` | PASS |
| Scaler features | `8` | `8` | PASS |
| Average validation MSE | finite | `0.04124828706619175` | PASS |
| Test MSE | finite | `0.061428837844613504` | PASS |
| Fold validation MSEs | five values | `0.025510066943321086, 0.05853339433492776, 0.06497570805221323, 0.019521841326430207, 0.037700424674066445` | PASS |
| Reload max absolute prediction delta | `0` | `0.0` | PASS |
| Task2 `run_all` elapsed | recorded | `1041.532698287 s` | PASS |
| Automatic fallback | `false` | `false` | PASS |
| Outer source provenance | checked/not bypassed | checked, `bypassed=false` | PASS |

Prediction sample:

```text
original_execution_time       = 1.0
predicted_execution_time      = 1.2976043224334717
predicted_slowdown_factor     = 0.5952086448669434
```

## 4. Producer Identity and Checksums

```text
megatron_lm:         df940b09c25537add927441594664c71ce01d473
megatron_sim_engine: 2b18afc9ad3b860de2f46b9fc4b364313a21647a
echo_slowdown:       1390b4416ded08bc1b9cd0620d329d81d4470bf9
predictor_run_id:    task2-20260722T142810Z-192-11368
artifact_manifest:   d344fbfc0f4e56286efe9dd5ee6ac3f125ed3ad34fe8e4599bc9a71f67dda76e
metrics.json:        6cda47a5f011ec763443f8a31ccaa7fd27ac7398016a62b2b1bf6832003a8761
dataset.csv:         3f6fd7be758f016eb32cb864611348edee0d6e08d9289bb839c57c87879d7eb3
xgb_model.json:      6f9474775b1c60a0489abf1f314af1f9366a87d515bb629f9430a12dc605e06e
standard_scaler.json: 71fdebff4a797f860f2f9f4088c9f0bdf6ca83ae01df303ca6ddb573df9fa16b
```

The real-mode marker intentionally records
`execution_evidence=runtime_measurement_requires_external_two_gpu_qualification`; this is the
repository's explicit distinction between a successful two-GPU runtime and an independent
qualification attestation. The run itself did use two H800 GPUs and passed the requested AE
workflow gate.

## 5. Conclusion

Fresh Qwen3-A3B Task2 is complete for the fake-level AE workflow. The dataset, predictor,
scaler, metrics, timing, provenance, checksums, manifest, and markers are ready for a Fresh
CPU-only Task3 chain. Fresh Task3, functional prebaked packaging, and clean-clone replay remain
open.
