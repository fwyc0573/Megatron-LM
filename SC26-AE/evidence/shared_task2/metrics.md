# Echo Task2 Metrics

- Schema: `sc26-ae-echo-metrics-v1`
- Predictor run: `task2-20260722T142810Z-192-11368`
- Evidence class: `echo_runtime_measurement_pending_external_qualification`

| Metric | Value |
| --- | ---: |
| Run elapsed seconds | 1041.53269829 |
| Dataset rows | 727 |
| Validation MSE fold 1 | 0.0255100669433 |
| Validation MSE fold 2 | 0.0585333943349 |
| Validation MSE fold 3 | 0.0649757080522 |
| Validation MSE fold 4 | 0.0195218413264 |
| Validation MSE fold 5 | 0.0377004246741 |
| Average validation MSE | 0.0412482870662 |
| Test MSE | 0.0614288378446 |
| Model reload max abs delta | 0 |
| Scaler feature count | 8 |
| Scaler mean count | 8 |
| Scaler scale count | 8 |
| Scaler nonzero scale count | 8 |

## Deterministic prediction sample

| Field | Value |
| --- | ---: |
| original_execution_time | 1 |
| predicted_execution_time | 1.29760432243 |
| predicted_execution_time_clipped | 1.29760432243 |
| predicted_slowdown_factor | 0.595208644867 |
| predicted_slowdown_factor_clipped | 0.595208644867 |

Synthetic reports are local contract evidence only and do not qualify the required two-GPU Echo run.
