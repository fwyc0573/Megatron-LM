threshold_pct=5.00
distributed_dir=task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/dist_run/realistic_trace/pp2_tp1_exp4_expn32_dp4_nl32_hs2048_sl256
scaling_dir=task_memory/task_2026-03-05_deepseek_v3_comp_gap_analysis/rerun_2026-03-05_metadata_comm_fix_stable/scale_baseline/profiler_log/pp2_tp1_ep4_expn32_dp4_nl32_hs2048_sl256
ranks=0,1,2,3,4,5,6,7
ops=forward_step,backward_step,optimizer_step
distributed_subtract_comm=True
distributed_comm_scale=1.0000
distributed_comm_scale_map={}
scaling_subtract_comm=False
scaling_comm_scale=1.0000
scaling_comm_scale_map={}
align_by_state=True
trim_ratio=0.2000
pair_timestamp=None
| rank | op | mg_state | dist_total_ms | dist_comm_ms | dist_eff_comm_ms | dist_comp_ms | dist_subops | scale_total_ms | scale_comm_ms | scale_eff_comm_ms | scale_comp_ms | scale_subops | diff_pct | status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | forward_step | warmup | 69.4500 | 16.4500 | 16.4500 | 53.0000 | 33.00 | 48.3600 | 0.0000 | 0.0000 | 48.3600 | 33.00 | 8.75 | FAIL |
| 0 | backward_step | cooldown | 53.6000 | 7.1500 | 7.1500 | 46.4500 | 22.00 | 42.3300 | 0.0000 | 0.0000 | 42.3300 | 22.00 | 8.87 | FAIL |
| 0 | optimizer_step | finalize | 24.7600 | 0.0000 | 0.0000 | 24.7600 | 0.00 | 24.4600 | 0.0000 | 0.0000 | 24.4600 | 0.00 | 1.21 | PASS |
| 1 | forward_step | warmup | 68.5000 | 13.1800 | 13.1800 | 55.3200 | 33.00 | 51.0200 | 0.0000 | 0.0000 | 51.0200 | 33.00 | 7.77 | FAIL |
| 1 | backward_step | cooldown | 53.3300 | 8.8600 | 8.8600 | 44.4700 | 22.00 | 47.5400 | 0.0000 | 0.0000 | 47.5400 | 22.00 | 6.90 | FAIL |
| 1 | optimizer_step | finalize | 23.1100 | 0.0000 | 0.0000 | 23.1100 | 0.00 | 25.5400 | 0.0000 | 0.0000 | 25.5400 | 0.00 | 10.51 | FAIL |
| 2 | forward_step | warmup | 69.5100 | 15.6100 | 15.6100 | 53.9000 | 33.00 | 54.5900 | 0.0000 | 0.0000 | 54.5900 | 33.00 | 1.28 | PASS |
| 2 | backward_step | cooldown | 54.5200 | 4.2300 | 4.2300 | 50.2900 | 22.00 | 44.2900 | 0.0000 | 0.0000 | 44.2900 | 22.00 | 11.93 | FAIL |
| 2 | optimizer_step | finalize | 24.4600 | 0.0000 | 0.0000 | 24.4600 | 0.00 | 24.7200 | 0.0000 | 0.0000 | 24.7200 | 0.00 | 1.06 | PASS |
| 3 | forward_step | warmup | 68.1200 | 7.5200 | 7.5200 | 60.6000 | 33.00 | 48.7100 | 0.0000 | 0.0000 | 48.7100 | 33.00 | 19.62 | FAIL |
| 3 | backward_step | cooldown | 53.6400 | 8.1000 | 8.1000 | 45.5400 | 22.00 | 52.3500 | 0.0000 | 0.0000 | 52.3500 | 22.00 | 14.95 | FAIL |
| 3 | optimizer_step | finalize | 23.6800 | 0.0000 | 0.0000 | 23.6800 | 0.00 | 24.5300 | 0.0000 | 0.0000 | 24.5300 | 0.00 | 3.59 | PASS |
| 4 | forward_step | steady | 91.1612 | 19.3663 | 19.3663 | 71.7950 | 48.00 | 72.3200 | 0.0000 | 0.0000 | 72.3200 | 48.00 | 0.73 | PASS |
| 4 | backward_step | steady | 71.0787 | 11.0687 | 11.0687 | 60.0100 | 32.00 | 66.8600 | 0.0000 | 0.0000 | 66.8600 | 32.00 | 11.41 | FAIL |
| 4 | optimizer_step | finalize | 25.6400 | 0.0000 | 0.0000 | 25.6400 | 0.00 | 26.4600 | 0.0000 | 0.0000 | 26.4600 | 0.00 | 3.20 | PASS |
| 5 | forward_step | steady | 91.3650 | 24.1575 | 24.1575 | 67.2075 | 48.00 | 74.1000 | 0.0000 | 0.0000 | 74.1000 | 48.00 | 10.26 | FAIL |
| 5 | backward_step | steady | 71.0250 | 12.3513 | 12.3513 | 58.6737 | 32.00 | 59.4300 | 0.0000 | 0.0000 | 59.4300 | 32.00 | 1.29 | PASS |
| 5 | optimizer_step | finalize | 23.6100 | 0.0000 | 0.0000 | 23.6100 | 0.00 | 26.3400 | 0.0000 | 0.0000 | 26.3400 | 0.00 | 11.56 | FAIL |
| 6 | forward_step | steady | 91.7588 | 18.3325 | 18.3325 | 73.4262 | 48.00 | 80.0400 | 0.0000 | 0.0000 | 80.0400 | 48.00 | 9.01 | FAIL |
| 6 | backward_step | steady | 70.6963 | 18.2000 | 18.2000 | 52.4963 | 32.00 | 60.0400 | 0.0000 | 0.0000 | 60.0400 | 32.00 | 14.37 | FAIL |
| 6 | optimizer_step | finalize | 23.4500 | 0.0000 | 0.0000 | 23.4500 | 0.00 | 26.2000 | 0.0000 | 0.0000 | 26.2000 | 0.00 | 11.73 | FAIL |
| 7 | forward_step | steady | 91.6287 | 21.6438 | 21.6438 | 69.9850 | 48.00 | 80.1500 | 0.0000 | 0.0000 | 80.1500 | 48.00 | 14.52 | FAIL |
| 7 | backward_step | steady | 70.4150 | 13.2387 | 13.2387 | 57.1763 | 32.00 | 66.5000 | 0.0000 | 0.0000 | 66.5000 | 32.00 | 16.31 | FAIL |
| 7 | optimizer_step | finalize | 23.4700 | 0.0000 | 0.0000 | 23.4700 | 0.00 | 26.8500 | 0.0000 | 0.0000 | 26.8500 | 0.00 | 14.40 | FAIL |

trimmed_mean_aux_summary(trim_ratio=0.2000, non-gating):
| rank | op | mg_state | dist_total_ms | dist_comm_ms | dist_eff_comm_ms | dist_comp_ms | dist_subops | scale_total_ms | scale_comm_ms | scale_eff_comm_ms | scale_comp_ms | scale_subops | diff_pct | status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | forward_step | warmup | 69.4500 | 16.4500 | 16.4500 | 53.0000 | 33.00 | 48.3600 | 0.0000 | 0.0000 | 48.3600 | 33.00 | 8.75 | FAIL |
| 0 | backward_step | cooldown | 53.6000 | 7.1500 | 7.1500 | 46.4500 | 22.00 | 42.3300 | 0.0000 | 0.0000 | 42.3300 | 22.00 | 8.87 | FAIL |
| 0 | optimizer_step | finalize | 24.7600 | 0.0000 | 0.0000 | 24.7600 | 0.00 | 24.4600 | 0.0000 | 0.0000 | 24.4600 | 0.00 | 1.21 | PASS |
| 1 | forward_step | warmup | 68.5000 | 13.1800 | 13.1800 | 55.3200 | 33.00 | 51.0200 | 0.0000 | 0.0000 | 51.0200 | 33.00 | 7.77 | FAIL |
| 1 | backward_step | cooldown | 53.3300 | 8.8600 | 8.8600 | 44.4700 | 22.00 | 47.5400 | 0.0000 | 0.0000 | 47.5400 | 22.00 | 6.90 | FAIL |
| 1 | optimizer_step | finalize | 23.1100 | 0.0000 | 0.0000 | 23.1100 | 0.00 | 25.5400 | 0.0000 | 0.0000 | 25.5400 | 0.00 | 10.51 | FAIL |
| 2 | forward_step | warmup | 69.5100 | 15.6100 | 15.6100 | 53.9000 | 33.00 | 54.5900 | 0.0000 | 0.0000 | 54.5900 | 33.00 | 1.28 | PASS |
| 2 | backward_step | cooldown | 54.5200 | 4.2300 | 4.2300 | 50.2900 | 22.00 | 44.2900 | 0.0000 | 0.0000 | 44.2900 | 22.00 | 11.93 | FAIL |
| 2 | optimizer_step | finalize | 24.4600 | 0.0000 | 0.0000 | 24.4600 | 0.00 | 24.7200 | 0.0000 | 0.0000 | 24.7200 | 0.00 | 1.06 | PASS |
| 3 | forward_step | warmup | 68.1200 | 7.5200 | 7.5200 | 60.6000 | 33.00 | 48.7100 | 0.0000 | 0.0000 | 48.7100 | 33.00 | 19.62 | FAIL |
| 3 | backward_step | cooldown | 53.6400 | 8.1000 | 8.1000 | 45.5400 | 22.00 | 52.3500 | 0.0000 | 0.0000 | 52.3500 | 22.00 | 14.95 | FAIL |
| 3 | optimizer_step | finalize | 23.6800 | 0.0000 | 0.0000 | 23.6800 | 0.00 | 24.5300 | 0.0000 | 0.0000 | 24.5300 | 0.00 | 3.59 | PASS |
| 4 | forward_step | steady | 91.0367 | 18.4217 | 18.4217 | 71.1150 | 48.00 | 72.3200 | 0.0000 | 0.0000 | 72.3200 | 48.00 | 1.69 | PASS |
| 4 | backward_step | steady | 70.2100 | 9.4733 | 9.4733 | 59.2800 | 32.00 | 66.8600 | 0.0000 | 0.0000 | 66.8600 | 32.00 | 12.79 | FAIL |
| 4 | optimizer_step | finalize | 25.6400 | 0.0000 | 0.0000 | 25.6400 | 0.00 | 26.4600 | 0.0000 | 0.0000 | 26.4600 | 0.00 | 3.20 | PASS |
| 5 | forward_step | steady | 91.1900 | 24.1367 | 24.1367 | 67.2267 | 48.00 | 74.1000 | 0.0000 | 0.0000 | 74.1000 | 48.00 | 10.22 | FAIL |
| 5 | backward_step | steady | 70.0667 | 11.9233 | 11.9233 | 56.6367 | 32.00 | 59.4300 | 0.0000 | 0.0000 | 59.4300 | 32.00 | 4.93 | PASS |
| 5 | optimizer_step | finalize | 23.6100 | 0.0000 | 0.0000 | 23.6100 | 0.00 | 26.3400 | 0.0000 | 0.0000 | 26.3400 | 0.00 | 11.56 | FAIL |
| 6 | forward_step | steady | 91.3767 | 18.6217 | 18.6217 | 73.6267 | 48.00 | 80.0400 | 0.0000 | 0.0000 | 80.0400 | 48.00 | 8.71 | FAIL |
| 6 | backward_step | steady | 69.8800 | 17.4983 | 17.4983 | 50.8083 | 32.00 | 60.0400 | 0.0000 | 0.0000 | 60.0400 | 32.00 | 18.17 | FAIL |
| 6 | optimizer_step | finalize | 23.4500 | 0.0000 | 0.0000 | 23.4500 | 0.00 | 26.2000 | 0.0000 | 0.0000 | 26.2000 | 0.00 | 11.73 | FAIL |
| 7 | forward_step | steady | 91.2150 | 21.3067 | 21.3067 | 69.1733 | 48.00 | 80.1500 | 0.0000 | 0.0000 | 80.1500 | 48.00 | 15.87 | FAIL |
| 7 | backward_step | steady | 69.5533 | 12.1967 | 12.1967 | 56.9950 | 32.00 | 66.5000 | 0.0000 | 0.0000 | 66.5000 | 32.00 | 16.68 | FAIL |
| 7 | optimizer_step | finalize | 23.4700 | 0.0000 | 0.0000 | 23.4700 | 0.00 | 26.8500 | 0.0000 | 0.0000 | 26.8500 | 0.00 | 14.40 | FAIL |

op_rank_median_aux_summary(non-gating, recommended_for_paper):
| op | rank_samples | rank_median_diff_pct | rank_p75_diff_pct | status |
|---|---:|---:|---:|---|
| backward_step | 8 | 11.67 | 14.37 | FAIL |
| forward_step | 8 | 8.88 | 10.26 | FAIL |
| optimizer_step | 8 | 7.05 | 11.56 | FAIL |
