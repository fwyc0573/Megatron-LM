==================================================================================================================================
Qwen3 30B-A3B MoE: Scaling Mode vs Distributed Mode Compute (Comp) Comparison
Model: num_layers=48, hidden_size=2048, num_experts=128, seq_len=2048
Hardware: H800 16-GPU,  World Size = 16
Threshold: 5.0% relative error
==================================================================================================================================


Analyzing Config A (PP=2, EP=8) ...

##################################################################################################################################
# Config A (PP=2, EP=8)  (pp2_tp1_exp8_expn128_dp8_nl48_hs2048_sl2048)
# PP=2, TP=1, EP=8, DP=8, num_experts=128, num_layers=48, hidden_size=2048, seq_len=2048
##################################################################################################################################

==================================================================================================================================
PART 1: Per-Rank Micro-Batch Breakdown (Distributed Mode)
==================================================================================================================================

--- Rank 0 (PP stage 0/1) ---
  forward_step: 4 micro-batch(es), avg_comp=85.36 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup     134.31      47.46      86.85     72
           1     steady     130.52      46.57      83.95     72
           2     steady     133.63      48.29      85.34     72
           3     steady     134.65      49.36      85.29     72
  backward_step: 4 micro-batch(es), avg_comp=69.79 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.56      66.81      69.75     48
           1     steady     137.35      66.99      70.36     48
           2     steady     136.49      66.91      69.58     48
           3   cooldown     136.56      67.08      69.48     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.77 ms

--- Rank 1 (PP stage 0/1) ---
  forward_step: 4 micro-batch(es), avg_comp=80.43 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup     134.36      52.00      82.36     72
           1     steady     130.77      51.90      78.87     72
           2     steady     133.60      53.15      80.45     72
           3     steady     135.35      55.30      80.05     72
  backward_step: 4 micro-batch(es), avg_comp=68.44 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     135.80      67.28      68.52     48
           1     steady     135.87      67.47      68.40     48
           2     steady     136.93      68.03      68.90     48
           3   cooldown     136.41      68.48      67.93     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.81 ms

--- Rank 2 (PP stage 0/1) ---
  forward_step: 4 micro-batch(es), avg_comp=80.82 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup     134.11      52.51      81.60     72
           1     steady     130.91      51.48      79.43     72
           2     steady     134.47      53.17      81.30     72
           3     steady     135.48      54.54      80.94     72
  backward_step: 4 micro-batch(es), avg_comp=66.29 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     135.83      70.09      65.74     48
           1     steady     136.48      70.28      66.20     48
           2     steady     137.15      70.31      66.84     48
           3   cooldown     136.27      69.91      66.36     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.79 ms

--- Rank 3 (PP stage 0/1) ---
  forward_step: 4 micro-batch(es), avg_comp=81.72 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup     134.14      50.83      83.31     72
           1     steady     130.73      50.46      80.27     72
           2     steady     133.46      51.30      82.16     72
           3     steady     135.47      54.33      81.14     72
  backward_step: 4 micro-batch(es), avg_comp=67.34 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     135.99      68.24      67.75     48
           1     steady     135.46      67.91      67.55     48
           2     steady     135.86      68.41      67.45     48
           3   cooldown     136.36      69.77      66.59     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.66 ms

--- Rank 4 (PP stage 0/1) ---
  forward_step: 4 micro-batch(es), avg_comp=81.69 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup     134.06      50.98      83.08     72
           1     steady     130.45      49.93      80.52     72
           2     steady     133.64      52.40      81.24     72
           3     steady     134.77      52.84      81.93     72
  backward_step: 4 micro-batch(es), avg_comp=66.39 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     135.50      69.34      66.16     48
           1     steady     136.59      69.95      66.64     48
           2     steady     136.95      69.94      67.01     48
           3   cooldown     136.59      70.85      65.74     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.65 ms

--- Rank 5 (PP stage 0/1) ---
  forward_step: 4 micro-batch(es), avg_comp=84.53 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup     134.11      47.89      86.22     72
           1     steady     130.33      47.60      82.73     72
           2     steady     134.15      49.27      84.88     72
           3     steady     134.98      50.68      84.30     72
  backward_step: 4 micro-batch(es), avg_comp=72.14 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     134.78      63.08      71.70     48
           1     steady     135.89      63.23      72.66     48
           2     steady     135.92      63.90      72.02     48
           3   cooldown     138.03      65.83      72.20     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.82 ms

--- Rank 6 (PP stage 0/1) ---
  forward_step: 4 micro-batch(es), avg_comp=85.86 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup     133.96      46.67      87.29     72
           1     steady     130.46      45.81      84.65     72
           2     steady     133.84      48.19      85.65     72
           3     steady     135.30      49.44      85.86     72
  backward_step: 4 micro-batch(es), avg_comp=71.28 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.13      64.96      71.17     48
           1     steady     136.47      65.19      71.28     48
           2     steady     137.18      65.37      71.81     48
           3   cooldown     136.35      65.47      70.88     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.81 ms

--- Rank 7 (PP stage 0/1) ---
  forward_step: 4 micro-batch(es), avg_comp=86.19 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup     133.79      46.79      87.00     72
           1     steady     130.34      45.73      84.61     72
           2     steady     133.54      46.50      87.04     72
           3     steady     135.07      48.94      86.13     72
  backward_step: 4 micro-batch(es), avg_comp=71.81 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     135.21      63.46      71.75     48
           1     steady     135.82      63.56      72.26     48
           2     steady     135.96      64.00      71.96     48
           3   cooldown     134.74      63.49      71.25     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.69 ms

--- Rank 8 (PP stage 1/1) ---
  forward_step: 4 micro-batch(es), avg_comp=89.98 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     141.68      51.56      90.12     72
           1     steady     142.49      52.68      89.81     72
           2     steady     140.19      51.03      89.16     72
           3     steady     141.36      50.55      90.81     72
  backward_step: 4 micro-batch(es), avg_comp=64.78 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.91      72.51      64.40     48
           1     steady     136.98      72.34      64.64     48
           2     steady     138.00      73.14      64.86     48
           3     steady     136.40      71.18      65.22     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.58 ms

--- Rank 9 (PP stage 1/1) ---
  forward_step: 4 micro-batch(es), avg_comp=92.56 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     140.72      46.67      94.05     72
           1     steady     141.03      48.98      92.05     72
           2     steady     139.21      46.90      92.31     72
           3     steady     140.37      48.54      91.83     72
  backward_step: 4 micro-batch(es), avg_comp=64.36 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.94      72.58      64.36     48
           1     steady     136.71      72.22      64.49     48
           2     steady     137.55      73.26      64.29     48
           3     steady     136.47      72.15      64.32     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.69 ms

--- Rank 10 (PP stage 1/1) ---
  forward_step: 4 micro-batch(es), avg_comp=87.62 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     141.63      53.66      87.97     72
           1     steady     142.75      55.62      87.13     72
           2     steady     140.49      53.17      87.32     72
           3     steady     141.50      53.45      88.05     72
  backward_step: 4 micro-batch(es), avg_comp=63.23 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.86      73.62      63.24     48
           1     steady     136.98      73.98      63.00     48
           2     steady     138.05      74.72      63.33     48
           3     steady     136.41      73.07      63.34     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.48 ms

--- Rank 11 (PP stage 1/1) ---
  forward_step: 4 micro-batch(es), avg_comp=88.53 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     140.85      51.11      89.74     72
           1     steady     141.41      52.71      88.70     72
           2     steady     139.18      51.51      87.67     72
           3     steady     140.40      52.38      88.02     72
  backward_step: 4 micro-batch(es), avg_comp=61.47 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.91      75.48      61.43     48
           1     steady     136.68      74.98      61.70     48
           2     steady     137.62      76.13      61.49     48
           3     steady     136.50      75.25      61.25     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.48 ms

--- Rank 12 (PP stage 1/1) ---
  forward_step: 4 micro-batch(es), avg_comp=90.84 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     140.87      48.64      92.23     72
           1     steady     141.26      50.49      90.77     72
           2     steady     139.50      48.94      90.56     72
           3     steady     140.51      50.71      89.80     72
  backward_step: 4 micro-batch(es), avg_comp=64.89 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.91      71.50      65.41     48
           1     steady     136.67      71.88      64.79     48
           2     steady     137.57      72.86      64.71     48
           3     steady     136.45      71.78      64.67     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.72 ms

--- Rank 13 (PP stage 1/1) ---
  forward_step: 4 micro-batch(es), avg_comp=88.28 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     140.48      51.30      89.18     72
           1     steady     140.81      52.66      88.15     72
           2     steady     138.91      50.68      88.23     72
           3     steady     139.90      52.32      87.58     72
  backward_step: 4 micro-batch(es), avg_comp=63.06 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.94      73.72      63.22     48
           1     steady     136.77      73.70      63.07     48
           2     steady     137.62      74.43      63.19     48
           3     steady     136.50      73.74      62.76     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.56 ms

--- Rank 14 (PP stage 1/1) ---
  forward_step: 4 micro-batch(es), avg_comp=94.45 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     141.30      45.62      95.68     72
           1     steady     142.05      47.66      94.39     72
           2     steady     139.69      46.05      93.64     72
           3     steady     141.05      46.94      94.11     72
  backward_step: 4 micro-batch(es), avg_comp=68.14 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     136.97      68.93      68.04     48
           1     steady     136.67      68.23      68.44     48
           2     steady     137.69      69.55      68.14     48
           3     steady     136.43      68.48      67.95     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.78 ms

--- Rank 15 (PP stage 1/1) ---
  forward_step: 4 micro-batch(es), avg_comp=87.81 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     141.08      51.85      89.23     72
           1     steady     141.14      53.06      88.08     72
           2     steady     139.01      52.00      87.01     72
           3     steady     140.20      53.30      86.90     72
  backward_step: 4 micro-batch(es), avg_comp=68.50 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady     137.47      68.98      68.49     48
           1     steady     136.77      68.36      68.41     48
           2     steady     137.90      69.59      68.31     48
           3     steady     136.80      68.00      68.80     48
  optimizer_step: 1 micro-batch(es), avg_comp=43.68 ms

==================================================================================================================================
PART 2: Summary Comparison — Scaling Comp vs Distributed Comp (avg across micro-batches)
         Distributed comp = total_duration - sum(comm_sub_op_durations)
==================================================================================================================================

Rank  PP |     S_fwd     D_fwd      Δfwd     err% |     S_bwd     D_bwd      Δbwd     err% |     S_opt     D_opt      Δopt     err%
-----------------------------------------------------------------------------------------------------------------------------------
   0   0 |    105.64     85.36     20.28   23.76% |     88.51     69.79     18.72   26.82% |     48.64     43.77      4.87   11.13%
   1   0 |    123.30     80.43     42.87   53.30% |     84.95     68.44     16.51   24.13% |     49.32     43.81      5.51   12.58%
   2   0 |    117.97     80.82     37.15   45.97% |    100.05     66.29     33.76   50.94% |     49.05     43.79      5.26   12.01%
   3   0 |    103.09     81.72     21.37   26.15% |     73.32     67.34      5.98    8.89% |     49.18     43.66      5.52   12.64%
   4   0 |    116.47     81.69     34.78   42.57% |     79.92     66.39     13.53   20.38% |     49.12     43.65      5.47   12.53%
   5   0 |    126.51     84.53     41.98   49.66% |     91.68     72.14     19.54   27.08% |     49.35     43.82      5.53   12.62%
   6   0 |    125.66     85.86     39.80   46.35% |    100.29     71.28     29.01   40.69% |     48.93     43.81      5.12   11.69%
   7   0 |    136.76     86.19     50.56   58.66% |    101.22     71.81     29.41   40.97% |     49.02     43.69      5.33   12.20%
   8   1 |    115.44     89.98     25.46   28.30% |     90.73     64.78     25.95   40.06% |     41.62     43.58     -1.96   -4.50%
   9   1 |    121.32     92.56     28.76   31.07% |     86.80     64.36     22.44   34.86% |     48.98     43.69      5.29   12.11%
  10   1 |    191.59     87.62    103.97  118.67% |    103.27     63.23     40.04   63.33% |     43.76     43.48      0.28    0.64%
  11   1 |    112.98     88.53     24.45   27.61% |     75.27     61.47     13.80   22.45% |     43.91     43.48      0.43    0.99%
  12   1 |    127.09     90.84     36.25   39.91% |     82.09     64.89     17.20   26.50% |     43.95     43.72      0.23    0.53%
  13   1 |    117.68     88.28     29.40   33.30% |     93.99     63.06     30.93   49.05% |     43.37     43.56     -0.19   -0.44%
  14   1 |    126.59     94.45     32.14   34.02% |    103.22     68.14     35.08   51.48% |     48.98     43.78      5.20   11.88%
  15   1 |    128.27     87.81     40.47   46.09% |    103.06     68.50     34.56   50.45% |     43.67     43.68     -0.01   -0.02%
-----------------------------------------------------------------------------------------------------------------------------------

Aggregation Statistics (relative error %):
                         mean   |mean|   median |median| max|err| min|err|
        forward_step   44.09%   44.09%   42.57%   42.57%  118.67%   23.76%
       backward_step   36.13%   36.13%   40.06%   40.06%   63.33%    8.89%
      optimizer_step    7.41%    8.03%   11.88%   11.88%   12.64%    0.02%

==================================================================================================================================
PART 3: Per-PP-Stage Statistics
==================================================================================================================================

  PP Stage 0 (ranks: [0, 1, 2, 3, 4, 5, 6, 7]):
                           mean   |mean|   median max|err|
          forward_step   43.30%   43.30%   46.35%   58.66%
         backward_step   29.99%   29.99%   27.08%   50.94%
        optimizer_step   12.17%   12.17%   12.53%   12.64%

  PP Stage 1 (ranks: [8, 9, 10, 11, 12, 13, 14, 15]):
                           mean   |mean|   median max|err|
          forward_step   44.87%   44.87%   34.02%  118.67%
         backward_step   42.27%   42.27%   49.05%   63.33%
        optimizer_step    2.65%    3.89%    0.64%   12.11%

==================================================================================================================================
PART 4: Pass/Fail Check (threshold = 5.0% relative error)
==================================================================================================================================
  FAIL: Rank 0 (PP0) fwd: |23.76%| > 5.0%
  FAIL: Rank 0 (PP0) bwd: |26.82%| > 5.0%
  FAIL: Rank 0 (PP0) opt: |11.13%| > 5.0%
  FAIL: Rank 1 (PP0) fwd: |53.30%| > 5.0%
  FAIL: Rank 1 (PP0) bwd: |24.13%| > 5.0%
  FAIL: Rank 1 (PP0) opt: |12.58%| > 5.0%
  FAIL: Rank 2 (PP0) fwd: |45.97%| > 5.0%
  FAIL: Rank 2 (PP0) bwd: |50.94%| > 5.0%
  FAIL: Rank 2 (PP0) opt: |12.01%| > 5.0%
  FAIL: Rank 3 (PP0) fwd: |26.15%| > 5.0%
  FAIL: Rank 3 (PP0) bwd: |8.89%| > 5.0%
  FAIL: Rank 3 (PP0) opt: |12.64%| > 5.0%
  FAIL: Rank 4 (PP0) fwd: |42.57%| > 5.0%
  FAIL: Rank 4 (PP0) bwd: |20.38%| > 5.0%
  FAIL: Rank 4 (PP0) opt: |12.53%| > 5.0%
  FAIL: Rank 5 (PP0) fwd: |49.66%| > 5.0%
  FAIL: Rank 5 (PP0) bwd: |27.08%| > 5.0%
  FAIL: Rank 5 (PP0) opt: |12.62%| > 5.0%
  FAIL: Rank 6 (PP0) fwd: |46.35%| > 5.0%
  FAIL: Rank 6 (PP0) bwd: |40.69%| > 5.0%
  FAIL: Rank 6 (PP0) opt: |11.69%| > 5.0%
  FAIL: Rank 7 (PP0) fwd: |58.66%| > 5.0%
  FAIL: Rank 7 (PP0) bwd: |40.97%| > 5.0%
  FAIL: Rank 7 (PP0) opt: |12.20%| > 5.0%
  FAIL: Rank 8 (PP1) fwd: |28.30%| > 5.0%
  FAIL: Rank 8 (PP1) bwd: |40.06%| > 5.0%
  FAIL: Rank 9 (PP1) fwd: |31.07%| > 5.0%
  FAIL: Rank 9 (PP1) bwd: |34.86%| > 5.0%
  FAIL: Rank 9 (PP1) opt: |12.11%| > 5.0%
  FAIL: Rank 10 (PP1) fwd: |118.67%| > 5.0%
  FAIL: Rank 10 (PP1) bwd: |63.33%| > 5.0%
  FAIL: Rank 11 (PP1) fwd: |27.61%| > 5.0%
  FAIL: Rank 11 (PP1) bwd: |22.45%| > 5.0%
  FAIL: Rank 12 (PP1) fwd: |39.91%| > 5.0%
  FAIL: Rank 12 (PP1) bwd: |26.50%| > 5.0%
  FAIL: Rank 13 (PP1) fwd: |33.30%| > 5.0%
  FAIL: Rank 13 (PP1) bwd: |49.05%| > 5.0%
  FAIL: Rank 14 (PP1) fwd: |34.02%| > 5.0%
  FAIL: Rank 14 (PP1) bwd: |51.48%| > 5.0%
  FAIL: Rank 14 (PP1) opt: |11.88%| > 5.0%
  FAIL: Rank 15 (PP1) fwd: |46.09%| > 5.0%
  FAIL: Rank 15 (PP1) bwd: |50.45%| > 5.0%

  Result: 6/48 checks passed (12.5%)
  ❌ 42 checks FAILED


Analyzing Config B (PP=4, EP=4) ...

##################################################################################################################################
# Config B (PP=4, EP=4)  (pp4_tp1_exp4_expn128_dp4_nl48_hs2048_sl2048)
# PP=4, TP=1, EP=4, DP=4, num_experts=128, num_layers=48, hidden_size=2048, seq_len=2048
##################################################################################################################################

==================================================================================================================================
PART 1: Per-Rank Micro-Batch Breakdown (Distributed Mode)
==================================================================================================================================

--- Rank 0 (PP stage 0/3) ---
  forward_step: 8 micro-batch(es), avg_comp=42.76 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      66.64      22.45      44.19     36
           1     warmup      65.23      22.88      42.35     36
           2     warmup      65.56      23.06      42.50     36
           3     steady      65.52      23.38      42.14     36
           4     steady      65.52      21.97      43.55     36
           5     steady      66.60      23.96      42.64     36
           6     steady      66.17      23.74      42.43     36
           7     steady      65.96      23.67      42.29     36
  backward_step: 8 micro-batch(es), avg_comp=37.93 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      74.07      35.08      38.99     24
           1     steady      74.47      36.01      38.46     24
           2     steady      73.45      35.93      37.52     24
           3     steady      73.50      35.99      37.51     24
           4     steady      74.39      36.16      38.23     24
           5   cooldown      73.47      35.96      37.51     24
           6   cooldown      73.84      36.19      37.65     24
           7   cooldown      73.55      35.99      37.56     24
  optimizer_step: 1 micro-batch(es), avg_comp=37.02 ms

--- Rank 1 (PP stage 0/3) ---
  forward_step: 8 micro-batch(es), avg_comp=43.23 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      66.63      22.40      44.23     36
           1     warmup      65.33      22.57      42.76     36
           2     warmup      65.18      22.41      42.77     36
           3     steady      65.40      22.75      42.65     36
           4     steady      65.36      21.71      43.65     36
           5     steady      66.73      22.73      44.00     36
           6     steady      65.38      22.37      43.01     36
           7     steady      65.11      22.35      42.76     36
  backward_step: 8 micro-batch(es), avg_comp=38.33 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      74.52      35.21      39.31     24
           1     steady      74.37      35.57      38.80     24
           2     steady      74.46      35.79      38.67     24
           3     steady      74.92      36.44      38.48     24
           4     steady      73.93      36.31      37.62     24
           5   cooldown      73.97      36.19      37.78     24
           6   cooldown      74.11      36.22      37.89     24
           7   cooldown      74.44      36.39      38.05     24
  optimizer_step: 1 micro-batch(es), avg_comp=37.00 ms

--- Rank 2 (PP stage 0/3) ---
  forward_step: 8 micro-batch(es), avg_comp=44.87 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      66.72      21.12      45.60     36
           1     warmup      65.29      20.27      45.02     36
           2     warmup      65.41      20.56      44.85     36
           3     steady      65.43      20.18      45.25     36
           4     steady      65.58      21.53      44.05     36
           5     steady      66.52      21.41      45.11     36
           6     steady      65.96      21.32      44.64     36
           7     steady      65.87      21.44      44.43     36
  backward_step: 8 micro-batch(es), avg_comp=41.22 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      74.51      33.17      41.34     24
           1     steady      75.00      32.99      42.01     24
           2     steady      73.78      32.91      40.87     24
           3     steady      73.94      33.18      40.76     24
           4     steady      73.91      33.20      40.71     24
           5   cooldown      75.08      33.45      41.63     24
           6   cooldown      74.93      33.15      41.78     24
           7   cooldown      73.91      33.25      40.66     24
  optimizer_step: 1 micro-batch(es), avg_comp=36.94 ms

--- Rank 3 (PP stage 0/3) ---
  forward_step: 8 micro-batch(es), avg_comp=45.59 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      66.51      20.22      46.29     36
           1     warmup      65.33      20.04      45.29     36
           2     warmup      65.36      20.28      45.08     36
           3     steady      65.45      20.08      45.37     36
           4     steady      65.34      19.51      45.83     36
           5     steady      66.27      19.89      46.38     36
           6     steady      65.84      20.53      45.31     36
           7     steady      65.21      20.07      45.14     36
  backward_step: 8 micro-batch(es), avg_comp=41.28 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      74.32      32.50      41.82     24
           1     steady      74.62      31.92      42.70     24
           2     steady      73.90      32.99      40.91     24
           3     steady      74.37      33.02      41.35     24
           4     steady      73.98      33.05      40.93     24
           5   cooldown      74.21      32.89      41.32     24
           6   cooldown      73.65      32.94      40.71     24
           7   cooldown      73.53      33.02      40.51     24
  optimizer_step: 1 micro-batch(es), avg_comp=37.15 ms

--- Rank 4 (PP stage 1/3) ---
  forward_step: 8 micro-batch(es), avg_comp=43.97 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      66.87      21.13      45.74     36
           1     warmup      65.29      21.68      43.61     36
           2     steady      65.73      22.38      43.35     36
           3     steady      64.60      20.91      43.69     36
           4     steady      65.90      21.35      44.55     36
           5     steady      64.71      21.02      43.69     36
           6     steady      64.17      20.49      43.68     36
           7     steady      64.39      20.97      43.42     36
  backward_step: 8 micro-batch(es), avg_comp=37.91 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      71.01      32.95      38.06     24
           1     steady      71.07      32.83      38.24     24
           2     steady      71.34      33.66      37.68     24
           3     steady      71.05      33.00      38.05     24
           4     steady      71.09      32.95      38.14     24
           5     steady      71.06      33.24      37.82     24
           6   cooldown      71.07      33.40      37.67     24
           7   cooldown      71.14      33.52      37.62     24
  optimizer_step: 1 micro-batch(es), avg_comp=32.49 ms

--- Rank 5 (PP stage 1/3) ---
  forward_step: 8 micro-batch(es), avg_comp=41.50 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      67.02      24.57      42.45     36
           1     warmup      65.06      24.03      41.03     36
           2     steady      65.53      24.83      40.70     36
           3     steady      65.10      23.61      41.49     36
           4     steady      66.35      24.23      42.12     36
           5     steady      65.08      23.59      41.49     36
           6     steady      64.64      23.33      41.31     36
           7     steady      64.82      23.44      41.38     36
  backward_step: 8 micro-batch(es), avg_comp=34.82 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      71.07      36.02      35.05     24
           1     steady      71.03      35.96      35.07     24
           2     steady      71.82      36.83      34.99     24
           3     steady      71.29      36.41      34.88     24
           4     steady      71.45      36.57      34.88     24
           5     steady      70.94      36.33      34.61     24
           6   cooldown      70.46      35.85      34.61     24
           7   cooldown      70.59      36.15      34.44     24
  optimizer_step: 1 micro-batch(es), avg_comp=32.34 ms

--- Rank 6 (PP stage 1/3) ---
  forward_step: 8 micro-batch(es), avg_comp=44.41 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      66.95      21.31      45.64     36
           1     warmup      65.20      21.56      43.64     36
           2     steady      65.33      21.81      43.52     36
           3     steady      64.79      19.94      44.85     36
           4     steady      66.37      21.58      44.79     36
           5     steady      65.00      20.47      44.53     36
           6     steady      64.47      20.20      44.27     36
           7     steady      64.75      20.69      44.06     36
  backward_step: 8 micro-batch(es), avg_comp=38.21 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      70.93      32.05      38.88     24
           1     steady      70.92      32.20      38.72     24
           2     steady      71.45      32.79      38.66     24
           3     steady      70.99      32.98      38.01     24
           4     steady      71.33      33.32      38.01     24
           5     steady      70.76      32.88      37.88     24
           6   cooldown      70.64      32.82      37.82     24
           7   cooldown      70.81      33.07      37.74     24
  optimizer_step: 1 micro-batch(es), avg_comp=32.48 ms

--- Rank 7 (PP stage 1/3) ---
  forward_step: 8 micro-batch(es), avg_comp=42.94 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      66.97      22.48      44.49     36
           1     warmup      65.16      23.13      42.03     36
           2     steady      65.40      23.60      41.80     36
           3     steady      64.70      21.65      43.05     36
           4     steady      66.01      22.23      43.78     36
           5     steady      64.94      21.89      43.05     36
           6     steady      64.30      21.91      42.39     36
           7     steady      64.66      21.76      42.90     36
  backward_step: 8 micro-batch(es), avg_comp=41.54 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      71.18      29.78      41.40     24
           1     steady      71.13      29.53      41.60     24
           2     steady      71.85      30.15      41.70     24
           3     steady      71.42      29.73      41.69     24
           4     steady      71.54      30.07      41.47     24
           5     steady      71.39      29.79      41.60     24
           6   cooldown      70.89      29.41      41.48     24
           7   cooldown      70.80      29.41      41.39     24
  optimizer_step: 1 micro-batch(es), avg_comp=32.39 ms

--- Rank 8 (PP stage 2/3) ---
  forward_step: 8 micro-batch(es), avg_comp=41.79 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      65.12      22.42      42.70     36
           1     steady      63.45      21.76      41.69     36
           2     steady      63.88      22.32      41.56     36
           3     steady      63.16      21.48      41.68     36
           4     steady      62.82      21.54      41.28     36
           5     steady      63.07      21.63      41.44     36
           6     steady      63.05      20.93      42.12     36
           7     steady      63.10      21.26      41.84     36
  backward_step: 8 micro-batch(es), avg_comp=33.95 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      69.88      35.67      34.21     24
           1     steady      70.38      36.16      34.22     24
           2     steady      70.01      36.09      33.92     24
           3     steady      70.09      36.21      33.88     24
           4     steady      69.75      35.84      33.91     24
           5     steady      69.80      35.93      33.87     24
           6     steady      69.99      36.18      33.81     24
           7   cooldown      69.73      35.93      33.80     24
  optimizer_step: 1 micro-batch(es), avg_comp=32.42 ms

--- Rank 9 (PP stage 2/3) ---
  forward_step: 8 micro-batch(es), avg_comp=43.18 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      65.12      21.09      44.03     36
           1     steady      63.51      20.64      42.87     36
           2     steady      63.90      20.65      43.25     36
           3     steady      63.12      20.05      43.07     36
           4     steady      63.08      20.12      42.96     36
           5     steady      63.18      20.30      42.88     36
           6     steady      63.24      20.01      43.23     36
           7     steady      63.19      20.05      43.14     36
  backward_step: 8 micro-batch(es), avg_comp=36.41 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      69.86      33.08      36.78     24
           1     steady      69.69      32.94      36.75     24
           2     steady      69.86      33.45      36.41     24
           3     steady      70.05      33.78      36.27     24
           4     steady      69.79      33.50      36.29     24
           5     steady      69.83      33.46      36.37     24
           6     steady      69.97      33.75      36.22     24
           7   cooldown      69.72      33.52      36.20     24
  optimizer_step: 1 micro-batch(es), avg_comp=32.39 ms

--- Rank 10 (PP stage 2/3) ---
  forward_step: 8 micro-batch(es), avg_comp=42.11 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      64.89      21.79      43.10     36
           1     steady      63.61      21.55      42.06     36
           2     steady      63.75      21.84      41.91     36
           3     steady      63.04      21.05      41.99     36
           4     steady      62.87      20.99      41.88     36
           5     steady      62.89      21.05      41.84     36
           6     steady      63.12      20.92      42.20     36
           7     steady      62.96      21.08      41.88     36
  backward_step: 8 micro-batch(es), avg_comp=35.18 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      69.76      34.27      35.49     24
           1     steady      70.16      34.83      35.33     24
           2     steady      69.81      34.63      35.18     24
           3     steady      69.87      34.82      35.05     24
           4     steady      69.86      34.73      35.13     24
           5     steady      69.74      34.62      35.12     24
           6     steady      69.82      34.72      35.10     24
           7   cooldown      69.66      34.61      35.05     24
  optimizer_step: 1 micro-batch(es), avg_comp=32.36 ms

--- Rank 11 (PP stage 2/3) ---
  forward_step: 8 micro-batch(es), avg_comp=43.03 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      64.59      20.62      43.97     36
           1     steady      63.30      20.23      43.07     36
           2     steady      63.85      20.76      43.09     36
           3     steady      63.11      20.04      43.07     36
           4     steady      63.00      20.29      42.71     36
           5     steady      62.94      20.04      42.90     36
           6     steady      63.16      20.39      42.77     36
           7     steady      63.14      20.44      42.70     36
  backward_step: 8 micro-batch(es), avg_comp=37.09 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      69.85      32.42      37.43     24
           1     steady      70.00      32.65      37.35     24
           2     steady      69.79      32.78      37.01     24
           3     steady      69.76      32.77      36.99     24
           4     steady      69.83      32.85      36.98     24
           5     steady      69.78      32.72      37.06     24
           6     steady      69.87      32.94      36.93     24
           7   cooldown      69.71      32.71      37.00     24
  optimizer_step: 1 micro-batch(es), avg_comp=32.39 ms

--- Rank 12 (PP stage 3/3) ---
  forward_step: 8 micro-batch(es), avg_comp=47.04 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      75.56      27.53      48.03     36
           1     steady      73.66      25.44      48.22     36
           2     steady      72.91      25.96      46.95     36
           3     steady      71.71      23.74      47.97     36
           4     steady      71.78      25.78      46.00     36
           5     steady      71.74      25.40      46.34     36
           6     steady      70.83      24.38      46.45     36
           7     steady      70.61      24.23      46.38     36
  backward_step: 8 micro-batch(es), avg_comp=33.66 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      78.39      44.57      33.82     24
           1     steady      76.72      42.86      33.86     24
           2     steady      76.34      42.70      33.64     24
           3     steady      76.55      42.91      33.64     24
           4     steady      76.32      42.64      33.68     24
           5     steady      76.06      42.53      33.53     24
           6     steady      75.99      42.36      33.63     24
           7     steady      75.89      42.44      33.45     24
  optimizer_step: 1 micro-batch(es), avg_comp=38.45 ms

--- Rank 13 (PP stage 3/3) ---
  forward_step: 8 micro-batch(es), avg_comp=52.62 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      74.99      19.71      55.28     36
           1     steady      73.54      19.99      53.55     36
           2     steady      71.90      18.69      53.21     36
           3     steady      71.54      18.86      52.68     36
           4     steady      71.45      19.62      51.83     36
           5     steady      71.62      20.19      51.43     36
           6     steady      70.80      19.11      51.69     36
           7     steady      70.46      19.16      51.30     36
  backward_step: 8 micro-batch(es), avg_comp=39.45 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      78.26      29.83      48.43     24
           1     steady      76.79      37.36      39.43     24
           2     steady      75.89      37.58      38.31     24
           3     steady      76.27      38.01      38.26     24
           4     steady      76.06      38.19      37.87     24
           5     steady      75.82      37.94      37.88     24
           6     steady      75.77      37.83      37.94     24
           7     steady      75.74      38.25      37.49     24
  optimizer_step: 1 micro-batch(es), avg_comp=38.60 ms

--- Rank 14 (PP stage 3/3) ---
  forward_step: 8 micro-batch(es), avg_comp=51.07 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      75.28      20.93      54.35     36
           1     steady      73.55      21.77      51.78     36
           2     steady      72.59      21.33      51.26     36
           3     steady      71.55      20.87      50.68     36
           4     steady      71.63      21.62      50.01     36
           5     steady      71.88      21.54      50.34     36
           6     steady      70.88      20.92      49.96     36
           7     steady      70.44      20.28      50.16     36
  backward_step: 8 micro-batch(es), avg_comp=39.05 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      78.31      39.05      39.26     24
           1     steady      76.56      37.16      39.40     24
           2     steady      76.14      37.09      39.05     24
           3     steady      76.35      37.12      39.23     24
           4     steady      76.03      36.95      39.08     24
           5     steady      75.82      36.93      38.89     24
           6     steady      75.90      36.97      38.93     24
           7     steady      75.79      37.24      38.55     24
  optimizer_step: 1 micro-batch(es), avg_comp=38.58 ms

--- Rank 15 (PP stage 3/3) ---
  forward_step: 8 micro-batch(es), avg_comp=51.89 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      75.23      20.82      54.41     36
           1     steady      73.60      21.23      52.37     36
           2     steady      72.33      19.90      52.43     36
           3     steady      71.47      19.80      51.67     36
           4     steady      71.31      20.02      51.29     36
           5     steady      71.78      20.48      51.30     36
           6     steady      70.84      20.20      50.64     36
           7     steady      70.42      19.38      51.04     36
  backward_step: 8 micro-batch(es), avg_comp=40.46 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      78.29      37.32      40.97     24
           1     steady      76.53      35.81      40.72     24
           2     steady      76.10      35.37      40.73     24
           3     steady      76.46      36.06      40.40     24
           4     steady      75.98      35.80      40.18     24
           5     steady      75.89      35.50      40.39     24
           6     steady      75.91      35.52      40.39     24
           7     steady      75.75      35.89      39.86     24
  optimizer_step: 1 micro-batch(es), avg_comp=38.46 ms

==================================================================================================================================
PART 2: Summary Comparison — Scaling Comp vs Distributed Comp (avg across micro-batches)
         Distributed comp = total_duration - sum(comm_sub_op_durations)
==================================================================================================================================

Rank  PP |     S_fwd     D_fwd      Δfwd     err% |     S_bwd     D_bwd      Δbwd     err% |     S_opt     D_opt      Δopt     err%
-----------------------------------------------------------------------------------------------------------------------------------
   0   0 |     52.64     42.76      9.88   23.10% |     51.51     37.93     13.58   35.81% |     43.96     37.02      6.94   18.75%
   1   0 |     53.46     43.23     10.23   23.67% |     50.91     38.33     12.58   32.84% |     43.95     37.00      6.95   18.78%
   2   0 |     54.71     44.87      9.84   21.93% |     51.07     41.22      9.85   23.90% |     43.85     36.94      6.91   18.71%
   3   0 |     58.67     45.59     13.08   28.70% |     59.38     41.28     18.10   43.84% |     44.02     37.15      6.87   18.49%
   4   1 |     52.51     43.97      8.54   19.43% |     48.18     37.91     10.27   27.09% |     38.15     32.49      5.66   17.42%
   5   1 |     56.42     41.50     14.92   35.96% |     48.25     34.82     13.43   38.58% |     38.33     32.34      5.99   18.52%
   6   1 |     54.98     44.41     10.57   23.79% |     48.26     38.21     10.05   26.29% |     38.15     32.48      5.67   17.46%
   7   1 |     58.83     42.94     15.89   37.02% |     55.75     41.54     14.21   34.20% |     38.16     32.39      5.77   17.81%
   8   2 |     53.55     41.79     11.76   28.14% |     48.49     33.95     14.54   42.82% |     38.24     32.42      5.82   17.95%
   9   2 |     53.91     43.18     10.73   24.85% |     48.23     36.41     11.82   32.46% |     38.19     32.39      5.80   17.91%
  10   2 |     57.92     42.11     15.81   37.55% |     48.40     35.18     13.22   37.57% |     38.27     32.36      5.91   18.26%
  11   2 |     55.04     43.03     12.01   27.90% |     56.08     37.09     18.99   51.18% |     38.31     32.39      5.92   18.28%
  12   3 |     59.97     47.04     12.93   27.48% |     53.71     33.66     20.05   59.58% |     43.81     38.45      5.36   13.94%
  13   3 |     62.31     52.62      9.69   18.41% |     53.72     39.45     14.27   36.17% |     38.62     38.60      0.02    0.05%
  14   3 |     64.44     51.07     13.37   26.19% |     53.65     39.05     14.60   37.39% |     38.56     38.58     -0.02   -0.05%
  15   3 |     68.42     51.89     16.53   31.85% |     61.81     40.46     21.35   52.79% |     38.75     38.46      0.29    0.75%
-----------------------------------------------------------------------------------------------------------------------------------

Aggregation Statistics (relative error %):
                         mean   |mean|   median |median| max|err| min|err|
        forward_step   27.25%   27.25%   27.48%   27.48%   37.55%   18.41%
       backward_step   38.28%   38.28%   37.39%   37.39%   59.58%   23.90%
      optimizer_step   14.56%   14.57%   17.95%   17.95%   18.78%    0.05%

==================================================================================================================================
PART 3: Per-PP-Stage Statistics
==================================================================================================================================

  PP Stage 0 (ranks: [0, 1, 2, 3]):
                           mean   |mean|   median max|err|
          forward_step   24.35%   24.35%   23.67%   28.70%
         backward_step   34.10%   34.10%   35.81%   43.84%
        optimizer_step   18.68%   18.68%   18.75%   18.78%

  PP Stage 1 (ranks: [4, 5, 6, 7]):
                           mean   |mean|   median max|err|
          forward_step   29.05%   29.05%   35.96%   37.02%
         backward_step   31.54%   31.54%   34.20%   38.58%
        optimizer_step   17.80%   17.80%   17.81%   18.52%

  PP Stage 2 (ranks: [8, 9, 10, 11]):
                           mean   |mean|   median max|err|
          forward_step   29.61%   29.61%   28.14%   37.55%
         backward_step   41.01%   41.01%   42.82%   51.18%
        optimizer_step   18.10%   18.10%   18.26%   18.28%

  PP Stage 3 (ranks: [12, 13, 14, 15]):
                           mean   |mean|   median max|err|
          forward_step   25.98%   25.98%   27.48%   31.85%
         backward_step   46.48%   46.48%   52.79%   59.58%
        optimizer_step    3.67%    3.70%    0.75%   13.94%

==================================================================================================================================
PART 4: Pass/Fail Check (threshold = 5.0% relative error)
==================================================================================================================================
  FAIL: Rank 0 (PP0) fwd: |23.10%| > 5.0%
  FAIL: Rank 0 (PP0) bwd: |35.81%| > 5.0%
  FAIL: Rank 0 (PP0) opt: |18.75%| > 5.0%
  FAIL: Rank 1 (PP0) fwd: |23.67%| > 5.0%
  FAIL: Rank 1 (PP0) bwd: |32.84%| > 5.0%
  FAIL: Rank 1 (PP0) opt: |18.78%| > 5.0%
  FAIL: Rank 2 (PP0) fwd: |21.93%| > 5.0%
  FAIL: Rank 2 (PP0) bwd: |23.90%| > 5.0%
  FAIL: Rank 2 (PP0) opt: |18.71%| > 5.0%
  FAIL: Rank 3 (PP0) fwd: |28.70%| > 5.0%
  FAIL: Rank 3 (PP0) bwd: |43.84%| > 5.0%
  FAIL: Rank 3 (PP0) opt: |18.49%| > 5.0%
  FAIL: Rank 4 (PP1) fwd: |19.43%| > 5.0%
  FAIL: Rank 4 (PP1) bwd: |27.09%| > 5.0%
  FAIL: Rank 4 (PP1) opt: |17.42%| > 5.0%
  FAIL: Rank 5 (PP1) fwd: |35.96%| > 5.0%
  FAIL: Rank 5 (PP1) bwd: |38.58%| > 5.0%
  FAIL: Rank 5 (PP1) opt: |18.52%| > 5.0%
  FAIL: Rank 6 (PP1) fwd: |23.79%| > 5.0%
  FAIL: Rank 6 (PP1) bwd: |26.29%| > 5.0%
  FAIL: Rank 6 (PP1) opt: |17.46%| > 5.0%
  FAIL: Rank 7 (PP1) fwd: |37.02%| > 5.0%
  FAIL: Rank 7 (PP1) bwd: |34.20%| > 5.0%
  FAIL: Rank 7 (PP1) opt: |17.81%| > 5.0%
  FAIL: Rank 8 (PP2) fwd: |28.14%| > 5.0%
  FAIL: Rank 8 (PP2) bwd: |42.82%| > 5.0%
  FAIL: Rank 8 (PP2) opt: |17.95%| > 5.0%
  FAIL: Rank 9 (PP2) fwd: |24.85%| > 5.0%
  FAIL: Rank 9 (PP2) bwd: |32.46%| > 5.0%
  FAIL: Rank 9 (PP2) opt: |17.91%| > 5.0%
  FAIL: Rank 10 (PP2) fwd: |37.55%| > 5.0%
  FAIL: Rank 10 (PP2) bwd: |37.57%| > 5.0%
  FAIL: Rank 10 (PP2) opt: |18.26%| > 5.0%
  FAIL: Rank 11 (PP2) fwd: |27.90%| > 5.0%
  FAIL: Rank 11 (PP2) bwd: |51.18%| > 5.0%
  FAIL: Rank 11 (PP2) opt: |18.28%| > 5.0%
  FAIL: Rank 12 (PP3) fwd: |27.48%| > 5.0%
  FAIL: Rank 12 (PP3) bwd: |59.58%| > 5.0%
  FAIL: Rank 12 (PP3) opt: |13.94%| > 5.0%
  FAIL: Rank 13 (PP3) fwd: |18.41%| > 5.0%
  FAIL: Rank 13 (PP3) bwd: |36.17%| > 5.0%
  FAIL: Rank 14 (PP3) fwd: |26.19%| > 5.0%
  FAIL: Rank 14 (PP3) bwd: |37.39%| > 5.0%
  FAIL: Rank 15 (PP3) fwd: |31.85%| > 5.0%
  FAIL: Rank 15 (PP3) bwd: |52.79%| > 5.0%

  Result: 3/48 checks passed (6.2%)
  ❌ 45 checks FAILED


Analyzing Config C (PP=8, EP=2) ...

##################################################################################################################################
# Config C (PP=8, EP=2)  (pp8_tp1_exp2_expn128_dp2_nl48_hs2048_sl2048)
# PP=8, TP=1, EP=2, DP=2, num_experts=128, num_layers=48, hidden_size=2048, seq_len=2048
##################################################################################################################################

==================================================================================================================================
PART 1: Per-Rank Micro-Batch Breakdown (Distributed Mode)
==================================================================================================================================

--- Rank 0 (PP stage 0/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.73 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.21       7.65      23.56     18
           1     warmup      30.33       7.37      22.96     18
           2     warmup      30.10       7.28      22.82     18
           3     warmup      29.96       7.55      22.41     18
           4     warmup      30.10       7.43      22.67     18
           5     warmup      30.32       7.47      22.85     18
           6     warmup      29.85       7.31      22.54     18
           7     steady      30.46       7.92      22.54     18
           8     steady      30.56       7.54      23.02     18
           9     steady      31.17       7.87      23.30     18
          10     steady      30.01       7.19      22.82     18
          11     steady      30.56       7.22      23.34     18
          12     steady      29.29       7.26      22.03     18
          13     steady      29.65       7.46      22.19     18
          14     steady      30.18       8.05      22.13     18
          15     steady      30.60       8.09      22.51     18
  backward_step: 16 micro-batch(es), avg_comp=25.63 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      41.64      15.97      25.67     12
           1     steady      41.53      16.25      25.28     12
           2     steady      41.78      15.90      25.88     12
           3     steady      41.50      16.17      25.33     12
           4     steady      42.04      15.89      26.15     12
           5     steady      41.93      16.19      25.74     12
           6     steady      41.34      16.01      25.33     12
           7     steady      41.33      15.96      25.37     12
           8     steady      41.93      16.03      25.90     12
           9   cooldown      41.90      16.26      25.64     12
          10   cooldown      41.87      16.18      25.69     12
          11   cooldown      41.77      16.28      25.49     12
          12   cooldown      41.96      16.17      25.79     12
          13   cooldown      41.64      16.17      25.47     12
          14   cooldown      41.77      16.17      25.60     12
          15   cooldown      41.80      16.09      25.71     12
  optimizer_step: 1 micro-batch(es), avg_comp=36.04 ms

--- Rank 1 (PP stage 0/7) ---
  forward_step: 16 micro-batch(es), avg_comp=23.62 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.17       6.73      24.44     18
           1     warmup      30.25       6.47      23.78     18
           2     warmup      29.95       6.34      23.61     18
           3     warmup      29.89       6.22      23.67     18
           4     warmup      30.44       6.83      23.61     18
           5     warmup      30.62       6.81      23.81     18
           6     warmup      29.97       6.66      23.31     18
           7     steady      30.23       6.74      23.49     18
           8     steady      29.98       6.29      23.69     18
           9     steady      30.60       6.29      24.31     18
          10     steady      30.19       6.56      23.63     18
          11     steady      30.66       6.34      24.32     18
          12     steady      29.46       6.46      23.00     18
          13     steady      30.46       7.00      23.46     18
          14     steady      29.31       6.36      22.95     18
          15     steady      29.59       6.68      22.91     18
  backward_step: 16 micro-batch(es), avg_comp=26.59 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      42.24      15.28      26.96     12
           1     steady      42.43      15.29      27.14     12
           2     steady      41.72      15.24      26.48     12
           3     steady      41.63      15.26      26.37     12
           4     steady      42.02      15.15      26.87     12
           5     steady      41.67      15.33      26.34     12
           6     steady      42.47      15.35      27.12     12
           7     steady      42.03      15.34      26.69     12
           8     steady      41.40      15.33      26.07     12
           9   cooldown      41.95      15.09      26.86     12
          10   cooldown      42.17      15.26      26.91     12
          11   cooldown      41.78      15.16      26.62     12
          12   cooldown      41.64      15.28      26.36     12
          13   cooldown      41.24      15.28      25.96     12
          14   cooldown      41.93      15.28      26.65     12
          15   cooldown      41.26      15.28      25.98     12
  optimizer_step: 1 micro-batch(es), avg_comp=35.92 ms

--- Rank 2 (PP stage 1/7) ---
  forward_step: 16 micro-batch(es), avg_comp=23.91 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      30.71       6.42      24.29     18
           1     warmup      29.96       6.51      23.45     18
           2     warmup      29.51       6.39      23.12     18
           3     warmup      29.78       6.60      23.18     18
           4     warmup      29.32       6.44      22.88     18
           5     warmup      29.87       6.40      23.47     18
           6     steady      29.63       6.66      22.97     18
           7     steady      31.06       6.90      24.16     18
           8     steady      31.00       6.93      24.07     18
           9     steady      31.51       7.01      24.50     18
          10     steady      30.72       6.69      24.03     18
          11     steady      31.34       6.77      24.57     18
          12     steady      32.22       7.78      24.44     18
          13     steady      31.84       7.00      24.84     18
          14     steady      30.69       6.76      23.93     18
          15     steady      31.30       6.67      24.63     18
  backward_step: 16 micro-batch(es), avg_comp=23.78 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      38.17      14.15      24.02     12
           1     steady      38.19      14.49      23.70     12
           2     steady      38.19      14.41      23.78     12
           3     steady      38.20      14.44      23.76     12
           4     steady      38.32      14.52      23.80     12
           5     steady      38.18      14.41      23.77     12
           6     steady      38.24      14.34      23.90     12
           7     steady      38.25      14.42      23.83     12
           8     steady      38.19      14.40      23.79     12
           9     steady      38.12      14.49      23.63     12
          10   cooldown      38.15      14.33      23.82     12
          11   cooldown      38.16      14.37      23.79     12
          12   cooldown      38.18      14.41      23.77     12
          13   cooldown      38.15      14.43      23.72     12
          14   cooldown      38.13      14.49      23.64     12
          15   cooldown      38.16      14.45      23.71     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.42 ms

--- Rank 3 (PP stage 1/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.92 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      30.73       7.63      23.10     18
           1     warmup      29.92       6.99      22.93     18
           2     warmup      29.59       7.12      22.47     18
           3     warmup      30.12       7.34      22.78     18
           4     warmup      29.51       7.24      22.27     18
           5     warmup      30.05       7.14      22.91     18
           6     steady      29.52       6.98      22.54     18
           7     steady      31.32       7.91      23.41     18
           8     steady      31.26       8.24      23.02     18
           9     steady      31.84       8.51      23.33     18
          10     steady      30.98       8.37      22.61     18
          11     steady      31.70       8.39      23.31     18
          12     steady      32.47       9.36      23.11     18
          13     steady      32.18       8.92      23.26     18
          14     steady      30.95       8.34      22.61     18
          15     steady      31.62       8.60      23.02     18
  backward_step: 16 micro-batch(es), avg_comp=22.23 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      38.20      15.94      22.26     12
           1     steady      38.02      15.73      22.29     12
           2     steady      37.98      15.74      22.24     12
           3     steady      37.99      15.76      22.23     12
           4     steady      38.10      15.86      22.24     12
           5     steady      37.98      15.80      22.18     12
           6     steady      38.25      15.99      22.26     12
           7     steady      38.35      16.21      22.14     12
           8     steady      38.02      15.81      22.21     12
           9     steady      38.02      15.82      22.20     12
          10   cooldown      38.30      16.08      22.22     12
          11   cooldown      38.29      16.01      22.28     12
          12   cooldown      38.27      15.90      22.37     12
          13   cooldown      38.26      16.06      22.20     12
          14   cooldown      38.27      16.08      22.19     12
          15   cooldown      38.27      16.08      22.19     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.36 ms

--- Rank 4 (PP stage 2/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.53 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      30.57       7.37      23.20     18
           1     warmup      30.00       7.33      22.67     18
           2     warmup      29.79       7.38      22.41     18
           3     warmup      29.67       7.42      22.25     18
           4     warmup      29.07       6.81      22.26     18
           5     steady      29.44       7.40      22.04     18
           6     steady      29.68       7.11      22.57     18
           7     steady      29.50       7.36      22.14     18
           8     steady      30.08       7.17      22.91     18
           9     steady      29.61       7.37      22.24     18
          10     steady      29.99       7.12      22.87     18
          11     steady      29.74       7.43      22.31     18
          12     steady      30.06       6.96      23.10     18
          13     steady      29.54       7.42      22.12     18
          14     steady      30.35       7.19      23.16     18
          15     steady      29.62       7.46      22.16     18
  backward_step: 16 micro-batch(es), avg_comp=21.41 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      38.07      16.46      21.61     12
           1     steady      37.98      16.67      21.31     12
           2     steady      38.10      16.81      21.29     12
           3     steady      38.12      16.71      21.41     12
           4     steady      37.95      16.62      21.33     12
           5     steady      38.07      16.68      21.39     12
           6     steady      38.54      16.46      22.08     12
           7     steady      38.06      16.63      21.43     12
           8     steady      38.00      16.56      21.44     12
           9     steady      37.99      16.69      21.30     12
          10     steady      38.03      16.64      21.39     12
          11   cooldown      38.03      16.70      21.33     12
          12   cooldown      38.05      16.76      21.29     12
          13   cooldown      38.02      16.70      21.32     12
          14   cooldown      38.05      16.74      21.31     12
          15   cooldown      38.09      16.75      21.34     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.34 ms

--- Rank 5 (PP stage 2/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.75 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      30.57       7.36      23.21     18
           1     warmup      30.06       6.99      23.07     18
           2     warmup      30.12       7.38      22.74     18
           3     warmup      29.86       7.32      22.54     18
           4     warmup      29.37       6.75      22.62     18
           5     steady      29.35       6.77      22.58     18
           6     steady      29.72       6.67      23.05     18
           7     steady      29.51       6.88      22.63     18
           8     steady      30.12       7.42      22.70     18
           9     steady      29.62       7.06      22.56     18
          10     steady      30.01       7.44      22.57     18
          11     steady      29.77       7.12      22.65     18
          12     steady      30.27       7.12      23.15     18
          13     steady      29.65       7.02      22.63     18
          14     steady      30.46       7.77      22.69     18
          15     steady      29.72       7.07      22.65     18
  backward_step: 16 micro-batch(es), avg_comp=22.79 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      38.36      15.60      22.76     12
           1     steady      38.67      15.90      22.77     12
           2     steady      38.87      16.12      22.75     12
           3     steady      38.95      16.14      22.81     12
           4     steady      38.80      16.17      22.63     12
           5     steady      38.90      16.11      22.79     12
           6     steady      38.88      15.93      22.95     12
           7     steady      38.86      16.08      22.78     12
           8     steady      38.86      16.13      22.73     12
           9     steady      38.77      16.06      22.71     12
          10     steady      38.46      15.67      22.79     12
          11   cooldown      38.27      15.43      22.84     12
          12   cooldown      38.46      15.62      22.84     12
          13   cooldown      38.24      15.39      22.85     12
          14   cooldown      38.24      15.37      22.87     12
          15   cooldown      38.30      15.53      22.77     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.32 ms

--- Rank 6 (PP stage 3/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.88 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.87       7.78      24.09     18
           1     warmup      31.11       7.98      23.13     18
           2     warmup      30.68       8.04      22.64     18
           3     warmup      30.43       7.94      22.49     18
           4     steady      30.81       8.14      22.67     18
           5     steady      30.28       7.50      22.78     18
           6     steady      31.61       8.83      22.78     18
           7     steady      30.72       7.99      22.73     18
           8     steady      31.22       8.34      22.88     18
           9     steady      30.41       7.76      22.65     18
          10     steady      31.07       8.18      22.89     18
          11     steady      30.58       7.89      22.69     18
          12     steady      30.82       7.47      23.35     18
          13     steady      30.59       7.94      22.65     18
          14     steady      30.53       7.64      22.89     18
          15     steady      30.77       7.99      22.78     18
  backward_step: 16 micro-batch(es), avg_comp=24.32 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      42.56      13.22      29.34     12
           1     steady      41.07      17.17      23.90     12
           2     steady      41.09      17.04      24.05     12
           3     steady      41.29      16.66      24.63     12
           4     steady      41.21      17.09      24.12     12
           5     steady      41.09      16.72      24.37     12
           6     steady      41.14      17.05      24.09     12
           7     steady      41.66      16.60      25.06     12
           8     steady      41.10      17.06      24.04     12
           9     steady      41.01      16.66      24.35     12
          10     steady      40.77      17.28      23.49     12
          11     steady      40.54      16.88      23.66     12
          12   cooldown      40.69      17.12      23.57     12
          13   cooldown      40.68      17.12      23.56     12
          14   cooldown      40.64      17.14      23.50     12
          15   cooldown      40.56      17.11      23.45     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.41 ms

--- Rank 7 (PP stage 3/7) ---
  forward_step: 16 micro-batch(es), avg_comp=24.91 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.90       6.19      25.71     18
           1     warmup      31.42       6.95      24.47     18
           2     warmup      30.88       6.03      24.85     18
           3     warmup      30.61       6.18      24.43     18
           4     steady      30.83       6.02      24.81     18
           5     steady      30.57       6.04      24.53     18
           6     steady      32.22       6.77      25.45     18
           7     steady      31.42       6.66      24.76     18
           8     steady      31.96       7.20      24.76     18
           9     steady      31.18       6.72      24.46     18
          10     steady      31.80       6.94      24.86     18
          11     steady      31.03       6.27      24.76     18
          12     steady      32.38       6.65      25.73     18
          13     steady      31.38       6.60      24.78     18
          14     steady      31.90       6.42      25.48     18
          15     steady      31.13       6.33      24.80     18
  backward_step: 16 micro-batch(es), avg_comp=25.48 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      42.57      16.86      25.71     12
           1     steady      40.71      15.20      25.51     12
           2     steady      40.41      14.99      25.42     12
           3     steady      40.81      15.40      25.41     12
           4     steady      40.78      15.24      25.54     12
           5     steady      40.29      14.87      25.42     12
           6     steady      40.78      15.21      25.57     12
           7     steady      41.20      15.81      25.39     12
           8     steady      40.90      15.36      25.54     12
           9     steady      40.39      14.99      25.40     12
          10     steady      40.95      15.47      25.48     12
          11     steady      40.19      14.82      25.37     12
          12   cooldown      40.76      15.37      25.39     12
          13   cooldown      40.82      15.26      25.56     12
          14   cooldown      40.76      15.15      25.61     12
          15   cooldown      40.96      15.59      25.37     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.34 ms

--- Rank 8 (PP stage 4/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.28 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.11       7.73      23.38     18
           1     warmup      29.12       7.32      21.80     18
           2     warmup      29.49       7.32      22.17     18
           3     steady      29.40       7.32      22.08     18
           4     steady      29.72       7.22      22.50     18
           5     steady      29.46       7.26      22.20     18
           6     steady      29.93       7.69      22.24     18
           7     steady      29.42       7.26      22.16     18
           8     steady      30.80       8.36      22.44     18
           9     steady      29.39       7.29      22.10     18
          10     steady      29.82       7.07      22.75     18
          11     steady      29.44       7.29      22.15     18
          12     steady      29.44       7.32      22.12     18
          13     steady      29.41       7.34      22.07     18
          14     steady      29.42       7.30      22.12     18
          15     steady      29.57       7.30      22.27     18
  backward_step: 16 micro-batch(es), avg_comp=21.39 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      38.21      16.72      21.49     12
           1     steady      38.44      16.53      21.91     12
           2     steady      38.38      17.33      21.05     12
           3     steady      38.46      16.51      21.95     12
           4     steady      38.21      17.21      21.00     12
           5     steady      38.56      16.33      22.23     12
           6     steady      38.20      17.00      21.20     12
           7     steady      38.43      16.40      22.03     12
           8     steady      38.29      17.28      21.01     12
           9     steady      38.38      16.25      22.13     12
          10     steady      37.84      16.73      21.11     12
          11     steady      37.82      16.79      21.03     12
          12     steady      37.80      16.74      21.06     12
          13   cooldown      37.75      16.81      20.94     12
          14   cooldown      37.74      16.73      21.01     12
          15   cooldown      37.82      16.76      21.06     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.30 ms

--- Rank 9 (PP stage 4/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.53 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.14       7.73      23.41     18
           1     warmup      29.13       6.94      22.19     18
           2     warmup      29.44       7.20      22.24     18
           3     steady      29.49       7.09      22.40     18
           4     steady      29.89       7.10      22.79     18
           5     steady      29.58       7.15      22.43     18
           6     steady      30.01       7.56      22.45     18
           7     steady      29.60       7.28      22.32     18
           8     steady      30.96       8.39      22.57     18
           9     steady      29.42       7.07      22.35     18
          10     steady      29.95       7.48      22.47     18
          11     steady      29.53       7.12      22.41     18
          12     steady      30.20       7.24      22.96     18
          13     steady      29.44       6.90      22.54     18
          14     steady      29.80       7.29      22.51     18
          15     steady      29.51       7.04      22.47     18
  backward_step: 16 micro-batch(es), avg_comp=22.39 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      38.16      14.93      23.23     12
           1     steady      38.42      16.25      22.17     12
           2     steady      38.35      15.39      22.96     12
           3     steady      38.41      16.22      22.19     12
           4     steady      38.20      15.48      22.72     12
           5     steady      38.60      16.28      22.32     12
           6     steady      38.15      15.34      22.81     12
           7     steady      38.39      16.33      22.06     12
           8     steady      38.27      15.47      22.80     12
           9     steady      38.54      16.31      22.23     12
          10     steady      37.78      15.55      22.23     12
          11     steady      37.77      15.63      22.14     12
          12     steady      37.76      15.69      22.07     12
          13   cooldown      37.80      15.70      22.10     12
          14   cooldown      37.69      15.57      22.12     12
          15   cooldown      37.70      15.56      22.14     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.29 ms

--- Rank 10 (PP stage 5/7) ---
  forward_step: 16 micro-batch(es), avg_comp=23.41 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.41       6.49      24.92     18
           1     warmup      29.76       6.51      23.25     18
           2     steady      29.79       6.60      23.19     18
           3     steady      29.98       6.66      23.32     18
           4     steady      29.75       6.49      23.26     18
           5     steady      30.28       6.88      23.40     18
           6     steady      29.83       6.51      23.32     18
           7     steady      30.12       6.94      23.18     18
           8     steady      29.85       6.49      23.36     18
           9     steady      30.35       6.98      23.37     18
          10     steady      29.90       6.50      23.40     18
          11     steady      30.17       6.86      23.31     18
          12     steady      29.84       6.46      23.38     18
          13     steady      30.21       6.90      23.31     18
          14     steady      29.75       6.47      23.28     18
          15     steady      30.14       6.86      23.28     18
  backward_step: 16 micro-batch(es), avg_comp=23.23 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      37.90      14.37      23.53     12
           1     steady      37.94      14.75      23.19     12
           2     steady      38.00      14.63      23.37     12
           3     steady      38.01      14.74      23.27     12
           4     steady      37.95      14.71      23.24     12
           5     steady      37.96      14.81      23.15     12
           6     steady      37.97      14.82      23.15     12
           7     steady      38.25      14.65      23.60     12
           8     steady      38.02      14.83      23.19     12
           9     steady      38.02      14.77      23.25     12
          10     steady      38.01      14.74      23.27     12
          11     steady      37.97      14.81      23.16     12
          12     steady      37.97      14.81      23.16     12
          13     steady      37.88      14.87      23.01     12
          14   cooldown      37.93      14.81      23.12     12
          15   cooldown      38.01      14.97      23.04     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.33 ms

--- Rank 11 (PP stage 5/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.75 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.45       7.52      23.93     18
           1     warmup      29.78       7.28      22.50     18
           2     steady      29.80       7.26      22.54     18
           3     steady      29.93       7.26      22.67     18
           4     steady      29.72       7.06      22.66     18
           5     steady      30.28       7.51      22.77     18
           6     steady      29.75       7.13      22.62     18
           7     steady      30.09       7.25      22.84     18
           8     steady      29.88       7.13      22.75     18
           9     steady      30.31       7.65      22.66     18
          10     steady      29.85       7.19      22.66     18
          11     steady      30.13       7.47      22.66     18
          12     steady      29.82       7.12      22.70     18
          13     steady      30.15       7.41      22.74     18
          14     steady      29.72       7.10      22.62     18
          15     steady      30.12       7.46      22.66     18
  backward_step: 16 micro-batch(es), avg_comp=22.23 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      37.82      15.31      22.51     12
           1     steady      37.95      15.76      22.19     12
           2     steady      37.92      15.62      22.30     12
           3     steady      38.05      15.71      22.34     12
           4     steady      37.88      15.62      22.26     12
           5     steady      37.94      15.78      22.16     12
           6     steady      37.94      15.69      22.25     12
           7     steady      38.26      15.85      22.41     12
           8     steady      37.91      15.67      22.24     12
           9     steady      37.95      15.59      22.36     12
          10     steady      37.94      15.71      22.23     12
          11     steady      37.94      15.73      22.21     12
          12     steady      37.96      15.73      22.23     12
          13     steady      37.84      15.79      22.05     12
          14   cooldown      37.88      15.84      22.04     12
          15   cooldown      37.89      15.93      21.96     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.34 ms

--- Rank 12 (PP stage 6/7) ---
  forward_step: 16 micro-batch(es), avg_comp=22.76 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.55       7.34      24.21     18
           1     steady      29.79       7.36      22.43     18
           2     steady      30.18       7.38      22.80     18
           3     steady      30.45       7.99      22.46     18
           4     steady      30.17       7.68      22.49     18
           5     steady      29.95       7.58      22.37     18
           6     steady      30.57       7.68      22.89     18
           7     steady      30.01       7.46      22.55     18
           8     steady      30.24       7.44      22.80     18
           9     steady      31.66       8.30      23.36     18
          10     steady      30.45       7.39      23.06     18
          11     steady      30.04       7.55      22.49     18
          12     steady      30.28       7.41      22.87     18
          13     steady      30.01       7.67      22.34     18
          14     steady      30.04       7.25      22.79     18
          15     steady      29.82       7.51      22.31     18
  backward_step: 16 micro-batch(es), avg_comp=22.19 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      39.56      17.00      22.56     12
           1     steady      39.67      17.46      22.21     12
           2     steady      39.71      17.56      22.15     12
           3     steady      39.78      17.64      22.14     12
           4     steady      39.78      17.57      22.21     12
           5     steady      39.74      17.61      22.13     12
           6     steady      39.81      17.56      22.25     12
           7     steady      40.90      18.67      22.23     12
           8     steady      39.98      17.61      22.37     12
           9     steady      39.78      17.63      22.15     12
          10     steady      39.69      17.56      22.13     12
          11     steady      39.71      17.60      22.11     12
          12     steady      39.67      17.54      22.13     12
          13     steady      39.69      17.61      22.08     12
          14     steady      39.67      17.66      22.01     12
          15   cooldown      39.70      17.59      22.11     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.29 ms

--- Rank 13 (PP stage 6/7) ---
  forward_step: 16 micro-batch(es), avg_comp=24.14 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     warmup      31.59       6.75      24.84     18
           1     steady      29.79       6.07      23.72     18
           2     steady      30.10       6.21      23.89     18
           3     steady      30.50       6.10      24.40     18
           4     steady      30.09       6.32      23.77     18
           5     steady      30.60       6.10      24.50     18
           6     steady      30.07       6.25      23.82     18
           7     steady      30.33       6.40      23.93     18
           8     steady      30.22       6.25      23.97     18
           9     steady      32.00       7.65      24.35     18
          10     steady      30.34       6.35      23.99     18
          11     steady      30.78       6.07      24.71     18
          12     steady      30.22       6.30      23.92     18
          13     steady      30.61       6.06      24.55     18
          14     steady      30.05       6.36      23.69     18
          15     steady      30.41       6.17      24.24     18
  backward_step: 16 micro-batch(es), avg_comp=24.78 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      39.50      14.85      24.65     12
           1     steady      39.58      14.95      24.63     12
           2     steady      39.65      14.97      24.68     12
           3     steady      39.65      14.96      24.69     12
           4     steady      39.65      15.02      24.63     12
           5     steady      39.64      14.95      24.69     12
           6     steady      39.71      15.02      24.69     12
           7     steady      40.91      14.89      26.02     12
           8     steady      39.97      14.90      25.07     12
           9     steady      39.69      14.93      24.76     12
          10     steady      39.62      14.96      24.66     12
          11     steady      39.60      14.98      24.62     12
          12     steady      39.55      14.95      24.60     12
          13     steady      39.58      14.93      24.65     12
          14     steady      39.58      14.94      24.64     12
          15   cooldown      39.73      14.95      24.78     12
  optimizer_step: 1 micro-batch(es), avg_comp=30.29 ms

--- Rank 14 (PP stage 7/7) ---
  forward_step: 16 micro-batch(es), avg_comp=28.86 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      38.13       8.52      29.61     18
           1     steady      36.20       7.32      28.88     18
           2     steady      36.18       7.07      29.11     18
           3     steady      36.90       7.47      29.43     18
           4     steady      35.94       7.10      28.84     18
           5     steady      36.77       7.40      29.37     18
           6     steady      35.93       7.01      28.92     18
           7     steady      37.13       7.75      29.38     18
           8     steady      36.28       7.22      29.06     18
           9     steady      37.24       7.78      29.46     18
          10     steady      35.82       7.08      28.74     18
          11     steady      36.77       7.40      29.37     18
          12     steady      35.22       7.43      27.79     18
          13     steady      35.36       7.52      27.84     18
          14     steady      35.63       7.48      28.15     18
          15     steady      35.24       7.44      27.80     18
  backward_step: 16 micro-batch(es), avg_comp=21.68 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      44.03      22.33      21.70     12
           1     steady      43.86      22.24      21.62     12
           2     steady      43.97      22.27      21.70     12
           3     steady      44.25      22.48      21.77     12
           4     steady      44.11      22.54      21.57     12
           5     steady      43.94      22.26      21.68     12
           6     steady      43.94      22.32      21.62     12
           7     steady      44.12      22.45      21.67     12
           8     steady      44.91      23.10      21.81     12
           9     steady      44.01      22.27      21.74     12
          10     steady      44.08      22.34      21.74     12
          11     steady      43.93      22.33      21.60     12
          12     steady      44.04      22.32      21.72     12
          13     steady      43.95      22.37      21.58     12
          14     steady      43.89      22.24      21.65     12
          15     steady      43.91      22.23      21.68     12
  optimizer_step: 1 micro-batch(es), avg_comp=35.99 ms

--- Rank 15 (PP stage 7/7) ---
  forward_step: 16 micro-batch(es), avg_comp=29.92 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      37.86       6.14      31.72     18
           1     steady      36.19       6.07      30.12     18
           2     steady      36.13       6.42      29.71     18
           3     steady      36.93       6.27      30.66     18
           4     steady      35.86       6.42      29.44     18
           5     steady      36.73       6.14      30.59     18
           6     steady      35.86       6.30      29.56     18
           7     steady      37.12       6.39      30.73     18
           8     steady      36.31       6.61      29.70     18
           9     steady      37.24       7.10      30.14     18
          10     steady      35.73       6.08      29.65     18
          11     steady      36.81       6.20      30.61     18
          12     steady      35.21       6.33      28.88     18
          13     steady      35.28       6.22      29.06     18
          14     steady      35.59       6.31      29.28     18
          15     steady      35.21       6.31      28.90     18
  backward_step: 16 micro-batch(es), avg_comp=23.76 ms
    batch_id   mg_state  total(ms)   comm(ms)   comp(ms)  #comm
           0     steady      44.00      19.90      24.10     12
           1     steady      43.83      20.23      23.60     12
           2     steady      43.94      20.35      23.59     12
           3     steady      44.16      20.27      23.89     12
           4     steady      44.08      20.23      23.85     12
           5     steady      43.86      20.24      23.62     12
           6     steady      43.89      20.21      23.68     12
           7     steady      43.98      20.39      23.59     12
           8     steady      44.75      20.24      24.51     12
           9     steady      43.90      20.22      23.68     12
          10     steady      44.04      20.18      23.86     12
          11     steady      43.83      20.20      23.63     12
          12     steady      43.96      20.22      23.74     12
          13     steady      43.92      20.20      23.72     12
          14     steady      43.83      20.29      23.54     12
          15     steady      43.83      20.35      23.48     12
  optimizer_step: 1 micro-batch(es), avg_comp=36.09 ms

==================================================================================================================================
PART 2: Summary Comparison — Scaling Comp vs Distributed Comp (avg across micro-batches)
         Distributed comp = total_duration - sum(comm_sub_op_durations)
==================================================================================================================================

Rank  PP |     S_fwd     D_fwd      Δfwd     err% |     S_bwd     D_bwd      Δbwd     err% |     S_opt     D_opt      Δopt     err%
-----------------------------------------------------------------------------------------------------------------------------------
   0   0 |     28.67     22.73      5.94   26.13% |     33.93     25.63      8.30   32.40% |     41.38     36.04      5.34   14.82%
   1   0 |     30.76     23.62      7.14   30.20% |     37.26     26.59     10.67   40.15% |     41.55     35.92      5.63   15.67%
   2   1 |     28.99     23.91      5.08   21.26% |     31.83     23.78      8.05   33.87% |     35.99     30.42      5.57   18.31%
   3   1 |     29.77     22.92      6.85   29.90% |     33.47     22.23     11.24   50.55% |     35.63     30.36      5.27   17.36%
   4   2 |     33.55     22.53     11.02   48.94% |     31.44     21.41     10.03   46.84% |     35.77     30.34      5.43   17.90%
   5   2 |     31.56     22.75      8.81   38.71% |     33.87     22.79     11.08   48.62% |     35.73     30.32      5.41   17.84%
   6   3 |     28.23     22.88      5.35   23.38% |     31.05     24.32      6.73   27.65% |     35.75     30.41      5.34   17.56%
   7   3 |     29.84     24.91      4.93   19.77% |     33.39     25.48      7.91   31.04% |     35.75     30.34      5.41   17.83%
   8   4 |     31.70     22.28      9.42   42.25% |     31.47     21.39     10.08   47.14% |     35.82     30.30      5.52   18.22%
   9   4 |     30.10     22.53      7.57   33.59% |     33.49     22.39     11.10   49.55% |     35.71     30.29      5.42   17.89%
  10   5 |     28.62     23.41      5.21   22.27% |     31.13     23.23      7.90   34.00% |     35.67     30.33      5.34   17.61%
  11   5 |     30.31     22.75      7.56   33.24% |     33.65     22.23     11.42   51.35% |     35.81     30.34      5.47   18.03%
  12   6 |     28.68     22.76      5.92   25.99% |     31.21     22.19      9.02   40.68% |     35.78     30.29      5.49   18.12%
  13   6 |     31.11     24.14      6.97   28.86% |     33.58     24.78      8.80   35.52% |     35.73     30.29      5.44   17.96%
  14   7 |     35.95     28.86      7.09   24.57% |     36.40     21.68     14.72   67.91% |     35.93     35.99     -0.06   -0.17%
  15   7 |     37.26     29.92      7.34   24.52% |     39.07     23.76     15.31   64.47% |     41.40     36.09      5.31   14.71%
-----------------------------------------------------------------------------------------------------------------------------------

Aggregation Statistics (relative error %):
                         mean   |mean|   median |median| max|err| min|err|
        forward_step   29.60%   29.60%   28.86%   28.86%   48.94%   19.77%
       backward_step   43.86%   43.86%   46.84%   46.84%   67.91%   27.65%
      optimizer_step   16.23%   16.25%   17.84%   17.84%   18.31%    0.17%

==================================================================================================================================
PART 3: Per-PP-Stage Statistics
==================================================================================================================================

  PP Stage 0 (ranks: [0, 1]):
                           mean   |mean|   median max|err|
          forward_step   28.17%   28.17%   30.20%   30.20%
         backward_step   36.27%   36.27%   40.15%   40.15%
        optimizer_step   15.25%   15.25%   15.67%   15.67%

  PP Stage 1 (ranks: [2, 3]):
                           mean   |mean|   median max|err|
          forward_step   25.58%   25.58%   29.90%   29.90%
         backward_step   42.21%   42.21%   50.55%   50.55%
        optimizer_step   17.83%   17.83%   18.31%   18.31%

  PP Stage 2 (ranks: [4, 5]):
                           mean   |mean|   median max|err|
          forward_step   43.83%   43.83%   48.94%   48.94%
         backward_step   47.73%   47.73%   48.62%   48.62%
        optimizer_step   17.87%   17.87%   17.90%   17.90%

  PP Stage 3 (ranks: [6, 7]):
                           mean   |mean|   median max|err|
          forward_step   21.57%   21.57%   23.38%   23.38%
         backward_step   29.35%   29.35%   31.04%   31.04%
        optimizer_step   17.70%   17.70%   17.83%   17.83%

  PP Stage 4 (ranks: [8, 9]):
                           mean   |mean|   median max|err|
          forward_step   37.92%   37.92%   42.25%   42.25%
         backward_step   48.35%   48.35%   49.55%   49.55%
        optimizer_step   18.06%   18.06%   18.22%   18.22%

  PP Stage 5 (ranks: [10, 11]):
                           mean   |mean|   median max|err|
          forward_step   27.75%   27.75%   33.24%   33.24%
         backward_step   42.67%   42.67%   51.35%   51.35%
        optimizer_step   17.82%   17.82%   18.03%   18.03%

  PP Stage 6 (ranks: [12, 13]):
                           mean   |mean|   median max|err|
          forward_step   27.42%   27.42%   28.86%   28.86%
         backward_step   38.10%   38.10%   40.68%   40.68%
        optimizer_step   18.04%   18.04%   18.12%   18.12%

  PP Stage 7 (ranks: [14, 15]):
                           mean   |mean|   median max|err|
          forward_step   24.55%   24.55%   24.57%   24.57%
         backward_step   66.19%   66.19%   67.91%   67.91%
        optimizer_step    7.27%    7.44%   14.71%   14.71%

==================================================================================================================================
PART 4: Pass/Fail Check (threshold = 5.0% relative error)
==================================================================================================================================
  FAIL: Rank 0 (PP0) fwd: |26.13%| > 5.0%
  FAIL: Rank 0 (PP0) bwd: |32.40%| > 5.0%
  FAIL: Rank 0 (PP0) opt: |14.82%| > 5.0%
  FAIL: Rank 1 (PP0) fwd: |30.20%| > 5.0%
  FAIL: Rank 1 (PP0) bwd: |40.15%| > 5.0%
  FAIL: Rank 1 (PP0) opt: |15.67%| > 5.0%
  FAIL: Rank 2 (PP1) fwd: |21.26%| > 5.0%
  FAIL: Rank 2 (PP1) bwd: |33.87%| > 5.0%
  FAIL: Rank 2 (PP1) opt: |18.31%| > 5.0%
  FAIL: Rank 3 (PP1) fwd: |29.90%| > 5.0%
  FAIL: Rank 3 (PP1) bwd: |50.55%| > 5.0%
  FAIL: Rank 3 (PP1) opt: |17.36%| > 5.0%
  FAIL: Rank 4 (PP2) fwd: |48.94%| > 5.0%
  FAIL: Rank 4 (PP2) bwd: |46.84%| > 5.0%
  FAIL: Rank 4 (PP2) opt: |17.90%| > 5.0%
  FAIL: Rank 5 (PP2) fwd: |38.71%| > 5.0%
  FAIL: Rank 5 (PP2) bwd: |48.62%| > 5.0%
  FAIL: Rank 5 (PP2) opt: |17.84%| > 5.0%
  FAIL: Rank 6 (PP3) fwd: |23.38%| > 5.0%
  FAIL: Rank 6 (PP3) bwd: |27.65%| > 5.0%
  FAIL: Rank 6 (PP3) opt: |17.56%| > 5.0%
  FAIL: Rank 7 (PP3) fwd: |19.77%| > 5.0%
  FAIL: Rank 7 (PP3) bwd: |31.04%| > 5.0%
  FAIL: Rank 7 (PP3) opt: |17.83%| > 5.0%
  FAIL: Rank 8 (PP4) fwd: |42.25%| > 5.0%
  FAIL: Rank 8 (PP4) bwd: |47.14%| > 5.0%
  FAIL: Rank 8 (PP4) opt: |18.22%| > 5.0%
  FAIL: Rank 9 (PP4) fwd: |33.59%| > 5.0%
  FAIL: Rank 9 (PP4) bwd: |49.55%| > 5.0%
  FAIL: Rank 9 (PP4) opt: |17.89%| > 5.0%
  FAIL: Rank 10 (PP5) fwd: |22.27%| > 5.0%
  FAIL: Rank 10 (PP5) bwd: |34.00%| > 5.0%
  FAIL: Rank 10 (PP5) opt: |17.61%| > 5.0%
  FAIL: Rank 11 (PP5) fwd: |33.24%| > 5.0%
  FAIL: Rank 11 (PP5) bwd: |51.35%| > 5.0%
  FAIL: Rank 11 (PP5) opt: |18.03%| > 5.0%
  FAIL: Rank 12 (PP6) fwd: |25.99%| > 5.0%
  FAIL: Rank 12 (PP6) bwd: |40.68%| > 5.0%
  FAIL: Rank 12 (PP6) opt: |18.12%| > 5.0%
  FAIL: Rank 13 (PP6) fwd: |28.86%| > 5.0%
  FAIL: Rank 13 (PP6) bwd: |35.52%| > 5.0%
  FAIL: Rank 13 (PP6) opt: |17.96%| > 5.0%
  FAIL: Rank 14 (PP7) fwd: |24.57%| > 5.0%
  FAIL: Rank 14 (PP7) bwd: |67.91%| > 5.0%
  FAIL: Rank 15 (PP7) fwd: |24.52%| > 5.0%
  FAIL: Rank 15 (PP7) bwd: |64.47%| > 5.0%
  FAIL: Rank 15 (PP7) opt: |14.71%| > 5.0%

  Result: 1/48 checks passed (2.1%)
  ❌ 47 checks FAILED


==================================================================================================================================
CROSS-CONFIGURATION SUMMARY
==================================================================================================================================

                 Configuration | |fwd| mean  |fwd| med   fwd max | |bwd| mean  |bwd| med   bwd max | |opt| mean  |opt| med   opt max |   Pass
---------------------------------------------------------------------------------------------------------------------------------------------
         Config A (PP=2, EP=8) |     44.09%     42.57%   118.67% |     36.13%     40.06%    63.33% |      8.03%     11.88%    12.64% |   6/48
         Config B (PP=4, EP=4) |     27.25%     27.48%    37.55% |     38.28%     37.39%    59.58% |     14.57%     17.95%    18.78% |   3/48
         Config C (PP=8, EP=2) |     29.60%     28.86%    48.94% |     43.86%     46.84%    67.91% |     16.25%     17.84%    18.31% |   1/48
---------------------------------------------------------------------------------------------------------------------------------------------

Report saved to: /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM/tests/performance/../../task_memory/compare_qwen3_moe_comp_report.md
