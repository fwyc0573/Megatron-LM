# SimAI Wall-clock Scaling: Parallel Configurations

## Fixed Parameters
- **TP = 8** (all scales)
- Model: 22B, micro_batch=1, seq_length=2048, GA=1

## Formal Scaling Configurations (11 Points)

| Total GPUs | TP | PP | DP  |
|------------|----|----|-----|
| 8          | 8  | 1  | 1   |
| 16         | 8  | 2  | 1   |
| 32         | 8  | 4  | 1   |
| 64         | 8  | 8  | 1   |
| 128        | 8  | 8  | 2   |
| 256        | 8  | 8  | 4   |
| 512        | 8  | 8  | 8   |
| 1024       | 8  | 8  | 16  |
| 2048       | 8  | 8  | 32  |
| 4096       | 8  | 8  | 64  |
| 8192       | 8  | 8  | 128 |

## PP Selection Policy
- Sensitivity analysis determined policy = **severe** (ratio >> 1.20 threshold)
- Formal phase uses **max feasible PP** at each scale
- PP upper bound = min(12, total_gpus / TP)
- DP = total_gpus / (TP × PP)


## new plan：Formal Scaling Configurations (11 Points)

| Total GPUs | TP | PP | DP  |
|------------|----|----|-----|
| 8          | 8  | 1  | 1   |
| 16         | 8  | 2  | 1   |
| 32         | 8  | 4  | 1   |
| 64         | 8  | 8  | 1   |
| 128        | 8  | 8  | 2   |
| 256        | 8  | 8  | 4   |
| 512        | 8  | 8  | 8   |
| 1024       | 8  | 16  | 8  |
| 2048       | 8  | 16  | 16  |
| 4096       | 8  | 16  | 32  |
| 8192       | 8  | 16  | 64 |
