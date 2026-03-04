#!/usr/bin/env python3
"""Count CUDA kernels during forward and backward of scaling mode.

This script patches the scaling mode training loop to capture kernel profiles
using torch.profiler, then counts specific kernel types (fmha, rmsnorm, etc.)
to verify the backward has the expected number of kernels.

Usage:
    cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
    CUDA_VISIBLE_DEVICES=0 python tests/unit/test_kernel_count_scaling.py \
        --rank 4 2>&1 | grep -E 'KERNEL_COUNT|fmha|rmsnorm|Summary'
"""
import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=4, help="Fake rank to simulate")
    args = parser.parse_args()

    # Build the command to run scaling mode for one rank with kernel counting
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "MODE": "scaling",
        "MODEL_PROFILE": "smoke",
        "TRANSFORMER_IMPL": "local",
        "TRAIN_ITERS": "3",
        "TRACE_START": "1",
        "SCALING_MIN_WARMUP_ITERS": "1",
        "SCALING_PROFILE_ITERS": "1",
        "SCALING_STRICT_GRAD_REPLAY": "0",
        "DO_TRACE": "False",
        "SCALING_FAKE_RANK_ORDER": str(args.rank),
        "DIAG_KERNEL_COUNT": "1",  # Our diagnostic flag
    })

    script = os.path.join(project_root, "examples", "pretrain_deepseek_v3_moe.sh")
    result = subprocess.run(
        ["bash", script],
        env=env,
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    # Print relevant lines
    for line in result.stdout.split('\n'):
        if any(kw in line for kw in ['KERNEL_COUNT', 'fmha', 'rmsnorm', 'Summary', 'DIAG', 'Error']):
            print(line)
    if result.returncode != 0:
        print(f"Process returned {result.returncode}")
        # Print last 20 lines of stderr
        stderr_lines = result.stderr.strip().split('\n')
        for line in stderr_lines[-20:]:
            print(f"STDERR: {line}")


if __name__ == "__main__":
    main()
