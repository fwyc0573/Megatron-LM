#!/usr/bin/env python3

import os
import re
import glob
from collections import defaultdict, Counter

def analyze_microbatch_scheduling():
    """
    Analyze micro-batch scheduling across different PP=64 configurations
    to identify if micro-batch count differences explain the simulation anomaly.
    """

    print("=" * 80)
    print("MICRO-BATCH SCHEDULING ANALYSIS FOR PP=64 CONFIGURATIONS")
    print("=" * 80)

    # Configuration directories to analyze
    configs = {
        'pp64_tp8': 'simulation_inputs/megatron_operation_log/8192_sim/pp64_tp8_ep1_expnNone_dp16_nl96_hs20480_sl2048/schedule',
        'pp64_tp4': 'simulation_inputs/megatron_operation_log/8192_sim/pp64_tp4_ep1_expnNone_dp32_nl96_hs20480_sl2048/schedule',
        'pp64_tp2': 'simulation_inputs/megatron_operation_log/8192_sim/pp64_tp2_ep1_expnNone_dp64_nl96_hs20480_sl2048/schedule'
    }

    # Database profile data (individual operation times in ms)
    profile_data = {
        'pp64_tp8': {'forward_step': 10.53, 'backward_step': 27.65, 'optimizer_step': 28.11},
        'pp64_tp4': {'forward_step': 19.94, 'backward_step': 57.91, 'optimizer_step': 47.29},
        'pp64_tp2': {'forward_step': 37.65, 'backward_step': 105.54, 'optimizer_step': 92.88}
    }

    results = {}

    for config_name, schedule_dir in configs.items():
        print(f"\nAnalyzing {config_name}:")
        print("-" * 50)

        if not os.path.exists(schedule_dir):
            print(f"  ⚠️ Schedule directory not found: {schedule_dir}")
            continue

        # Analyze schedule files
        schedule_analysis = analyze_schedule_directory(schedule_dir)

        # Calculate theoretical total computation time
        profile = profile_data[config_name]
        theoretical_comp_time = calculate_theoretical_computation_time(
            schedule_analysis, profile
        )

        results[config_name] = {
            'schedule_analysis': schedule_analysis,
            'profile_data': profile,
            'theoretical_comp_time': theoretical_comp_time
        }

        # Display results
        print(f"  Schedule Analysis:")
        print(f"    Total micro-batches: {schedule_analysis['total_microbatches']}")
        print(f"    Forward steps: {schedule_analysis['forward_steps']}")
        print(f"    Backward steps: {schedule_analysis['backward_steps']}")
        print(f"    Optimizer steps: {schedule_analysis['optimizer_steps']}")
        print(f"    Stages analyzed: {schedule_analysis['stages_count']}")

        print(f"  Individual Operation Times (ms):")
        print(f"    Forward step: {profile['forward_step']:.2f}ms")
        print(f"    Backward step: {profile['backward_step']:.2f}ms")
        print(f"    Optimizer step: {profile['optimizer_step']:.2f}ms")

        print(f"  Theoretical Total Computation Time:")
        print(f"    Total forward time: {theoretical_comp_time['total_forward']:.2f}ms")
        print(f"    Total backward time: {theoretical_comp_time['total_backward']:.2f}ms")
        print(f"    Total optimizer time: {theoretical_comp_time['total_optimizer']:.2f}ms")
        print(f"    Grand total: {theoretical_comp_time['grand_total']:.2f}ms ({theoretical_comp_time['grand_total']/1000:.3f}s)")

    # Cross-configuration comparison
    print(f"\n" + "=" * 80)
    print("CROSS-CONFIGURATION MICRO-BATCH COMPARISON")
    print("=" * 80)

    print(f"{'Config':<12} {'Micro-batches':<15} {'Forward Ops':<12} {'Backward Ops':<13} {'Optimizer Ops':<14} {'Total Comp (s)':<15}")
    print("-" * 95)

    for config_name, data in results.items():
        sched = data['schedule_analysis']
        comp_time = data['theoretical_comp_time']['grand_total'] / 1000
        print(f"{config_name:<12} {sched['total_microbatches']:<15} {sched['forward_steps']:<12} "
              f"{sched['backward_steps']:<13} {sched['optimizer_steps']:<14} {comp_time:<15.3f}")

    # Micro-batch efficiency analysis
    print(f"\n" + "=" * 80)
    print("MICRO-BATCH EFFICIENCY ANALYSIS")
    print("=" * 80)

    if len(results) >= 2:
        # Compare micro-batch counts
        mb_counts = {config: data['schedule_analysis']['total_microbatches']
                    for config, data in results.items()}

        print("Micro-batch count comparison:")
        for config, count in mb_counts.items():
            print(f"  {config}: {count} micro-batches")

        # Check if micro-batch counts explain the performance difference
        if 'pp64_tp2' in mb_counts and 'pp64_tp8' in mb_counts:
            tp2_mb = mb_counts['pp64_tp2']
            tp8_mb = mb_counts['pp64_tp8']
            mb_ratio = tp2_mb / tp8_mb if tp8_mb > 0 else float('inf')

            print(f"\nMicro-batch ratio analysis:")
            print(f"  pp64_tp2 / pp64_tp8 micro-batches: {mb_ratio:.2f}")

            if mb_ratio < 1.0:
                print(f"  🎯 CRITICAL: pp64_tp2 has FEWER micro-batches than pp64_tp8!")
                print(f"     This could explain the apparent performance advantage!")
            elif mb_ratio > 1.0:
                print(f"  ✅ pp64_tp2 has more micro-batches (as expected)")
            else:
                print(f"  → pp64_tp2 and pp64_tp8 have equal micro-batches")

    return results

def analyze_schedule_directory(schedule_dir):
    """
    Analyze all schedule files in a directory to count operations.
    """
    schedule_files = glob.glob(os.path.join(schedule_dir, "stage*_scheduling_plan.txt"))

    total_stats = {
        'total_microbatches': 0,
        'forward_steps': 0,
        'backward_steps': 0,
        'optimizer_steps': 0,
        'stages_count': len(schedule_files),
        'batch_ids': set()
    }

    for schedule_file in schedule_files:
        try:
            with open(schedule_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Extract batch_id for micro-batch counting
                    batch_id_match = re.search(r'batch_id=(\d+)', line)
                    if batch_id_match:
                        batch_id = int(batch_id_match.group(1))
                        total_stats['batch_ids'].add(batch_id)

                    # Count different operation types
                    if 'forward_step(' in line:
                        total_stats['forward_steps'] += 1
                    elif 'backward_step(' in line:
                        total_stats['backward_steps'] += 1
                    elif 'optimizer_step(' in line:
                        total_stats['optimizer_steps'] += 1

        except Exception as e:
            print(f"    Error reading {schedule_file}: {e}")

    # Total micro-batches is the maximum batch_id + 1 (since batch_id starts from 0)
    if total_stats['batch_ids']:
        total_stats['total_microbatches'] = max(total_stats['batch_ids']) + 1

    return total_stats

def calculate_theoretical_computation_time(schedule_analysis, profile_data):
    """
    Calculate theoretical total computation time based on schedule and profile data.
    """
    total_forward = schedule_analysis['forward_steps'] * profile_data['forward_step']
    total_backward = schedule_analysis['backward_steps'] * profile_data['backward_step']
    total_optimizer = schedule_analysis['optimizer_steps'] * profile_data['optimizer_step']

    return {
        'total_forward': total_forward,
        'total_backward': total_backward,
        'total_optimizer': total_optimizer,
        'grand_total': total_forward + total_backward + total_optimizer
    }

if __name__ == "__main__":
    results = analyze_microbatch_scheduling()
