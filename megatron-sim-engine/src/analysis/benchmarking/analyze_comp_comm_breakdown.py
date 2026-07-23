#!/usr/bin/env python3

import json
import os
import glob
from datetime import datetime
from pathlib import Path
import pandas as pd

def analyze_operation_logs():
    """
    Analyze computation and communication operation logs to understand
    the breakdown and identify potential anomalies in simulation results.
    """

    print("=" * 80)
    print("Computation vs Communication Breakdown Analysis")
    print("=" * 80)

    # Find the most recent operation log files
    project_root = Path(__file__).resolve().parents[3]
    timeline_log_dir = project_root / "log" / "timeline_op_log"

    # Get recent SIMULATE_MODE files (last 10)
    comp_files = sorted(glob.glob(str(timeline_log_dir / "comp_operations_SIMULATE_MODE_*.json")))[-10:]
    comm_files = sorted(glob.glob(str(timeline_log_dir / "comm_operations_SIMULATE_MODE_*.json")))[-10:]

    print(f"Found {len(comp_files)} recent computation log files")
    print(f"Found {len(comm_files)} recent communication log files")
    print()

    results = []

    # Process each pair of comp/comm files
    for comp_file, comm_file in zip(comp_files, comm_files):
        # Extract timestamp from filename
        comp_timestamp = comp_file.split('_')[-1].replace('.json', '')
        comm_timestamp = comm_file.split('_')[-1].replace('.json', '')

        if comp_timestamp != comm_timestamp:
            print(f"Warning: Timestamp mismatch - {comp_timestamp} vs {comm_timestamp}")
            continue

        print(f"Analyzing logs for timestamp: {comp_timestamp}")

        # Load computation data
        try:
            with open(comp_file, 'r') as f:
                comp_data = json.load(f)
        except Exception as e:
            print(f"Error loading {comp_file}: {e}")
            continue

        # Load communication data
        try:
            with open(comm_file, 'r') as f:
                comm_data = json.load(f)
        except Exception as e:
            print(f"Error loading {comm_file}: {e}")
            continue

        # Extract metadata
        comp_total_time = comp_data['metadata']['total_comp_time']
        comm_total_time = comm_data['metadata']['total_comm_time']
        comp_operations = comp_data['metadata']['total_operations']
        comm_operations = comm_data['metadata']['total_operations']

        # Calculate ratios
        total_time = comp_total_time + comm_total_time
        comp_ratio = comp_total_time / total_time if total_time > 0 else 0
        comm_ratio = comm_total_time / total_time if total_time > 0 else 0

        # Convert milliseconds to seconds
        comp_time_sec = comp_total_time / 1000
        comm_time_sec = comm_total_time / 1000
        total_time_sec = total_time / 1000

        result = {
            'timestamp': comp_timestamp,
            'comp_time_ms': comp_total_time,
            'comm_time_ms': comm_total_time,
            'comp_time_sec': comp_time_sec,
            'comm_time_sec': comm_time_sec,
            'total_time_sec': total_time_sec,
            'comp_ratio': comp_ratio,
            'comm_ratio': comm_ratio,
            'comp_operations': comp_operations,
            'comm_operations': comm_operations
        }

        results.append(result)

        print(f"  Computation time: {comp_time_sec:.3f}s ({comp_ratio:.1%})")
        print(f"  Communication time: {comm_time_sec:.3f}s ({comm_ratio:.1%})")
        print(f"  Total time: {total_time_sec:.3f}s")
        print(f"  Comp operations: {comp_operations}, Comm operations: {comm_operations}")
        print()

    # Create summary table
    if results:
        df = pd.DataFrame(results)

        print("=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Timestamp':<15} {'Comp(s)':<8} {'Comm(s)':<8} {'Total(s)':<9} {'Comp%':<6} {'Comm%':<6} {'C/C Ratio':<10}")
        print("-" * 80)

        for _, row in df.iterrows():
            cc_ratio = row['comp_time_sec'] / row['comm_time_sec'] if row['comm_time_sec'] > 0 else float('inf')
            print(f"{row['timestamp']:<15} {row['comp_time_sec']:<8.2f} {row['comm_time_sec']:<8.2f} "
                  f"{row['total_time_sec']:<9.2f} {row['comp_ratio']:<6.1%} {row['comm_ratio']:<6.1%} "
                  f"{cc_ratio:<10.2f}")

        print()
        print("ANALYSIS:")
        print(f"Average computation ratio: {df['comp_ratio'].mean():.1%}")
        print(f"Average communication ratio: {df['comm_ratio'].mean():.1%}")
        print(f"Std dev computation ratio: {df['comp_ratio'].std():.1%}")
        print(f"Std dev communication ratio: {df['comm_ratio'].std():.1%}")

        # Identify anomalies
        comp_mean = df['comp_ratio'].mean()
        comp_std = df['comp_ratio'].std()

        print("\nANOMALY DETECTION:")
        for _, row in df.iterrows():
            if abs(row['comp_ratio'] - comp_mean) > 2 * comp_std:
                print(f"⚠️  ANOMALY: {row['timestamp']} has unusual comp ratio: {row['comp_ratio']:.1%}")

        # Save results to CSV
        output_file = project_root / "log" / "comp_comm_analysis.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    analyze_operation_logs()
