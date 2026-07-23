#!/usr/bin/env python3
"""
Root Cause Analysis for Scheduling Plan Mismatches

This script performs detailed analysis of the mismatched configuration
to identify the root cause of line count discrepancies.
"""

import re
from pathlib import Path

def analyze_trace_file(file_path):
    """Analyze a trace file to understand its structure"""
    operations = []
    batch_counts = {}
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            operations.append(line)
            
            # Extract batch_id if present
            batch_match = re.search(r'batch_id=(\d+)', line)
            if batch_match:
                batch_id = int(batch_match.group(1))
                batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1
    
    return {
        'total_lines': len(operations),
        'batch_counts': batch_counts,
        'unique_batches': len(batch_counts),
        'sample_operations': operations[:5] + ['...'] + operations[-5:] if len(operations) > 10 else operations
    }

def analyze_schedule_file(file_path):
    """Analyze a schedule file to understand its structure"""
    operations = []
    batch_counts = {}
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            operations.append(line)
            
            # Extract batch_id if present
            batch_match = re.search(r'batch_id=(\d+)', line)
            if batch_match:
                batch_id = int(batch_match.group(1))
                batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1
    
    return {
        'total_lines': len(operations),
        'batch_counts': batch_counts,
        'unique_batches': len(batch_counts),
        'sample_operations': operations[:5] + ['...'] + operations[-5:] if len(operations) > 10 else operations
    }

def main():
    """Analyze the mismatched configuration"""
    config_name = "pp4_tp1_exp2_expn4_dp4_nl8_hs4096_sl4096"
    base_dir = (
        Path(__file__).resolve().parents[3]
        / "simulation_inputs"
        / "megatron_operation_log"
        / "new_moe"
    )
    config_dir = base_dir / config_name
    
    print(f"Analyzing mismatched configuration: {config_name}")
    print("=" * 60)
    
    # Analyze trace files
    trace_dir = config_dir / "global_ranks_profile"
    trace_files = list(trace_dir.glob("*.txt"))
    
    print(f"\nTrace Files Analysis:")
    print("-" * 30)
    
    rank_analyses = {}
    for trace_file in sorted(trace_files):
        rank_match = re.search(r'rank(\d+)', trace_file.name)
        if rank_match:
            rank = int(rank_match.group(1))
            analysis = analyze_trace_file(trace_file)
            rank_analyses[rank] = analysis
            
            print(f"Rank {rank}: {analysis['total_lines']} lines, {analysis['unique_batches']} batches")
            if analysis['batch_counts']:
                batch_summary = ", ".join([f"batch{k}:{v}" for k, v in sorted(analysis['batch_counts'].items())])
                print(f"  Batch distribution: {batch_summary}")
    
    # Analyze schedule files
    schedule_dir = config_dir / "schedule"
    schedule_files = list(schedule_dir.glob("*.txt"))
    
    print(f"\nSchedule Files Analysis:")
    print("-" * 30)
    
    stage_analyses = {}
    for schedule_file in sorted(schedule_files):
        stage_match = re.search(r'stage(\d+)', schedule_file.name)
        if stage_match:
            stage = int(stage_match.group(1))
            analysis = analyze_schedule_file(schedule_file)
            stage_analyses[stage] = analysis
            
            print(f"Stage {stage}: {analysis['total_lines']} lines, {analysis['unique_batches']} batches")
            if analysis['batch_counts']:
                batch_summary = ", ".join([f"batch{k}:{v}" for k, v in sorted(analysis['batch_counts'].items())])
                print(f"  Batch distribution: {batch_summary}")
    
    # Compare Stage 0 with Rank 0 (they should correspond)
    print(f"\nDetailed Comparison: Stage 0 vs Rank 0")
    print("-" * 40)
    
    if 0 in rank_analyses and 0 in stage_analyses:
        rank0_analysis = rank_analyses[0]
        stage0_analysis = stage_analyses[0]
        
        print(f"Rank 0 lines: {rank0_analysis['total_lines']}")
        print(f"Stage 0 lines: {stage0_analysis['total_lines']}")
        print(f"Difference: {abs(rank0_analysis['total_lines'] - stage0_analysis['total_lines'])}")
        
        print(f"\nRank 0 batch distribution: {rank0_analysis['batch_counts']}")
        print(f"Stage 0 batch distribution: {stage0_analysis['batch_counts']}")
        
        print(f"\nRank 0 sample operations:")
        for op in rank0_analysis['sample_operations']:
            print(f"  {op}")
        
        print(f"\nStage 0 sample operations:")
        for op in stage0_analysis['sample_operations']:
            print(f"  {op}")
    
    # Configuration analysis
    print(f"\nConfiguration Analysis:")
    print("-" * 30)
    
    # Parse config parameters
    pattern = r'pp(\d+)_tp(\d+)_exp(\d+)_expn(\d+)_dp(\d+)_nl(\d+)_hs(\d+)_sl(\d+)'
    match = re.match(pattern, config_name)
    
    if match:
        pp, tp, exp, expn, dp, nl, hs, sl = map(int, match.groups())
        world_size = pp * tp * dp
        
        print(f"PP={pp}, TP={tp}, DP={dp}, EXP={exp}, EXPN={expn}")
        print(f"NL={nl}, HS={hs}, SL={sl}")
        print(f"World Size: {world_size}")
        print(f"Expected micro batches: {4 * pp} (4 × PP)")
        
        # Calculate expected operations per stage
        # This is a rough estimate based on typical pipeline patterns
        expected_ops_per_batch = 10  # Rough estimate
        expected_total_ops = 4 * pp * expected_ops_per_batch
        
        print(f"Estimated operations per stage: ~{expected_total_ops}")
    
    # Root cause analysis
    print(f"\nRoot Cause Analysis:")
    print("-" * 30)
    
    if 0 in rank_analyses and 0 in stage_analyses:
        rank0_batches = rank_analyses[0]['batch_counts']
        stage0_batches = stage_analyses[0]['batch_counts']
        
        if len(rank0_batches) != len(stage0_batches):
            print(f"❌ Batch count mismatch: Rank0 has {len(rank0_batches)} batches, Stage0 has {len(stage0_batches)} batches")
        
        if rank0_batches != stage0_batches:
            print(f"❌ Batch distribution mismatch")
            print(f"   Rank0 batches: {sorted(rank0_batches.keys())}")
            print(f"   Stage0 batches: {sorted(stage0_batches.keys())}")
        
        # Check if it's a sequence length issue
        if sl == 4096:
            print(f"⚠️ This configuration uses SL=4096, which might have different batch processing")
        
        # Check if it's related to expert parallelism
        if exp != expn:
            print(f"⚠️ Expert configuration: EXP={exp} != EXPN={expn}, might affect operation count")
    
    print(f"\nRecommendations:")
    print("-" * 20)
    print("1. Check if num_micro_batch calculation is correct (should be 4 × PP)")
    print("2. Verify trace_start parameter alignment")
    print("3. Check if sequence length affects batch processing")
    print("4. Verify expert parallelism configuration")

if __name__ == "__main__":
    main()
