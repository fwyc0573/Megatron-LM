#!/usr/bin/env python3
"""
Standalone Memory Trace Visualizer
This script provides memory trace visualization without importing the full Megatron environment.
"""

import sys
import os
import json
import glob
import re
import argparse

def _plot_allocated_memory_for_path(ax, json_path, label, style_kwargs):
    """Helper function to load, process, and plot data for a single trace file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not read or parse {json_path}: {e}")
        return False, None, None

    if not data:
        print(f"Warning: No data found in {json_path}")
        return False, None, None

    # Get the last iteration
    try:
        last_iter_key = str(max(map(int, data.keys())))
        iter_data = data[last_iter_key]
    except (ValueError, KeyError):
        print(f"Warning: Could not determine last iteration in {json_path}")
        return False, None, None

    samples = iter_data.get('samples', [])
    if not samples or 'allocated_memory_MB' not in samples[0]:
        print(f"Warning: No valid samples found for last iteration in {json_path}")
        return False, None, None
        
    timestamps = [s['timestamp_s'] for s in samples]
    allocated_mem = [s['allocated_memory_MB'] for s in samples]
    
    # Normalize timestamps to start from 0 for this iteration
    start_time = timestamps[0]
    relative_timestamps = [t - start_time for t in timestamps]
    
    ax.plot(relative_timestamps, allocated_mem, label=label, **style_kwargs)
    
    # Return theoretical and static memory if available
    theoretical_memory = iter_data.get('theoretical_memory_MB', -1)
    static_memory = iter_data.get('static_memory_MB', -1)
    return True, theoretical_memory, static_memory

def visualize_memory_comparison(paired_rank_files, output_dir='memory_plots', show_theoretical=False):
    """
    Visualizes a comparison of allocated memory for simulation vs. real runs,
    generating one plot per rank.

    Args:
        paired_rank_files (dict): A dict mapping rank_id to {'sim': path, 'real': path}.
        output_dir (str): Directory to save the output plot images.
        show_theoretical (bool): Whether to display theoretical memory lines. Default: False.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please install matplotlib to use the visualization function: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    for rank, paths in sorted(paired_rank_files.items()):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 9))
        
        has_plot_data = False
        theoretical_memories = {}
        static_memories = {}

        # Plot Real Run data if available
        if 'real' in paths:
            success, theoretical_mem, static_mem = _plot_allocated_memory_for_path(
                ax, paths['real'], 'Real Run', 
                {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8}
            )
            if success:
                has_plot_data = True
                if theoretical_mem > 0:
                    theoretical_memories['real'] = theoretical_mem
                if static_mem > 0:
                    static_memories['real'] = static_mem
        
        # Plot Simulation data if available
        if 'sim' in paths:
            success, theoretical_mem, static_mem = _plot_allocated_memory_for_path(
                ax, paths['sim'], 'Simulation', 
                {'linestyle': '--', 'linewidth': 2.0, 'alpha': 1.0}
            )
            if success:
                has_plot_data = True
                if theoretical_mem > 0:
                    theoretical_memories['sim'] = theoretical_mem
                if static_mem > 0:
                    static_memories['sim'] = static_mem

        if has_plot_data:
            # Add theoretical memory as horizontal lines only if show_theoretical is True
            if show_theoretical:
                for run_type, theo_mem in theoretical_memories.items():
                    line_style = '-' if run_type == 'real' else '--'
                    ax.axhline(y=theo_mem, color='red', linestyle=line_style, 
                              alpha=0.7, linewidth=1.5, 
                              label=f'Theoretical Memory ({run_type.title()})')
            
            # Always show static memory as baseline reference lines
            for run_type, static_mem in static_memories.items():
                line_style = '-' if run_type == 'real' else '--'
                ax.axhline(y=static_mem, color='green', linestyle=line_style, 
                          alpha=0.6, linewidth=1.2, 
                          label=f'Static Memory ({run_type.title()})')

            ax.set_xlabel('Time in Iteration (seconds)', fontsize=14)
            ax.set_ylabel('Allocated Memory (MB)', fontsize=14)
            ax.set_title(f'Memory Comparison for Rank {rank} (Last Iteration)', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True)
            
            output_file = os.path.join(output_dir, f'memory_comparison_rank_{rank}.png')
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Plot for rank {rank} saved to {output_file}")
            except Exception as e:
                print(f"Error saving plot for rank {rank}: {e}")
        
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Visualize memory trace comparisons between real and simulation runs')
    parser.add_argument('real_traces_dir', help='Directory containing real run memory traces')
    parser.add_argument('sim_traces_dir', help='Directory containing simulation memory traces')
    parser.add_argument('output_dir', nargs='?', default='memory_plots', help='Output directory for generated plots (default: memory_plots)')
    parser.add_argument('--show-theoretical', action='store_true', default=False, 
                       help='Display theoretical memory lines in plots (default: disabled)')

    # Handle backward compatibility with old positional argument style
    if len(sys.argv) >= 3 and not sys.argv[1].startswith('-'):
        # Old style: python script.py real_dir sim_dir [output_dir]
        if len(sys.argv) == 3 or (len(sys.argv) == 4 and not sys.argv[3].startswith('-')):
            real_trace_dir = sys.argv[1]
            sim_trace_dir = sys.argv[2]
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'memory_plots'
            show_theoretical = False
        else:
            args = parser.parse_args()
            real_trace_dir = args.real_traces_dir
            sim_trace_dir = args.sim_traces_dir
            output_dir = args.output_dir
            show_theoretical = args.show_theoretical
    else:
        args = parser.parse_args()
        real_trace_dir = args.real_traces_dir
        sim_trace_dir = args.sim_traces_dir
        output_dir = args.output_dir
        show_theoretical = args.show_theoretical

    paired_files = {}

    # Find and pair real trace files
    real_files = glob.glob(os.path.join(real_trace_dir, 'memory_trace_rank*.json'))
    for f_path in real_files:
        rank_match = re.search(r'rank(\d+)', os.path.basename(f_path))
        if rank_match:
            rank = rank_match.group(1)
            paired_files.setdefault(rank, {})['real'] = f_path
            
    # Find and pair simulation trace files
    sim_files = glob.glob(os.path.join(sim_trace_dir, 'memory_trace_rank*.json'))
    for f_path in sim_files:
        rank_match = re.search(r'rank(\d+)', os.path.basename(f_path))
        if rank_match:
            rank = rank_match.group(1)
            paired_files.setdefault(rank, {})['sim'] = f_path

    if not paired_files:
        print("No trace files found in the specified directories.")
        sys.exit(1)

    visualize_memory_comparison(paired_files, output_dir, show_theoretical)

if __name__ == '__main__':
    main()
