#!/usr/bin/env python3
"""
GPU Memory Usage Heatmap Visualization for LLM 3D Parallel Training

This script creates a heatmap visualization showing GPU memory usage across different
TP (Tensor Parallelism) and PP (Pipeline Parallelism) configurations.

Author: Generated for Yicheng
Date: 2025-07-26
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple, List, Optional
import argparse
from pathlib import Path


class MemoryHeatmapVisualizer:
    """Visualizer for GPU memory usage heatmap across TP/PP configurations."""
    
    def __init__(self, memory_traces_dir: str, log_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            memory_traces_dir: Directory containing JSON memory trace files
            log_dir: Directory containing configuration log files
        """
        self.memory_traces_dir = Path(memory_traces_dir)
        self.log_dir = Path(log_dir)
        
        # Configuration data
        self.successful_configs = set()
        self.failed_configs = set()
        self.memory_data = {}
        self.theoretical_memory_data = {}  # Store theoretical memory predictions
        self.theory_mismatch_configs = set()  # Configs where theory > 80GB but actual success
        
        # Visualization parameters
        self.max_memory_gb = 80.0  # GPU memory limit in GB
        self.color_map = self._create_color_map()

        # Academic style settings
        self._setup_academic_style()
        
    def _setup_academic_style(self) -> None:
        # 'font.family': 'serif',
        # 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],        """Setup academic publication style for matplotlib."""
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 13,
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.0,
            'patch.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'axes.edgecolor': 'black',
            'axes.grid': False,
            'grid.alpha': 0.3,
            'hatch.linewidth': 2.5 
        })

    def _create_color_map(self) -> LinearSegmentedColormap:
        """Create custom colormap for memory usage visualization."""
        # Define colors: white (0%) to deep red (100%) - academic style
        colors = ['#FFFFFF', '#FEE5D9', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D']
        return LinearSegmentedColormap.from_list('memory_usage', colors, N=256)
    
    def _parse_filename(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse TP and PP values from memory trace filename.
        
        Args:
            filename: Memory trace filename
            
        Returns:
            Tuple of (TP, PP) values or None if parsing fails
        """
        # Pattern: memory_trace_rank0_wd8192_tp{TP}_pp{PP}_dp{DP}_...
        pattern = r'memory_trace_rank0_wd8192_tp(\d+)_pp(\d+)_dp(\d+)'
        match = re.search(pattern, filename)
        
        if match:
            tp = int(match.group(1))
            pp = int(match.group(2))
            return (tp, pp)
        return None
    
    def _load_memory_data(self) -> None:
        """Load memory data from JSON files."""
        print("Loading memory trace data...")
        
        for json_file in self.memory_traces_dir.glob("*.json"):
            tp_pp = self._parse_filename(json_file.name)
            if tp_pp is None:
                print(f"Warning: Could not parse filename {json_file.name}")
                continue
                
            tp, pp = tp_pp
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract peak allocated memory and theoretical memory from rank 0
                if "0" in data and "peak_allocated_MB" in data["0"]:
                    peak_mb = data["0"]["peak_allocated_MB"]
                    peak_gb = peak_mb / 1024.0  # Convert MB to GB
                    self.memory_data[(tp, pp)] = peak_gb

                    # Extract theoretical memory if available
                    if "theoretical_memory_MB" in data["0"]:
                        theoretical_mb = data["0"]["theoretical_memory_MB"]
                        theoretical_gb = theoretical_mb / 1024.0  # Convert MB to GB
                        self.theoretical_memory_data[(tp, pp)] = theoretical_gb

                        # Check for theory-reality mismatch (theory > 80GB but actual success)
                        if theoretical_gb > 80.0:
                            self.theory_mismatch_configs.add((tp, pp))
                            print(f"Loaded TP={tp}, PP={pp}: {peak_gb:.1f} GB (Theory: {theoretical_gb:.1f} GB - MISMATCH)")
                        else:
                            print(f"Loaded TP={tp}, PP={pp}: {peak_gb:.1f} GB (Theory: {theoretical_gb:.1f} GB)")
                    else:
                        print(f"Loaded TP={tp}, PP={pp}: {peak_gb:.1f} GB (No theory data)")
                else:
                    print(f"Warning: Invalid data structure in {json_file.name}")
                    
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
    
    def _load_config_status(self) -> None:
        """Load configuration success/failure status from log files."""
        print("Loading configuration status...")
        
        # Load successful configurations
        success_file = self.log_dir / "successful_configs.log"
        if success_file.exists():
            with open(success_file, 'r') as f:
                for line in f:
                    if "TP=" in line and "PP=" in line:
                        # Parse: Config X: TP=Y, PP=Z, ... - SUCCESS
                        tp_match = re.search(r'TP=(\d+)', line)
                        pp_match = re.search(r'PP=(\d+)', line)
                        if tp_match and pp_match:
                            tp = int(tp_match.group(1))
                            pp = int(pp_match.group(1))
                            self.successful_configs.add((tp, pp))

        # Load failed configurations
        failed_file = self.log_dir / "failed_configs.log"
        if failed_file.exists():
            with open(failed_file, 'r') as f:
                for line in f:
                    if "TP=" in line and "PP=" in line:
                        # Parse: Config X: TP=Y, PP=Z, ... - FAILED
                        tp_match = re.search(r'TP=(\d+)', line)
                        pp_match = re.search(r'PP=(\d+)', line)
                        if tp_match and pp_match:
                            tp = int(tp_match.group(1))
                            pp = int(pp_match.group(1))
                            self.failed_configs.add((tp, pp))
        
        print(f"Loaded {len(self.successful_configs)} successful configs")
        print(f"Loaded {len(self.failed_configs)} failed configs")
    
    def _get_tp_pp_ranges(self) -> Tuple[List[int], List[int]]:
        """Get sorted lists of TP and PP values."""
        all_configs = set(self.memory_data.keys()) | self.successful_configs | self.failed_configs
        
        tp_values = sorted(set(tp for tp, pp in all_configs))
        pp_values = sorted(set(pp for tp, pp in all_configs))
        
        return tp_values, pp_values
    
    def _create_heatmap_matrix(self, tp_values: List[int], pp_values: List[int]) -> np.ndarray:
        """
        Create the heatmap matrix with memory usage data.
        
        Args:
            tp_values: Sorted list of TP values
            pp_values: Sorted list of PP values
            
        Returns:
            2D numpy array for heatmap visualization
        """
        matrix = np.full((len(tp_values), len(pp_values)), np.nan)
        
        for i, tp in enumerate(tp_values):
            for j, pp in enumerate(pp_values):
                config = (tp, pp)
                
                if config in self.memory_data:
                    # Successful configuration with memory data
                    memory_gb = self.memory_data[config]
                    # Normalize to [0, 1] range for colormap
                    normalized_memory = min(memory_gb / self.max_memory_gb, 1.0)
                    matrix[i, j] = normalized_memory
                elif config in self.failed_configs:
                    # Failed configuration (OOM) - use special value
                    matrix[i, j] = -1.0  # Special marker for failed configs
                # else: leave as NaN for missing data
        
        return matrix
    
    def create_heatmap(self, output_path: str = "memory_heatmap",
                      figsize: Tuple[float, float] = None, dpi: int = 300) -> None:
        """
        Create and save the memory usage heatmap in both PNG and PDF formats.

        Args:
            output_path: Base path to save the heatmap (without extension)
            figsize: Figure size (width, height) in inches. If None, auto-calculate 1:3 ratio
            dpi: DPI for output images
        """
        # Load all data
        self._load_memory_data()
        self._load_config_status()
        
        # Get TP and PP ranges
        tp_values, pp_values = self._get_tp_pp_ranges()

        if not tp_values or not pp_values:
            print("Error: No valid TP/PP configurations found!")
            return

        print(f"TP values: {tp_values}")
        print(f"PP values: {pp_values}")

        # Auto-calculate figure size with 1:3 ratio if not provided
        if figsize is None:
            # Base width on number of PP values, height on TP values
            # Aim for 1:3 ratio (height:width)
            base_width = max(8, len(pp_values) * 1.2)
            base_height = base_width / 1.8
            figsize = (base_width, base_height)

        print(f"Using figure size: {figsize[0]:.1f} x {figsize[1]:.1f} inches")

        # Create heatmap matrix
        matrix = self._create_heatmap_matrix(tp_values, pp_values)

        # Create the plot with academic style
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Create custom colormap that handles failed configs
        cmap = self.color_map.copy()
        cmap.set_bad(color='#E5E5E5')  # Light gray for NaN values (missing data)

        # Plot the heatmap with academic styling
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                      interpolation='nearest')

        # Add text annotations for each cell with GB units
        for i, tp in enumerate(tp_values):
            for j, pp in enumerate(pp_values):
                config = (tp, pp)

                if config in self.memory_data:
                    # Show both actual and theoretical memory values
                    memory_gb = self.memory_data[config]

                    # Prepare text with both actual and theoretical values
                    if config in self.theoretical_memory_data:
                        theory_gb = self.theoretical_memory_data[config]
                        text = f"Moye: {memory_gb:.1f}GB\n\nTheory: {theory_gb:.1f}GB"
                    else:
                        text = f"Moye: {memory_gb:.1f}GB\nTheory: N/A"

                    # Use white text on dark backgrounds, black on light
                    normalized_value = memory_gb / self.max_memory_gb
                    color = 'white' if normalized_value > 0.6 else 'black'
                    fontweight = 'normal'

                    # Add white diagonal hatching for theory >= 80GB configs
                    if config in self.theory_mismatch_configs:
                        # Add white diagonal hatching pattern
                        hatch_rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                     linewidth=2.5, facecolor='none',
                                                     edgecolor='white', hatch='\\',
                                                     alpha=0.8)
                        ax.add_patch(hatch_rect)
                elif config in self.failed_configs:
                    # Show "--" for failed configs with gray background
                    text = "—"  # Em dash for better typography
                    color = 'black'
                    fontweight = 'bold'
                    # Add lighter gray background for failed configs
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                           linewidth=1, facecolor='#E8E8E8',
                                           edgecolor='white', alpha=0.7)
                    ax.add_patch(rect)
                else:
                    # Missing data
                    text = ""
                    color = 'black'
                    fontweight = 'normal'

                if text:
                    ax.text(j, i, text, ha='center', va='center',
                           color='black', fontsize=9.3, fontweight=fontweight,
                           linespacing=0.9)
        
        # Customize the plot with academic styling
        ax.set_xticks(range(len(pp_values)))
        ax.set_xticklabels(pp_values, fontsize=10)
        ax.set_yticks(range(len(tp_values)))
        ax.set_yticklabels(tp_values, fontsize=10)

        # Academic style labels and title
        ax.set_xlabel('Pipeline Parallelism (PP)', fontsize=11, fontweight='normal')
        ax.set_ylabel('Tensor Parallelism (TP)', fontsize=11, fontweight='normal')
        ax.set_title('GPU Memory Scanning: 485B Model Training Using Megatron-LM\n(8192 GPUs, A800-SXM4-80GB, FP16)',
                    fontsize=14, fontweight='normal', pad=8)

        # Add professional colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, aspect=30)
        cbar.set_label('Peak Memory (GB)', fontsize=10, fontweight='normal')

        # Customize colorbar ticks to show actual memory values
        cbar_ticks = np.linspace(0, 1, 5)
        cbar_tick_labels = [f"{tick * self.max_memory_gb:.0f}" for tick in cbar_ticks]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_tick_labels, fontsize=9)

        # Add subtle grid lines
        ax.set_xticks(np.arange(-0.5, len(pp_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tp_values), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5, alpha=0.8)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        # Add professional legend with theory mismatch indicator
        legend_elements = [
            patches.Patch(facecolor='#99000D', label='High Memory'),
            patches.Patch(facecolor='#FFFFFF', edgecolor='black', linewidth=0.5, label='Low Memory'),
            patches.Patch(facecolor='#E8E8E8', label='Failed (OOM)'),
            patches.Patch(facecolor='#FCBBA1', edgecolor='white', hatch='///', label='Theory ≥ 80GB')
        ]
        # legend = ax.legend(handles=legend_elements, loc='upper left',
        #                   bbox_to_anchor=(1.02, 1), frameon=True,
        #                   fancybox=False, shadow=False, fontsize=9)
        # legend.get_frame().set_linewidth(0.5)
        # legend.get_frame().set_edgecolor('black')
        
        # Adjust layout for academic publication
        plt.tight_layout()

        # Save in both PNG and PDF formats
        base_path = Path(output_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove extension if provided
        if base_path.suffix:
            base_path = base_path.with_suffix('')

        # Save PNG version (for presentations, web)
        png_path = base_path.with_suffix('.png')
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', format='png')
        print(f"PNG heatmap saved to: {png_path}")

        # Save PDF version (for publications)
        pdf_path = base_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight',
                   facecolor='white', edgecolor='none', format='pdf')
        print(f"PDF heatmap saved to: {pdf_path}")

        print(f"Summary: {len(self.successful_configs)} successful, {len(self.failed_configs)} failed configurations")
        if self.theory_mismatch_configs:
            print(f"Theory-Reality Mismatch: {len(self.theory_mismatch_configs)} configs where theory > 80GB but actual success")
            for config in sorted(self.theory_mismatch_configs):
                tp, pp = config
                actual_gb = self.memory_data[config]
                theory_gb = self.theoretical_memory_data[config]
                print(f"  TP={tp}, PP={pp}: Actual={actual_gb:.1f}GB, Theory={theory_gb:.1f}GB")

        plt.show()


def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize GPU memory usage heatmap for LLM training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python visualize_memory_heatmap.py

  # Custom paths and output
  python visualize_memory_heatmap.py \\
    --memory-dir ../../examples/memory_traces_scaling/485B_8192gpus_tplimit8 \\
    --log-dir ../../log/CONFIG_SWEEP_WS8192_540_20250726_024653 \\
    --output ../../visualization_outputs/memory_heatmap_485B_8192gpus.png

  # Larger figure size
  python visualize_memory_heatmap.py --figsize 16 10
        """
    )
    parser.add_argument('--memory-dir', type=str,
                       default='examples/memory_traces_scaling/485B_8192gpus_tplimit8',
                       help='Directory containing memory trace JSON files')
    parser.add_argument('--log-dir', type=str,
                       default='log/CONFIG_SWEEP_WS8192_540_20250726_024653',
                       help='Directory containing configuration log files')
    parser.add_argument('--output', type=str, default='memory_heatmap',
                       help='Base output path for the heatmap (without extension, will generate .png and .pdf)')
    parser.add_argument('--figsize', type=float, nargs=2, default=None,
                       help='Figure size (width height) in inches. If not specified, auto-calculate 1:3 ratio')
    parser.add_argument('--max-memory', type=float, default=80.0,
                       help='Maximum GPU memory in GB for color scaling (default: 80.0)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for PNG output (default: 300)')

    args = parser.parse_args()

    # Create visualizer and generate heatmap
    visualizer = MemoryHeatmapVisualizer(args.memory_dir, args.log_dir)
    visualizer.max_memory_gb = args.max_memory

    # Convert figsize to tuple if provided
    figsize = tuple(args.figsize) if args.figsize else None
    visualizer.create_heatmap(args.output, figsize, dpi=args.dpi)


if __name__ == "__main__":
    main()
