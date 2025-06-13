import multiprocessing
import threading
import queue
import time
import os
import json

try:
    import pynvml
except ImportError:
    pynvml = None

import torch

class MemoryTracker(threading.Thread):
    """
    A thread that tracks GPU memory usage of a given device at a specified interval.
    Runs in the same process as the training script to access torch.cuda.memory_allocated.
    """
    def __init__(self, rank, device_id, output_dir, sampling_interval=0.01, file_name_args=""):
        super().__init__()
        self.rank = rank
        self.device_id = device_id
        self.sampling_interval = sampling_interval
        os.makedirs(output_dir, exist_ok=True)
        if file_name_args:
             self.output_path = os.path.join(output_dir, f'memory_trace_rank{rank}_{file_name_args}.json')
        else:
             self.output_path = os.path.join(output_dir, f'memory_trace_rank{rank}.json')

        self.queue = queue.Queue()
        self.data = {}
        self.start_time = 0
        self.tracking = False

    def start_tracking(self, iteration):
        """Signal the tracker to start recording for a given iteration."""
        self.queue.put(('start_iter', iteration))

    def pause_tracking(self):
        """Signal the tracker to pause recording."""
        self.queue.put(('pause', -1))

    def next_iteration(self, iteration):
        """Signal the tracker to start recording for the next iteration."""
        self.queue.put(('next_iter', iteration))

    def log_peak_memory(self, iteration, peak_memory_mb):
        """Signal the tracker to log the peak memory for an iteration."""
        self.queue.put(('log_peak', (iteration, peak_memory_mb)))

    def stop_tracking(self):
        """Signal the tracker to stop recording and save the data."""
        self.queue.put(('stop', -1))
        print(f"Stopping memory tracking for rank {self.rank}...")
        self.join()
        print(f"Memory tracking for rank {self.rank} stopped. Data saved to {self.output_path}")

    def run(self):
        """The main loop of the tracker thread."""
        if pynvml is None:
            if self.rank == 0:
                print("Warning: pynvml not installed. GPU memory tracking is disabled.")
            return

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

        current_iter = -1
        running = True
        self.start_time = time.time()
        self.tracking = False

        while running:
            # Check for commands from the main process
            while not self.queue.empty():
                try:
                    cmd, val = self.queue.get_nowait()
                    if cmd == 'next_iter':
                        current_iter = val
                        if current_iter not in self.data:
                            self.data[current_iter] = {'samples': [], 'peak_allocated_MB': -1}
                        self.tracking = True
                    elif cmd == 'start_iter':
                        current_iter = val
                        if current_iter not in self.data:
                            self.data[current_iter] = {'samples': [], 'peak_allocated_MB': -1}
                        self.tracking = True
                    elif cmd == 'pause':
                        self.tracking = False
                    elif cmd == 'stop':
                        running = False
                        break
                    elif cmd == 'log_peak':
                        iter_num, peak_mem = val
                        if iter_num in self.data:
                            self.data[iter_num]['peak_allocated_MB'] = round(peak_mem, 2)
                except queue.Empty:
                    break
            
            if not running:
                break

            if self.tracking and current_iter != -1:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    allocated_mem = torch.cuda.memory_allocated(self.device_id)
                    timestamp = time.time() - self.start_time
                    self.data[current_iter]['samples'].append({
                        'timestamp_s': round(timestamp, 4),
                        'reserved_memory_MB': mem_info.used / (1024**2),
                        'allocated_memory_MB': allocated_mem / (1024**2)
                    })
                except pynvml.NVMLError as e:
                    print(f"Rank {self.rank} memory tracker: NVML error: {e}")
                    pass

            time.sleep(self.sampling_interval)
        
        pynvml.nvmlShutdown()

        try:
            with open(self.output_path, 'w') as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Rank {self.rank} memory tracker: Error writing to file {self.output_path}: {e}")

def get_memory_tracker(args):
    """Initialize and return the memory tracker if enabled."""
    if not getattr(args, 'trace_memory', False):
        return None
    
    import torch

    # In scaling mode, we don't use torch.distributed, but simulate ranks.
    if getattr(args, 'is_scaling_mode', False):
        rank = args.fake_current_rank_id
        device_id = torch.cuda.current_device()
        # Distinguish the log's out dir for scaling mode
        output_dir = getattr(args, 'trace_memory_dir', 'memory_traces_scaling')
        sampling_interval = getattr(args, 'trace_memory_interval', 0.01)

        name_args = (f"wd{args.fake_world_size}_tp{args.fake_tp}_pp{args.fake_pp}"
                     f"_exp{args.fake_exp}_expNum{args.fake_num_experts}"
                     f"_l{args.num_layers}_bs{args.micro_batch_size}")
    else:
        import torch.distributed as dist
        rank = dist.get_rank()
        device_id = torch.cuda.current_device()
        output_dir = getattr(args, 'trace_memory_dir', 'memory_traces')
        sampling_interval = getattr(args, 'trace_memory_interval', 0.01)
    
        name_args = f"wd{args.world_size}_tp{args.tensor_model_parallel_size}_pp{args.pipeline_model_parallel_size}"
        if args.expert_model_parallel_size > 1:
            name_args += f"_ep{args.expert_model_parallel_size}_expNum{args.num_experts}"
        name_args += f"_l{args.num_layers}_bs{args.micro_batch_size}"

    tracker = MemoryTracker(
        rank=rank,
        device_id=device_id,
        output_dir=output_dir,
        sampling_interval=sampling_interval,
        file_name_args=name_args
    )
    return tracker

def _plot_allocated_memory_for_path(ax, json_path, label, style_kwargs):
    """Helper function to load, process, and plot data for a single trace file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not read or parse {json_path}: {e}")
        return False
    
    if not data:
        print(f"Warning: No data found in {json_path}")
        return False

    # Get the last iteration
    try:
        last_iter_key = str(max(map(int, data.keys())))
        iter_data = data[last_iter_key]
    except (ValueError, KeyError):
        print(f"Warning: Could not determine last iteration in {json_path}")
        return False

    samples = iter_data.get('samples', [])
    if not samples or 'allocated_memory_MB' not in samples[0]:
        print(f"Warning: No valid samples found for last iteration in {json_path}")
        return False
        
    timestamps = [s['timestamp_s'] for s in samples]
    allocated_mem = [s['allocated_memory_MB'] for s in samples]
    
    # Normalize timestamps to start from 0 for this iteration
    start_time = timestamps[0]
    relative_timestamps = [t - start_time for t in timestamps]
    
    ax.plot(relative_timestamps, allocated_mem, label=label, **style_kwargs)
    return True

def visualize_memory_comparison(paired_rank_files, output_dir='memory_plots'):
    """
    Visualizes a comparison of allocated memory for simulation vs. real runs,
    generating one plot per rank.

    Args:
        paired_rank_files (dict): A dict mapping rank_id to {'sim': path, 'real': path}.
        output_dir (str): Directory to save the output plot images.
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

        # Plot Real Run data if available
        if 'real' in paths:
            if _plot_allocated_memory_for_path(ax, paths['real'], 'Real Run', 
                                               {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8}):
                has_plot_data = True
        
        # Plot Simulation data if available
        if 'sim' in paths:
            if _plot_allocated_memory_for_path(ax, paths['sim'], 'Simulation', 
                                               {'linestyle': '--', 'linewidth': 2.0, 'alpha': 1.0}):
                has_plot_data = True

        if has_plot_data:
            ax.set_xlabel('Time in Iteration (seconds)', fontsize=14)
            ax.set_ylabel('Allocated Memory (MB)', fontsize=14)
            ax.set_title(f'Allocated Memory Comparison for Rank {rank} (Last Iteration)', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True)
            
            output_file = os.path.join(output_dir, f'memory_comparison_rank_{rank}.png')
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Plot for rank {rank} saved to {output_file}")
            except Exception as e:
                print(f"Error saving plot for rank {rank}: {e}")
        
        plt.close(fig)

if __name__ == '__main__':
    import sys
    import glob
    import re

    usage_msg = (
        "Usage: python megatron/profiler/trace_memory.py <real_traces_dir> <sim_traces_dir> [output_dir]\n"
        "Example: python megatron/profiler/trace_memory.py "
        "examples/memory_traces examples/memory_traces_scaling memory_plots"
    )

    if len(sys.argv) < 3:
        print(usage_msg)
        sys.exit(1)
    
    real_trace_dir = sys.argv[1]
    sim_trace_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'memory_plots'

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

    visualize_memory_comparison(paired_files, output_dir)

'''
python megatron/profiler/trace_memory.py \
    examples/memory_traces \
    examples/memory_traces_scaling \
    my_comparison_plots
'''
