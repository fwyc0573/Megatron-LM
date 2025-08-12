#!/usr/bin/env python3

import os
import sys
import time
import torch

# Add the megatron path
sys.path.insert(0, '/research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM')

from megatron.profiler.trace_memory import get_memory_tracker

class MockArgs:
    def __init__(self):
        self.trace_memory = True
        self.is_scaling_mode = True
        self.fake_current_rank_id = 0
        self.trace_memory_dir = 'test_memory_traces_scaling'
        self.trace_memory_interval = 0.01
        self.fake_world_size = 8
        self.fake_tp = 1
        self.fake_pp = 8
        self.fake_dp = 1
        self.fake_exp = 1
        self.fake_num_experts = None
        self.num_layers = 40
        self.micro_batch_size = 1
        self.global_batch_size = 1
        self.num_micro_batches = 1

def test_memory_tracker():
    print("Testing memory tracker...")
    
    # Create mock args
    args = MockArgs()
    
    # Test if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available, cannot test memory tracker")
        return
    
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Test pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        print("pynvml initialized successfully")
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"NVML device count: {device_count}")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"pynvml error: {e}")
        return
    
    # Create memory tracker
    print("Creating memory tracker...")
    memory_tracker = get_memory_tracker(args)
    
    if memory_tracker is None:
        print("ERROR: Memory tracker is None!")
        return
    
    print(f"Memory tracker created: {memory_tracker}")
    print(f"Output path: {memory_tracker.output_path}")
    
    # Start the tracker
    print("Starting memory tracker...")
    memory_tracker.start()
    
    # Simulate some work
    print("Simulating work...")
    memory_tracker.next_iteration(0)
    
    # Log static memory
    static_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    memory_tracker.log_static_memory(0, static_memory_mb)
    print(f"Static memory logged: {static_memory_mb:.2f} MB")
    
    # Allocate some memory
    x = torch.randn(1000, 1000, device='cuda')
    time.sleep(1)  # Let the tracker sample
    
    # Log peak memory
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    memory_tracker.log_peak_memory(0, peak_memory_mb)
    print(f"Peak memory logged: {peak_memory_mb:.2f} MB")
    
    # Log theoretical memory
    memory_tracker.log_theoretical_memory(0, 1000.0)
    print("Theoretical memory logged: 1000.0 MB")
    
    # Stop the tracker
    print("Stopping memory tracker...")
    memory_tracker.stop_tracking()
    
    # Check if file was created
    if os.path.exists(memory_tracker.output_path):
        print(f"SUCCESS: Memory trace file created at {memory_tracker.output_path}")
        with open(memory_tracker.output_path, 'r') as f:
            content = f.read()
            print(f"File size: {len(content)} bytes")
            print("First 500 characters:")
            print(content[:500])
    else:
        print(f"ERROR: Memory trace file not found at {memory_tracker.output_path}")

if __name__ == "__main__":
    test_memory_tracker()
