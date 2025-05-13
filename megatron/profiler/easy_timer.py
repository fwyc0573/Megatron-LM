import torch

class CUDATimer:
    def __init__(self):
        self.events = {}
        self.warm_up_sign = True

    def __call__(self, rank_id, cmd):
        self.rank_id = rank_id
        self.current_cmd = cmd
        return self

    def __enter__(self):
        if self.warm_up_sign is False:
            # 初始化 CUDA 事件
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            # 记录开始时间
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.warm_up_sign is False:
            # 记录结束时间
            self.end_event.record()
            # 同步 CUDA 事件
            torch.cuda.synchronize()
            # 计算耗时
            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

            # 组织记录为 {rank_id: {cmd: elapsed_time}}
            if self.rank_id not in self.events:
                self.events[self.rank_id] = {}
            self.events[self.rank_id][self.current_cmd] = elapsed_time_ms

    def cuda_time_output(self):
        # 打印每个 rank 中记录的 cmd 耗时
        for rank_id, cmds in self.events.items():
            print(f"Rank {rank_id} CUDA Timing:")
            for cmd, time_ms in cmds.items():
                print(f"  {cmd}: {time_ms:.2f} ms")
            print("-" * 40)

# 使用示例
if __name__ == "__main__":
    timer = CUDATimer()

    # 模拟的前向传播
    with timer(0, "forward pass"):
        input_tensor = torch.randn(1000, 1000, device='cuda')
        model_output = torch.matmul(input_tensor, input_tensor.T)

    # 模拟的反向传播
    with timer(0, "backward pass"):
        model_output.backward(torch.ones_like(model_output))

    # 输出测量结果
    timer.cuda_time_output()
