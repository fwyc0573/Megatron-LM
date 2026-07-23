#!/usr/bin/env python3
import ast

class MessageSizeCalculator:
    DTYPE_SIZES = {
        'torch.float32': 4, 'torch.float16': 2, 'torch.float64': 8,
        'torch.int8': 1, 'torch.uint8': 1, 'torch.int16': 2,
        'torch.int32': 4, 'torch.int64': 8, 'torch.bool': 1,
        'torch.bfloat16': 2, 'float32': 4, 'float16': 2, 'float64': 8,
        'int8': 1, 'uint8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
        'bool': 1, 'bfloat16': 2, 'fp32': 4, 'fp16': 2, 'fp64': 8,
        'half': 2, 'float': 4, 'double': 8,
    }
    
    def calculate_tensor_size(self, tensor_shape, tensor_dtype):
        if tensor_shape is None or tensor_dtype is None:
            return 0
        
        shape = self._parse_tensor_shape(tensor_shape)
        if not shape:
            return 0
        
        num_elements = 1
        for dim in shape:
            if dim <= 0:
                return 0
            num_elements *= dim
        
        dtype_size = self._get_dtype_size(tensor_dtype)
        if dtype_size == 0:
            return 0
        
        return num_elements * dtype_size
    
    def _parse_tensor_shape(self, tensor_shape):
        if tensor_shape is None:
            return []
        
        if isinstance(tensor_shape, list):
            try:
                return [int(dim) for dim in tensor_shape]
            except (ValueError, TypeError):
                return []
        
        if isinstance(tensor_shape, str):
            try:
                parsed = ast.literal_eval(tensor_shape)
                if isinstance(parsed, (list, tuple)):
                    return [int(dim) for dim in parsed]
                else:
                    return []
            except (ValueError, SyntaxError):
                return []
        
        return []
    
    def _get_dtype_size(self, tensor_dtype):
        if not tensor_dtype:
            return 0
        
        dtype_clean = tensor_dtype.strip().lower()
        
        if dtype_clean in self.DTYPE_SIZES:
            return self.DTYPE_SIZES[dtype_clean]
        
        if dtype_clean.startswith("torch."):
            dtype_without_prefix = dtype_clean[6:]
            if dtype_without_prefix in self.DTYPE_SIZES:
                return self.DTYPE_SIZES[dtype_without_prefix]
        
        if "float32" in dtype_clean or "fp32" in dtype_clean:
            return 4
        elif "float16" in dtype_clean or "fp16" in dtype_clean or "half" in dtype_clean:
            return 2
        elif "float64" in dtype_clean or "fp64" in dtype_clean or "double" in dtype_clean:
            return 8
        elif "bfloat16" in dtype_clean:
            return 2
        
        return 0

_global_calculator = MessageSizeCalculator()

def get_tensor_data_size(tensor_shape, tensor_dtype):
    return _global_calculator.calculate_tensor_size(tensor_shape, tensor_dtype)

def calculate_comm_message_size(tensor_shape, tensor_dtype, comm_op, world_size=1):
    return _global_calculator.calculate_tensor_size(tensor_shape, tensor_dtype)
