import importlib.util
from pathlib import Path

import torch

from megatron.core import ModelParallelConfig


def _load_p2p_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "megatron" / "core" / "pipeline_parallel" / "p2p_communication.py"
    spec = importlib.util.spec_from_file_location("p2p_communication_test_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_align_forward_tensor_dtype_casts_to_pipeline_dtype():
    p2p_module = _load_p2p_module()
    config = ModelParallelConfig(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16)
    config.multi_latent_attention = True
    tensor = torch.randn(2, 3, dtype=torch.float32)
    aligned = p2p_module._align_forward_tensor_dtype(tensor, config)
    assert aligned.dtype == torch.bfloat16


def test_align_forward_tensor_dtype_keeps_nonfloating_tensor():
    p2p_module = _load_p2p_module()
    config = ModelParallelConfig(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16)
    config.multi_latent_attention = True
    tensor = torch.randint(0, 10, (2, 3), dtype=torch.int64)
    aligned = p2p_module._align_forward_tensor_dtype(tensor, config)
    assert aligned.dtype == torch.int64


def test_align_forward_tensor_dtype_noop_when_not_mla():
    p2p_module = _load_p2p_module()
    config = ModelParallelConfig(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16)
    config.multi_latent_attention = False
    tensor = torch.randn(2, 3, dtype=torch.float32)
    aligned = p2p_module._align_forward_tensor_dtype(tensor, config)
    assert aligned.dtype == torch.float32
