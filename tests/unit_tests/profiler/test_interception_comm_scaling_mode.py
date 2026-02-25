import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


def _load_cmd_module():
    cmd_path = (
        Path(__file__).resolve().parents[3] / "megatron" / "profiler" / "cmd.py"
    )
    spec = importlib.util.spec_from_file_location("test_cmd_module_for_interception", cmd_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {cmd_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_interception_module(cmd_module):
    # Inject lightweight package stubs so interception_comm.py can import CMD
    megatron_pkg = sys.modules.setdefault("megatron", types.ModuleType("megatron"))
    profiler_pkg = sys.modules.setdefault("megatron.profiler", types.ModuleType("megatron.profiler"))
    setattr(profiler_pkg, "cmd", cmd_module)
    setattr(megatron_pkg, "profiler", profiler_pkg)
    sys.modules["megatron.profiler.cmd"] = cmd_module

    module_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "profiler"
        / "comm_utils"
        / "interception_comm.py"
    )
    spec = importlib.util.spec_from_file_location("test_interception_comm_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmd_module = _load_cmd_module()
interception_comm = _load_interception_module(cmd_module)


def _patch_training_args(is_scaling_mode):
    training_module = types.ModuleType("megatron.training")
    training_module.get_args = lambda: SimpleNamespace(is_scaling_mode=is_scaling_mode)
    return mock.patch.dict(sys.modules, {"megatron.training": training_module})


def test_allreduce_wrapper_skips_collective_in_scaling_mode():
    tensor = torch.ones(4)
    with _patch_training_args(True), mock.patch.object(torch.distributed, "all_reduce") as mocked_all_reduce:
        output, handle = interception_comm.allreduce_wrapper(tensor, tp_group=None, async_op=False)
    assert output is tensor
    assert handle is None
    assert mocked_all_reduce.call_count == 0


def test_reduce_wrapper_async_skips_collective_in_scaling_mode():
    tensor = torch.ones(4)
    with _patch_training_args(True), mock.patch.object(torch.distributed, "all_reduce") as mocked_all_reduce:
        handle = interception_comm.reduce_wrapper(tensor, async_op=True, tp_group=None)
    assert handle is None
    assert mocked_all_reduce.call_count == 0


def test_broadcast_wrapper_skips_collective_in_scaling_mode():
    tensor = torch.ones(4)
    with _patch_training_args(True), mock.patch.object(torch.distributed, "broadcast") as mocked_broadcast:
        output = interception_comm.broadcast_wrapper(
            tensor, func="tokens", tp_group=object(), tp_src_rank=0
        )
    assert output is tensor
    assert mocked_broadcast.call_count == 0

