import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


def _load_cmd_module():
    cmd_path = (
        Path(__file__).resolve().parents[3] / "megatron" / "profiler" / "cmd.py"
    )
    spec = importlib.util.spec_from_file_location("test_cmd_module", cmd_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {cmd_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmd_module = _load_cmd_module()
CMD = cmd_module.CMD


class _DummyEvent:
    sync_calls = 0

    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self):
        return None

    def synchronize(self):
        _DummyEvent.sync_calls += 1

    def elapsed_time(self, other):
        return 0.12


class _DummyCmd:
    def __init__(self, sync_mode, is_scaling_mode=False):
        self.use_cuda = True
        self.args = SimpleNamespace(
            trace_subop_sync_mode=sync_mode, is_scaling_mode=is_scaling_mode
        )
        self.sub_operations = []

    def add_sub_operation(self, operation_name, duration, attr_info, timestamp_ms=None):
        self.sub_operations.append((operation_name, duration, attr_info, timestamp_ms))


@CMD.get_trace_decorator()
def _decorated_add_one(x):
    return x + 1


@CMD.get_trace_decorator(comm_func="allreduce")
def _decorated_comm_add_one(x):
    return x + 1


def _run_decorated_with_mode(sync_mode):
    _DummyEvent.sync_calls = 0
    current_cmd = _DummyCmd(sync_mode)
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ) as global_sync, mock.patch.object(CMD, "get_current_cmd", return_value=current_cmd):
        result = _decorated_add_one(3)
    return result, current_cmd, global_sync


def _run_comm_decorated_with_mode(sync_mode, is_scaling_mode):
    _DummyEvent.sync_calls = 0
    current_cmd = _DummyCmd(sync_mode, is_scaling_mode=is_scaling_mode)
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ) as global_sync, mock.patch.object(CMD, "get_current_cmd", return_value=current_cmd):
        result = _decorated_comm_add_one(3)
    return result, current_cmd, global_sync


def test_trace_subop_sync_mode_event_uses_event_sync():
    result, current_cmd, global_sync = _run_decorated_with_mode("event")
    assert result == 4
    assert len(current_cmd.sub_operations) == 1
    assert _DummyEvent.sync_calls == 1
    assert global_sync.call_count == 0


def test_trace_subop_sync_mode_global_uses_global_sync():
    result, current_cmd, global_sync = _run_decorated_with_mode("global")
    assert result == 4
    assert len(current_cmd.sub_operations) == 1
    assert _DummyEvent.sync_calls == 0
    assert global_sync.call_count == 1


def test_trace_subop_sync_mode_invalid_fails_fast():
    with pytest.raises(ValueError, match="Unsupported trace_subop_sync_mode"):
        _run_decorated_with_mode("invalid-mode")


def test_scaling_comm_subop_is_metadata_only():
    result, current_cmd, global_sync = _run_comm_decorated_with_mode(
        "event", is_scaling_mode=True
    )
    assert result == 4
    assert len(current_cmd.sub_operations) == 1
    op_name, duration, attr_info, timestamp_ms = current_cmd.sub_operations[0]
    assert op_name == "_decorated_comm_add_one"
    assert duration == 0.0
    assert attr_info.get("comm_func") == "allreduce"
    assert timestamp_ms is not None
    assert _DummyEvent.sync_calls == 0
    assert global_sync.call_count == 0
