import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training import training as training_module
from megatron.training.arguments import parse_args, validate_args


def _base_cli():
    return [
        "test_trace_ddp_grad_overlap_args.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
        "--max-position-embeddings",
        "4096",
        "--seq-length",
        "128",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "1",
        "--train-iters",
        "1",
    ]


def _parse_and_validate(argv):
    with mock.patch.object(sys, "argv", argv):
        args = parse_args(ignore_unknown_args=True)
    with mock.patch.dict(os.environ, {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}, clear=False):
        return validate_args(args)


def test_trace_ddp_grad_overlap_flag_parses():
    args = _parse_and_validate(
        _base_cli() + ["--overlap-grad-reduce", "--trace-ddp-grad-overlap"]
    )
    assert args.trace_ddp_grad_overlap is True


def test_trace_ddp_grad_overlap_auto_enabled_when_ddp_overlap_is_on():
    args = _parse_and_validate(_base_cli() + ["--overlap-grad-reduce"])
    assert args.trace_ddp_grad_overlap is True


def test_do_trace_false_disables_trace_ddp_grad_overlap_auto_enable():
    args = _parse_and_validate(_base_cli() + ["--overlap-grad-reduce", "--do-trace", "False"])
    assert args.do_trace is False
    assert args.trace_ddp_grad_overlap is False


def test_do_trace_zero_disables_trace_ddp_grad_overlap_auto_enable():
    args = _parse_and_validate(_base_cli() + ["--overlap-grad-reduce", "--do-trace", "0"])
    assert args.do_trace is False
    assert args.trace_ddp_grad_overlap is False


def test_do_trace_invalid_value_is_rejected():
    with mock.patch.object(sys, "argv", _base_cli() + ["--do-trace", "maybe"]):
        with pytest.raises(SystemExit):
            parse_args(ignore_unknown_args=True)


def test_trace_ddp_grad_overlap_requires_overlap_grad_reduce():
    with pytest.raises(AssertionError, match="--overlap-grad-reduce"):
        _parse_and_validate(_base_cli() + ["--trace-ddp-grad-overlap"])


def test_runtime_validation_rejects_scaling_disable_ddp_wrap():
    args = SimpleNamespace(
        trace_ddp_grad_overlap=True,
        overlap_grad_reduce=True,
        is_scaling_mode=True,
        scaling_disable_ddp_wrap=True,
    )
    with pytest.raises(RuntimeError, match="scaling-disable-ddp-wrap"):
        training_module._validate_trace_ddp_grad_overlap_runtime_args(args)
