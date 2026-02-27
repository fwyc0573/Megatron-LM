import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training import training as training_module
from megatron.training.arguments import parse_args


class _DummyCmd:
    def __init__(self, **kwargs):
        self.name_cmd = kwargs["name_cmd"]
        self.rank_id = kwargs["rank_id"]
        self.stage_operations_trace_dict = kwargs["stage_operations_trace_dict"]
        self.micro_batch_ids_dict = kwargs["micro_batch_ids_dict"]

    def __enter__(self):
        self.micro_batch_ids_dict[self.name_cmd] += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stage_operations_trace_dict.setdefault(self.rank_id, []).append(self.name_cmd)



def _build_trace_args(enabled=True):
    return SimpleNamespace(
        trace_optimizer_microphases=enabled,
        stage_operations_trace={},
        simu_micro_batch_ids={"optimizer_step": -1},
        simu_start=True,
        trace_start=0,
        current_iter=0,
    )


def test_maybe_add_optimizer_microphase_batch_ids_adds_all_phases():
    batch_ids = {"optimizer_step": -1}
    training_module._maybe_add_optimizer_microphase_batch_ids(batch_ids)
    for phase_name in training_module.OPTIMIZER_MICROPHASE_OPS:
        assert batch_ids[phase_name] == -1


def test_optimizer_microphase_cmd_disabled_returns_null_context():
    args = _build_trace_args(enabled=False)
    context_manager = training_module._optimizer_microphase_cmd(
        args=args,
        phase_name="optimizer_main_update",
        rank_id="0",
        stage_id=0,
        mg_state="finalize",
        description="test",
    )
    assert type(context_manager).__name__ == "nullcontext"


def test_optimizer_microphase_cmd_invalid_phase_fails_fast():
    args = _build_trace_args(enabled=True)
    with pytest.raises(ValueError, match="Unsupported optimizer microphase"):
        training_module._optimizer_microphase_cmd(
            args=args,
            phase_name="bad_phase",
            rank_id="0",
            stage_id=0,
            mg_state="finalize",
            description="test",
        )


def test_optimizer_microphase_cmd_records_phase_order_and_presence():
    args = _build_trace_args(enabled=True)
    with mock.patch.object(training_module, "CMD", _DummyCmd):
        for phase_name in training_module.OPTIMIZER_MICROPHASE_OPS:
            with training_module._optimizer_microphase_cmd(
                args=args,
                phase_name=phase_name,
                rank_id="0",
                stage_id=0,
                mg_state="finalize",
                description="test",
            ):
                pass

    assert args.stage_operations_trace["0"] == list(training_module.OPTIMIZER_MICROPHASE_OPS)
    for phase_name in training_module.OPTIMIZER_MICROPHASE_OPS:
        assert args.simu_micro_batch_ids[phase_name] == 0


def test_trace_optimizer_microphases_default_disabled():
    test_argv = [
        "test_training_optimizer_microphase.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
    ]
    with mock.patch.object(sys, "argv", test_argv):
        args = parse_args(ignore_unknown_args=True)
    assert args.trace_optimizer_microphases is False


def test_trace_optimizer_microphases_enabled():
    test_argv = [
        "test_training_optimizer_microphase.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
        "--trace-optimizer-microphases",
    ]
    with mock.patch.object(sys, "argv", test_argv):
        args = parse_args(ignore_unknown_args=True)
    assert args.trace_optimizer_microphases is True


def test_scaling_replay_write_phase_default_pre_optimizer():
    test_argv = [
        "test_training_optimizer_microphase.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
    ]
    with mock.patch.object(sys, "argv", test_argv):
        args = parse_args(ignore_unknown_args=True)
    assert args.scaling_replay_write_phase == "pre_optimizer"


def test_scaling_replay_write_phase_can_be_post_optimizer():
    test_argv = [
        "test_training_optimizer_microphase.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
        "--scaling-replay-write-phase",
        "post_optimizer",
    ]
    with mock.patch.object(sys, "argv", test_argv):
        args = parse_args(ignore_unknown_args=True)
    assert args.scaling_replay_write_phase == "post_optimizer"


def test_should_defer_scaling_grad_replay_write_behaves_as_expected():
    args = SimpleNamespace(scaling_replay_write_phase="post_optimizer")
    assert training_module._should_defer_scaling_grad_replay_write(args) is True

    args.scaling_replay_write_phase = "pre_optimizer"
    assert training_module._should_defer_scaling_grad_replay_write(args) is False

    args.scaling_replay_write_phase = "bad_phase"
    with pytest.raises(ValueError, match="Unsupported --scaling-replay-write-phase value"):
        training_module._should_defer_scaling_grad_replay_write(args)


def test_scaling_align_scheduler_increment_default_disabled():
    test_argv = [
        "test_training_optimizer_microphase.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
    ]
    with mock.patch.object(sys, "argv", test_argv):
        args = parse_args(ignore_unknown_args=True)
    assert args.scaling_align_scheduler_increment is False


def test_scaling_align_scheduler_increment_enabled():
    test_argv = [
        "test_training_optimizer_microphase.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
        "--scaling-align-scheduler-increment",
    ]
    with mock.patch.object(sys, "argv", test_argv):
        args = parse_args(ignore_unknown_args=True)
    assert args.scaling_align_scheduler_increment is True


def test_get_scaling_scheduler_increment_dp_size_switch():
    args = SimpleNamespace(
        scaling_align_scheduler_increment=False, fake_dp=4, data_parallel_size=1
    )
    assert training_module._get_scaling_scheduler_increment_dp_size(args) == 4

    args.scaling_align_scheduler_increment = True
    assert training_module._get_scaling_scheduler_increment_dp_size(args) == 1
