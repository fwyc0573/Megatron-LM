"""Unit tests for DDP overlap slowdown integration in sim-engine."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from typing import Dict

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import simu_main
from src.core.simu_engine import (
    MODE_SIMULATE,
    Operation,
    SimulatorEngine,
    Stage,
    SubOperation,
    TimelinesManager,
)


class _ConstantPredictor:
    def __init__(self, slowdown_by_kernel: Dict[str, float]) -> None:
        self._slowdown_by_kernel = slowdown_by_kernel

    def predict_slowdown_factor(self, kernel_name: str, ground_truth_ms: float, feature_row: dict) -> float:
        assert feature_row["ground_truth"] == ground_truth_ms
        return self._slowdown_by_kernel[kernel_name]


def _build_legacy_sub_operation(operation_name: str, index: int = 0) -> SubOperation:
    return SubOperation(
        name=f"legacy_{operation_name}_comp_{index}",
        op_kind="comp",
        duration=1.5,
        wrank_id=0,
        group_kind=None,
        description="Legacy profile sub-operation",
        start_time=0.0,
        pt_start_time=0.0,
        pt_duration=1.5,
        tensor_shape=None,
        tensor_dtype=None,
        trace_src_func=None,
        comm_func=None,
        name_with_id=f"legacy_{operation_name}_comp_{index}",
    )


def _build_legacy_comm_sub_operation(
    *,
    name: str,
    group_kind: str,
    comm_func: str,
    index: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> SubOperation:
    return SubOperation(
        name=name,
        op_kind="comm",
        duration=0.0,
        wrank_id=0,
        group_kind=group_kind,
        description=f"{name}|shape={tensor_shape}|dtype={tensor_dtype}",
        start_time=float(index),
        pt_start_time=0.0,
        pt_duration=25.63,
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
        trace_src_func=f"trace_{name}",
        comm_func=comm_func,
        name_with_id=f"{name}_{index}",
    )


def _build_legacy_backward_sub_operations() -> list[SubOperation]:
    return [
        _build_legacy_sub_operation("backward_step", 0),
        _build_legacy_comm_sub_operation(
            name="tp_allgather",
            group_kind="tp",
            comm_func="allgather",
            index=1,
            tensor_shape=[8, 16],
            tensor_dtype="torch.float16",
        ),
        _build_legacy_sub_operation("backward_step", 1),
        _build_legacy_comm_sub_operation(
            name="exp_all_to_all",
            group_kind="exp",
            comm_func="all_to_all",
            index=2,
            tensor_shape=[4, 32],
            tensor_dtype="torch.bfloat16",
        ),
        _build_legacy_sub_operation("backward_step", 2),
    ]


def _build_constructed_operations(
    *,
    operation_name: str = "backward_step",
    slowdown_enabled: bool = True,
    include_trace: bool = True,
    include_overlays: bool = True,
    trace_duration: float | None = 25.63,
    legacy_sub_operations: list[SubOperation] | None = None,
) -> list[Operation]:
    rank_stub = SimpleNamespace(
        world_rank=0,
        _get_pp_local_rank=lambda: 0,
    )
    schedule_batch_id = 31
    schedule_operation = Operation(
        name=operation_name,
        duration=None,
        batch_id=schedule_batch_id,
        mg_state="steady",
        op_kind="comp",
    )
    schedule_stage = Stage(
        wrank_id=0,
        rank=rank_stub,
        stage_id=0,
        framework="megatron-lm",
    )
    schedule_stage.operations_list = [schedule_operation]

    trace_stages_dict = None
    if include_trace:
        trace_batch_id = 0 if operation_name == "backward_step" else schedule_batch_id
        trace_operation = Operation(
            name=operation_name,
            duration=trace_duration,
            end_timestamp=100.0,
            batch_id=trace_batch_id,
            wrank_id=0,
            stage_id=0,
            mg_state="steady",
            op_kind="comp",
            cmd_uid=f"cmd-{operation_name}",
            trace_metadata={"trace_event_type": operation_name, "source": "trace"},
        )
        trace_operations = [trace_operation]
        if operation_name == "backward_step" and include_overlays:
            for index in range(2):
                trace_operations.append(
                    Operation(
                        name="dp_allreduce",
                        duration=0.0,
                        batch_id=trace_batch_id,
                        wrank_id=0,
                        stage_id=0,
                        mg_state="steady",
                        op_kind="comm",
                        group_kind="dp",
                        cmd_uid=f"comm-{index + 1}",
                        trace_metadata={
                            "trace_event_type": "ddp_grad_comm",
                            "metadata_only": True,
                            "trigger_cmd_uid": "cmd-backward_step",
                            "launch_timestamp_ms": 90.0 + index,
                            "synthetic_duration_ms": 3.0,
                        },
                        name_with_id=f"dp_allreduce_comm-{index + 1}",
                    )
                )

        trace_stage = Stage(
            wrank_id=0,
            rank=rank_stub,
            stage_id=0,
            framework="megatron-lm",
        )
        trace_stage.operations_list = trace_operations
        trace_stages_dict = {0: trace_stage}

    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.running_mode = MODE_SIMULATE
    engine.optimization_enabled = False
    engine.moe_rank_selection = "all"
    engine.mpu = SimpleNamespace(
        world_size=1,
        pp_size=1,
        tp_size=1,
        dp_size=1,
        exp_size=1,
        ep_size=1,
    )
    engine.simulator_config = SimpleNamespace(
        slowdown=SimpleNamespace(enabled=slowdown_enabled)
    )

    if legacy_sub_operations is None:
        if operation_name == "backward_step":
            legacy_sub_operations = _build_legacy_backward_sub_operations()
        else:
            legacy_sub_operations = [_build_legacy_sub_operation(operation_name)]

    complete_stages, _ = engine._init_3d_parallel_all_ranks(
        stages_or_wranks_dict={0: schedule_stage},
        rank_instances_dict={0: rank_stub},
        trace_stages_dict=trace_stages_dict,
        torch_graph_stage_op_dict={
            0: {
                operation_name: {
                    "duration": 25.63,
                    "sub_ops_list": legacy_sub_operations,
                }
            }
        },
    )
    assert len(complete_stages) == 1
    return complete_stages[0].operations_list


def test_trace_backed_slowdown_backward_preserves_only_legacy_communication() -> None:
    operations = _build_constructed_operations()

    backward_operation = operations[0]
    assert backward_operation.hidden_duration is None
    assert backward_operation.duration == pytest.approx(25.63)
    assert backward_operation.cmd_uid == "cmd-backward_step"
    assert backward_operation.end_timestamp == pytest.approx(100.0)
    assert backward_operation.trace_metadata == {
        "trace_event_type": "backward_step",
        "source": "trace",
    }
    assert len(operations) == 5

    comm_children = operations[1:3]
    assert all(isinstance(operation, SubOperation) for operation in comm_children)
    assert all(operation.op_kind == "comm" for operation in comm_children)
    assert [operation.name for operation in comm_children] == [
        "tp_allgather",
        "exp_all_to_all",
    ]
    assert [operation.group_kind for operation in comm_children] == ["tp", "exp"]
    assert [operation.comm_func for operation in comm_children] == ["allgather", "all_to_all"]
    assert [operation.tensor_shape for operation in comm_children] == [[8, 16], [4, 32]]
    assert [operation.tensor_dtype for operation in comm_children] == [
        "torch.float16",
        "torch.bfloat16",
    ]
    assert [operation.description for operation in comm_children] == [
        "tp_allgather|shape=[8, 16]|dtype=torch.float16",
        "exp_all_to_all|shape=[4, 32]|dtype=torch.bfloat16",
    ]
    assert [operation.batch_id for operation in comm_children] == [31, 31]
    assert [operation.wrank_id for operation in comm_children] == [0, 0]
    assert [operation.stage_id for operation in comm_children] == [0, 0]
    assert [operation.mg_state for operation in comm_children] == ["steady", "steady"]
    assert [operation.start_time for operation in comm_children] == [1.0, 2.0]
    assert [operation.pt_start_time for operation in comm_children] == [0.0, 0.0]
    assert [operation.pt_duration for operation in comm_children] == [25.63, 25.63]
    assert [operation.trace_src_func for operation in comm_children] == [
        "trace_tp_allgather",
        "trace_exp_all_to_all",
    ]
    assert [operation.name_with_id for operation in comm_children] == [
        "tp_allgather_1_31_backward_step",
        "exp_all_to_all_2_31_backward_step",
    ]
    assert not any(
        isinstance(operation, SubOperation) and operation.op_kind == "comp"
        for operation in operations
    )

    overlay_operations = operations[3:]
    assert [operation.cmd_uid for operation in overlay_operations] == ["comm-1", "comm-2"]
    assert all(operation.batch_id == 31 for operation in overlay_operations)
    assert all(operation.mg_state == "steady" for operation in overlay_operations)
    assert all(
        operation.trace_metadata["trigger_cmd_uid"] == "cmd-backward_step"
        for operation in overlay_operations
    )
    assert all(
        operation.trace_metadata["trigger_batch_id"] == 31
        for operation in overlay_operations
    )


def test_schedule_only_backward_retains_legacy_profile_expansion() -> None:
    operations = _build_constructed_operations(
        include_trace=False,
        include_overlays=False,
    )

    assert len(operations) == 6
    assert operations[0].cmd_uid is None
    assert operations[0].duration == pytest.approx(0.01)
    assert operations[0].hidden_duration == pytest.approx(25.63)
    assert all(isinstance(operation, SubOperation) for operation in operations[1:])
    assert [operation.op_kind for operation in operations[1:]] == [
        "comp",
        "comm",
        "comp",
        "comm",
        "comp",
    ]


def test_slowdown_disabled_trace_backed_backward_retains_legacy_profile_expansion() -> None:
    operations = _build_constructed_operations(slowdown_enabled=False)

    assert len(operations) == 8
    assert operations[0].cmd_uid == "cmd-backward_step"
    assert operations[0].duration == pytest.approx(0.01)
    assert operations[0].hidden_duration == pytest.approx(25.63)
    assert all(isinstance(operation, SubOperation) for operation in operations[1:6])
    assert [operation.cmd_uid for operation in operations[6:]] == ["comm-1", "comm-2"]


def test_requested_slowdown_without_paired_overlay_retains_legacy_profile_expansion() -> None:
    operations = _build_constructed_operations(include_overlays=False)

    assert len(operations) == 6
    assert operations[0].cmd_uid == "cmd-backward_step"
    assert operations[0].duration == pytest.approx(0.01)
    assert operations[0].hidden_duration == pytest.approx(25.63)
    assert all(isinstance(operation, SubOperation) for operation in operations[1:])


def test_forward_operation_retains_legacy_profile_expansion_with_slowdown_requested() -> None:
    operations = _build_constructed_operations(
        operation_name="forward_step",
        include_overlays=False,
    )

    assert len(operations) == 2
    assert operations[0].cmd_uid == "cmd-forward_step"
    assert operations[0].duration == pytest.approx(0.01)
    assert operations[0].hidden_duration == pytest.approx(25.63)
    assert isinstance(operations[1], SubOperation)


def test_trace_backed_slowdown_backward_requires_trace_duration() -> None:
    with pytest.raises(ValueError, match="missing trace duration"):
        _build_constructed_operations(trace_duration=None)


def test_trace_backed_slowdown_backward_rejects_unknown_legacy_child_kind() -> None:
    legacy_sub_operations = _build_legacy_backward_sub_operations()
    legacy_sub_operations[2].op_kind = "unknown"

    with pytest.raises(ValueError, match="Unsupported legacy sub-operation kind"):
        _build_constructed_operations(legacy_sub_operations=legacy_sub_operations)


def test_fail_fast_validate_allows_slowdown_without_assets_dir_until_overlap_is_detected() -> None:
    args = simu_main._parse_args(
        [
            "--framework",
            "megatron-lm",
            "--mode",
            "simulate",
            "--schedule-dir",
            "schedule",
            "--database-dir",
            "database",
            "--world-size",
            "2",
            "--pp-size",
            "1",
            "--tp-size",
            "1",
            "--local-size",
            "2",
            "--trace-dir",
            "trace",
            "--enable-slowdown",
        ]
    )

    config = simu_main._fail_fast_validate(args)
    assert config.slowdown.enabled is True
    assert config.slowdown.assets_dir is None
    assert config.overlap.mode == "auto"


def test_fail_fast_validate_rejects_profile_mode_slowdown() -> None:
    args = simu_main._parse_args(
        [
            "--framework",
            "megatron-lm",
            "--mode",
            "profile",
            "--trace-dir",
            "trace",
            "--database-dir",
            "database",
            "--world-size",
            "2",
            "--pp-size",
            "1",
            "--tp-size",
            "1",
            "--local-size",
            "2",
            "--enable-slowdown",
            "--slowdown-assets-dir",
            "assets",
        ]
    )

    with pytest.raises(ValueError, match="simulate"):
        simu_main._fail_fast_validate(args)


def test_fail_fast_validate_accepts_explicit_overlap_mode() -> None:
    args = simu_main._parse_args(
        [
            "--framework",
            "megatron-lm",
            "--mode",
            "simulate",
            "--trace-dir",
            "trace",
            "--schedule-dir",
            "schedule",
            "--database-dir",
            "database",
            "--world-size",
            "2",
            "--pp-size",
            "1",
            "--tp-size",
            "1",
            "--local-size",
            "2",
            "--overlap-mode",
            "off",
        ]
    )

    config = simu_main._fail_fast_validate(args)
    assert config.overlap.mode == "off"


def test_fixed_point_kernel_duration_returns_baseline_without_overlap() -> None:
    duration_ms = TimelinesManager._fixed_point_kernel_duration_ms(
        ground_truth_ms=3.0,
        slowdown_factor=0.7,
        overlap_time_ms=0.0,
        max_iters=20,
        tol_ms=1e-6,
    )

    assert duration_ms == pytest.approx(3.0)


def test_fixed_point_kernel_duration_grows_with_overlap() -> None:
    duration_ms = TimelinesManager._fixed_point_kernel_duration_ms(
        ground_truth_ms=4.0,
        slowdown_factor=0.5,
        overlap_time_ms=2.0,
        max_iters=50,
        tol_ms=1e-6,
    )

    assert duration_ms > 4.0
    assert duration_ms < 6.0


def test_trace_driven_backward_requires_trace_cmd_uid() -> None:
    assert TimelinesManager._is_trace_driven_backward(
        SimpleNamespace(name="backward_step", cmd_uid="cmd-bwd-1")
    )
    assert not TimelinesManager._is_trace_driven_backward(
        SimpleNamespace(name="backward_step", cmd_uid=None)
    )
    assert not TimelinesManager._is_trace_driven_backward(
        SimpleNamespace(name="forward_step", cmd_uid="cmd-fwd-1")
    )


def _build_manager_for_init(monkeypatch, timeline, *, overlap_mode: str, slowdown_enabled: bool):
    monkeypatch.setattr(TimelinesManager, "_init_stages_timeline", lambda self: {0: timeline})
    return TimelinesManager(
        dependency_relationship={},
        comm_matching_relationship={},
        compelete_wranks_list=[SimpleNamespace(wrank_id=0)],
        running_mode=MODE_SIMULATE,
        simulator_config=SimpleNamespace(
            slowdown=SimpleNamespace(
                enabled=slowdown_enabled,
                assets_dir=None,
                model_path=None,
                scaler_path=None,
                max_iters=50,
                tol_ms=1e-6,
            ),
            overlap=SimpleNamespace(mode=overlap_mode),
        ),
    )


def test_timelines_manager_auto_enables_overlap_when_trace_contains_overlay(monkeypatch) -> None:
    overlay_operation = SimpleNamespace(
        trace_metadata={"trace_event_type": "ddp_grad_comm", "trigger_cmd_uid": "cmd-bwd-1"},
        op_semantics=None,
    )
    timeline = SimpleNamespace(waiting_queue=[overlay_operation], can_overlap=False)

    manager = _build_manager_for_init(
        monkeypatch,
        timeline,
        overlap_mode="auto",
        slowdown_enabled=False,
    )

    assert manager.can_overlap is True
    assert timeline.can_overlap is True


def test_timelines_manager_rejects_forced_overlap_off_when_trace_contains_overlay(monkeypatch) -> None:
    overlay_operation = SimpleNamespace(
        trace_metadata={"trace_event_type": "ddp_grad_comm", "trigger_cmd_uid": "cmd-bwd-1"},
        op_semantics=None,
    )
    timeline = SimpleNamespace(waiting_queue=[overlay_operation], can_overlap=False)

    with pytest.raises(ValueError, match="force-disabled"):
        _build_manager_for_init(
            monkeypatch,
            timeline,
            overlap_mode="off",
            slowdown_enabled=False,
        )


def test_timelines_manager_warns_and_disables_slowdown_without_overlap(monkeypatch, caplog) -> None:
    timeline = SimpleNamespace(waiting_queue=[], can_overlap=False)

    def _unexpected_load_assets(*_args, **_kwargs):
        raise AssertionError("slowdown assets should not load when overlap metadata is absent")

    monkeypatch.setattr("src.core.simu_engine.load_slowdown_assets", _unexpected_load_assets)

    with caplog.at_level("WARNING"):
        manager = _build_manager_for_init(
            monkeypatch,
            timeline,
            overlap_mode="auto",
            slowdown_enabled=True,
        )

    assert manager.slowdown_enabled is False
    assert "Slowdown requested but no trace-driven DDP overlap overlay is present" in caplog.text


def test_simulate_backward_slowdown_schedule_delays_second_comm_launch() -> None:
    blueprint = {
        "baseline_duration_ms": 8.0,
        "kernels": [
            {"kernel_name": "k1", "start_offset_ms": 0.0, "baseline_duration_ms": 4.0},
            {"kernel_name": "k2", "start_offset_ms": 4.0, "baseline_duration_ms": 4.0},
        ],
        "launch_markers": [
            {"comm_uid": "comm-1", "baseline_offset_ms": 2.0, "bucket_id": 0, "buffer_id": 0},
            {"comm_uid": "comm-2", "baseline_offset_ms": 6.0, "bucket_id": 1, "buffer_id": 0},
        ],
    }
    kernel_features = {
        "k1": {
            "Compute throughput": 1.0,
            "Memory throughput": 1.0,
            "DRAM throughput": 1.0,
            "Achieved occupancy": 1.0,
            "Maximum occupancy": 1.0,
            "L1 hit rate": 1.0,
            "L2 hit rate": 1.0,
        },
        "k2": {
            "Compute throughput": 1.0,
            "Memory throughput": 1.0,
            "DRAM throughput": 1.0,
            "Achieved occupancy": 1.0,
            "Maximum occupancy": 1.0,
            "L1 hit rate": 1.0,
            "L2 hit rate": 1.0,
        },
    }
    predictor = _ConstantPredictor({"k1": 1.0, "k2": 1.0})

    result = TimelinesManager._simulate_backward_slowdown_schedule(
        backward_start_time_ms=100.0,
        blueprint=blueprint,
        kernel_features=kernel_features,
        predictor=predictor,
        comm_duration_by_uid={"comm-1": 3.0, "comm-2": 3.0},
        max_iters=50,
        tol_ms=1e-6,
    )

    assert result["backward_duration_ms"] > blueprint["baseline_duration_ms"]
    comm_1 = result["comm_schedules"]["comm-1"]
    comm_2 = result["comm_schedules"]["comm-2"]
    assert comm_1["launch_time_ms"] > 102.0
    assert comm_2["launch_time_ms"] > 106.0
    assert comm_2["launch_time_ms"] > comm_1["launch_time_ms"]
    assert comm_2["finish_time_ms"] >= comm_2["launch_time_ms"] + 3.0


def test_ddp_overlap_comm_replay_remains_offset_based_when_slowdown_disabled() -> None:
    manager = TimelinesManager.__new__(TimelinesManager)
    manager.slowdown_enabled = False
    manager.completed_cmd_operations_by_uid = {}
    manager.pending_ddp_wait_finish_times_by_uid = {}
    manager.pending_ddp_unassigned_finish_times_by_rank = {}
    manager.async_ddp_comm_available_by_rank = {}
    manager._predict_profile_overlap_comm_duration = lambda _op: 3.0

    trigger_operation = type("TriggerOp", (), {})()
    trigger_operation.end_timestamp = 108.0
    trigger_operation.duration = 8.0
    trigger_operation.join_time = 200.0
    trigger_operation.trace_metadata = {"trace_event_type": "backward_step"}
    manager.completed_cmd_operations_by_uid[(0, "cmd-bwd-1")] = trigger_operation

    timeline = type("Timeline", (), {})()
    timeline.wrank_id = 0
    timeline.comm_timeline = []
    timeline.final_merge_timeline = []
    timeline._get_last_op_waiting_acc = lambda _ignore: 0

    operation = type("ReplayOp", (), {})()
    operation.name = "dp_allreduce"
    operation.name_with_id = "dp_allreduce_iter7_buffer0_bucket0"
    operation.cmd_uid = "comm-1"
    operation.trace_metadata = {
        "trace_event_type": "ddp_grad_comm",
        "metadata_only": True,
        "trigger_cmd_uid": "cmd-bwd-1",
        "launch_timestamp_ms": 103.0,
        "wait_cmd_uid": "wait-1",
    }
    operation.comm_set_join_time = lambda join_time: setattr(operation, "join_time", round(join_time, 2))
    operation.comm_set_waiting_finish_time = lambda waiting_time: (
        setattr(operation, "waiting_time", round(waiting_time, 2)),
        setattr(operation, "finish_time", round(operation.join_time + waiting_time + operation.duration, 2)),
    )
    operation.comm_set_waiting_acc_time = lambda waiting_time, waiting_acc: setattr(operation, "waiting_acc", waiting_acc + waiting_time)

    manager._add_trace_driven_async_ddp_overlap_comm(timeline, operation)

    assert operation.join_time == pytest.approx(203.0)
    assert operation.finish_time == pytest.approx(206.0)
    assert manager.pending_ddp_wait_finish_times_by_uid[(0, "wait-1")] == [pytest.approx(206.0)]


def test_simulate_backward_slowdown_schedule_preserves_residual_duration() -> None:
    blueprint = {
        "baseline_duration_ms": 10.0,
        "kernels": [
            {"kernel_name": "k1", "start_offset_ms": 0.0, "baseline_duration_ms": 2.0},
            {"kernel_name": "k2", "start_offset_ms": 2.0, "baseline_duration_ms": 2.0},
        ],
        "launch_markers": [],
    }
    kernel_features = {
        "k1": {
            "Compute throughput": 1.0,
            "Memory throughput": 1.0,
            "DRAM throughput": 1.0,
            "Achieved occupancy": 1.0,
            "Maximum occupancy": 1.0,
            "L1 hit rate": 1.0,
            "L2 hit rate": 1.0,
        },
        "k2": {
            "Compute throughput": 1.0,
            "Memory throughput": 1.0,
            "DRAM throughput": 1.0,
            "Achieved occupancy": 1.0,
            "Maximum occupancy": 1.0,
            "L1 hit rate": 1.0,
            "L2 hit rate": 1.0,
        },
    }
    predictor = _ConstantPredictor({"k1": 0.0, "k2": 0.0})

    result = TimelinesManager._simulate_backward_slowdown_schedule(
        backward_start_time_ms=50.0,
        blueprint=blueprint,
        kernel_features=kernel_features,
        predictor=predictor,
        comm_duration_by_uid={},
        max_iters=20,
        tol_ms=1e-6,
    )

    assert result["residual_duration_ms"] == pytest.approx(6.0)
    assert result["backward_duration_ms"] == pytest.approx(10.0)


def test_simulate_backward_slowdown_schedule_skips_kernel_without_features() -> None:
    blueprint = {
        "baseline_duration_ms": 4.0,
        "kernels": [
            {"kernel_name": "kernel_missing", "start_offset_ms": 0.0, "baseline_duration_ms": 4.0},
        ],
        "launch_markers": [],
    }

    class _UnexpectedPredictor:
        def predict_slowdown_factor(self, **_kwargs):
            raise AssertionError("predictor must not run without kernel features")

    result = TimelinesManager._simulate_backward_slowdown_schedule(
        backward_start_time_ms=10.0,
        blueprint=blueprint,
        kernel_features={},
        predictor=_UnexpectedPredictor(),
        comm_duration_by_uid={},
        max_iters=20,
        tol_ms=1e-6,
    )

    assert result["backward_duration_ms"] == pytest.approx(4.0)
    assert result["kernel_schedules"][0]["slowdown_factor"] == pytest.approx(0.0)
    assert result["kernel_schedules"][0]["slowdown_feature_source"] == "missing_skip"


def test_simulate_backward_slowdown_schedule_uses_unique_canonical_kernel_alias() -> None:
    blueprint = {
        "baseline_duration_ms": 4.0,
        "kernels": [
            {"kernel_name": "ns::kernel_a", "start_offset_ms": 0.0, "baseline_duration_ms": 4.0},
        ],
        "launch_markers": [],
    }
    kernel_features = {
        "kernel_a": {
            "Compute throughput": 1.0,
            "Memory throughput": 1.0,
            "DRAM throughput": 1.0,
            "Achieved occupancy": 1.0,
            "Maximum occupancy": 1.0,
            "L1 hit rate": 1.0,
            "L2 hit rate": 1.0,
        }
    }

    class _AliasPredictor:
        def predict_slowdown_factor(self, kernel_name, ground_truth_ms, feature_row):
            assert kernel_name == "kernel_a"
            assert feature_row["ground_truth"] == ground_truth_ms
            return 0.5

    result = TimelinesManager._simulate_backward_slowdown_schedule(
        backward_start_time_ms=10.0,
        blueprint=blueprint,
        kernel_features=kernel_features,
        predictor=_AliasPredictor(),
        comm_duration_by_uid={},
        max_iters=20,
        tol_ms=1e-6,
    )

    assert result["backward_duration_ms"] == pytest.approx(4.0)
    assert result["kernel_schedules"][0]["slowdown_factor"] == pytest.approx(0.5)
    assert result["kernel_schedules"][0]["slowdown_feature_source"] == "canonical_alias:kernel_a"


def test_simulate_backward_slowdown_schedule_skips_ambiguous_canonical_alias() -> None:
    blueprint = {
        "baseline_duration_ms": 4.0,
        "kernels": [
            {"kernel_name": "ns::kernel_a", "start_offset_ms": 0.0, "baseline_duration_ms": 4.0},
        ],
        "launch_markers": [],
    }
    kernel_features = {
        "module_a::kernel_a": {"Compute throughput": 1.0},
        "other_a::kernel_a": {"Compute throughput": 2.0},
    }

    class _UnexpectedPredictor:
        def predict_slowdown_factor(self, **_kwargs):
            raise AssertionError("predictor must not run for an ambiguous alias")

    result = TimelinesManager._simulate_backward_slowdown_schedule(
        backward_start_time_ms=10.0,
        blueprint=blueprint,
        kernel_features=kernel_features,
        predictor=_UnexpectedPredictor(),
        comm_duration_by_uid={},
        max_iters=20,
        tol_ms=1e-6,
    )

    assert result["backward_duration_ms"] == pytest.approx(4.0)
    assert result["kernel_schedules"][0]["slowdown_factor"] == pytest.approx(0.0)
    assert result["kernel_schedules"][0]["slowdown_feature_source"] == "missing_skip"
