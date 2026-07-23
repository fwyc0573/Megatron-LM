"""Integration test for DDP-overlap backward slowdown replay."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simu_engine import (
    IndividualTimeline,
    MODE_SIMULATE,
    Operation,
    SimulatorEngine,
    Stage,
    SubOperation,
    TimelinesManager,
)


class _ConstantPredictor:
    def __init__(self, slowdown_by_kernel: dict[str, float]) -> None:
        self._slowdown_by_kernel = slowdown_by_kernel

    def predict_slowdown_factor(self, kernel_name: str, ground_truth_ms: float, feature_row: dict) -> float:
        assert feature_row["ground_truth"] == pytest.approx(ground_truth_ms)
        return self._slowdown_by_kernel[kernel_name]


class _ConstantCommBackend:
    def __init__(self) -> None:
        self.requests = []

    def predict(self, request) -> float:
        self.requests.append(request)
        duration_by_group = {
            "tp": 6.0,
            "exp": 7.0,
        }
        return duration_by_group[request.group_kind]


def _build_manager() -> TimelinesManager:
    manager = TimelinesManager.__new__(TimelinesManager)
    manager.dependency_relationship = {"unit": []}
    manager.comm_matching_relationship = {}
    manager.can_overlap = True
    manager.running_mode = MODE_SIMULATE
    manager.global_waiting_pool = {}
    manager.global_finished_operations = {}
    manager.completed_cmd_operations_by_uid = {}
    manager.pending_ddp_wait_finish_times_by_uid = {}
    manager.pending_ddp_unassigned_finish_times_by_rank = {}
    manager.async_ddp_comm_available_by_rank = {}
    manager.slowdown_enabled = True
    manager.slowdown_assets = SimpleNamespace(
        backward_kernel_blueprints={
            "cmd-bwd-1": {
                "rank": 0,
                "stage_id": 0,
                "batch_id": 0,
                "iter_id": 7,
                "mg_state": "steady",
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
        },
        kernel_features={
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
        },
    )
    manager.slowdown_predictor = _ConstantPredictor({"k1": 1.0, "k2": 1.0})
    manager.slowdown_runtime_comm_schedules_by_uid = {}
    manager.slowdown_processed_backward_cmd_uids = set()
    manager.slowdown_trigger_cmd_uids = {"cmd-bwd-1"}
    manager.simulator_config = SimpleNamespace(
        slowdown=SimpleNamespace(max_iters=50, tol_ms=1e-6)
    )
    manager.cc_backend = _ConstantCommBackend()
    manager.cc_estimator = None
    manager._predict_profile_overlap_comm_duration = lambda op: float(op.trace_metadata["synthetic_duration_ms"])
    return manager


def _build_timeline(operations: list[Operation]) -> IndividualTimeline:
    rank_stub = SimpleNamespace(
        dp_groups=[0, 1],
        pp_groups=[0],
        tp_groups=[0],
        ep_groups=[0],
        exp_groups=[0],
        cp_groups=None,
        dp_modulo_exp_groups=[0],
    )
    stage = Stage(wrank_id=0, rank=rank_stub, stage_id=0, framework="megatron-lm")
    stage.operations_list = operations
    timeline = IndividualTimeline(stage, can_overlap=True)
    timeline.stage_kind = "unit"
    return timeline


def _build_overlay_comm(
    comm_uid: str,
    wait_cmd_uid: str,
    synthetic_duration_ms: float,
    *,
    wrank_id: int = 0,
    trigger_cmd_uid: str = "cmd-bwd-1",
) -> Operation:
    return Operation(
        name="dp_allreduce",
        duration=0.0,
        batch_id=0,
        wrank_id=wrank_id,
        stage_id=0,
        mg_state="steady",
        op_kind="comm",
        group_kind="dp",
        cmd_uid=comm_uid,
        trace_metadata={
            "trace_event_type": "ddp_grad_comm",
            "metadata_only": True,
            "trigger_cmd_uid": trigger_cmd_uid,
            "wait_cmd_uid": wait_cmd_uid,
            "synthetic_duration_ms": synthetic_duration_ms,
        },
        name_with_id=f"dp_allreduce_{comm_uid}",
    )


def _build_legacy_comp_sub_operation(wrank_id: int, index: int) -> SubOperation:
    return SubOperation(
        name=f"legacy_backward_comp_{index}",
        op_kind="comp",
        duration=2.0,
        wrank_id=wrank_id,
        group_kind=None,
        description="Legacy profile compute sub-operation",
        start_time=float(index),
        pt_start_time=0.0,
        pt_duration=8.0,
        tensor_shape=None,
        tensor_dtype=None,
        trace_src_func=None,
        comm_func=None,
        name_with_id=f"legacy_backward_comp_{index}",
    )


def _build_legacy_comm_sub_operation(
    *,
    wrank_id: int,
    name: str,
    group_kind: str,
    comm_func: str,
    index: int,
    tensor_shape: list[int],
) -> SubOperation:
    return SubOperation(
        name=name,
        op_kind="comm",
        duration=0.0,
        wrank_id=wrank_id,
        group_kind=group_kind,
        description=f"{name}|shape={tensor_shape}|dtype=torch.float16",
        start_time=float(index),
        pt_start_time=0.0,
        pt_duration=8.0,
        tensor_shape=tensor_shape,
        tensor_dtype="torch.float16",
        trace_src_func=f"trace_{name}",
        comm_func=comm_func,
        name_with_id=f"{name}_{index}",
    )


def _build_legacy_backward_sub_operations(wrank_id: int) -> list[SubOperation]:
    return [
        _build_legacy_comp_sub_operation(wrank_id, 0),
        _build_legacy_comm_sub_operation(
            wrank_id=wrank_id,
            name="tp_allgather",
            group_kind="tp",
            comm_func="allgather",
            index=1,
            tensor_shape=[8, 16],
        ),
        _build_legacy_comp_sub_operation(wrank_id, 1),
        _build_legacy_comm_sub_operation(
            wrank_id=wrank_id,
            name="exp_all_to_all",
            group_kind="exp",
            comm_func="all_to_all",
            index=2,
            tensor_shape=[4, 32],
        ),
        _build_legacy_comp_sub_operation(wrank_id, 2),
    ]


def _build_constructed_trace_backed_stage() -> Stage:
    rank_stub = SimpleNamespace(
        world_rank=0,
        _get_pp_local_rank=lambda: 0,
        dp_groups=[0, 1],
        pp_groups=[0],
        tp_groups=[0],
        ep_groups=[0],
        exp_groups=[0],
        cp_groups=None,
        dp_modulo_exp_groups=[0],
    )
    schedule_stage = Stage(
        wrank_id=0,
        rank=rank_stub,
        stage_id=0,
        framework="megatron-lm",
    )
    schedule_stage.operations_list = [
        Operation(
            name="backward_step",
            duration=None,
            batch_id=0,
            mg_state="steady",
            op_kind="comp",
        )
    ]

    trace_stage = Stage(
        wrank_id=0,
        rank=rank_stub,
        stage_id=0,
        framework="megatron-lm",
    )
    trace_stage.operations_list = [
        Operation(
            name="backward_step",
            duration=8.0,
            end_timestamp=8.0,
            batch_id=0,
            wrank_id=0,
            stage_id=0,
            mg_state="steady",
            op_kind="comp",
            cmd_uid="cmd-bwd-1",
            trace_metadata={"trace_event_type": "backward_step"},
        ),
        _build_overlay_comm("comm-1", "wait-1", 3.0),
        _build_overlay_comm("comm-2", "wait-1", 3.0),
    ]

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
        slowdown=SimpleNamespace(enabled=True)
    )

    complete_stages, _ = engine._init_3d_parallel_all_ranks(
        stages_or_wranks_dict={0: schedule_stage},
        rank_instances_dict={0: rank_stub},
        trace_stages_dict={0: trace_stage},
        torch_graph_stage_op_dict={
            0: {
                "backward_step": {
                    "duration": 8.0,
                    "sub_ops_list": _build_legacy_backward_sub_operations(0),
                }
            }
        },
    )
    assert len(complete_stages) == 1
    return complete_stages[0]


def test_constructed_trace_backed_backward_replays_without_legacy_conflict() -> None:
    constructed_stage = _build_constructed_trace_backed_stage()
    timeline = IndividualTimeline(constructed_stage, can_overlap=True)
    timeline.stage_kind = "unit"
    manager = _build_manager()
    manager.strategy = "no-pipelining"
    manager.stages_timeline_process_dict = {0: timeline}

    manager._replay_profile_no_pipelining()

    assert len(constructed_stage.operations_list) == 5
    assert [operation.name for operation in constructed_stage.operations_list] == [
        "backward_step",
        "tp_allgather",
        "exp_all_to_all",
        "dp_allreduce",
        "dp_allreduce",
    ]
    assert all(
        isinstance(operation, SubOperation) and operation.op_kind == "comm"
        for operation in constructed_stage.operations_list[1:3]
    )
    assert not any(
        isinstance(operation, SubOperation) and operation.op_kind == "comp"
        for operation in constructed_stage.operations_list
    )

    backward_operation = next(
        operation
        for operation in timeline.final_merge_timeline
        if operation.name == "backward_step"
    )
    generic_comm_operations = [
        operation
        for operation in timeline.final_merge_timeline
        if isinstance(operation, SubOperation)
    ]
    overlay_operations = [
        operation
        for operation in timeline.final_merge_timeline
        if operation.trace_metadata.get("trace_event_type") == "ddp_grad_comm"
    ]
    assert manager.slowdown_processed_backward_cmd_uids == {(0, "cmd-bwd-1")}
    assert backward_operation.duration > 8.0
    assert backward_operation.hidden_duration is None
    assert backward_operation.trace_metadata["slowdown_applied"] is True
    assert [operation.name for operation in generic_comm_operations] == [
        "tp_allgather",
        "exp_all_to_all",
    ]
    assert [operation.group_kind for operation in generic_comm_operations] == ["tp", "exp"]
    assert [operation.tensor_shape for operation in generic_comm_operations] == [
        [8, 16],
        [4, 32],
    ]
    assert [operation.cmd_uid for operation in overlay_operations] == ["comm-1", "comm-2"]
    assert all(
        operation.trace_metadata["trigger_cmd_uid"] == "cmd-bwd-1"
        for operation in overlay_operations
    )
    assert manager.slowdown_runtime_comm_schedules_by_uid == {}


def _build_four_rank_constructed_stages() -> list[Stage]:
    schedule_stages = {}
    trace_stages = {}
    rank_instances = {}
    torch_graph = {}

    for wrank_id in range(4):
        tp_group = [0, 1] if wrank_id < 2 else [2, 3]
        exp_group = [0, 2] if wrank_id % 2 == 0 else [1, 3]
        rank_stub = SimpleNamespace(
            world_rank=wrank_id,
            _get_pp_local_rank=lambda: 0,
            dp_groups=[wrank_id],
            pp_groups=[wrank_id],
            tp_groups=tp_group,
            ep_groups=[wrank_id],
            exp_groups=exp_group,
            cp_groups=None,
            dp_modulo_exp_groups=[wrank_id],
        )
        rank_instances[wrank_id] = rank_stub

        schedule_stage = Stage(
            wrank_id=wrank_id,
            rank=rank_stub,
            stage_id=0,
            framework="megatron-lm",
        )
        schedule_stage.operations_list = [
            Operation(
                name="backward_step",
                duration=None,
                batch_id=0,
                mg_state="steady",
                op_kind="comp",
            )
        ]
        schedule_stages[wrank_id] = schedule_stage

        cmd_uid = f"cmd-bwd-{wrank_id}"
        trace_stage = Stage(
            wrank_id=wrank_id,
            rank=rank_stub,
            stage_id=0,
            framework="megatron-lm",
        )
        trace_stage.operations_list = [
            Operation(
                name="backward_step",
                duration=8.0,
                end_timestamp=8.0,
                batch_id=0,
                wrank_id=wrank_id,
                stage_id=0,
                mg_state="steady",
                op_kind="comp",
                cmd_uid=cmd_uid,
                trace_metadata={"trace_event_type": "backward_step"},
            ),
            _build_overlay_comm(
                f"comm-{wrank_id}-1",
                f"wait-{wrank_id}",
                1.0,
                wrank_id=wrank_id,
                trigger_cmd_uid=cmd_uid,
            ),
            _build_overlay_comm(
                f"comm-{wrank_id}-2",
                f"wait-{wrank_id}",
                1.0,
                wrank_id=wrank_id,
                trigger_cmd_uid=cmd_uid,
            ),
        ]
        trace_stages[wrank_id] = trace_stage
        torch_graph[wrank_id] = {
            "backward_step": {
                "duration": 8.0,
                "sub_ops_list": _build_legacy_backward_sub_operations(wrank_id),
            }
        }

    engine = SimulatorEngine.__new__(SimulatorEngine)
    engine.running_mode = MODE_SIMULATE
    engine.optimization_enabled = False
    engine.moe_rank_selection = "all"
    engine.mpu = SimpleNamespace(
        world_size=4,
        pp_size=1,
        tp_size=2,
        dp_size=1,
        exp_size=2,
        ep_size=1,
    )
    engine.simulator_config = SimpleNamespace(
        slowdown=SimpleNamespace(enabled=True)
    )

    complete_stages, _ = engine._init_3d_parallel_all_ranks(
        stages_or_wranks_dict=schedule_stages,
        rank_instances_dict=rank_instances,
        trace_stages_dict=trace_stages,
        torch_graph_stage_op_dict=torch_graph,
    )
    return complete_stages


def _build_four_rank_manager(stages: list[Stage]) -> TimelinesManager:
    manager = _build_manager()
    manager.optimization_enabled = False
    manager.is_moe_model = True
    manager.selected_ranks = {0, 1, 2, 3}
    manager.mpu_info = None
    manager.slowdown_assets.backward_kernel_blueprints = {
        f"cmd-bwd-{wrank_id}": {
            "rank": wrank_id,
            "stage_id": 0,
            "batch_id": 0,
            "iter_id": 7,
            "mg_state": "steady",
            "baseline_duration_ms": 8.0,
            "kernels": [
                {"kernel_name": "k1", "start_offset_ms": 0.0, "baseline_duration_ms": 4.0},
                {"kernel_name": "k2", "start_offset_ms": 4.0, "baseline_duration_ms": 4.0},
            ],
            "launch_markers": [
                {
                    "comm_uid": f"comm-{wrank_id}-1",
                    "baseline_offset_ms": 1.0,
                    "bucket_id": 0,
                    "buffer_id": 0,
                },
                {
                    "comm_uid": f"comm-{wrank_id}-2",
                    "baseline_offset_ms": 3.0,
                    "bucket_id": 1,
                    "buffer_id": 0,
                },
            ],
        }
        for wrank_id in range(4)
    }
    manager.slowdown_predictor = _ConstantPredictor({"k1": 0.0, "k2": 0.0})
    manager.slowdown_trigger_cmd_uids = {
        f"cmd-bwd-{wrank_id}" for wrank_id in range(4)
    }
    manager.stages_timeline_process_dict = {}
    for stage in stages:
        timeline = IndividualTimeline(stage, can_overlap=True)
        timeline.stage_kind = "unit"
        manager.stages_timeline_process_dict[stage.wrank_id] = timeline
    return manager


def test_comm_only_projection_schedules_four_rank_tp_exp_collectives() -> None:
    constructed_stages = _build_four_rank_constructed_stages()
    assert all(len(stage.operations_list) == 5 for stage in constructed_stages)
    assert all(stage.operations_list[0].duration == pytest.approx(8.0) for stage in constructed_stages)
    assert all(stage.operations_list[0].hidden_duration is None for stage in constructed_stages)
    assert all(
        [operation.name for operation in stage.operations_list]
        == [
            "backward_step",
            "tp_allgather",
            "exp_all_to_all",
            "dp_allreduce",
            "dp_allreduce",
        ]
        for stage in constructed_stages
    )
    assert all(
        [operation.wrank_id for operation in stage.operations_list[1:3]]
        == [stage.wrank_id, stage.wrank_id]
        for stage in constructed_stages
    )

    manager = _build_four_rank_manager(constructed_stages)
    timelines = manager.stages_timeline_process_dict

    for operation_index in range(5):
        for wrank_id in range(4):
            operation = timelines[wrank_id].waiting_queue.popleft()
            manager._add_operation_to_timeline(timelines[wrank_id], operation)

    assert manager.global_waiting_pool == {}
    assert manager.slowdown_runtime_comm_schedules_by_uid == {}
    assert manager.slowdown_processed_backward_cmd_uids == {
        (0, "cmd-bwd-0"),
        (1, "cmd-bwd-1"),
        (2, "cmd-bwd-2"),
        (3, "cmd-bwd-3"),
    }
    assert len(manager.cc_backend.requests) == 4
    assert sorted(tuple(request.comm_group) for request in manager.cc_backend.requests) == [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ]

    for timeline in timelines.values():
        generic_comm = [
            operation
            for operation in timeline.comm_timeline
            if isinstance(operation, SubOperation)
        ]
        overlays = [
            operation
            for operation in timeline.comm_timeline
            if operation.trace_metadata.get("trace_event_type") == "ddp_grad_comm"
        ]
        assert [operation.name for operation in generic_comm] == [
            "tp_allgather",
            "exp_all_to_all",
        ]
        assert [operation.join_time for operation in generic_comm] == [0.0, 6.0]
        assert [operation.duration for operation in generic_comm] == [6.0, 7.0]
        assert [operation.finish_time for operation in generic_comm] == [6.0, 13.0]
        assert [operation.join_time for operation in overlays] == [1.0, 3.0]
        assert [operation.finish_time for operation in overlays] == [2.0, 4.0]
        assert max(operation.finish_time for operation in timeline.final_merge_timeline) == 13.0


def test_backward_slowdown_delays_bucket_launch_and_preserves_finalize_wait() -> None:
    manager = _build_manager()
    timeline = _build_timeline(
        [
            Operation(
                name="backward_step",
                duration=8.0,
                batch_id=0,
                wrank_id=0,
                stage_id=0,
                mg_state="steady",
                op_kind="comp",
                cmd_uid="cmd-bwd-1",
                trace_metadata={},
            ),
            _build_overlay_comm("comm-1", "wait-1", 3.0),
            _build_overlay_comm("comm-2", "wait-1", 3.0),
            Operation(
                name="dp_allreduce",
                duration=0.25,
                batch_id=0,
                wrank_id=0,
                stage_id=0,
                mg_state="finalize",
                op_kind="comp",
                cmd_uid="wait-1",
                op_semantics="metadata_placeholder",
                trace_metadata={"finalize_base_duration_ms": 0.25},
            ),
        ]
    )

    backward_op = timeline.waiting_queue.popleft()
    manager._add_operation_to_timeline(timeline, backward_op)
    comm1_op = timeline.waiting_queue.popleft()
    manager._add_operation_to_timeline(timeline, comm1_op)
    comm2_op = timeline.waiting_queue.popleft()
    manager._add_operation_to_timeline(timeline, comm2_op)
    wait_op = timeline.waiting_queue.popleft()
    manager._add_operation_to_timeline(timeline, wait_op)

    assert backward_op.duration > 8.0
    assert comm1_op.join_time > 2.0
    assert comm2_op.join_time > 6.0
    assert comm2_op.join_time > comm1_op.join_time
    assert wait_op.join_time == pytest.approx(backward_op.finish_time)
    assert wait_op.finish_time == pytest.approx(max(wait_op.join_time + 0.25, comm2_op.finish_time), abs=0.01)
    assert wait_op.duration == pytest.approx(wait_op.finish_time - wait_op.join_time, abs=0.01)


def test_replay_profile_no_pipelining_applies_backward_slowdown() -> None:
    manager = _build_manager()
    manager.strategy = 'no-pipelining'

    timeline = _build_timeline(
        [
            Operation(
                name="backward_step",
                duration=8.0,
                end_timestamp=8.0,
                batch_id=0,
                wrank_id=0,
                stage_id=0,
                mg_state="steady",
                op_kind="comp",
                cmd_uid="cmd-bwd-1",
                trace_metadata={},
            ),
            _build_overlay_comm("comm-1", "wait-1", 3.0),
            _build_overlay_comm("comm-2", "wait-1", 3.0),
            Operation(
                name="dp_allreduce",
                duration=0.25,
                batch_id=0,
                wrank_id=0,
                stage_id=0,
                mg_state="finalize",
                op_kind="comp",
                cmd_uid="wait-1",
                op_semantics="metadata_placeholder",
                trace_metadata={"finalize_base_duration_ms": 0.25},
            ),
        ]
    )
    manager.stages_timeline_process_dict = {0: timeline}

    manager._replay_profile_no_pipelining()

    backward_op = next(op for op in timeline.final_merge_timeline if op.name == 'backward_step')
    comm_ops = [op for op in timeline.final_merge_timeline if op.op_kind == 'comm']
    wait_op = next(op for op in timeline.final_merge_timeline if op.name == 'dp_allreduce' and op.op_kind == 'comp')

    assert manager.slowdown_processed_backward_cmd_uids == {(0, 'cmd-bwd-1')}
    assert backward_op.duration > 8.0
    assert len(comm_ops) == 2
    assert comm_ops[0].join_time > 2.0
    assert comm_ops[1].join_time > 6.0
    assert wait_op.join_time == pytest.approx(backward_op.finish_time)
    assert wait_op.finish_time == pytest.approx(max(wait_op.join_time + 0.25, comm_ops[-1].finish_time), abs=0.01)


def _build_shared_uid_timeline(wrank_id: int, synthetic_duration_ms: float) -> IndividualTimeline:
    rank_stub = SimpleNamespace(
        world_rank=wrank_id,
        dp_groups=[wrank_id],
        pp_groups=[wrank_id],
        tp_groups=[wrank_id],
        ep_groups=[wrank_id],
        exp_groups=[wrank_id],
        cp_groups=None,
        dp_modulo_exp_groups=[wrank_id],
    )
    stage = Stage(wrank_id=wrank_id, rank=rank_stub, stage_id=0, framework="megatron-lm")
    stage.operations_list = [
        Operation(
            name="backward_step",
            duration=8.0,
            batch_id=0,
            wrank_id=wrank_id,
            stage_id=0,
            mg_state="steady",
            op_kind="comp",
            cmd_uid="cmd-shared",
            trace_metadata={},
        ),
        _build_overlay_comm(
            "comm-shared-1",
            "wait-shared",
            synthetic_duration_ms,
            wrank_id=wrank_id,
            trigger_cmd_uid="cmd-shared",
        ),
        _build_overlay_comm(
            "comm-shared-2",
            "wait-shared",
            synthetic_duration_ms,
            wrank_id=wrank_id,
            trigger_cmd_uid="cmd-shared",
        ),
    ]
    timeline = IndividualTimeline(stage, can_overlap=True)
    timeline.stage_kind = "unit"
    return timeline


def _build_shared_uid_manager() -> TimelinesManager:
    manager = _build_manager()
    manager.slowdown_assets.backward_kernel_blueprints = {
        "cmd-shared": {
            "rank": 0,
            "stage_id": 0,
            "batch_id": 0,
            "iter_id": 7,
            "mg_state": "steady",
            "baseline_duration_ms": 8.0,
            "kernels": [
                {"kernel_name": "k1", "start_offset_ms": 0.0, "baseline_duration_ms": 4.0},
                {"kernel_name": "k2", "start_offset_ms": 4.0, "baseline_duration_ms": 4.0},
            ],
            "launch_markers": [
                {"comm_uid": "comm-shared-1", "baseline_offset_ms": 2.0, "bucket_id": 0, "buffer_id": 0},
                {"comm_uid": "comm-shared-2", "baseline_offset_ms": 6.0, "bucket_id": 1, "buffer_id": 0},
            ],
        }
    }
    manager.slowdown_trigger_cmd_uids = {"cmd-shared"}
    manager.slowdown_runtime_comm_schedules_by_uid = {}
    manager.slowdown_processed_backward_cmd_uids = set()
    manager.completed_cmd_operations_by_uid = {}
    manager.pending_ddp_wait_finish_times_by_uid = {}
    return manager


def test_cross_rank_reused_uid_has_rank_local_slowdown_state() -> None:
    manager = _build_shared_uid_manager()
    timelines = {
        0: _build_shared_uid_timeline(0, synthetic_duration_ms=3.0),
        1: _build_shared_uid_timeline(1, synthetic_duration_ms=4.0),
    }

    for wrank_id in (0, 1):
        manager._add_operation_to_timeline(timelines[wrank_id], timelines[wrank_id].waiting_queue.popleft())

    for wrank_id in (0, 1):
        timeline = timelines[wrank_id]
        while timeline.waiting_queue:
            manager._add_operation_to_timeline(timeline, timeline.waiting_queue.popleft())

    assert manager.slowdown_processed_backward_cmd_uids == {
        (0, "cmd-shared"),
        (1, "cmd-shared"),
    }
    assert manager.slowdown_runtime_comm_schedules_by_uid == {}
    assert set(manager.pending_ddp_wait_finish_times_by_uid) == {
        (0, "wait-shared"),
        (1, "wait-shared"),
    }
    assert [operation.duration for operation in timelines[0].comm_timeline] == [3.0, 3.0]
    assert [operation.duration for operation in timelines[1].comm_timeline] == [4.0, 4.0]


def test_same_rank_reused_uid_remains_fail_fast() -> None:
    manager = _build_shared_uid_manager()
    timeline = _build_shared_uid_timeline(0, synthetic_duration_ms=3.0)
    manager._add_operation_to_timeline(timeline, timeline.waiting_queue.popleft())

    duplicate_operation = Operation(
        name="backward_step",
        duration=8.0,
        batch_id=1,
        wrank_id=0,
        stage_id=0,
        mg_state="steady",
        op_kind="comp",
        cmd_uid="cmd-shared",
        trace_metadata={},
    )
    with pytest.raises(ValueError, match="Backward slowdown already processed for cmd_uid=cmd-shared"):
        manager._add_trace_driven_backward_slowdown(
            timeline,
            duplicate_operation,
            join_time=0.0,
            current_format_operantion_name="duplicate",
        )
