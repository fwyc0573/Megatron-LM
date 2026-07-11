import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path('tests/e2e/compare_ddp_slowdown_reference.py')
spec = importlib.util.spec_from_file_location('compare_ddp_slowdown_reference', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _write_trace(path: Path, rank_id: int) -> None:
    stage_id = 0 if rank_id == 0 else 1
    mg_state = 'cooldown' if rank_id == 0 else 'steady'
    path.write_text(
        '\n'.join(
            [
                f"rank:{rank_id}:backward_step(stage_id={stage_id},batch_id=1,mg_state=steady,duration=11.0,description=None,group_kind=None,cmd_uid=cmd-other-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=101.0,sub_operations=[])",
                f"rank:{rank_id}:backward_step(stage_id={stage_id},batch_id=0,mg_state={mg_state},duration=50.0,description=None,group_kind=None,cmd_uid=cmd-bwd-{rank_id},op_semantics=None,finalize_base_duration_ms=None,input__shape=None,input__dtype=None,timestamp=150.0,sub_operations=[])",
                f"rank:{rank_id}:ddp_grad_comm(comm_uid=comm-random-a-{rank_id},iter=2,stage_id={stage_id},mg_state={mg_state},group_kind=dp,comm_func=allreduce,buffer_id=1,bucket_id=1,bucket_offset=0,bucket_numel=4,bucket_numel_unpadded=4,param_count=1,trigger_cmd_uid=cmd-bwd-{rank_id},trigger_op=backward_step,trigger_batch_id=0,trigger_timestamp_ms=120.0,launch_timestamp_ms=130.0,launch_source=param_hook,timing_domain=actual,metadata_only=False,status=completed,completion_observed_timestamp_ms=145.0,completion_source=cuda_event,wait_cmd_uid=cmd-wait-{rank_id},wait_start_timestamp_ms=None,wait_end_timestamp_ms=None,logical_stream_role=dp_comm,grad_dtype=torch.float16,data_parallel_world_size=2,duration=15.0,timestamp=145.0,sub_operations=[])",
                f"rank:{rank_id}:dp_allreduce(stage_id={stage_id},batch_id=0,mg_state=finalize,duration=16.0,description=model_chunk.finish_grad_sync(), All-reduce / reduce-scatter across DP replicas,group_kind=dp,cmd_uid=cmd-wait-{rank_id},op_semantics=metadata_placeholder,finalize_base_duration_ms=0.5,input__shape=None,input__dtype=None,timestamp=166.0,sub_operations=[])",
            ]
        )
        + '\n',
        encoding='utf-8',
    )


def test_build_error_summary_uses_stable_alignment_key_and_target_window(tmp_path: Path) -> None:
    reference_dir = tmp_path / 'reference'
    reference_dir.mkdir()
    _write_trace(reference_dir / 'rank0.txt', 0)

    sim_summary = {
        'wrank_id': 0,
        'target_backward': {
            'cmd_uid': 'cmd-sim-bwd-0',
            'stage_id': 0,
            'mg_state': 'cooldown',
            'batch_id': 0,
            'start_time_ms': 100.0,
            'finish_time_ms': 149.0,
            'duration_ms': 49.0,
            'ddp_comm_alignment_keys': [
                'stage_id=0|mg_state=cooldown|comm_func=allreduce|buffer_id=1|bucket_id=1|bucket_offset=0|bucket_numel_unpadded=4|bucket_numel=4|param_count=1|logical_stream_role=dp_comm|grad_dtype=torch.float16'
            ],
            'ddp_comm_count': 1,
        },
        'backward_duration_ms_off': 42.0,
        'backward_duration_ms_on': 49.0,
        'slowdown_off': {
            'target_backward': {'start_time_ms': 100.0, 'finish_time_ms': 132.0},
            'ddp_comm_ops': {
                'stage_id=0|mg_state=cooldown|comm_func=allreduce|buffer_id=1|bucket_id=1|bucket_offset=0|bucket_numel_unpadded=4|bucket_numel=4|param_count=1|logical_stream_role=dp_comm|grad_dtype=torch.float16': {
                    'join_time': 120.0,
                    'finish_time': 132.0,
                }
            },
            'finalize_wait_duration_ms': 10.0,
        },
        'slowdown_on': {
            'target_backward': {'start_time_ms': 100.0, 'finish_time_ms': 144.0},
            'ddp_comm_ops': {
                'stage_id=0|mg_state=cooldown|comm_func=allreduce|buffer_id=1|bucket_id=1|bucket_offset=0|bucket_numel_unpadded=4|bucket_numel=4|param_count=1|logical_stream_role=dp_comm|grad_dtype=torch.float16': {
                    'join_time': 129.0,
                    'finish_time': 144.0,
                }
            },
            'finalize_wait_duration_ms': 15.5,
        },
    }

    reference_summary = module.load_reference_trace_summary(
        reference_dir,
        0,
        target_backward=sim_summary['target_backward'],
    )
    row = module.build_error_summary(sim_summary=sim_summary, reference_summary=reference_summary)

    assert row['wrank_id'] == 0
    assert row['reference_backward_cmd_uid'] == 'cmd-bwd-0'
    assert row['hardware_backward_duration_ms'] == pytest.approx(50.0)
    assert row['shared_comm_uids_count'] == 1
    assert row['backward_abs_err_ms_off'] == pytest.approx(8.0)
    assert row['backward_abs_err_ms_on'] == pytest.approx(1.0)
    assert row['ddp_launch_mae_ms_off'] == pytest.approx(10.0)
    assert row['ddp_launch_mae_ms_on'] == pytest.approx(1.0)
    assert row['finalize_wait_abs_err_ms_on'] == pytest.approx(0.5)


def test_build_error_summary_fails_fast_without_shared_alignment_keys(tmp_path: Path) -> None:
    reference_dir = tmp_path / 'reference'
    reference_dir.mkdir()
    _write_trace(reference_dir / 'rank0.txt', 0)

    sim_summary = {
        'wrank_id': 0,
        'target_backward': {
            'cmd_uid': 'cmd-sim-bwd-0',
            'stage_id': 0,
            'mg_state': 'cooldown',
            'batch_id': 0,
            'start_time_ms': 100.0,
            'finish_time_ms': 150.0,
            'duration_ms': 50.0,
            'ddp_comm_alignment_keys': ['stage_id=0|mg_state=cooldown|comm_func=allreduce|buffer_id=9|bucket_id=9|bucket_offset=9|bucket_numel_unpadded=9|bucket_numel=9|param_count=9|logical_stream_role=dp_comm|grad_dtype=torch.float16'],
            'ddp_comm_count': 1,
        },
        'backward_duration_ms_off': 42.0,
        'backward_duration_ms_on': 49.0,
        'slowdown_off': {
            'target_backward': {'start_time_ms': 100.0, 'finish_time_ms': 142.0},
            'ddp_comm_ops': {},
            'finalize_wait_duration_ms': 10.0,
        },
        'slowdown_on': {
            'target_backward': {'start_time_ms': 100.0, 'finish_time_ms': 149.0},
            'ddp_comm_ops': {},
            'finalize_wait_duration_ms': 15.5,
        },
    }

    with pytest.raises(ValueError, match='does not share any DDP comm alignment keys'):
        module.load_reference_trace_summary(reference_dir, 0, target_backward=sim_summary['target_backward'])
