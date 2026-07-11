import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_cmd_module():
    cmd_path = (
        Path(__file__).resolve().parents[3] / "megatron" / "profiler" / "cmd.py"
    )
    spec = importlib.util.spec_from_file_location("test_cmd_module_nvtx", cmd_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {cmd_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmd_module = _load_cmd_module()
CMD = cmd_module.CMD


class _DummyEvent:
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self):
        return None

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return 0.42


def _build_cmd(
    enable_nvtx: bool,
    enable_phase: bool = False,
    boundary_mode: str = "none",
    group_kind=None,
):
    micro_batch_ids = {"forward_step": 0}
    stage_operations_trace = {}
    args = SimpleNamespace(
        trace_subop_sync_mode="global",
        trace_kernel_ground_truth=enable_nvtx,
        trace_kernel_ground_truth_prefix="cmd_gt",
        trace_kernel_ground_truth_phase=enable_phase,
        trace_kernel_boundary_sync_mode=boundary_mode,
        is_scaling_mode=False,
    )
    return CMD(
        rank_id=0,
        mg_state="steady",
        name_cmd="forward_step",
        use_cuda=True,
        stage_operations_trace_dict=stage_operations_trace,
        micro_batch_ids_dict=micro_batch_ids,
        stage_id=1,
        group_kind=group_kind,
        simu_start=True,
        trace_start=1,
        current_iter=1,
        args=args,
    )


def test_cmd_kernel_ground_truth_nvtx_enabled_pushes_and_pops():
    cmd = _build_cmd(enable_nvtx=True)
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ), mock.patch.object(cmd_module.nvtx, "range_push") as range_push, mock.patch.object(
        cmd_module.nvtx, "range_pop"
    ) as range_pop:
        with cmd:
            pass
    assert range_push.call_count == 1
    pushed_label = range_push.call_args[0][0]
    assert pushed_label.startswith("cmd_gt|rank=0|op=forward_step|state=steady")
    assert "|stage=1|" in pushed_label
    assert "|cmd_uid=" in pushed_label
    assert range_pop.call_count == 1


def test_cmd_kernel_ground_truth_nvtx_label_keeps_required_fields_and_cmd_uid():
    cmd = _build_cmd(enable_nvtx=True)
    label = cmd._build_kernel_ground_truth_nvtx_label()
    assert label is not None
    assert "|rank=0|" in label
    assert "|op=forward_step|" in label
    assert "|state=steady|" in label
    assert "|stage=1|" in label
    assert "|batch=" in label
    assert "|iter=1|" in label
    assert f"|cmd_uid={cmd.cmd_uid}" in label


def test_cmd_kernel_ground_truth_nvtx_disabled_does_not_emit_range():
    cmd = _build_cmd(enable_nvtx=False)
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ), mock.patch.object(cmd_module.nvtx, "range_push") as range_push, mock.patch.object(
        cmd_module.nvtx, "range_pop"
    ) as range_pop:
        with cmd:
            pass
    assert range_push.call_count == 0
    assert range_pop.call_count == 0


def test_cmd_kernel_ground_truth_phase_context_emits_phase_range():
    cmd = _build_cmd(enable_nvtx=True, enable_phase=True, boundary_mode="event")
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ), mock.patch.object(cmd_module.nvtx, "range_push") as range_push, mock.patch.object(
        cmd_module.nvtx, "range_pop"
    ) as range_pop:
        with cmd:
            with cmd.phase_range("compute"):
                pass
    assert range_push.call_count == 2
    assert "|phase=compute" in range_push.call_args_list[1][0][0]
    assert range_pop.call_count == 2


def test_cmd_kernel_ground_truth_phase_disabled_skips_phase_range():
    cmd = _build_cmd(enable_nvtx=True, enable_phase=False)
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ), mock.patch.object(cmd_module.nvtx, "range_push") as range_push, mock.patch.object(
        cmd_module.nvtx, "range_pop"
    ) as range_pop:
        with cmd:
            with cmd.phase_range("compute"):
                pass
    assert range_push.call_count == 1
    assert range_pop.call_count == 1


def test_cmd_kernel_ground_truth_comm_group_auto_phase_range():
    cmd = _build_cmd(
        enable_nvtx=True, enable_phase=True, boundary_mode="event", group_kind="pp"
    )
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ), mock.patch.object(cmd_module.nvtx, "range_push") as range_push, mock.patch.object(
        cmd_module.nvtx, "range_pop"
    ) as range_pop:
        with cmd:
            pass
    assert range_push.call_count == 2
    assert "|phase=comm" in range_push.call_args_list[1][0][0]
    assert range_pop.call_count == 2


def test_cmd_kernel_ground_truth_phase_context_supports_extra_tags():
    cmd = _build_cmd(enable_nvtx=True, enable_phase=True, boundary_mode="event")
    with mock.patch.object(cmd_module.torch.cuda, "Event", _DummyEvent), mock.patch.object(
        cmd_module.torch.cuda, "synchronize"
    ), mock.patch.object(cmd_module.nvtx, "range_push") as range_push, mock.patch.object(
        cmd_module.nvtx, "range_pop"
    ) as range_pop:
        with cmd:
            with cmd.phase_range(
                "compute", extra_tags={"attn_bwd_segment": "attn_core_bwd"}
            ):
                pass
    assert range_push.call_count == 2
    pushed_label = range_push.call_args_list[1][0][0]
    assert "|phase=compute" in pushed_label
    assert "|attn_bwd_segment=attn_core_bwd" in pushed_label
    assert range_pop.call_count == 2
