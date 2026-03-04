import torch
import pytest

from megatron.core.tensor_parallel import mappings


def test_gather_from_sequence_parallel_region_to_moe_accepts_use_global_buffer(
    monkeypatch,
):
    class _DummyGather:
        @staticmethod
        def apply(input_):
            return input_

    monkeypatch.setattr(
        mappings, "_GatherFromSequenceParallelRegionToMOE", _DummyGather
    )
    payload = object()
    output = mappings.gather_from_sequence_parallel_region_to_moe(
        payload, use_global_buffer=True
    )
    assert output is payload


def test_profiled_all_to_all_single_scaling_mode_returns_finite_payload():
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = mappings._profiled_all_to_all_single(
        input_tensor,
        output_split_sizes=[2, 4],
        input_split_sizes=[1, 2],
        group=None,
        group_type="exp",
        is_scaling_mode=True,
    )
    assert output.shape == (6, 4)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:3], input_tensor)
    assert torch.equal(output[3:], torch.zeros(3, 4, dtype=torch.float32))


def test_profiled_all_to_all_single_scaling_mode_equal_rows_materializes_copy():
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = mappings._profiled_all_to_all_single(
        input_tensor,
        output_split_sizes=[1, 2],
        input_split_sizes=[1, 2],
        group=None,
        group_type="exp",
        is_scaling_mode=True,
    )
    assert output.shape == input_tensor.shape
    assert torch.equal(output, input_tensor)
    assert output.data_ptr() != input_tensor.data_ptr()


def test_profiled_all_to_all_single_scaling_mode_none_split_materializes_copy():
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = mappings._profiled_all_to_all_single(
        input_tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
        group_type="exp",
        is_scaling_mode=True,
    )
    assert output.shape == input_tensor.shape
    assert torch.equal(output, input_tensor)
    assert output.data_ptr() != input_tensor.data_ptr()


def test_profiled_all_to_all_single_non_scaling_uses_contiguous_input(monkeypatch):
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    assert not input_tensor.is_contiguous()
    called = {}

    def _fake_all_to_all_single(
        output, input_, output_split_sizes, input_split_sizes, group
    ):
        called["contiguous"] = input_.is_contiguous()
        output.copy_(input_)

    monkeypatch.setattr(torch.distributed, "all_to_all_single", _fake_all_to_all_single)

    output = mappings._profiled_all_to_all_single(
        input_tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group=object(),
        group_type="exp",
        is_scaling_mode=False,
    )
    assert called["contiguous"] is True
    assert torch.equal(output, input_tensor)


def test_emulate_comm_adjacent_copies_materializes_copy():
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = mappings._emulate_comm_adjacent_copies(input_tensor, copy_iters=2)
    assert torch.equal(output, input_tensor)
    assert output.data_ptr() != input_tensor.data_ptr()


def test_emulate_comm_adjacent_copies_negative_raises():
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    with pytest.raises(ValueError):
        mappings._emulate_comm_adjacent_copies(input_tensor, copy_iters=-1)


def test_reduce_from_model_parallel_region_nvtx_balanced_world_size_one(monkeypatch):
    nvtx_calls = []
    monkeypatch.setattr(
        mappings.nvtx,
        "range_push",
        lambda label: nvtx_calls.append(("push", label)),
    )
    monkeypatch.setattr(
        mappings.nvtx,
        "range_pop",
        lambda: nvtx_calls.append(("pop", None)),
    )
    monkeypatch.setattr(
        mappings,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    input_tensor = torch.randn(2, 3)
    output = mappings._ReduceFromModelParallelRegion.forward(None, input_tensor)

    assert output is input_tensor
    assert nvtx_calls == [("push", "row_g_fwd"), ("pop", None)]


def test_reduce_from_model_parallel_region_nvtx_balanced_world_size_gt_one(
    monkeypatch,
):
    nvtx_calls = []
    reduce_calls = []
    monkeypatch.setattr(
        mappings.nvtx,
        "range_push",
        lambda label: nvtx_calls.append(("push", label)),
    )
    monkeypatch.setattr(
        mappings.nvtx,
        "range_pop",
        lambda: nvtx_calls.append(("pop", None)),
    )
    monkeypatch.setattr(
        mappings,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )

    def _fake_reduce(input_tensor, func="embedding_fwd"):
        reduce_calls.append(func)
        return input_tensor + 1

    monkeypatch.setattr(mappings, "_reduce", _fake_reduce)

    input_tensor = torch.randn(2, 3)
    output = mappings._ReduceFromModelParallelRegion.forward(
        None, input_tensor, func="unit_test_reduce"
    )

    assert torch.equal(output, input_tensor + 1)
    assert reduce_calls == ["unit_test_reduce"]
    assert nvtx_calls == [("push", "row_g_fwd"), ("pop", None)]
