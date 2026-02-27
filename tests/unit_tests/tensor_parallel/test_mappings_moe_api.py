import torch

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
