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
