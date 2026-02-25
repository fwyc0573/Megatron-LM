import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def _build_config(**overrides):
    kwargs = {
        "num_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 8,
        "use_cpu_initialization": True,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_scaling_sequence_parallel_uses_fake_tp():
    cfg = _build_config(
        sequence_parallel=True,
        tensor_model_parallel_size=1,
        is_scaling_mode=True,
        fake_tp=2,
    )
    assert cfg.sequence_parallel is True


def test_non_scaling_sequence_parallel_requires_tp_gt_one():
    with pytest.raises(ValueError, match="Can not use sequence paralllelism"):
        _build_config(
            sequence_parallel=True,
            tensor_model_parallel_size=1,
            is_scaling_mode=False,
        )


def test_scaling_head_divisibility_validates_against_fake_tp():
    with pytest.raises(ValueError, match="num_attention_heads"):
        _build_config(
            num_attention_heads=10,
            sequence_parallel=True,
            tensor_model_parallel_size=1,
            is_scaling_mode=True,
            fake_tp=3,
        )
