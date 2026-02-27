import pytest
import torch

from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
    _yarn_get_concentration_factor,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_yarn_rotary_embedding_shape_and_concentration():
    embedding = YarnRotaryEmbedding(
        kv_channels=64,
        rotary_percent=1.0,
        rotary_base=10000.0,
        scaling_factor=8.0,
        original_max_position_embeddings=4096,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
    )

    freqs, concentration = embedding.get_emb(16)
    assert freqs.shape == (16, 1, 1, 64)
    expected = _yarn_get_concentration_factor(
        scaling_factor=8.0,
        mscale=1.0,
        mscale_all_dim=1.0,
    )
    assert concentration == pytest.approx(expected)

    freqs_only = embedding(16)
    assert freqs_only.shape == (16, 1, 1, 64)
