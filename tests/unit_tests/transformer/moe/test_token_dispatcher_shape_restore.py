# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.token_dispatcher import _restore_scaling_token_rows


def test_restore_scaling_token_rows_noop_when_not_scaling():
    output = torch.randn(8, 16)
    hidden_shape = torch.Size([4, 2, 16])

    restored = _restore_scaling_token_rows(output, hidden_shape, is_scaling_mode=False)

    assert restored is output
    assert restored.shape == (8, 16)


def test_restore_scaling_token_rows_restores_expected_rows():
    output = torch.arange(12, dtype=torch.float32).view(3, 4)
    hidden_shape = torch.Size([5, 1, 4])  # expected rows = 5

    restored = _restore_scaling_token_rows(output, hidden_shape, is_scaling_mode=True)

    assert restored.shape == (5, 4)
    assert torch.allclose(restored[:3], output)
    assert torch.count_nonzero(restored[3:]) == 0


def test_restore_scaling_token_rows_raises_on_hidden_mismatch():
    output = torch.randn(3, 8)
    hidden_shape = torch.Size([3, 1, 16])

    with pytest.raises(ValueError, match="Hidden size mismatch"):
        _restore_scaling_token_rows(output, hidden_shape, is_scaling_mode=True)
