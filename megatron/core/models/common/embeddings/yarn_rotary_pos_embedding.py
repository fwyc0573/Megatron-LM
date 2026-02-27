# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, rotary_base: float = 10000, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(rotary_base)
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings)
    low = math.floor(low)
    high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_value: float, max_value: float, dim: int, device: torch.device):
    if min_value == max_value:
        max_value += 0.001
    linear = (torch.arange(dim, dtype=torch.float32, device=device) - min_value) / (
        max_value - min_value
    )
    return torch.clamp(linear, 0, 1)


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_get_concentration_factor(
    scaling_factor: float, mscale: Optional[float], mscale_all_dim: Optional[float]
) -> float:
    if mscale is None or mscale_all_dim is None:
        return _yarn_get_mscale(scaling_factor)
    return float(
        _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim)
    )


class YarnRotaryEmbedding(RotaryEmbedding):
    """YaRN rotary embedding implementation used by DeepSeek-V3."""

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        self.dim = kv_channels if rotary_percent >= 1.0 else int(kv_channels * rotary_percent)
        self.rotary_base = rotary_base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.inv_freq_extra = 1.0 / (
            rotary_base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        self.inv_freq_inter = 1.0 / (
            scaling_factor
            * rotary_base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )

        super().__init__(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=int(rotary_base),
        )

    def get_emb(self, max_seq_len: int, offset: int = 0) -> Tuple[Tensor, float]:
        if self.rotary_interleaved:
            raise ValueError("YarnRotaryEmbedding does not support rotary_interleaved=True.")

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.rotary_base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, self.dim // 2, device=self.inv_freq_extra.device
        ).to(dtype=torch.float32)
        inv_freq = self.inv_freq_inter * (1 - inv_freq_mask) + self.inv_freq_extra * inv_freq_mask

        seq = (
            torch.arange(max_seq_len, device=self.inv_freq_extra.device, dtype=self.inv_freq_extra.dtype)
            + offset
        )
        freqs = torch.outer(seq, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[:, None, None, :]
        concentration = _yarn_get_concentration_factor(
            self.scaling_factor, self.mscale, self.mscale_all_dim
        )
        return emb, concentration

    def forward(self, max_seq_len: int, offset: int = 0) -> Tensor:
        emb, _ = self.get_emb(max_seq_len, offset)
        return emb
