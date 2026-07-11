from types import SimpleNamespace

import pytest

from megatron.core.distributed.distributed_data_parallel import _resolve_bucketing_pipeline_rank


def test_resolve_bucketing_pipeline_rank_uses_parallel_state_outside_scaling() -> None:
    args = SimpleNamespace(is_scaling_mode=False, pp_rank=7)
    assert _resolve_bucketing_pipeline_rank(args, default_pipeline_rank=1) == 1


def test_resolve_bucketing_pipeline_rank_uses_fake_pp_rank_in_scaling_mode() -> None:
    args = SimpleNamespace(is_scaling_mode=True, pp_rank=3)
    assert _resolve_bucketing_pipeline_rank(args, default_pipeline_rank=0) == 3


def test_resolve_bucketing_pipeline_rank_fails_fast_without_fake_pp_rank() -> None:
    args = SimpleNamespace(is_scaling_mode=True, pp_rank=None)
    with pytest.raises(RuntimeError, match='pp_rank'):
        _resolve_bucketing_pipeline_rank(args, default_pipeline_rank=0)
