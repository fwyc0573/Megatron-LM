from megatron.profiler.utils import resolve_scaling_replay_path


def test_resolve_scaling_replay_path_prefers_iter_specific(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    iter_path = cache_dir / "activation_to_rank4_iter5.pt"
    legacy_path = cache_dir / "activation_to_rank4.pt"
    iter_path.write_text("iter")
    legacy_path.write_text("legacy")

    selected = resolve_scaling_replay_path(str(cache_dir), rank_id=4, current_iter=5)

    assert selected == str(iter_path)


def test_resolve_scaling_replay_path_falls_back_to_legacy(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    legacy_path = cache_dir / "activation_to_rank7.pt"
    legacy_path.write_text("legacy")

    selected = resolve_scaling_replay_path(str(cache_dir), rank_id=7, current_iter=9)

    assert selected == str(legacy_path)


def test_resolve_scaling_replay_path_returns_none_when_missing(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    selected = resolve_scaling_replay_path(str(cache_dir), rank_id=2, current_iter=1)

    assert selected is None


def test_resolve_scaling_replay_path_handles_missing_cache_dir():
    selected = resolve_scaling_replay_path(cache_dir=None, rank_id=1, current_iter=3)

    assert selected is None
