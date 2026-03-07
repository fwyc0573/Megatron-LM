import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron"
    / "profiler"
    / "moe"
    / "routing_profiles.py"
)
SPEC = importlib.util.spec_from_file_location("routing_profiles", MODULE_PATH)
ROUTING_PROFILES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROUTING_PROFILES)

build_controlled_routing_indices = ROUTING_PROFILES.build_controlled_routing_indices
summarize_expert_load = ROUTING_PROFILES.summarize_expert_load


def _hottest_to_median_ratio(indices: torch.Tensor, num_experts: int) -> float:
    summary = summarize_expert_load(indices, num_experts)
    return summary["hottest_to_median"]


def test_balanced_routing_is_nearly_uniform():
    indices = build_controlled_routing_indices(
        num_tokens=1024,
        topk=2,
        num_experts=32,
        skew_mode="balanced",
    )

    ratio = _hottest_to_median_ratio(indices, 32)

    assert indices.shape == (1024, 2)
    assert ratio <= 1.1


def test_skew_modes_increase_hottest_to_median_ratio_monotonically():
    balanced = build_controlled_routing_indices(1024, 2, 32, "balanced")
    moderate = build_controlled_routing_indices(1024, 2, 32, "moderate_skew")
    strong = build_controlled_routing_indices(1024, 2, 32, "strong_skew")

    balanced_ratio = _hottest_to_median_ratio(balanced, 32)
    moderate_ratio = _hottest_to_median_ratio(moderate, 32)
    strong_ratio = _hottest_to_median_ratio(strong, 32)

    assert balanced_ratio < moderate_ratio < strong_ratio
    assert moderate_ratio >= 1.5
    assert strong_ratio >= 3.0


def test_invalid_skew_mode_fails_fast():
    try:
        build_controlled_routing_indices(128, 2, 16, "unsupported")
    except ValueError as error:
        assert "Unsupported skew mode" in str(error)
    else:
        raise AssertionError("Expected ValueError for unsupported skew mode")
