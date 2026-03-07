from typing import Dict, Optional

import torch


SUPPORTED_ROUTING_PROFILES = (
    "default",
    "balanced",
    "moderate_skew",
    "strong_skew",
)


def _build_weighted_schedule(count: int, weights: torch.Tensor) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be > 0")
    if weights.numel() == 0:
        raise ValueError("weights must be non-empty")
    if (weights <= 0).any():
        raise ValueError("weights must be positive")

    repeated = torch.repeat_interleave(
        torch.arange(weights.numel(), dtype=torch.long),
        weights.to(dtype=torch.long),
    )
    repeats = (count + repeated.numel() - 1) // repeated.numel()
    return repeated.repeat(repeats)[:count]


def _normalize_profile_name(skew_mode: str) -> str:
    if skew_mode == "moderate":
        return "moderate_skew"
    if skew_mode == "strong":
        return "strong_skew"
    return skew_mode


def build_controlled_routing_indices(
    num_tokens: int,
    topk: int,
    num_experts: int,
    skew_mode: str,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
) -> torch.Tensor:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if num_experts <= 0:
        raise ValueError("num_experts must be > 0")

    profile = _normalize_profile_name(skew_mode)
    if profile not in SUPPORTED_ROUTING_PROFILES:
        raise ValueError(f"Unsupported skew mode: {skew_mode}")
    if profile == "default":
        raise ValueError("default profile should use the model router path instead")

    if num_groups is not None or group_topk is not None:
        if num_groups is None or group_topk is None:
            raise ValueError("num_groups and group_topk must be set together")
        if num_groups <= 0 or group_topk <= 0:
            raise ValueError("num_groups and group_topk must be > 0")
        if num_experts % num_groups != 0:
            raise ValueError("num_experts must be divisible by num_groups")
        if topk % group_topk != 0:
            raise ValueError("topk must be divisible by group_topk")
        return _build_group_limited_indices(
            num_tokens=num_tokens,
            topk=topk,
            num_experts=num_experts,
            profile=profile,
            num_groups=num_groups,
            group_topk=group_topk,
        )

    return _build_expert_limited_indices(
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        profile=profile,
    )


def _build_expert_limited_indices(
    num_tokens: int,
    topk: int,
    num_experts: int,
    profile: str,
) -> torch.Tensor:
    weights = torch.ones(num_experts, dtype=torch.long)
    hot_expert_count = max(1, num_experts // 8)
    if profile == "moderate_skew":
        weights[:hot_expert_count] = 2
    elif profile == "strong_skew":
        weights[:hot_expert_count] = 5

    primary = _build_weighted_schedule(num_tokens, weights)
    offsets = torch.arange(topk, dtype=torch.long).unsqueeze(0)
    return (primary.unsqueeze(1) + offsets) % num_experts


def _build_group_limited_indices(
    num_tokens: int,
    topk: int,
    num_experts: int,
    profile: str,
    num_groups: int,
    group_topk: int,
) -> torch.Tensor:
    experts_per_group = num_experts // num_groups
    experts_per_selected_group = topk // group_topk
    group_weights = torch.ones(num_groups, dtype=torch.long)
    if profile == "moderate_skew":
        group_weights[0] = 2
    elif profile == "strong_skew":
        group_weights[0] = 5

    primary_groups = _build_weighted_schedule(num_tokens, group_weights)
    indices = torch.empty((num_tokens, topk), dtype=torch.long)
    group_cursors = torch.zeros(num_groups, dtype=torch.long)

    for token_id in range(num_tokens):
        column = 0
        for group_offset in range(group_topk):
            group_id = int((primary_groups[token_id] + group_offset) % num_groups)
            base_expert = group_id * experts_per_group
            for _ in range(experts_per_selected_group):
                local_expert = int(group_cursors[group_id] % experts_per_group)
                indices[token_id, column] = base_expert + local_expert
                group_cursors[group_id] += 1
                column += 1

    return indices


def summarize_expert_load(indices: torch.Tensor, num_experts: int) -> Dict[str, float]:
    if indices.ndim != 2:
        raise ValueError("indices must have shape [num_tokens, topk]")
    if num_experts <= 0:
        raise ValueError("num_experts must be > 0")

    counts = torch.bincount(indices.reshape(-1), minlength=num_experts).to(dtype=torch.float32)
    median = torch.median(counts).item()
    mean = counts.mean().item()
    std = counts.std(unbiased=False).item()
    hottest = counts.max().item()
    coldest = counts.min().item()
    hottest_to_median = hottest / median if median > 0 else float("inf")
    return {
        "hottest": hottest,
        "coldest": coldest,
        "median": median,
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else 0.0,
        "hottest_to_median": hottest_to_median,
    }
