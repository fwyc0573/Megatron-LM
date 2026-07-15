#!/usr/bin/env python3

import json
import math
import sys
from typing import Dict, List, Sequence, Tuple

import torch

try:
    from grouped_gemm import ops
except ImportError as exc:
    raise RuntimeError(
        "grouped_gemm v1.0 is required; run tools/ae/setup_grouped_gemm_v1.sh first"
    ) from exc


SEED = 20260713
ABS_ERROR_LIMIT = 0.125
MEAN_ABS_ERROR_LIMIT = 0.01
REPEAT_VARIANCE_LIMIT = 0.0
RELATIVE_DENOMINATOR_FLOOR = 1.0e-2
REPEAT_COUNT = 5


def grouped_reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    start = 0
    for expert_index, size in enumerate(batch_sizes.cpu().tolist()):
        rhs = weights[expert_index].transpose(0, 1) if trans_b else weights[expert_index]
        outputs.append(inputs[start : start + size] @ rhs)
        start += size
    if start != inputs.shape[0]:
        raise AssertionError(
            f"batch_sizes sum {start} does not match input rows {inputs.shape[0]}"
        )
    return torch.cat(outputs, dim=0)


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, object]:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    absolute_error = (actual_float - expected_float).abs()
    relative_error = absolute_error / expected_float.abs().clamp_min(
        RELATIVE_DENOMINATOR_FLOOR
    )
    finite_count = int(torch.isfinite(actual_float).sum().item())
    element_count = actual.numel()
    return {
        "element_count": element_count,
        "finite_count": finite_count,
        "max_abs_error": absolute_error.max().item(),
        "mean_abs_error": absolute_error.mean().item(),
        "max_relative_error": relative_error.max().item(),
        "relative_denominator_floor": RELATIVE_DENOMINATOR_FLOOR,
        "expected_sample": expected_float.flatten()[:5].cpu().tolist(),
        "actual_sample": actual_float.flatten()[:5].cpu().tolist(),
    }


def assert_metrics(name: str, metrics: Dict[str, object]) -> None:
    element_count = int(metrics["element_count"])
    finite_count = int(metrics["finite_count"])
    max_abs_error = float(metrics["max_abs_error"])
    mean_abs_error = float(metrics["mean_abs_error"])
    if finite_count != element_count:
        raise AssertionError(
            f"{name} contains non-finite values: {finite_count}/{element_count} finite"
        )
    if max_abs_error > ABS_ERROR_LIMIT:
        raise AssertionError(
            f"{name} max absolute error {max_abs_error} exceeds {ABS_ERROR_LIMIT}"
        )
    if mean_abs_error > MEAN_ABS_ERROR_LIMIT:
        raise AssertionError(
            f"{name} mean absolute error {mean_abs_error} exceeds {MEAN_ABS_ERROR_LIMIT}"
        )


def make_tensors(
    batch_sizes: Sequence[int], trans_b: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)
    input_width = 128
    output_width = 96
    expert_count = len(batch_sizes)
    inputs = torch.randn(
        sum(batch_sizes),
        input_width,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(input_width)
    weight_shape = (
        (expert_count, output_width, input_width)
        if trans_b
        else (expert_count, input_width, output_width)
    )
    weights = torch.randn(
        weight_shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(input_width)
    return inputs, weights, torch.tensor(batch_sizes, dtype=torch.int64)


def run_case(case_name: str, batch_sizes: Sequence[int], trans_b: bool) -> Dict[str, object]:
    inputs, weights, batch_sizes_tensor = make_tensors(batch_sizes, trans_b)
    inputs.requires_grad_(True)
    weights.requires_grad_(True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)

    actual = ops.gmm(inputs, weights, batch_sizes_tensor, trans_b)
    expected = grouped_reference(
        reference_inputs, reference_weights, batch_sizes_tensor, trans_b
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + 1)
    output_gradient = torch.randn(
        actual.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    actual.backward(output_gradient)
    expected.backward(output_gradient)

    output_metrics = tensor_metrics(actual, expected)
    input_gradient_metrics = tensor_metrics(inputs.grad, reference_inputs.grad)
    weight_gradient_metrics = tensor_metrics(weights.grad, reference_weights.grad)
    assert_metrics(f"{case_name}.output", output_metrics)
    assert_metrics(f"{case_name}.input_gradient", input_gradient_metrics)
    assert_metrics(f"{case_name}.weight_gradient", weight_gradient_metrics)

    repeated_outputs = []
    for _ in range(REPEAT_COUNT):
        repeated_outputs.append(
            ops.gmm(
                inputs.detach(), weights.detach(), batch_sizes_tensor, trans_b
            ).detach()
        )
    repeat_max_abs_delta = max(
        (candidate.float() - repeated_outputs[0].float()).abs().max().item()
        for candidate in repeated_outputs[1:]
    )
    if repeat_max_abs_delta > REPEAT_VARIANCE_LIMIT:
        raise AssertionError(
            f"{case_name} repeated-run max delta {repeat_max_abs_delta} exceeds "
            f"{REPEAT_VARIANCE_LIMIT}"
        )

    return {
        "case": case_name,
        "batch_sizes": list(batch_sizes),
        "trans_b": trans_b,
        "output": output_metrics,
        "input_gradient": input_gradient_metrics,
        "weight_gradient": weight_gradient_metrics,
        "repeat_count": REPEAT_COUNT,
        "repeat_max_abs_delta": repeat_max_abs_delta,
    }


def validate_runtime() -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for grouped_gemm runtime validation")
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    if "H800" not in device_name or device_capability != (9, 0):
        raise RuntimeError(
            "This validation requires an H800 with CUDA capability 9.0; "
            f"got {device_name!r} with capability {device_capability}"
        )

    cases = []
    for trans_b in (False, True):
        cases.append(run_case("fixed", [24, 24, 24, 24], trans_b))
        cases.append(run_case("variable", [13, 29, 7, 47], trans_b))
    return {
        "status": "PASS",
        "seed": SEED,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": device_name,
        "device_capability": list(device_capability),
        "dtype": "torch.bfloat16",
        "abs_error_limit": ABS_ERROR_LIMIT,
        "mean_abs_error_limit": MEAN_ABS_ERROR_LIMIT,
        "repeat_variance_limit": REPEAT_VARIANCE_LIMIT,
        "cases": cases,
    }


def main() -> int:
    try:
        result = validate_runtime()
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
