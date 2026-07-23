from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


SCHEDULER_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "scheduler"
    / "mg_scheduling"
)
if str(SCHEDULER_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEDULER_DIR))

import mg_test  # noqa: E402
from mg_scheduling_plan import SchedulingPlan  # noqa: E402


MODEL_CASES = (
    {
        "model_size": "gpt175b",
        "world_size": 1024,
        "pp": 8,
        "tp": 8,
        "exp": 1,
        "gbs": 768,
        "seq": 2048,
        "hidden": 12288,
        "microbatches": 48,
        "untie": False,
    },
    {
        "model_size": "qwen3_a30b",
        "world_size": 256,
        "pp": 4,
        "tp": 8,
        "exp": 8,
        "gbs": 128,
        "seq": 256,
        "hidden": 2048,
        "microbatches": 16,
        "untie": True,
    },
    {
        "model_size": "dsv3",
        "world_size": 256,
        "pp": 4,
        "tp": 8,
        "exp": 8,
        "gbs": 128,
        "seq": 256,
        "hidden": 2048,
        "microbatches": 16,
        "untie": True,
    },
)


def _args(case: dict, output_dir: Path) -> Namespace:
    args = Namespace(
        tensor_model_parallel_size=case["tp"],
        pipeline_model_parallel_size=case["pp"],
        expert_model_parallel_size=case["exp"],
        untie_embeddings_and_output_weights=case["untie"],
        num_experts=1 if case["exp"] == 1 else 128,
        local_size=8,
        world_size=case["world_size"],
        micro_batch_size=1,
        global_batch_size=case["gbs"],
        seq_length=case["seq"],
        hidden_size=case["hidden"],
        model_size=case["model_size"],
        fp16=False,
        bf16=True,
        output_dir=str(output_dir),
        train_iters=1,
        trace_start=0,
    )
    return mg_test.validate_and_calculate_parameters(args)


def test_precision_flags_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mg_test.py", "--fp16", "--bf16"])
    with pytest.raises(SystemExit):
        mg_test.parse_arguments()


def test_parser_accepts_bf16_output_dir_and_opaque_model_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mg_test.py",
            "--bf16",
            "--output-dir",
            str(tmp_path),
            "--model-size",
            "qwen3_a30b",
        ],
    )
    args = mg_test.parse_arguments()
    assert args.bf16 is True
    assert args.fp16 is False
    assert args.output_dir == str(tmp_path)
    assert args.model_size == "qwen3_a30b"


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case["model_size"])
def test_ae_schedule_matrix_is_deterministic_and_bf16(
    case: dict,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / case["model_size"]
    plan = SchedulingPlan(_args(case, output_dir))

    assert plan.args.pipeline_dtype is torch.bfloat16
    plan.get_write_scheduling_plan(write_to_file=True)

    expected_files = {
        f"stage{stage_id}_scheduling_plan.txt" for stage_id in range(case["pp"])
    }
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    assert actual_files == expected_files

    for stage_id in range(case["pp"]):
        path = output_dir / f"stage{stage_id}_scheduling_plan.txt"
        text = path.read_text(encoding="utf-8")
        assert text.count(":forward_step(") == case["microbatches"]
        assert text.count(":backward_step(") == case["microbatches"]
        assert text.count(":optimizer_step(") == 1
        assert text.count(":dp_allreduce(") == 1
        assert "torch.bfloat16" in text
        assert f"input__shape=[{case['seq']}, 1, {case['hidden']}]" in text

        embedding_count = text.count(":ep_allreduce(")
        if case["untie"]:
            assert embedding_count == 0
        elif stage_id in {0, case["pp"] - 1}:
            assert embedding_count == 1
        else:
            assert embedding_count == 0
