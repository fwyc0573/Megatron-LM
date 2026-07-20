#!/usr/bin/env bash

# Documentation contract for the evaluator-facing SC'26 AE workflow.
# This test checks that the public README and paper suggestion list remain
# synchronized with the nine-entry, evidence-bound task contract.  It does
# not qualify a GPU run or a release pre-dataset.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
README_PATH="${REPO_ROOT}/SC26-AE/README.md"
TEX_SUGGESTIONS_PATH="${REPO_ROOT}/task_memory/task_2026-07-15_sc26_ae_workflow/tex_change_suggestions.md"

python3 - "${REPO_ROOT}" "${README_PATH}" "${TEX_SUGGESTIONS_PATH}" <<'PY'
import pathlib
import re
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
readme_path = pathlib.Path(sys.argv[2]).resolve()
tex_path = pathlib.Path(sys.argv[3]).resolve()
readme = readme_path.read_text(encoding="utf-8")
tex = tex_path.read_text(encoding="utf-8")

models = ("gpt175b", "qwen3_a30b", "dsv3")
tasks = ("task1", "task2", "task3")
entries = [f"{task}_{model}.sh" for task in tasks for model in models]

def fail(message):
    raise SystemExit(f"DOC_CONTRACT_FAIL: {message}")

def require(text, needle, label):
    if needle not in text:
        fail(f"{label} is missing: {needle}")

# All public files must exist, be executable, and be named directly in the
# README.  A prose reference to a generic dispatcher is not sufficient.
for entry in entries:
    path = repo_root / "SC26-AE" / entry
    if not path.is_file() or not (path.stat().st_mode & 0o111):
        fail(f"public entry is missing or not executable: {path}")
    require(readme, f"SC26-AE/{entry}", f"README entry {entry}")
    command_hits = re.findall(
        rf"(?:^|\s)(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*bash\s+SC26-AE/{re.escape(entry)}(?:\s|$)",
        readme,
        flags=re.MULTILINE,
    )
    if not command_hits:
        fail(f"README has no runnable bash command for {entry}")

# The README must identify the exact current source pins and distinguish the
# mutable worktree from a future release qualification.
for label in ("Main repository commit", "Echo-slowdown", "megatron-sim-engine", "collective-sim"):
    require(readme, label, f"README source identity {label}")
    # A source-identity row must carry a full SHA-1, not a short or symbolic
    # reference.  The row may explicitly mark the value as a current pin.
    identity_match = re.search(
        rf"{re.escape(label)}[^\n]*\b[0-9a-f]{{40}}\b", readme, flags=re.IGNORECASE
    )
    if identity_match is None:
        fail(f"README source identity has no full SHA-1: {label}")

require(readme, "hub.i.basemind.com/mg-echo/megatron-h800:v1.2-ae", "current image")
require(readme, "immutable digest", "image digest boundary")
require(readme, "EVIDENCE_CLASS=local_synthetic_not_gpu_qualification", "synthetic evidence class")
require(readme, "AE-ready=NO", "qualification status boundary")
require(readme, "ARTIFACT_SOURCE=fresh", "fresh source command")
require(readme, "ARTIFACT_SOURCE=prebaked", "prebaked source command")
require(readme, "no automatic fresh-to-prebaked fallback", "no-fallback statement")
require(readme, "SC26-AE/tools/package_prebaked.py build", "strict prebaked packager command")
require(readme, "local/synthetic", "synthetic packager evidence boundary")
require(readme, "production CLI", "production packager strictness")
require(
    readme,
    "2026-SC-first-submission/sc25-ad-ae/for-paper-authors/sc26-ad.tex",
    "paper draft relationship",
)
require(
    readme,
    "QUICK_TASK3_STATUS=LOCAL_SMOKE_COMPATIBLE_NOT_RELEASE_QUALIFIED",
    "QUICK Task3 evidence boundary",
)
require(
    readme,
    "REAL_RUNTIME_EVIDENCE_STATUS=PENDING_H800_QUALIFICATION",
    "real runtime evidence boundary",
)
require(
    readme,
    "CANONICAL_PREBAKED_DISTRIBUTION=NOT_SELECTED",
    "pending canonical distribution boundary",
)
require(readme, "## Fail-fast troubleshooting", "root-cause troubleshooting section")
require(
    readme,
    "`collective-sim` is optional background infrastructure",
    "communication weak-validation boundary",
)

# Task2 is a reusable pre-dataset producer.  Its documented numeric contract
# must name every audit field rather than reducing the output to an opaque MSE.
for field in (
    "task2_run_all_elapsed_seconds",
    "dataset_row_count",
    "validation_mse_by_fold",
    "average_validation_mse",
    "test_mse",
    "model_reload_max_abs_prediction_delta",
    "scaler_feature_count",
    "scaler_mean_count",
    "scaler_scale_count",
    "scaler_nonzero_scale_count",
    "prediction_sample",
):
    require(readme, f"`{field}`", f"Task2 metric field {field}")

# Keep report field names exact so an AE operator can parse JSON without
# guessing aliases.
for field in (
    "rank0_step_time_ms",
    "rank0_forward_step_duration_sum_ms",
    "rank0_backward_step_duration_sum_ms",
    "rank0_optimizer_step_duration_sum_ms",
    "rank0_comp_plus_comm_diagnostic_ms",
    "simulator_load_time_s",
    "simulator_execution_time_s",
    "simulator_wall_clock_s",
):
    require(readme, f"`{field}`", f"Task3 report field {field}")

# Every paper suggestion must be auditable without editing the paper in this
# task: retain the copied old wording, proposed replacement, evidence path,
# and reason for each numbered suggestion.
for index in range(1, 11):
    heading = f"## {index}."
    start = tex.find(heading)
    if start < 0:
        fail(f"paper suggestion section is missing: {heading}")
    end = tex.find("\n## ", start + len(heading))
    section = tex[start:] if end < 0 else tex[start:end]
    require(
        section,
        "**Current wording (copied verbatim from the reviewed draft):**",
        f"paper suggestion {index} current wording",
    )
    require(
        section,
        "**Suggested replacement:**",
        f"paper suggestion {index} replacement",
    )
    require(section, "**Evidence path:**", f"paper suggestion {index} evidence path")
    require(section, "**Reason:**", f"paper suggestion {index} reason")

require(tex, "does not modify", "paper edit boundary")
require(tex, "sc26-ad.tex", "paper source reference")
require(
    tex,
    "SC26_AD_SOURCE_SHA256=31d2053436e8a058bc3ee2a4d32868625d737c8260a058c3a4bd07b3c57adf17",
    "reviewed paper source SHA256",
)
print("DOC_CONTRACT_STATUS=PASS")
print(f"PUBLIC_ENTRY_COUNT={len(entries)}")
print("PAPER_SUGGESTION_COUNT=10")
PY
