#!/usr/bin/env python3
"""Strict, fail-closed verifier for the Session 47 V21 documentation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ARTIFACT_BEGIN = "V21_ARTIFACT_INVENTORY_BEGIN"
ARTIFACT_END = "V21_ARTIFACT_INVENTORY_END"
DOCUMENT_BEGIN = "V21_AUTHORITATIVE_INVENTORY_BEGIN"
DOCUMENT_END = "V21_AUTHORITATIVE_INVENTORY_END"
SUPPLEMENTAL_HEADING = "V21 supplemental identities:"


class VerificationFailure(RuntimeError):
    """Raised when any V21 contract check fails."""


def fail(message: str) -> None:
    raise VerificationFailure(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_and_size(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def extract_inventory(
    summary: str,
    begin: str,
    end: str,
    expected_count: int,
    label: str,
) -> list[tuple[str, int, str]]:
    matches = list(re.finditer(rf"{re.escape(begin)}\n(.*?){re.escape(end)}", summary, re.S))
    require(len(matches) == 1, f"{label} marker pair count is {len(matches)}, expected 1")
    body = matches[0].group(1)
    rows: list[tuple[str, int, str]] = []
    row_pattern = re.compile(
        r"^\|\s*`([^`]+)`\s*\|\s*([0-9][0-9,]*)\s*\|\s*`([0-9a-f]{64})`\s*\|\s*$",
        re.MULTILINE,
    )
    for match in row_pattern.finditer(body):
        path_text, size_text, digest = match.groups()
        rows.append((path_text, int(size_text.replace(",", "")), digest))
    require(len(rows) == expected_count, f"{label} row count={len(rows)}, expected {expected_count}")
    paths = [row[0] for row in rows]
    require(len(set(paths)) == len(paths), f"{label} contains duplicate paths")
    return rows


def resolve_repo_path(repo_root: Path, path_text: str) -> Path:
    relative = Path(path_text)
    require(not relative.is_absolute(), f"inventory path must be relative: {path_text}")
    require(".." not in relative.parts, f"inventory path escapes repository: {path_text}")
    path = repo_root / relative
    require(path.is_file(), f"inventory file is missing: {path_text}")
    resolved_root = repo_root.resolve()
    try:
        path.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise VerificationFailure(f"inventory file resolves outside repository: {path_text}") from exc
    return path


def verify_inventory(
    repo_root: Path,
    rows: list[tuple[str, int, str]],
    label: str,
) -> None:
    for path_text, expected_size, expected_digest in rows:
        require(
            path_text != "task_memory/task_2026-07-15_sc26_ae_workflow/summary.md",
            f"{label} includes summary.md",
        )
        path = resolve_repo_path(repo_root, path_text)
        actual_size, actual_digest = sha256_and_size(path)
        require(
            actual_size == expected_size,
            f"{label} size mismatch for {path_text}: expected {expected_size}, actual {actual_size}",
        )
        require(
            actual_digest == expected_digest,
            f"{label} SHA256 mismatch for {path_text}: expected {expected_digest}, actual {actual_digest}",
        )
        print(f"{label.upper()}_HASH path={path_text} bytes={actual_size} sha256={actual_digest}")


def resolve_supplemental_path(task_dir: Path, name: str) -> Path:
    candidates = [task_dir / name, task_dir / "logs" / name]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    require(len(existing) == 1, f"supplemental identity must resolve uniquely: {name}")
    return existing[0]


def verify_supplemental(summary: str, task_dir: Path) -> int:
    require(SUPPLEMENTAL_HEADING in summary, "missing V21 supplemental identity heading")
    section = summary.split(SUPPLEMENTAL_HEADING, 1)[1]
    fences = re.findall(r"```text\n(.*?)\n```", section, re.S)
    require(fences, "missing fenced V21 supplemental identity block")
    lines = [line.strip() for line in fences[0].splitlines() if line.strip()]
    pattern = re.compile(r"^(.+?) bytes=([0-9][0-9,]*) sha256=([0-9a-f]{64})$")
    identities: list[tuple[str, int, str]] = []
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            continue
        name, size_text, digest = match.groups()
        identities.append((name, int(size_text.replace(",", "")), digest))
    require(identities, "V21 supplemental identity block has no parseable rows")
    names = [row[0] for row in identities]
    require(len(set(names)) == len(names), "V21 supplemental identity block has duplicate names")
    for name, expected_size, expected_digest in identities:
        path = resolve_supplemental_path(task_dir, name)
        actual_size, actual_digest = sha256_and_size(path)
        require(
            actual_size == expected_size,
            f"supplemental size mismatch for {name}: expected {expected_size}, actual {actual_size}",
        )
        require(
            actual_digest == expected_digest,
            f"supplemental SHA256 mismatch for {name}: expected {expected_digest}, actual {actual_digest}",
        )
        print(f"SUPPLEMENTAL_HASH name={name} bytes={actual_size} sha256={actual_digest}")
    return len(identities)


def verify_issue_headings(issues_path: Path) -> None:
    text = issues_path.read_text(encoding="utf-8")
    missing = [
        issue
        for issue in range(50, 59)
        if not re.search(rf"^#+\s+I{issue}(?:\s|[.:—-])", text, re.M)
    ]
    require(not missing, f"missing issue headings: {missing}")
    print("ISSUE_HEADINGS_I50_I58=PASS")


def verify_status_boundary(summary: str, expected_status: str) -> None:
    status_lines = re.findall(r"^V21_STATUS=([A-Z_]+)$", summary, re.M)
    require(status_lines, "missing V21_STATUS marker")
    require(
        status_lines[-1] == expected_status,
        f"expected final V21_STATUS={expected_status}, observed {status_lines[-1]}",
    )
    require(summary.count("V21_STATUS=") == 1, "V21_STATUS must have exactly one current marker")

    final_block = summary[summary.rfind("V21_STATUS=") :]
    required = (
        "I56 = PARTIAL / OPEN",
        "INCOMPLETE",
        "Gate B1 = BLOCKED",
        "real_pre_dataset = NOT QUALIFIED",
        "release_pre_dataset = NOT QUALIFIED",
        "AE-ready = NO",
    )
    for phrase in required:
        require(phrase in final_block, f"missing current status boundary phrase: {phrase}")
    require("local_synthetic_not_gpu_qualification" in summary, "missing synthetic evidence class")
    print(f"V21_STATUS={expected_status}")
    print("STATUS_BOUNDARY=PASS")


def verify_optional_verifier_identity(summary: str, task_dir: Path) -> None:
    marker = "V21_VERIFIER_IDENTITY_BEGIN"
    end_marker = "V21_VERIFIER_IDENTITY_END"
    begin_count = summary.count(marker)
    end_count = summary.count(end_marker)
    require(begin_count in (0, 1), f"V21 verifier identity begin marker count={begin_count}")
    require(end_count == begin_count, "V21 verifier identity marker pair is unbalanced")
    if begin_count == 0:
        print("V21_VERIFIER_IDENTITY=ABSENT_PRE_APPEND")
        return
    match = re.search(rf"{re.escape(marker)}\n(.*?){re.escape(end_marker)}", summary, re.S)
    require(match is not None, "V21 verifier identity block is malformed")
    body = match.group(1)
    path_match = re.search(r"^path=(.+)$", body, re.M)
    bytes_match = re.search(r"^bytes=([0-9]+)$", body, re.M)
    digest_match = re.search(r"^sha256=([0-9a-f]{64})$", body, re.M)
    require(path_match and bytes_match and digest_match, "V21 verifier identity fields are incomplete")
    path_text = path_match.group(1).strip()
    path = resolve_repo_path(task_dir.parent.parent, path_text)
    actual_size, actual_digest = sha256_and_size(path)
    require(actual_size == int(bytes_match.group(1)), "V21 verifier log byte identity mismatch")
    require(actual_digest == digest_match.group(1), "V21 verifier log SHA256 identity mismatch")
    print(f"V21_VERIFIER_IDENTITY=PASS path={path_text} bytes={actual_size} sha256={actual_digest}")


def verify_log_markers(task_dir: Path) -> None:
    logs_dir = task_dir / "logs"

    def log(name: str) -> str:
        path = logs_dir / name
        require(path.is_file(), f"required evidence log is missing: {name}")
        return path.read_text(encoding="utf-8", errors="replace")

    package = log("session47-post-agent-package-artifact-20260720.log")
    require("52 passed" in package and "42 passed" in package, "package/artifact regression counts missing")
    # Keep the retained Session 47 transcript immutable: its PASS_COUNT=31 marker is
    # historical evidence for that checkpoint, not the current D16 contract count.
    task1 = log("session47-post-agent-task1-20260720.log")
    require("PASS_COUNT=31" in task1, "historical Task1 integration PASS_COUNT=31 missing")
    current_unit = log("i53-d16-unit-green-20260720.log")
    require("PASS_COUNT=49" in current_unit, "current D16 unit PASS_COUNT=49 missing")
    current_task1 = log("i53-d16-integration-green-20260720.log")
    require("PASS_COUNT=38" in current_task1, "current Task1 integration PASS_COUNT=38 missing")
    full_e2e = log("session47-post-agent-full-e2e-20260720.log")
    required_metrics = (
        "TASK1_TRACE_FILES=4",
        "TASK1_MEMORY_JSON=4",
        "TASK2_DATASET_ROWS=2",
        "TASK2_AVERAGE_VALIDATION_MSE=3.0",
        "TASK2_TEST_MSE=0.5",
        "TASK2_MODEL_RELOAD_MAX_ABS_PREDICTION_DELTA=0.0",
        "TASK3_RANK0_STEP_MS=22.5",
        "TASK3_FORWARD_MS=6.0",
        "TASK3_BACKWARD_MS=11.0",
        "TASK3_OPTIMIZER_MS=2.5",
        "TASK3_SIM_WALL_S=0.5",
        "CHAIN_PASS_COUNT=1",
        "FULL_E2E_STATUS=PASS",
        "EVIDENCE_CLASS=local_synthetic_not_gpu_qualification",
    )
    for marker in required_metrics:
        require(marker in full_e2e, f"missing fresh-chain/e2e metric marker: {marker}")
    static = log("session47-post-agent-static-scope-corrected-20260720.log")
    for marker in (
        # The retained Session 47 transcript is historical and intentionally reports the
        # then-current 52-file scope. The live scope check below reports the current 53-file
        # scope after the model-aware D16 test was added.
        "SHELL_SCOPE_COUNT=52",
        "SHELL_SYNTAX_COUNT=52",
        "PYTHON_SCOPE_COUNT=35",
        "PYTHON_SYNTAX_COUNT=35",
        "GIT_DIFF_CHECK=PASS",
        "TMP_ROOT_SCAN=PASS",
    ):
        require(marker in static, f"missing static evidence marker: {marker}")
    docs = log("session47-post-agent-docs-contract-20260720.log")
    for marker in ("DOC_CONTRACT_STATUS=PASS", "PUBLIC_ENTRY_COUNT=9", "PAPER_SUGGESTION_COUNT=10"):
        require(marker in docs, f"missing documentation evidence marker: {marker}")
    full_unit = log("session47-post-agent-full-unit-20260720.log")
    require("6 failed, 113 passed" in full_unit, "CUDA-only full-unit limitation is not recorded")
    require("FULL_UNIT_RC=1" in full_unit, "full-unit non-zero controller result is not recorded")
    print("EVIDENCE_MARKERS=PASS")
    print("TASK1_TRACE_FILES=4")
    print("TASK1_MEMORY_JSON=4")
    print("TASK2_DATASET_ROWS=2")
    print("TASK2_AVERAGE_VALIDATION_MSE=3.0")
    print("TASK2_TEST_MSE=0.5")
    print("TASK2_MODEL_RELOAD_MAX_ABS_PREDICTION_DELTA=0.0")
    print("TASK3_RANK0_STEP_MS=22.5")
    print("TASK3_FORWARD_MS=6.0")
    print("TASK3_BACKWARD_MS=11.0")
    print("TASK3_OPTIMIZER_MS=2.5")
    print("TASK3_SIM_WALL_S=0.5")
    print("CUDA_ONLY_FULL_UNIT_FAILURES=6")
    print("HISTORICAL_TASK1_PASS_COUNT=31")
    print("CURRENT_D16_UNIT_PASS_COUNT=49")
    print("CURRENT_TASK1_INTEGRATION_PASS_COUNT=38")


def collect_shell_paths(repo_root: Path) -> list[Path]:
    verifier_shell = repo_root / "tests/integration/test_sc26_ae_v21_verifier.sh"
    runtime_output_root = repo_root / "SC26-AE/output"
    shell_roots = ("SC26-AE", "tests/unit", "tests/integration", "tests/e2e", "tools/ae")
    return sorted(
        path
        for root in shell_roots
        for path in (repo_root / root).rglob("*.sh")
        if path.is_file()
        and path != verifier_shell
        and runtime_output_root not in path.parents
    )


def verify_static_scope(repo_root: Path) -> None:
    shell_paths = collect_shell_paths(repo_root)
    require(len(shell_paths) == 47, f"shell scope count={len(shell_paths)}, expected 47")
    for path in shell_paths:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            fail(f"shell syntax failed for {path}: {result.stderr.strip()}")

    verifier_python = repo_root / "tests/integration/sc26_ae_v21_verifier.py"
    python_roots = ("SC26-AE/tools", "tests/unit", "tests/integration", "tests/e2e", "tests/performance", "tools/ae")
    python_paths = sorted(
        path
        for root in python_roots
        for path in (repo_root / root).rglob("*.py")
        if path.is_file() and path != verifier_python
    )
    python_paths.append(repo_root / "pretrain_llama.py")
    python_paths = sorted(set(python_paths))
    require(len(python_paths) == 36, f"Python scope count={len(python_paths)}, expected 36")
    for path in python_paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            fail(f"Python syntax failed for {path}: {exc}")

    production_paths = [repo_root / "SC26-AE" / "lib", repo_root / "SC26-AE" / "tools"]
    temporary_matches = [
        path
        for root in production_paths
        for path in root.rglob("*")
        if path.is_file() and "/tmp" in path.read_text(encoding="utf-8", errors="replace")
    ]
    require(not temporary_matches, f"hard-coded production temporary-root matches: {temporary_matches}")

    diff = subprocess.run(
        ["git", "diff", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:
        fail(f"git diff --check failed:\n{diff.stdout}{diff.stderr}")
    print("SHELL_SCOPE_COUNT=47")
    print("SHELL_SYNTAX_COUNT=47")
    print("VERIFIER_SHELL_SELF_EXCLUDED=1")
    print("RUNTIME_OUTPUT_SHELL_EXCLUDED=1")
    print("PYTHON_SCOPE_COUNT=36")
    print("PYTHON_SYNTAX_COUNT=36")
    print("VERIFIER_PYTHON_SELF_EXCLUDED=1")
    print("TMP_ROOT_SCAN=PASS")
    print("GIT_DIFF_CHECK=PASS")


def run(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    task_dir = repo_root / "task_memory" / "task_2026-07-15_sc26_ae_workflow"
    summary_path = task_dir / "summary.md"
    issues_path = task_dir / "issues.md"
    require(summary_path.is_file(), f"summary is missing: {summary_path}")
    require(issues_path.is_file(), f"issues ledger is missing: {issues_path}")
    summary = summary_path.read_text(encoding="utf-8")

    artifact_rows = extract_inventory(summary, ARTIFACT_BEGIN, ARTIFACT_END, 7, "artifact inventory")
    document_rows = extract_inventory(summary, DOCUMENT_BEGIN, DOCUMENT_END, 10, "authoritative inventory")
    verify_inventory(repo_root, artifact_rows, "artifact")
    verify_inventory(repo_root, document_rows, "document")
    print(f"ARTIFACT_COUNT={len(artifact_rows)}")
    print(f"DOCUMENT_COUNT={len(document_rows)}")
    supplemental_count = verify_supplemental(summary, task_dir)
    print(f"SUPPLEMENTAL_COUNT={supplemental_count}")
    verify_issue_headings(issues_path)
    verify_status_boundary(summary, args.expected_status)
    verify_optional_verifier_identity(summary, task_dir)
    verify_log_markers(task_dir)
    verify_static_scope(repo_root)
    print("V21_VERIFIER_STATUS=PASS")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--expected-status",
        choices=("VERIFIER_PENDING", "PASS"),
        default="VERIFIER_PENDING",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(sys.argv[1:] if argv is None else argv))
    except (OSError, subprocess.SubprocessError, VerificationFailure) as exc:
        print(f"V21_VERIFY_FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
