import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = REPO_ROOT / "tests" / "integration" / "sc26_ae_v21_verifier.py"


def load_verifier_module():
    spec = importlib.util.spec_from_file_location("sc26_ae_v21_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_shell_paths_excludes_runtime_output(tmp_path: Path) -> None:
    source_script = tmp_path / "SC26-AE" / "task1_gpt175b.sh"
    runtime_script = tmp_path / "SC26-AE" / "output" / "_work" / "source" / "run.sh"
    unit_script = tmp_path / "tests" / "unit" / "test_example.sh"
    for path in (source_script, runtime_script, unit_script):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    verifier = load_verifier_module()
    observed = {
        path.relative_to(tmp_path).as_posix()
        for path in verifier.collect_shell_paths(tmp_path)
    }

    assert observed == {
        "SC26-AE/task1_gpt175b.sh",
        "tests/unit/test_example.sh",
    }
