import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path('tests/e2e/test_gpt67b_ddp_slowdown_lightweight.sh')


def _extract_function(function_name: str) -> str:
    lines = SCRIPT_PATH.read_text(encoding='utf-8').splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f'{function_name}()'):
            start = index
            break
    if start is None:
        raise AssertionError(f'Function not found: {function_name}')

    collected = []
    brace_depth = 0
    for line in lines[start:]:
        collected.append(line)
        brace_depth += line.count('{')
        brace_depth -= line.count('}')
        if brace_depth == 0:
            break
    return '\n'.join(collected) + '\n'


def test_latest_rank_file_returns_latest_match_without_pipefail_breakage(tmp_path: Path) -> None:
    older = tmp_path / 'foo_rank0_older.txt'
    latest = tmp_path / 'foo_rank0_latest.txt'
    older.write_text('older\n', encoding='utf-8')
    latest.write_text('latest\n', encoding='utf-8')
    os.utime(older, (1000, 1000))
    os.utime(latest, (2000, 2000))

    function_body = _extract_function('latest_rank_file')
    bash_program = f'''#!/usr/bin/env bash
set -euo pipefail
{function_body}
latest_rank_file {tmp_path!s} 0
'''
    result = subprocess.run(
        ['bash'],
        input=bash_program,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == str(latest)
    assert result.returncode == 0
