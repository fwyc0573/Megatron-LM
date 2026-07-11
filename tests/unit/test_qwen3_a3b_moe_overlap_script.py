import os
import stat
import subprocess
from pathlib import Path

SCRIPT_PATH = Path('examples/pretrain_qwen3_30b_a3b_moe.sh').resolve()
WRAPPER_PATH = Path('examples/pretrain_qwen3_30b_a3b_moe_ddp_overlap_trace.sh').resolve()


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_script(tmp_path: Path, *, script_path: Path, mode: str, env_overrides=None) -> list[str]:
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    torchrun_log = tmp_path / f'{script_path.stem}_{mode}.log'

    _write_executable(
        fake_bin / 'torchrun',
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\n' \"$@\" > \"${TORCHRUN_LOG}\"\n",
    )

    env = os.environ.copy()
    env.update(
        {
            'PATH': f"{fake_bin}:{env['PATH']}",
            'TORCHRUN_LOG': str(torchrun_log),
            'MODE': mode,
            'MODEL_PROFILE': 'smoke',
            'TRAIN_ITERS': '1',
            'TRACE_START': '0',
            'DO_TRACE': 'True',
            'OVERLAP_GRAD_REDUCE': '1',
            'DDP_BUCKET_SIZE': '1234',
            'GPUS_PER_NODE': '1',
            'PP': '1',
            'TP': '1',
            'EP': '1',
            'CP': '1',
            'MICRO_BATCH_SIZE': '1',
            'SEQ_LEN': '128',
            'MASTER_PORT': '29600',
        }
    )
    if env_overrides:
        env.update(env_overrides)
    if mode == 'scaling':
        env.update(
            {
                'SCALE_GPU': '0',
                'FAKE_WORLD_SIZE': '1',
                'FAKE_PP': '1',
                'FAKE_TP': '1',
                'FAKE_EXP': '1',
                'FAKE_RANK_ORDER': '0',
            }
        )

    subprocess.run(
        ['bash', str(script_path)],
        cwd=script_path.parent.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return torchrun_log.read_text(encoding='utf-8').splitlines()


def test_qwen3_a3b_moe_distributed_overlap_args_are_forwarded(tmp_path: Path) -> None:
    argv = _run_script(tmp_path, script_path=SCRIPT_PATH, mode='distributed')

    assert '--overlap-grad-reduce' in argv
    assert '--ddp-bucket-size' in argv
    assert '1234' in argv
    assert '--do-trace' in argv
    assert 'True' in argv
    assert '--trace-ddp-grad-overlap' not in argv
    assert '--is-scaling-mode' not in argv


def test_qwen3_a3b_moe_scaling_overlap_args_are_forwarded(tmp_path: Path) -> None:
    argv = _run_script(tmp_path, script_path=SCRIPT_PATH, mode='scaling')

    assert '--overlap-grad-reduce' in argv
    assert '--ddp-bucket-size' in argv
    assert '1234' in argv
    assert '--do-trace' in argv
    assert 'True' in argv
    assert '--trace-ddp-grad-overlap' not in argv
    assert '--is-scaling-mode' in argv
    assert '--fake-current-rank-id' in argv
    assert '0' in argv


def test_qwen3_a3b_moe_wrapper_defaults_enable_overlap_tracing(tmp_path: Path) -> None:
    argv = _run_script(
        tmp_path,
        script_path=WRAPPER_PATH,
        mode='distributed',
        env_overrides={'OVERLAP_GRAD_REDUCE': '', 'DO_TRACE': ''},
    )

    assert '--overlap-grad-reduce' in argv
    assert '--do-trace' in argv
    assert 'True' in argv
    assert '--trace-ddp-grad-overlap' not in argv
