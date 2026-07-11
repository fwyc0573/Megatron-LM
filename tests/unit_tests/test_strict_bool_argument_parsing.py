import argparse
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

from megatron.training import arguments as training_arguments
from mg_scheduling import arguments as mg_scheduling_arguments


def _base_cli(script_name: str):
    return [
        script_name,
        '--num-layers',
        '2',
        '--hidden-size',
        '128',
        '--num-attention-heads',
        '8',
        '--max-position-embeddings',
        '4096',
        '--seq-length',
        '128',
        '--micro-batch-size',
        '1',
        '--global-batch-size',
        '1',
        '--train-iters',
        '1',
    ]


def _load_module(module_name: str, relative_path: str):
    path = Path(relative_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load module spec for {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sft_retro_module():
    stub_names = [
        'pretrain_gpt',
        'tools.retro.sft.dataset_conv',
    ]
    inserted = {}
    for name in stub_names:
        if name in sys.modules:
            continue
        module = type(sys)(name)
        inserted[name] = module
        sys.modules[name] = module

    training_module = sys.modules.get('megatron.training')
    if training_module is None:
        raise RuntimeError('megatron.training must already be importable for sft_retro test stubbing')

    missing_training_attrs = {}
    for attr_name, value in {
        'get_args': lambda: None,
        'get_retro_args': lambda: None,
        'print_rank_0': lambda *args, **kwargs: None,
        'get_timers': lambda: None,
        'get_tokenizer': lambda: None,
        'pretrain': lambda *args, **kwargs: None,
    }.items():
        if not hasattr(training_module, attr_name):
            missing_training_attrs[attr_name] = None
            setattr(training_module, attr_name, value)

    pretrain_gpt = sys.modules['pretrain_gpt']
    pretrain_gpt.model_provider = lambda *args, **kwargs: None
    pretrain_gpt.is_dataset_built_on_rank = lambda *args, **kwargs: True

    dataset_conv = sys.modules['tools.retro.sft.dataset_conv']
    dataset_conv.JsonQADataset = object
    dataset_conv.JsonQADatasetConfig = object
    dataset_conv.RetroJsonQADataset = object
    dataset_conv.RetroJsonQADatasetConfig = object

    try:
        return _load_module('sft_retro_test_module', 'tools/retro/sft/sft_retro.py')
    finally:
        for attr_name in missing_training_attrs:
            delattr(training_module, attr_name)
        for name in inserted:
            sys.modules.pop(name, None)


sim_engine_mg_sched_arguments = _load_module(
    'sim_engine_mg_sched_arguments',
    'megatron-sim-engine/src/scheduler/mg_scheduling/arguments.py',
)
sft_retro = _load_sft_retro_module()


@pytest.mark.parametrize(
    ('module', 'argv_suffix', 'attr_name', 'expected'),
    [
        (training_arguments, ['--onnx-safe', 'False'], 'onnx_safe', False),
        (training_arguments, ['--onnx-safe', '1'], 'onnx_safe', True),
        (training_arguments, ['--lazy-mpu-init', '0'], 'lazy_mpu_init', False),
        (training_arguments, ['--lazy-mpu-init', 'true'], 'lazy_mpu_init', True),
        (mg_scheduling_arguments, ['--onnx-safe', 'False'], 'onnx_safe', False),
        (mg_scheduling_arguments, ['--lazy-mpu-init', '1'], 'lazy_mpu_init', True),
        (sim_engine_mg_sched_arguments, ['--onnx-safe', 'off'], 'onnx_safe', False),
        (sim_engine_mg_sched_arguments, ['--lazy-mpu-init', 'yes'], 'lazy_mpu_init', True),
    ],
)
def test_strict_bool_flags_parse_expected_values(module, argv_suffix, attr_name, expected):
    argv = _base_cli(f'{module.__name__}.py') + argv_suffix
    with mock.patch.object(sys, 'argv', argv):
        args = module.parse_args(ignore_unknown_args=True)
    assert getattr(args, attr_name) is expected


@pytest.mark.parametrize('module', [training_arguments, mg_scheduling_arguments, sim_engine_mg_sched_arguments])
def test_strict_bool_flags_reject_invalid_tokens(module):
    argv = _base_cli(f'{module.__name__}.py') + ['--onnx-safe', 'maybe']
    with mock.patch.object(sys, 'argv', argv):
        with pytest.raises(SystemExit):
            module.parse_args(ignore_unknown_args=True)


def test_sft_retro_reset_eval_false_parses_correctly():
    parser = argparse.ArgumentParser()
    parser = sft_retro.get_tasks_args(parser)
    args = parser.parse_args(['--reset_eval', 'False'])
    assert args.reset_eval is False


def test_sft_retro_reset_eval_invalid_value_is_rejected():
    parser = argparse.ArgumentParser()
    parser = sft_retro.get_tasks_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(['--reset_eval', 'invalid'])
