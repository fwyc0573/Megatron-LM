"""Unit tests for slowdown predictor asset loading and adapter behavior."""

from __future__ import annotations

import builtins
import json
import pathlib
import sys
import types

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extensions.slowdown_predictor import EchoSlowdownPredictor, load_slowdown_assets


SCALER_SPEC = {
    'feature_names': [
        'ground_truth',
        'Compute throughput',
        'Memory throughput',
        'DRAM throughput',
        'Achieved occupancy',
        'Maximum occupancy',
        'L1 hit rate',
        'L2 hit rate',
    ],
    'mean': [4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    'scale': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
}


def _write_valid_assets(tmp_path: pathlib.Path) -> pathlib.Path:
    assets_dir = tmp_path / 'slowdown_assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'scope': 'ddp_backward_only',
                'model_path': 'xgb_model.json',
                'scaler_path': 'standard_scaler.json',
                'label_prefix': 'cmd_trace',
                'clip_negative_slowdown': True,
                'generator_version': 'test-v2',
                'source_trace_dir': 'trace_dir',
                'source_nsys_sqlite': 'baseline.sqlite',
                'source_ncu_metrics': 'metrics.csv',
            }
        )
    )
    (assets_dir / 'kernel_features.json').write_text(
        json.dumps(
            {
                'kernel_a': {
                    'Compute throughput': 1.0,
                    'Memory throughput': 2.0,
                    'DRAM throughput': 3.0,
                    'Achieved occupancy': 4.0,
                    'Maximum occupancy': 5.0,
                    'L1 hit rate': 6.0,
                    'L2 hit rate': 7.0,
                }
            }
        )
    )
    (assets_dir / 'backward_kernel_blueprints.json').write_text(
        json.dumps(
            {
                'cmd-bwd-1': {
                    'rank': 0,
                    'stage_id': 0,
                    'batch_id': 1,
                    'iter_id': 7,
                    'mg_state': 'steady',
                    'baseline_duration_ms': 4.0,
                    'kernels': [
                        {
                            'kernel_name': 'kernel_a',
                            'start_offset_ms': 0.0,
                            'baseline_duration_ms': 4.0,
                        }
                    ],
                    'launch_markers': [
                        {
                            'comm_uid': 'comm-1',
                            'baseline_offset_ms': 2.0,
                            'bucket_id': 0,
                            'buffer_id': 0,
                        }
                    ],
                }
            }
        )
    )
    (assets_dir / 'standard_scaler.json').write_text(json.dumps(SCALER_SPEC), encoding='utf-8')
    return assets_dir


def test_load_slowdown_assets_normalizes_valid_assets(tmp_path: pathlib.Path) -> None:
    assets_dir = _write_valid_assets(tmp_path)

    assets = load_slowdown_assets(str(assets_dir))

    assert assets.manifest['scope'] == 'ddp_backward_only'
    assert assets.manifest['scaler_path'] == 'standard_scaler.json'
    assert assets.kernel_features['kernel_a']['DRAM throughput'] == pytest.approx(3.0)
    assert assets.backward_kernel_blueprints['cmd-bwd-1']['kernels'][0]['kernel_name'] == 'kernel_a'


def test_load_slowdown_assets_rejects_duplicate_json_keys(tmp_path: pathlib.Path) -> None:
    assets_dir = tmp_path / 'slowdown_assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / 'manifest.json').write_text(
        '{"scope":"ddp_backward_only","scope":"ddp_backward_only","model_path":"xgb_model.json","scaler_path":"standard_scaler.json","label_prefix":"cmd_trace","clip_negative_slowdown":true,"generator_version":"test-v2","source_trace_dir":"trace_dir","source_nsys_sqlite":"baseline.sqlite","source_ncu_metrics":"metrics.csv"}'
    )
    (assets_dir / 'kernel_features.json').write_text('{}')
    (assets_dir / 'backward_kernel_blueprints.json').write_text('{}')

    with pytest.raises(RuntimeError, match='Duplicate JSON key'):
        load_slowdown_assets(str(assets_dir))


class _FakeSeries(list):
    @property
    def iloc(self):
        return self

    def clip(self, lower: float = 0.0):
        return _FakeSeries([max(lower, value) for value in self])


class _FakeBooster:
    def load_model(self, model_path: str) -> None:
        self.model_path = model_path

    def predict(self, dtest):
        row = dtest[0]
        if row['ground_truth'] == pytest.approx(2.0):
            return [-0.25]
        raise AssertionError(f'Expected scaled ground_truth=2.0, got {row}')


class _FakeXGBoost(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(Booster=_FakeBooster, DMatrix=lambda rows: rows)


class _FakePandas(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(DataFrame=lambda rows, columns=None: rows, Series=_FakeSeries)


def test_echo_slowdown_predictor_applies_scaler_and_clips_negative_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setitem(sys.modules, 'xgboost', _FakeXGBoost())
    monkeypatch.setitem(sys.modules, 'pandas', _FakePandas())

    scaler_path = tmp_path / 'standard_scaler.json'
    scaler_path.write_text(json.dumps(SCALER_SPEC), encoding='utf-8')
    model_path = tmp_path / 'dummy_model.json'
    model_path.write_text('{}', encoding='utf-8')

    predictor = EchoSlowdownPredictor(str(model_path), scaler_path=str(scaler_path))
    feature_row = {
        'ground_truth': 8.0,
        'Compute throughput': 12.0,
        'Memory throughput': 22.0,
        'DRAM throughput': 32.0,
        'Achieved occupancy': 42.0,
        'Maximum occupancy': 52.0,
        'L1 hit rate': 62.0,
        'L2 hit rate': 72.0,
    }

    result = predictor.predict_slowdown(feature_row, input_overlap_ratio=1.0)

    assert result['predicted_slowdown_factor'] == pytest.approx(-0.25)
    assert result['predicted_slowdown_factor_clipped'] == pytest.approx(0.0)
    assert result['predicted_execution_time_clipped'] == pytest.approx(8.0)
    assert predictor.predict_slowdown_factor('kernel_a', 8.0, feature_row) == pytest.approx(0.0)


def test_echo_slowdown_predictor_fails_fast_without_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'xgboost':
            raise ModuleNotFoundError("No module named 'xgboost'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', _patched_import)

    with pytest.raises(RuntimeError, match='xgboost'):
        EchoSlowdownPredictor('dummy_model.json')


def test_load_slowdown_assets_fails_fast_without_scaler_path(tmp_path: pathlib.Path) -> None:
    assets_dir = tmp_path / 'slowdown_assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / 'manifest.json').write_text(
        json.dumps(
            {
                'scope': 'ddp_backward_only',
                'model_path': 'xgb_model.json',
                'label_prefix': 'cmd_trace',
                'clip_negative_slowdown': True,
                'generator_version': 'test-v2',
                'source_trace_dir': 'trace_dir',
                'source_nsys_sqlite': 'baseline.sqlite',
                'source_ncu_metrics': 'metrics.csv',
            }
        )
    )
    (assets_dir / 'kernel_features.json').write_text('{}')
    (assets_dir / 'backward_kernel_blueprints.json').write_text('{}')

    with pytest.raises(RuntimeError, match='scaler_path'):
        load_slowdown_assets(str(assets_dir))
