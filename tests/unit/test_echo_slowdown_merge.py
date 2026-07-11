import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path('Echo-slowdown/merge/merge_script.py')


spec = importlib.util.spec_from_file_location('echo_slowdown_merge_script', MODULE_PATH)
merge_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_script)


def test_merge_features_matches_normalized_kernel_names_and_occurrences():
    slowdown_df = pd.DataFrame(
        {
            'id': [1, 2, 3, 4],
            'kShortName': [
                'layer_norm_grad_input_kernel',
                'vectorized_elementwise_kernel',
                'vectorized_elementwise_kernel',
                'CatArrayBatchedCopy',
            ],
            'slowdown': [0.1, 0.2, 0.3, 0.4],
        }
    )
    features_df = pd.DataFrame(
        {
            'Kernel Name': [
                'layer_norm_grad_input_kernel_vectorized',
                'vectorized_elementwise_kernel',
                'vectorized_elementwise_kernel',
                'CatArrayBatchedCopy_aligned16_contig',
                'unmatched_kernel',
            ],
            'Compute throughput': [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    merged_df = merge_script.merge_features(slowdown_df, features_df)

    assert list(merged_df['kShortName']) == [
        'layer_norm_grad_input_kernel',
        'vectorized_elementwise_kernel',
        'vectorized_elementwise_kernel',
        'CatArrayBatchedCopy',
    ]
    assert list(merged_df['Kernel Name']) == [
        'layer_norm_grad_input_kernel_vectorized',
        'vectorized_elementwise_kernel',
        'vectorized_elementwise_kernel',
        'CatArrayBatchedCopy_aligned16_contig',
    ]
    assert list(merged_df['Compute throughput']) == [1.0, 2.0, 3.0, 4.0]


def test_merge_features_fails_fast_when_no_kernels_match():
    slowdown_df = pd.DataFrame({'kShortName': ['kernel_a']})
    features_df = pd.DataFrame({'Kernel Name': ['kernel_b']})

    with pytest.raises(RuntimeError, match='No matched kernels'):
        merge_script.merge_features(slowdown_df, features_df)
