"""Pytest collection controls for archived topology exploration scripts.

These scripts were migrated from legacy `StaticGraphs/test/` for discoverability,
but they are not stable unit tests (external deps, interactive execution, or
manual experiment assumptions). Keep them excluded from default CI-style runs.
"""

collect_ignore = [
    "3d_group_test.py",
    "rank_init_test.py",
    "onnx_test.py",
    "onnx_specific_op_test.py",
    "fw_prop_graph_get.py",
    "model_read_test.py",
    "torch_fx_test.py",
]
