from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path("SC26-AE/tools/normalize_ncu_metrics.py")
SPEC = importlib.util.spec_from_file_location("normalize_ncu_metrics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


DETAIL_COLUMNS = ["ID", "Kernel Name", "Section Name", "Metric Name", "Metric Value"]
NAME_COLUMNS = ["ID", "Kernel Name"]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def complete_details() -> list[dict[str, str]]:
    metrics = {
        ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"): "10.0",
        ("GPU Speed Of Light Throughput", "Memory Throughput"): "20.0",
        ("GPU Speed Of Light Throughput", "DRAM Throughput"): "30.0",
        ("Occupancy", "Achieved Occupancy"): "40.0",
        ("Occupancy", "Theoretical Occupancy"): "50.0",
        ("Memory Workload Analysis", "L1/TEX Hit Rate"): "60.0",
        ("Memory Workload Analysis", "L2 Hit Rate"): "70.0",
    }
    return [
        {
            "ID": "1",
            "Kernel Name": "kernel_a",
            "Section Name": section,
            "Metric Name": metric,
            "Metric Value": value,
        }
        for (section, metric), value in metrics.items()
    ]


def test_normalize_writes_canonical_feature_schema(tmp_path: Path) -> None:
    details = tmp_path / "details.csv"
    names = tmp_path / "names.csv"
    output = tmp_path / "kernel_metric_output.csv"
    write_csv(details, DETAIL_COLUMNS, complete_details())
    write_csv(names, NAME_COLUMNS, [{"ID": "1", "Kernel Name": "kernel_a"}])

    count = MODULE.normalize_ncu_metrics(details, names, output)

    assert count == 1
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "Kernel Name": "kernel_a",
            "Compute throughput": "10.0",
            "Memory throughput": "20.0",
            "DRAM throughput": "30.0",
            "Achieved occupancy": "40.0",
            "Maximum occupancy": "50.0",
            "L1 hit rate": "60.0",
            "L2 hit rate": "70.0",
        }
    ]


def test_normalize_fails_when_required_metric_is_missing(tmp_path: Path) -> None:
    details = tmp_path / "details.csv"
    names = tmp_path / "names.csv"
    output = tmp_path / "kernel_metric_output.csv"
    rows = complete_details()[:-1]
    write_csv(details, DETAIL_COLUMNS, rows)
    write_csv(names, NAME_COLUMNS, [{"ID": "1", "Kernel Name": "kernel_a"}])

    with pytest.raises(ValueError, match="missing metric"):
        MODULE.normalize_ncu_metrics(details, names, output)


def test_normalize_fails_when_kernel_name_is_empty(tmp_path: Path) -> None:
    details = tmp_path / "details.csv"
    names = tmp_path / "names.csv"
    output = tmp_path / "kernel_metric_output.csv"
    write_csv(details, DETAIL_COLUMNS, complete_details())
    write_csv(names, NAME_COLUMNS, [{"ID": "1", "Kernel Name": ""}])

    with pytest.raises(ValueError, match="no kernel rows"):
        MODULE.normalize_ncu_metrics(details, names, output)
