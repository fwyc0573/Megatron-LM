#!/usr/bin/env python3
"""Normalize Nsight Compute CSV exports into slowdown feature rows.

The Task1 runner owns the workload capture.  This utility only converts the
two explicit ``ncu`` CSV exports (details and kernel short names) into the
canonical feature schema consumed by the slowdown asset builder.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import OrderedDict
from typing import Dict, Iterable, Mapping


REQUIRED_COLUMNS = (
    "Kernel Name",
    "Compute throughput",
    "Memory throughput",
    "DRAM throughput",
    "Achieved occupancy",
    "Maximum occupancy",
    "L1 hit rate",
    "L2 hit rate",
)


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"NCU CSV must be a regular file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"NCU CSV is empty: {path}")
    return rows


def _metric_value(rows: Iterable[Mapping[str, str]], section: str, metric: str) -> str:
    for row in rows:
        if row.get("Section Name", "").strip() == section and row.get("Metric Name", "").strip() == metric:
            value = row.get("Metric Value", "").strip()
            if value:
                return value
    raise ValueError(f"NCU details CSV is missing metric {section}/{metric}")


def normalize_ncu_metrics(
    details_csv: pathlib.Path,
    kernel_names_csv: pathlib.Path,
    output_csv: pathlib.Path,
) -> int:
    details = _read_csv(details_csv)
    names = _read_csv(kernel_names_csv)
    for required in ("ID", "Kernel Name", "Section Name", "Metric Name", "Metric Value"):
        if required not in details[0]:
            raise ValueError(f"NCU details CSV is missing column {required!r}")
    for required in ("ID", "Kernel Name"):
        if required not in names[0]:
            raise ValueError(f"NCU kernel-name CSV is missing column {required!r}")

    name_by_id: Dict[str, str] = OrderedDict()
    for row in names:
        identifier = row.get("ID", "").strip()
        kernel_name = row.get("Kernel Name", "").strip()
        if identifier and kernel_name and identifier not in name_by_id:
            name_by_id[identifier] = kernel_name

    metric_specs = (
        ("Compute throughput", "GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
        ("Memory throughput", "GPU Speed Of Light Throughput", "Memory Throughput"),
        ("DRAM throughput", "GPU Speed Of Light Throughput", "DRAM Throughput"),
        ("Achieved occupancy", "Occupancy", "Achieved Occupancy"),
        ("Maximum occupancy", "Occupancy", "Theoretical Occupancy"),
        ("L1 hit rate", "Memory Workload Analysis", "L1/TEX Hit Rate"),
        ("L2 hit rate", "Memory Workload Analysis", "L2 Hit Rate"),
    )
    rows_by_id: Dict[str, list[Mapping[str, str]]] = {}
    for row in details:
        identifier = row.get("ID", "").strip()
        if identifier:
            rows_by_id.setdefault(identifier, []).append(row)

    output_rows = []
    seen_names = set()
    for identifier, kernel_name in name_by_id.items():
        metric_rows = rows_by_id.get(identifier)
        if not metric_rows:
            raise ValueError(f"NCU details CSV has no metrics for kernel ID {identifier}")
        output = {"Kernel Name": kernel_name}
        for column, section, metric in metric_specs:
            output[column] = _metric_value(metric_rows, section, metric)
        seen_names.add(kernel_name)
        output_rows.append(output)
    if not output_rows:
        raise ValueError("NCU exports contain no kernel rows")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(output_rows)
    return len(seen_names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details-csv", type=pathlib.Path, required=True)
    parser.add_argument("--kernel-names-csv", type=pathlib.Path, required=True)
    parser.add_argument("--output-csv", type=pathlib.Path, required=True)
    args = parser.parse_args()
    count = normalize_ncu_metrics(args.details_csv, args.kernel_names_csv, args.output_csv)
    print(f"NCU_FEATURE_STATUS=verified")
    print(f"NCU_KERNEL_COUNT={count}")
    print(f"NCU_FEATURE_CSV={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
