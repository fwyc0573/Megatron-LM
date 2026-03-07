#!/usr/bin/env python3

import argparse
import glob
import importlib.util
import json
import re
from pathlib import Path
from statistics import median


def _load_routing_profiles_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "megatron"
        / "profiler"
        / "moe"
        / "routing_profiles.py"
    )
    spec = importlib.util.spec_from_file_location("routing_profiles", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROUTING_PROFILES = _load_routing_profiles_module()


def parse_trace_line(line: str):
    match = re.match(r"rank:(\d+):(\w+)\((.*)\)$", line.strip())
    if not match:
        return None

    rank_id = int(match.group(1))
    op_name = match.group(2)
    rest = match.group(3)

    sub_ops_match = re.search(r",sub_operations=\[(.*)\]$", rest)
    sub_ops_str = sub_ops_match.group(1) if sub_ops_match else ""
    fields_str = rest[:sub_ops_match.start()] if sub_ops_match else rest

    fields = {}
    for key, value in re.findall(r"(\w+)=([^,]+)", fields_str):
        try:
            fields[key] = float(value)
        except ValueError:
            fields[key] = value

    comm_duration_sum = 0.0
    if sub_ops_str:
        for sub_op in re.findall(r"'([^']*)'", sub_ops_str):
            duration_match = re.search(r"duration=([0-9.]+)", sub_op)
            if duration_match:
                comm_duration_sum += float(duration_match.group(1))

    return {
        "rank_id": rank_id,
        "op_name": op_name,
        "duration": float(fields.get("duration", 0.0)),
        "stage_id": int(fields.get("stage_id", 0)),
        "batch_id": int(fields.get("batch_id", 0)),
        "mg_state": fields.get("mg_state", ""),
        "comm_duration_sum": comm_duration_sum,
    }


def parse_trace_file(filepath: Path):
    records = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = parse_trace_line(line)
            if record is not None:
                records.append(record)
    return records


def latest_rank_files(directory: Path):
    rank_to_files = {}
    for filepath in directory.glob("*_rank*_*.txt"):
        match = re.search(r"_rank(\d+)_", filepath.name)
        if not match:
            continue
        rank_id = int(match.group(1))
        rank_to_files.setdefault(rank_id, []).append(filepath)

    return {rank_id: sorted(files)[-1] for rank_id, files in rank_to_files.items()}


def summarize_rank_proxy(records, mode: str):
    op_buckets = {"forward_step": [], "backward_step": [], "optimizer_step": []}
    target_batch_id = None
    if mode == "scaling":
        batch_ids = [record["batch_id"] for record in records if record["op_name"] in op_buckets]
        target_batch_id = max(batch_ids) if batch_ids else None

    stage_id = None
    for record in records:
        if record["op_name"] not in op_buckets:
            continue
        if mode == "scaling" and target_batch_id is not None and record["batch_id"] != target_batch_id:
            continue
        stage_id = record["stage_id"]
        op_buckets[record["op_name"]].append(record)

    summary = {}
    for op_name, bucket in op_buckets.items():
        if not bucket:
            summary[op_name] = 0.0
            continue
        if op_name == "optimizer_step":
            summary[op_name] = sum(item["duration"] for item in bucket) / len(bucket)
            continue
        comp_values = [item["duration"] - item["comm_duration_sum"] for item in bucket]
        summary[op_name] = sum(comp_values) / len(comp_values)

    return {
        "stage_id": stage_id,
        "forward_step": summary["forward_step"],
        "backward_step": summary["backward_step"],
        "optimizer_step": summary["optimizer_step"],
        "critical_path_proxy_ms": (
            summary["forward_step"] + summary["backward_step"] + summary["optimizer_step"]
        ),
    }


def compute_stage_straggler_ratio(rank_summaries):
    stage_to_values = {}
    for item in rank_summaries.values():
        stage_to_values.setdefault(item["stage_id"], []).append(item["critical_path_proxy_ms"])

    stage_ratios = {}
    for stage_id, values in stage_to_values.items():
        stage_ratios[stage_id] = max(values) / median(values)
    worst_stage = max(stage_ratios, key=stage_ratios.get)
    return stage_ratios[worst_stage], worst_stage, stage_ratios


def build_report_row(args, skew_mode: str):
    distributed_dir = Path(args.distributed_dir)
    scaling_dir = Path(args.scaling_dir)
    distributed_files = latest_rank_files(distributed_dir)
    scaling_files = latest_rank_files(scaling_dir)
    common_ranks = sorted(set(distributed_files) & set(scaling_files))
    if not common_ranks:
        raise RuntimeError("No overlapping ranks found between distributed and scaling traces")

    distributed_summary = {}
    scaling_summary = {}
    for rank_id in common_ranks:
        distributed_summary[rank_id] = summarize_rank_proxy(
            parse_trace_file(distributed_files[rank_id]), mode="distributed"
        )
        scaling_summary[rank_id] = summarize_rank_proxy(
            parse_trace_file(scaling_files[rank_id]), mode="scaling"
        )

    gt_ratio, gt_stage, gt_stage_ratios = compute_stage_straggler_ratio(distributed_summary)
    moye_ratio, moye_stage, moye_stage_ratios = compute_stage_straggler_ratio(scaling_summary)

    gt_stage_values = [
        item["critical_path_proxy_ms"]
        for item in distributed_summary.values()
        if item["stage_id"] == gt_stage
    ]
    moye_stage_values = [
        item["critical_path_proxy_ms"]
        for item in scaling_summary.values()
        if item["stage_id"] == gt_stage
    ]
    gt_critical_path_proxy_ms = max(gt_stage_values)
    moye_critical_path_proxy_ms = max(moye_stage_values)
    critical_path_proxy_error_pct = abs(
        (moye_critical_path_proxy_ms - gt_critical_path_proxy_ms) / gt_critical_path_proxy_ms * 100.0
    )

    routing_indices = ROUTING_PROFILES.build_controlled_routing_indices(
        num_tokens=args.seq_len * args.micro_batch_size,
        topk=args.topk,
        num_experts=args.num_experts,
        skew_mode=skew_mode,
        num_groups=args.num_groups,
        group_topk=args.group_topk,
    )
    load_summary = ROUTING_PROFILES.summarize_expert_load(routing_indices, args.num_experts)

    return {
        "skew_mode": skew_mode,
        "hottest_to_median_expert_load": round(load_summary["hottest_to_median"], 2),
        "ground_truth_straggler_ratio": round(gt_ratio, 2),
        "moye_straggler_ratio": round(moye_ratio, 2),
        "critical_path_proxy_error_pct": round(critical_path_proxy_error_pct, 2),
        "ground_truth_stage": gt_stage,
        "moye_stage": moye_stage,
        "ground_truth_stage_ratios": gt_stage_ratios,
        "moye_stage_ratios": moye_stage_ratios,
    }


def render_markdown(rows):
    lines = [
        "| Skew level | Hottest/median expert load | GT straggler ratio | Moye straggler ratio | Critical-path proxy error |",
        "|---|---:|---:|---:|---:|",
    ]
    display_names = {
        "balanced": "Balanced",
        "moderate_skew": "Moderate",
        "strong_skew": "Strong",
    }
    for row in rows:
        lines.append(
            "| {label} | {load:.2f} | {gt:.2f} | {moye:.2f} | {err:.2f}% |".format(
                label=display_names[row["skew_mode"]],
                load=row["hottest_to_median_expert_load"],
                gt=row["ground_truth_straggler_ratio"],
                moye=row["moye_straggler_ratio"],
                err=row["critical_path_proxy_error_pct"],
            )
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize DeepSeek MoE routing-skew traces")
    parser.add_argument("--distributed-dir", required=True)
    parser.add_argument("--scaling-dir", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--group-topk", type=int, default=None)
    parser.add_argument("--skew-mode", required=True, choices=["balanced", "moderate_skew", "strong_skew"])
    parser.add_argument("--markdown-path", required=True)
    parser.add_argument("--json-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    row = build_report_row(args, args.skew_mode)

    markdown = render_markdown([row])
    markdown_path = Path(args.markdown_path)
    json_path = Path(args.json_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
