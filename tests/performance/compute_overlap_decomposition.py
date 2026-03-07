#!/usr/bin/env python3
"""
Compute overlap decomposition for WS256 dense simulation variants.

Decomposes the existing 3-component split (comp_execute, comm_execute, bubble)
into a 4-component split (pure_comp, pure_comm, overlap, bubble) where:
    comp_execute = pure_comp + 0.5 * overlap
    comm_execute = pure_comm + 0.5 * overlap
    e2e = pure_comp + pure_comm + overlap + bubble

Overlap inputs:
    - groundtruth: overlap = 5.57% of e2e
    - ours, v1, v2: overlap error vs groundtruth = +1.05%
    - v3: overlap error vs groundtruth = -n%, solved from data
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Compute overlap decomposition for WS256 variants"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="task_memory/task_2026-03-04_reverse_groundtruth/logs/"
        "variant_groundtruth_ours_v1_v2_v3_e2e_comp_comm_bubble.csv",
        help="Path to the existing 3-component CSV",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="task_memory/task_2026-03-04_reverse_groundtruth/logs/"
        "variant_overlap_decomposition.csv",
        help="Path to write the 4-component CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="task_memory/task_2026-03-04_reverse_groundtruth/logs/"
        "variant_overlap_decomposition.json",
        help="Path to write JSON summary",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="task_memory/task_2026-03-04_reverse_groundtruth/"
        "test_report_2026-03-05_overlap_decomposition.md",
        help="Path to write markdown report",
    )
    parser.add_argument(
        "--gt-overlap-pct",
        type=float,
        default=5.57,
        help="Groundtruth overlap as percentage of e2e (default: 5.57)",
    )
    parser.add_argument(
        "--ours-overlap-error-pct",
        type=float,
        default=1.05,
        help="Ours/v1/v2 overlap relative error vs groundtruth in percent (default: +1.05)",
    )
    args = parser.parse_args()

    # =========================================================================
    # 1. Read existing 3-component CSV
    # =========================================================================
    variants = {}
    with open(args.input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["variant"]
            variants[name] = {
                "e2e_ms": float(row["e2e_ms"]),
                "comp_execute_ms": float(row["comp_execute_ms"]),
                "comm_execute_ms": float(row["comm_execute_ms"]),
                "bubble_ms": float(row["bubble_ms"]),
                "e2e_error_pct": float(row["e2e_error_pct"]),
            }

    gt = variants["groundtruth"]
    ours = variants["ours"]
    v3 = variants["v3"]

    # =========================================================================
    # 2. Compute overlaps
    # =========================================================================

    # Groundtruth overlap
    overlap_gt = (args.gt_overlap_pct / 100.0) * gt["e2e_ms"]
    print(f"[Step 1] Groundtruth overlap = {args.gt_overlap_pct}% * {gt['e2e_ms']:.6f}")
    print(f"         = {overlap_gt:.6f} ms\n")

    # Ours / v1 / v2 overlap (error +1.05% relative to groundtruth)
    ours_overlap_error_frac = args.ours_overlap_error_pct / 100.0
    overlap_ours = overlap_gt * (1.0 + ours_overlap_error_frac)
    print(f"[Step 2] Ours/v1/v2 overlap = overlap_gt * (1 + {args.ours_overlap_error_pct}%)")
    print(f"         = {overlap_gt:.6f} * {1.0 + ours_overlap_error_frac:.6f}")
    print(f"         = {overlap_ours:.6f} ms\n")

    # v3 overlap: derive from sum constraint
    # pure_comp + pure_comm are the same for ours and v3
    # (comp_execute + comm_execute) = pure_comp + pure_comm + overlap
    # => overlap_v3 = overlap_ours + [(comp+comm)_v3 - (comp+comm)_ours]
    sum_comp_comm_ours = ours["comp_execute_ms"] + ours["comm_execute_ms"]
    sum_comp_comm_v3 = v3["comp_execute_ms"] + v3["comm_execute_ms"]
    delta_sum = sum_comp_comm_v3 - sum_comp_comm_ours
    overlap_v3 = overlap_ours + delta_sum

    print(f"[Step 3] v3 overlap derivation:")
    print(f"         (comp+comm)_ours = {sum_comp_comm_ours:.6f}")
    print(f"         (comp+comm)_v3   = {sum_comp_comm_v3:.6f}")
    print(f"         delta            = {delta_sum:.6f}")
    print(f"         overlap_v3       = {overlap_ours:.6f} + ({delta_sum:.6f})")
    print(f"                          = {overlap_v3:.6f} ms\n")

    # Compute n for v3
    v3_overlap_error_pct = (overlap_v3 - overlap_gt) / overlap_gt * 100.0
    n_v3 = -v3_overlap_error_pct
    print(f"[Step 4] v3 overlap error vs groundtruth:")
    print(f"         error = (overlap_v3 - overlap_gt) / overlap_gt * 100")
    print(f"               = ({overlap_v3:.6f} - {overlap_gt:.6f}) / {overlap_gt:.6f} * 100")
    print(f"               = {v3_overlap_error_pct:.6f}%")
    print(f"         => n  = {n_v3:.6f}%\n")

    # Assign overlaps
    overlap_map = {
        "groundtruth": overlap_gt,
        "ours": overlap_ours,
        "v1": overlap_ours,  # same as ours
        "v2": overlap_ours,  # same as ours
        "v3": overlap_v3,
    }
    overlap_error_map = {
        "groundtruth": 0.0,
        "ours": args.ours_overlap_error_pct,
        "v1": args.ours_overlap_error_pct,
        "v2": args.ours_overlap_error_pct,
        "v3": v3_overlap_error_pct,
    }

    # =========================================================================
    # 3. Compute 4-component decomposition
    # =========================================================================
    results = {}
    order = ["groundtruth", "ours", "v1", "v2", "v3"]
    for name in order:
        v = variants[name]
        ov = overlap_map[name]
        pure_comp = v["comp_execute_ms"] - 0.5 * ov
        pure_comm = v["comm_execute_ms"] - 0.5 * ov
        bubble = v["bubble_ms"]
        e2e = v["e2e_ms"]

        # Sanity check
        recon = pure_comp + pure_comm + ov + bubble
        assert abs(recon - e2e) < 1e-4, (
            f"{name}: reconstruction mismatch {recon:.6f} vs {e2e:.6f}"
        )

        results[name] = {
            "e2e_ms": e2e,
            "pure_comp_ms": pure_comp,
            "pure_comm_ms": pure_comm,
            "overlap_ms": ov,
            "bubble_ms": bubble,
            "overlap_error_pct": overlap_error_map[name],
            # Also keep original 3-component for reference
            "comp_execute_ms": v["comp_execute_ms"],
            "comm_execute_ms": v["comm_execute_ms"],
        }

    # =========================================================================
    # 4. Compute errors vs groundtruth for each component
    # =========================================================================
    gt_res = results["groundtruth"]
    for name in order:
        r = results[name]
        for comp_key in ["pure_comp_ms", "pure_comm_ms", "overlap_ms", "bubble_ms", "e2e_ms"]:
            gt_val = gt_res[comp_key]
            sim_val = r[comp_key]
            if gt_val != 0:
                err = (sim_val - gt_val) / gt_val * 100.0
            else:
                err = 0.0
            error_key = comp_key.replace("_ms", "_error_pct")
            r[error_key] = err

    # =========================================================================
    # 5. Print summary table
    # =========================================================================
    print("=" * 120)
    print("4-Component Decomposition Summary (Critical Rank)")
    print("=" * 120)
    header = (
        f"{'variant':<14} {'e2e_ms':>12} {'pure_comp':>12} {'pure_comm':>12} "
        f"{'overlap':>12} {'bubble':>12} | "
        f"{'e2e_err%':>10} {'comp_err%':>10} {'comm_err%':>10} "
        f"{'ovlp_err%':>10} {'bubl_err%':>10}"
    )
    print(header)
    print("-" * 120)
    for name in order:
        r = results[name]
        line = (
            f"{name:<14} "
            f"{r['e2e_ms']:>12.6f} "
            f"{r['pure_comp_ms']:>12.6f} "
            f"{r['pure_comm_ms']:>12.6f} "
            f"{r['overlap_ms']:>12.6f} "
            f"{r['bubble_ms']:>12.6f} | "
            f"{r['e2e_error_pct']:>10.4f} "
            f"{r['pure_comp_error_pct']:>10.4f} "
            f"{r['pure_comm_error_pct']:>10.4f} "
            f"{r['overlap_error_pct']:>10.4f} "
            f"{r['bubble_error_pct']:>10.4f}"
        )
        print(line)
    print("=" * 120)

    # Also print percentage of e2e for each component
    print()
    print("=" * 120)
    print("4-Component as Percentage of E2E")
    print("=" * 120)
    header2 = (
        f"{'variant':<14} {'pure_comp%':>12} {'pure_comm%':>12} "
        f"{'overlap%':>12} {'bubble%':>12} {'sum%':>12}"
    )
    print(header2)
    print("-" * 120)
    for name in order:
        r = results[name]
        e2e = r["e2e_ms"]
        pc_pct = r["pure_comp_ms"] / e2e * 100
        pm_pct = r["pure_comm_ms"] / e2e * 100
        ov_pct = r["overlap_ms"] / e2e * 100
        bb_pct = r["bubble_ms"] / e2e * 100
        total_pct = pc_pct + pm_pct + ov_pct + bb_pct
        line = (
            f"{name:<14} "
            f"{pc_pct:>12.4f} "
            f"{pm_pct:>12.4f} "
            f"{ov_pct:>12.4f} "
            f"{bb_pct:>12.4f} "
            f"{total_pct:>12.4f}"
        )
        print(line)
    print("=" * 120)

    # =========================================================================
    # 6. Write CSV output
    # =========================================================================
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    csv_fields = [
        "variant",
        "e2e_ms",
        "pure_comp_ms",
        "pure_comm_ms",
        "overlap_ms",
        "bubble_ms",
        "comp_execute_ms",
        "comm_execute_ms",
        "e2e_error_pct",
        "pure_comp_error_pct",
        "pure_comm_error_pct",
        "overlap_error_pct",
        "bubble_error_pct",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for name in order:
            r = results[name]
            row = {"variant": name}
            for field in csv_fields[1:]:
                row[field] = f"{r[field]:.6f}"
            writer.writerow(row)
    print(f"\nCSV written to: {args.output_csv}")

    # =========================================================================
    # 7. Write JSON output
    # =========================================================================
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    json_out = {
        "description": "4-component overlap decomposition for WS256 dense simulation variants",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "parameters": {
            "gt_overlap_pct_of_e2e": args.gt_overlap_pct,
            "ours_v1_v2_overlap_error_pct": args.ours_overlap_error_pct,
            "v3_overlap_error_pct": v3_overlap_error_pct,
            "v3_n_pct": n_v3,
        },
        "overlap_values_ms": {name: overlap_map[name] for name in order},
        "decomposition": {name: results[name] for name in order},
    }
    with open(args.output_json, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON written to: {args.output_json}")

    # =========================================================================
    # 8. Write markdown report
    # =========================================================================
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    report = generate_markdown_report(args, results, order, overlap_map,
                                       overlap_error_map, n_v3, overlap_gt,
                                       overlap_ours, overlap_v3, delta_sum)
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Report written to: {args.output_report}")


def generate_markdown_report(args, results, order, overlap_map, overlap_error_map,
                              n_v3, overlap_gt, overlap_ours, overlap_v3, delta_sum):
    """Generate a markdown report for the overlap decomposition."""
    gt = results["groundtruth"]
    lines = []
    lines.append("## Modification History\n")
    lines.append("| Date       | Summary of Changes |")
    lines.append("|------------|--------------------|")
    lines.append("| 2026-03-05 | Initial overlap decomposition report |")
    lines.append("")
    lines.append("# Overlap Decomposition Report (WS256 Dense H800)")
    lines.append("")
    lines.append("**Date**: 2026-03-05")
    lines.append("")
    lines.append("## 1. Decomposition Model")
    lines.append("")
    lines.append("The existing 3-component split:")
    lines.append("- `comp_execute = pure_comp + 0.5 * overlap`")
    lines.append("- `comm_execute = pure_comm + 0.5 * overlap`")
    lines.append("- `e2e = comp_execute + comm_execute + bubble`")
    lines.append("")
    lines.append("Is refined to a 4-component split:")
    lines.append("- `e2e = pure_comp + pure_comm + overlap + bubble`")
    lines.append("")
    lines.append("Where `overlap` represents the time during which computation and "
                 "communication execute concurrently.")
    lines.append("")
    lines.append("## 2. Overlap Inputs")
    lines.append("")
    lines.append(f"- **Groundtruth**: overlap = {args.gt_overlap_pct}% of e2e "
                 f"= {overlap_gt:.6f} ms")
    lines.append(f"- **Ours, v1, v2**: overlap error vs groundtruth = "
                 f"+{args.ours_overlap_error_pct}% → overlap = {overlap_ours:.6f} ms")
    lines.append(f"- **v3**: overlap error vs groundtruth = {-n_v3:+.6f}% (solved) "
                 f"→ overlap = {overlap_v3:.6f} ms")
    lines.append("")
    lines.append("### v3 Overlap Derivation")
    lines.append("")
    lines.append("v3 shares the same `pure_comp` and `pure_comm` as ours. "
                 "Only `overlap` changes.")
    lines.append("")
    lines.append("Using the sum constraint:")
    lines.append("```")
    lines.append("(comp_execute + comm_execute) = pure_comp + pure_comm + overlap")
    lines.append(f"(comp+comm)_ours = {results['ours']['comp_execute_ms']:.6f} + "
                 f"{results['ours']['comm_execute_ms']:.6f} = "
                 f"{results['ours']['comp_execute_ms'] + results['ours']['comm_execute_ms']:.6f}")
    lines.append(f"(comp+comm)_v3   = {results['v3']['comp_execute_ms']:.6f} + "
                 f"{results['v3']['comm_execute_ms']:.6f} = "
                 f"{results['v3']['comp_execute_ms'] + results['v3']['comm_execute_ms']:.6f}")
    lines.append(f"delta            = {delta_sum:.6f}")
    lines.append(f"overlap_v3       = overlap_ours + delta = {overlap_ours:.6f} + "
                 f"({delta_sum:.6f}) = {overlap_v3:.6f} ms")
    lines.append("```")
    lines.append("")
    lines.append(f"**v3 overlap error n = {n_v3:.6f}%**")
    lines.append("")

    # 4-component table
    lines.append("## 3. 4-Component Decomposition (Critical Rank, ms)")
    lines.append("")
    lines.append("| variant | e2e_ms | pure_comp_ms | pure_comm_ms | overlap_ms | bubble_ms |")
    lines.append("|---------|--------|-------------|-------------|-----------|----------|")
    for name in order:
        r = results[name]
        lines.append(
            f"| {name} | {r['e2e_ms']:.6f} | {r['pure_comp_ms']:.6f} | "
            f"{r['pure_comm_ms']:.6f} | {r['overlap_ms']:.6f} | {r['bubble_ms']:.6f} |"
        )
    lines.append("")

    # Error table
    lines.append("## 4. Per-Component Error vs Groundtruth (%)")
    lines.append("")
    lines.append("| variant | e2e_err% | pure_comp_err% | pure_comm_err% | "
                 "overlap_err% | bubble_err% |")
    lines.append("|---------|---------|---------------|---------------|"
                 "------------|------------|")
    for name in order:
        r = results[name]
        lines.append(
            f"| {name} | {r['e2e_error_pct']:.6f} | "
            f"{r['pure_comp_error_pct']:.6f} | "
            f"{r['pure_comm_error_pct']:.6f} | "
            f"{r['overlap_error_pct']:.6f} | "
            f"{r['bubble_error_pct']:.6f} |"
        )
    lines.append("")

    # Percentage of e2e table
    lines.append("## 5. Component as Percentage of E2E")
    lines.append("")
    lines.append("| variant | pure_comp% | pure_comm% | overlap% | bubble% | sum% |")
    lines.append("|---------|-----------|-----------|---------|--------|------|")
    for name in order:
        r = results[name]
        e2e = r["e2e_ms"]
        pc = r["pure_comp_ms"] / e2e * 100
        pm = r["pure_comm_ms"] / e2e * 100
        ov = r["overlap_ms"] / e2e * 100
        bb = r["bubble_ms"] / e2e * 100
        total = pc + pm + ov + bb
        lines.append(
            f"| {name} | {pc:.4f} | {pm:.4f} | {ov:.4f} | {bb:.4f} | {total:.4f} |"
        )
    lines.append("")

    # Relationship with original 3-component
    lines.append("## 6. Relationship to Original 3-Component Split")
    lines.append("")
    lines.append("| variant | comp_execute_ms | = pure_comp + 0.5*overlap | "
                 "comm_execute_ms | = pure_comm + 0.5*overlap |")
    lines.append("|---------|----------------|--------------------------|"
                 "----------------|--------------------------|")
    for name in order:
        r = results[name]
        recon_comp = r["pure_comp_ms"] + 0.5 * r["overlap_ms"]
        recon_comm = r["pure_comm_ms"] + 0.5 * r["overlap_ms"]
        lines.append(
            f"| {name} | {r['comp_execute_ms']:.6f} | {recon_comp:.6f} | "
            f"{r['comm_execute_ms']:.6f} | {recon_comm:.6f} |"
        )
    lines.append("")

    # Artifacts
    lines.append("## 7. Artifacts")
    lines.append("")
    lines.append("- Computation script: `tests/performance/compute_overlap_decomposition.py`")
    lines.append(f"- 4-component CSV: `{args.output_csv}`")
    lines.append(f"- JSON summary: `{args.output_json}`")
    lines.append(f"- This report: `{args.output_report}`")
    lines.append("")
    lines.append("## 8. Reproducible Command")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM")
    lines.append(f"python tests/performance/compute_overlap_decomposition.py \\")
    lines.append(f"  --gt-overlap-pct {args.gt_overlap_pct} \\")
    lines.append(f"  --ours-overlap-error-pct {args.ours_overlap_error_pct}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
