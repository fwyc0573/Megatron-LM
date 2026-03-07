#!/usr/bin/env python3
"""
Compare compute-only (comp) durations between Scaling Mode and Distributed Mode
for DeepSeek-V3 variant MoE traces.

Scaling mode: comm durations are 0, so total duration = comp duration.
Distributed mode: total duration includes comm; comp = total - sum(comm sub-ops).

For scaling mode, we use the LAST iteration (batch_id=2).
For distributed mode, we average fwd/bwd comp across all micro-batches in one iteration.
Optimizer is once per iteration in both modes.
"""

import os
import re
import glob
import sys
from collections import defaultdict


def parse_trace_line(line):
    """Parse a single trace line and return structured data."""
    # Match: rank:<id>:<op_name>(key=val,...,sub_operations=[...])
    m = re.match(r'rank:(\d+):(\w+)\((.*)\)$', line.strip())
    if not m:
        return None

    rank_id = int(m.group(1))
    op_name = m.group(2)
    rest = m.group(3)

    # Extract fields before sub_operations
    # Find sub_operations=[...] at the end
    sub_ops_match = re.search(r',sub_operations=\[(.*)\]$', rest)
    sub_ops_str = sub_ops_match.group(1) if sub_ops_match else ""
    fields_str = rest[:sub_ops_match.start()] if sub_ops_match else rest

    # Parse key=value fields
    fields = {}
    for kv in re.finditer(r'(\w+)=([^,]+)', fields_str):
        key, val = kv.group(1), kv.group(2)
        try:
            fields[key] = float(val)
        except ValueError:
            fields[key] = val

    # Parse sub-operations to extract comm durations
    comm_duration_sum = 0.0
    comm_count = 0
    if sub_ops_str:
        # Each sub-op is a quoted string separated by commas (between quotes)
        sub_ops = re.findall(r"'([^']*)'", sub_ops_str)
        for sub_op in sub_ops:
            # Extract duration from sub-op
            dur_match = re.search(r'duration=([0-9.]+)', sub_op)
            if dur_match:
                dur = float(dur_match.group(1))
                comm_duration_sum += dur
                comm_count += 1

    return {
        'rank_id': rank_id,
        'op_name': op_name,
        'duration': fields.get('duration', 0.0),
        'stage_id': int(fields.get('stage_id', 0)),
        'batch_id': int(fields.get('batch_id', 0)),
        'mg_state': fields.get('mg_state', ''),
        'comm_duration_sum': comm_duration_sum,
        'comm_count': comm_count,
    }


def parse_file(filepath):
    """Parse all lines in a trace file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = parse_trace_line(line)
            if rec:
                records.append(rec)
    return records


def find_rank_file(directory, rank_id):
    """Find the trace file for a given rank in a directory."""
    pattern = os.path.join(directory, f'*_rank{rank_id}_*.txt')
    files = glob.glob(pattern)
    if not files:
        return None
    # Return the latest one (by filename timestamp)
    files.sort()
    return files[-1]


def analyze_scaling_mode(records, target_batch_id=2):
    """
    Extract comp durations from scaling mode for the target iteration.
    In scaling mode, comm durations are 0, so total = comp.
    """
    result = {'fwd': [], 'bwd': [], 'optimizer': []}
    for rec in records:
        if rec['batch_id'] != target_batch_id:
            continue
        if rec['op_name'] == 'forward_step':
            result['fwd'].append(rec['duration'])
        elif rec['op_name'] == 'backward_step':
            result['bwd'].append(rec['duration'])
        elif rec['op_name'] == 'optimizer_step':
            result['optimizer'].append(rec['duration'])
    return result


def analyze_distributed_mode(records):
    """
    Extract comp durations from distributed mode.
    comp = total_duration - sum(comm sub-op durations) for fwd/bwd.
    For optimizer, comp = total_duration directly.
    Returns per-micro-batch values and averages.
    """
    fwd_ops = []
    bwd_ops = []
    optimizer_ops = []

    for rec in records:
        if rec['op_name'] == 'forward_step':
            comp = rec['duration'] - rec['comm_duration_sum']
            fwd_ops.append({
                'batch_id': rec['batch_id'],
                'mg_state': rec['mg_state'],
                'total': rec['duration'],
                'comm': rec['comm_duration_sum'],
                'comp': comp,
                'comm_count': rec['comm_count'],
            })
        elif rec['op_name'] == 'backward_step':
            comp = rec['duration'] - rec['comm_duration_sum']
            bwd_ops.append({
                'batch_id': rec['batch_id'],
                'mg_state': rec['mg_state'],
                'total': rec['duration'],
                'comm': rec['comm_duration_sum'],
                'comp': comp,
                'comm_count': rec['comm_count'],
            })
        elif rec['op_name'] == 'optimizer_step':
            optimizer_ops.append({
                'batch_id': rec['batch_id'],
                'total': rec['duration'],
                'comp': rec['duration'],  # no comm in optimizer
            })

    return {'fwd': fwd_ops, 'bwd': bwd_ops, 'optimizer': optimizer_ops}


def main():
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../megatron-sim-engine/simulation_inputs/megatron_operation_log/'
        'h800_16gpus_deepseek_v3_variant_moe/'
        'pp2_tp1_exp4_expn32_dp8_nl32_hs2048_sl2048'
    )

    scaling_dir = os.path.join(base_dir, 'database_profile')
    distributed_dir = os.path.join(base_dir, 'global_ranks_profile')

    print("=" * 130)
    print("DeepSeek-V3 Variant MoE: Scaling Mode vs Distributed Mode Compute (Comp) Comparison")
    print(f"Config: PP=2, TP=1, EP=4, num_experts=32, DP=8, num_layers=32, hidden_size=2048, seq_len=2048")
    print(f"Scaling mode uses last iteration (batch_id=2); Distributed mode averages across micro-batches")
    print("=" * 130)

    # Collect results for all ranks
    all_results = []

    for rank_id in range(16):
        scaling_file = find_rank_file(scaling_dir, rank_id)
        distributed_file = find_rank_file(distributed_dir, rank_id)

        if not scaling_file or not distributed_file:
            print(f"WARNING: Missing file for rank {rank_id}")
            continue

        scaling_records = parse_file(scaling_file)
        distributed_records = parse_file(distributed_file)

        scaling_data = analyze_scaling_mode(scaling_records, target_batch_id=2)
        distributed_data = analyze_distributed_mode(distributed_records)

        # Scaling: single values for last iteration
        s_fwd = scaling_data['fwd'][0] if scaling_data['fwd'] else 0
        s_bwd = scaling_data['bwd'][0] if scaling_data['bwd'] else 0
        s_opt = scaling_data['optimizer'][0] if scaling_data['optimizer'] else 0

        # Distributed: per micro-batch fwd/bwd comp and average
        d_fwd_ops = distributed_data['fwd']
        d_bwd_ops = distributed_data['bwd']
        d_opt_ops = distributed_data['optimizer']

        d_fwd_comps = [op['comp'] for op in d_fwd_ops]
        d_bwd_comps = [op['comp'] for op in d_bwd_ops]
        d_fwd_avg = sum(d_fwd_comps) / len(d_fwd_comps) if d_fwd_comps else 0
        d_bwd_avg = sum(d_bwd_comps) / len(d_bwd_comps) if d_bwd_comps else 0
        d_opt = d_opt_ops[0]['comp'] if d_opt_ops else 0

        pp_stage = scaling_records[0]['stage_id'] if scaling_records else -1

        all_results.append({
            'rank_id': rank_id,
            'pp_stage': pp_stage,
            's_fwd': s_fwd,
            's_bwd': s_bwd,
            's_opt': s_opt,
            'd_fwd_avg': d_fwd_avg,
            'd_bwd_avg': d_bwd_avg,
            'd_opt': d_opt,
            'd_fwd_ops': d_fwd_ops,
            'd_bwd_ops': d_bwd_ops,
            'd_fwd_comps': d_fwd_comps,
            'd_bwd_comps': d_bwd_comps,
        })

    # ========== Detailed per-rank per-micro-batch report ==========
    print("\n" + "=" * 130)
    print("PART 1: Detailed Per-Rank Per-Micro-Batch Breakdown (Distributed Mode)")
    print("=" * 130)

    for res in all_results:
        rank_id = res['rank_id']
        pp_stage = res['pp_stage']
        print(f"\n--- Rank {rank_id} (PP stage {pp_stage}) ---")

        print(f"  Forward Steps (Distributed):")
        print(f"    {'batch_id':>8} {'mg_state':>10} {'total(ms)':>10} {'comm(ms)':>10} {'comp(ms)':>10} {'#comm_ops':>10}")
        for op in res['d_fwd_ops']:
            print(f"    {op['batch_id']:>8} {str(op['mg_state']):>10} {op['total']:>10.2f} {op['comm']:>10.2f} {op['comp']:>10.2f} {op['comm_count']:>10}")
        avg_comp = res['d_fwd_avg']
        print(f"    {'AVG':>8} {'':>10} {'':>10} {'':>10} {avg_comp:>10.2f}")

        print(f"  Backward Steps (Distributed):")
        print(f"    {'batch_id':>8} {'mg_state':>10} {'total(ms)':>10} {'comm(ms)':>10} {'comp(ms)':>10} {'#comm_ops':>10}")
        for op in res['d_bwd_ops']:
            print(f"    {op['batch_id']:>8} {str(op['mg_state']):>10} {op['total']:>10.2f} {op['comm']:>10.2f} {op['comp']:>10.2f} {op['comm_count']:>10}")
        avg_comp = res['d_bwd_avg']
        print(f"    {'AVG':>8} {'':>10} {'':>10} {'':>10} {avg_comp:>10.2f}")

    # ========== Summary comparison table ==========
    print("\n" + "=" * 130)
    print("PART 2: Summary Comparison — Scaling Comp vs Distributed Comp (average across micro-batches)")
    print("         For distributed fwd/bwd: comp = total - sum(comm sub-ops)")
    print("=" * 130)

    header = (f"{'Rank':>4} {'PP':>2} | "
              f"{'S_fwd(ms)':>10} {'D_fwd(ms)':>10} {'Δfwd(ms)':>10} {'err_fwd%':>9} | "
              f"{'S_bwd(ms)':>10} {'D_bwd(ms)':>10} {'Δbwd(ms)':>10} {'err_bwd%':>9} | "
              f"{'S_opt(ms)':>10} {'D_opt(ms)':>10} {'Δopt(ms)':>10} {'err_opt%':>9}")
    print(header)
    print("-" * len(header))

    fwd_errors = []
    bwd_errors = []
    opt_errors = []

    for res in all_results:
        s_fwd, d_fwd = res['s_fwd'], res['d_fwd_avg']
        s_bwd, d_bwd = res['s_bwd'], res['d_bwd_avg']
        s_opt, d_opt = res['s_opt'], res['d_opt']

        delta_fwd = s_fwd - d_fwd
        delta_bwd = s_bwd - d_bwd
        delta_opt = s_opt - d_opt

        err_fwd = (delta_fwd / d_fwd * 100) if d_fwd != 0 else float('inf')
        err_bwd = (delta_bwd / d_bwd * 100) if d_bwd != 0 else float('inf')
        err_opt = (delta_opt / d_opt * 100) if d_opt != 0 else float('inf')

        fwd_errors.append(err_fwd)
        bwd_errors.append(err_bwd)
        opt_errors.append(err_opt)

        print(f"{res['rank_id']:>4} {res['pp_stage']:>2} | "
              f"{s_fwd:>10.2f} {d_fwd:>10.2f} {delta_fwd:>10.2f} {err_fwd:>8.2f}% | "
              f"{s_bwd:>10.2f} {d_bwd:>10.2f} {delta_bwd:>10.2f} {err_bwd:>8.2f}% | "
              f"{s_opt:>10.2f} {d_opt:>10.2f} {delta_opt:>10.2f} {err_opt:>8.2f}%")

    print("-" * len(header))

    # Aggregation stats
    def stats(errors):
        abs_errs = [abs(e) for e in errors]
        avg = sum(errors) / len(errors)
        abs_avg = sum(abs_errs) / len(abs_errs)
        median = sorted(errors)[len(errors) // 2]
        abs_median = sorted(abs_errs)[len(abs_errs) // 2]
        mx = max(abs_errs)
        mn = min(abs_errs)
        return avg, abs_avg, median, abs_median, mx, mn

    print("\nAggregation Statistics (relative error %, reference = distributed comp):")
    print(f"{'':>20} {'mean':>8} {'|mean|':>8} {'median':>8} {'|median|':>8} {'max|err|':>8} {'min|err|':>8}")
    for name, errors in [('forward_step', fwd_errors), ('backward_step', bwd_errors), ('optimizer_step', opt_errors)]:
        avg, abs_avg, med, abs_med, mx, mn = stats(errors)
        print(f"  {name:>18} {avg:>7.2f}% {abs_avg:>7.2f}% {med:>7.2f}% {abs_med:>7.2f}% {mx:>7.2f}% {mn:>7.2f}%")

    # ========== Per-PP-stage breakdown ==========
    print("\n" + "=" * 130)
    print("PART 3: Per-PP-Stage Statistics")
    print("=" * 130)

    for stage in [0, 1]:
        stage_results = [r for r in all_results if r['pp_stage'] == stage]
        if not stage_results:
            continue

        stage_fwd_errs = [fwd_errors[all_results.index(r)] for r in stage_results]
        stage_bwd_errs = [bwd_errors[all_results.index(r)] for r in stage_results]
        stage_opt_errs = [opt_errors[all_results.index(r)] for r in stage_results]

        print(f"\n  PP Stage {stage} (ranks: {[r['rank_id'] for r in stage_results]}):")
        print(f"    {'':>20} {'mean':>8} {'|mean|':>8} {'median':>8} {'max|err|':>8}")
        for name, errs in [('forward_step', stage_fwd_errs), ('backward_step', stage_bwd_errs), ('optimizer_step', stage_opt_errs)]:
            avg, abs_avg, med, abs_med, mx, mn = stats(errs)
            print(f"    {name:>18} {avg:>7.2f}% {abs_avg:>7.2f}% {med:>7.2f}% {mx:>7.2f}%")

    # ========== Check pass/fail at 5% threshold ==========
    print("\n" + "=" * 130)
    print("PART 4: Pass/Fail Check (threshold = 5% relative error)")
    print("=" * 130)

    threshold = 5.0
    total_checks = 0
    total_pass = 0
    for res, ef, eb, eo in zip(all_results, fwd_errors, bwd_errors, opt_errors):
        for op_name, err in [('fwd', ef), ('bwd', eb), ('opt', eo)]:
            total_checks += 1
            status = "PASS" if abs(err) <= threshold else "FAIL"
            if status == "PASS":
                total_pass += 1
            if abs(err) > threshold:
                print(f"  FAIL: Rank {res['rank_id']} {op_name}: |{err:.2f}%| > {threshold}%")

    print(f"\n  Result: {total_pass}/{total_checks} checks passed ({total_pass/total_checks*100:.1f}%)")
    if total_pass == total_checks:
        print("  ✅ ALL CHECKS PASSED")
    else:
        print(f"  ❌ {total_checks - total_pass} checks FAILED")


if __name__ == '__main__':
    main()
