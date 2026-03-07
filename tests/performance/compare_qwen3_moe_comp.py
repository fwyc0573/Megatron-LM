#!/usr/bin/env python3
"""
Compare compute-only (comp) durations between Scaling Mode and Distributed Mode
for Qwen3 30B-A3B MoE traces across 3 different PP/EP configurations on H800 16-GPU.

Configurations:
  A: PP=2, TP=1, EP=8, DP=8  (pp2_tp1_exp8_expn128_dp8_nl48_hs2048_sl2048)
  B: PP=4, TP=1, EP=4, DP=4  (pp4_tp1_exp4_expn128_dp4_nl48_hs2048_sl2048)
  C: PP=8, TP=1, EP=2, DP=2  (pp8_tp1_exp2_expn128_dp2_nl48_hs2048_sl2048)

Scaling mode: comm sub-op durations are 0.0, so op duration = comp duration.
Distributed mode: comp = op_duration - sum(comm sub-op durations).

For scaling mode, there is exactly 1 profiled iteration per rank (6-7 lines).
For distributed mode, there are multiple micro-batches in the 1F1B pipeline
schedule; we compute per-micro-batch comp and then take the average.
"""

import os
import re
import glob
import sys
from collections import defaultdict

# =========================================================================== #
# Configuration
# =========================================================================== #

CONFIGS = [
    {
        'name': 'Config A (PP=2, EP=8)',
        'dir_name': 'pp2_tp1_exp8_expn128_dp8_nl48_hs2048_sl2048',
        'pp': 2, 'tp': 1, 'ep': 8, 'dp': 8, 'num_experts': 128,
    },
    {
        'name': 'Config B (PP=4, EP=4)',
        'dir_name': 'pp4_tp1_exp4_expn128_dp4_nl48_hs2048_sl2048',
        'pp': 4, 'tp': 1, 'ep': 4, 'dp': 4, 'num_experts': 128,
    },
    {
        'name': 'Config C (PP=8, EP=2)',
        'dir_name': 'pp8_tp1_exp2_expn128_dp2_nl48_hs2048_sl2048',
        'pp': 8, 'tp': 1, 'ep': 2, 'dp': 2, 'num_experts': 128,
    },
]

WORLD_SIZE = 16
TARGET_OPS = {'forward_step', 'backward_step', 'optimizer_step'}
THRESHOLD_PCT = 5.0

# =========================================================================== #
# Trace Parsing
# =========================================================================== #

def parse_trace_line(line):
    """Parse a single trace line and return structured data."""
    m = re.match(r'rank:(\d+):(\w+)\((.*)\)$', line.strip())
    if not m:
        return None

    rank_id = int(m.group(1))
    op_name = m.group(2)
    rest = m.group(3)

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
        sub_ops = re.findall(r"'([^']*)'", sub_ops_str)
        for sub_op in sub_ops:
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
    """Find trace file for a given rank in a directory."""
    pattern = os.path.join(directory, f'*rank{rank_id}_*.txt')
    files = glob.glob(pattern)
    if not files:
        return None
    # If multiple files, return the latest by timestamp
    return sorted(files)[-1]


# =========================================================================== #
# Analysis Helpers
# =========================================================================== #

def analyze_scaling_mode(records):
    """
    Scaling mode: exactly 1 profiled iteration.
    Comm sub-op durations are 0.0, so op duration = comp duration.
    Return dict of op_name -> comp_duration (ms).
    """
    result = {}
    for rec in records:
        if rec['op_name'] in TARGET_OPS:
            result[rec['op_name']] = rec['duration']
    return result


def analyze_distributed_mode(records):
    """
    Distributed mode: multiple micro-batches in 1F1B schedule.
    For forward_step/backward_step: collect per-micro-batch comp (= total - comm).
    For optimizer_step: single value per iteration.
    Return dict of op_name -> {
        'per_mb': [(batch_id, mg_state, total, comm, comp, comm_count), ...],
        'avg_comp': float,
    }
    """
    result = {}
    for op_name in TARGET_OPS:
        ops = [r for r in records if r['op_name'] == op_name]
        per_mb = []
        for r in ops:
            total = r['duration']
            comm = r['comm_duration_sum']
            comp = total - comm
            per_mb.append({
                'batch_id': r['batch_id'],
                'mg_state': r['mg_state'],
                'total': total,
                'comm': comm,
                'comp': comp,
                'comm_count': r['comm_count'],
            })
        avg_comp = sum(m['comp'] for m in per_mb) / len(per_mb) if per_mb else 0
        result[op_name] = {
            'per_mb': per_mb,
            'avg_comp': avg_comp,
            'count': len(per_mb),
        }
    return result


def get_pp_stage(rank_id, pp_size):
    """Compute PP stage from rank_id given the PP size (TP=1)."""
    ranks_per_stage = WORLD_SIZE // pp_size
    return rank_id // ranks_per_stage


def stats(errors):
    """Compute aggregation statistics for a list of error values."""
    if not errors:
        return {'mean': 0, 'abs_mean': 0, 'median': 0, 'abs_median': 0, 'max': 0, 'min': 0}
    abs_errs = [abs(e) for e in errors]
    n = len(errors)
    s = sorted(errors)
    a = sorted(abs_errs)
    return {
        'mean': sum(errors) / n,
        'abs_mean': sum(abs_errs) / n,
        'median': s[n // 2],
        'abs_median': a[n // 2],
        'max': max(abs_errs),
        'min': min(abs_errs),
    }


# =========================================================================== #
# Per-Config Analysis & Report
# =========================================================================== #

def analyze_config(config, base_dir):
    """Run full analysis for one configuration. Returns structured results."""
    cfg_dir = os.path.join(base_dir, config['dir_name'])
    scaling_dir = os.path.join(cfg_dir, 'database_profile')
    distributed_dir = os.path.join(cfg_dir, 'global_ranks_profile')

    pp_size = config['pp']
    results = []

    for rank_id in range(WORLD_SIZE):
        scaling_file = find_rank_file(scaling_dir, rank_id)
        distributed_file = find_rank_file(distributed_dir, rank_id)

        if not scaling_file:
            print(f"  WARNING: Missing scaling file for rank {rank_id}")
            continue
        if not distributed_file:
            print(f"  WARNING: Missing distributed file for rank {rank_id}")
            continue

        s_records = parse_file(scaling_file)
        d_records = parse_file(distributed_file)

        s_data = analyze_scaling_mode(s_records)
        d_data = analyze_distributed_mode(d_records)

        pp_stage = get_pp_stage(rank_id, pp_size)

        results.append({
            'rank_id': rank_id,
            'pp_stage': pp_stage,
            's_fwd': s_data.get('forward_step', 0),
            's_bwd': s_data.get('backward_step', 0),
            's_opt': s_data.get('optimizer_step', 0),
            'd_fwd': d_data['forward_step'],
            'd_bwd': d_data['backward_step'],
            'd_opt': d_data['optimizer_step'],
        })

    return results


def print_config_report(config, results, out):
    """Print detailed report for one configuration."""
    pp_size = config['pp']

    out.write("\n" + "#" * 130 + "\n")
    out.write(f"# {config['name']}  ({config['dir_name']})\n")
    out.write(f"# PP={config['pp']}, TP={config['tp']}, EP={config['ep']}, "
              f"DP={config['dp']}, num_experts={config['num_experts']}, "
              f"num_layers=48, hidden_size=2048, seq_len=2048\n")
    out.write("#" * 130 + "\n")

    # ---- Part 1: Detailed per-rank micro-batch breakdown (distributed) ----
    out.write("\n" + "=" * 130 + "\n")
    out.write("PART 1: Per-Rank Micro-Batch Breakdown (Distributed Mode)\n")
    out.write("=" * 130 + "\n")

    for res in results:
        rank_id = res['rank_id']
        pp_stage = res['pp_stage']
        out.write(f"\n--- Rank {rank_id} (PP stage {pp_stage}/{pp_size-1}) ---\n")

        for op_key, op_label in [('d_fwd', 'forward_step'), ('d_bwd', 'backward_step'), ('d_opt', 'optimizer_step')]:
            d_info = res[op_key]
            out.write(f"  {op_label}: {d_info['count']} micro-batch(es), avg_comp={d_info['avg_comp']:.2f} ms\n")
            if d_info['count'] > 1:
                out.write(f"    {'batch_id':>8} {'mg_state':>10} {'total(ms)':>10} {'comm(ms)':>10} {'comp(ms)':>10} {'#comm':>6}\n")
                for mb in d_info['per_mb']:
                    out.write(f"    {mb['batch_id']:>8} {str(mb['mg_state']):>10} "
                              f"{mb['total']:>10.2f} {mb['comm']:>10.2f} {mb['comp']:>10.2f} {mb['comm_count']:>6}\n")

    # ---- Part 2: Summary comparison table ----
    out.write("\n" + "=" * 130 + "\n")
    out.write("PART 2: Summary Comparison — Scaling Comp vs Distributed Comp (avg across micro-batches)\n")
    out.write("         Distributed comp = total_duration - sum(comm_sub_op_durations)\n")
    out.write("=" * 130 + "\n\n")

    header = (f"{'Rank':>4} {'PP':>3} | "
              f"{'S_fwd':>9} {'D_fwd':>9} {'Δfwd':>9} {'err%':>8} | "
              f"{'S_bwd':>9} {'D_bwd':>9} {'Δbwd':>9} {'err%':>8} | "
              f"{'S_opt':>9} {'D_opt':>9} {'Δopt':>9} {'err%':>8}")
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    fwd_errors, bwd_errors, opt_errors = [], [], []

    for res in results:
        s_fwd = res['s_fwd']
        s_bwd = res['s_bwd']
        s_opt = res['s_opt']
        d_fwd = res['d_fwd']['avg_comp']
        d_bwd = res['d_bwd']['avg_comp']
        d_opt = res['d_opt']['avg_comp']

        def err_pct(s, d):
            return ((s - d) / d * 100) if d != 0 else float('inf')

        ef = err_pct(s_fwd, d_fwd)
        eb = err_pct(s_bwd, d_bwd)
        eo = err_pct(s_opt, d_opt)

        fwd_errors.append(ef)
        bwd_errors.append(eb)
        opt_errors.append(eo)

        out.write(f"{res['rank_id']:>4} {res['pp_stage']:>3} | "
                  f"{s_fwd:>9.2f} {d_fwd:>9.2f} {s_fwd-d_fwd:>9.2f} {ef:>7.2f}% | "
                  f"{s_bwd:>9.2f} {d_bwd:>9.2f} {s_bwd-d_bwd:>9.2f} {eb:>7.2f}% | "
                  f"{s_opt:>9.2f} {d_opt:>9.2f} {s_opt-d_opt:>9.2f} {eo:>7.2f}%\n")

    out.write("-" * len(header) + "\n")

    # Aggregation
    out.write(f"\nAggregation Statistics (relative error %):\n")
    out.write(f"  {'':>18} {'mean':>8} {'|mean|':>8} {'median':>8} {'|median|':>8} {'max|err|':>8} {'min|err|':>8}\n")
    for name, errs in [('forward_step', fwd_errors), ('backward_step', bwd_errors), ('optimizer_step', opt_errors)]:
        st = stats(errs)
        out.write(f"  {name:>18} {st['mean']:>7.2f}% {st['abs_mean']:>7.2f}% "
                  f"{st['median']:>7.2f}% {st['abs_median']:>7.2f}% "
                  f"{st['max']:>7.2f}% {st['min']:>7.2f}%\n")

    # ---- Part 3: Per-PP-Stage breakdown ----
    out.write("\n" + "=" * 130 + "\n")
    out.write("PART 3: Per-PP-Stage Statistics\n")
    out.write("=" * 130 + "\n")

    for stage in range(pp_size):
        stage_indices = [i for i, r in enumerate(results) if r['pp_stage'] == stage]
        if not stage_indices:
            continue
        stage_ranks = [results[i]['rank_id'] for i in stage_indices]
        stage_fwd = [fwd_errors[i] for i in stage_indices]
        stage_bwd = [bwd_errors[i] for i in stage_indices]
        stage_opt = [opt_errors[i] for i in stage_indices]

        out.write(f"\n  PP Stage {stage} (ranks: {stage_ranks}):\n")
        out.write(f"    {'':>18} {'mean':>8} {'|mean|':>8} {'median':>8} {'max|err|':>8}\n")
        for name, errs in [('forward_step', stage_fwd), ('backward_step', stage_bwd), ('optimizer_step', stage_opt)]:
            st = stats(errs)
            out.write(f"    {name:>18} {st['mean']:>7.2f}% {st['abs_mean']:>7.2f}% "
                      f"{st['median']:>7.2f}% {st['max']:>7.2f}%\n")

    # ---- Part 4: Pass/Fail ----
    out.write("\n" + "=" * 130 + "\n")
    out.write(f"PART 4: Pass/Fail Check (threshold = {THRESHOLD_PCT}% relative error)\n")
    out.write("=" * 130 + "\n")

    total_checks = 0
    total_pass = 0
    for res, ef, eb, eo in zip(results, fwd_errors, bwd_errors, opt_errors):
        for op_label, err in [('fwd', ef), ('bwd', eb), ('opt', eo)]:
            total_checks += 1
            if abs(err) <= THRESHOLD_PCT:
                total_pass += 1
            else:
                out.write(f"  FAIL: Rank {res['rank_id']} (PP{res['pp_stage']}) {op_label}: "
                          f"|{err:.2f}%| > {THRESHOLD_PCT}%\n")

    pct = (total_pass / total_checks * 100) if total_checks > 0 else 0
    out.write(f"\n  Result: {total_pass}/{total_checks} checks passed ({pct:.1f}%)\n")
    if total_pass == total_checks:
        out.write("  ✅ ALL CHECKS PASSED\n")
    else:
        out.write(f"  ❌ {total_checks - total_pass} checks FAILED\n")

    return {
        'config_name': config['name'],
        'fwd_errors': fwd_errors,
        'bwd_errors': bwd_errors,
        'opt_errors': opt_errors,
        'total_pass': total_pass,
        'total_checks': total_checks,
    }


# =========================================================================== #
# Cross-Config Summary
# =========================================================================== #

def print_cross_config_summary(all_summaries, out):
    """Print a cross-configuration comparison summary."""
    out.write("\n\n" + "=" * 130 + "\n")
    out.write("CROSS-CONFIGURATION SUMMARY\n")
    out.write("=" * 130 + "\n\n")

    header = (f"{'Configuration':>30} | "
              f"{'|fwd| mean':>10} {'|fwd| med':>10} {'fwd max':>9} | "
              f"{'|bwd| mean':>10} {'|bwd| med':>10} {'bwd max':>9} | "
              f"{'|opt| mean':>10} {'|opt| med':>10} {'opt max':>9} | "
              f"{'Pass':>6}")
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    for s in all_summaries:
        sf = stats(s['fwd_errors'])
        sb = stats(s['bwd_errors'])
        so = stats(s['opt_errors'])
        pass_str = f"{s['total_pass']}/{s['total_checks']}"
        out.write(f"{s['config_name']:>30} | "
                  f"{sf['abs_mean']:>9.2f}% {sf['abs_median']:>9.2f}% {sf['max']:>8.2f}% | "
                  f"{sb['abs_mean']:>9.2f}% {sb['abs_median']:>9.2f}% {sb['max']:>8.2f}% | "
                  f"{so['abs_mean']:>9.2f}% {so['abs_median']:>9.2f}% {so['max']:>8.2f}% | "
                  f"{pass_str:>6}\n")

    out.write("-" * len(header) + "\n")


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../megatron-sim-engine/simulation_inputs/megatron_operation_log/'
        'h800_16gpus_qwen3_moe'
    )

    if not os.path.isdir(base_dir):
        print(f"ERROR: Base directory not found: {base_dir}")
        sys.exit(1)

    # Output to both stdout and file
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../task_memory/compare_qwen3_moe_comp_report.md'
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    class TeeWriter:
        """Write to both stdout and a file."""
        def __init__(self, filepath):
            self.file = open(filepath, 'w')
        def write(self, text):
            sys.stdout.write(text)
            self.file.write(text)
        def flush(self):
            sys.stdout.flush()
            self.file.flush()
        def close(self):
            self.file.close()

    out = TeeWriter(report_path)

    out.write("=" * 130 + "\n")
    out.write("Qwen3 30B-A3B MoE: Scaling Mode vs Distributed Mode Compute (Comp) Comparison\n")
    out.write(f"Model: num_layers=48, hidden_size=2048, num_experts=128, seq_len=2048\n")
    out.write(f"Hardware: H800 16-GPU,  World Size = {WORLD_SIZE}\n")
    out.write(f"Threshold: {THRESHOLD_PCT}% relative error\n")
    out.write("=" * 130 + "\n")

    all_summaries = []

    for config in CONFIGS:
        out.write(f"\n\nAnalyzing {config['name']} ...\n")
        results = analyze_config(config, base_dir)
        summary = print_config_report(config, results, out)
        all_summaries.append(summary)

    print_cross_config_summary(all_summaries, out)

    out.write(f"\nReport saved to: {report_path}\n")
    out.close()


if __name__ == '__main__':
    main()
