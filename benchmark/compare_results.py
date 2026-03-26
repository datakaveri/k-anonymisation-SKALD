#!/usr/bin/env python3
"""
Compare benchmark results from Rust and Python SKALD pipeline runs.

Usage:
    python benchmark/compare_results.py results_rust.json results_python.json

    # Optional: filter and export
    python benchmark/compare_results.py results_rust.json results_python.json \\
        --format csv \\
        --output comparison.csv

Produces:
  1. Per-run comparison table (wall time, RAM, throughput, speedup)
  2. Summary statistics per (total_qis, k) combination
  3. Speedup heatmap (text-based)
"""

import argparse
import json
import sys
import os
from collections import defaultdict


def load(path):
    with open(path) as f:
        data = json.load(f)
    # Accept both wrapped {"results": [...]} and bare [...]
    if isinstance(data, list):
        return data
    return data.get("results", [])


def fmt_mb(mb):
    if mb is None:
        return "N/A"
    if mb >= 1024:
        return f"{mb/1024:.1f} GB"
    return f"{mb:.0f} MB"


def fmt_s(s):
    if s is None:
        return "N/A"
    if s >= 3600:
        return f"{s/3600:.1f}h"
    if s >= 60:
        return f"{s/60:.1f}m"
    return f"{s:.1f}s"


def fmt_rows(r):
    if r >= 1_000_000_000:
        return f"{r/1e9:.1f}B"
    if r >= 1_000_000:
        return f"{r/1e6:.0f}M"
    if r >= 1_000:
        return f"{r/1e3:.0f}K"
    return str(r)


def record_key(r):
    return (r["rows"], r["total_qis"], r["k"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("rust_file",   help="JSON results file from Rust pipeline run")
    p.add_argument("python_file", help="JSON results file from Python pipeline run")
    p.add_argument("--format",    choices=["table", "csv"], default="table")
    p.add_argument("--output",    help="Write output to file instead of stdout")
    p.add_argument("--sort-by",   choices=["rows", "qis", "k", "speedup"], default="rows")
    return p.parse_args()


def build_comparison(rust_records, python_records):
    rust_map   = {record_key(r): r for r in rust_records}
    python_map = {record_key(r): r for r in python_records}

    all_keys = sorted(set(rust_map) | set(python_map))
    rows = []

    for key in all_keys:
        n_rows, total_qis, k = key
        r = rust_map.get(key)
        p = python_map.get(key)

        r_time  = r["wall_time_s"]  if r and r["status"] == "success" else None
        p_time  = p["wall_time_s"]  if p and p["status"] == "success" else None
        r_rss   = r["peak_rss_mb"]  if r else None
        p_rss   = p["peak_rss_mb"]  if p else None
        r_tput  = r["throughput_mb_per_s"] if r and r["status"] == "success" else None
        p_tput  = p["throughput_mb_per_s"] if p and p["status"] == "success" else None
        r_in    = r["input_size_mb"] if r else None

        speedup = None
        if r_time and p_time and r_time > 0:
            speedup = p_time / r_time

        rss_ratio = None
        if r_rss and p_rss and r_rss > 0:
            rss_ratio = p_rss / r_rss

        rows.append({
            "rows":          n_rows,
            "total_qis":     total_qis,
            "k":             k,
            "input_mb":      r_in,
            "r_status":      r["status"] if r else "missing",
            "p_status":      p["status"] if p else "missing",
            "r_time_s":      r_time,
            "p_time_s":      p_time,
            "r_rss_mb":      r_rss,
            "p_rss_mb":      p_rss,
            "r_tput":        r_tput,
            "p_tput":        p_tput,
            "time_speedup":  speedup,
            "rss_ratio":     rss_ratio,
            "r_error":       r.get("error_msg", "") if r else "",
            "p_error":       p.get("error_msg", "") if p else "",
        })

    return rows


def print_table(rows, out):
    COL_W = {
        "Dataset":    12,
        "QIs":         4,
        "k":           5,
        "Size":        8,
        "R Time":      9,
        "P Time":      9,
        "Time x":      7,
        "R RAM":       8,
        "P RAM":       8,
        "RAM x":       6,
        "R MB/s":      8,
        "P MB/s":      8,
    }

    def header():
        return (
            f"{'Dataset':<{COL_W['Dataset']}} "
            f"{'QIs':>{COL_W['QIs']}} "
            f"{'k':>{COL_W['k']}} "
            f"{'Size':>{COL_W['Size']}} "
            f"{'R Time':>{COL_W['R Time']}} "
            f"{'P Time':>{COL_W['P Time']}} "
            f"{'Time x':>{COL_W['Time x']}} "
            f"{'R RAM':>{COL_W['R RAM']}} "
            f"{'P RAM':>{COL_W['P RAM']}} "
            f"{'RAM x':>{COL_W['RAM x']}} "
            f"{'R MB/s':>{COL_W['R MB/s']}} "
            f"{'P MB/s':>{COL_W['P MB/s']}}"
        )

    sep = "-" * len(header())

    print("\n" + "=" * len(header()), file=out)
    print("SKALD Rust vs Python — Per-Run Comparison", file=out)
    print("=" * len(header()), file=out)
    print(header(), file=out)
    print(sep, file=out)

    for row in rows:
        dataset   = fmt_rows(row["rows"])
        qis       = str(row["total_qis"])
        k         = str(row["k"])
        size      = fmt_mb(row["input_mb"])
        r_time    = fmt_s(row["r_time_s"])   if row["r_status"] == "success" else f"ERR"
        p_time    = fmt_s(row["p_time_s"])   if row["p_status"] == "success" else f"ERR"
        r_rss     = fmt_mb(row["r_rss_mb"])
        p_rss     = fmt_mb(row["p_rss_mb"])
        r_tput    = f"{row['r_tput']:.0f}"   if row["r_tput"]  else "N/A"
        p_tput    = f"{row['p_tput']:.0f}"   if row["p_tput"]  else "N/A"

        speedup_s = f"{row['time_speedup']:.1f}x" if row["time_speedup"] else "N/A"
        rss_s     = f"{row['rss_ratio']:.1f}x"    if row["rss_ratio"]    else "N/A"

        print(
            f"{dataset:<{COL_W['Dataset']}} "
            f"{qis:>{COL_W['QIs']}} "
            f"{k:>{COL_W['k']}} "
            f"{size:>{COL_W['Size']}} "
            f"{r_time:>{COL_W['R Time']}} "
            f"{p_time:>{COL_W['P Time']}} "
            f"{speedup_s:>{COL_W['Time x']}} "
            f"{r_rss:>{COL_W['R RAM']}} "
            f"{p_rss:>{COL_W['P RAM']}} "
            f"{rss_s:>{COL_W['RAM x']}} "
            f"{r_tput:>{COL_W['R MB/s']}} "
            f"{p_tput:>{COL_W['P MB/s']}}",
            file=out,
        )

        if row["r_error"]:
            print(f"  [RUST ERROR]   {row['r_error'][:100]}", file=out)
        if row["p_error"]:
            print(f"  [PYTHON ERROR] {row['p_error'][:100]}", file=out)

    print(sep, file=out)

    # Legend
    print(
        "\nColumns: R=Rust  P=Python  Time x = Python/Rust time speedup  "
        "RAM x = Python/Rust RAM ratio (higher = Rust uses less RAM)\n",
        file=out,
    )


def print_summary(rows, out):
    """Aggregate by (total_qis, k) across all dataset sizes."""
    groups = defaultdict(list)
    for row in rows:
        if row["r_status"] == "success" and row["p_status"] == "success":
            groups[(row["total_qis"], row["k"])].append(row)

    print("=" * 60, file=out)
    print("SUMMARY — Average speedup by (QI count, k)", file=out)
    print("=" * 60, file=out)
    print(f"{'QIs':>4}  {'k':>5}  {'Avg Time x':>10}  {'Avg RAM x':>9}  {'Runs':>4}", file=out)
    print("-" * 40, file=out)

    for (qis, k) in sorted(groups):
        group = groups[(qis, k)]
        speedups  = [r["time_speedup"] for r in group if r["time_speedup"]]
        rss_ratios= [r["rss_ratio"]    for r in group if r["rss_ratio"]]
        avg_sp  = sum(speedups)   / len(speedups)   if speedups   else None
        avg_rss = sum(rss_ratios) / len(rss_ratios) if rss_ratios else None
        print(
            f"{qis:>4}  {k:>5}  "
            f"{f'{avg_sp:.1f}x':>10}  "
            f"{f'{avg_rss:.1f}x':>9}  "
            f"{len(group):>4}",
            file=out,
        )

    print("", file=out)


def print_heatmap(rows, out):
    """Text heatmap: rows = QI count, columns = dataset size, cell = speedup."""
    successful = [r for r in rows if r["time_speedup"] is not None]
    if not successful:
        print("No successful paired runs to produce heatmap.", file=out)
        return

    qi_vals   = sorted(set(r["total_qis"] for r in successful))
    size_vals = sorted(set(r["rows"]      for r in successful))

    speedup_map = {}
    for r in successful:
        key = (r["total_qis"], r["rows"])
        # Average over k values for the heatmap cell
        if key not in speedup_map:
            speedup_map[key] = []
        speedup_map[key].append(r["time_speedup"])

    avg_map = {k: sum(v) / len(v) for k, v in speedup_map.items()}

    col_w = 10
    header_row = f"{'QIs\\Rows':<8} " + "".join(f"{fmt_rows(s):>{col_w}}" for s in size_vals)

    print("=" * len(header_row), file=out)
    print("TIME SPEEDUP HEATMAP  (Rust vs Python, higher = Rust faster)", file=out)
    print("=" * len(header_row), file=out)
    print(header_row, file=out)
    print("-" * len(header_row), file=out)

    for qi in qi_vals:
        cells = []
        for sz in size_vals:
            sp = avg_map.get((qi, sz))
            if sp is None:
                cells.append(f"{'N/A':>{col_w}}")
            else:
                cells.append(f"{f'{sp:.1f}x':>{col_w}}")
        print(f"{qi:<8} " + "".join(cells), file=out)

    print("", file=out)


def write_csv(rows, out):
    import csv as csv_mod
    writer = csv_mod.writer(out)
    writer.writerow([
        "rows", "total_qis", "k", "input_mb",
        "rust_status", "python_status",
        "rust_wall_s", "python_wall_s", "time_speedup",
        "rust_rss_mb", "python_rss_mb", "rss_ratio",
        "rust_tput_mb_s", "python_tput_mb_s",
        "rust_error", "python_error",
    ])
    for r in rows:
        writer.writerow([
            r["rows"], r["total_qis"], r["k"], r.get("input_mb", ""),
            r["r_status"], r["p_status"],
            r.get("r_time_s", ""), r.get("p_time_s", ""),
            f"{r['time_speedup']:.2f}" if r["time_speedup"] else "",
            r.get("r_rss_mb", ""), r.get("p_rss_mb", ""),
            f"{r['rss_ratio']:.2f}" if r["rss_ratio"] else "",
            r.get("r_tput", ""), r.get("p_tput", ""),
            r.get("r_error", ""), r.get("p_error", ""),
        ])


def main():
    args = parse_args()

    if not os.path.exists(args.rust_file):
        print(f"ERROR: {args.rust_file} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.python_file):
        print(f"ERROR: {args.python_file} not found", file=sys.stderr)
        sys.exit(1)

    rust_records   = load(args.rust_file)
    python_records = load(args.python_file)

    print(f"Rust   results: {len(rust_records)} runs from {args.rust_file}", file=sys.stderr)
    print(f"Python results: {len(python_records)} runs from {args.python_file}", file=sys.stderr)

    comparison = build_comparison(rust_records, python_records)

    sort_key = {
        "rows":    lambda r: (r["rows"], r["total_qis"], r["k"]),
        "qis":     lambda r: (r["total_qis"], r["rows"], r["k"]),
        "k":       lambda r: (r["k"], r["rows"], r["total_qis"]),
        "speedup": lambda r: (-(r["time_speedup"] or 0)),
    }[args.sort_by]
    comparison.sort(key=sort_key)

    out_stream = open(args.output, "w") if args.output else sys.stdout

    try:
        if args.format == "csv":
            write_csv(comparison, out_stream)
        else:
            print_table(comparison, out_stream)
            print_summary(comparison, out_stream)
            print_heatmap(comparison, out_stream)
    finally:
        if args.output:
            out_stream.close()
            print(f"Comparison written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
