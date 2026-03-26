#!/usr/bin/env python3
"""
Thin wrapper to invoke the Python SKALD pipeline for benchmark purposes.

Must be run from the repo root on the master (Python) branch where SKALD.core exists.

Usage:
    python benchmark/python_runner.py \\
        --data    /path/to/data.csv \\
        --workdir /tmp/bench_workdir \\
        --num-qis 2 \\
        --cat-qis 2 \\
        --k       5 \\
        --suppression-limit 0.05

Writes a JSON result object to stdout. All other output goes to stderr.
Exit code 0 on success, 1 on pipeline failure.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

# ── Column metadata — must match generate_data.py ────────────────────────────

NUMERICAL_QI_POOL = [
    {"column": "age",                "encode": False, "type": "int"},
    {"column": "zipcode",            "encode": False, "type": "int"},
    {"column": "income",             "encode": False, "type": "int"},
    {"column": "birth_year",         "encode": False, "type": "int"},
    {"column": "district_code",      "encode": False, "type": "int"},
    {"column": "credit_score",       "encode": False, "type": "int"},
    {"column": "transaction_amount", "encode": False, "type": "int"},
    {"column": "household_size",     "encode": False, "type": "int"},
]

CATEGORICAL_QI_POOL = [
    {"column": "gender"},
    {"column": "blood_group"},
    {"column": "education_level"},
    {"column": "occupation"},
    {"column": "region"},
    {"column": "marital_status"},
    {"column": "insurance_tier"},
    {"column": "language"},
]

# Default size factor (bin width) for each numerical QI
SIZE_FACTORS = {
    "age":                2,
    "zipcode":            100,
    "income":             5000,
    "birth_year":         2,
    "district_code":      10,
    "credit_score":       10,
    "transaction_amount": 1000,
    "household_size":     1,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",              required=True)
    p.add_argument("--workdir",           required=True)
    p.add_argument("--num-qis",           type=int, default=2)
    p.add_argument("--cat-qis",           type=int, default=2)
    p.add_argument("--k",                 type=int, default=5)
    p.add_argument("--suppression-limit", type=float, default=0.05)
    p.add_argument("--chunks",            type=int, default=4,
                   help="number_of_chunks for Python pipeline")
    return p.parse_args()


def build_python_config(data_csv, num_qis, cat_qis, k, suppression_limit, chunks):
    """Build the config dict expected by SKALD_main.main_process()."""
    num_qi_cfgs = NUMERICAL_QI_POOL[:min(num_qis, len(NUMERICAL_QI_POOL))]
    cat_qi_cfgs = CATEGORICAL_QI_POOL[:min(cat_qis, len(CATEGORICAL_QI_POOL))]

    size = {q["column"]: SIZE_FACTORS[q["column"]] for q in num_qi_cfgs}

    return {
        "operations": ["SKALD", "k-anonymity"],
        "data_type": "BenchmarkData",
        "BenchmarkData": {
            "enable_k_anonymity": True,
            "chunk_name": os.path.basename(data_csv),
            "number_of_chunks": chunks,
            "output_path":       "benchmark_output.csv",
            "output_directory":  "output",
            "key_directory":     "keys",
            "log_file":          "log.txt",
            "save_output":       True,
            "suppress":          [],
            "encrypt":           [],
            "quasi_identifiers": {
                "numerical":   num_qi_cfgs,
                "categorical": cat_qi_cfgs,
            },
            "size": size,
            "k_anonymize": {"k": k},
            "suppression_limit": suppression_limit,
        },
    }


def run_pipeline_python(config, data_csv, workdir):
    """
    Call the Python pipeline. Tries multiple import paths to handle
    different branch layouts.
    """
    # Put the data file where the pipeline expects it
    data_dir = os.path.join(workdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, os.path.basename(data_csv))
    if not os.path.exists(dest):
        shutil.copy2(data_csv, dest)

    os.chdir(workdir)

    try:
        from SKALD.core import run_pipeline as _rp  # noqa: F401
        from SKALD.core import run_pipeline
        print("[python_runner] Using SKALD.core.run_pipeline", file=sys.stderr)
        # run_pipeline is the direct entry point on master branch
        return run_pipeline(config)
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import Python pipeline: {e}. "
            "Make sure you are on the master branch and 'pip install -e .' has been run."
        )


def dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (1024 ** 2)


def main():
    args = parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    config = build_python_config(
        args.data,
        args.num_qis,
        args.cat_qis,
        args.k,
        args.suppression_limit,
        args.chunks,
    )

    input_size_mb = os.path.getsize(args.data) / (1024 ** 2)

    t0 = time.monotonic()
    error_msg = None
    pipeline_result = None

    try:
        pipeline_result = run_pipeline_python(config, args.data, args.workdir)
        status = "success"
    except Exception as e:
        status = "error"
        error_msg = str(e)
        traceback.print_exc(file=sys.stderr)

    wall_time_s = time.monotonic() - t0

    output_dir = os.path.join(args.workdir, "output")
    output_size_mb = dir_size_mb(output_dir) if os.path.isdir(output_dir) else 0.0

    result = {
        "pipeline":           "python",
        "status":             status,
        "wall_time_s":        round(wall_time_s, 3),
        "input_size_mb":      round(input_size_mb, 2),
        "output_size_mb":     round(output_size_mb, 2),
        "throughput_mb_per_s": round(input_size_mb / wall_time_s, 2) if wall_time_s > 0 else 0,
        "num_numerical_qis":  min(args.num_qis, len(NUMERICAL_QI_POOL)),
        "num_categorical_qis":min(args.cat_qis, len(CATEGORICAL_QI_POOL)),
        "total_qis":          min(args.num_qis, len(NUMERICAL_QI_POOL)) + min(args.cat_qis, len(CATEGORICAL_QI_POOL)),
        "k":                  args.k,
        "suppression_limit":  args.suppression_limit,
        "error":              error_msg,
        "pipeline_output":    pipeline_result,
    }

    print(json.dumps(result))
    sys.exit(0 if status == "success" else 1)


if __name__ == "__main__":
    main()
