#!/usr/bin/env python3
"""
Streaming synthetic CSV generator for SKALD pipeline benchmarks.

Writes directly to disk row-by-row — never loads the full dataset into RAM.
All values are pseudo-random based on a fixed seed for reproducibility.

Usage:
    python generate_data.py \\
        --rows 50000000 \\
        --num-qis 4 \\
        --cat-qis 4 \\
        --output /tmp/bench_data.csv

Exit code 0 on success. Prints final file size to stdout.
"""

import argparse
import csv
import os
import random
import sys
import time

# ── Column pools ──────────────────────────────────────────────────────────────

NUMERICAL_QI_POOL = [
    ("age",                lambda r: r.randint(18, 90)),
    ("zipcode",            lambda r: r.randint(10000, 99999)),
    ("income",             lambda r: r.randint(10000, 200000)),
    ("birth_year",         lambda r: r.randint(1940, 2006)),
    ("district_code",      lambda r: r.randint(1, 500)),
    ("credit_score",       lambda r: r.randint(300, 850)),
    ("transaction_amount", lambda r: r.randint(100, 100000)),
    ("household_size",     lambda r: r.randint(1, 10)),
]

CATEGORICAL_QI_POOL = [
    ("gender",             ["M", "F", "O"]),
    ("blood_group",        ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
    ("education_level",    ["Primary", "Secondary", "Graduate", "Postgraduate"]),
    ("occupation",         ["Government", "Private", "Self-Employed", "Unemployed", "Student"]),
    ("region",             ["North", "South", "East", "West", "Central"]),
    ("marital_status",     ["Single", "Married", "Divorced", "Widowed"]),
    ("insurance_tier",     ["Basic", "Standard", "Premium"]),
    ("language",           ["English", "Hindi", "Tamil", "Telugu", "Kannada",
                            "Marathi", "Bengali", "Gujarati", "Punjabi", "Odia"]),
]

# Non-QI filler columns to pad file size closer to target
FILLER_COLUMNS = [
    ("record_id",    lambda r: f"REC{r.randint(100000000, 999999999)}"),
    ("name_hash",    lambda r: f"{r.randint(0, 0xFFFFFFFF):08x}{r.randint(0, 0xFFFFFFFF):08x}"),
    ("record_date",  lambda r: f"202{r.randint(0,4)}-{r.randint(1,12):02d}-{r.randint(1,28):02d}"),
    ("status_code",  lambda r: r.choice(["ACTIVE", "INACTIVE", "PENDING", "CLOSED", "SUSPENDED"])),
    ("scheme_id",    lambda r: f"SCH{r.randint(1000, 9999)}"),
    ("facility_code",lambda r: f"FAC{r.randint(100, 999):03d}"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic benchmark CSV")
    p.add_argument("--rows",     type=int,   required=True,  help="Number of data rows to generate")
    p.add_argument("--num-qis",  type=int,   default=2,      help="Number of numerical QI columns (max 8)")
    p.add_argument("--cat-qis",  type=int,   default=2,      help="Number of categorical QI columns (max 8)")
    p.add_argument("--output",   type=str,   required=True,  help="Output CSV file path")
    p.add_argument("--seed",     type=int,   default=42,     help="Random seed for reproducibility")
    p.add_argument("--filler",   type=int,   default=3,      help="Number of non-QI filler columns (0-6)")
    p.add_argument("--progress", action="store_true",        help="Print progress every 1M rows")
    return p.parse_args()


def main():
    args = parse_args()

    num_qis = min(args.num_qis, len(NUMERICAL_QI_POOL))
    cat_qis = min(args.cat_qis, len(CATEGORICAL_QI_POOL))
    filler  = min(args.filler,  len(FILLER_COLUMNS))

    if num_qis == 0 and cat_qis == 0:
        print("ERROR: need at least one QI column", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    num_qi_cols  = NUMERICAL_QI_POOL[:num_qis]
    cat_qi_cols  = CATEGORICAL_QI_POOL[:cat_qis]
    filler_cols  = FILLER_COLUMNS[:filler]

    headers = (
        [name for name, _ in num_qi_cols]
        + [name for name, _ in cat_qi_cols]
        + [name for name, _ in filler_cols]
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    start = time.monotonic()
    rows_written = 0
    report_interval = 1_000_000

    with open(args.output, "w", newline="", buffering=1 << 20) as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for _ in range(args.rows):
            row = (
                [gen(rng)  for _, gen    in num_qi_cols]
                + [rng.choice(vals)       for _, vals   in cat_qi_cols]
                + [gen(rng) for _, gen   in filler_cols]
            )
            writer.writerow(row)
            rows_written += 1

            if args.progress and rows_written % report_interval == 0:
                elapsed = time.monotonic() - start
                rate    = rows_written / elapsed
                size_mb = os.path.getsize(args.output) / (1024 ** 2)
                print(
                    f"  {rows_written:>12,} rows  |  {size_mb:>8,.1f} MB  |  {rate:>9,.0f} rows/s",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed  = time.monotonic() - start
    size_mb  = os.path.getsize(args.output) / (1024 ** 2)
    size_gb  = size_mb / 1024

    print(
        f"Generated {rows_written:,} rows  "
        f"| {size_gb:.3f} GB ({size_mb:.1f} MB)  "
        f"| {rows_written / elapsed:,.0f} rows/s  "
        f"| {elapsed:.1f}s"
    )
    print(f"OUTPUT_PATH={args.output}")
    print(f"OUTPUT_SIZE_MB={size_mb:.2f}")
    print(f"OUTPUT_ROWS={rows_written}")
    print(f"NUM_NUMERICAL_QIS={num_qis}")
    print(f"NUM_CATEGORICAL_QIS={cat_qis}")
    print(f"COLUMNS={','.join(headers)}")


if __name__ == "__main__":
    main()
