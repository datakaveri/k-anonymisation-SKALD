#!/usr/bin/env bash
# =============================================================================
# SKALD Pipeline Benchmark Script
# =============================================================================
#
# Auto-detects whether the Rust or Python pipeline is available and runs a
# full benchmark matrix across dataset sizes, QI counts, and k values.
#
# HOW TO USE — two-branch workflow:
#
#   1. On the Rust branch (this branch):
#        cd /path/to/k-anonymisation-SKALD
#        bash benchmark/run_benchmark.sh --output benchmark/results_rust.json
#
#   2. On master (Python) branch (in a VM / separate checkout):
#        cd /path/to/k-anonymisation-SKALD
#        bash benchmark/run_benchmark.sh --output benchmark/results_python.json
#
#   3. Compare:
#        python benchmark/compare_results.py \
#            benchmark/results_rust.json \
#            benchmark/results_python.json
#
# OPTIONS (all optional — defaults shown):
#   --output        FILE    Where to write the JSON results (default: benchmark_results_<pipeline>_<ts>.json)
#   --sizes         LIST    Space-separated row counts  (default: "1000000 10000000 50000000 100000000")
#                           Approx sizes: 1M≈200MB  10M≈2GB  50M≈10GB  100M≈20GB
#                           For 50GB use 250000000, for 100GB use 500000000
#   --qi-configs    LIST    Space-separated total QI counts to test  (default: "2 4 6 8")
#                           QIs are split evenly: half numerical, half categorical
#   --k-values      LIST    Space-separated k values (default: "5 15 50")
#   --workdir       DIR     Scratch directory for pipeline runs (default: /tmp/skald_bench)
#   --data-dir      DIR     Where to store generated CSVs (default: /tmp/skald_bench_data)
#   --keep-data             Do NOT delete generated CSVs after use (useful to reuse across runs)
#   --skip-generate         Reuse CSVs already in --data-dir (implies --keep-data)
#   --rust-binary   PATH    Override path to skald_pipeline binary
#   --python-bin    PATH    Override python3 executable
#   --dry-run               Print what would run without executing
#
# METRICS RECORDED per run:
#   wall_time_s        Total elapsed seconds (from pipeline start to finish)
#   peak_rss_mb        Peak resident RAM in MB (via /usr/bin/time -v or /proc polling)
#   cpu_user_s         User-space CPU seconds
#   cpu_sys_s          Kernel CPU seconds
#   input_size_mb      Size of the input CSV in MB
#   output_size_mb     Size of the output directory in MB
#   throughput_mb_s    input_size_mb / wall_time_s
#   rows               Number of rows in the dataset
#   num_qis            Total quasi-identifier count
#   k                  k-anonymity parameter
#   status             "success" or "error"
#   error_msg          Error detail if status=error
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SIZES="1000000 10000000 50000000 100000000"
QI_CONFIGS="2 4 6 8"
K_VALUES="5 15 50"
WORKDIR="/tmp/skald_bench"
DATA_DIR="/tmp/skald_bench_data"
KEEP_DATA=false
SKIP_GENERATE=false
DRY_RUN=false
RUST_BINARY=""
PYTHON_BIN="python3"
OUTPUT_FILE=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes)          SIZES="$2";          shift 2 ;;
        --qi-configs)     QI_CONFIGS="$2";     shift 2 ;;
        --k-values)       K_VALUES="$2";       shift 2 ;;
        --workdir)        WORKDIR="$2";        shift 2 ;;
        --data-dir)       DATA_DIR="$2";       shift 2 ;;
        --output)         OUTPUT_FILE="$2";    shift 2 ;;
        --rust-binary)    RUST_BINARY="$2";    shift 2 ;;
        --python-bin)     PYTHON_BIN="$2";     shift 2 ;;
        --keep-data)      KEEP_DATA=true;      shift   ;;
        --skip-generate)  SKIP_GENERATE=true; KEEP_DATA=true; shift ;;
        --dry-run)        DRY_RUN=true;        shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmark"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── Detect pipeline type ──────────────────────────────────────────────────────
# Priority: explicit --rust-binary flag > Cargo.toml present (Rust branch) >
#           pre-built binary > Python importable > unknown.
# SKALD_main.py exists on BOTH branches so is NOT used as a Rust/Python signal.
detect_pipeline() {
    # 1. Explicit binary override — always Rust
    if [[ -n "$RUST_BINARY" && -x "$RUST_BINARY" ]]; then
        echo "rust"; return
    fi

    # 2. Cargo.toml present → this is the Rust branch; build if needed
    local cargo_toml="$REPO_ROOT/SKALD/Cargo.toml"
    if [[ -f "$cargo_toml" ]]; then
        local rel_bin="$REPO_ROOT/SKALD/target/release/skald_pipeline"
        local dbg_bin="$REPO_ROOT/SKALD/target/debug/skald_pipeline"
        if [[ -x "$rel_bin" ]]; then
            RUST_BINARY="$rel_bin"
        elif [[ -x "$dbg_bin" ]]; then
            RUST_BINARY="$dbg_bin"
        else
            # Binary not yet built — build it now
            echo "[bench] SKALD/Cargo.toml found. Building Rust pipeline (release)..." >&2
            if ! $DRY_RUN; then
                (cd "$REPO_ROOT/SKALD" && cargo build --release --quiet) >&2
            fi
            RUST_BINARY="$rel_bin"
        fi
        echo "rust"; return
    fi

    # 3. Python pipeline importable (master branch with SKALD installed)
    if "$PYTHON_BIN" -c "from SKALD.core import run_pipeline" 2>/dev/null; then
        echo "python"; return
    fi

    echo "unknown"
}

export REPO_ROOT
PIPELINE_TYPE="$(detect_pipeline)"

if [[ "$PIPELINE_TYPE" == "unknown" ]]; then
    echo "ERROR: Cannot detect pipeline type." >&2
    echo "  Rust branch:   SKALD/Cargo.toml must exist (or use --rust-binary)" >&2
    echo "  Python branch: run 'pip install -e .' so 'from SKALD.core import run_pipeline' works" >&2
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="$BENCH_DIR/results_${PIPELINE_TYPE}_${TIMESTAMP}.json"
fi

# ── Ensure Rust binary is set (may have been built inside detect_pipeline) ────
if [[ "$PIPELINE_TYPE" == "rust" && -z "$RUST_BINARY" ]]; then
    RUST_BINARY="$REPO_ROOT/SKALD/target/release/skald_pipeline"
fi
if [[ "$PIPELINE_TYPE" == "rust" ]]; then
    echo "[bench] Rust binary: $RUST_BINARY"
fi

echo "[bench] Pipeline:   $PIPELINE_TYPE"
echo "[bench] Output:     $OUTPUT_FILE"
echo "[bench] Sizes:      $SIZES"
echo "[bench] QI configs: $QI_CONFIGS"
echo "[bench] k values:   $K_VALUES"
echo ""

mkdir -p "$WORKDIR" "$DATA_DIR"

# ── Memory measurement helpers ────────────────────────────────────────────────
# Returns peak RSS in MB by wrapping command with /usr/bin/time -v (GNU time)
GNU_TIME=""
if /usr/bin/time --version 2>&1 | grep -q GNU; then
    GNU_TIME="/usr/bin/time"
elif command -v gtime &>/dev/null && gtime --version 2>&1 | grep -q GNU; then
    GNU_TIME="gtime"
fi

# Background RSS poller: polls /proc/<pid>/status every 0.5s, writes peak to file
poll_rss() {
    local pid="$1" outfile="$2"
    local peak=0
    while kill -0 "$pid" 2>/dev/null; do
        local rss
        rss=$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
        (( rss > peak )) && peak=$rss
        sleep 0.5
    done
    echo "$peak" > "$outfile"
}

# ── Config generators ─────────────────────────────────────────────────────────

# Rust pipeline config (config/config.json format)
build_rust_config() {
    local num_qis="$1" cat_qis="$2" k="$3" sup_limit="$4"

    local num_qi_json cat_qi_json size_json
    num_qi_json="$(python3 - "$num_qis" <<'PYEOF'
import sys, json
POOL = [
    {"column":"age",                "encode":False,"type":"int"},
    {"column":"zipcode",            "encode":False,"type":"int"},
    {"column":"income",             "encode":False,"type":"int"},
    {"column":"birth_year",         "encode":False,"type":"int"},
    {"column":"district_code",      "encode":False,"type":"int"},
    {"column":"credit_score",       "encode":False,"type":"int"},
    {"column":"transaction_amount", "encode":False,"type":"int"},
    {"column":"household_size",     "encode":False,"type":"int"},
]
n = int(sys.argv[1])
print(json.dumps(POOL[:n]))
PYEOF
)"

    cat_qi_json="$(python3 - "$cat_qis" <<'PYEOF'
import sys, json
POOL = [
    {"column":"gender"},
    {"column":"blood_group"},
    {"column":"education_level"},
    {"column":"occupation"},
    {"column":"region"},
    {"column":"marital_status"},
    {"column":"insurance_tier"},
    {"column":"language"},
]
n = int(sys.argv[1])
print(json.dumps(POOL[:n]))
PYEOF
)"

    size_json="$(python3 - "$num_qis" <<'PYEOF'
import sys, json
SIZES = {"age":2,"zipcode":100,"income":5000,"birth_year":2,"district_code":10,
         "credit_score":10,"transaction_amount":1000,"household_size":1}
POOL  = ["age","zipcode","income","birth_year","district_code",
         "credit_score","transaction_amount","household_size"]
n = int(sys.argv[1])
print(json.dumps({c: SIZES[c] for c in POOL[:n]}))
PYEOF
)"

    python3 - "$k" "$sup_limit" "$num_qi_json" "$cat_qi_json" "$size_json" <<'PYEOF'
import sys, json
k, sup, num_j, cat_j, size_j = sys.argv[1:]
print(json.dumps({
    "operations": ["SKALD","k-anonymity"],
    "data_type": "benchmark",
    "benchmark": {
        "output_path":        "benchmark_output.csv",
        "output_directory":   "output",
        "log_file":           "log.txt",
        "suppress":           [],
        "hashing_with_salt":  [],
        "hashing_without_salt":[],
        "masking":            [],
        "encrypt":            [],
        "charcloak":          [],
        "tokenization":       [],
        "fpe":                [],
        "quasi_identifiers": {
            "numerical":   json.loads(num_j),
            "categorical": json.loads(cat_j),
        },
        "size":           json.loads(size_j),
        "k_anonymize":    {"k": int(k)},
        "suppression_limit": float(sup),
        "enable_l_diversity": False,
    }
}, indent=2))
PYEOF
}

# ── Dir size helper ───────────────────────────────────────────────────────────
dir_size_mb() {
    du -sm "$1" 2>/dev/null | awk '{print $1}'
}

# ── Single run: Rust ──────────────────────────────────────────────────────────
run_rust() {
    local data_csv="$1" num_qis="$2" cat_qis="$3" k="$4" rows="$5"
    local run_id="rust_nqi${num_qis}_cqi${cat_qis}_k${k}_r${rows}"
    local run_dir="$WORKDIR/$run_id"
    local sup_limit=0.05

    rm -rf "$run_dir"
    mkdir -p "$run_dir/config" "$run_dir/data" "$run_dir/chunks" "$run_dir/output"

    # Hard link avoids doubling disk usage (same inode, pipeline only reads the file)
    ln "$data_csv" "$run_dir/data/benchmark.csv" 2>/dev/null \
        || cp "$data_csv" "$run_dir/data/benchmark.csv"
    build_rust_config "$num_qis" "$cat_qis" "$k" "$sup_limit" \
        > "$run_dir/config/config.json"

    local input_size_mb
    input_size_mb=$(python3 -c "import os; print(round(os.path.getsize('$data_csv')/(1024**2),2))")

    local time_file="$run_dir/time_output.txt"
    local rss_file="$run_dir/peak_rss_kb.txt"
    local exit_code=0
    local wall_time_s=0
    local peak_rss_mb=0
    local cpu_user_s=0
    local cpu_sys_s=0
    local error_msg=""

    echo "[bench] RUN  $run_id" >&2

    if $DRY_RUN; then
        echo "  DRY RUN: would execute $RUST_BINARY in $run_dir" >&2
        echo "  Config:  $run_dir/config/config.json" >&2
        return
    fi

    local t_start t_end
    t_start=$(date +%s%3N)

    if [[ -n "$GNU_TIME" ]]; then
        # Use GNU time for accurate max RSS
        (cd "$run_dir" && \
            "$GNU_TIME" -v "$RUST_BINARY" >/dev/null 2>"$time_file") \
            && exit_code=0 || exit_code=$?

        if [[ -f "$time_file" ]]; then
            wall_time_s=$(grep "Elapsed (wall clock)" "$time_file" | \
                awk '{print $NF}' | \
                python3 -c "
import sys, re
t = sys.stdin.read().strip()
m = re.match(r'(?:(\d+):)?(\d+):(\d+\.\d+)', t)
if m:
    h = int(m.group(1) or 0)
    mi = int(m.group(2))
    s  = float(m.group(3))
    print(round(h*3600 + mi*60 + s, 3))
else:
    print(t)
" 2>/dev/null || echo "0")
            peak_rss_mb=$(grep "Maximum resident set size" "$time_file" | \
                awk '{print $NF}' | \
                python3 -c "import sys; v=sys.stdin.read().strip(); print(round(int(v)/1024,1))" 2>/dev/null || echo "0")
            cpu_user_s=$(grep "User time" "$time_file" | awk '{print $NF}' || echo "0")
            cpu_sys_s=$(grep "System time" "$time_file" | awk '{print $NF}' || echo "0")
        fi
    else
        # Fallback: background RSS poller + wall clock
        echo "0" > "$rss_file"
        (cd "$run_dir" && "$RUST_BINARY" >/dev/null) &
        local pid=$!
        poll_rss "$pid" "$rss_file" &
        local poller_pid=$!
        wait "$pid" && exit_code=0 || exit_code=$?
        wait "$poller_pid" 2>/dev/null || true

        t_end=$(date +%s%3N)
        wall_time_s=$(python3 -c "print(round(($t_end - $t_start)/1000, 3))")
        peak_rss_mb=$(python3 -c \
            "import os; v=int(open('$rss_file').read().strip() or '0'); print(round(v/1024,1))")
    fi

    if [[ $exit_code -ne 0 ]]; then
        error_msg="skald_pipeline exited with code $exit_code"
        if [[ -f "$run_dir/output/pipeline.log" ]]; then
            error_msg="$error_msg | last log: $(tail -3 "$run_dir/output/pipeline.log" | tr '\n' ' ')"
        fi
    fi

    local output_size_mb
    output_size_mb=$(dir_size_mb "$run_dir/output")

    local throughput
    throughput=$(python3 -c "
mb=$input_size_mb; t=$wall_time_s
print(round(mb/t, 2) if t > 0 else 0)")

    python3 - <<PYEOF
import json, datetime
rec = {
    "timestamp":           "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "pipeline":            "rust",
    "status":              "$([ $exit_code -eq 0 ] && echo success || echo error)",
    "rows":                $rows,
    "num_numerical_qis":   $num_qis,
    "num_categorical_qis": $cat_qis,
    "total_qis":           $((num_qis + cat_qis)),
    "k":                   $k,
    "suppression_limit":   $sup_limit,
    "wall_time_s":         $wall_time_s,
    "peak_rss_mb":         $peak_rss_mb,
    "cpu_user_s":          "${cpu_user_s:-0}",
    "cpu_sys_s":           "${cpu_sys_s:-0}",
    "input_size_mb":       $input_size_mb,
    "output_size_mb":      ${output_size_mb:-0},
    "throughput_mb_per_s": $throughput,
    "error_msg":           "$error_msg",
}
print(json.dumps(rec))
PYEOF

    rm -rf "$run_dir"
}

# ── Single run: Python ────────────────────────────────────────────────────────
run_python() {
    local data_csv="$1" num_qis="$2" cat_qis="$3" k="$4" rows="$5"
    local run_id="python_nqi${num_qis}_cqi${cat_qis}_k${k}_r${rows}"
    local run_dir="$WORKDIR/$run_id"
    local sup_limit=0.05

    rm -rf "$run_dir"
    mkdir -p "$run_dir"

    local rss_file="$run_dir/peak_rss_kb.txt"
    local time_file="$run_dir/time_output.txt"
    local result_file="$run_dir/runner_output.json"
    local exit_code=0

    echo "[bench] RUN  $run_id" >&2

    if $DRY_RUN; then
        echo "  DRY RUN: would run python_runner.py for $data_csv" >&2
        return
    fi

    local runner_cmd=(
        "$PYTHON_BIN" "$BENCH_DIR/python_runner.py"
        "--data"              "$data_csv"
        "--workdir"           "$run_dir/pipeline_wd"
        "--num-qis"           "$num_qis"
        "--cat-qis"           "$cat_qis"
        "--k"                 "$k"
        "--suppression-limit" "$sup_limit"
    )

    if [[ -n "$GNU_TIME" ]]; then
        (cd "$REPO_ROOT" && \
            "$GNU_TIME" -v "${runner_cmd[@]}" >"$result_file" 2>"$time_file") \
            && exit_code=0 || exit_code=$?

        local wall_time_s peak_rss_mb cpu_user_s cpu_sys_s
        wall_time_s=$(grep "Elapsed (wall clock)" "$time_file" | \
            awk '{print $NF}' | \
            python3 -c "
import sys, re
t = sys.stdin.read().strip()
m = re.match(r'(?:(\d+):)?(\d+):(\d+\.\d+)', t)
if m:
    h = int(m.group(1) or 0)
    mi= int(m.group(2))
    s = float(m.group(3))
    print(round(h*3600 + mi*60 + s, 3))
else:
    print(0)
" 2>/dev/null || echo "0")
        peak_rss_mb=$(grep "Maximum resident set size" "$time_file" | \
            awk '{print $NF}' | \
            python3 -c "import sys; v=sys.stdin.read().strip(); print(round(int(v)/1024,1))" 2>/dev/null || echo "0")
        cpu_user_s=$(grep "User time" "$time_file" | awk '{print $NF}' || echo "0")
        cpu_sys_s=$(grep "System time" "$time_file"  | awk '{print $NF}' || echo "0")
    else
        echo "0" > "$rss_file"
        local t_start t_end
        t_start=$(date +%s%3N)
        (cd "$REPO_ROOT" && "${runner_cmd[@]}" >"$result_file") &
        local pid=$!
        poll_rss "$pid" "$rss_file" &
        local poller_pid=$!
        wait "$pid" && exit_code=0 || exit_code=$?
        wait "$poller_pid" 2>/dev/null || true
        t_end=$(date +%s%3N)
        wall_time_s=$(python3 -c "print(round(($t_end - $t_start)/1000, 3))")
        peak_rss_mb=$(python3 -c \
            "v=int(open('$rss_file').read().strip() or '0'); print(round(v/1024,1))")
        cpu_user_s="0"; cpu_sys_s="0"
    fi

    # Parse runner output for input/output sizes (runner also writes them)
    local input_size_mb output_size_mb throughput runner_status runner_error
    if [[ -f "$result_file" ]]; then
        input_size_mb=$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('input_size_mb',0))" 2>/dev/null || echo "0")
        output_size_mb=$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('output_size_mb',0))" 2>/dev/null || echo "0")
        runner_status=$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('status','error'))" 2>/dev/null || echo "error")
        runner_error=$(python3 -c "import json; d=json.load(open('$result_file')); print(d.get('error') or '')" 2>/dev/null || echo "")
    else
        input_size_mb=$(python3 -c "import os; print(round(os.path.getsize('$data_csv')/(1024**2),2))")
        output_size_mb=0
        runner_status="error"
        runner_error="runner produced no output (exit $exit_code)"
    fi

    throughput=$(python3 -c "
mb=$input_size_mb; t=$wall_time_s
print(round(mb/t, 2) if t > 0 else 0)")

    python3 - <<PYEOF
import json
rec = {
    "timestamp":           "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "pipeline":            "python",
    "status":              "$runner_status",
    "rows":                $rows,
    "num_numerical_qis":   $num_qis,
    "num_categorical_qis": $cat_qis,
    "total_qis":           $((num_qis + cat_qis)),
    "k":                   $k,
    "suppression_limit":   $sup_limit,
    "wall_time_s":         $wall_time_s,
    "peak_rss_mb":         $peak_rss_mb,
    "cpu_user_s":          "${cpu_user_s:-0}",
    "cpu_sys_s":           "${cpu_sys_s:-0}",
    "input_size_mb":       $input_size_mb,
    "output_size_mb":      ${output_size_mb:-0},
    "throughput_mb_per_s": $throughput,
    "error_msg":           "$runner_error",
}
print(json.dumps(rec))
PYEOF

    rm -rf "$run_dir"
}

# ── Main benchmark loop ───────────────────────────────────────────────────────

RESULTS=()
TOTAL_RUNS=0

# Pre-count total runs for progress display
for _s in $SIZES; do
    for _q in $QI_CONFIGS; do
        for _k in $K_VALUES; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
    done
done
echo "[bench] Total combinations: $TOTAL_RUNS"
echo ""

RUN_NUM=0

for ROWS in $SIZES; do
    for QI_TOTAL in $QI_CONFIGS; do
        # Split QIs: half numerical, half categorical (favour numerical for odd)
        NUM_QIS=$(( (QI_TOTAL + 1) / 2 ))
        CAT_QIS=$(( QI_TOTAL / 2 ))
        

        # CSV filename includes configuration so files can be reused
        DATA_CSV="$DATA_DIR/bench_${ROWS}rows_nqi${NUM_QIS}_cqi${CAT_QIS}.csv"

        # Generate data if needed
        if [[ "$SKIP_GENERATE" == "false" || ! -f "$DATA_CSV" ]]; then
            echo "[bench] Generating data: rows=$ROWS  num_qis=$NUM_QIS  cat_qis=$CAT_QIS"
            if ! $DRY_RUN; then
                "$PYTHON_BIN" "$BENCH_DIR/generate_data.py" \
                    --rows      "$ROWS" \
                    --num-qis   "$NUM_QIS" \
                    --cat-qis   "$CAT_QIS" \
                    --output    "$DATA_CSV" \
                    --progress
            fi
        else
            echo "[bench] Reusing existing: $DATA_CSV"
        fi

        for K in $K_VALUES; do
            RUN_NUM=$((RUN_NUM + 1))
            echo ""
            echo "[bench] ──── Run $RUN_NUM / $TOTAL_RUNS ────────────────────────"
            echo "[bench]      rows=$ROWS  total_qis=$QI_TOTAL  k=$K  pipeline=$PIPELINE_TYPE"

            if ! $DRY_RUN; then
                RECORD=""
                if [[ "$PIPELINE_TYPE" == "rust" ]]; then
                    RECORD="$(run_rust "$DATA_CSV" "$NUM_QIS" "$CAT_QIS" "$K" "$ROWS")"
                else
                    RECORD="$(run_python "$DATA_CSV" "$NUM_QIS" "$CAT_QIS" "$K" "$ROWS")"
                fi
                RESULTS+=("$RECORD")

                # Print summary line
                echo "$RECORD" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
status = d['status']
mark = '✓' if status == 'success' else '✗'
print(f\"  {mark} wall={d['wall_time_s']:.1f}s  rss={d['peak_rss_mb']}MB  \"\
      f\"in={d['input_size_mb']:.0f}MB  out={d['output_size_mb']:.0f}MB  \"\
      f\"throughput={d['throughput_mb_per_s']:.1f} MB/s\")
if d['error_msg']:
    print(f\"  ERROR: {d['error_msg']}\")
"
            fi
        done

        # Clean up generated CSV unless --keep-data
        if [[ "$KEEP_DATA" == "false" && ! "$SKIP_GENERATE" == "true" && -f "$DATA_CSV" ]]; then
            rm -f "$DATA_CSV"
        fi
    done
done

# ── Write results JSON ─────────────────────────────────────────────────────────

if ! $DRY_RUN; then
    echo ""
    echo "[bench] Writing results to $OUTPUT_FILE"
    mkdir -p "$(dirname "$OUTPUT_FILE")"

    python3 - "$OUTPUT_FILE" "$PIPELINE_TYPE" "$TIMESTAMP" <<PYEOF "${RESULTS[@]}"
import sys, json

outfile      = sys.argv[1]
pipeline     = sys.argv[2]
timestamp    = sys.argv[3]
records_raw  = sys.argv[4:]

records = [json.loads(r) for r in records_raw if r.strip()]

summary = {
    "pipeline":   pipeline,
    "generated":  timestamp,
    "total_runs": len(records),
    "successful": sum(1 for r in records if r["status"] == "success"),
    "failed":     sum(1 for r in records if r["status"] != "success"),
    "results":    records,
}

with open(outfile, "w") as f:
    json.dump(summary, f, indent=2)

print(f"  {len(records)} records written ({summary['successful']} success, {summary['failed']} failed)")
PYEOF

    echo ""
    echo "[bench] Done. Results: $OUTPUT_FILE"
    echo "[bench] Compare with:  python benchmark/compare_results.py <results_rust.json> <results_python.json>"
fi
