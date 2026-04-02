#!/usr/bin/env bash
# =============================================================================
# Overnight benchmark launcher — survives VSCode / terminal disconnect.
#
# Usage:
#   bash benchmark/run_overnight.sh
#
# Then close VSCode / disconnect safely. Check progress any time with:
#   tail -f /tmp/skald_bench_overnight.log
#
# Results will be at:
#   benchmark/results_rust_<timestamp>.json   (or python)
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="/tmp/skald_bench_overnight.log"
PID_FILE="/tmp/skald_bench.pid"

# ── Overnight config ──────────────────────────────────────────────────────────
# VM disk: 29GB total / ~11GB free  RAM: 31GB / 28GB free  Swap: none
#
# Peak disk per run (no --keep-data):
#   = 1 CSV on disk + chunks dir (~same size as CSV) + output (~0.5GB)
#   1M  rows ≈ 0.05GB CSV  →  ~0.1GB peak   ✓
#   10M rows ≈  0.5GB CSV  →  ~1.0GB peak   ✓
#   50M rows ≈  2.3GB CSV  →  ~5.0GB peak   ✓
#   100M rows ≈ 4.6GB CSV  →  ~9.5GB peak   ✓ (tight — keep buffer in mind)
#
# Without --keep-data each CSV is deleted after its k-value loop,
# so only ONE CSV is on disk at any time.
SIZES="1000000 10000000 50000000 100000000"
QI_CONFIGS="2 4 6 8"
K_VALUES="5 20 50"

# No --keep-data: delete each CSV after use to stay within 11GB free
EXTRA_FLAGS="--data-dir /tmp/skald_bench_data --workdir /tmp/skald_bench"

echo "======================================================"
echo "  SKALD overnight benchmark"
echo "  Log:     $LOG"
echo "  PID file: $PID_FILE"
echo "  Sizes:   $SIZES"
echo "  QIs:     $QI_CONFIGS"
echo "  k vals:  $K_VALUES"
echo "  Extra flags: $EXTRA_FLAGS"
echo "======================================================"
echo ""
echo "Launching in background with nohup..."
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
nohup bash "$REPO_ROOT/benchmark/run_benchmark.sh" \
    --sizes      "$SIZES" \
    --qi-configs "$QI_CONFIGS" \
    --k-values   "$K_VALUES" \
    $EXTRA_FLAGS \
    >> "$LOG" 2>&1 &

BG_PID=$!
echo $BG_PID > "$PID_FILE"

echo ""
echo "  Background PID: $BG_PID  (saved to $PID_FILE)"
echo ""
echo "You can now safely close VSCode and disconnect."
echo ""
echo "  Check progress:   tail -f $LOG"
echo "  Check if running: kill -0 \$(cat $PID_FILE) && echo running || echo done"
echo "  Stop early:       kill \$(cat $PID_FILE)"
echo ""
echo "Results will appear in:  $REPO_ROOT/benchmark/"
