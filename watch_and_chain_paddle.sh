#!/usr/bin/env bash
# Waits for the running TrOCR full-786 cascade benchmark (PID passed in, default
# 7528) to exit, verifies it actually COMPLETED rather than merely died, and only
# then chains the PaddleOCR full-786 run behind it.
#
# Two independent completion checks, both must pass:
#   1. the log carries the final "Total time: Ns | Avg per image: Xs" summary
#      (only printed after the image loop finishes)
#   2. the results CSV holds close to 786 DISTINCT filenames
#
# Check 2 counts distinct filenames, not raw lines: the CSV has one row per YOLO
# box, so images with >1 detection push the line count above 786 (~830-840
# historically). A line-count threshold would therefore be satisfiable by ~750
# images with lots of multi-box rows -- distinct filenames is the honest measure.
#
# Never launches on a failed/killed/partial run; writes the reason to
# watcher_status.txt instead.

set -uo pipefail

REPO=/home/shashwatkumar/Testing/container_ocr_test
WATCH_PID="${1:-7528}"
POLL_SECONDS=60

TROCR_LOG="$REPO/trocr_cascade_FULL_thin_20260728.log"
TROCR_CSV="$REPO/benchmark_results_trocr_cascade_FULL_thin_20260728.csv"
STATUS="$REPO/watcher_status.txt"

PADDLE_CSV="$REPO/benchmark_results_paddle_cascade_FULL_20260728.csv"
PADDLE_LOG="$REPO/paddle_cascade_FULL_20260728.log"

EXPECTED_IMAGES=786
MIN_IMAGES=780   # "close to 786" -- tolerates a couple of unreadable files

note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$STATUS"; }

note "watcher started (pid $$), waiting on PID $WATCH_PID"

# ── 1. Wait for the TrOCR run to exit ────────────────────────────────────────
# Match on cmdline, not just PID existence: over a ~2h wait the kernel could
# recycle 7528 onto an unrelated process, which would make the watcher hang
# indefinitely. Treating "PID gone OR cmdline no longer ours" as exit is the
# safe reading -- both mean our benchmark is no longer running.
still_running() {
    [ -r "/proc/$WATCH_PID/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$WATCH_PID/cmdline" 2>/dev/null \
        | grep -q 'benchmark_trocr_cascade_quicksample\.py'
}

# 60s poll keeps the watcher off the CPU while the benchmark saturates 8 cores.
while still_running; do
    sleep "$POLL_SECONDS"
done

note "PID $WATCH_PID exited; verifying completion"

# Give the process a moment to flush its final CSV write.
sleep 10

# ── 2. Check A: completion summary present in the log ────────────────────────
if grep -qE '^Total time: +[0-9]+s \| Avg per image: [0-9.]+s' "$TROCR_LOG" 2>/dev/null; then
    SUMMARY=$(grep -E '^Total time:' "$TROCR_LOG" | tail -1)
    note "CHECK A PASS: $SUMMARY"
    CHECK_A=1
else
    note "CHECK A FAIL: no 'Total time: ... | Avg per image: ...' summary in $(basename "$TROCR_LOG")"
    note "  last 5 log lines:"
    tail -5 "$TROCR_LOG" 2>/dev/null | sed 's/^/    /' >> "$STATUS"
    CHECK_A=0
fi

# ── 3. Check B: CSV holds ~786 distinct images ───────────────────────────────
if [ -f "$TROCR_CSV" ]; then
    # cut on comma: filename is field 1 and image filenames contain no commas.
    N_IMAGES=$(tail -n +2 "$TROCR_CSV" | cut -d, -f1 | sort -u | grep -c . )
    N_ROWS=$(( $(wc -l < "$TROCR_CSV") - 1 ))
    if [ "$N_IMAGES" -ge "$MIN_IMAGES" ]; then
        note "CHECK B PASS: $N_IMAGES distinct images (>= $MIN_IMAGES), $N_ROWS rows"
        CHECK_B=1
    else
        note "CHECK B FAIL: only $N_IMAGES distinct images in CSV (need >= $MIN_IMAGES of $EXPECTED_IMAGES); $N_ROWS rows"
        CHECK_B=0
    fi
else
    note "CHECK B FAIL: $TROCR_CSV does not exist"
    CHECK_B=0
fi

# ── 4. Launch only if both checks passed ─────────────────────────────────────
if [ "$CHECK_A" -eq 1 ] && [ "$CHECK_B" -eq 1 ]; then
    # Belt-and-braces: never stack two cascades. One run already pulls ~650% of
    # 8 cores, so a second would oversubscribe rather than parallelise.
    if pgrep -f 'benchmark_full_live_resplit\.py|benchmark_trocr_cascade_quicksample\.py' >/dev/null 2>&1; then
        note "ABORT: a benchmark process is already running -- refusing to launch a second"
        exit 1
    fi

    cd "$REPO" || { note "ABORT: cannot cd to $REPO"; exit 1; }

    # BENCH_SAMPLE_FILE must be absent/empty so the script runs all 786.
    # FINETUNED_REC_MODEL_DIR must be absent so PaddleOCR runs STOCK -- that is
    # the configuration the 69.0%/64.7% historical figures were measured on.
    unset BENCH_SAMPLE_FILE
    unset FINETUNED_REC_MODEL_DIR

    BENCH_OUTPUT_CSV="$PADDLE_CSV" \
    nohup python3 benchmark_full_live_resplit.py > "$PADDLE_LOG" 2>&1 &
    PADDLE_PID=$!
    disown

    note "LAUNCHED PaddleOCR full-786 cascade, pid $PADDLE_PID -> $(basename "$PADDLE_LOG")"
    note "  output CSV: $(basename "$PADDLE_CSV")"
else
    note "NOT LAUNCHING PaddleOCR run -- TrOCR benchmark did not verify as complete (A=$CHECK_A B=$CHECK_B)"
fi

note "watcher done"
