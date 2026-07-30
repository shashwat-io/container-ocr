# Status: Tier3-Pass2 fix is LIVE, full benchmark still needs to run

**Date:** 2026-07-01 (evening)

## What's done
- `app/inference/ocr.py` has the Tier3-Pass2 check-digit-recovery fix live
  (disabled — 81%/17-21 manual-review error rate, no confidence signal
  available to gate on). Backup of the pre-fix version is at
  `app/inference/ocr.py.pre_tier3fix.bak`.
- Diff scope confirmed clean: only the Pass-2 block + two comments changed.
  Tier1b, Tier2-recovery, Tier4, the two-pass docpp logic, and
  `_best_portrait_rotation` are all untouched.
- CSV-replay sanity check (no live OCR) confirmed: Tier1b (157 rows) and
  Tier2-recovery (30 rows) 0 changed, literal/Tier4 buckets (352 rows)
  0 touched, 68/71 former Tier3-Pass2 rows now correctly invalid, 3/71
  rescued by the pre-existing Tier4 mechanism with the same code.
  **Projected new total: 542/786** — but this is a projection from replaying
  old CSV data, not a real run under the new code.

## What's NOT done
- The full 786-image `benchmark.py` run was killed mid-flight (interrupted
  at ~100/786 images, ~1087s in) because the machine was about to sleep/lose
  power. **No partial CSV was written** — `benchmark_results.csv` is
  byte-identical to `benchmark_results_pre_tier3fix.csv` (confirmed via
  md5sum), i.e. it still holds the OLD 610/786 pre-fix data. The interrupted
  run's log was renamed to `benchmark_run_tier3fix_INTERRUPTED_incomplete.log`
  to make clear it's not a completed result (ends in a Paddle SIGTERM abort
  traceback, no "BENCHMARK RESULTS" section).

## Next step
Re-run `benchmark.py` from scratch (fresh process, nothing salvageable from
the interrupted run) to get the real, official new accuracy number. Expect
something close to the ~542/786 projection, but treat that as unconfirmed
until the actual run completes. Standard pre-flight: `rm -rf
app/inference/__pycache__`, confirm no other benchmark/PaddleX process
running, unbuffered logging (`PYTHONUNBUFFERED=1 python3 -u`).

## Process state left running
Flask dev server (`run.py`, debug mode) was left running — no PaddleX
process is orphaned or running. Fine to leave as-is or kill; restarting
tomorrow is a single command either way.
