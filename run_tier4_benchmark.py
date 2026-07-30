#!/usr/bin/env python3
"""
Self-verifying launcher for the Tier 4 benchmark run.

Performs the Step 2 startup verification BEFORE any image processing, then
hands off to benchmark.py via runpy so the existing benchmark logic runs
unchanged. All verification output is printed first (stdout) and also
written to a run-log header file so it travels with the run.
"""
import os
import sys
import time
import runpy
import datetime

PROJECT_ROOT = '/home/shashwatkumar/Testing/container_ocr_test'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

OCR_PATH = os.path.join(PROJECT_ROOT, 'app', 'inference', 'ocr.py')
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, 'benchmark.py')
MARKER = "Valid (character confusion corrected"
VERIFY_LOG = os.path.join(PROJECT_ROOT, 'tier4_run_verification.log')

# Collect verification lines so they go to BOTH stdout and the log file.
_lines = []


def emit(line=""):
    print(line, flush=True)
    _lines.append(line)


emit("================ TIER 4 BENCHMARK — STARTUP VERIFICATION ================")

# (3) PID + exact start timestamp
pid = os.getpid()
start_ts = datetime.datetime.now().astimezone()
emit(f"Process PID:        {pid}")
emit(f"Start timestamp:    {start_ts.isoformat()}")

# (1) absolute path + mtime of ocr.py
abs_ocr = os.path.abspath(OCR_PATH)
ocr_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(OCR_PATH)).astimezone()
emit(f"ocr.py abs path:    {abs_ocr}")
emit(f"ocr.py mtime:       {ocr_mtime.isoformat()}")

# (2) import the module, read its own __file__ source, confirm marker
import app.inference.ocr as ocr_mod  # noqa: E402

loaded_file = os.path.abspath(ocr_mod.__file__)
emit(f"imported __file__:  {loaded_file}")

if loaded_file != abs_ocr:
    emit(f"WARNING: imported module path != expected ocr.py path")

with open(ocr_mod.__file__, 'r') as fh:
    src = fh.read()

marker_present = MARKER in src
emit(f"marker searched:    {MARKER!r}")
emit(f"marker present:     {marker_present}")

if marker_present:
    emit("FRESH CODE CONFIRMED")
    status = "FRESH"
else:
    emit("STALE CODE DETECTED — ABORTING")
    status = "STALE"

emit("========================================================================")

# Persist the verification header to a log file so it travels with the run.
with open(VERIFY_LOG, 'w') as fh:
    fh.write("\n".join(_lines) + "\n")

if status == "STALE":
    # Exit immediately WITHOUT processing any images.
    sys.exit(2)

# Hand off to the real benchmark, unchanged.
emit("")
emit(">>> Handing off to benchmark.py ...")
emit("")
runpy.run_path(BENCHMARK_PATH, run_name="__main__")
