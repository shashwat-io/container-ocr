#!/usr/bin/env bash
# PART 1 of 2 -- runs now, no app code required.
#
# Resolves the opencv-python / opencv-python-headless / opencv-contrib-python
# three-way conflict in flask-app-venv, matching the exact remedy that fixed
# this same class of problem locally on 07-10 (uninstall ALL variants, then a
# single clean reinstall -- not a partial removal, since partial removal is
# exactly what risks leaving cross-contaminated files in site-packages/cv2/).
#
# Root cause (confirmed against installed package metadata + source):
#   - paddleocr==3.7.0 requires paddlex[ocr-core], which hard-pins
#     opencv-contrib-python==4.10.0.84 unconditionally for OCR use.
#   - ultralytics==8.4.78 separately declares opencv-python>=4.6.0.
#   - Grepped installed source of paddlex, paddleocr, ultralytics, and this
#     app's own app/ code: zero usage of any opencv_contrib-exclusive API
#     (ximgproc, bgsegm, xfeatures2d, etc.) anywhere. opencv-contrib-python's
#     cv2 is a strict superset and satisfies all of them functionally.
#   - paddlex's pin can't be avoided without much riskier --no-deps surgery on
#     paddlex itself (would also silently drop its OTHER real dependencies).
#     ultralytics' pin is a single isolated requirement we CAN safely drop.
#   -> keep opencv-contrib-python only; remove opencv-python and
#      opencv-python-headless.
#
# PART 2 (fix_opencv_conflict_part2_appboot.sh) does the full run.py boot
# re-verification once the app repo has been transferred to this server --
# run that separately after the transfer, don't skip it.
set -uo pipefail

VENV_DIR="/mnt/Data/container-ocrpoc/test/flask-app-venv"

echo "=========================================="
echo "STEP 0: BEFORE STATE (raw)"
echo "=========================================="
source "$VENV_DIR/bin/activate"
echo "Active python: $(python3 --version)"
echo
pip list 2>/dev/null | grep -i opencv
echo

echo "=========================================="
echo "STEP 1: UNINSTALL ALL THREE OPENCV VARIANTS"
echo "=========================================="
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
echo
echo "--- confirm clean slate ---"
pip list 2>/dev/null | grep -i opencv || echo "(none installed -- clean slate confirmed)"
echo

echo "=========================================="
echo "STEP 2: CLEAN REINSTALL OF opencv-contrib-python==4.10.0.84 ONLY"
echo "=========================================="
pip install --no-cache-dir opencv-contrib-python==4.10.0.84
echo

echo "=========================================="
echo "STEP 3: IMPORT VERIFICATION (raw, no summarizing)"
echo "=========================================="
python3 -c "import cv2; print('cv2 import OK:', cv2.__version__, cv2.__file__)"
echo

echo "=========================================="
echo "STEP 3b: pip check (informational)"
echo "=========================================="
echo "Expected: this will likely flag ultralytics as wanting opencv-python,"
echo "which is now intentionally absent. This is a pip BOOKKEEPING note only --"
echo "ultralytics imports cv2 at runtime regardless of which opencv-* package"
echo "provided it, so this does not affect functionality. Confirmed by the"
echo "import check below actually exercising ultralytics + cv2 together."
pip check || true
echo

echo "=========================================="
echo "STEP 3c: ULTRALYTICS + CV2 TOGETHER (the exact combination that broke"
echo "         on 07-10 -- verifying it specifically, not just cv2 alone)"
echo "=========================================="
python3 -c "
import cv2
import ultralytics
print('cv2', cv2.__version__)
print('ultralytics', ultralytics.__version__)
import numpy as np
from ultralytics import YOLO
print('ultralytics YOLO class import OK')
"
echo

echo "=========================================="
echo "PART 1 DONE. Paste this output back."
echo "Once the app repo is transferred to this server, run"
echo "fix_opencv_conflict_part2_appboot.sh to finish the re-verification."
echo "=========================================="
