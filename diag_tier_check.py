"""
Diagnostic: verify tiered _try_extract() is active and working on
known contamination-failure images.
"""
import os
import sys
import logging
import time

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

# ── Step 1: verify ocr.py on disk and wipe any stale .pyc ────────────────────
OCR_PATH = 'app/inference/ocr.py'
mtime = os.path.getmtime(OCR_PATH)
print("=" * 70)
print(f"ocr.py mtime : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")

import glob, importlib
pyc_glob = 'app/inference/__pycache__/ocr*.pyc'
stale = glob.glob(pyc_glob)
if stale:
    for p in stale:
        os.remove(p)
        print(f"Deleted stale pyc: {p}")
else:
    print("No stale .pyc found for app/inference/ocr")

# ── Step 2: set up explicit stdout logging for app.inference.ocr ──────────────
ocr_logger = logging.getLogger('app.inference.ocr')
ocr_logger.setLevel(logging.DEBUG)
# Remove any pre-existing handlers so we control the output
ocr_logger.handlers.clear()
ocr_logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(logging.Formatter('[OCR-LOG] %(message)s'))
ocr_logger.addHandler(_handler)

# ── Step 3: import and patch run_ocr_on_crop to capture raw texts ─────────────
import app.inference.ocr as ocr_mod

_last_texts = []

_orig_run_ocr = ocr_mod.run_ocr_on_crop
def _patched_run_ocr(crop_image):
    global _last_texts
    result = _orig_run_ocr(crop_image)
    _last_texts = result
    return result
ocr_mod.run_ocr_on_crop = _patched_run_ocr

print(f"\nocr module loaded from: {ocr_mod.__file__}")
print("=" * 70)

# ── Step 4: test images ───────────────────────────────────────────────────────
# Chosen from benchmark_results.csv: detected=True, is_valid=False,
# clear type-code / dimension-sticker contamination in ocr_raw.
TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'

TEST_CASES = [
    # EGHU37696D on line 1 (10 chars), 22G1 on line 2 — Tier 1b should recover check digit
    "3864162003561327490_0070_jpg.rf.5855c7d40fe329fce69fbff537890b39.jpg",
    # 22G1 then MCU3030647 — type code on separate line from real code
    "IMG_0184_ce_png_jpg.rf.f6faff45fbfc265e22ecab2eb734f485.jpg",
    # 45G1 / 6302 / SKHU across lines — all three lines needed to reconstruct
    "Event20231020095456004-Avi_8170_jpg.rf.94f6da41ac86653ba35cca0ecd6a9a69.jpg",
]

from ultralytics import YOLO
model = YOLO('models/best.pt')
import numpy as np
from PIL import Image

for fname in TEST_CASES:
    path = os.path.join(TEST_DIR, fname)
    if not os.path.exists(path):
        print(f"\n[SKIP] not found: {fname}")
        continue

    print(f"\n{'─'*70}")
    print(f"Image : {fname}")

    img_pil = Image.open(path).convert('RGB')
    img_np  = np.array(img_pil)
    with open(path, 'rb') as f:
        image_bytes = f.read()

    results = model(img_np, verbose=False)
    boxes = [tuple(float(v) for v in r_box.xyxy[0])
             for r in results for r_box in r.boxes]
    print(f"Boxes : {len(boxes)}")
    if not boxes:
        print("  (no YOLO detection — skipping)")
        continue

    x1, y1, x2, y2 = boxes[0]
    crop = ocr_mod.crop_detection(image_bytes, x1, y1, x2, y2)
    w, h = crop.size
    print(f"Crop  : {w}x{h}  {'portrait' if h > w else 'landscape'}")

    # Reset captured texts before this image
    _last_texts = []

    print("\n--- logger output (tier tags) ---")
    result = ocr_mod.extract_container_code(image_bytes, x1, y1, x2, y2)
    print("--- end logger output ---")

    print(f"\nrun_ocr_on_crop() raw texts (last captured):")
    for t, s in _last_texts:
        print(f"  {t!r:30s}  score={s:.4f}")

    print(f"\nFinal result:")
    print(f"  container_code : {result['container_code']!r}")
    print(f"  is_valid       : {result['is_valid']}")
    print(f"  validation_reason : {result['validation_reason']}")
    print(f"  ocr_raw        : {result['ocr_raw_output']}")

print(f"\n{'='*70}")
print("Done.")
