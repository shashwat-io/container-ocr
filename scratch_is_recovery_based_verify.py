"""
READ-ONLY verification. Live extract_container_code() calls (real OCR, tiny
count) against known rows to confirm the is_recovery_based/is_merge_based
extension behaves as intended. No ocr.py/predictor.py edits happen here --
this only exercises the already-edited code.
"""
import os
import sys
import csv
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from ultralytics import YOLO
from app.inference.ocr import extract_container_code

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
CORRECTED_CSV = 'benchmark_results_full_live_corrected_tiers.csv'

with open(CORRECTED_CSV, newline='') as f:
    all_rows = list(csv.DictReader(f))
seen = Counter()
for r in all_rows:
    r['_box_idx'] = seen[r['filename']]
    seen[r['filename']] += 1

targets = [
    ("Tier2-exact KNOWN WRONG #1", "IMG_0414_ce_png_jpg.rf.ceceb118ee9c0527c668f187a9815c4f.jpg", "IRKU8476800", True),
    ("Tier2-exact KNOWN WRONG #2 (docpp-off-retry path)", "IMG_20230101_153713_region_jpg.rf.67ae464ab9c41bbc1ffa534fb45be374.jpg", "XINU1955119", True),
    ("Tier1-exact regression check", "1-132938001-OCR-AS-B01_jpg_ce_png_jpg.rf.fa9a674067c6a2dd70dc0d5d819481db.jpg", "TGHU4816303", False),
    ("Tier1b regression check", "1114039705751937665_0050_jpg.rf.f5c42f338dbdc3a8898410b03bba225c.jpg", "TCLU4897918", True),
]

model = YOLO('models/best.pt')

def boxes_for(fname):
    img = np.array(Image.open(os.path.join(TEST_DIR, fname)).convert('RGB'))
    res = model(img, verbose=False)
    return [[float(v) for v in b.xyxy[0]] for r in res for b in r.boxes]

row_by_key = {}
for r in all_rows:
    row_by_key.setdefault(r['filename'], []).append(r)

print("=" * 90)
all_pass = True
for label, fname, expected_code, expected_flag in targets:
    matching_rows = row_by_key[fname]
    # find the row matching expected_code to get its box_idx
    target_row = next(r for r in matching_rows if r['container_code'] == expected_code)
    box_idx = target_row['_box_idx']

    bxs = boxes_for(fname)
    x1, y1, x2, y2 = bxs[box_idx]
    with open(os.path.join(TEST_DIR, fname), 'rb') as fh:
        image_bytes = fh.read()

    result = extract_container_code(image_bytes, x1, y1, x2, y2)
    got_flag = result['is_recovery_based']
    status = "PASS" if got_flag == expected_flag else "FAIL"
    if status == "FAIL":
        all_pass = False

    print(f"[{status}] {label}")
    print(f"  filename: {fname}")
    print(f"  expected_code (from prior benchmark): {expected_code}")
    print(f"  live container_code: {result['container_code']!r}")
    print(f"  live is_valid: {result['is_valid']}")
    print(f"  live validation_reason: {result['validation_reason']!r}")
    print(f"  live is_recovery_based: {got_flag}  (expected {expected_flag})")
    print("-" * 90)

print("ALL PASS" if all_pass else "SOME FAILED")
