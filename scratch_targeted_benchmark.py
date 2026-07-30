#!/usr/bin/env python3
"""
Time-boxed targeted spot check of the tier3-pass2 fix (NOT the full 786-image
benchmark — that still needs to run tomorrow with a full time block).

Runs REAL OCR (extract_container_code) only on the 131 rows in
scratch_targeted_sample_rows.csv: all 71 former Tier3-Pass2 rows +
60 random control rows (seed=42) from the rest of the dataset.

Does NOT modify benchmark.py. Writes its own separate output CSV.
"""
import os, csv, time, sys
import numpy as np
from PIL import Image
from collections import Counter

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from ultralytics import YOLO
from app.inference.ocr import extract_container_code

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
SAMPLE_CSV = 'scratch_targeted_sample_rows.csv'
OUTPUT_CSV = 'scratch_targeted_benchmark_results.csv'

targeted = list(csv.DictReader(open(SAMPLE_CSV)))
print(f"Targeted rows to process: {len(targeted)}", flush=True)

model = YOLO('models/best.pt')
_box_cache = {}
def boxes_for(fname):
    if fname in _box_cache:
        return _box_cache[fname]
    img = np.array(Image.open(os.path.join(TEST_DIR, fname)).convert('RGB'))
    res = model(img, verbose=False)
    bxs = [[float(v) for v in b.xyxy[0]] for r in res for b in r.boxes]
    _box_cache[fname] = bxs
    return bxs

start = time.time()
out_rows = []
box_drift = 0

for i, row in enumerate(targeted, 1):
    fname = row['filename']
    idx = int(row['box_idx'])
    group = row['group']
    bxs = boxes_for(fname)
    if idx >= len(bxs):
        box_drift += 1
        out_rows.append({'filename': fname, 'box_idx': idx, 'group': group,
                          'container_code': '', 'is_valid': False, 'ocr_raw': 'BOX DRIFT'})
        continue
    x1, y1, x2, y2 = bxs[idx]
    with open(os.path.join(TEST_DIR, fname), 'rb') as f:
        image_bytes = f.read()
    result = extract_container_code(image_bytes, x1, y1, x2, y2)
    out_rows.append({
        'filename': fname, 'box_idx': idx, 'group': group,
        'container_code': result['container_code'],
        'is_valid': result['is_valid'],
        'ocr_raw': result['ocr_raw_output'],
    })
    if i % 20 == 0 or i == len(targeted):
        elapsed = time.time() - start
        avg = elapsed / i
        remaining = avg * (len(targeted) - i)
        print(f"[{i}/{len(targeted)}] {elapsed:.0f}s elapsed | avg {avg:.1f}s/img | "
              f"est. remaining {remaining:.0f}s", flush=True)

elapsed = time.time() - start
with open(OUTPUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['filename', 'box_idx', 'group', 'container_code', 'is_valid', 'ocr_raw'])
    w.writeheader()
    w.writerows(out_rows)

print(f"\nDone. Total time: {elapsed:.0f}s. box_drift={box_drift}. Written to {OUTPUT_CSV}", flush=True)
print("TARGETED_BENCHMARK_COMPLETE", flush=True)
