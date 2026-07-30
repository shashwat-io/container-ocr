"""
READ-ONLY. No OCR/PaddleX calls, no ocr.py edits, no server restart.

Full review-crop build for Tier2-exact (23 rows) -- the last unaudited slice
of the valid-result population (Tier1-exact, Tier1b, Tier2-recovery, and
Tier4-confusion already reviewed). Full review, no sampling: 23 rows is
small enough to do completely.

Tier2-exact mechanism (confirmed by reading ocr.py lines 199-208): for each
pair of adjacent OCR-detected lines, concatenate + clean, then slide an
11-char window across the merged string; any window matching the
structural pattern (4 letters + 7 digits) is checksum-validated and the
first pass wins. All 11 characters (including the check digit) come
directly from OCR text -- zero character correction, zero check-digit
invention. Same logic as Tier1-exact, just applied to a merged 2-line
string instead of a single line.

Source: benchmark_results_full_live_corrected_tiers.csv (replay-corrected
tier labels).

Crops regenerated via crop_detection() (app/inference/ocr.py) using
recomputed YOLO box coordinates -- same zero-OCR-rerun method as the prior
four reviews. box_idx = occurrence index of the filename when iterating the
full corrected-tiers CSV top-to-bottom (matches row write order in
benchmark_full_live.py). Rows with num_boxes > 1 are flagged
box_ambiguous=True and ALL boxes for that image are saved for manual
disambiguation.

Writes: tier2_exact_review_samples/  (+ review_sheet.csv, same columns as
the Tier1-exact review sheet, verified_correct left blank)
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from ultralytics import YOLO
from app.inference.ocr import crop_detection

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
CORRECTED_CSV = 'benchmark_results_full_live_corrected_tiers.csv'
OUT_DIR = 'tier2_exact_review_samples'
TIER_LABEL = 'Tier2-exact'
BLOCKS = [(1, 100), (101, 200), (201, 300), (301, 400),
          (401, 500), (501, 600), (601, 700), (701, 786)]

os.makedirs(OUT_DIR, exist_ok=True)

images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
idx_of = {f: i for i, f in enumerate(images)}
print(f"Master image list: {len(images)} images", flush=True)

with open(CORRECTED_CSV, newline='') as f:
    all_rows = list(csv.DictReader(f))

seen = Counter()
for r in all_rows:
    fname = r['filename']
    r['_box_idx'] = seen[fname]
    seen[fname] += 1

population = [r for r in all_rows if r['corrected_tier'] == TIER_LABEL]
print(f"{TIER_LABEL} population: {len(population)} rows", flush=True)
assert len(set(r['filename'] for r in population)) == len(population), "unexpected duplicate filename within tier"

def block_of(idx1):
    for (lo, hi) in BLOCKS:
        if lo <= idx1 <= hi:
            return (lo, hi)
    raise ValueError(idx1)

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

sheet_rows = []
extraction_failures = []
letters = 'abcdefgh'

for i, row in enumerate(population, 1):
    fname = row['filename']
    recorded_num_boxes = int(row['num_boxes'])
    box_idx = row['_box_idx']
    bxs = boxes_for(fname)
    box_count_mismatch = (len(bxs) != recorded_num_boxes)
    box_ambiguous = recorded_num_boxes > 1

    with open(os.path.join(TEST_DIR, fname), 'rb') as fh:
        image_bytes = fh.read()

    saved_names = []
    try:
        if box_ambiguous:
            for j, (x1, y1, x2, y2) in enumerate(bxs):
                crop = crop_detection(image_bytes, x1, y1, x2, y2)
                out_name = f"sample{i}_{letters[j]}.jpg"
                crop.save(os.path.join(OUT_DIR, out_name))
                saved_names.append(out_name)
        else:
            if box_idx < len(bxs):
                x1, y1, x2, y2 = bxs[box_idx]
                crop = crop_detection(image_bytes, x1, y1, x2, y2)
                out_name = f"sample{i}.jpg"
                crop.save(os.path.join(OUT_DIR, out_name))
                saved_names.append(out_name)
            else:
                box_count_mismatch = True
                extraction_failures.append(fname)
    except Exception as e:
        extraction_failures.append(f"{fname} ({e})")

    idx1 = idx_of[fname] + 1
    sheet_rows.append({
        'sample_number': i,
        'filename': fname,
        'claimed_code': row['container_code'],
        'ocr_raw': row['ocr_raw'],
        'ocr_confidence': row['ocr_confidence'],
        'validation_reason': row['validation_reason'],
        'verified_correct': '',
        'saved_crop_image(s)': ', '.join(saved_names),
        'box_ambiguous': box_ambiguous,
        'box_count_mismatch': box_count_mismatch,
        'recorded_num_boxes': recorded_num_boxes,
        'recomputed_num_boxes': len(bxs),
        'box_idx': box_idx,
        'image_index': idx1,
        'block': f"{block_of(idx1)[0]}-{block_of(idx1)[1]}",
    })

    print(f"  [{i}/{len(population)}] {fname}  boxes={len(bxs)} ambiguous={box_ambiguous}", flush=True)

fieldnames = ['sample_number', 'filename', 'claimed_code', 'ocr_raw',
              'ocr_confidence', 'validation_reason', 'verified_correct',
              'saved_crop_image(s)', 'box_ambiguous', 'box_count_mismatch',
              'recorded_num_boxes', 'recomputed_num_boxes', 'box_idx',
              'image_index', 'block']
sheet_path = os.path.join(OUT_DIR, 'review_sheet.csv')
with open(sheet_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(sheet_rows)

n_ambig = sum(1 for r in sheet_rows if r['box_ambiguous'])
n_mismatch = sum(1 for r in sheet_rows if r['box_count_mismatch'])
print(f"\nTotal rows: {len(sheet_rows)}", flush=True)
print(f"box_ambiguous: {n_ambig} | box_count_mismatch: {n_mismatch}", flush=True)
print(f"Extraction failures: {len(extraction_failures)}", flush=True)
for f_ in extraction_failures:
    print("  FAILED:", f_, flush=True)
if n_ambig:
    print("\nAmbiguous rows (need box=a/b/... disambiguation):", flush=True)
    for r in sheet_rows:
        if r['box_ambiguous']:
            print(f"  sample{r['sample_number']}: {r['filename']} | claimed={r['claimed_code']} | crops={r['saved_crop_image(s)']}", flush=True)
print(f"\nSaved crops + review_sheet.csv to {OUT_DIR}/", flush=True)
print("DONE", flush=True)
