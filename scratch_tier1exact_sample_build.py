"""
READ-ONLY. No OCR/PaddleX calls, no ocr.py edits, no server restart.

Builds a stratified review sample of Tier1-exact rows (280 total, the
largest and never-reviewed tier) to check for a hidden wrong-rate: a clean
misread that happens to still land on a *different but still-valid* ISO 6346
code, which the checksum alone cannot catch.

Sample size 45 -- matching Tier1b's original review size for a directly
comparable confidence level. Stratification mirrors quick_sample_set.txt's
construction exactly: block by image-index in sorted(os.listdir(test/images))
order (blocks of 100, last block 701-786), proportional target per block
(largest-remainder rounding), random.seed(42) called once then
random.sample() per block in block order (not reseeded per block).

Crops regenerated via crop_detection() (app/inference/ocr.py) using
recomputed YOLO box coordinates -- same zero-OCR-rerun method as the
Tier1b/Tier4-confusion/Tier2-recovery reviews. box_idx = occurrence index of
the filename when iterating the full corrected-tiers CSV top-to-bottom
(matches row write order in benchmark_full_live.py).

Writes: tier1_exact_review_samples/  (+ review_sheet.csv)
"""
import csv
import os
import random
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
OUT_DIR = 'tier1_exact_review_samples'
TIER_LABEL = 'Tier1-exact'
SAMPLE_SIZE = 45
BLOCKS = [(1, 100), (101, 200), (201, 300), (301, 400),
          (401, 500), (501, 600), (601, 700), (701, 786)]

os.makedirs(OUT_DIR, exist_ok=True)

# ---- Master index order (identical basis to quick_sample_set.txt) --------
images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
idx_of = {f: i for i, f in enumerate(images)}
print(f"Master image list: {len(images)} images", flush=True)

with open(CORRECTED_CSV, newline='') as f:
    all_rows = list(csv.DictReader(f))

# box_idx = occurrence index per filename, iterating full file in order.
seen = Counter()
for r in all_rows:
    fname = r['filename']
    r['_box_idx'] = seen[fname]
    seen[fname] += 1

population = [r for r in all_rows if r['corrected_tier'] == TIER_LABEL]
print(f"{TIER_LABEL} population: {len(population)} rows", flush=True)

def block_of(idx1):
    for (lo, hi) in BLOCKS:
        if lo <= idx1 <= hi:
            return (lo, hi)
    raise ValueError(idx1)

by_block = {b: [] for b in BLOCKS}
for r in population:
    idx1 = idx_of[r['filename']] + 1
    by_block[block_of(idx1)].append(r)

block_counts = {b: len(rs) for b, rs in by_block.items()}
print("Rows per block:", block_counts, flush=True)

# ---- Proportional targets, largest-remainder rounding ---------------------
total_pop = sum(block_counts.values())
raw = {b: c * SAMPLE_SIZE / total_pop for b, c in block_counts.items()}
floor = {b: int(v) for b, v in raw.items()}
remainder = {b: raw[b] - floor[b] for b in raw}
deficit = SAMPLE_SIZE - sum(floor.values())
order = sorted(remainder.items(), key=lambda kv: -kv[1])
targets = dict(floor)
for b, _ in order[:deficit]:
    targets[b] += 1
assert sum(targets.values()) == SAMPLE_SIZE
print("Block targets:", targets, flush=True)

# ---- Sample: seed once, random.sample per block in block order ------------
random.seed(42)
sample = []
for b in BLOCKS:
    pool = by_block[b]
    k = targets[b]
    picked = random.sample(pool, k)
    sample.extend(picked)
    print(f"  block {b}: pool={len(pool)} target={k}", flush=True)

print(f"\nFinal sample size: {len(sample)}", flush=True)

# ---- Crops (recompute YOLO boxes, crop_detection(), zero OCR) -------------
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

for i, row in enumerate(sample, 1):
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
print(f"\nbox_ambiguous: {n_ambig} | box_count_mismatch: {n_mismatch}", flush=True)
print(f"Extraction failures: {len(extraction_failures)}", flush=True)
for f_ in extraction_failures:
    print("  FAILED:", f_, flush=True)
print(f"Saved crops + review_sheet.csv to {OUT_DIR}/", flush=True)
print("DONE", flush=True)
