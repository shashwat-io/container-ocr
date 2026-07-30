"""
READ-ONLY. No OCR/PaddleX calls, no ocr.py edits, no server restart.

Builds review crops for the two remaining unknown-wrong-rate tiers:
  - Tier4-confusion (63 rows, corrected_tier column)
  - Tier2-recovery  (35 rows, corrected_tier column)

Source: benchmark_results_full_live_corrected_tiers.csv (already-verified
replay-corrected tier labels, 0 unrecoverable rows across all 610 valid rows
-- see scripts/replay_tier_attribution.py).

Methodology mirrors tier1b_review_samples/ and tier_b_verification_samples/
(scratch_tier_b_sample.py):
  - box_idx per row = occurrence index of that filename when iterating the
    full CSV top-to-bottom (row write order == box iteration order in
    benchmark_full_live.py's `for x1,y1,x2,y2 in boxes:` loop).
  - Boxes recomputed fresh via YOLO (models/best.pt) -- detection is
    deterministic given fixed weights, no PaddleX/OCR involved at all.
  - crop_detection() from app/inference/ocr.py (padding=10 default) used
    for cropping -- zero OCR re-run, zero model calls beyond YOLO.
  - Rows where num_boxes > 1 are flagged box_ambiguous=True and ALL boxes
    for that image are saved (sampleN_a.jpg, sampleN_b.jpg, ...) for manual
    disambiguation, same conservative approach as tier1b's samples 9/10.
  - Rows where recomputed box count != recorded num_boxes are flagged
    box_count_mismatch=True (should not occur; YOLO is deterministic).

Writes:
  tier4_confusion_review_samples/  (+ review_sheet.csv)
  tier2_recovery_review_samples/   (+ review_sheet.csv)
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

TIER_CONFIGS = [
    ('Tier4-confusion', 'tier4_confusion_review_samples'),
    ('Tier2-recovery', 'tier2_recovery_review_samples'),
]

with open(CORRECTED_CSV, newline='') as f:
    all_rows = list(csv.DictReader(f))

# Assign box_idx = occurrence index per filename, iterating full file in order.
seen = Counter()
for r in all_rows:
    fname = r['filename']
    r['_box_idx'] = seen[fname]
    seen[fname] += 1

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

for tier_label, out_dir in TIER_CONFIGS:
    os.makedirs(out_dir, exist_ok=True)
    population = [r for r in all_rows if r['corrected_tier'] == tier_label]
    print(f"\n=== {tier_label}: {len(population)} rows ===", flush=True)

    sheet_rows = []
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
        if box_ambiguous:
            # Save every box crop for this image for manual disambiguation.
            letters = 'abcdefgh'
            for j, (x1, y1, x2, y2) in enumerate(bxs):
                crop = crop_detection(image_bytes, x1, y1, x2, y2)
                out_name = f"sample{i}_{letters[j]}.jpg"
                crop.save(os.path.join(out_dir, out_name))
                saved_names.append(out_name)
        else:
            if box_idx < len(bxs):
                x1, y1, x2, y2 = bxs[box_idx]
                crop = crop_detection(image_bytes, x1, y1, x2, y2)
                out_name = f"sample{i}.jpg"
                crop.save(os.path.join(out_dir, out_name))
                saved_names.append(out_name)
            else:
                box_count_mismatch = True

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
        })

        if i % 10 == 0 or i == len(population):
            print(f"  [{i}/{len(population)}] {fname}", flush=True)

    sheet_path = os.path.join(out_dir, 'review_sheet.csv')
    fieldnames = ['sample_number', 'filename', 'claimed_code', 'ocr_raw',
                  'ocr_confidence', 'validation_reason', 'verified_correct',
                  'saved_crop_image(s)', 'box_ambiguous', 'box_count_mismatch',
                  'recorded_num_boxes', 'recomputed_num_boxes', 'box_idx']
    with open(sheet_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sheet_rows)

    n_ambig = sum(1 for r in sheet_rows if r['box_ambiguous'])
    n_mismatch = sum(1 for r in sheet_rows if r['box_count_mismatch'])
    print(f"  box_ambiguous: {n_ambig} | box_count_mismatch: {n_mismatch}")
    print(f"  Saved crops + review_sheet.csv to {out_dir}/")

print("\nDONE")
