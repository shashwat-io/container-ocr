#!/usr/bin/env python3
"""
READ-ONLY. No OCR/PaddleX calls, no ocr.py edits, no server restart.

Pulls the 29 "different code" rows from today's targeted 71-row Tier3-Pass2
re-run (scratch_targeted_benchmark_results.csv) vs the earlier-today baseline
(benchmark_results.csv), samples 10 (seed=42), and saves full+crop image
pairs + MANIFEST.txt with both candidate codes side by side for manual review.
"""
import csv, os, random, sys
import numpy as np
from PIL import Image
from collections import Counter

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')
from ultralytics import YOLO

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
RERUN_CSV = 'scratch_targeted_benchmark_results.csv'
BASE_CSV = 'benchmark_results.csv'
PRE_OFF_CSV = 'benchmark_results_pre_twopass.csv'
OUT_DIR = 'instability_check_samples'

os.makedirs(OUT_DIR, exist_ok=True)

def load_keyed(path):
    rows = list(csv.DictReader(open(path)))
    seen = Counter()
    keyed = {}
    for r in rows:
        f = r['filename']; idx = seen[f]; seen[f] += 1
        keyed[(f, idx)] = r
    return keyed

rerun_keyed = load_keyed(RERUN_CSV)
base_keyed = load_keyed(BASE_CSV)
pre_off_keyed = load_keyed(PRE_OFF_CSV)

rerun_rows = list(csv.DictReader(open(RERUN_CSV)))
tier3pass2_rows = [r for r in rerun_rows if r['group'] == 'tier3pass2']

diff_code_rows = []
for r in tier3pass2_rows:
    key = (r['filename'], int(r['box_idx']))
    if r['is_valid'].strip().lower() != 'true':
        continue  # only the "valid" rescues matter here
    earlier_code = base_keyed[key]['container_code']
    rerun_code = r['container_code']
    if earlier_code != rerun_code:
        was_on_sufficient = pre_off_keyed[key]['is_valid'].strip().lower() == 'true'
        origin = 'docpp-ON primary (was sufficient before)' if was_on_sufficient else 'docpp-OFF retry (already needed before)'
        diff_code_rows.append({
            'filename': r['filename'],
            'box_idx': int(r['box_idx']),
            'earlier_code': earlier_code,
            'rerun_code': rerun_code,
            'origin': origin,
        })

print(f"Total 'different code' rows found: {len(diff_code_rows)} (expected 29)")
origin_counts = Counter(d['origin'] for d in diff_code_rows)
print(f"Origin composition: {dict(origin_counts)}")

random.seed(42)
sample = random.sample(diff_code_rows, min(10, len(diff_code_rows)))
sample_origin_counts = Counter(d['origin'] for d in sample)
print(f"\nSample size: {len(sample)}")
print(f"Sample origin composition: {dict(sample_origin_counts)}")

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

def crop_with_padding(image, x1, y1, x2, y2, padding=10):
    w, h = image.size
    left   = max(0, int(x1) - padding)
    top    = max(0, int(y1) - padding)
    right  = min(w, int(x2) + padding)
    bottom = min(h, int(y2) + padding)
    return image.crop((left, top, right, bottom))

manifest_lines = [
    "Instability check samples — comparing EARLIER (610/786 baseline run) vs",
    "TODAY'S RE-RUN (targeted 71-row Tier3-Pass2 test) claimed codes for the same crop.",
    "Full image + pipeline crop (padding=10), NO fresh OCR run for this script.",
    "",
    f"Total 'different code' rows: {len(diff_code_rows)} | Sample: {len(sample)}",
    f"Sample origin composition: {dict(sample_origin_counts)}",
    "",
]

for i, d in enumerate(sample, 1):
    fname = d['filename']
    bxs = boxes_for(fname)
    total_boxes = len(bxs)
    idx = d['box_idx']
    x1, y1, x2, y2 = bxs[idx]
    full_img = Image.open(os.path.join(TEST_DIR, fname)).convert('RGB')
    crop_img = crop_with_padding(full_img, x1, y1, x2, y2)

    full_img.save(os.path.join(OUT_DIR, f"sample{i}_full.jpg"))
    crop_img.save(os.path.join(OUT_DIR, f"sample{i}_crop.jpg"))

    manifest_lines.append(f"sample{i}")
    manifest_lines.append(f"  source image   : {fname}")
    manifest_lines.append(f"  box_idx        : {idx} of {total_boxes}")
    manifest_lines.append(f"  docpp origin   : {d['origin']}")
    manifest_lines.append(f"  EARLIER code   : {d['earlier_code']}")
    manifest_lines.append(f"  RERUN code     : {d['rerun_code']}")
    manifest_lines.append(f"  match?         : {'SAME' if d['earlier_code']==d['rerun_code'] else 'DIFFERENT'}")
    manifest_lines.append(f"  yolo box       : ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  crop=({int(x2-x1+20)}, {int(y2-y1+20)})")
    manifest_lines.append("")

with open(os.path.join(OUT_DIR, 'MANIFEST.txt'), 'w') as f:
    f.write('\n'.join(manifest_lines))

print(f"\nSaved {len(sample)} full+crop pairs and MANIFEST.txt to {OUT_DIR}/")
print("DONE")
