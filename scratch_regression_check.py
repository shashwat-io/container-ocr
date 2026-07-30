#!/usr/bin/env python3
"""
READ-ONLY verification. Does not touch ocr.py, does not invoke PaddleX/OCR.
Compares benchmark_results.csv (new, two-pass) against
benchmark_results_pre_twopass.csv (baseline, single-pass docpp-ON) row by row,
keyed by (filename, box_idx-within-filename), split by crop orientation
(portrait/landscape), determined the same way ocr.py does: crop h > w.
Orientation is recomputed via the same YOLO model + bbox geometry used by the
benchmark (no OCR/PaddleX call needed).
"""
import csv, os, sys
import numpy as np
from PIL import Image
from collections import Counter

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')
from ultralytics import YOLO

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
NEW_CSV = 'benchmark_results.csv'
BASE_CSV = 'benchmark_results_pre_twopass.csv'

def load_keyed(path):
    rows = list(csv.DictReader(open(path)))
    seen = Counter()
    keyed = {}
    for r in rows:
        f = r['filename']; idx = seen[f]; seen[f] += 1
        keyed[(f, idx)] = {
            'is_valid': r['is_valid'].strip().lower() == 'true',
            'code': r['container_code'],
            'raw': r['ocr_raw'],
        }
    return keyed

base = load_keyed(BASE_CSV)
new = load_keyed(NEW_CSV)

print(f"baseline rows: {len(base)}  new rows: {len(new)}")
missing_in_new = set(base) - set(new)
missing_in_base = set(new) - set(base)
print(f"keys only in baseline: {len(missing_in_new)}  keys only in new: {len(missing_in_base)}")

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

def is_portrait(fname, idx):
    bxs = boxes_for(fname)
    if idx >= len(bxs):
        return None  # box drift, shouldn't happen given equal row counts
    x1, y1, x2, y2 = bxs[idx]
    w, h = x2 - x1, y2 - y1
    return h > w

categories = {
    'landscape': {'gain': [], 'regression': [], 'silent_change_valid_valid': [], 'silent_change_invalid_invalid': [], 'unchanged_valid': 0, 'unchanged_invalid': 0},
    'portrait':  {'gain': [], 'regression': [], 'silent_change_valid_valid': [], 'silent_change_invalid_invalid': [], 'unchanged_valid': 0, 'unchanged_invalid': 0},
}
orientation_unknown = []

common_keys = set(base) & set(new)
for key in sorted(common_keys):
    fname, idx = key
    port = is_portrait(fname, idx)
    if port is None:
        orientation_unknown.append(key)
        continue
    orient = 'portrait' if port else 'landscape'
    b = base[key]; n = new[key]
    cat = categories[orient]
    if not b['is_valid'] and n['is_valid']:
        cat['gain'].append((fname, idx, n['code'], n['raw'][-60:]))
    elif b['is_valid'] and not n['is_valid']:
        cat['regression'].append((fname, idx, b['code'], n['code']))
    elif b['is_valid'] and n['is_valid']:
        if b['code'] != n['code']:
            cat['silent_change_valid_valid'].append((fname, idx, b['code'], n['code']))
        else:
            cat['unchanged_valid'] += 1
    else:
        # both invalid
        if b['code'] != n['code']:
            cat['silent_change_invalid_invalid'].append((fname, idx, b['code'], n['code']))
        else:
            cat['unchanged_invalid'] += 1

for orient in ('landscape', 'portrait'):
    cat = categories[orient]
    print(f"\n{'='*70}\n{orient.upper()} rows\n{'='*70}")
    print(f"  gains (invalid->valid):            {len(cat['gain'])}")
    print(f"  regressions (valid->invalid):       {len(cat['regression'])}")
    print(f"  silent changes (valid->diff valid): {len(cat['silent_change_valid_valid'])}")
    print(f"  code changed but still invalid:     {len(cat['silent_change_invalid_invalid'])}")
    print(f"  unchanged valid:                    {cat['unchanged_valid']}")
    print(f"  unchanged invalid:                  {cat['unchanged_invalid']}")

    if cat['regression']:
        print(f"\n  -- REGRESSIONS --")
        for f, bi, oldc, newc in cat['regression']:
            print(f"    - {f[:46]} [box {bi}] : {oldc!r} -> {newc!r}")

    if cat['silent_change_valid_valid']:
        print(f"\n  -- SILENT VALID->DIFFERENT-VALID CHANGES --")
        for f, bi, oldc, newc in cat['silent_change_valid_valid']:
            print(f"    ~ {f[:46]} [box {bi}] : {oldc!r} -> {newc!r}")

    if cat['gain']:
        print(f"\n  -- GAINS (sample, up to 10) --")
        for f, bi, code, _ in cat['gain'][:10]:
            print(f"    + {f[:46]} [box {bi}] -> {code!r}")

print(f"\norientation-unknown (box drift, should be 0): {len(orientation_unknown)}")
print("\nDONE")
