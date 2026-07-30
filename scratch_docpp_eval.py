#!/usr/bin/env python3
"""
DIAGNOSTIC ONLY — does NOT modify ocr.py.
Monkeypatches run_ocr_on_crop in-memory to optionally disable PaddleX's
doc preprocessor (use_doc_orientation_classify / use_doc_unwarping) at
predict() time, keeping lsl at default and all other logic (incl.
_best_portrait_rotation, _post_process, Tiers 1-4) exactly as production.

Estimates net impact of docpp OFF:
  - 153 previously-incomplete invalid rows  -> count gains (now valid)
  - 100 random currently-valid rows         -> run docpp ON (control) AND
    docpp OFF, classify still-pass vs regression (ON valid & OFF invalid).
"""
import sys, os, csv, re, random, time
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

import app.inference.ocr as ocr_mod
from ultralytics import YOLO

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
CSV = 'benchmark_results.csv'

FLAGS = {'docpp_off': False}

def patched_run_ocr_on_crop(crop_image):
    ocr = ocr_mod.get_ocr()
    crop_np = np.array(crop_image)
    kwargs = {}
    if FLAGS['docpp_off']:
        kwargs = dict(use_doc_orientation_classify=False, use_doc_unwarping=False)
    with ocr_mod._ocr_lock:
        result = list(ocr.predict(crop_np, **kwargs))
    texts = []
    for r in result:
        for text, score in zip(r.get('rec_texts', []), r.get('rec_scores', [])):
            if text.strip():
                texts.append((text, float(score)))
    return texts

ocr_mod.run_ocr_on_crop = patched_run_ocr_on_crop  # _try_extract resolves this global

def alnum(s): return re.sub(r'[^A-Za-z0-9]', '', s)
def total_chars(raw):
    return sum(len(alnum(p.split('(')[0])) for p in raw.split(' | ') if p.strip())

# ---- select rows ----
from collections import Counter
rows = list(csv.DictReader(open(CSV)))
seen = Counter()
keyed = []
for r in rows:
    f = r['filename']; idx = seen[f]; seen[f] += 1
    keyed.append({'filename': f, 'box_idx': idx, 'is_valid': r['is_valid'].strip().lower()=='true',
                  'code': r['container_code'], 'raw': r['ocr_raw'], 'num_boxes': int(r['num_boxes'])})

incomplete = [k for k in keyed if (not k['is_valid']) and total_chars(k['raw']) < 11]
valid_rows = [k for k in keyed if k['is_valid']]
random.seed(42)
valid_sample = random.sample(valid_rows, 100)

print(f"incomplete (target gains): {len(incomplete)}")
print(f"currently-valid sample (regression test): {len(valid_sample)}", flush=True)

model = YOLO('models/best.pt')
_box_cache = {}
def boxes_for(fname):
    if fname in _box_cache: return _box_cache[fname]
    img = np.array(Image.open(os.path.join(TEST_DIR, fname)).convert('RGB'))
    res = model(img, verbose=False)
    bxs = [[float(v) for v in b.xyxy[0]] for r in res for b in r.boxes]
    _box_cache[fname] = bxs
    return bxs

def run_row(k, docpp_off):
    fname = k['filename']
    bxs = boxes_for(fname)
    if k['box_idx'] >= len(bxs):
        return None  # box-count drift
    with open(os.path.join(TEST_DIR, fname), 'rb') as fh:
        image_bytes = fh.read()
    FLAGS['docpp_off'] = docpp_off
    return ocr_mod.extract_container_code(image_bytes, *bxs[k['box_idx']])

# ---- 1) incomplete -> docpp OFF gains ----
print("\n=== PART 1: incomplete rows with docpp OFF ===", flush=True)
gains = []
t0 = time.time()
for i, k in enumerate(incomplete, 1):
    res = run_row(k, docpp_off=True)
    if res is None:
        continue
    if res['is_valid']:
        gains.append((k['filename'], k['box_idx'], res['container_code'], res['validation_reason']))
    if i % 25 == 0:
        print(f"  [{i}/{len(incomplete)}] gains so far: {len(gains)}  ({time.time()-t0:.0f}s)", flush=True)

# ---- 2) valid sample -> ON (control) + OFF ----
print("\n=== PART 2: currently-valid sample, docpp ON (control) vs OFF ===", flush=True)
still_pass = 0
regressions = []        # ON valid & OFF invalid  (docpp-attributable)
control_mismatch = []   # ON invalid (couldn't reproduce CSV valid) -> noise, excluded
both_changed_code = []  # ON valid & OFF valid but code differs
for i, k in enumerate(valid_sample, 1):
    on = run_row(k, docpp_off=False)
    off = run_row(k, docpp_off=True)
    if on is None or off is None:
        control_mismatch.append((k['filename'], k['box_idx'], 'box-drift'))
        continue
    if not on['is_valid']:
        control_mismatch.append((k['filename'], k['box_idx'], f"ON-invalid({on['container_code']!r})"))
        continue
    # ON is valid (control reproduces) -> now judge OFF
    if off['is_valid']:
        still_pass += 1
        if off['container_code'] != on['container_code']:
            both_changed_code.append((k['filename'], k['box_idx'], on['container_code'], off['container_code']))
    else:
        regressions.append((k['filename'], k['box_idx'], on['container_code'], off['container_code'], off['validation_reason']))
    if i % 25 == 0:
        print(f"  [{i}/{len(valid_sample)}] still_pass={still_pass} regress={len(regressions)} ctrl_mismatch={len(control_mismatch)} ({time.time()-t0:.0f}s)", flush=True)

# ---- report ----
print("\n" + "="*70)
print("RESULTS — docpp OFF (lsl default, all other logic unchanged)")
print("="*70)
print(f"\nPART 1 GAINS — previously-incomplete now VALID: {len(gains)} / {len(incomplete)}")
for f, bi, code, reason in gains:
    print(f"  +  {f[:46]} [box {bi}] -> {code!r}  ({reason})")

print(f"\nPART 2 REGRESSIONS (control ON valid, OFF invalid): {len(regressions)}")
if regressions:
    for f, bi, oncode, offcode, reason in regressions:
        print(f"  -  {f[:46]} [box {bi}] : {oncode!r} -> {offcode!r}  ({reason})")
else:
    print("  0 regressions found")

print(f"\nPART 2 still-pass (ON valid & OFF valid): {still_pass}")
print(f"PART 2 valid->valid but CODE changed: {len(both_changed_code)}")
for f, bi, oncode, offcode in both_changed_code:
    print(f"  ~  {f[:46]} [box {bi}] : {oncode!r} -> {offcode!r}")

print(f"\nPART 2 control mismatches (ON didn't reproduce CSV-valid; excluded as noise): {len(control_mismatch)}")
for f, bi, why in control_mismatch:
    print(f"  ?  {f[:46]} [box {bi}] : {why}")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("DONE")
