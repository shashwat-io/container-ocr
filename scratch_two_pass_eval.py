#!/usr/bin/env python3
"""
DIAGNOSTIC ONLY — does NOT modify ocr.py.

Prototypes the "best valid wins" two-pass fix in-memory:
  1. Primary pass: real _try_extract(crop) with docpp ON (current default).
  2. ONLY if primary is invalid: retry the SAME crop via _try_extract(crop)
     again, this time with docpp OFF.
  3. If the docpp-OFF retry is valid, accept it, tag "Valid (docpp-off retry)".
     Otherwise keep the original (invalid) primary result.
  4. A docpp-ON valid primary result is NEVER touched by the fallback.

Implementation: monkeypatch run_ocr_on_crop (same FLAGS-toggle pattern as
scratch_docpp_eval.py) and monkeypatch _try_extract to a two-pass wrapper
around the ORIGINAL _try_extract (tier logic untouched). Since
_best_portrait_rotation and extract_container_code look up `_try_extract`
as a module global at call time, patching ocr_mod._try_extract makes both
call sites use the two-pass version automatically — no other code changes.

Measures against the same 153 incomplete + 100 currently-valid sample rows
(benchmark_results.csv, seed=42) used in yesterday's docpp-off net-impact eval.
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

ocr_mod.run_ocr_on_crop = patched_run_ocr_on_crop

_orig_try_extract = ocr_mod._try_extract

def two_pass_try_extract(crop):
    FLAGS['docpp_off'] = False
    code, valid, reason, raw, conf = _orig_try_extract(crop)
    if valid:
        return code, valid, reason, raw, conf
    FLAGS['docpp_off'] = True
    code2, valid2, reason2, raw2, conf2 = _orig_try_extract(crop)
    FLAGS['docpp_off'] = False  # always leave OFF-state reset before returning
    if valid2:
        return code2, valid2, "Valid (docpp-off retry)", raw2, conf2
    return code, valid, reason, raw, conf

ocr_mod._try_extract = two_pass_try_extract  # _best_portrait_rotation / extract_container_code resolve this global

def alnum(s): return re.sub(r'[^A-Za-z0-9]', '', s)
def total_chars(raw):
    return sum(len(alnum(p.split('(')[0])) for p in raw.split(' | ') if p.strip())

# ---- select rows (identical selection to scratch_docpp_eval.py) ----
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

def run_row(k):
    fname = k['filename']
    bxs = boxes_for(fname)
    if k['box_idx'] >= len(bxs):
        return None  # box-count drift
    with open(os.path.join(TEST_DIR, fname), 'rb') as fh:
        image_bytes = fh.read()
    FLAGS['docpp_off'] = False  # every row starts fresh on primary pass
    return ocr_mod.extract_container_code(image_bytes, *bxs[k['box_idx']])

# ---- 1) incomplete -> two-pass gains ----
print("\n=== PART 1: incomplete rows with two-pass (docpp ON, fallback OFF) ===", flush=True)
gains = []
still_invalid = 0
t0 = time.time()
for i, k in enumerate(incomplete, 1):
    res = run_row(k)
    if res is None:
        continue
    if res['is_valid']:
        gains.append((k['filename'], k['box_idx'], res['container_code'], res['validation_reason']))
    else:
        still_invalid += 1
    if i % 25 == 0:
        print(f"  [{i}/{len(incomplete)}] gains so far: {len(gains)}  ({time.time()-t0:.0f}s)", flush=True)

# ---- 2) valid sample -> confirm no regressions, no silent code changes ----
print("\n=== PART 2: currently-valid sample, two-pass re-run ===", flush=True)
still_pass = 0
regressions = []        # was valid in CSV, two-pass run now invalid
silent_changes = []     # was valid in CSV, two-pass valid but code DIFFERS
control_mismatch = []   # couldn't reproduce CSV-valid at all (non-determinism noise)
used_fallback_while_valid = []  # sanity check: valid rows where reason says "docpp-off retry" (should be 0 if primary already valid)
for i, k in enumerate(valid_sample, 1):
    res = run_row(k)
    if res is None:
        control_mismatch.append((k['filename'], k['box_idx'], 'box-drift'))
        continue
    if res['is_valid']:
        still_pass += 1
        if res['container_code'] != k['code']:
            silent_changes.append((k['filename'], k['box_idx'], k['code'], res['container_code'], res['validation_reason']))
        if 'docpp-off retry' in res['validation_reason']:
            used_fallback_while_valid.append((k['filename'], k['box_idx'], k['code']))
    else:
        regressions.append((k['filename'], k['box_idx'], k['code'], res['container_code'], res['validation_reason']))
    if i % 25 == 0:
        print(f"  [{i}/{len(valid_sample)}] still_pass={still_pass} regress={len(regressions)} ctrl_mismatch={len(control_mismatch)} ({time.time()-t0:.0f}s)", flush=True)

# ---- report ----
print("\n" + "="*70)
print("RESULTS — two-pass 'best valid wins' (docpp ON primary, docpp OFF fallback)")
print("="*70)
print(f"\nPART 1 GAINS — previously-incomplete now VALID: {len(gains)} / {len(incomplete)}")
for f, bi, code, reason in gains:
    print(f"  +  {f[:46]} [box {bi}] -> {code!r}  ({reason})")
print(f"PART 1 still invalid: {still_invalid}")

print(f"\nPART 2 REGRESSIONS (CSV valid, two-pass invalid): {len(regressions)}")
if regressions:
    for f, bi, oldcode, newcode, reason in regressions:
        print(f"  -  {f[:46]} [box {bi}] : {oldcode!r} -> {newcode!r}  ({reason})")
else:
    print("  0 regressions found")

print(f"\nPART 2 still-pass (CSV valid & two-pass valid): {still_pass}")
print(f"PART 2 SILENT CODE CHANGES (valid->valid but code differs): {len(silent_changes)}")
for f, bi, oldcode, newcode, reason in silent_changes:
    print(f"  ~  {f[:46]} [box {bi}] : {oldcode!r} -> {newcode!r}  ({reason})")

print(f"\nPART 2 sanity: valid rows where fallback pass fired despite primary already being valid: {len(used_fallback_while_valid)}")
for f, bi, code in used_fallback_while_valid:
    print(f"  !  {f[:46]} [box {bi}] : {code!r}  <-- BUG: primary-valid result was overridden")

print(f"\nPART 2 control mismatches (couldn't reproduce CSV-valid at all; excluded as noise): {len(control_mismatch)}")
for f, bi, why in control_mismatch:
    print(f"  ?  {f[:46]} [box {bi}] : {why}")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
print("DONE")
