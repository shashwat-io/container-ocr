#!/usr/bin/env python3
"""
READ-ONLY. No OCR/PaddleX calls, no ocr.py edits, no server restart.

Dedicated deep-dive sample of Tier3-Pass2 (check-digit-recovery via the
full-merged-string 10-char sliding window) rows, to firm up the ~83% (5/6)
error rate found in the last small review before deciding whether to
restrict/disable recovery specifically for this tier.

STEP 1: isolate the remaining unreviewed Tier3-Pass2 population (71 total in
        bucket b), excluding the 6 already visually reviewed:
          - sample10 from gain_samples/ (confirmed via classify() replay to
            be Tier3-Pass2, WRONG)
          - samples 13,14,15,16,17 from tier_b_verification_samples/ (all
            Tier3-Pass2, reviewed this round)
STEP 2: sample 15 more, split as evenly as available between docpp-ON
        primary and docpp-OFF retry origin. Seed 42, reused for consistency.
STEP 3: save full+crop image pairs + MANIFEST.txt to
        tier3_pass2_deepdive_samples/. No confidence column — Tier3-Pass2
        structurally has none (confirmed in the prior investigation: matches
        span a sliding window over concatenated multi-line text, not tied to
        a single OCR detection box).
"""
import csv, os, random, sys
import numpy as np
from PIL import Image
from collections import Counter

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')
from ultralytics import YOLO
from scratch_tier_classify import parse_texts, classify

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
NEW_CSV = 'benchmark_results.csv'
BASE_CSV = 'benchmark_results_pre_twopass.csv'
OUT_DIR = 'tier3_pass2_deepdive_samples'

os.makedirs(OUT_DIR, exist_ok=True)

def load_keyed(path):
    rows = list(csv.DictReader(open(path)))
    seen = Counter()
    keyed = {}
    for r in rows:
        f = r['filename']; idx = seen[f]; seen[f] += 1
        keyed[(f, idx)] = r
    return keyed

new_keyed = load_keyed(NEW_CSV)
base_keyed = load_keyed(BASE_CSV)

gains_keys = set()
for key, r in new_keyed.items():
    if r['is_valid'].strip().lower() == 'true':
        b = base_keyed[key]
        if b['is_valid'].strip().lower() != 'true':
            gains_keys.add(key)

# 6 already-reviewed Tier3-Pass2 filenames (verified via classify() replay,
# not just assumed from their gain-bucket label — sample10 was checked
# explicitly, was NOT double-counted with sample3 which is Tier1/literal)
ALREADY_REVIEWED_T3P2 = {
    'IMG_20221218_145339_ce_png_jpg.rf.e2a156f350f0c51f286a82bf2b30a27b.jpg',  # gain_samples sample10
    'IMG_0326_ce_png_jpg.rf.77150ef5f2ef7a661e303b6f409fa4c4.jpg',              # tier_b sample13
    '4654256381207060402_0160_jpg.rf.4913820186121d4fe9852895209eb845.jpg',    # tier_b sample14
    'Event20231020101456005-Avi_8155_jpg.rf.5f85da7a415d43341c89bdc836a0e0b2.jpg',  # tier_b sample15
    'IMG_0425_region_JPG_jpg.rf.c4e444f29b192053374eeaabf636c44b.jpg',          # tier_b sample16
    'Event20231020100956005-Avi_7595_jpg.rf.41179bff1dafc40b9bb7356c804c2684.jpg',  # tier_b sample17
}

# ---- STEP 1: full Tier3-Pass2 population, verify sample3 (gain_samples) is NOT Tier3-Pass2 ----
sample3_fname = '2305812729909048674_0160_jpg.rf.4e35798fa5efaa1342ee1ed5c94d257a.jpg'
r = new_keyed[(sample3_fname, 0)]
_, s3_bucket, s3_tier, _ = classify(parse_texts(r['ocr_raw']))
print(f"sample3 (gain_samples) tier check: bucket={s3_bucket} tier={s3_tier}  -> {'NOT Tier3-Pass2, no exclusion needed' if s3_tier != 'Tier3-Pass2' else 'IS Tier3-Pass2 -- MUST EXCLUDE'}")

population = []
for key, r in new_keyed.items():
    if r['is_valid'].strip().lower() != 'true':
        continue
    texts = parse_texts(r['ocr_raw'])
    code, bucket, tier, conf = classify(texts)
    if tier != 'Tier3-Pass2':
        continue
    fname, idx = key
    origin = 'docpp-OFF retry' if key in gains_keys else 'docpp-ON primary'
    population.append({'filename': fname, 'box_idx': idx, 'code': code, 'origin': origin})

print(f"\nTotal Tier3-Pass2 population: {len(population)}")
already_reviewed_in_pop = [p for p in population if p['filename'] in ALREADY_REVIEWED_T3P2]
print(f"Already-reviewed Tier3-Pass2 rows found in population: {len(already_reviewed_in_pop)} (expected 6)")
for p in already_reviewed_in_pop:
    print(f"  - {p['filename'][:50]} origin={p['origin']}")

candidates = [p for p in population if p['filename'] not in ALREADY_REVIEWED_T3P2]
print(f"\nRemaining unreviewed Tier3-Pass2 candidates: {len(candidates)}")
origin_pool = Counter(p['origin'] for p in candidates)
print(f"Candidate origin pool: {dict(origin_pool)}")

# ---- STEP 2: sample 15, evenly split ON/OFF where available ----
random.seed(42)
on_pool = [p for p in candidates if p['origin'] == 'docpp-ON primary']
off_pool = [p for p in candidates if p['origin'] == 'docpp-OFF retry']
random.shuffle(on_pool)
random.shuffle(off_pool)

TARGET = 15
half = TARGET // 2
off_take = min(half, len(off_pool))
on_take = min(TARGET - off_take, len(on_pool))
remaining = TARGET - off_take - on_take
if remaining > 0:
    extra_off = min(remaining, len(off_pool) - off_take)
    off_take += extra_off
    remaining -= extra_off
if remaining > 0:
    extra_on = min(remaining, len(on_pool) - on_take)
    on_take += extra_on
    remaining -= extra_on

sample = on_pool[:on_take] + off_pool[:off_take]
print(f"\nSample: target={TARGET} -> ON={on_take} OFF={off_take}")
print(f"Final sample size: {len(sample)}")

# ---- STEP 3: save images + manifest ----
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
    "Tier3-Pass2 deep-dive verification samples — full image + pipeline crop (padding=10), NO OCR run",
    "NOTE: Tier3-Pass2 matches span a sliding window over concatenated multi-line OCR text,",
    "not tied to a single detection box, so no per-line confidence is available for any row here",
    "(confirmed structural gap from the prior investigation).",
    "",
    f"Full Tier3-Pass2 population: {len(population)} | Excluded (already reviewed): 6 | Sample: {len(sample)}",
    f"Sample origin split: ON={on_take} OFF={off_take}",
    "",
]

for i, p in enumerate(sample, 1):
    fname = p['filename']
    bxs = boxes_for(fname)
    total_boxes = len(bxs)
    idx = p['box_idx']
    x1, y1, x2, y2 = bxs[idx]
    full_img = Image.open(os.path.join(TEST_DIR, fname)).convert('RGB')
    crop_img = crop_with_padding(full_img, x1, y1, x2, y2)

    full_img.save(os.path.join(OUT_DIR, f"sample{i}_full.jpg"))
    crop_img.save(os.path.join(OUT_DIR, f"sample{i}_crop.jpg"))

    manifest_lines.append(f"sample{i}")
    manifest_lines.append(f"  source image  : {fname}")
    manifest_lines.append(f"  box_idx       : {idx} of {total_boxes}")
    manifest_lines.append(f"  claimed code  : {p['code']}")
    manifest_lines.append(f"  docpp origin  : {p['origin']}")
    manifest_lines.append(f"  yolo box      : ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  crop=({int(x2-x1+20)}, {int(y2-y1+20)})")
    manifest_lines.append("")

with open(os.path.join(OUT_DIR, 'MANIFEST.txt'), 'w') as f:
    f.write('\n'.join(manifest_lines))

print(f"\nSaved {len(sample)} full+crop pairs and MANIFEST.txt to {OUT_DIR}/")
print("DONE")
