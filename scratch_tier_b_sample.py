#!/usr/bin/env python3
"""
READ-ONLY. No OCR/PaddleX calls, no ocr.py edits, no server restart.

STEP 1: Build the full bucket-b (check-digit-recovered) population with
        metadata, reusing the tier-replay classifier from
        scratch_tier_classify.py (already verified: 0 mismatches replaying
        all 610 valid rows' actual recorded codes).
STEP 2: Stratified sample of 20 rows, excluding the 6 already-reviewed
        bucket-b rows (samples 1,5,7,8,10,11), proportional to tier
        composition (Tier1b ~61% -> 12, Tier3-Pass2 ~27% -> 5,
        Tier2-recovery ~12% -> 3), each tier split as evenly as available
        between docpp-ON-primary and docpp-OFF-retry origin. Seed=42.
STEP 3: Save full+crop image pairs + MANIFEST.txt to
        tier_b_verification_samples/.
"""
import csv, os, re, random, sys
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
OUT_DIR = 'tier_b_verification_samples'
FULL_LIST_PATH = 'tier_b_population_full_list.txt'

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

# already-reviewed bucket-b filenames (samples 1,5,7,8,10,11 from gain_samples review)
ALREADY_REVIEWED = {
    'IMG_0501_ce_png_jpg.rf.c9b3f0bad3c8ef1bde48b3d35656cc65.jpg',
    'IMG_20221218_150316_region_jpg.rf.2bcab07863457bab95276a15cdafb1e3.jpg',
    '6427132303892274193_0270_jpg.rf.be89aed792b915e1fea5d306e08717ff.jpg',
    'IMG_0541_ce_png_jpg.rf.5d87c1d8cd92e9b99088a50b4004ef6c.jpg',
    'IMG_20221218_145339_ce_png_jpg.rf.e2a156f350f0c51f286a82bf2b30a27b.jpg',
    '1480134690284023198_0130_jpg.rf.ff3bbc9ba0e8d23f8f1436ee0a516e2f.jpg',
}

def flatten_conf(conf):
    if conf is None:
        return None
    if isinstance(conf, tuple):
        vals = [c for c in conf if c is not None]
        return sum(vals) / len(vals) if vals else None
    return conf

# ---- STEP 1: full bucket-b population ----
population = []
for key, r in new_keyed.items():
    if r['is_valid'].strip().lower() != 'true':
        continue
    texts = parse_texts(r['ocr_raw'])
    code, bucket, tier, conf = classify(texts)
    if bucket != 'check_digit_recovered':
        continue
    fname, idx = key
    origin = 'docpp-OFF retry' if key in gains_keys else 'docpp-ON primary'
    population.append({
        'filename': fname,
        'box_idx': idx,
        'tier': tier,
        'code': code,
        'origin': origin,
        'conf': flatten_conf(conf),
        'already_reviewed': fname in ALREADY_REVIEWED,
    })

print(f"Total bucket-b population: {len(population)}")
tier_counts = Counter(p['tier'] for p in population)
origin_counts = Counter(p['origin'] for p in population)
print(f"Tier composition: {dict(tier_counts)}")
print(f"Origin composition: {dict(origin_counts)}")

with open(FULL_LIST_PATH, 'w') as f:
    f.write(f"Full check-digit-recovered (bucket b) population ({len(population)} rows)\n")
    f.write(f"Tier composition: {dict(tier_counts)}\n")
    f.write(f"Origin composition: {dict(origin_counts)}\n\n")
    for p in population:
        f.write(
            f"{p['filename']}\tbox={p['box_idx']}\ttier={p['tier']}\tcode={p['code']!r}\t"
            f"origin={p['origin']}\tconf={p['conf']}\talready_reviewed={p['already_reviewed']}\n"
        )
print(f"Full population list written to {FULL_LIST_PATH}")

# ---- STEP 2: stratified sample of 20, excluding already-reviewed ----
candidates = [p for p in population if not p['already_reviewed']]
print(f"\nCandidates after excluding {len(population) - len(candidates)} already-reviewed: {len(candidates)}")

TIER_TARGETS = {'Tier1b': 12, 'Tier3-Pass2': 5, 'Tier2-recovery': 3}
assert sum(TIER_TARGETS.values()) == 20

random.seed(42)
sample = []
tally = Counter()

for tier, target in TIER_TARGETS.items():
    tier_pool = [p for p in candidates if p['tier'] == tier]
    on_pool = [p for p in tier_pool if p['origin'] == 'docpp-ON primary']
    off_pool = [p for p in tier_pool if p['origin'] == 'docpp-OFF retry']
    random.shuffle(on_pool)
    random.shuffle(off_pool)

    # split target as evenly as possible between origins, capped by availability
    half = target // 2
    off_take = min(half, len(off_pool))
    on_take = min(target - off_take, len(on_pool))
    # if ON pool ran short, backfill from OFF; if OFF pool ran short, backfill from ON
    remaining = target - off_take - on_take
    if remaining > 0:
        extra_off = min(remaining, len(off_pool) - off_take)
        off_take += extra_off
        remaining -= extra_off
    if remaining > 0:
        extra_on = min(remaining, len(on_pool) - on_take)
        on_take += extra_on
        remaining -= extra_on

    picked = on_pool[:on_take] + off_pool[:off_take]
    sample.extend(picked)
    print(f"  {tier}: target={target} -> ON={on_take} OFF={off_take} (pool ON={len(on_pool)} OFF={len(off_pool)})")

print(f"\nFinal sample size: {len(sample)}")
tier_tally = Counter(s['tier'] for s in sample)
origin_tally = Counter(s['origin'] for s in sample)
print(f"Sample tier tally: {dict(tier_tally)}")
print(f"Sample origin tally: {dict(origin_tally)}")

# ---- STEP 3: save full+crop images + MANIFEST ----
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
    "Tier-B (check-digit-recovered) verification samples — full image + pipeline crop (padding=10), NO OCR run\n",
    f"Full bucket-b population: {len(population)} | Sample: {len(sample)} (excludes 6 already-reviewed)\n",
    f"Sample tier tally: {dict(tier_tally)} | Sample origin tally: {dict(origin_tally)}\n",
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
    manifest_lines.append(f"  source image      : {fname}")
    manifest_lines.append(f"  box_idx           : {idx} of {total_boxes}")
    manifest_lines.append(f"  tier              : {p['tier']}")
    manifest_lines.append(f"  claimed code      : {p['code']}")
    manifest_lines.append(f"  docpp origin      : {p['origin']}")
    manifest_lines.append(f"  owner-line conf   : {p['conf']}")
    manifest_lines.append(f"  yolo box          : ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  crop=({int(x2-x1+20)}, {int(y2-y1+20)})")
    manifest_lines.append("")

with open(os.path.join(OUT_DIR, 'MANIFEST.txt'), 'w') as f:
    f.write('\n'.join(manifest_lines))

print(f"\nSaved {len(sample)} full+crop pairs and MANIFEST.txt to {OUT_DIR}/")
print("DONE")
