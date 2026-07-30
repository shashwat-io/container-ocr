#!/usr/bin/env python3
"""
READ-ONLY verification. No OCR/PaddleX calls. Does not touch ocr.py, does not
restart the server. Only reads benchmark CSVs + YOLO (for crop geometry) and
saves image crops for manual visual review.

STEP 1: full list of all invalid->valid gains (new 610/786 CSV vs
        benchmark_results_pre_twopass.csv baseline).
STEP 2: bucket each gain by its PRE-fix (old, invalid) failure category using
        the exact precedence rule reconstructed from AGENTS.md's documented
        349-invalid-row breakdown (153/61/36/99), calibrated to reproduce
        those counts exactly on benchmark_results_pre_twopass.csv:
          1. incomplete: total_chars(old_raw) < 11
          2. separated:  any single OLD raw text token, cleaned, exactly
                         matches a container type/size code (e.g. "45G1"):
                         ^\d{2}[A-Z]\d$
          3. merged:     not separated, but that type-code pattern appears
                         embedded inside a longer single OLD raw text token
          4. remainder:  neither of the above
STEP 3: sample 12 rows, capped at 2 from "incomplete", remainder split evenly
        across separated/merged/remainder (whichever have gains).
STEP 4: save full image + crop for each sampled row into gain_samples/, plus
        MANIFEST.txt.
"""
import csv, os, re, random, sys
import numpy as np
from PIL import Image
from collections import Counter

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')
from ultralytics import YOLO

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
NEW_CSV = 'benchmark_results.csv'
BASE_CSV = 'benchmark_results_pre_twopass.csv'
OUT_DIR = 'gain_samples'
FULL_LIST_PATH = 'gain_samples_full_list.txt'

os.makedirs(OUT_DIR, exist_ok=True)

def alnum(s): return re.sub(r'[^A-Za-z0-9]', '', s)
def clean(s): return alnum(s).upper()

def total_chars(raw):
    return sum(len(alnum(p.split('(')[0])) for p in raw.split(' | ') if p.strip())

def parse_texts(raw):
    """Parse 'TEXT(0.96) | TEXT2(0.95)' back into [(text, conf), ...]."""
    out = []
    for p in raw.split(' | '):
        p = p.strip()
        if not p:
            continue
        m = re.match(r'^(.*)\(([0-9.]+)\)$', p)
        if m:
            out.append((m.group(1), float(m.group(2))))
        else:
            out.append((p, None))
    return out

def avg_conf(raw):
    texts = parse_texts(raw)
    confs = [c for _, c in texts if c is not None]
    return sum(confs) / len(confs) if confs else None

TYPE_RE = re.compile(r'^\d{2}[A-Z]\d$')
TYPE_SUB = re.compile(r'\d{2}[A-Z]\d')

def classify_old_bucket(old_raw):
    if total_chars(old_raw) < 11:
        return 'incomplete'
    toks = [clean(t) for t, _ in parse_texts(old_raw)]
    if any(TYPE_RE.match(t) for t in toks if t):
        return 'separated'
    if any(len(t) > 4 and TYPE_SUB.search(t) for t in toks):
        return 'merged'
    return 'remainder'

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
assert set(base) == set(new), "row-key mismatch between baseline and new CSV"

# ---- STEP 1 + 2: full gains list with bucket tag ----
gains = []
for key in sorted(base):
    b = base[key]; n = new[key]
    if not b['is_valid'] and n['is_valid']:
        fname, idx = key
        bucket = classify_old_bucket(b['raw'])
        gains.append({
            'filename': fname,
            'box_idx': idx,
            'old_raw': b['raw'],
            'new_code': n['code'],
            'new_conf': avg_conf(n['raw']),
            'tag': 'Valid (docpp-off retry)',  # only new code path per Step A scope confirmation
            'bucket': bucket,
        })

print(f"Total gains: {len(gains)}")
bucket_counts = Counter(g['bucket'] for g in gains)
print("Gains by pre-fix bucket:", dict(bucket_counts))

with open(FULL_LIST_PATH, 'w') as f:
    f.write(f"Full gains list ({len(gains)} rows) — new (610/786) vs baseline (492/786, benchmark_results_pre_twopass.csv)\n")
    f.write(f"Bucket counts: {dict(bucket_counts)}\n\n")
    for g in gains:
        f.write(
            f"{g['filename']}\tbox={g['box_idx']}\tbucket={g['bucket']}\t"
            f"old_raw={g['old_raw']!r}\tnew_code={g['new_code']!r}\t"
            f"new_conf={g['new_conf']}\ttag={g['tag']!r}\n"
        )
print(f"Full list written to {FULL_LIST_PATH}")

# ---- STEP 3: sample 12, capped at 2 incomplete, rest split across the others ----
random.seed(42)
by_bucket = {b: [g for g in gains if g['bucket'] == b] for b in ('incomplete', 'separated', 'merged', 'remainder')}
for b in ('incomplete', 'separated', 'merged', 'remainder'):
    random.shuffle(by_bucket[b])  # deterministic given the single seed above

sample = []
incomplete_take = min(2, len(by_bucket['incomplete']))
sample += by_bucket['incomplete'][:incomplete_take]

remaining_slots = 12 - len(sample)
other_buckets = ['separated', 'merged', 'remainder']
avail = {b: by_bucket[b][:] for b in other_buckets}

# even split across buckets that actually have gains, rebalancing if one runs short
taken = {b: 0 for b in other_buckets}
while remaining_slots > 0 and any(len(avail[b]) > taken[b] for b in other_buckets):
    for b in other_buckets:
        if remaining_slots <= 0:
            break
        if len(avail[b]) > taken[b]:
            sample.append(avail[b][taken[b]])
            taken[b] += 1
            remaining_slots -= 1

print(f"\nSample size: {len(sample)}")
print("Sample bucket breakdown:", Counter(g['bucket'] for g in sample))

# ---- STEP 4: save full + crop images, write MANIFEST ----
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
    "Gain samples (NEW invalid->valid results) — full image + pipeline crop (padding=10), NO OCR run\n",
    f"Total gains: {len(gains)} | Sample: {len(sample)} (incomplete capped at 2)\n",
]

for i, g in enumerate(sample, 1):
    fname = g['filename']
    bxs = boxes_for(fname)
    total_boxes = len(bxs)
    idx = g['box_idx']
    x1, y1, x2, y2 = bxs[idx]
    full_img = Image.open(os.path.join(TEST_DIR, fname)).convert('RGB')
    crop_img = crop_with_padding(full_img, x1, y1, x2, y2)

    full_img.save(os.path.join(OUT_DIR, f"sample{i}_full.jpg"))
    crop_img.save(os.path.join(OUT_DIR, f"sample{i}_crop.jpg"))

    manifest_lines.append(f"sample{i}")
    manifest_lines.append(f"  source image  : {fname}")
    manifest_lines.append(f"  box_idx       : {idx} of {total_boxes}")
    manifest_lines.append(f"  original bucket: {g['bucket']}")
    manifest_lines.append(f"  OLD raw result: {g['old_raw']}")
    manifest_lines.append(f"  NEW claimed code: {g['new_code']}")
    manifest_lines.append(f"  NEW confidence  : {g['new_conf']}")
    manifest_lines.append(f"  tag           : {g['tag']}")
    manifest_lines.append(f"  yolo box      : ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  crop=({int(x2-x1+20)}, {int(y2-y1+20)})")
    manifest_lines.append("")

with open(os.path.join(OUT_DIR, 'MANIFEST.txt'), 'w') as f:
    f.write('\n'.join(manifest_lines))

print(f"\nSaved {len(sample)} full+crop pairs and MANIFEST.txt to {OUT_DIR}/")
print("DONE")
