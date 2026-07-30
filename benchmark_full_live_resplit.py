import os
import csv
import time
import sys
import logging
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from ultralytics import YOLO
from app.inference.ocr import extract_container_code

# ── The only config swapped vs. benchmark_full_live.py ────────────────────
# TEST_DIR now points at the leakage-clean resplit test set (confirmed 0
# group_id overlap across train/valid/test via scripts/leakage_check.py).
# MODEL_PATH/OUTPUT_CSV/SAMPLE_FILE are env-var overridable so this one
# script can run both the new (resplit-trained) and old (contaminated-split)
# detector against the IDENTICAL image set for an apples-to-apples
# detector-only comparison -- nothing else in the script (OCR side, tier
# logic, tier-capture instrumentation) differs from benchmark_full_live.py.
TEST_DIR = '/home/shashwatkumar/Testing/container_ocr_test/container.v2i.yolov11.resplit/test/images'
MODEL_PATH = os.environ.get('BENCH_MODEL_PATH', 'models/best_resplit.pt')
OUTPUT_CSV = os.environ.get('BENCH_OUTPUT_CSV',
                             '/home/shashwatkumar/Testing/container_ocr_test/benchmark_results_full_live_resplit.csv')
SAMPLE_FILE = os.environ.get('BENCH_SAMPLE_FILE')  # optional: restrict to a quick-sample list
CHECKPOINT_EVERY = 100

# ── Tier attribution via log capture ──────────────────────────────────────
# extract_container_code doesn't return a tier field directly, but each
# resolution path in ocr.py logs a distinctive "Tier N (...)" message right
# before it returns. We attach a handler to that logger and, per image,
# take the last such message emitted as the resolving tier. This is
# read-only instrumentation — no tier logic in ocr.py is touched.
_last_tier_msg = {'msg': None}

class TierCaptureHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith('Tier ') or msg.startswith('Portrait:') or msg.startswith('Landscape:'):
            if msg.startswith('Tier '):
                _last_tier_msg['msg'] = msg

ocr_logger = logging.getLogger('app.inference.ocr')
ocr_logger.setLevel(logging.INFO)
ocr_logger.addHandler(TierCaptureHandler())
ocr_logger.propagate = False

TIER_LABELS = [
    ('Tier 1 (single-line exact)', 'Tier1-exact'),
    ('Tier 1b (single-line check-digit recovery)', 'Tier1b-recovery'),
    ('Tier 2 (adjacent-pair 11-char)', 'Tier2-exact'),
    ('Tier 2 (adjacent-pair check-digit)', 'Tier2-recovery'),
    ('Tier 3 (merged Pass 1)', 'Tier3-Pass1'),
    ('Tier 3 (merged fallback, no valid)', 'Tier3-fallback-novalid'),
    ('Tier 4 (confusion correction)', 'Tier4-confusion'),
]

def classify_tier(reason, tier_msg):
    if tier_msg:
        for prefix, label in TIER_LABELS:
            if tier_msg.startswith(prefix):
                # Tier4 log fires in addition to a Tier3-fallback-novalid log
                # right before it (best_code assembly); Tier 4 is the true
                # resolver when present, so no override needed since it's
                # always the LAST message logged when it fires.
                return label
    if not reason:
        return 'none'
    if 'docpp-off retry' in reason:
        return 'docpp-off-retry'
    if 'character confusion corrected' in reason:
        return 'Tier4-confusion'
    if 'check-digit recovered' in reason:
        return 'check-digit-recovery'
    if reason == 'Valid':
        return 'literal-match'
    return 'unresolved'


model = YOLO(MODEL_PATH)

images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if SAMPLE_FILE:
    with open(SAMPLE_FILE) as f:
        sample_names = {l.strip() for l in f if l.strip() and not l.startswith('#')}
    images = [f for f in images if f in sample_names]
    missing = sample_names - set(images)
    if missing:
        print(f"WARNING: {len(missing)} sample filenames not found in {TEST_DIR}: {sorted(missing)[:5]}...", flush=True)

total = len(images)
print(f"Found {total} images (model={MODEL_PATH}, test_dir={TEST_DIR}, sample_file={SAMPLE_FILE})", flush=True)

stats = {'detected': 0, 'ocr_read': 0, 'valid': 0, 'no_detection': 0, 'invalid': 0}
tier_pass_counts = {}
start = time.time()
rows = []

fieldnames = ['filename', 'detected', 'num_boxes', 'container_code', 'is_valid',
              'validation_reason', 'tier', 'ocr_confidence', 'ocr_raw']

def write_csv():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

for i, fname in enumerate(images, 1):
    path = os.path.join(TEST_DIR, fname)
    try:
        img_pil = Image.open(path).convert('RGB')
        img_np = np.array(img_pil)
        with open(path, 'rb') as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"  Load error {fname}: {e}", flush=True)
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'validation_reason': f'load error: {e}',
                     'tier': 'none', 'ocr_confidence': 0.0, 'ocr_raw': ''})
        continue

    try:
        results = model(img_np, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                boxes.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"  YOLO error {fname}: {e}", flush=True)
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'validation_reason': f'YOLO error: {e}',
                     'tier': 'none', 'ocr_confidence': 0.0, 'ocr_raw': ''})
        continue

    if not boxes:
        stats['no_detection'] += 1
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'validation_reason': 'No detection',
                     'tier': 'none', 'ocr_confidence': 0.0, 'ocr_raw': ''})
    else:
        stats['detected'] += 1
        for x1, y1, x2, y2 in boxes:
            _last_tier_msg['msg'] = None
            try:
                result = extract_container_code(image_bytes, x1, y1, x2, y2)
                tier = classify_tier(result['validation_reason'], _last_tier_msg['msg'])
                if result['container_code']:
                    stats['ocr_read'] += 1
                if result['is_valid']:
                    stats['valid'] += 1
                    tier_pass_counts[tier] = tier_pass_counts.get(tier, 0) + 1
                else:
                    stats['invalid'] += 1
                rows.append({
                    'filename': fname,
                    'detected': True,
                    'num_boxes': len(boxes),
                    'container_code': result['container_code'],
                    'is_valid': result['is_valid'],
                    'validation_reason': result['validation_reason'],
                    'tier': tier,
                    'ocr_confidence': result['ocr_confidence'],
                    'ocr_raw': result['ocr_raw_output'],
                })
            except Exception as e:
                stats['invalid'] += 1
                rows.append({'filename': fname, 'detected': True, 'num_boxes': len(boxes),
                             'container_code': '', 'is_valid': False, 'validation_reason': f'OCR error: {e}',
                             'tier': 'none', 'ocr_confidence': 0.0, 'ocr_raw': ''})

    if i % CHECKPOINT_EVERY == 0 or i == total:
        elapsed = time.time() - start
        write_csv()
        tier_summary = ', '.join(f"{k}:{v}" for k, v in sorted(tier_pass_counts.items()))
        print(
            f"[CHECKPOINT {i}/{total}] {elapsed:.0f}s elapsed | avg {elapsed/i:.1f}s/img | "
            f"pass so far: {stats['valid']} | fail so far: {stats['invalid'] + stats['no_detection']} | "
            f"tiers: {tier_summary} | checkpoint CSV written ({len(rows)} rows)",
            flush=True
        )

elapsed = time.time() - start
write_csv()

d = stats['detected'] or 1
print(f"\n=== BENCHMARK RESULTS (LIVE, RESPLIT) ===", flush=True)
print(f"Model:            {MODEL_PATH}", flush=True)
print(f"Total images:     {total}", flush=True)
print(f"YOLO detected:    {stats['detected']} ({stats['detected']/total*100:.1f}%)", flush=True)
print(f"OCR read code:    {stats['ocr_read']} ({stats['ocr_read']/d*100:.1f}% of detected)", flush=True)
print(f"ISO 6346 valid:   {stats['valid']} / {total}", flush=True)
print(f"No detection:     {stats['no_detection']} ({stats['no_detection']/total*100:.1f}%)", flush=True)
print(f"\n--- Tier breakdown (passing rows only) ---", flush=True)
for k, v in sorted(tier_pass_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v}", flush=True)
print(f"\nTotal time:       {elapsed:.0f}s | Avg per image: {elapsed/total:.1f}s", flush=True)
print(f"Results saved to: {OUTPUT_CSV}", flush=True)
print(f"RUN COMPLETED UNINTERRUPTED", flush=True)
