"""
Quick-sample (157-image) live benchmark run to verify the is_recovery_based
flag added to app/inference/ocr.py::extract_container_code() / threaded
through app/inference/predictor.py::_real_predict earlier today.

Read-only verification: does NOT change tier-matching logic, retry logic, or
which OCR result wins. Adds one additional captured field
(is_recovery_based) to the per-row CSV output, pulled directly from the same
result dict key predictor.py reads (result['is_recovery_based']).

Reuses quick_sample_set.txt (fixed, seed=42, 157-image stratified sample --
see that file's header comment) instead of regenerating a sample. Mirrors
benchmark_full_live.py's tier-capture instrumentation (known to mislabel the
live tier for ~36% of passing rows per AGENTS.md / replay_tier_attribution.py
-- does not affect is_valid pass/fail counts, only the tier label) so the
per-tier breakdown here can be sanity-checked against
benchmark_results_full_live_corrected_tiers.csv.

Does not modify benchmark.py or benchmark_full_live.py. Writes its own
separate output CSV.
"""
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

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
SAMPLE_LIST = '/home/shashwatkumar/Testing/container_ocr_test/quick_sample_set.txt'
OUTPUT_CSV = '/home/shashwatkumar/Testing/container_ocr_test/quick_sample_tier1b_verification.csv'
CHECKPOINT_EVERY = 25

# ── Tier attribution via log capture (mirrors benchmark_full_live.py) ──────
_last_tier_msg = {'msg': None}

class TierCaptureHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
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


with open(SAMPLE_LIST) as f:
    images = [line.strip() for line in f if line.strip() and not line.startswith('#')]

total = len(images)
print(f"Loaded {total} images from {SAMPLE_LIST}", flush=True)

model = YOLO('models/best.pt')

stats = {'detected': 0, 'ocr_read': 0, 'valid': 0, 'no_detection': 0, 'invalid': 0}
tier_pass_counts = {}
recovery_count = 0
start = time.time()
rows = []

fieldnames = ['filename', 'detected', 'num_boxes', 'container_code', 'is_valid',
              'validation_reason', 'tier', 'is_recovery_based', 'ocr_confidence', 'ocr_raw']

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
                     'tier': 'none', 'is_recovery_based': False, 'ocr_confidence': 0.0, 'ocr_raw': ''})
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
                     'tier': 'none', 'is_recovery_based': False, 'ocr_confidence': 0.0, 'ocr_raw': ''})
        continue

    if not boxes:
        stats['no_detection'] += 1
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'validation_reason': 'No detection',
                     'tier': 'none', 'is_recovery_based': False, 'ocr_confidence': 0.0, 'ocr_raw': ''})
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
                    if result['is_recovery_based']:
                        recovery_count += 1
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
                    'is_recovery_based': result['is_recovery_based'],
                    'ocr_confidence': result['ocr_confidence'],
                    'ocr_raw': result['ocr_raw_output'],
                })
            except Exception as e:
                stats['invalid'] += 1
                rows.append({'filename': fname, 'detected': True, 'num_boxes': len(boxes),
                             'container_code': '', 'is_valid': False, 'validation_reason': f'OCR error: {e}',
                             'tier': 'none', 'is_recovery_based': False, 'ocr_confidence': 0.0, 'ocr_raw': ''})

    if i % CHECKPOINT_EVERY == 0 or i == total:
        elapsed = time.time() - start
        write_csv()
        tier_summary = ', '.join(f"{k}:{v}" for k, v in sorted(tier_pass_counts.items()))
        print(
            f"[CHECKPOINT {i}/{total}] {elapsed:.0f}s elapsed | avg {elapsed/i:.1f}s/img | "
            f"pass so far: {stats['valid']} | fail so far: {stats['invalid'] + stats['no_detection']} | "
            f"recovery-based so far: {recovery_count} | "
            f"tiers: {tier_summary} | checkpoint CSV written ({len(rows)} rows)",
            flush=True
        )

elapsed = time.time() - start
write_csv()

d = stats['detected'] or 1
v = stats['valid'] or 1
print(f"\n=== QUICK-SAMPLE BENCHMARK RESULTS (157-image, is_recovery_based verification) ===", flush=True)
print(f"Total images:        {total}", flush=True)
print(f"YOLO detected:       {stats['detected']} ({stats['detected']/total*100:.1f}%)", flush=True)
print(f"OCR read code:       {stats['ocr_read']} ({stats['ocr_read']/d*100:.1f}% of detected)", flush=True)
print(f"ISO 6346 valid:      {stats['valid']} / {total}", flush=True)
print(f"No detection:        {stats['no_detection']} ({stats['no_detection']/total*100:.1f}%)", flush=True)
print(f"Recovery-based:      {recovery_count} / {stats['valid']} valid rows ({recovery_count/v*100:.1f}%)", flush=True)
print(f"\n--- Tier breakdown (passing rows only) ---", flush=True)
for k, val in sorted(tier_pass_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {val}", flush=True)
print(f"\nTotal time:          {elapsed:.0f}s | Avg per image: {elapsed/total:.1f}s", flush=True)
print(f"Results saved to:    {OUTPUT_CSV}", flush=True)
print(f"RUN COMPLETED UNINTERRUPTED", flush=True)
